# At the top of the file, add these imports
import sys
import inspect
from logger_config import setup_logger
import time
import logging
import json
from typing import Dict, Any, Optional, List
from queue import Queue, Empty
from trino.dbapi import connect
from trino.auth import BasicAuthentication
import requests
from requests.exceptions import RequestException
from app_config import get_config

logger = setup_logger(__name__)
config = get_config()

class TrinoConnectionManager:
    """Manages connections to Trino with connection pooling"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrinoConnectionManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the connection manager"""
        self.pool_size = config.connection_pool_size
        self.connection_pool = Queue(maxsize=self.pool_size)
        self.active_connections = 0
        self.credentials = config.get_trino_credentials()
        self.session_properties = {}
        
        # Parse session properties if provided
        try:
            session_props = config.trino_session_props
            if session_props:
                self.session_properties = json.loads(session_props)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse session properties: {session_props}")
    
    def get_connection(self):
        """Get a connection from the pool or create a new one if the pool is empty"""
        try:
            # Try to get a connection from the pool
            conn = self.connection_pool.get_nowait()
            logger.debug("Reusing existing connection from pool")
            return conn
        except Empty:
            # If the pool is empty, create a new connection
            if self.active_connections < self.pool_size:
                logger.debug("Creating new connection")
                conn = self._create_connection()
                self.active_connections += 1
                return conn
            else:
                # If we've reached the maximum number of connections, wait for one to be returned
                logger.debug("Waiting for a connection to be returned to the pool")
                return self.connection_pool.get()
    
    def release_connection(self, conn):
        """Return a connection to the pool"""
        try:
            self.connection_pool.put_nowait(conn)
            logger.debug("Connection returned to pool")
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
            self.active_connections -= 1
    
    def _create_connection(self):
        """Create a new connection to Trino"""
        try:
            auth = None
            if self.credentials.get('password'):
                auth = BasicAuthentication(
                    self.credentials['user'],
                    self.credentials['password']
                )
            
            conn = connect(
                host=self.credentials['host'],
                port=self.credentials['port'],
                user=self.credentials['user'],
                catalog=self.credentials['catalog'],
                schema=self.credentials['schema'],
                auth=auth,
                session_properties=self.session_properties
            )
            
            return conn
        except Exception as e:
            logger.error(f"Error creating Trino connection: {e}")
            raise
    
    def execute_query(self, query, params=None):
        """Execute a query and return the results"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            result = list(cursor)
            column_names = None
            
            if cursor.description:
                column_names = [col[0] for col in cursor.description]
            
            return {"columns": column_names, "data": result}
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            raise
        finally:
            if conn:
                self.release_connection(conn)
    
    def execute_update(self, query, params=None):
        """Execute a query that doesn't return results (DDL, insert, update, delete)"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # For operations that return a row count
            row_count = cursor.rowcount if hasattr(cursor, 'rowcount') else -1
            return row_count
            
        except Exception as e:
            logger.error(f"Error executing update: {e}")
            logger.error(f"Query: {query}")
            raise
        finally:
            if conn:
                self.release_connection(conn)
    
    def list_tables(self, schema=None):
        """List all tables in a schema"""
        if not schema:
            schema = self.credentials['schema']
            
        query = f"SHOW TABLES FROM {schema}"
        result = self.execute_query(query)
        
        if result and result['data']:
            return [row[0] for row in result['data']]
        return []
    
    def table_exists(self, table_name, schema=None):
        """Check if a table exists"""
        tables = self.list_tables(schema)
        return table_name in tables
    
    def drop_table(self, table_name, schema=None):
        """Drop a table if it exists"""
        if not schema:
            schema = self.credentials['schema']
            
        if self.table_exists(table_name, schema):
            query = f"DROP TABLE {schema}.{table_name}"
            self.execute_update(query)
            logger.info(f"Dropped table {schema}.{table_name}")
            return True
        
        logger.info(f"Table {schema}.{table_name} does not exist, no need to drop")
        return False
    
    def drop_all_tables(self, schemas=None):
        """Drop all tables from specified schemas (or all schemas if None)"""
        if not schemas:
            # Get all schemas excluding system schemas
            result = self.execute_query("SHOW SCHEMAS")
            schemas = [row[0] for row in result['data'] 
                     if row[0] not in ['information_schema', 'system']]
        
        tables_dropped = 0
        for schema in schemas:
            tables = self.list_tables(schema)
            
            for table in tables:
                try:
                    query = f"DROP TABLE {schema}.{table}"
                    self.execute_update(query)
                    tables_dropped += 1
                    logger.info(f"Dropped table {schema}.{table}")
                except Exception as e:
                    logger.error(f"Error dropping table {schema}.{table}: {e}")
        
        logger.info(f"Successfully dropped {tables_dropped} tables")
        return tables_dropped
    
    def create_schema(self, schema_name):
        """Create a schema if it doesn't exist"""
        try:
            query = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
            self.execute_update(query)
            logger.info(f"Created schema {schema_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating schema {schema_name}: {e}")
            return False
    
    def wait_for_trino_ready(self, max_retries=30, delay=10):
        """Wait for Trino to be ready by checking its health endpoint"""
        logger.info("Waiting for Trino to be ready...")
        
        for attempt in range(max_retries):
            try:
                # Try to connect to Trino's health endpoint
                response = requests.get(f'http://{self.credentials["host"]}:{self.credentials["port"]}/v1/info/state')
                if response.status_code == 200 and response.text.strip('"') == "ACTIVE":
                    logger.info("Trino is ready!")
                    return True
                    
                logger.debug(f"Trino not ready yet (Attempt {attempt + 1}/{max_retries}). Status: {response.text}")
            except RequestException as e:
                logger.debug(f"Connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
            logger.info(f"Waiting {delay} seconds before next attempt...")
            time.sleep(delay)
        
        raise Exception("Trino failed to become ready within the timeout period")
    
    def get_table_schema(self, table_name, schema=None):
        """Get the schema of a table"""
        if not schema:
            schema = self.credentials['schema']
            
        query = f"DESCRIBE {schema}.{table_name}"
        try:
            result = self.execute_query(query)
            return result
        except Exception as e:
            logger.error(f"Error getting schema for table {schema}.{table_name}: {e}")
            return None

# Create a singleton instance and export it
def get_trino_connection_manager():
    """Get the singleton TrinoConnectionManager instance"""
    return TrinoConnectionManager() 