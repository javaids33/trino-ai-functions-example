import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from trino.dbapi import connect
from trino.exceptions import TrinoQueryError

class TrinoClient:
    """
    A client for interacting with Trino database.
    Provides methods for executing queries and handling results.
    """
    
    def __init__(self, host: str = None, port: int = None, user: str = None, catalog: str = None, schema: str = None):
        """
        Initialize the Trino client with connection parameters.
        
        Args:
            host: Trino server hostname
            port: Trino server port
            user: Username for Trino
            catalog: Default catalog to use
            schema: Default schema to use
        """
        self.host = host or os.environ.get('TRINO_HOST', 'trino')
        self.port = port or int(os.environ.get('TRINO_PORT', '8080'))
        self.user = user or os.environ.get('TRINO_USER', 'trino')
        self.catalog = catalog or os.environ.get('TRINO_CATALOG', 'iceberg')
        self.schema = schema or os.environ.get('TRINO_SCHEMA', 'iceberg')
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized TrinoClient with host={self.host}, port={self.port}, user={self.user}, catalog={self.catalog}, schema={self.schema}")
    
    def get_connection(self):
        """Get a connection to the Trino server."""
        return connect(
            host=self.host,
            port=self.port,
            user=self.user,
            catalog=self.catalog,
            schema=self.schema
        )
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a query and return the results as a list of dictionaries.
        
        Args:
            query: SQL query to execute
            
        Returns:
            List of dictionaries representing the rows returned by the query
        """
        self.logger.debug(f"Executing query: {query}")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    results = []
                    
                    for row in cursor:
                        results.append(dict(zip(columns, row)))
                    
                    self.logger.debug(f"Query returned {len(results)} rows")
                    return results
                else:
                    self.logger.debug("Query executed successfully with no results")
                    return []
                    
        except TrinoQueryError as e:
            self.logger.error(f"Query error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
    
    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax by using EXPLAIN without executing the query.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Remove any trailing semicolons before validation
            sql = sql.strip()
            if sql.endswith(';'):
                sql = sql[:-1].strip()
                
            # Use EXPLAIN to validate the SQL without executing it
            explain_query = f"EXPLAIN {sql}"
            self.logger.debug(f"Executing validation query: {explain_query}")
            self.execute_query(explain_query)
            return True, None
        except TrinoQueryError as e:
            error_message = str(e)
            self.logger.warning(f"SQL validation failed: {error_message}")
            return False, error_message
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Unexpected error during SQL validation: {error_message}")
            return False, error_message 