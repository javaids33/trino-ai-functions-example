import logging
import time
import json
from typing import Dict, Any, List, Optional
import trino
from colorama import Fore

logger = logging.getLogger(__name__)

class TrinoExecutor:
    """Executes SQL queries against a Trino server and returns results"""
    
    def __init__(self, host: str = "trino", port: int = 8080, user: str = "trino", 
                 catalog: str = "memory", schema: str = "default"):
        """
        Initialize the Trino executor
        
        Args:
            host: The Trino host
            port: The Trino port
            user: The Trino user
            catalog: The default catalog
            schema: The default schema
        """
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "catalog": catalog,
            "schema": schema,
        }
        logger.info("Trino Executor initialized with connection to %s:%s", host, port)
    
    def execute_query(self, sql: str, max_rows: int = 1000) -> Dict[str, Any]:
        """
        Execute a SQL query and return the results
        
        Args:
            sql: The SQL query to execute
            max_rows: Maximum number of rows to return
            
        Returns:
            Dictionary with query results and metadata
        """
        start_time = time.time()
        logger.info("Executing SQL query: %s", sql)
        
        try:
            # Create a connection to Trino
            conn = trino.dbapi.connect(**self.connection_params)
            cursor = conn.cursor()
            
            # Execute the query
            cursor.execute(sql)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch data (with row limit)
            rows = []
            row_count = 0
            truncated = False
            
            for row in cursor:
                rows.append(list(row))
                row_count += 1
                if row_count >= max_rows:
                    truncated = True
                    break
            
            # Close the connection
            cursor.close()
            conn.close()
            
            elapsed_time = time.time() - start_time
            logger.info("Query executed successfully in %.2fs, %d rows returned", elapsed_time, row_count)
            
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": row_count,
                "truncated": truncated,
                "execution_time": elapsed_time,
                "sql": sql
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_message = str(e)
            logger.error("Error executing query: %s", error_message)
            
            return {
                "success": False,
                "error": error_message,
                "execution_time": elapsed_time,
                "sql": sql
            } 