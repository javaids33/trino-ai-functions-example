import logging
import time
import re
from typing import Dict, Any, Optional, List, Tuple

import trino
from trino.exceptions import TrinoQueryError

from tools.base_tool import Tool
from trino_client import TrinoClient
from conversation_logger import conversation_logger

logger = logging.getLogger(__name__)

class ValidateSQLTool(Tool):
    """
    Tool for validating SQL queries against Trino without executing them.
    """
    
    def __init__(self, name: str = "validate_sql", description: str = "Validates SQL queries against Trino without executing them", trino_client: Optional[TrinoClient] = None):
        """
        Initialize the ValidateSQLTool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            trino_client: An optional TrinoClient instance
        """
        super().__init__(name, description)
        self.trino_client = trino_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ValidateSQLTool initialized")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a SQL query against Trino without executing it
        
        Args:
            inputs: A dictionary containing:
                - sql: The SQL query to validate
                
        Returns:
            A dictionary containing:
                - is_valid: Whether the query is valid
                - error_message: The error message if the query is invalid
                - error_type: The type of error if the query is invalid
        """
        sql = inputs.get("sql", "")
        
        if not sql:
            self.logger.error("No SQL provided for validation")
            return {
                "is_valid": False,
                "error_message": "No SQL provided for validation",
                "error_type": "missing_sql"
            }
        
        self.logger.info(f"Validating SQL query: {sql[:200]}...")
        conversation_logger.log_trino_ai_processing("sql_validation_start", {
            "sql": sql[:200] + "..." if len(sql) > 200 else sql
        })
        
        try:
            start_time = time.time()
            is_valid, error_message, error_type = self._validate_sql(sql)
            validation_time = time.time() - start_time
            
            if is_valid:
                self.logger.info(f"SQL validation successful (took {validation_time:.2f}s)")
                conversation_logger.log_trino_ai_processing("sql_validation_success", {
                    "validation_time": validation_time,
                    "sql": sql[:200] + "..." if len(sql) > 200 else sql
                })
            else:
                self.logger.error(f"SQL validation failed: {error_message} (type: {error_type})")
                conversation_logger.log_trino_ai_processing("sql_validation_failed", {
                    "validation_time": validation_time,
                    "error_message": error_message,
                    "error_type": error_type,
                    "sql": sql[:200] + "..." if len(sql) > 200 else sql
                })
            
            return {
                "is_valid": is_valid,
                "error_message": error_message,
                "error_type": error_type
            }
            
        except Exception as e:
            self.logger.error(f"Error during SQL validation: {str(e)}")
            conversation_logger.log_error("validate_sql_tool", f"Error during SQL validation: {str(e)}")
            
            return {
                "is_valid": False,
                "error_message": f"Error during SQL validation: {str(e)}",
                "error_type": "validation_error"
            }
    
    def _validate_sql(self, sql: str) -> Tuple[bool, str, str]:
        """
        Validate a SQL query against Trino without executing it
        
        Args:
            sql: The SQL query to validate
            
        Returns:
            A tuple containing:
                - Whether the query is valid
                - The error message if the query is invalid
                - The type of error if the query is invalid
        """
        # Clean up the SQL query
        sql = sql.strip()
        
        # Remove any trailing semicolons
        if sql.endswith(';'):
            sql = sql[:-1].strip()
        
        # Prepare the validation query
        validation_query = f"EXPLAIN {sql}"
        
        try:
            # Connect to Trino
            conn = self.trino_client.get_connection()
            cursor = conn.cursor()
            
            # Execute the EXPLAIN query
            cursor.execute(validation_query)
            
            # If we get here, the query is valid
            cursor.fetchall()  # Consume the results
            cursor.close()
            conn.close()
            
            return True, "", ""
            
        except TrinoQueryError as e:
            # Extract the error message and type
            error_message = str(e)
            error_type = self._categorize_error(error_message)
            
            return False, error_message, error_type
            
        except Exception as e:
            # Handle other exceptions
            return False, str(e), "unknown_error"
    
    def _categorize_error(self, error_message: str) -> str:
        """
        Categorize the error message into a specific error type
        
        Args:
            error_message: The error message from Trino
            
        Returns:
            The error type as a string
        """
        error_message = error_message.lower()
        
        if "table" in error_message and "not found" in error_message:
            return "table_not_found"
        elif "column" in error_message and "not found" in error_message:
            return "column_not_found"
        elif "syntax error" in error_message:
            return "syntax_error"
        elif "ambiguous" in error_message:
            return "ambiguous_reference"
        elif "type mismatch" in error_message or "cannot be applied to" in error_message:
            return "type_mismatch"
        elif "schema" in error_message and "not found" in error_message:
            return "schema_not_found"
        elif "catalog" in error_message and "not found" in error_message:
            return "catalog_not_found"
        else:
            return "other_error"

class ExecuteSQLTool(Tool):
    """
    Tool for executing SQL queries against Trino.
    """
    
    def __init__(self, name: str = "execute_sql", description: str = "Executes SQL queries against Trino"):
        """
        Initialize the ExecuteSQLTool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        super().__init__(name, description)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("ExecuteSQLTool initialized")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a SQL query against Trino
        
        Args:
            inputs: A dictionary containing:
                - sql: The SQL query to execute
                - max_rows: Optional maximum number of rows to return (default: 100)
                
        Returns:
            A dictionary containing:
                - success: Whether the query was executed successfully
                - columns: The column names if successful
                - rows: The result rows if successful
                - row_count: The number of rows returned
                - truncated: Whether the results were truncated
                - error: The error message if unsuccessful
        """
        sql = inputs.get("sql", "")
        max_rows = inputs.get("max_rows", 100)
        
        if not sql:
            logger.error("No SQL provided to ExecuteSQLTool")
            return {"success": False, "error": "No SQL provided"}
        
        logger.info(f"Executing SQL: {sql[:100]}...")
        conversation_logger.log_trino_ai_processing("sql_execution_start", {
            "sql": sql[:200] + "..." if len(sql) > 200 else sql,
            "max_rows": max_rows
        })
        
        try:
            # Connect to Trino
            conn = trino.dbapi.connect(
                host=self.get_env("TRINO_HOST", "trino"),
                port=int(self.get_env("TRINO_PORT", "8080")),
                user=self.get_env("TRINO_USER", "admin"),
                catalog=self.get_env("TRINO_CATALOG", "iceberg"),
                schema=self.get_env("TRINO_SCHEMA", "iceberg")
            )
            
            cursor = conn.cursor()
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
            
            logger.info(f"SQL execution successful: {row_count} rows returned")
            conversation_logger.log_trino_ai_processing("sql_execution_success", {
                "row_count": row_count,
                "column_count": len(columns),
                "truncated": truncated
            })
            
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": row_count,
                "truncated": truncated,
                "sql": sql
            }
            
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            conversation_logger.log_error("execute_sql_tool", f"Error executing SQL: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "sql": sql
            }
    
    def get_env(self, key: str, default: str) -> str:
        """
        Get an environment variable with a default value
        
        Args:
            key: The environment variable key
            default: The default value if the key is not found
            
        Returns:
            The value of the environment variable or the default
        """
        import os
        return os.getenv(key, default) 