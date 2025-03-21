import logging
import re
from typing import Dict, Any, List, Optional
import sys
import os
import time
import trino
from colorama import Fore
from conversation_logger import conversation_logger
from tools.base_tool import BaseTool

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Remove the global connection initialization
# trino_client = trino.dbapi.Connection()  # This line is causing the error

class SQLTool(BaseTool):
    """Tool for executing SQL queries"""
    
    def __init__(self, name: str = "SQL Tool", description: str = "Executes SQL queries", 
                 trino_client: Optional[trino.dbapi.Connection] = None):
        """
        Initialize the SQL tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            trino_client: A Trino client
        """
        super().__init__(name, description)
        self.trino_client = trino_client
        
        # If trino_client is not provided, we don't create one here
        # The client should be passed in from the application that uses this tool
        if self.trino_client is None:
            logger.warning(f"{Fore.YELLOW}SQLTool initialized without a Trino client. Client must be provided before execution.{Fore.RESET}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a SQL query and return results
        
        Args:
            inputs: Dictionary containing:
                - sql: The SQL query to execute
                - max_rows: Optional maximum number of rows to return (default: 10)
                
        Returns:
            Dictionary containing:
                - sql: The executed SQL query
                - results: The query results
                - row_count: The number of rows returned
        """
        sql = inputs.get("sql")
        max_rows = inputs.get("max_rows", 10)
        
        if not sql:
            logger.error(f"{Fore.RED}No SQL provided to ExecuteSQLTool{Fore.RESET}")
            return {"error": "No SQL provided"}
        
        logger.info(f"{Fore.BLUE}Executing SQL: {sql[:100]}...{Fore.RESET}")
        
        try:
            # Execute SQL using Trino client
            results = self.trino_client.execute_query(sql)
            
            # Limit results to max_rows
            limited_results = results[:max_rows] if results else []
            row_count = len(limited_results)
            
            logger.info(f"{Fore.GREEN}SQL execution successful: {row_count} rows returned{Fore.RESET}")
            
            return {
                "sql": sql,
                "results": limited_results,
                "row_count": row_count
            }
        except Exception as e:
            logger.error(f"{Fore.RED}Error executing SQL: {e}{Fore.RESET}")
            return {"error": f"Error executing SQL: {e}"}
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool"""
        return {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute"
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum number of rows to return",
                    "default": 10
                }
            },
            "required": ["sql"]
        }

class SQLValidationTool(BaseTool):
    """Tool for validating SQL queries"""
    
    def __init__(self, name: str = "SQL Validation Tool", 
                 description: str = "Validates SQL queries for syntactic and semantic correctness",
                 trino_client: Optional[trino.dbapi.Connection] = None):
        """
        Initialize the SQL validation tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            trino_client: A Trino client
        """
        super().__init__(name, description)
        self.trino_client = trino_client
        
        # If trino_client is not provided, we don't create one here
        if self.trino_client is None:
            logger.warning(f"{Fore.YELLOW}SQLValidationTool initialized without a Trino client. Client must be provided before execution.{Fore.RESET}")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a SQL query without executing it
        
        Args:
            inputs: Dictionary containing:
                - sql: The SQL query to validate
                
        Returns:
            Dictionary containing:
                - is_valid: Boolean indicating whether the query is valid
                - error_message: Error message if the query is invalid
                - error_type: Categorized error type (syntax, schema_mismatch, permissions, etc.)
        """
        sql = inputs.get("sql", "")
        
        if not sql:
            logger.error(f"{Fore.RED}No SQL provided for validation{Fore.RESET}")
            return {
                "is_valid": False,
                "error_message": "No SQL provided for validation",
                "error_type": "missing_input"
            }
        
        logger.info(f"{Fore.YELLOW}Validating SQL query: {sql[:200]}...{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("validate_sql_start", {
            "sql_length": len(sql),
            "sql_preview": sql[:200] + "..." if len(sql) > 200 else sql
        })
        
        try:
            start_time = time.time()
            is_valid, error_message = self.trino_client.validate_sql(sql)
            validation_time = time.time() - start_time
            
            if is_valid:
                logger.info(f"{Fore.GREEN}SQL validation successful (took {validation_time:.2f}s){Fore.RESET}")
                conversation_logger.log_trino_ai_processing("validate_sql_success", {
                    "validation_time_seconds": validation_time
                })
                return {
                    "is_valid": True,
                    "error_message": "",
                    "error_type": None
                }
            else:
                # Identify specific error types for better debugging
                error_type = self._categorize_error(error_message)
                
                logger.error(f"{Fore.RED}SQL validation failed: {error_message} (type: {error_type}){Fore.RESET}")
                conversation_logger.log_trino_ai_processing("validate_sql_failure", {
                    "error_message": error_message,
                    "error_type": error_type,
                    "validation_time_seconds": validation_time
                })
                
                return {
                    "is_valid": False,
                    "error_message": error_message,
                    "error_type": error_type
                }
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error during SQL validation: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("validate_sql", f"Execution error: {str(e)}")
            return {
                "is_valid": False,
                "error_message": f"Error during validation: {str(e)}",
                "error_type": "validation_error"
            }
    
    def _categorize_error(self, error_message: str) -> str:
        """
        Categorize the error message into a specific type
        
        Args:
            error_message: The error message from Trino
            
        Returns:
            The categorized error type
        """
        error_message = error_message.upper()
        
        if "SYNTAX_ERROR" in error_message:
            return "syntax"
        elif "COLUMN_NOT_FOUND" in error_message or "TABLE_NOT_FOUND" in error_message:
            return "schema_mismatch"
        elif "SCHEMA_NOT_FOUND" in error_message or "CATALOG_NOT_FOUND" in error_message:
            return "catalog_schema_error"
        elif "ACCESS_DENIED" in error_message or "PERMISSION" in error_message:
            return "permissions"
        elif "TYPE_MISMATCH" in error_message or "INVALID_CAST" in error_message:
            return "type_error"
        elif "DIVISION_BY_ZERO" in error_message:
            return "division_by_zero"
        elif "FUNCTION_NOT_FOUND" in error_message:
            return "function_not_found"
        elif "AMBIGUOUS" in error_message:
            return "ambiguous_reference"
        elif "EXCEEDED" in error_message and "LIMIT" in error_message:
            return "limit_exceeded"
        else:
            return "unknown"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool"""
        return {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to validate"
                }
            },
            "required": ["sql"]
        } 