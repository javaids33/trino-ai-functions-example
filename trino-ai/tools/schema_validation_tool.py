import logging
from typing import Dict, Any, List
import sys
import os
import difflib

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.base_tool import Tool
from conversation_logger import conversation_logger
from trino_client import TrinoClient

logger = logging.getLogger(__name__)

class SchemaValidationTool(Tool):
    """Tool for validating tables and columns against the actual database schema"""
    
    def __init__(self, name: str = "schema_validator", description: str = "Validates tables and columns against the actual database schema"):
        super().__init__(name, description)
        self.trino_client = TrinoClient()
        logger.info("Schema Validation Tool initialized")
        
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tables and columns against the actual database schema
        
        Args:
            inputs: Dictionary containing:
                - tables: List of tables to validate
                - columns: List of columns to validate (in format "table.column")
                
        Returns:
            Dictionary containing:
                - valid_tables: List of valid tables
                - invalid_tables: List of invalid tables
                - valid_columns: List of valid columns
                - invalid_columns: List of invalid columns
                - suggested_corrections: Suggested corrections for invalid items
        """
        tables = inputs.get("tables", [])
        columns = inputs.get("columns", [])
        
        logger.info(f"Validating schema: {len(tables)} tables, {len(columns)} columns")
        conversation_logger.log_trino_ai_processing("schema_validation_start", {
            "tables_count": len(tables),
            "columns_count": len(columns)
        })
        
        # Connect to database to validate
        try:
            conn = self.trino_client.get_connection()
            cursor = conn.cursor()
            
            # Get all tables in the current catalog/schema
            try:
                cursor.execute("SHOW TABLES")
                available_tables = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Error getting tables: {str(e)}")
                available_tables = []
            
            # Validate tables
            valid_tables = [t for t in tables if t in available_tables]
            invalid_tables = [t for t in tables if t not in available_tables]
            
            # Get columns for each valid table
            available_columns = {}
            for table in valid_tables:
                try:
                    cursor.execute(f"SHOW COLUMNS FROM {table}")
                    available_columns[table] = [row[0] for row in cursor.fetchall()]
                except Exception as e:
                    logger.error(f"Error getting columns for table {table}: {str(e)}")
                    available_columns[table] = []
            
            # Validate columns
            valid_columns = []
            invalid_columns = []
            for column in columns:
                if "." in column:
                    table, col = column.split(".", 1)
                    if table in available_columns and col in available_columns[table]:
                        valid_columns.append(column)
                    else:
                        invalid_columns.append(column)
                else:
                    # Column without table specified
                    invalid_columns.append(column)
            
            # Generate suggestions for invalid items
            suggested_corrections = {}
            
            # Table suggestions
            for invalid_table in invalid_tables:
                # Find similar table names
                matches = difflib.get_close_matches(invalid_table, available_tables, n=3, cutoff=0.6)
                if matches:
                    suggested_corrections[invalid_table] = matches
            
            # Column suggestions
            for invalid_column in invalid_columns:
                if "." in invalid_column:
                    table, col = invalid_column.split(".", 1)
                    if table in available_columns:
                        # Find similar column names in this table
                        matches = difflib.get_close_matches(col, available_columns[table], n=3, cutoff=0.6)
                        if matches:
                            suggested_corrections[invalid_column] = [f"{table}.{match}" for match in matches]
            
            # Close the connection
            cursor.close()
            conn.close()
            
            result = {
                "valid_tables": valid_tables,
                "invalid_tables": invalid_tables,
                "valid_columns": valid_columns,
                "invalid_columns": invalid_columns,
                "suggested_corrections": suggested_corrections
            }
            
            logger.info(f"Schema validation completed: {len(valid_tables)} valid tables, {len(invalid_tables)} invalid tables")
            conversation_logger.log_trino_ai_processing("schema_validation_complete", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in schema validation: {str(e)}")
            conversation_logger.log_error("schema_validation_tool", f"Error in schema validation: {str(e)}")
            
            return {
                "error": f"Error in schema validation: {str(e)}",
                "valid_tables": [],
                "invalid_tables": tables,
                "valid_columns": [],
                "invalid_columns": columns,
                "suggested_corrections": {}
            }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the parameters schema for this tool
        
        Returns:
            The parameters schema for this tool
        """
        return {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tables to validate"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of columns to validate (in format 'table.column')"
                }
            },
            "required": ["tables"]
        } 