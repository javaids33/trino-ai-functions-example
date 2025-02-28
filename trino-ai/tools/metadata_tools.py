import logging
import re
from typing import Dict, Any, Tuple
import sys
import os

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.base_tool import Tool
from embeddings import embedding_service
from colorama import Fore

logger = logging.getLogger(__name__)

class GetSchemaContextTool(Tool):
    """Tool for retrieving schema context for a natural language query"""
    
    def __init__(self, name: str = "Get Schema Context", description: str = "Retrieves relevant schema context for a natural language query"):
        super().__init__(name, description)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve schema context for a natural language query
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - max_tables: Optional maximum number of tables to include (default: 5)
                
        Returns:
            Dictionary containing:
                - context: The schema context
                - table_count: The number of tables in the context
                - schema_info_status: Status of schema information
        """
        query = inputs.get("query")
        max_tables = inputs.get("max_tables", 5)
        
        if not query:
            logger.error(f"{Fore.RED}No query provided to GetSchemaContextTool{Fore.RESET}")
            return {"error": "No query provided"}
        
        logger.info(f"{Fore.BLUE}Retrieving schema context for query: {query}{Fore.RESET}")
        
        try:
            # Get context from embedding service
            context = embedding_service.get_context_for_query(query, n_results=max_tables)
            
            # Count tables in context
            table_count = context.count("Table:")
            
            logger.info(f"{Fore.GREEN}Retrieved schema context with {table_count} tables{Fore.RESET}")
            
            return {
                "context": context,
                "table_count": table_count,
                "schema_info_status": "complete" if table_count > 0 else "no_tables_found"
            }
        except Exception as e:
            logger.error(f"{Fore.RED}Error retrieving schema context: {e}{Fore.RESET}")
            return {"error": f"Error retrieving schema context: {e}"}
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to get context for"
                },
                "max_tables": {
                    "type": "integer",
                    "description": "Maximum number of tables to include in context",
                    "default": 5
                }
            },
            "required": ["query"]
        }


class RefreshMetadataTool(Tool):
    """Tool for refreshing the database metadata cache"""
    
    def __init__(self, name: str = "Refresh Metadata", description: str = "Refreshes the database metadata cache"):
        super().__init__(name, description)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refresh the database metadata cache
        
        Args:
            inputs: Dictionary containing:
                - force: Optional boolean to force a complete refresh (default: False)
                
        Returns:
            Dictionary containing:
                - status: Status of the refresh operation
                - message: Message describing the result
        """
        force = inputs.get("force", False)
        
        logger.info(f"{Fore.BLUE}Refreshing metadata cache (force={force}){Fore.RESET}")
        
        try:
            # Refresh embeddings
            embedding_service.refresh_embeddings()
            
            logger.info(f"{Fore.GREEN}Metadata cache refreshed successfully{Fore.RESET}")
            
            return {
                "status": "success",
                "message": "Metadata cache refreshed successfully"
            }
        except Exception as e:
            logger.error(f"{Fore.RED}Error refreshing metadata cache: {e}{Fore.RESET}")
            return {
                "status": "error",
                "message": f"Error refreshing metadata cache: {e}"
            }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool"""
        return {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Whether to force a complete refresh",
                    "default": False
                }
            }
        } 