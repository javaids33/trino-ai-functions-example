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
from conversation_logger import conversation_logger

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

class GetCatalogsMetadataTool(Tool):
    """
    Tool for retrieving catalog metadata from Trino.
    """
    
    def __init__(self, name: str = "get_catalogs_metadata", description: str = "Retrieves catalog metadata from Trino"):
        """
        Initialize the GetCatalogsMetadataTool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        super().__init__(name, description)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("GetCatalogsMetadataTool initialized")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve catalog metadata from Trino
        
        Args:
            inputs: An empty dictionary (no inputs required)
                
        Returns:
            A dictionary containing:
                - catalogs: A list of catalog names
        """
        self.logger.info("Retrieving catalog metadata")
        conversation_logger.log_trino_ai_processing("metadata_retrieval_start", {
            "metadata_type": "catalogs"
        })
        
        try:
            # Connect to Trino
            conn = self.get_trino_connection()
            cursor = conn.cursor()
            
            # Execute the query to get catalogs
            cursor.execute("SHOW CATALOGS")
            
            # Fetch all catalogs
            catalogs = [row[0] for row in cursor.fetchall()]
            
            self.logger.info(f"Retrieved {len(catalogs)} catalogs")
            conversation_logger.log_trino_ai_processing("metadata_retrieval_success", {
                "metadata_type": "catalogs",
                "count": len(catalogs)
            })
            
            return {
                "catalogs": catalogs
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving catalog metadata: {str(e)}")
            conversation_logger.log_error("get_catalogs_metadata_tool", f"Error retrieving catalog metadata: {str(e)}")
            
            return {
                "error": f"Error retrieving catalog metadata: {str(e)}"
            }
    
    def get_trino_connection(self):
        """
        Get a connection to Trino
        
        Returns:
            A connection to Trino
        """
        import trino
        return trino.dbapi.connect(
            host=self.get_env("TRINO_HOST", "trino"),
            port=int(self.get_env("TRINO_PORT", "8080")),
            user=self.get_env("TRINO_USER", "admin"),
            catalog=self.get_env("TRINO_CATALOG", "iceberg"),
            schema=self.get_env("TRINO_SCHEMA", "iceberg")
        )
    
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

class GetSchemasMetadataTool(Tool):
    """
    Tool for retrieving schema metadata from Trino.
    """
    
    def __init__(self, name: str = "get_schemas_metadata", description: str = "Retrieves schema metadata from Trino"):
        """
        Initialize the GetSchemasMetadataTool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        super().__init__(name, description)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("GetSchemasMetadataTool initialized")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve schema metadata from Trino
        
        Args:
            inputs: A dictionary containing:
                - catalog: The catalog to retrieve schemas from
                
        Returns:
            A dictionary containing:
                - schemas: A list of schema names
                - catalog: The catalog the schemas belong to
        """
        catalog = inputs.get("catalog", "")
        
        if not catalog:
            self.logger.error("No catalog provided")
            return {
                "error": "No catalog provided"
            }
        
        self.logger.info(f"Retrieving schema metadata for catalog: {catalog}")
        conversation_logger.log_trino_ai_processing("metadata_retrieval_start", {
            "metadata_type": "schemas",
            "catalog": catalog
        })
        
        try:
            # Connect to Trino
            conn = self.get_trino_connection()
            cursor = conn.cursor()
            
            # Execute the query to get schemas
            cursor.execute(f"SHOW SCHEMAS FROM {catalog}")
            
            # Fetch all schemas
            schemas = [row[0] for row in cursor.fetchall()]
            
            self.logger.info(f"Retrieved {len(schemas)} schemas from catalog {catalog}")
            conversation_logger.log_trino_ai_processing("metadata_retrieval_success", {
                "metadata_type": "schemas",
                "catalog": catalog,
                "count": len(schemas)
            })
            
            return {
                "catalog": catalog,
                "schemas": schemas
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving schema metadata: {str(e)}")
            conversation_logger.log_error("get_schemas_metadata_tool", f"Error retrieving schema metadata: {str(e)}")
            
            return {
                "error": f"Error retrieving schema metadata: {str(e)}"
            }
    
    def get_trino_connection(self):
        """
        Get a connection to Trino
        
        Returns:
            A connection to Trino
        """
        import trino
        return trino.dbapi.connect(
            host=self.get_env("TRINO_HOST", "trino"),
            port=int(self.get_env("TRINO_PORT", "8080")),
            user=self.get_env("TRINO_USER", "admin"),
            catalog=self.get_env("TRINO_CATALOG", "iceberg"),
            schema=self.get_env("TRINO_SCHEMA", "iceberg")
        )
    
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

class GetTablesMetadataTool(Tool):
    """
    Tool for retrieving table metadata from Trino.
    """
    
    def __init__(self, name: str = "get_tables_metadata", description: str = "Retrieves table metadata from Trino"):
        """
        Initialize the GetTablesMetadataTool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        super().__init__(name, description)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("GetTablesMetadataTool initialized")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve table metadata from Trino
        
        Args:
            inputs: A dictionary containing:
                - catalog: The catalog to retrieve tables from
                - schema: The schema to retrieve tables from
                
        Returns:
            A dictionary containing:
                - tables: A list of table names
                - catalog: The catalog the tables belong to
                - schema: The schema the tables belong to
        """
        catalog = inputs.get("catalog", "")
        schema = inputs.get("schema", "")
        
        if not catalog:
            self.logger.error("No catalog provided")
            return {
                "error": "No catalog provided"
            }
        
        if not schema:
            self.logger.error("No schema provided")
            return {
                "error": "No schema provided"
            }
        
        self.logger.info(f"Retrieving table metadata for {catalog}.{schema}")
        conversation_logger.log_trino_ai_processing("metadata_retrieval_start", {
            "metadata_type": "tables",
            "catalog": catalog,
            "schema": schema
        })
        
        try:
            # Connect to Trino
            conn = self.get_trino_connection()
            cursor = conn.cursor()
            
            # Execute the query to get tables
            cursor.execute(f"SHOW TABLES FROM {catalog}.{schema}")
            
            # Fetch all tables
            tables = [row[0] for row in cursor.fetchall()]
            
            self.logger.info(f"Retrieved {len(tables)} tables from {catalog}.{schema}")
            conversation_logger.log_trino_ai_processing("metadata_retrieval_success", {
                "metadata_type": "tables",
                "catalog": catalog,
                "schema": schema,
                "count": len(tables)
            })
            
            return {
                "catalog": catalog,
                "schema": schema,
                "tables": tables
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving table metadata: {str(e)}")
            conversation_logger.log_error("get_tables_metadata_tool", f"Error retrieving table metadata: {str(e)}")
            
            return {
                "error": f"Error retrieving table metadata: {str(e)}"
            }
    
    def get_trino_connection(self):
        """
        Get a connection to Trino
        
        Returns:
            A connection to Trino
        """
        import trino
        return trino.dbapi.connect(
            host=self.get_env("TRINO_HOST", "trino"),
            port=int(self.get_env("TRINO_PORT", "8080")),
            user=self.get_env("TRINO_USER", "admin"),
            catalog=self.get_env("TRINO_CATALOG", "iceberg"),
            schema=self.get_env("TRINO_SCHEMA", "iceberg")
        )
    
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

class GetColumnsMetadataTool(Tool):
    """
    Tool for retrieving column metadata from Trino.
    """
    
    def __init__(self, name: str = "get_columns_metadata", description: str = "Retrieves column metadata from Trino"):
        """
        Initialize the GetColumnsMetadataTool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        super().__init__(name, description)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("GetColumnsMetadataTool initialized")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve column metadata from Trino
        
        Args:
            inputs: A dictionary containing:
                - catalog: The catalog to retrieve columns from
                - schema: The schema to retrieve columns from
                - table: The table to retrieve columns from
                
        Returns:
            A dictionary containing:
                - columns: A list of column metadata
                - catalog: The catalog the columns belong to
                - schema: The schema the columns belong to
                - table: The table the columns belong to
        """
        catalog = inputs.get("catalog", "")
        schema = inputs.get("schema", "")
        table = inputs.get("table", "")
        
        if not catalog:
            self.logger.error("No catalog provided")
            return {
                "error": "No catalog provided"
            }
        
        if not schema:
            self.logger.error("No schema provided")
            return {
                "error": "No schema provided"
            }
        
        if not table:
            self.logger.error("No table provided")
            return {
                "error": "No table provided"
            }
        
        self.logger.info(f"Retrieving column metadata for {catalog}.{schema}.{table}")
        conversation_logger.log_trino_ai_processing("metadata_retrieval_start", {
            "metadata_type": "columns",
            "catalog": catalog,
            "schema": schema,
            "table": table
        })
        
        try:
            # Connect to Trino
            conn = self.get_trino_connection()
            cursor = conn.cursor()
            
            # Execute the query to get columns
            cursor.execute(f"DESCRIBE {catalog}.{schema}.{table}")
            
            # Fetch all columns
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "extra": row[2] if len(row) > 2 else ""
                })
            
            self.logger.info(f"Retrieved {len(columns)} columns from {catalog}.{schema}.{table}")
            conversation_logger.log_trino_ai_processing("metadata_retrieval_success", {
                "metadata_type": "columns",
                "catalog": catalog,
                "schema": schema,
                "table": table,
                "count": len(columns)
            })
            
            return {
                "catalog": catalog,
                "schema": schema,
                "table": table,
                "columns": columns
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving column metadata: {str(e)}")
            conversation_logger.log_error("get_columns_metadata_tool", f"Error retrieving column metadata: {str(e)}")
            
            return {
                "error": f"Error retrieving column metadata: {str(e)}"
            }
    
    def get_trino_connection(self):
        """
        Get a connection to Trino
        
        Returns:
            A connection to Trino
        """
        import trino
        return trino.dbapi.connect(
            host=self.get_env("TRINO_HOST", "trino"),
            port=int(self.get_env("TRINO_PORT", "8080")),
            user=self.get_env("TRINO_USER", "admin"),
            catalog=self.get_env("TRINO_CATALOG", "iceberg"),
            schema=self.get_env("TRINO_SCHEMA", "iceberg")
        )
    
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