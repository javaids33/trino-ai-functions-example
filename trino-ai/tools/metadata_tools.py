import logging
import re
from typing import Dict, Any, Tuple, Optional, List
import sys
import os
import trino
from colorama import Fore

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.base_tool import BaseTool
from embeddings import embedding_service
from conversation_logger import conversation_logger

logger = logging.getLogger(__name__)

# Make sure MetadataTool is properly defined
class MetadataTool(BaseTool):
    """Base tool for metadata operations"""
    
    def __init__(self, name: str, description: str, 
                 trino_client: Optional[trino.dbapi.Connection] = None,
                 metadata_cache_path: str = "metadata_cache.json"):
        """
        Initialize the metadata tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            trino_client: A Trino client
            metadata_cache_path: Path to the metadata cache file
        """
        super().__init__(name, description)
        self.trino_client = trino_client
        self.metadata_cache_path = metadata_cache_path
        
        # If trino_client is not provided, we don't create one here
        if self.trino_client is None:
            logger.warning(f"{Fore.YELLOW}MetadataTool initialized without a Trino client. Client must be provided before execution.{Fore.RESET}")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the metadata tool
        
        Args:
            inputs: The inputs to the tool
            
        Returns:
            The outputs from the tool
        """
        # This is an abstract base class, so this method should be overridden
        raise NotImplementedError("MetadataTool is an abstract base class. Use a concrete implementation.")
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the schema for the tool's parameters
        
        Returns:
            A dictionary describing the parameters schema
        """
        # This is an abstract base class, so this method should be overridden
        raise NotImplementedError("MetadataTool is an abstract base class. Use a concrete implementation.")

# Now define the other classes
class GetSchemaContextTool(MetadataTool):
    """Tool for getting schema context"""
    
    def __init__(self, name: str = "Get Schema Context Tool", 
                 description: str = "Gets schema context for a query",
                 trino_client: Optional[trino.dbapi.Connection] = None,
                 metadata_cache_path: str = "metadata_cache.json"):
        """
        Initialize the get schema context tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            trino_client: A Trino client
            metadata_cache_path: Path to the metadata cache file
        """
        super().__init__(name, description, trino_client, metadata_cache_path)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of the execute method"""
        # Add implementation here
        return {"schema_context": "Sample schema context"}
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the schema for the tool's parameters
        
        Returns:
            A dictionary describing the parameters schema
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to get schema context for"
                },
                "catalog": {
                    "type": "string",
                    "description": "The catalog to get schema context from"
                },
                "schema": {
                    "type": "string",
                    "description": "The schema to get schema context from"
                }
            },
            "required": ["query"]
        }

class RefreshMetadataTool(MetadataTool):
    """Tool for refreshing metadata"""
    
    def __init__(self, name: str = "Refresh Metadata Tool", 
                 description: str = "Refreshes metadata cache",
                 trino_client: Optional[trino.dbapi.Connection] = None,
                 metadata_cache_path: str = "metadata_cache.json"):
        """
        Initialize the refresh metadata tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            trino_client: A Trino client
            metadata_cache_path: Path to the metadata cache file
        """
        super().__init__(name, description, trino_client, metadata_cache_path)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of the execute method"""
        # Add implementation here
        return {"status": "Metadata refreshed"}
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the schema for the tool's parameters
        
        Returns:
            A dictionary describing the parameters schema
        """
        return {
            "type": "object",
            "properties": {
                "catalog": {
                    "type": "string",
                    "description": "The catalog to refresh metadata for"
                },
                "schema": {
                    "type": "string",
                    "description": "The schema to refresh metadata for"
                }
            },
            "required": []
        }

class TableRelationshipsExtractor(BaseTool):
    """Tool for extracting table relationships"""
    
    def __init__(self, name: str = "Table Relationships Extractor", 
                 description: str = "Extracts relationships between tables",
                 trino_client: Optional[trino.dbapi.Connection] = None):
        """
        Initialize the table relationships extractor
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            trino_client: A Trino client
        """
        super().__init__(name, description)
        self.trino_client = trino_client
        
        # If trino_client is not provided, we don't create one here
        if self.trino_client is None:
            logger.warning(f"{Fore.YELLOW}TableRelationshipsExtractor initialized without a Trino client. Client must be provided before execution.{Fore.RESET}")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of the execute method"""
        # Add implementation here
        return {"relationships": []}
    
    def _extract_foreign_key_relationships(self, catalog: str, schema: str, tables: list[str]) -> list[Dict[str, Any]]:
        """
        Extract foreign key relationships between tables
        
        Args:
            catalog: The catalog name
            schema: The schema name
            tables: List of table names
            
        Returns:
            List of foreign key relationships
        """
        # Implementation here
        return []
    
    def _extract_naming_pattern_relationships(self, catalog: str, schema: str, tables: List[str]) -> List[Dict[str, Any]]:
        """Extract relationships based on column naming patterns"""
        if not self.trino_client:
            return []
        
        try:
            # Get all tables and columns
            query = f"""
            SELECT 
                table_name, 
                column_name
            FROM {catalog}.information_schema.columns
            WHERE table_schema = '{schema}'
            """
            
            if tables:
                table_list = "', '".join(tables)
                query += f" AND table_name IN ('{table_list}')"
            
            result = self.trino_client.execute_query(query)
            
            # Build a dictionary of tables and their columns
            table_columns = {}
            for row in result.get("results", []):
                table_name = row[0]
                column_name = row[1]
                
                if table_name not in table_columns:
                    table_columns[table_name] = []
                
                table_columns[table_name].append(column_name)
            
            # Look for common patterns like:
            # - id columns (table_id in one table matching id in another)
            # - _id suffix columns
            relationships = []
            
            for table1, columns1 in table_columns.items():
                for column1 in columns1:
                    # Check for _id suffix
                    if column1.endswith('_id'):
                        # Extract the prefix (e.g., "customer" from "customer_id")
                        prefix = column1[:-3]
                        
                        # Look for a table with this name
                        if prefix in table_columns:
                            # Check if this table has an "id" column
                            if "id" in table_columns[prefix]:
                                relationships.append({
                                    "foreign_table": f"{schema}.{table1}",
                                    "foreign_column": column1,
                                    "primary_table": f"{schema}.{prefix}",
                                    "primary_column": "id",
                                    "relationship_type": "naming_pattern",
                                    "confidence": "high"
                                })
            
            return relationships
        except Exception as e:
            logger.error(f"{Fore.RED}Error extracting naming pattern relationships: {str(e)}{Fore.RESET}")
            return []
    
    def _extract_type_based_relationships(self, catalog: str, schema: str, tables: List[str]) -> List[Dict[str, Any]]:
        """Extract relationships based on column data types"""
        if not self.trino_client:
            return []
        
        try:
            # Get all tables, columns and their data types
            query = f"""
            SELECT 
                table_name, 
                column_name,
                data_type
            FROM {catalog}.information_schema.columns
            WHERE table_schema = '{schema}'
            """
            
            if tables:
                table_list = "', '".join(tables)
                query += f" AND table_name IN ('{table_list}')"
            
            result = self.trino_client.execute_query(query)
            
            # Build a dictionary of tables and their columns with types
            table_columns = {}
            for row in result.get("results", []):
                table_name = row[0]
                column_name = row[1]
                data_type = row[2]
                
                if table_name not in table_columns:
                    table_columns[table_name] = []
                
                table_columns[table_name].append({
                    "name": column_name,
                    "type": data_type
                })
            
            # Look for columns with the same name and type across tables
            relationships = []
            
            for table1, columns1 in table_columns.items():
                for column1 in columns1:
                    # Skip common columns like "created_at", "updated_at", etc.
                    if column1["name"] in ["created_at", "updated_at", "created_by", "updated_by"]:
                        continue
                    
                    # Look for columns with the same name and type in other tables
                    for table2, columns2 in table_columns.items():
                        if table1 == table2:
                            continue
                        
                        for column2 in columns2:
                            if column1["name"] == column2["name"] and column1["type"] == column2["type"]:
                                # This is a potential join relationship
                                relationships.append({
                                    "table1": f"{schema}.{table1}",
                                    "column1": column1["name"],
                                    "table2": f"{schema}.{table2}",
                                    "column2": column2["name"],
                                    "data_type": column1["type"],
                                    "relationship_type": "common_column",
                                    "confidence": "medium"
                                })
            
            return relationships
        except Exception as e:
            logger.error(f"{Fore.RED}Error extracting type-based relationships: {str(e)}{Fore.RESET}")
            return []
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool"""
        return {
            "type": "object",
            "properties": {
                "catalog": {
                    "type": "string",
                    "description": "The catalog to extract relationships from"
                },
                "schema": {
                    "type": "string",
                    "description": "The schema to extract relationships from"
                },
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Optional list of tables to focus on"
                }
            }
        } 