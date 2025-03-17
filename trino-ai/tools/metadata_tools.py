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

class GetSchemaContextTool(BaseTool):
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


class RefreshMetadataTool(BaseTool):
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


class TableRelationshipsExtractor(BaseTool):
    """Tool to extract and store table relationships from Trino metadata"""
    
    def __init__(self, trino_client: Optional[trino.dbapi.Connection] = None):
        super().__init__(
            name="Table Relationships Extractor",
            description="Extracts foreign key and join relationships from Trino metadata"
        )
        self.trino_client = trino_client
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract table relationships from Trino metadata
        
        Args:
            inputs: Dictionary containing:
                - catalog: Optional catalog name to focus on
                - schema: Optional schema name to focus on
                - tables: Optional list of tables to focus on
                
        Returns:
            Dictionary containing the extracted relationships
        """
        catalog = inputs.get("catalog", "iceberg")
        schema = inputs.get("schema", "iceberg")
        tables = inputs.get("tables", [])
        
        logger.info(f"{Fore.CYAN}Extracting table relationships for catalog={catalog}, schema={schema}{Fore.RESET}")
        
        # Extract relationships based on foreign keys if available
        fk_relationships = self._extract_foreign_key_relationships(catalog, schema, tables)
        
        # Extract relationships based on column name patterns
        naming_relationships = self._extract_naming_pattern_relationships(catalog, schema, tables)
        
        # Extract relationships based on column data types
        type_relationships = self._extract_type_based_relationships(catalog, schema, tables)
        
        # Combine all relationships
        all_relationships = {
            "foreign_keys": fk_relationships,
            "naming_patterns": naming_relationships,
            "type_based": type_relationships
        }
        
        logger.info(f"{Fore.GREEN}Extracted {len(fk_relationships)} foreign key relationships, " +
                   f"{len(naming_relationships)} naming pattern relationships, " +
                   f"{len(type_relationships)} type-based relationships{Fore.RESET}")
        
        return {
            "relationships": all_relationships,
            "catalog": catalog,
            "schema": schema
        }
    
    def _extract_foreign_key_relationships(self, catalog: str, schema: str, tables: list[str]) -> list[Dict[str, Any]]:
        """Extract foreign key relationships from Trino metadata"""
        if not self.trino_client:
            logger.warning(f"{Fore.YELLOW}No Trino client available for foreign key extraction{Fore.RESET}")
            return []
        
        try:
            # Query for foreign key relationships
            # Note: This is catalog-dependent and may not work for all catalogs
            # For example, Iceberg doesn't expose FK constraints in information_schema
            query = f"""
            SELECT 
                tc.table_schema as foreign_schema,
                tc.table_name as foreign_table,
                kcu.column_name as foreign_column,
                ccu.table_schema as primary_schema,
                ccu.table_name as primary_table,
                ccu.column_name as primary_column
            FROM {catalog}.information_schema.table_constraints tc
            JOIN {catalog}.information_schema.key_column_usage kcu 
              ON tc.constraint_name = kcu.constraint_name
            JOIN {catalog}.information_schema.constraint_column_usage ccu 
              ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = '{schema}'
            """
            
            if tables:
                table_list = "', '".join(tables)
                query += f" AND tc.table_name IN ('{table_list}')"
            
            result = self.trino_client.execute_query(query)
            
            relationships = []
            for row in result.get("results", []):
                relationships.append({
                    "foreign_table": f"{row[0]}.{row[1]}",
                    "foreign_column": row[2],
                    "primary_table": f"{row[3]}.{row[4]}",
                    "primary_column": row[5],
                    "relationship_type": "foreign_key"
                })
            
            return relationships
        except Exception as e:
            logger.error(f"{Fore.RED}Error extracting foreign key relationships: {str(e)}{Fore.RESET}")
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