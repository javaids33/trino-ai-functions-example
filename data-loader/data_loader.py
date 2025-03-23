import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
from typing import Dict, Any, List, Optional
import pandas as pd

from app_config import get_config
from logger_config import setup_logger
from trino_connector import get_trino_connection_manager
from minio_helper import get_minio_manager
from cache_manager import DatasetCacheManager
from socrata_loader import SocrataLoader

# Set up logger
logger = setup_logger(__name__)
config = get_config()

class DataLoader:
    """Data loader service that handles loading data from Socrata to Trino."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.config = get_config()
        self.cache = DatasetCacheManager()
        self.trino = get_trino_connection_manager()
    
    def load_dataset(self, dataset_id: str, force_reload: bool = False) -> dict:
        """
        Load a dataset from Socrata to Trino.
        
        Args:
            dataset_id: Socrata dataset ID
            force_reload: Whether to force reload even if cached
            
        Returns:
            Dict with result information
        """
        try:
            logger.info(f"Loading dataset {dataset_id}, force_reload={force_reload}")
            
            # Get dataset as Parquet
            parquet_path, metadata = self._get_dataset_parquet(dataset_id, force_reload)
            
            # Load data into Trino
            result = self._load_to_trino(dataset_id, parquet_path, metadata)
            
            return result
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _get_dataset_parquet(self, dataset_id: str, force_reload: bool = False) -> tuple:
        """
        Get the dataset as a Parquet file, either from cache or from Socrata
        
        Args:
            dataset_id: Socrata dataset ID
            force_reload: Whether to force reload even if cached
                
        Returns:
            Tuple of (parquet_path, metadata)
        """
        # Check if dataset exists in cache and we don't need to force reload
        if not force_reload and self.cache.is_dataset_cached(dataset_id):
            logger.info(f"Using cached version of dataset {dataset_id}")
            dataset_info = self.cache.get_dataset_info(dataset_id)
            metadata = self.cache.get_dataset_metadata(dataset_id)
            return dataset_info.get("file_path"), metadata
        
        # Dataset not in cache or force reload, fetch from Socrata
        logger.info(f"Fetching dataset {dataset_id} from Socrata")
        
        try:
            # Create the Socrata loader
            socrata = SocrataLoader()
            
            # Set the output path for the Parquet file
            parquet_path = os.path.join(self.config.temp_dir, f"{dataset_id}.parquet")
            
            # Download the dataset
            result = socrata.download_dataset_as_parquet(dataset_id, parquet_path)
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Failed to download dataset {dataset_id}: {error_msg}")
                raise Exception(error_msg)
                
            # Get metadata
            metadata = socrata.get_dataset_metadata(dataset_id)
            
            # Extract the required fields from the download result and metadata
            name = result.get("name") or metadata.get("name", f"Dataset {dataset_id}")
            description = result.get("description") or metadata.get("description", "")
            row_count = result.get("row_count", 0)
            
            # Read the Parquet file to get column information
            df = pd.read_parquet(parquet_path)
            
            # Extract column metadata
            column_metadata = []
            
            # Fix the columns iteration - ensure we handle any metadata format
            columns_data = metadata.get("columns", [])
            if isinstance(columns_data, list):
                # Process as expected
                for col in columns_data:
                    column_info = {
                        "name": col.get("name", ""),
                        "datatype": col.get("datatype", ""),
                        "description": col.get("description", "")
                    }
                    column_metadata.append(column_info)
            else:
                # Try to extract columns from the dataframe instead
                logger.warning(f"Dataset {dataset_id} metadata has unexpected 'columns' format: {type(columns_data)}")
                try:
                    # Get column names from the dataframe
                    for col_name in df.columns:
                        column_info = {
                            "name": col_name,
                            "datatype": str(df[col_name].dtype),
                            "description": ""
                        }
                        column_metadata.append(column_info)
                except Exception as e:
                    logger.error(f"Error extracting columns from dataframe: {str(e)}")
            
            # Add column metadata to the main metadata
            metadata["column_details"] = column_metadata
            
            # Log the metadata we've extracted
            logger.info(f"Extracted metadata for dataset {dataset_id}: {name}, {row_count} rows")
            
            # Update cache with all required parameters
            self.cache.update_dataset_cache(
                dataset_id=dataset_id,
                file_path=parquet_path,
                name=name,
                description=description,
                row_count=row_count,
                metadata=metadata
            )
            
            return parquet_path, metadata
            
        except Exception as e:
            logger.error(f"Error getting dataset parquet for {dataset_id}: {str(e)}", exc_info=True)
            raise
    
    def _load_to_trino(self, dataset_id: str, parquet_path: str, metadata: dict) -> dict:
        """
        Load a dataset from Parquet file to Trino.
        
        Args:
            dataset_id: Socrata dataset ID
            parquet_path: Path to the Parquet file
            metadata: Dataset metadata
            
        Returns:
            Dict with result information
        """
        try:
            # Get dataset name
            dataset_name = metadata.get("name", f"Dataset {dataset_id}")
            
            # Prepare schema and table names
            schema_name = "iceberg"  # Use iceberg schema
            table_name = f"socrata_{dataset_id.replace('-', '_')}"
            full_table_name = f"{schema_name}.{table_name}"
            
            logger.info(f"Loading dataset {dataset_id} to Trino table {full_table_name}")
            
            # Drop existing table if it exists
            self.trino.execute_query(f"DROP TABLE IF EXISTS {full_table_name}")
            
            # Read Parquet file to get schema
            df = pd.read_parquet(parquet_path)
            
            # Map pandas dtypes to Trino types
            type_mapping = {
                'int64': 'BIGINT',
                'int32': 'INTEGER',
                'float64': 'DOUBLE',
                'float32': 'REAL',
                'bool': 'BOOLEAN',
                'datetime64[ns]': 'TIMESTAMP',
                'object': 'VARCHAR'
            }
            
            # Build column definitions
            column_defs = []
            for col in df.columns:
                # Get pandas dtype and map to Trino type
                pandas_type = str(df[col].dtype)
                trino_type = type_mapping.get(pandas_type, 'VARCHAR')
                
                # Handle special cases - columns containing timestamps but stored as objects
                if pandas_type == 'object' and len(df) > 0 and df[col].iloc[0] and isinstance(df[col].iloc[0], (pd.Timestamp, datetime)):
                    trino_type = 'TIMESTAMP'
                
                # Add to column definitions - escape column names
                column_defs.append(f'"{col}" {trino_type}')
            
            # Combine column definitions
            columns_sql = ', '.join(column_defs)
            
            # Create simplified table comment
            simple_comment = dataset_name
            if len(simple_comment) > 100:
                simple_comment = simple_comment[:100] + "..."
            
            # Properly escape single quotes in the comment
            simple_comment = simple_comment.replace("'", "''")
            
            # Create table
            create_table_sql = f"""
            CREATE TABLE {full_table_name} (
                {columns_sql}
            )
            COMMENT '{simple_comment}'
            """
            
            logger.info(f"Creating Trino table with SQL: {create_table_sql}")
            self.trino.execute_query(create_table_sql)
            
            # Skip adding detailed comment - it's causing SQL syntax issues
            # Instead, store the detailed description in the metadata table
            
            # Add column comments if available
            for col_info in metadata.get("column_details", []):
                col_name = col_info.get("name", "")
                col_desc = col_info.get("description", "")
                
                if col_name and col_desc and col_name in df.columns:
                    # Limit and sanitize description
                    col_desc = col_desc.replace("'", "''")
                    if len(col_desc) > 100:
                        col_desc = col_desc[:100] + "..."
                    
                    try:
                        comment_sql = f"COMMENT ON COLUMN {full_table_name}.\"{col_name}\" IS '{col_desc}'"
                        self.trino.execute_query(comment_sql)
                    except Exception as e:
                        logger.warning(f"Error adding comment for column {col_name}: {str(e)}")
            
            # Create a temporary external table pointing to the Parquet file
            parquet_location = f"'file://{os.path.abspath(parquet_path)}'"
            temp_table_name = f"{schema_name}.temp_{dataset_id.replace('-', '_')}"
            
            # Drop the temp table if it exists
            self.trino.execute_query(f"DROP TABLE IF EXISTS {temp_table_name}")
            
            # Create temp table
            create_temp_sql = f"""
            CREATE TABLE {temp_table_name}
            WITH (
                external_location = {parquet_location},
                format = 'PARQUET'
            )
            """
            
            logger.info(f"Creating temporary table with SQL: {create_temp_sql}")
            self.trino.execute_query(create_temp_sql)
            
            # Wait a moment for the table to be available
            time.sleep(2)
            
            # Insert data from temp table into the main table
            insert_sql = f"INSERT INTO {full_table_name} SELECT * FROM {temp_table_name}"
            logger.info(f"Inserting data with SQL: {insert_sql}")
            self.trino.execute_query(insert_sql)
            
            # Drop the temporary table
            self.trino.execute_query(f"DROP TABLE IF EXISTS {temp_table_name}")
            
            # Update cache with table information
            self.cache.update_dataset_table_info(
                dataset_id=dataset_id,
                schema_name=schema_name,
                table_name=table_name
            )
            
            # Get row count
            result = self.trino.execute_query(f"SELECT COUNT(*) FROM {full_table_name}")
            row_count = result[0][0] if result else 0
            
            logger.info(f"Successfully loaded {row_count} rows into Trino table {full_table_name}")
            
            return {
                "success": True,
                "schema_name": schema_name,
                "table_name": table_name,
                "row_count": row_count
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id} to Trino: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

# Singleton instance for global access
_loader_instance = None

def get_loader():
    """Get the singleton DataLoader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = DataLoader()
    return _loader_instance

def load_dataset(dataset_id: str, force_reload: bool = False) -> dict:
    """
    Simple function to load a dataset from Socrata to Trino
    
    Args:
        dataset_id: The Socrata dataset ID
        force_reload: Whether to force reload even if cached
        
    Returns:
        Dict with result information
    """
    loader = get_loader()
    return loader.load_dataset(dataset_id, force_reload)

def main():
    """Main entry point for data loader"""
    parser = argparse.ArgumentParser(description="Load datasets from Socrata to Trino")
    parser.add_argument('dataset_ids', nargs='*', help='Socrata dataset IDs to load')
    parser.add_argument('--file', '-f', help='File containing dataset IDs, one per line')
    parser.add_argument('--force', action='store_true', help='Force reload even if cached')
    parser.add_argument('--all', action='store_true', help='Load all datasets in cache')
    args = parser.parse_args()
    
    # Initialize data loader
    loader = DataLoader()
    
    # Collect dataset IDs to load
    dataset_ids = set(args.dataset_ids)
    
    # Add dataset IDs from file if specified
    if args.file:
        try:
            with open(args.file, 'r') as f:
                file_ids = [line.strip() for line in f if line.strip()]
                dataset_ids.update(file_ids)
        except Exception as e:
            logger.error(f"Error reading dataset IDs from file {args.file}: {str(e)}")
    
    # If --all flag is used, load all datasets in cache
    if args.all:
        cached_datasets = loader.cache.get_all_cached_datasets()
        dataset_ids.update([d.get('dataset_id') for d in cached_datasets if d.get('dataset_id')])
    
    # If still no dataset IDs, use a default
    if not dataset_ids:
        logger.warning("No dataset IDs provided, using default datasets")
        dataset_ids = ["5694-9szk", "kz4z-fdn2", "sqcr-6vxa", "vfnx-vebw"]
    
    # Load each dataset
    results = []
    for dataset_id in dataset_ids:
        result = loader.load_dataset(dataset_id, force_reload=args.force)
        results.append(result)
        
        # Add a short delay between loads
        time.sleep(1)
    
    # Print summary
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Successfully loaded {success_count} of {len(results)} datasets")
    
    # Print details of failed loads
    failed = [r for r in results if not r["success"]]
    if failed:
        logger.error("Failed loads:")
        for f in failed:
            logger.error(f"  {f['dataset_id']}: {f['error']}")

if __name__ == "__main__":
    main() 