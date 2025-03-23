"""
Socrata Open Data API Loader
============================

This module provides a simplified interface for loading data from Socrata Open Data API
endpoints into Parquet files, which can then be loaded into Trino with Iceberg.

Main Features:
-------------
- Authenticated or anonymous access to Socrata APIs
- Efficient chunked processing for large datasets
- Direct conversion to Parquet format
- Dataset metadata handling
"""

import os
import time
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sodapy import Socrata
from typing import Dict, List, Any, Optional, Tuple, Generator, Union, Iterator
import re
from datetime import datetime
import tempfile
import gc
from logger_config import setup_logger
from app_config import get_config
from trino_connector import get_trino_connection_manager
from cache_manager import DatasetCacheManager

# Create logger
logger = setup_logger(__name__)
config = get_config()

class SocrataLoader:
    """Simplified Socrata data loader that extracts data directly to Parquet.
    
    This class handles the process of extracting data from Socrata APIs
    and converting it to Parquet format for loading into Trino/Iceberg.
    
    Key features:
    - Authenticated or anonymous access to Socrata APIs
    - Chunked processing for large datasets
    - Direct conversion to Parquet format
    - Dataset metadata fetching
    """
    
    def __init__(self):
        """Initialize the Socrata data loader with configuration."""
        self.config = get_config()
        self.domain = self.config.socrata_domain
        self.app_token = self.config.socrata_app_token
        self.api_key_id = self.config.socrata_api_key_id
        self.api_key_secret = self.config.socrata_api_key_secret
        self.timeout = self.config.socrata_timeout  # Use from config
        self.batch_size = self.config.socrata_batch_size  # Add batch_size from config
        self.client = None  # Initialize to None, will be created when needed
        
        # Ensure the temp directory exists
        os.makedirs(self.config.temp_dir, exist_ok=True)
        
        # For handling Trino cleanup
        self.trino = get_trino_connection_manager()
        self.cache = DatasetCacheManager()
        self.logger = logger  # Add logger reference
    
    def _init_socrata_client(self):
        """
        Initialize the Socrata client with authentication
        
        Returns:
            Authenticated Socrata client
        """
        try:
            # Check if we have authentication credentials
            if self.api_key_id and self.api_key_secret:
                logger.info(f"Initialized authenticated Socrata client for {self.domain}")
                return Socrata(
                    self.domain,
                    self.app_token,
                    username=self.api_key_id,
                    password=self.api_key_secret,
                    timeout=self.timeout
                )
            else:
                # No authentication, use app token only
                logger.info(f"Initialized unauthenticated Socrata client for {self.domain}")
                return Socrata(
                    self.domain,
                    self.app_token,
                    timeout=self.timeout
                )
        except Exception as e:
            logger.error(f"Error creating Socrata client: {str(e)}")
            raise
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get metadata for a Socrata dataset
        
        Args:
            dataset_id: Socrata dataset ID
            
        Returns:
            Dictionary with dataset metadata
        """
        try:
            logger.info(f"Fetching metadata for dataset {dataset_id}")
            
            # Initialize client if needed
            if not self.client:
                self.client = self._init_socrata_client()
            
            # Get metadata from Socrata
            metadata = self.client.get_metadata(dataset_id)
            
            # Extract key information
            result = {
                "dataset_id": dataset_id,
                "name": metadata.get("name", ""),
                "description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "domain": self.domain,
                "created_at": metadata.get("createdAt", ""),
                "updated_at": metadata.get("rowsUpdatedAt", ""),
                "license": metadata.get("license", {}).get("name", ""),
                "columns": len(metadata.get("columns", [])),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching metadata for dataset {dataset_id}: {str(e)}")
            return {
                "dataset_id": dataset_id,
                "error": str(e)
            }
    
    def discover_datasets(self, limit: int = 10, offset: int = 0, 
                        domain_category: str = None) -> List[Dict[str, Any]]:
        """Discover available datasets from Socrata"""
        try:
            logger.info(f"Discovering datasets from {self.domain}, limit={limit}, offset={offset}")
            
            # Build query
            query = {}
            if domain_category:
                query["domain_category"] = domain_category
            
            # Fetch datasets
            datasets = self.client.datasets(limit=limit, offset=offset, **query)
            logger.info(f"Found {len(datasets)} datasets")
            
            return datasets
        except Exception as e:
            logger.error(f"Error discovering datasets: {e}")
            return []
    
    def _clean_column_name(self, name: str) -> str:
        """Clean a column name to be SQL-friendly"""
        # Replace spaces and special characters with underscores
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a number
        if clean and clean[0].isdigit():
            clean = 'c_' + clean
        # Ensure it's not empty
        if not clean:
            clean = 'column'
        # Lowercase for consistency
        return clean.lower()
    
    def _fetch_data_in_chunks(self, dataset_id: str) -> Iterator[pd.DataFrame]:
        """
        Fetch data from Socrata in chunks to avoid memory issues
        
        Args:
            dataset_id: Socrata dataset ID
            
        Yields:
            DataFrame with chunk of data
        """
        try:
            # Initialize client if needed
            if not self.client:
                self.client = self._init_socrata_client()
            
            # Get total row count - don't use SoQL query
            # Instead use metadata or just start fetching
            logger.info(f"Fetching dataset {dataset_id} in chunks of {self.batch_size}")
            
            # Start with offset 0
            offset = 0
            more_data = True
            
            while more_data:
                try:
                    # Get data chunk
                    chunk = self.client.get(dataset_id, limit=self.batch_size, offset=offset)
                    
                    # If we got data, convert to DataFrame and yield
                    if chunk and len(chunk) > 0:
                        df = pd.DataFrame.from_records(chunk)
                        offset += len(df)
                        logger.info(f"Fetched chunk with {len(df)} rows. Total so far: {offset}")
                        yield df
                    else:
                        # No more data
                        more_data = False
                        logger.info(f"Finished fetching data. Total rows: {offset}")
                except Exception as e:
                    logger.error(f"Error fetching chunk at offset {offset}: {str(e)}")
                    more_data = False
                
        except Exception as e:
            logger.error(f"Error fetching data for dataset {dataset_id}: {str(e)}")
            # Yield empty DataFrame to avoid breaking the iterator
            yield pd.DataFrame()
    
    def _cleanup_existing_data(self, dataset_id: str):
        """
        Clean up any existing data for this dataset in Trino
        to ensure we always have fresh data with current schema.
        
        Args:
            dataset_id: The Socrata dataset ID
        """
        try:
            # Clean up cache entry
            cached_info = self.cache.get_dataset_info(dataset_id)
            if cached_info:
                logger.info(f"Cleaning up cached data for dataset {dataset_id}")
                
                # Remove from Trino if schema/table info is available
                if "schema_name" in cached_info and "table_name" in cached_info:
                    schema_name = cached_info.get("schema_name")
                    table_name = cached_info.get("table_name")
                    
                    # Drop the table if it exists
                    logger.info(f"Dropping Trino table {schema_name}.{table_name} if exists")
                    self.trino.execute_query(f"DROP TABLE IF EXISTS {schema_name}.{table_name}")
                
                # Clear cache entry
                self.cache.remove_dataset(dataset_id)
                
        except Exception as e:
            logger.warning(f"Error during cleanup for dataset {dataset_id}: {e}")
            # Continue with the download even if cleanup fails
    
    def download_dataset_as_parquet(self, dataset_id: str, 
                                  output_path: str) -> Dict[str, Any]:
        """
        Download a dataset from Socrata and save it as a Parquet file.
        Always creates a fresh download, deleting any existing files.
        
        Args:
            dataset_id: The Socrata dataset identifier (e.g., '8wbx-tsch')
            output_path: Path where the Parquet file should be saved
            
        Returns:
            Dictionary with metadata about the download operation
        """
        result = {
            "dataset_id": dataset_id,
            "success": False,
            "start_time": datetime.now().isoformat(),
            "output_path": output_path
        }
        
        try:
            logger.info(f"Downloading dataset {dataset_id} to {output_path}")
            
            # First, fetch metadata for this dataset
            metadata = self.get_dataset_metadata(dataset_id)
            result.update(metadata)
            
            # Clean up existing data for clean slate
            self._cleanup_existing_data(dataset_id)
            
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Temporary file for combined results
                temp_path = os.path.join(temp_dir, f"{dataset_id}.parquet")
                
                # Delete existing file if it exists
                if os.path.exists(output_path):
                    logger.info(f"Removing existing file: {output_path}")
                    os.remove(output_path)
                
                # Create parent directory for output if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Initialize variables for tracking
                chunk_count = 0
                total_rows = 0
                writer = None
                schema = None
                all_data = None
                
                # Process each chunk
                for chunk_df in self._fetch_data_in_chunks(dataset_id):
                    if chunk_df.empty:
                        logger.warning(f"Received empty chunk for dataset {dataset_id}")
                        continue
                        
                    chunk_count += 1
                    total_rows += len(chunk_df)
                    
                    # For the first chunk, save the data
                    if all_data is None:
                        all_data = chunk_df
                    else:
                        # For subsequent chunks, append to existing data
                        all_data = pd.concat([all_data, chunk_df], ignore_index=True)
                    
                    # Log progress
                    logger.info(f"Processed chunk {chunk_count} with {len(chunk_df)} rows. Total rows: {total_rows}")
                    
                    # Free memory
                    del chunk_df
                    gc.collect()
                
                # Write the final combined data to Parquet file
                if all_data is not None and not all_data.empty:
                    try:
                        # Save to permanent location
                        all_data.to_parquet(output_path, index=False)
                        logger.info(f"Moved Parquet file to {output_path}")
                    except Exception as e:
                        logger.error(f"Error writing to Parquet file: {str(e)}")
                        result["error"] = str(e)
                        return result
                else:
                    logger.error("No data was fetched from the dataset")
                    result["error"] = "No data was fetched from the dataset"
                    return result
                
                # Update result with success
                result["success"] = True
                result["row_count"] = total_rows
                result["chunk_count"] = chunk_count
                result["file_size"] = os.path.getsize(output_path)
                result["end_time"] = datetime.now().isoformat()
                
                logger.info(f"Successfully downloaded dataset {dataset_id} with {total_rows} rows")
                return result
                
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {str(e)}")
            result["error"] = str(e)
            result["end_time"] = datetime.now().isoformat()
            return result

# Backwards compatibility class for existing code
class SocrataToTrinoETL:
    """Legacy compatibility class that wraps the new SocrataLoader.
    
    This class maintains backward compatibility with existing code
    while delegating operations to the new simplified SocrataLoader.
    """
    
    def __init__(self, api_key_id=None, api_key_secret=None, domain=None,
                 minio_endpoint=None, minio_access_key=None, minio_secret_key=None,
                 trino_host=None, trino_port=None, trino_user=None, trino_catalog=None,
                 cache_dir="data_cache"):
        """Initialize with compatibility parameters"""
        self.loader = SocrataLoader()
        self.cache_manager = DatasetCacheManager(cache_dir=cache_dir)
        logger.warning("SocrataToTrinoETL is deprecated. Please use SocrataLoader instead.")
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a Socrata dataset"""
        return self.loader.get_dataset_metadata(dataset_id)
    
    def discover_datasets(self, domain: Optional[str] = None, category: Optional[str] = None, 
                         limit: int = 10) -> List[Dict[str, Any]]:
        """Discover available datasets from Socrata"""
        return self.loader.discover_datasets(limit=limit, domain_category=category)
    
    # Minimal stub methods to support backward compatibility
    def _ensure_bucket_exists(self):
        """Legacy method stub"""
        logger.warning("_ensure_bucket_exists is deprecated")
        return True
        
    def _ensure_metadata_schema(self):
        """Legacy method stub"""
        logger.warning("_ensure_metadata_schema is deprecated")
        return True
    
    def create_trino_table_from_dataset(self, dataset_id, overwrite=False):
        """Legacy method stub that logs a warning"""
        logger.warning(f"create_trino_table_from_dataset is deprecated. Use the data_loader.py instead.")
        return {
            "success": False,
            "error": "Method deprecated. Use data_loader.py instead.",
            "dataset_id": dataset_id
        }
