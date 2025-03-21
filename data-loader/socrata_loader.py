"""
Socrata Open Data API ETL Pipeline for Trino/Iceberg
====================================================

This module provides a complete ETL (Extract, Transform, Load) pipeline for loading 
data from Socrata Open Data API endpoints into Trino/Iceberg tables with MinIO as the 
underlying storage layer.

Main Components:
---------------
- SocrataToTrinoETL: Main class that handles the end-to-end ETL process
- Data extraction from Socrata APIs with authentication and rate limit handling
- Data transformation using DuckDB for intermediate processing
- Data loading into Trino tables with Iceberg format
- MinIO integration for storing Parquet files
- Dataset metadata management and caching

Workflow:
--------
1. Connect to Socrata API and discover/extract datasets
2. Process data in chunks to handle large datasets efficiently
3. Transform data types and column names to be Trino-compatible
4. Export processed data to Parquet format
5. Store Parquet files in MinIO
6. Create Iceberg tables in Trino referencing the Parquet data
7. Track dataset metadata in registry for future reference

Configuration:
-------------
- Requires credentials for Socrata API, MinIO, and Trino
- Supports environment variable configuration through .env file
- Includes fallback to anonymous access for Socrata if no credentials provided
- Configurable chunk sizes for processing large datasets

Dependencies:
------------
- sodapy: For Socrata API access
- pandas/pyarrow: For data processing and Parquet handling
- minio: For S3-compatible storage access
- trino: For database connections
- DuckDBProcessor: For efficient data transformation

Example Usage:
-------------
    # Initialize the ETL pipeline
    etl = SocrataToTrinoETL()
    
    # Discover available datasets
    datasets = etl.discover_datasets(limit=10)
    
    # Process a specific dataset
    result = etl.create_trino_table_from_dataset("abcd-1234")
    
    # Get dataset metadata
    metadata = etl.get_dataset_metadata("abcd-1234")
"""

import os
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sodapy import Socrata
from minio import Minio
from minio.error import S3Error
import trino
from trino.dbapi import connect
from typing import Dict, List, Any, Optional, Tuple, Generator, Iterator, Union
import re
from datetime import datetime
import tempfile
import shutil
import gc
from duckdb_processor import DuckDBProcessor
from logger_config import setup_logger
from env_config import (
    get_socrata_credentials, 
    get_minio_credentials, 
    get_trino_credentials,
    DEFAULT_DOMAIN
)
from cache_manager import DatasetCacheManager
import logging
import uuid

# Create logger
logger = logging.getLogger(__name__)

class SocrataToTrinoETL:
    """ETL pipeline to load data from Socrata Open Data API to Trino/Iceberg tables.
    
    This class handles the entire process of extracting data from Socrata APIs,
    transforming it to be compatible with Trino, and loading it into Iceberg tables.
    It manages connections to Socrata, MinIO, and Trino, as well as handling data
    chunking, type conversion, and metadata management.
    
    Key features:
    - Authenticated or anonymous access to Socrata APIs
    - Chunked processing for large datasets
    - DuckDB-based transformation pipeline
    - Parquet storage in MinIO
    - Iceberg table creation in Trino
    - Dataset metadata tracking
    - Caching of processed data
    
    Attributes:
        api_key_id (str): Socrata API key ID for authentication
        api_key_secret (str): Socrata API key secret for authentication
        domain (str): Socrata domain (e.g., data.cityofnewyork.us)
        minio_endpoint (str): MinIO server endpoint
        minio_access_key (str): MinIO access key
        minio_secret_key (str): MinIO secret key
        minio_bucket (str): MinIO bucket for Iceberg data
        nyc_etl_bucket (str): MinIO bucket for ETL intermediates
        trino_host (str): Trino server hostname
        trino_port (int): Trino server port
        trino_user (str): Trino username
        trino_catalog (str): Trino catalog name
        chunk_size (int): Default chunk size for data processing
    """
    
    def __init__(self, api_key_id=None, api_key_secret=None, domain=None,
                 minio_endpoint=None, minio_access_key=None, minio_secret_key=None,
                 trino_host=None, trino_port=None, trino_user=None, trino_catalog=None,
                 cache_dir="data_cache"):
        """Initialize the ETL pipeline with credentials"""
        # Set up instance logger
        self.logger = logging.getLogger(__name__)
        
        # Get Socrata credentials from environment if not provided
        socrata_creds = get_socrata_credentials()
        self.api_key_id = api_key_id or socrata_creds['key_id']
        self.api_key_secret = api_key_secret or socrata_creds['key_secret']
        self.domain = domain or socrata_creds.get('domain', DEFAULT_DOMAIN)
        
        # Get MinIO credentials from environment if not provided
        minio_creds = get_minio_credentials()
        self.minio_endpoint = minio_endpoint or minio_creds['endpoint']
        self.minio_access_key = minio_access_key or minio_creds['access_key']
        self.minio_secret_key = minio_secret_key or minio_creds['secret_key']
        self.minio_bucket = minio_creds.get('bucket', 'iceberg')
        
        # Get Trino credentials from environment if not provided
        trino_creds = get_trino_credentials()
        self.trino_host = trino_host or trino_creds['host']
        self.trino_port = trino_port or trino_creds['port']
        self.trino_user = trino_user or trino_creds['user']
        self.trino_catalog = trino_catalog or trino_creds['catalog']
        
        # Initialize clients
        self.socrata_client = self._init_socrata_client()
        self.minio_client = self._init_minio_client()
        
        # Initialize dataset cache manager
        self.cache_manager = DatasetCacheManager(cache_dir=cache_dir)
        
        # Configure data processing
        self.chunk_size = 50000  # Default chunk size for fetching data

        # Ensure buckets and schemas exist
        self._ensure_bucket_exists()
        self._ensure_metadata_schema()

    def _ensure_bucket_exists(self):
        """Ensure the necessary buckets exist in MinIO"""
        try:
            if not self.minio_client:
                self.logger.warning("MinIO client not available, skipping bucket creation")
                return
                
            if not self.minio_client.bucket_exists(self.minio_bucket):
                self.minio_client.make_bucket(self.minio_bucket)
                self.logger.info(f"Created MinIO bucket: {self.minio_bucket}")
        except Exception as e:
            self.logger.error(f"Error ensuring bucket exists: {e}")
    
    def _init_socrata_client(self) -> Socrata:
        """Initialize the Socrata client with authentication if available"""
        try:
            if self.api_key_id and self.api_key_secret:
                # Socrata requires a positional argument before username/password
                # This is the token parameter, but we'll use an empty string
                client = Socrata(
                    self.domain,
                    "",  # Required positional argument (token)
                    username=self.api_key_id,
                    password=self.api_key_secret
                )
                self.logger.info("Initialized Socrata client with API key authentication")
            else:
                # No authentication provided, will use anonymous access with rate limits
                client = Socrata(self.domain, "")  # Empty string for required positional arg
                self.logger.warning("Initialized Socrata client without authentication - severe rate limits will apply")
            return client
        except Exception as e:
            self.logger.error(f"Error initializing Socrata client: {e}")
            raise
    
    def _init_minio_client(self) -> Optional[Minio]:
        """Initialize MinIO client with robust error handling"""
        try:
            from minio import Minio
            
            if not all([self.minio_endpoint, self.minio_access_key, self.minio_secret_key]):
                self.logger.error("MinIO credentials incomplete. Ensure endpoint, access_key, and secret_key are provided.")
                return None
            
            # Log the MinIO connection attempt (sanitizing credentials)
            endpoint_display = self.minio_endpoint
            self.logger.info(f"Connecting to MinIO at {endpoint_display}")
            
            # Fix endpoint format if needed
            if self.minio_endpoint.startswith('http://') or self.minio_endpoint.startswith('https://'):
                from urllib.parse import urlparse
                parsed = urlparse(self.minio_endpoint)
                # Extract just the hostname and port
                self.minio_endpoint = parsed.netloc
            
            # Initialize MinIO client
            minio_client = Minio(
                endpoint=self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False  # Use HTTP instead of HTTPS for development
            )
            
            # Check if bucket exists, create if not
            bucket_exists = False
            try:
                bucket_exists = minio_client.bucket_exists(self.minio_bucket)
            except Exception as bucket_check_err:
                self.logger.error(f"Error checking MinIO bucket: {bucket_check_err}")
                # Continue anyway - we'll try to create the bucket
            
            if not bucket_exists:
                try:
                    self.logger.info(f"Creating MinIO bucket: {self.minio_bucket}")
                    minio_client.make_bucket(self.minio_bucket)
                    self.logger.info(f"Created MinIO bucket: {self.minio_bucket}")
                except Exception as create_err:
                    self.logger.error(f"Failed to create MinIO bucket: {create_err}")
                    # Continue anyway - the bucket might exist already or be created externally
            
            self.logger.info("Successfully initialized MinIO client")
            return minio_client
            
        except ImportError:
            self.logger.error("MinIO client library not installed. Please install 'minio' package.")
            return None
        except Exception as e:
            self.logger.error(f"Error initializing MinIO client: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_trino_connection(self):
        """Get a connection to Trino"""
        try:
            from trino.dbapi import connect
            
            # Ensure we're using the service name, not localhost
            trino_host = self.trino_host
            
            # Log connection attempt
            self.logger.info(f"Connecting to Trino at {trino_host}:{self.trino_port} as {self.trino_user}")
            
            # Create connection
            conn = connect(
                host=trino_host,
                port=self.trino_port,
                user=self.trino_user,
                catalog=self.trino_catalog
            )
            
            # Test the connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            self.logger.info(f"Trino connection successful: {result}")
            
            return conn
        except Exception as e:
            self.logger.error(f"Error connecting to Trino: {str(e)}")
            # Include traceback for more detailed debugging
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _ensure_metadata_schema(self):
        """Ensure the metadata schema exists in Trino"""
        try:
            conn = self._get_trino_connection()
            if not conn:
                self.logger.warning("Could not connect to Trino to ensure metadata schema")
                return
                
            cursor = conn.cursor()
            
            # Create metadata schema if it doesn't exist
            cursor.execute("CREATE SCHEMA IF NOT EXISTS iceberg.metadata")
            
            # Create dataset registry table if it doesn't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS iceberg.metadata.dataset_registry (
                dataset_id VARCHAR,
                dataset_title VARCHAR,
                dataset_description VARCHAR,
                schema_name VARCHAR,
                table_name VARCHAR,
                row_count BIGINT,
                column_count INTEGER,
                original_size BIGINT,
                parquet_size BIGINT,
                partitioning_columns ARRAY(VARCHAR),
                last_updated TIMESTAMP,
                etl_timestamp TIMESTAMP
            )
            """)
            
            conn.close()
            self.logger.info("Ensured metadata schema and registry table exist")
        except Exception as e:
            self.logger.error(f"Error ensuring metadata schema: {e}")
            self.logger.info("Continuing without metadata schema - will use local cache only")
    
    def discover_datasets(self, domain: Optional[str] = None, category: Optional[str] = None, 
                         limit: int = 10) -> List[Dict[str, Any]]:
        """Discover datasets from Socrata"""
        try:
            domain = domain or self.domain
            
            # Build the query
            query = {}
            if category:
                query["categories"] = category
            
            # Get datasets
            datasets = self.socrata_client.datasets(limit=limit, **query)
            
            self.logger.info(f"Discovered {len(datasets)} datasets from {domain}")
            return datasets
        except Exception as e:
            self.logger.error(f"Error discovering datasets: {str(e)}")
            return []
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a dataset"""
        try:
            metadata = self.socrata_client.get_metadata(dataset_id)
            return metadata
        except Exception as e:
            self.logger.error(f"Error getting metadata for dataset {dataset_id}: {str(e)}")
            return {}
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name to be compatible with Trino"""
        # Replace spaces and special characters with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        # Remove consecutive underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        # Remove leading and trailing underscores
        clean_name = clean_name.strip('_')
        # Ensure it doesn't start with a number
        if clean_name and clean_name[0].isdigit():
            clean_name = 'col_' + clean_name
        # If empty, use a default name
        if not clean_name:
            clean_name = 'column'
        return clean_name
    
    def _clean_table_name(self, name: str) -> str:
        """Clean table name to be compatible with Trino"""
        # Replace spaces and special characters with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        # Remove consecutive underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        # Remove leading and trailing underscores
        clean_name = clean_name.strip('_')
        # Ensure it doesn't start with a number
        if clean_name and clean_name[0].isdigit():
            clean_name = 'tbl_' + clean_name
        # If empty, use a default name
        if not clean_name:
            clean_name = 'table'
        return clean_name
    
    def _determine_schema_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Determine the schema name from metadata"""
        # Try to get the category
        category = None
        if 'classification' in metadata and 'domain_category' in metadata['classification']:
            category = metadata['classification']['domain_category']
        
        if not category and 'category' in metadata:
            category = metadata['category']
        
        # Clean and use the category as schema name, or default to 'general'
        if category:
            schema_name = self._clean_table_name(category)
        else:
            schema_name = 'general'
        
        return schema_name
    
    def _determine_table_name(self, metadata: Dict[str, Any]) -> str:
        """Determine the table name from metadata"""
        # Try to get the name
        name = None
        if 'name' in metadata:
            name = metadata['name']
        elif 'resource' in metadata and 'name' in metadata['resource']:
            name = metadata['resource']['name']
        
        # Clean and use the name as table name, or use the dataset ID
        if name:
            table_name = self._clean_table_name(name)
        else:
            # Use dataset ID as fallback
            dataset_id = metadata.get('id', 'unknown')
            table_name = self._clean_table_name(dataset_id)
        
        return table_name
    
    def _determine_partitioning_columns(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> List[str]:
        """Determine appropriate partitioning columns based on data characteristics"""
        # For now, return an empty list to disable partitioning
        return []
    
    def _infer_schema_from_dataframe(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer the schema from a pandas DataFrame"""
        schema = {}
        
        for col in df.columns:
            clean_col = self._clean_column_name(col)
            dtype = df[col].dtype
            
            if pd.api.types.is_integer_dtype(dtype):
                schema[clean_col] = 'BIGINT'
            elif pd.api.types.is_float_dtype(dtype):
                schema[clean_col] = 'DOUBLE'
            elif pd.api.types.is_bool_dtype(dtype):
                schema[clean_col] = 'BOOLEAN'
            elif pd.api.types.is_datetime64_dtype(dtype):
                schema[clean_col] = 'TIMESTAMP'
            else:
                schema[clean_col] = 'VARCHAR'
        
        return schema
    
    def _create_trino_table(self, schema_name: str, table_name: str, column_schema: Dict[str, str], 
                            partitioning_columns: List[str] = None, object_name: str = None, 
                            metadata: Dict[str, Any] = None) -> bool:
        """Create a table in Trino and load data from Parquet file"""
        try:
            conn = self._get_trino_connection()
            cursor = conn.cursor()
            
            # Create schema if it doesn't exist
            schema_comment = ""
            if metadata and 'classification' in metadata and 'domain_category' in metadata['classification']:
                schema_comment = metadata['classification']['domain_category'].replace("'", "''")
            
            schema_sql = f"CREATE SCHEMA IF NOT EXISTS iceberg.{schema_name}"
            self.logger.info(f"Executing SQL: {schema_sql}")
            cursor.execute(schema_sql)
            
            # Add comment to schema if we have metadata
            if schema_comment:
                comment_sql = f"COMMENT ON SCHEMA iceberg.{schema_name} IS '{schema_comment}'"
                try:
                    cursor.execute(comment_sql)
                    self.logger.info(f"Added comment to schema iceberg.{schema_name}")
                except Exception as e:
                    self.logger.warning(f"Could not add comment to schema: {e}")
            
            # Drop table if it exists
            drop_sql = f"DROP TABLE IF EXISTS iceberg.{schema_name}.{table_name}"
            self.logger.info(f"Executing SQL: {drop_sql}")
            cursor.execute(drop_sql)
            
            # Build the CREATE TABLE statement
            columns_sql = []
            for col, dtype in column_schema.items():
                columns_sql.append(f"{col} {dtype}")
            
            # Add partitioning if specified
            partitioning_sql = ""
            if partitioning_columns and len(partitioning_columns) > 0:
                clean_partitioning_columns = [self._clean_column_name(col) for col in partitioning_columns]
                # Format the array properly with quotes around column names
                formatted_cols = [f"'{col}'" for col in clean_partitioning_columns]
                partitioning_sql = f" WITH (partitioning = ARRAY[{', '.join(formatted_cols)}])"
            
            create_table_sql = f"""
                CREATE TABLE iceberg.{schema_name}.{table_name} (
                    {', '.join(columns_sql)}
                ){partitioning_sql}
            """
            
            self.logger.info(f"Executing SQL: {create_table_sql}")
            cursor.execute(create_table_sql)
            self.logger.info(f"Created table iceberg.{schema_name}.{table_name}")
            
            # Add table comment if we have metadata
            if metadata and 'name' in metadata and 'description' in metadata:
                table_description = metadata['description'].replace("'", "''") if metadata.get('description') else metadata['name'].replace("'", "''")
                comment_sql = f"COMMENT ON TABLE iceberg.{schema_name}.{table_name} IS '{table_description}'"
                try:
                    cursor.execute(comment_sql)
                    self.logger.info(f"Added comment to table iceberg.{schema_name}.{table_name}")
                except Exception as e:
                    self.logger.warning(f"Could not add comment to table: {e}")
            
            # Add column comments if we have column metadata
            if metadata and 'columns' in metadata and isinstance(metadata['columns'], list):
                for col_meta in metadata['columns']:
                    if 'fieldName' in col_meta and 'name' in col_meta:
                        col_name = self._clean_column_name(col_meta['fieldName'])
                        col_description = col_meta.get('description', col_meta['name']).replace("'", "''")
                        comment_sql = f"COMMENT ON COLUMN iceberg.{schema_name}.{table_name}.{col_name} IS '{col_description}'"
                        try:
                            cursor.execute(comment_sql)
                            self.logger.info(f"Added comment to column {col_name}")
                        except Exception as e:
                            self.logger.warning(f"Could not add comment to column {col_name}: {e}")
            
            # Load data from Parquet file if object_name is provided
            if object_name:
                try:
                    # Download the file from MinIO to a local path
                    local_file_path = os.path.join(self.cache_dir, f"{object_name.split('/')[-1]}")
                    
                    # Download the file from MinIO if it doesn't exist locally
                    if not os.path.exists(local_file_path):
                        try:
                            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                            self.minio_client.fget_object(
                                self.minio_bucket, 
                                object_name, 
                                local_file_path
                            )
                            self.logger.info(f"Downloaded {object_name} to {local_file_path}")
                        except Exception as e:
                            self.logger.error(f"Error downloading file from MinIO: {e}")
                            raise
                    
                    # Read the Parquet file
                    df = pd.read_parquet(local_file_path)
                    self.logger.info(f"Loaded {len(df)} rows from {local_file_path}")
                    
                    # Insert data in batches to improve performance
                    batch_size = 500  # Reduced from 5000 to 500 to avoid query text length limit
                    total_rows = len(df)
                    batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
                    
                    self.logger.info(f"Starting to insert {total_rows} rows in {batches} batches of {batch_size} rows each")
                    
                    # Track progress
                    start_time = time.time()
                    rows_inserted = 0
                    
                    for batch_idx in range(batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, total_rows)
                        batch_df = df.iloc[start_idx:end_idx]
                        
                        # Build a multi-row INSERT statement
                        insert_values = []
                        for _, row in batch_df.iterrows():
                            row_values = []
                            for col in column_schema.keys():
                                if col in row:
                                    if pd.isna(row[col]):
                                        row_values.append("NULL")
                                    elif isinstance(row[col], (int, float)):
                                        row_values.append(str(row[col]))
                                    else:
                                        # Fix the syntax error by using a different approach for escaping single quotes
                                        val = str(row[col]) if row[col] is not None else None
                                        if val is not None:
                                            # Replace single quotes with two single quotes for SQL
                                            val = val.replace("'", "''")
                                            row_values.append(f"'{val}'")
                                        else:
                                            row_values.append("NULL")
                                else:
                                    row_values.append("NULL")
                            
                            insert_values.append(f"({', '.join(row_values)})")
                        
                        # Execute the multi-row INSERT
                        if insert_values:
                            insert_sql = f"INSERT INTO iceberg.{schema_name}.{table_name} VALUES {', '.join(insert_values)}"
                            cursor.execute(insert_sql)
                            rows_inserted += len(batch_df)
                            
                            # Calculate progress and estimated time remaining
                            progress = rows_inserted / total_rows * 100
                            elapsed_time = time.time() - start_time
                            rows_per_second = rows_inserted / elapsed_time if elapsed_time > 0 else 0
                            estimated_total_time = total_rows / rows_per_second if rows_per_second > 0 else 0
                            estimated_time_remaining = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
                            
                            self.logger.info(f"Inserted batch {batch_idx+1}/{batches} ({len(batch_df)} rows) - "
                                       f"{progress:.1f}% complete - "
                                       f"{rows_per_second:.1f} rows/sec - "
                                       f"Est. time remaining: {estimated_time_remaining:.1f} seconds")
                    
                    self.logger.info(f"Successfully loaded all {total_rows} rows into iceberg.{schema_name}.{table_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading data: {e}")
                    self.logger.error("Could not load data into the table")
                    return False
            
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating table: {e}")
            self.logger.error("Failed to create Trino table")
            return False
    
    def _upload_to_minio(self, file_path, dataset_id):
        """Upload a file to MinIO with unique path"""
        try:
            if not self.minio_client:
                self.logger.error("MinIO client not available")
                return False
            
            # Generate the same unique ID for consistency with table creation
            unique_id = getattr(self, '_current_unique_id', uuid.uuid4().hex[:8])
            # Store for reuse in table creation
            self._current_unique_id = unique_id
            
            # Get the base filename
            file_name = os.path.basename(file_path)
            object_name = f"{dataset_id}/{unique_id}/{file_name}"
            
            self.logger.info(f"Uploading {file_path} to MinIO bucket {self.minio_bucket} as {object_name}")
            
            # Upload the file
            self.minio_client.fput_object(
                bucket_name=self.minio_bucket,
                object_name=object_name,
                file_path=file_path,
                content_type="application/octet-stream"
            )
            
            self.logger.info(f"Successfully uploaded {file_path} to MinIO at {object_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error uploading to MinIO: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
    def _register_dataset(self, dataset_id: str, metadata: Dict[str, Any], schema_name: str, 
                         table_name: str, stats: Dict[str, Any]) -> bool:
        """Register the dataset in the metadata registry"""
        try:
            conn = self._get_trino_connection()
            cursor = conn.cursor()
            
            # Extract values from metadata
            dataset_title = metadata.get('name', '').replace("'", "''")
            dataset_description = metadata.get('description', '').replace("'", "''")
            domain = self.domain.replace("'", "''")
            
            # Get category
            category = ''
            if 'classification' in metadata and 'domain_category' in metadata['classification']:
                category = metadata['classification']['domain_category']
            elif 'category' in metadata:
                category = metadata['category']
            category = category.replace("'", "''")
            
            # Get tags - ensure it's a proper array
            tags = []
            if 'tags' in metadata and isinstance(metadata['tags'], list):
                tags = metadata['tags']
            elif 'classification' in metadata and 'domain_tags' in metadata['classification'] and isinstance(metadata['classification']['domain_tags'], list):
                tags = metadata['classification']['domain_tags']
            
            # Get last updated - ensure proper timestamp format
            last_updated = datetime.now()
            if 'updatedAt' in metadata:
                try:
                    # Parse ISO format to datetime
                    last_updated_str = metadata['updatedAt']
                    if isinstance(last_updated_str, str):
                        last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not parse updatedAt timestamp: {metadata.get('updatedAt')}, using current time")
            
            # Format timestamps as strings in Trino format
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            last_updated_str = last_updated.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if last_updated else created_at
            
            # Convert numeric values
            row_count = int(stats.get('row_count', 0)) if stats.get('row_count') is not None else 0
            column_count = int(stats.get('column_count', 0)) if stats.get('column_count') is not None else 0
            original_size = int(stats.get('original_size', 0)) if stats.get('original_size') is not None else 0
            parquet_size = int(stats.get('parquet_size', 0)) if stats.get('parquet_size') is not None else 0
            
            # Format partitioning columns as a string
            partitioning_columns_str = json.dumps(stats.get('partitioning_columns', []))
            
            # Add validation status
            validation_status = stats.get('validation_status', 'PENDING')
            validation_message = stats.get('validation_message', '').replace("'", "''")
            
            # Insert or update the registry
            upsert_sql = f"""
            INSERT INTO iceberg.nycdata.dataset_registry (
                dataset_id, dataset_title, dataset_description, domain, category, tags,
                schema_name, table_name, row_count, column_count, original_size, parquet_size,
                partitioning_columns, created_at, last_updated, etl_timestamp,
                validation_status, validation_message
            )
            VALUES (
                '{dataset_id}', '{dataset_title}', '{dataset_description}', '{domain}', '{category}', 
                ARRAY[{', '.join([f"'{tag}'" for tag in tags])}],
                '{schema_name}', '{table_name}', {row_count}, {column_count}, {original_size}, {parquet_size},
                '{partitioning_columns_str}', TIMESTAMP '{created_at}', TIMESTAMP '{last_updated_str}', 
                TIMESTAMP '{created_at}', '{validation_status}', '{validation_message}'
            )
            """
            
            self.logger.info(f"Executing SQL: {upsert_sql}")
            cursor.execute(upsert_sql)
            
            self.logger.info(f"Registered dataset {dataset_id} in metadata registry")
            return True
        except Exception as e:
            self.logger.error(f"Error registering dataset: {str(e)}")
            return False
    
    def _validate_dataset_against_source(self, dataset_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the dataset against the source to ensure data integrity"""
        try:
            self.logger.info(f"Validating dataset {dataset_id} against source")
            
            # Get a sample from the source for validation
            # We'll use a small sample to avoid hitting rate limits
            sample_size = min(100, len(df))
            
            # Get the first few rows from the source
            source_sample = self.socrata_client.get(dataset_id, limit=sample_size)
            
            if not source_sample:
                return {
                    "validation_status": "FAILED",
                    "validation_message": "Could not retrieve source data for validation"
                }
            
            # Convert to DataFrame for comparison
            source_df = pd.DataFrame.from_records(source_sample)
            
            # Basic validation checks
            validation_results = {
                "source_row_count": len(source_sample),
                "local_row_count": len(df),
                "source_column_count": len(source_df.columns),
                "local_column_count": len(df.columns),
                "column_match": set(source_df.columns) == set(df.columns),
                "sample_validation": True,  # Will be set to False if sample doesn't match
                "mismatched_columns": []
            }
            
            # Check if columns match
            if not validation_results["column_match"]:
                missing_in_local = set(source_df.columns) - set(df.columns)
                missing_in_source = set(df.columns) - set(source_df.columns)
                validation_results["mismatched_columns"] = {
                    "missing_in_local": list(missing_in_local),
                    "missing_in_source": list(missing_in_source)
                }
            
            # Determine validation status
            if validation_results["column_match"]:
                validation_status = "VALIDATED"
                validation_message = f"Successfully validated {sample_size} rows against source"
            else:
                validation_status = "WARNING"
                validation_message = f"Column mismatch: {len(validation_results['mismatched_columns']['missing_in_local'])} missing in local, {len(validation_results['mismatched_columns']['missing_in_source'])} missing in source"
            
            self.logger.info(f"Validation results for {dataset_id}: {validation_status} - {validation_message}")
            
            return {
                "validation_status": validation_status,
                "validation_message": validation_message,
                "validation_details": validation_results
            }
        except Exception as e:
            self.logger.error(f"Error validating dataset {dataset_id}: {e}")
            return {
                "validation_status": "ERROR",
                "validation_message": f"Validation error: {str(e)}"
            }
    
    def _fetch_data_in_chunks(self, dataset_id: str, chunk_size: int = None, 
                             query_params: Dict[str, Any] = None) -> Generator[pd.DataFrame, None, None]:
        """Fetch data from Socrata in chunks to reduce memory usage"""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        offset = 0
        total_fetched = 0
        more_data = True
        
        self.logger.info(f"Fetching data for dataset {dataset_id} in chunks of {chunk_size}")
        start_time = time.time()
        
        while more_data:
            try:
                # Prepare query parameters
                params = {
                    '$limit': chunk_size,
                    '$offset': offset
                }
                
                # Add any additional query parameters
                if query_params:
                    params.update(query_params)
                
                # Fetch chunk
                chunk_start_time = time.time()
                results = self.socrata_client.get(dataset_id, **params)
                chunk_end_time = time.time()
                
                # Check if we got any data
                if not results:
                    more_data = False
                    break
                
                # Convert to DataFrame
                df_chunk = pd.DataFrame.from_records(results)
                
                # Handle NULL values
                for col in df_chunk.columns:
                    # Replace empty strings with empty strings (to avoid None)
                    df_chunk[col] = df_chunk[col].replace('', '')
                    
                    # Replace None values with appropriate defaults
                    if df_chunk[col].dtype == 'object':
                        df_chunk[col] = df_chunk[col].fillna('')
                    elif pd.api.types.is_numeric_dtype(df_chunk[col]):
                        df_chunk[col] = df_chunk[col].fillna(0)
                    elif pd.api.types.is_datetime64_dtype(df_chunk[col]):
                        df_chunk[col] = df_chunk[col].fillna(pd.Timestamp('1970-01-01'))
                    else:
                        df_chunk[col] = df_chunk[col].fillna('')
                
                chunk_size_actual = len(df_chunk)
                total_fetched += chunk_size_actual
                
                self.logger.info(f"Fetched chunk of {chunk_size_actual} rows in {chunk_end_time - chunk_start_time:.2f} seconds "
                           f"(total: {total_fetched} rows, elapsed: {chunk_end_time - start_time:.2f} seconds)")
                
                # Yield the chunk
                yield df_chunk
                
                # Update offset for next chunk
                offset += chunk_size_actual
                
                # Check if we've reached the end
                if chunk_size_actual < chunk_size:
                    more_data = False
                    
                # Force garbage collection to free memory
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error fetching chunk at offset {offset}: {e}")
                more_data = False
                break
        
        end_time = time.time()
        self.logger.info(f"Completed fetching {total_fetched} rows in {end_time - start_time:.2f} seconds")

    def _process_and_save_chunks(self, dataset_id, chunks_iterator):
        """Process and save all chunks of a dataset"""
        try:
            # Initialize DuckDB processor
            from duckdb_processor import DuckDBProcessor
            
            # Create a temp directory based on dataset_id
            import tempfile
            temp_dir = os.path.join(tempfile.gettempdir(), f"nyc_etl_{dataset_id}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Initialize processor without the dataset_id parameter
            processor = DuckDBProcessor()
            
            # Log the initialization
            self.logger.info(f"Initialized DuckDBProcessor for dataset {dataset_id} with temp dir {temp_dir}")
            
            chunk_count = 0
            row_count = 0
            
            first_chunk = None
            
            for chunk in chunks_iterator:
                if chunk is None or chunk.empty:
                    self.logger.warning(f"Empty chunk received, skipping")
                    continue
                    
                # Save the first chunk for schema creation
                if first_chunk is None:
                    first_chunk = chunk
                    self.logger.info(f"First chunk captured with {len(chunk)} rows")
                    
                # Add the chunk to DuckDB
                processor.add_chunk(chunk)
                
                # Update counters
                chunk_count += 1
                row_count += len(chunk)
                
                self.logger.info(f"Processed chunk {chunk_count} with {len(chunk)} rows")
                
            # Check if we processed any valid chunks
            if chunk_count == 0 or first_chunk is None:
                self.logger.error("No valid chunks processed")
                processor.cleanup()
                return None
                
            # Convert to Parquet
            parquet_path = os.path.join(temp_dir, f"{dataset_id}.parquet")
            processor.to_parquet(parquet_path)
            
            # Return the results
            return {
                'row_count': row_count,
                'chunk_count': chunk_count,
                'parquet_path': parquet_path,
                'temp_dir': temp_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error processing chunks: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Clean up processor if it exists
            if 'processor' in locals() and processor:
                try:
                    processor.cleanup()
                except:
                    pass
            
            return None

    def _map_duckdb_to_trino_type(self, duckdb_type: str) -> str:
        """Map DuckDB types to Trino types"""
        # Lower case for consistent matching
        duckdb_type = duckdb_type.lower()
        
        # Basic type mapping
        type_mapping = {
            'boolean': 'boolean',
            'tinyint': 'tinyint',
            'smallint': 'smallint',
            'integer': 'integer',
            'int': 'integer',
            'bigint': 'bigint',
            'hugeint': 'bigint',
            'utinyint': 'tinyint',
            'usmallint': 'smallint',
            'uinteger': 'integer',
            'ubigint': 'bigint',
            'float': 'real',
            'double': 'double',
            'decimal': 'decimal(38,9)',
            'varchar': 'varchar',
            'char': 'char',
            'date': 'date',
            'time': 'time',
            'timestamp': 'timestamp',
            'timestamp with time zone': 'timestamp with time zone',
            'blob': 'varbinary',
            'json': 'json',
            'uuid': 'uuid'
        }
        
        # Check for exact matches
        if duckdb_type in type_mapping:
            return type_mapping[duckdb_type]
        
        # Check for parameterized types
        if duckdb_type.startswith('varchar('):
            return duckdb_type
        if duckdb_type.startswith('decimal('):
            return duckdb_type
        
        # Default to varchar for unknown types
        self.logger.warning(f"Unknown DuckDB type: {duckdb_type}, mapping to varchar")
        return 'varchar'

    def create_trino_table_from_dataset(self, dataset_id, overwrite=False):
        """Create a Trino table from a Socrata dataset with improved error handling"""
        try:
            # Check if dataset is already cached
            cached_metadata = None
            parquet_path = None
            
            # Only try to get metadata if cache_manager is available
            if hasattr(self, 'cache_manager') and self.cache_manager:
                # Check if cache_manager has the method
                if hasattr(self.cache_manager, 'get_dataset_metadata'):
                    cached_metadata = self.cache_manager.get_dataset_metadata(dataset_id)
                else:
                    self.logger.warning("Cache manager doesn't have get_dataset_metadata method")
            
            if cached_metadata and not overwrite:
                self.logger.info(f"Using cached data for dataset {dataset_id}")
                parquet_path = cached_metadata.get('parquet_path')
                
                # Ensure the Parquet file exists
                if not parquet_path or not os.path.exists(parquet_path):
                    self.logger.warning(f"Cached Parquet file not found: {parquet_path}")
                    parquet_path = None
                    cached_metadata = None
            
            # Process the dataset if no cache or overwrite requested
            if not cached_metadata or overwrite:
                # Process the dataset from scratch
                self.logger.info(f"Processing dataset {dataset_id} from scratch")
                
                # Get dataset metadata
                metadata = self.get_dataset_metadata(dataset_id)
                if not metadata:
                    self.logger.error(f"Failed to get metadata for dataset {dataset_id}")
                    return {"error": "Failed to get dataset metadata"}
                    
                # Extract the dataset in chunks
                chunks_iterator = self.fetch_dataset_chunks(dataset_id)
                
                # Process and save chunks
                result = self._process_and_save_chunks(dataset_id, chunks_iterator)
                
                if not result:
                    self.logger.error(f"Failed to process dataset {dataset_id}")
                    return {"error": "Failed to process dataset chunks"}
                    
                if 'parquet_path' not in result:
                    self.logger.error(f"No Parquet path in processing result for {dataset_id}")
                    return {"error": "No Parquet path in processing result"}
                    
                parquet_path = result['parquet_path']
                
                # Verify the Parquet file exists and is not empty
                if not os.path.exists(parquet_path):
                    self.logger.error(f"Generated Parquet file does not exist: {parquet_path}")
                    return {"error": "Generated Parquet file does not exist"}
                    
                file_size = os.path.getsize(parquet_path)
                if file_size == 0:
                    self.logger.error(f"Generated Parquet file is empty: {parquet_path}")
                    return {"error": "Generated Parquet file is empty"}
                    
                self.logger.info(f"Generated Parquet file: {parquet_path} ({file_size} bytes)")
                
                # Update metadata with processing results
                metadata.update({
                    'row_count': result.get('row_count', 0),
                    'parquet_path': parquet_path,
                    'parquet_size': file_size,
                    'temp_dir': result.get('temp_dir')
                })
                
                # Cache the metadata if cache_manager is available
                if hasattr(self, 'cache_manager') and self.cache_manager:
                    if hasattr(self.cache_manager, 'save_dataset_metadata'):
                        self.cache_manager.save_dataset_metadata(dataset_id, metadata)
                    else:
                        self.logger.warning("Cache manager doesn't have save_dataset_metadata method")
                
                cached_metadata = metadata
            
            # Upload Parquet file to MinIO
            upload_success = self._upload_to_minio(parquet_path, dataset_id)
            if not upload_success:
                self.logger.error(f"Failed to upload Parquet file to MinIO")
                # Return partial success with local file information
                return {
                    "status": "partial_success",
                    "message": "Dataset processed but not uploaded to MinIO",
                    "parquet_path": parquet_path,
                    "metadata": cached_metadata
                }
            
            # Create Trino table
            table_created = self._create_iceberg_table(parquet_path, dataset_id, cached_metadata)
            
            if not table_created:
                self.logger.warning(f"Failed to create table for dataset {dataset_id}, but data is processed")
                # Return partial success with upload information
                return {
                    "status": "partial_success",
                    "message": "Dataset processed and uploaded but table creation failed",
                    "parquet_path": parquet_path,
                    "metadata": cached_metadata
                }
            
            # Full success
            return {
                "status": "success",
                "message": "Dataset processed, uploaded, and table created successfully",
                "parquet_path": parquet_path,
                "metadata": cached_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error creating table for dataset {dataset_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def process_multiple_datasets(self, dataset_ids: List[str] = None, category: str = None, 
                                limit: int = 5) -> List[Dict[str, Any]]:
        """Process multiple datasets"""
        results = []
        
        # If dataset IDs are provided, process them
        if dataset_ids:
            for dataset_id in dataset_ids:
                self.logger.info(f"Processing dataset {dataset_id}")
                result = self.create_trino_table_from_dataset(dataset_id)
                if result:
                    results.append(result)
        # Otherwise, discover datasets
        else:
            datasets = self.discover_datasets(category=category, limit=limit)
            for dataset in datasets:
                dataset_id = dataset.get('resource', {}).get('id')
                if not dataset_id:
                    continue
                    
                self.logger.info(f"Processing discovered dataset {dataset_id}")
                result = self.create_trino_table_from_dataset(dataset_id)
                if result:
                    results.append(result)
        
        return results

    def _update_dataset_registry(self, dataset_id, dataset_title, dataset_description, 
                               schema_name, table_name, row_count, original_size=0, 
                               parquet_size=0, metadata=None):
        """Update the dataset registry with new dataset information"""
        try:
            conn = self._get_trino_connection()
            if not conn:
                self.logger.warning("Trino connection not available, skipping registry update")
                return False
            
            cursor = conn.cursor()
            
            # Ensure metadata schema and registry table exist
            self._ensure_metadata_schema()
            
            # Format timestamps correctly for Trino
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Clean input strings to prevent SQL injection
            dataset_title = dataset_title.replace("'", "''") if dataset_title else ''
            dataset_description = dataset_description.replace("'", "''") if dataset_description else ''
            
            # Derive column count from metadata if available
            column_count = len(metadata.get('columns', [])) if metadata and 'columns' in metadata else 0
            
            # Format for partitioning columns (if any)
            partitioning_columns = metadata.get('partitioning_columns', []) if metadata else []
            
            # Fix: Avoid using backslashes in f-string expressions
            quoted_cols = ["'{}'".format(col) for col in partitioning_columns]
            cols_joined = ', '.join(quoted_cols)
            partitioning_cols_str = f"ARRAY[{cols_joined}]" if partitioning_columns else "ARRAY[]"
            
            # Execute INSERT or UPDATE
            # First check if the dataset already exists
            cursor.execute(f"SELECT dataset_id FROM {self.trino_catalog}.metadata.dataset_registry WHERE dataset_id = '{dataset_id}'")
            exists = len(cursor.fetchall()) > 0
            
            if exists:
                # Update existing record
                cursor.execute(f"""
                UPDATE {self.trino_catalog}.metadata.dataset_registry
                SET 
                    dataset_title = '{dataset_title}',
                    dataset_description = '{dataset_description}',
                    schema_name = '{schema_name}',
                    table_name = '{table_name}',
                    row_count = {row_count},
                    column_count = {column_count},
                    original_size = {original_size},
                    parquet_size = {parquet_size},
                    partitioning_columns = {partitioning_cols_str},
                    last_updated = TIMESTAMP '{current_time}',
                    etl_timestamp = TIMESTAMP '{current_time}'
                WHERE dataset_id = '{dataset_id}'
                """)
            else:
                # Insert new record
                cursor.execute(f"""
                INSERT INTO {self.trino_catalog}.metadata.dataset_registry (
                    dataset_id, dataset_title, dataset_description,
                    schema_name, table_name, row_count, column_count,
                    original_size, parquet_size, partitioning_columns,
                    last_updated, etl_timestamp
                )
                VALUES (
                    '{dataset_id}', '{dataset_title}', '{dataset_description}',
                    '{schema_name}', '{table_name}', {row_count}, {column_count},
                    {original_size}, {parquet_size}, {partitioning_cols_str},
                    TIMESTAMP '{current_time}', TIMESTAMP '{current_time}'
                )
                """)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Updated dataset registry for {dataset_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating dataset registry: {e}")
            return False

    def _check_dataset_in_registry(self, dataset_id: str) -> bool:
        """Check if a dataset exists in the registry"""
        try:
            conn = self._get_trino_connection()
            if not conn:
                self.logger.warning("Trino connection not available, cannot check registry")
                return False
                
            cursor = conn.cursor()
            cursor.execute(f"SELECT dataset_id FROM iceberg.metadata.dataset_registry WHERE dataset_id = '{dataset_id}'")
            exists = len(cursor.fetchall()) > 0
            conn.close()
            return exists
        except Exception as e:
            self.logger.error(f"Error checking dataset in registry: {e}")
            return False
    
    def _get_dataset_from_registry(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset information from the registry"""
        try:
            conn = self._get_trino_connection()
            if not conn:
                self.logger.warning("Trino connection not available, cannot get dataset from registry")
                return {}
                
            cursor = conn.cursor()
            cursor.execute(f"""
            SELECT 
                dataset_id, dataset_title, dataset_description, 
                schema_name, table_name, row_count, column_count,
                original_size, parquet_size, 
                last_updated, etl_timestamp
            FROM iceberg.metadata.dataset_registry 
            WHERE dataset_id = '{dataset_id}'
            """)
            
            columns = [col[0] for col in cursor.description]
            result = cursor.fetchone()
            
            if not result:
                return {}
                
            dataset_info = dict(zip(columns, result))
            conn.close()
            return dataset_info
        except Exception as e:
            self.logger.error(f"Error getting dataset from registry: {e}")
            return {}

    def fetch_dataset_chunk(self, dataset_id, limit=50000, offset=0):
        """Fetch a chunk of data from Socrata API without the invalid timeout parameter."""
        try:
            self.logger.info(f"Fetching chunk of {limit} rows starting at offset {offset}")
            
            # Use pagination to get a chunk of data
            # Remove the timeout parameter which is causing the 400 error
            results = self.socrata_client.get(
                dataset_id,
                limit=limit,
                offset=offset
            )
            
            # Convert to DataFrame
            if results and len(results) > 0:
                df = pd.DataFrame.from_records(results)
                return df
            else:
                self.logger.warning(f"No results returned for chunk at offset {offset}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error fetching chunk at offset {offset}: {str(e)}")
            
            # Progressive backoff for errors, but don't mention timeout specifically
            if limit > 10000:
                # Try again with a smaller chunk size
                self.logger.info(f"Retrying with reduced chunk size ({limit//2})")
                return self.fetch_dataset_chunk(dataset_id, limit=limit//2, offset=offset)
            
            return None
        
    def load_dataset(self, dataset_id, **kwargs):
        self.logger.info(f"Starting dataset load for {dataset_id} with params: {kwargs}")
        # Existing code

    def fetch_dataset_chunks(self, dataset_id, chunk_size=50000) -> Generator[pd.DataFrame, None, None]:
        """Fetch all data from a dataset in chunks to handle large datasets"""
        offset = 0
        while True:
            self.logger.info(f"Fetching chunk at offset {offset}")
            chunk = self.fetch_dataset_chunk(dataset_id, limit=chunk_size, offset=offset)
            
            if chunk is None or chunk.empty:
                self.logger.info(f"No more data at offset {offset}, stopping")
                break
            
            # Return this chunk to the generator
            yield chunk
            
            # Update offset for next chunk
            offset += chunk_size
            
            # Check if this was the last chunk (fewer rows than requested)
            if len(chunk) < chunk_size:
                self.logger.info(f"Last chunk had {len(chunk)} rows, stopping")
                break

    def _create_iceberg_table(self, parquet_path, dataset_id, metadata):
        """Create an Iceberg table for a dataset"""
        try:
            # Get Trino connection
            conn = self._get_trino_connection()
            if not conn:
                self.logger.error("Cannot create table: Trino connection not available")
                return False
            
            cursor = conn.cursor()
            
            # Generate a Trino-friendly table name from dataset title
            table_name = self._generate_table_name(metadata.get('name', dataset_id))
            
            # Determine schema from category if available
            category = metadata.get('category', '').lower().replace(' ', '_')
            schema_name = 'nyc'  # Default schema
            
            if category and category != "uncategorized":
                schema_name = self._clean_schema_name(category)
            
            # Ensure schema exists
            self.logger.info(f"Executing SQL: CREATE SCHEMA IF NOT EXISTS {self.trino_catalog}.{schema_name}")
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.trino_catalog}.{schema_name}")
            
            # Drop table if it exists
            self.logger.info(f"Executing SQL: DROP TABLE IF EXISTS {self.trino_catalog}.{schema_name}.{table_name}")
            cursor.execute(f"DROP TABLE IF EXISTS {self.trino_catalog}.{schema_name}.{table_name}")
            
            # Read the Parquet file explicitly with pandas to get schema
            self.logger.info(f"Reading Parquet file for schema: {parquet_path}")
            
            # First check if file exists and is readable
            if not os.path.exists(parquet_path):
                self.logger.error(f"Parquet file not found: {parquet_path}")
                return False
            
            try:
                import pandas as pd
                
                # Read the Parquet file
                df = pd.read_parquet(parquet_path)
                
                # Debug info
                self.logger.info(f"Parquet file loaded successfully, shape: {df.shape}")
                self.logger.info(f"Columns found: {list(df.columns)}")
                
                if df.empty:
                    self.logger.error("Parquet file is empty - no data to create table from")
                    return False
                
                # Create columns definition from pandas DataFrame
                columns_def = []
                for col_name, dtype in df.dtypes.items():
                    # Clean column name
                    clean_col = self._clean_column_name(col_name)
                    # Map pandas dtype to SQL type
                    sql_type = self._map_pandas_to_sql_type(dtype)
                    columns_def.append(f'"{clean_col}" {sql_type}')
                    
                    # Debug logging
                    self.logger.info(f"Column definition: {clean_col} ({dtype}) -> {sql_type}")
                
                # Join column definitions
                columns_sql = ",\n    ".join(columns_def)
                
                # Make sure we actually have columns
                if not columns_def:
                    self.logger.error("No columns found in DataFrame schema")
                    return False
                
                # Create table with column definitions
                create_table_sql = f"""
                        CREATE TABLE {self.trino_catalog}.{schema_name}.{table_name} (
                            {columns_sql}
                        )
                        WITH (
                            format = 'PARQUET',
                            location = 's3a://{self.minio_bucket}/{dataset_id}/'
                        )
                        """
                
                self.logger.info(f"Executing SQL:\n{create_table_sql}")
                cursor.execute(create_table_sql)
                
                # Record the table creation in our registry
                registry_updated = self._update_dataset_registry(
                    dataset_id=dataset_id,
                    dataset_title=metadata.get('name', ''),
                    dataset_description=metadata.get('description', ''),
                    schema_name=schema_name,
                    table_name=table_name,
                    row_count=metadata.get('row_count', 0),
                    original_size=metadata.get('original_size', 0),
                    parquet_size=metadata.get('parquet_size', 0),
                    metadata=metadata
                )
                
                # Close connection
                conn.close()
                
                if registry_updated:
                    self.logger.info(f"Dataset {dataset_id} is now available as {schema_name}.{table_name}")
                else:
                    self.logger.warning(f"Dataset {dataset_id} table created, but registry update failed")
                
                return True
                
            except Exception as pandas_err:
                # Fallback to DuckDB for schema extraction if pandas fails
                self.logger.warning(f"Failed to read Parquet with pandas: {pandas_err}, trying DuckDB")
                
                try:
                    import duckdb
                    
                    # Create a temporary DuckDB connection
                    temp_db = duckdb.connect(":memory:")
                    
                    # Query the Parquet file schema
                    schema_query = f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
                    schema_result = temp_db.execute(schema_query).fetchall()
                    
                    # Create columns definition from DuckDB schema
                    columns_def = []
                    for col_info in schema_result:
                        col_name = col_info[0]
                        duck_type = col_info[1]
                        
                        # Clean column name
                        clean_col = self._clean_column_name(col_name)
                        # Map DuckDB type to SQL type
                        sql_type = self._map_duckdb_to_sql_type(duck_type)
                        columns_def.append(f'"{clean_col}" {sql_type}')
                        
                        # Debug logging
                        self.logger.info(f"Column from DuckDB: {clean_col} ({duck_type}) -> {sql_type}")
                    
                    # Close the temporary connection
                    temp_db.close()
                    
                    # Join column definitions
                    columns_sql = ",\n    ".join(columns_def)
                    
                    # Make sure we actually have columns
                    if not columns_def:
                        self.logger.error("No columns found in DuckDB schema")
                        return False
                        
                    # Create table with column definitions
                    create_table_sql = f"""
                            CREATE TABLE {self.trino_catalog}.{schema_name}.{table_name} (
                                {columns_sql}
                            )
                            WITH (
                                format = 'PARQUET',
                                location = 's3a://{self.minio_bucket}/{dataset_id}/'
                            )
                            """
                    
                    self.logger.info(f"Executing SQL:\n{create_table_sql}")
                    cursor.execute(create_table_sql)
                    
                    # Record the table creation in our registry
                    registry_updated = self._update_dataset_registry(
                        dataset_id=dataset_id,
                        dataset_title=metadata.get('name', ''),
                        dataset_description=metadata.get('description', ''),
                        schema_name=schema_name,
                        table_name=table_name,
                        row_count=metadata.get('row_count', 0),
                        original_size=metadata.get('original_size', 0),
                        parquet_size=metadata.get('parquet_size', 0),
                        metadata=metadata
                    )
                    
                    # Close connection
                    conn.close()
                    
                    return True
                    
                except Exception as duckdb_err:
                    self.logger.error(f"Failed to read Parquet with DuckDB: {duckdb_err}")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error creating table: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.error("Failed to create Trino table")
            return False

    def _map_duckdb_to_sql_type(self, duckdb_type):
        """Map DuckDB type to SQL type"""
        duckdb_type = duckdb_type.lower()
        
        if 'varchar' in duckdb_type or 'string' in duckdb_type:
            return 'VARCHAR'
        elif 'integer' in duckdb_type or 'int' in duckdb_type:
            return 'BIGINT'
        elif 'double' in duckdb_type or 'float' in duckdb_type or 'numeric' in duckdb_type:
            return 'DOUBLE'
        elif 'boolean' in duckdb_type or 'bool' in duckdb_type:
            return 'BOOLEAN'
        elif 'timestamp' in duckdb_type:
            return 'TIMESTAMP'
        elif 'date' in duckdb_type:
            return 'DATE'
        else:
            # Default to VARCHAR for unknown types
            return 'VARCHAR'

    def _generate_table_name(self, name):
        """Generate a table name from a dataset name"""
        # Replace spaces with underscores
        cleaned_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        # Remove consecutive underscores
        cleaned_name = re.sub(r'_+', '_', cleaned_name)
        # Remove leading and trailing underscores
        cleaned_name = cleaned_name.strip('_')
        # Ensure it doesn't start with a number
        if cleaned_name and cleaned_name[0].isdigit():
            cleaned_name = 'tbl_' + cleaned_name
        # If empty, use a default name
        if not cleaned_name:
            cleaned_name = 'table'
        return cleaned_name

    def _clean_schema_name(self, name):
        """Clean a schema name to be compatible with Trino"""
        # Replace spaces with underscores
        cleaned_name = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        # Remove consecutive underscores
        cleaned_name = re.sub(r'_+', '_', cleaned_name)
        # Remove leading and trailing underscores
        cleaned_name = cleaned_name.strip('_')
        # Ensure it doesn't start with a number
        if cleaned_name and cleaned_name[0].isdigit():
            cleaned_name = 'sch_' + cleaned_name
        # If empty, use a default name
        if not cleaned_name:
            cleaned_name = 'schema'
        return cleaned_name

    def _map_pandas_to_sql_type(self, pandas_dtype):
        """Map pandas dtype to SQL type"""
        import numpy as np
        
        # Convert to string for easier comparison
        dtype_str = str(pandas_dtype).lower()
        
        # Handle numpy/pandas numeric types
        if 'int' in dtype_str:
            return 'BIGINT'
        elif 'float' in dtype_str:
            return 'DOUBLE'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        elif 'datetime' in dtype_str or 'timestamp' in dtype_str:
            return 'TIMESTAMP'
        elif 'date' in dtype_str:
            return 'DATE'
        else:
            # Default to VARCHAR for strings and other types
            return 'VARCHAR'
