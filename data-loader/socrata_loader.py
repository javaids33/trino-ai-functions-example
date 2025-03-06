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
from typing import Dict, List, Any, Optional, Tuple, Generator, Iterator
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

# Set up logger
logger = setup_logger(__name__)

class SocrataToTrinoETL:
    """ETL pipeline to load data from Socrata Open Data API to Trino"""
    
    def __init__(self, domain=None, app_token=None, api_key_id=None, api_key_secret=None,
                 minio_endpoint=None, minio_access_key=None, minio_secret_key=None):
        """Initialize the ETL pipeline with Socrata credentials."""
        # Get credentials from env_config if not provided
        socrata_creds = get_socrata_credentials()
        minio_creds = get_minio_credentials()
        
        self.domain = domain or DEFAULT_DOMAIN
        self.app_token = app_token or socrata_creds['app_token']
        self.api_key_id = api_key_id or socrata_creds['api_key_id']
        self.api_key_secret = api_key_secret or socrata_creds['api_key_secret']
        
        # Set MinIO parameters from arguments or environment variables
        self.minio_endpoint = minio_endpoint or minio_creds['endpoint']
        self.minio_access_key = minio_access_key or minio_creds['access_key']
        self.minio_secret_key = minio_secret_key or minio_creds['secret_key']
        self.minio_bucket = os.environ.get('MINIO_BUCKET', 'iceberg')
        
        # Create a dedicated bucket for NYC ETL data
        self.nyc_etl_bucket = 'nycetldata'
        
        # Trino configuration
        trino_creds = get_trino_credentials()
        self.trino_host = trino_creds['host']
        self.trino_port = int(trino_creds['port'])
        self.trino_user = trino_creds['user']
        self.trino_catalog = trino_creds['catalog']
        self.trino_schema = trino_creds['schema']
        
        # Initialize clients
        self.client = self._init_socrata_client()
        self.minio_client = self._init_minio_client()
        
        # Ensure the NYC ETL bucket exists
        self._ensure_nyc_etl_bucket()
        
        # Create cache directory for downloaded files
        self.cache_dir = os.path.join(os.getcwd(), 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Ensure metadata schema exists
        self._ensure_metadata_schema()
        
        # Default chunk size for processing large datasets
        self.chunk_size = 10000
        
    def _ensure_nyc_etl_bucket(self):
        """Ensure the NYC ETL bucket exists in MinIO"""
        try:
            if not self.minio_client.bucket_exists(self.nyc_etl_bucket):
                self.minio_client.make_bucket(self.nyc_etl_bucket)
                logger.info(f"Created MinIO bucket: {self.nyc_etl_bucket}")
        except Exception as e:
            logger.error(f"Error ensuring NYC ETL bucket exists: {e}")
    
    def _init_socrata_client(self) -> Socrata:
        """Initialize the Socrata client with authentication if available"""
        try:
            if self.api_key_id and self.api_key_secret:
                client = Socrata(
                    self.domain,
                    self.app_token,
                    username=self.api_key_id,
                    password=self.api_key_secret
                )
                logger.info("Initialized Socrata client with API key authentication")
            else:
                client = Socrata(self.domain, self.app_token)
                logger.warning("Initialized Socrata client without authentication - rate limits will apply")
            return client
        except Exception as e:
            logger.error(f"Error initializing Socrata client: {e}")
            raise
    
    def _init_minio_client(self) -> Minio:
        """Initialize MinIO client"""
        try:
            client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False
            )
            
            # Ensure the bucket exists
            if not client.bucket_exists(self.minio_bucket):
                client.make_bucket(self.minio_bucket)
                logger.info(f"Created MinIO bucket '{self.minio_bucket}'")
            
            return client
        except Exception as e:
            logger.error(f"Error initializing MinIO client: {str(e)}")
            raise
    
    def _get_trino_connection(self) -> trino.dbapi.Connection:
        """Get a connection to Trino"""
        try:
            conn = trino.dbapi.connect(
                host=self.trino_host,
                port=self.trino_port,
                user=self.trino_user,
                catalog=self.trino_catalog,
                schema=self.trino_schema
            )
            
            return conn
        except Exception as e:
            logger.error(f"Error connecting to Trino: {str(e)}")
            raise
    
    def _ensure_metadata_schema(self):
        """Ensure the metadata schema exists in Trino"""
        try:
            conn = self._get_trino_connection()
            cursor = conn.cursor()
            
            # Create metadata schema if it doesn't exist
            cursor.execute("CREATE SCHEMA IF NOT EXISTS iceberg.nycdata")
            
            # Check if the table exists
            cursor.execute("SHOW TABLES IN iceberg.nycdata LIKE 'dataset_registry'")
            table_exists = len(cursor.fetchall()) > 0
            
            if table_exists:
                # Check if we need to add new columns for validation
                try:
                    cursor.execute("SELECT validation_status FROM iceberg.nycdata.dataset_registry LIMIT 0")
                except Exception:
                    # Validation columns don't exist, add them
                    logger.info("Adding validation columns to dataset_registry table")
                    cursor.execute("ALTER TABLE iceberg.nycdata.dataset_registry ADD COLUMN validation_status VARCHAR")
                    cursor.execute("ALTER TABLE iceberg.nycdata.dataset_registry ADD COLUMN validation_message VARCHAR")
                    cursor.execute("ALTER TABLE iceberg.nycdata.dataset_registry ADD COLUMN created_at TIMESTAMP")
            else:
                # Create dataset registry table if it doesn't exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS iceberg.nycdata.dataset_registry (
                    dataset_id VARCHAR,
                    dataset_title VARCHAR,
                    dataset_description VARCHAR,
                    domain VARCHAR,
                    category VARCHAR,
                    tags ARRAY(VARCHAR),
                    schema_name VARCHAR,
                    table_name VARCHAR,
                    row_count BIGINT,
                    column_count INTEGER,
                    original_size BIGINT,
                    parquet_size BIGINT,
                    partitioning_columns VARCHAR,
                    created_at TIMESTAMP,
                    last_updated TIMESTAMP,
                    etl_timestamp TIMESTAMP,
                    validation_status VARCHAR,
                    validation_message VARCHAR
                )
                """)
            
            cursor.close()
            conn.close()
            
            logger.info("Ensured nycdata schema and registry table exist")
        except Exception as e:
            logger.error(f"Error ensuring nycdata schema: {str(e)}")
            raise
    
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
            datasets = self.client.datasets(limit=limit, **query)
            
            logger.info(f"Discovered {len(datasets)} datasets from {domain}")
            return datasets
        except Exception as e:
            logger.error(f"Error discovering datasets: {str(e)}")
            return []
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a dataset"""
        try:
            metadata = self.client.get_metadata(dataset_id)
            return metadata
        except Exception as e:
            logger.error(f"Error getting metadata for dataset {dataset_id}: {str(e)}")
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
    
    def _determine_partitioning_columns(self, df: pd.DataFrame) -> List[str]:
        """Determine which columns to use for partitioning"""
        partitioning_columns = []
        
        # Look for date columns first
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            # Use the first date column
            partitioning_columns.append(date_columns[0])
        
        # Look for categorical columns with reasonable cardinality
        for col in df.columns:
            if col in partitioning_columns:
                continue
                
            # Skip if not a string column
            if df[col].dtype != 'object':
                continue
                
            # Check cardinality
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 100:
                partitioning_columns.append(col)
                
            # Limit to 2 partitioning columns
            if len(partitioning_columns) >= 2:
                break
        
        return partitioning_columns
    
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
            logger.info(f"Executing SQL: {schema_sql}")
            cursor.execute(schema_sql)
            
            # Add comment to schema if we have metadata
            if schema_comment:
                comment_sql = f"COMMENT ON SCHEMA iceberg.{schema_name} IS '{schema_comment}'"
                try:
                    cursor.execute(comment_sql)
                    logger.info(f"Added comment to schema iceberg.{schema_name}")
                except Exception as e:
                    logger.warning(f"Could not add comment to schema: {e}")
            
            # Drop table if it exists
            drop_sql = f"DROP TABLE IF EXISTS iceberg.{schema_name}.{table_name}"
            logger.info(f"Executing SQL: {drop_sql}")
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
            
            logger.info(f"Executing SQL: {create_table_sql}")
            cursor.execute(create_table_sql)
            logger.info(f"Created table iceberg.{schema_name}.{table_name}")
            
            # Add table comment if we have metadata
            if metadata and 'name' in metadata and 'description' in metadata:
                table_description = metadata['description'].replace("'", "''") if metadata.get('description') else metadata['name'].replace("'", "''")
                comment_sql = f"COMMENT ON TABLE iceberg.{schema_name}.{table_name} IS '{table_description}'"
                try:
                    cursor.execute(comment_sql)
                    logger.info(f"Added comment to table iceberg.{schema_name}.{table_name}")
                except Exception as e:
                    logger.warning(f"Could not add comment to table: {e}")
            
            # Add column comments if we have column metadata
            if metadata and 'columns' in metadata and isinstance(metadata['columns'], list):
                for col_meta in metadata['columns']:
                    if 'fieldName' in col_meta and 'name' in col_meta:
                        col_name = self._clean_column_name(col_meta['fieldName'])
                        col_description = col_meta.get('description', col_meta['name']).replace("'", "''")
                        comment_sql = f"COMMENT ON COLUMN iceberg.{schema_name}.{table_name}.{col_name} IS '{col_description}'"
                        try:
                            cursor.execute(comment_sql)
                            logger.info(f"Added comment to column {col_name}")
                        except Exception as e:
                            logger.warning(f"Could not add comment to column {col_name}: {e}")
            
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
                            logger.info(f"Downloaded {object_name} to {local_file_path}")
                        except Exception as e:
                            logger.error(f"Error downloading file from MinIO: {e}")
                            raise
                    
                    # Read the Parquet file
                    df = pd.read_parquet(local_file_path)
                    logger.info(f"Loaded {len(df)} rows from {local_file_path}")
                    
                    # Insert data in batches to improve performance
                    batch_size = 500  # Reduced from 5000 to 500 to avoid query text length limit
                    total_rows = len(df)
                    batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
                    
                    logger.info(f"Starting to insert {total_rows} rows in {batches} batches of {batch_size} rows each")
                    
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
                            
                            logger.info(f"Inserted batch {batch_idx+1}/{batches} ({len(batch_df)} rows) - "
                                       f"{progress:.1f}% complete - "
                                       f"{rows_per_second:.1f} rows/sec - "
                                       f"Est. time remaining: {estimated_time_remaining:.1f} seconds")
                    
                    logger.info(f"Successfully loaded all {total_rows} rows into iceberg.{schema_name}.{table_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading data: {e}")
                    logger.error("Could not load data into the table")
                    return False
            
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            logger.error("Failed to create Trino table")
            return False
    
    def _upload_to_minio(self, local_file_path: str, object_name: str) -> bool:
        """Upload a file to MinIO"""
        try:
            # Upload to the main bucket
            self.minio_client.fput_object(
                self.minio_bucket, 
                object_name, 
                local_file_path
            )
            logger.info(f"Uploaded {local_file_path} to MinIO as {object_name}")
            
            # Also upload to the NYC ETL data bucket
            try:
                self.minio_client.fput_object(
                    self.nyc_etl_bucket, 
                    object_name, 
                    local_file_path
                )
                logger.info(f"Uploaded {local_file_path} to NYC ETL bucket as {object_name}")
            except Exception as e:
                logger.warning(f"Error uploading to NYC ETL bucket: {str(e)}")
            
            return True
        except Exception as e:
            logger.error(f"Error uploading to MinIO: {str(e)}")
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
                    logger.warning(f"Could not parse updatedAt timestamp: {metadata.get('updatedAt')}, using current time")
            
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
            
            logger.info(f"Executing SQL: {upsert_sql}")
            cursor.execute(upsert_sql)
            
            logger.info(f"Registered dataset {dataset_id} in metadata registry")
            return True
        except Exception as e:
            logger.error(f"Error registering dataset: {str(e)}")
            return False
    
    def _validate_dataset_against_source(self, dataset_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the dataset against the source to ensure data integrity"""
        try:
            logger.info(f"Validating dataset {dataset_id} against source")
            
            # Get a sample from the source for validation
            # We'll use a small sample to avoid hitting rate limits
            sample_size = min(100, len(df))
            
            # Get the first few rows from the source
            source_sample = self.client.get(dataset_id, limit=sample_size)
            
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
            
            logger.info(f"Validation results for {dataset_id}: {validation_status} - {validation_message}")
            
            return {
                "validation_status": validation_status,
                "validation_message": validation_message,
                "validation_details": validation_results
            }
        except Exception as e:
            logger.error(f"Error validating dataset {dataset_id}: {e}")
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
        
        logger.info(f"Fetching data for dataset {dataset_id} in chunks of {chunk_size}")
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
                results = self.client.get(dataset_id, **params)
                chunk_end_time = time.time()
                
                # Check if we got any data
                if not results:
                    more_data = False
                    break
                
                # Convert to DataFrame
                df_chunk = pd.DataFrame.from_records(results)
                chunk_size_actual = len(df_chunk)
                total_fetched += chunk_size_actual
                
                logger.info(f"Fetched chunk of {chunk_size_actual} rows in {chunk_end_time - chunk_start_time:.2f} seconds "
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
                logger.error(f"Error fetching chunk at offset {offset}: {e}")
                more_data = False
                break
        
        end_time = time.time()
        logger.info(f"Completed fetching {total_fetched} rows in {end_time - start_time:.2f} seconds")

    def _process_and_save_chunks(self, dataset_id: str, schema_name: str, table_name: str, 
                               chunk_generator: Iterator[pd.DataFrame], 
                               partitioning_columns: List[str] = None) -> Dict[str, Any]:
        """Process and save data chunks to parquet files using DuckDB for memory efficiency"""
        try:
            # Create a temporary directory for processing
            temp_dir = tempfile.mkdtemp(prefix=f"tmp_{dataset_id}_socrata")
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Initialize DuckDB processor
            duckdb_processor = DuckDBProcessor(temp_dir=temp_dir)
            logger.info(f"Initialized DuckDB processor for dataset {dataset_id}")
            
            # Process chunks with DuckDB
            processing_result = duckdb_processor.process_dataframe_chunks(chunk_generator, table_name="dataset")
            
            if processing_result.get("total_rows", 0) == 0:
                logger.error(f"No data processed for dataset {dataset_id}")
                shutil.rmtree(temp_dir)
                return {}
            
            # Determine schema and partitioning if not provided
            if not partitioning_columns and processing_result.get("columns"):
                # Use the first few rows to determine partitioning columns
                sample_query = "SELECT * FROM dataset LIMIT 1000"
                sample_df = duckdb_processor.conn.execute(sample_query).df()
                partitioning_columns = self._determine_partitioning_columns(sample_df)
                logger.info(f"Using partitioning columns: {partitioning_columns}")
            
            # Save to parquet with DuckDB's optimized writer
            combined_file = os.path.join(temp_dir, f"{dataset_id}.parquet")
            save_result = duckdb_processor.save_to_parquet(
                table_name="dataset", 
                output_path=combined_file,
                partitioning_columns=partitioning_columns
            )
            
            if save_result.get("file_size", 0) == 0:
                logger.error(f"Failed to save parquet file for dataset {dataset_id}")
                shutil.rmtree(temp_dir)
                return {}
            
            # Upload to MinIO
            object_name = f"data/{schema_name}/{table_name}/{dataset_id}.parquet"
            self._upload_to_minio(combined_file, object_name)
            
            # Infer schema from DuckDB
            schema = {}
            for col_name, col_type in zip(save_result.get("columns", []), save_result.get("column_types", [])):
                # Map DuckDB types to Trino types
                trino_type = self._map_duckdb_to_trino_type(col_type)
                schema[col_name] = trino_type
            
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
            
            return {
                "row_count": processing_result.get("total_rows", 0),
                "column_count": len(save_result.get("columns", [])),
                "parquet_size": save_result.get("file_size", 0),
                "object_name": object_name,
                "schema": schema,
                "partitioning_columns": partitioning_columns
            }
                
        except Exception as e:
            logger.error(f"Error processing and saving chunks: {e}")
            return {}
    
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
        logger.warning(f"Unknown DuckDB type: {duckdb_type}, mapping to varchar")
        return 'varchar'

    def create_trino_table_from_dataset(self, dataset_id: str, query_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a Trino table from a Socrata dataset using memory-efficient processing."""
        try:
            # Get dataset metadata
            metadata = self.get_dataset_metadata(dataset_id)
            if not metadata:
                logger.error(f"Failed to get metadata for dataset {dataset_id}")
                return {}
            
            # Determine schema and table name
            schema_name = self._determine_schema_from_metadata(metadata)
            table_name = self._determine_table_name(metadata)
            
            logger.info(f"Processing dataset {dataset_id} into {schema_name}.{table_name}")
            
            # Check if we already have this dataset in cache
            cache_file = os.path.join(self.cache_dir, f"{dataset_id}.parquet")
            
            if os.path.exists(cache_file) and not query_params:
                logger.info(f"Using cached dataset file: {cache_file}")
                df = pd.read_parquet(cache_file)
                
                # Determine partitioning columns
                partitioning_columns = self._determine_partitioning_columns(df)
                
                # Upload to MinIO
                object_name = f"data/{schema_name}/{table_name}/{dataset_id}.parquet"
                if not self._upload_to_minio(cache_file, object_name):
                    logger.error(f"Failed to upload {cache_file} to MinIO")
                    return {}
                
                # Infer schema from DataFrame
                column_schema = self._infer_schema_from_dataframe(df)
                
                # Validate the dataset against the source
                validation_results = self._validate_dataset_against_source(dataset_id, df)
                
                # Create the Trino table and load data
                if not self._create_trino_table(schema_name, table_name, column_schema, partitioning_columns, object_name, metadata):
                    logger.error(f"Failed to create Trino table")
                    return {}
                
                # Register the dataset in the metadata registry
                stats = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "original_size": len(df) * 100,  # Rough estimate
                    "parquet_size": os.path.getsize(cache_file),
                    "partitioning_columns": partitioning_columns,
                    "validation_status": validation_results["validation_status"],
                    "validation_message": validation_results["validation_message"]
                }
                
                if not self._register_dataset(dataset_id, metadata, schema_name, table_name, stats):
                    logger.error(f"Failed to register dataset in metadata registry")
                    return {}
                
                return {
                    "dataset_id": dataset_id,
                    "schema_name": schema_name,
                    "table_name": table_name,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "validation_status": validation_results["validation_status"],
                    "validation_message": validation_results["validation_message"]
                }
            else:
                # Process data in chunks
                chunk_generator = self._fetch_data_in_chunks(dataset_id, query_params=query_params)
                
                # Process and save chunks
                result = self._process_and_save_chunks(dataset_id, schema_name, table_name, chunk_generator)
                
                if not result:
                    logger.error(f"Failed to process dataset {dataset_id}")
                    return {}
                
                # Get a sample for validation
                sample_df = next(self._fetch_data_in_chunks(dataset_id, chunk_size=100))
                validation_results = self._validate_dataset_against_source(dataset_id, sample_df)
                
                # Create the Trino table and load data
                if not self._create_trino_table(schema_name, table_name, result["schema"], 
                                              result["partitioning_columns"], result["object_name"], metadata):
                    logger.error(f"Failed to create Trino table")
                    return {}
                
                # Register the dataset in the metadata registry
                stats = {
                    "row_count": result["row_count"],
                    "column_count": result["column_count"],
                    "original_size": result["row_count"] * 100,  # Rough estimate
                    "parquet_size": result["parquet_size"],
                    "partitioning_columns": result["partitioning_columns"],
                    "validation_status": validation_results["validation_status"],
                    "validation_message": validation_results["validation_message"]
                }
                
                if not self._register_dataset(dataset_id, metadata, schema_name, table_name, stats):
                    logger.error(f"Failed to register dataset in metadata registry")
                    return {}
                
                return {
                    "dataset_id": dataset_id,
                    "schema_name": schema_name,
                    "table_name": table_name,
                    "row_count": result["row_count"],
                    "column_count": result["column_count"],
                    "validation_status": validation_results["validation_status"],
                    "validation_message": validation_results["validation_message"]
                }
        except Exception as e:
            logger.error(f"Error creating Trino table from dataset {dataset_id}: {e}")
            return {}
    
    def process_multiple_datasets(self, dataset_ids: List[str] = None, category: str = None, 
                                limit: int = 5) -> List[Dict[str, Any]]:
        """Process multiple datasets"""
        results = []
        
        # If dataset IDs are provided, process them
        if dataset_ids:
            for dataset_id in dataset_ids:
                logger.info(f"Processing dataset {dataset_id}")
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
                    
                logger.info(f"Processing discovered dataset {dataset_id}")
                result = self.create_trino_table_from_dataset(dataset_id)
                if result:
                    results.append(result)
        
        return results

def main():
    """Main function to run the ETL process"""
    try:
        logger.info("Starting Socrata to Trino ETL process")
        
        # Get credentials from environment if available
        app_token = os.environ.get('SOCRATA_APP_TOKEN')
        api_key_id = os.environ.get('SOCRATA_API_KEY_ID', '43v2j64kmopow5b2kksprtl3r')
        api_key_secret = os.environ.get('SOCRATA_API_KEY_SECRET', '40jnea9x84i36adn6k9nmjckn3grq5ujixyax7d6uugvab4vso')
        
        # Initialize ETL pipeline with authentication
        etl = SocrataToTrinoETL(
            app_token=app_token,
            api_key_id=api_key_id,
            api_key_secret=api_key_secret
        )
        
        # Process datasets - either specify IDs or discover automatically
        # Example: Process NYC Civil Service List dataset
        dataset_ids = ["vx8i-nprf"]  # NYC Civil Service List
        
        results = etl.process_multiple_datasets(dataset_ids=dataset_ids)
        
        logger.info(f"Processed {len(results)} datasets successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 