import os
import logging
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sodapy import Socrata
from minio import Minio
from minio.error import S3Error
import trino
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/socrata_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SocrataToTrinoETL:
    """ETL pipeline to load data from Socrata Open Data API to Trino"""
    
    def __init__(self, domain="data.cityofnewyork.us", app_token=None, api_key_id=None, api_key_secret=None):
        """Initialize the ETL pipeline"""
        self.domain = domain
        self.app_token = app_token
        self.api_key_id = api_key_id
        self.api_key_secret = api_key_secret
        
        # Initialize Socrata client with authentication
        if api_key_id and api_key_secret:
            self.socrata_client = Socrata(domain, app_token, username=api_key_id, password=api_key_secret)
            logger.info(f"Initialized Socrata client with API key authentication")
        else:
            self.socrata_client = Socrata(domain, app_token)
            logger.warning("Using Socrata client without authentication - rate limits will apply")
        
        self.minio_client = self._init_minio_client()
        self._ensure_metadata_schema()
        
    def _init_minio_client(self) -> Minio:
        """Initialize MinIO client"""
        try:
            endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
            access_key = os.environ.get("MINIO_ACCESS_KEY", "admin")
            secret_key = os.environ.get("MINIO_SECRET_KEY", "password")
            
            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False
            )
            
            # Ensure the bucket exists
            if not client.bucket_exists("iceberg"):
                client.make_bucket("iceberg")
                logger.info("Created MinIO bucket 'iceberg'")
            
            return client
        except Exception as e:
            logger.error(f"Error initializing MinIO client: {str(e)}")
            raise
    
    def _get_trino_connection(self) -> trino.dbapi.Connection:
        """Get a connection to Trino"""
        try:
            host = os.environ.get("TRINO_HOST", "trino")
            port = int(os.environ.get("TRINO_PORT", "8080"))
            user = os.environ.get("TRINO_USER", "admin")
            catalog = os.environ.get("TRINO_CATALOG", "iceberg")
            schema = os.environ.get("TRINO_SCHEMA", "iceberg")
            
            conn = trino.dbapi.connect(
                host=host,
                port=port,
                user=user,
                catalog=catalog,
                schema=schema
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
                original_size_bytes BIGINT,
                parquet_size_bytes BIGINT,
                compression_ratio DOUBLE,
                partitioning_columns ARRAY(VARCHAR),
                last_updated TIMESTAMP,
                etl_timestamp TIMESTAMP
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
            datasets = self.socrata_client.datasets(limit=limit, **query)
            
            logger.info(f"Discovered {len(datasets)} datasets from {domain}")
            return datasets
        except Exception as e:
            logger.error(f"Error discovering datasets: {str(e)}")
            return []
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a dataset"""
        try:
            metadata = self.socrata_client.get_metadata(dataset_id)
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
                           partitioning_columns: List[str] = None) -> bool:
        """Create a table in Trino"""
        try:
            conn = self._get_trino_connection()
            cursor = conn.cursor()
            
            # Create schema if it doesn't exist
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS iceberg.{schema_name}")
            
            # Drop table if it exists
            cursor.execute(f"DROP TABLE IF EXISTS iceberg.{schema_name}.{table_name}")
            
            # Build the CREATE TABLE statement
            columns_sql = ", ".join([f"{col} {dtype}" for col, dtype in column_schema.items()])
            
            # Add partitioning if specified
            partitioning_sql = ""
            if partitioning_columns and len(partitioning_columns) > 0:
                clean_partitioning_columns = [self._clean_column_name(col) for col in partitioning_columns]
                partitioning_sql = f" WITH (partitioning = ARRAY{clean_partitioning_columns})"
            
            create_table_sql = f"""
            CREATE TABLE iceberg.{schema_name}.{table_name} (
                {columns_sql}
            ){partitioning_sql}
            """
            
            cursor.execute(create_table_sql)
            
            cursor.close()
            conn.close()
            
            logger.info(f"Created table iceberg.{schema_name}.{table_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            return False
    
    def _upload_to_minio(self, local_file_path: str, object_name: str) -> bool:
        """Upload a file to MinIO"""
        try:
            self.minio_client.fput_object(
                "iceberg", 
                object_name, 
                local_file_path
            )
            
            logger.info(f"Uploaded {local_file_path} to MinIO as {object_name}")
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
            dataset_title = metadata.get('name', '')
            dataset_description = metadata.get('description', '')
            domain = self.domain
            
            # Get category
            category = ''
            if 'classification' in metadata and 'domain_category' in metadata['classification']:
                category = metadata['classification']['domain_category']
            elif 'category' in metadata:
                category = metadata['category']
            
            # Get tags
            tags = []
            if 'tags' in metadata:
                tags = metadata['tags']
            elif 'classification' in metadata and 'domain_tags' in metadata['classification']:
                tags = metadata['classification']['domain_tags']
            
            # Get last updated
            last_updated = datetime.now().isoformat()
            if 'updatedAt' in metadata:
                last_updated = metadata['updatedAt']
            
            # Insert into registry
            cursor.execute("""
            INSERT INTO iceberg.nycdata.dataset_registry (
                dataset_id, dataset_title, dataset_description, domain, category, tags,
                schema_name, table_name, row_count, column_count, original_size_bytes,
                parquet_size_bytes, compression_ratio, partitioning_columns, last_updated, etl_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id, dataset_title, dataset_description, domain, category, tags,
                schema_name, table_name, stats['row_count'], stats['column_count'], 
                stats['original_size_bytes'], stats['parquet_size_bytes'], stats['compression_ratio'],
                stats['partitioning_columns'], last_updated, datetime.now().isoformat()
            ))
            
            cursor.close()
            conn.close()
            
            logger.info(f"Registered dataset {dataset_id} in nycdata registry")
            return True
        except Exception as e:
            logger.error(f"Error registering dataset: {str(e)}")
            return False
    
    def create_trino_table_from_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Create a Trino table from a Socrata dataset"""
        try:
            # Get metadata
            metadata = self.get_dataset_metadata(dataset_id)
            if not metadata:
                logger.error(f"Failed to get metadata for dataset {dataset_id}")
                return {}
            
            # Determine schema and table names
            schema_name = self._determine_schema_from_metadata(metadata)
            table_name = self._determine_table_name(metadata)
            
            logger.info(f"Processing dataset {dataset_id} into {schema_name}.{table_name}")
            
            # Get the data
            logger.info(f"Fetching data for dataset {dataset_id}")
            start_time = time.time()
            data = list(self.socrata_client.get_all(dataset_id))
            fetch_time = time.time() - start_time
            logger.info(f"Fetched {len(data)} rows in {fetch_time:.2f} seconds")
            
            if not data:
                logger.error(f"No data found for dataset {dataset_id}")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(data)
            
            # Clean column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Determine partitioning columns
            partitioning_columns = self._determine_partitioning_columns(df)
            logger.info(f"Using partitioning columns: {partitioning_columns}")
            
            # Infer schema
            column_schema = self._infer_schema_from_dataframe(df)
            
            # Create a temporary directory for the Parquet file
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create the Parquet file
                parquet_file = os.path.join(temp_dir, f"{dataset_id}.parquet")
                
                # Convert to PyArrow Table and write to Parquet
                table = pa.Table.from_pandas(df)
                pq.write_table(table, parquet_file)
                
                # Get file sizes
                original_size = len(df) * len(df.columns) * 8  # Rough estimate
                parquet_size = os.path.getsize(parquet_file)
                compression_ratio = original_size / parquet_size if parquet_size > 0 else 0
                
                # Upload to MinIO
                object_name = f"data/{schema_name}/{table_name}/{dataset_id}.parquet"
                if not self._upload_to_minio(parquet_file, object_name):
                    logger.error(f"Failed to upload Parquet file to MinIO")
                    return {}
            
            # Create the Trino table
            if not self._create_trino_table(schema_name, table_name, column_schema, partitioning_columns):
                logger.error(f"Failed to create Trino table")
                return {}
            
            # Register the dataset
            stats = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'original_size_bytes': original_size,
                'parquet_size_bytes': parquet_size,
                'compression_ratio': compression_ratio,
                'partitioning_columns': partitioning_columns
            }
            
            if not self._register_dataset(dataset_id, metadata, schema_name, table_name, stats):
                logger.error(f"Failed to register dataset")
                return {}
            
            # Return the result
            result = {
                'dataset_id': dataset_id,
                'schema_name': schema_name,
                'table_name': table_name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'partitioning_columns': partitioning_columns
            }
            
            logger.info(f"Successfully created Trino table for dataset {dataset_id}")
            return result
        except Exception as e:
            logger.error(f"Error creating Trino table from dataset {dataset_id}: {str(e)}")
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