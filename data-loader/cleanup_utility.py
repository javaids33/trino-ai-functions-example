import os
import shutil
import tempfile
from trino.dbapi import connect
from datetime import datetime, timedelta
from minio import Minio
from logger_config import setup_logger
from env_config import get_trino_credentials, get_minio_credentials

# Set up logger
logger = setup_logger(__name__)

def cleanup_temp_files():
    """Cleanup temporary files created during ETL process"""
    try:
        # Get standard temp directory
        temp_dir = tempfile.gettempdir()
        pattern = "tmp_*_socrata"
        
        # Find all temp directories matching our pattern
        count = 0
        for item in os.listdir(temp_dir):
            if pattern in item:
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                        count += 1
                        logger.info(f"Removed temporary directory: {item_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove {item_path}: {str(e)}")
        
        logger.info(f"Cleaned up {count} temporary directories")
        return count
    except Exception as e:
        logger.error(f"Error cleaning temporary files: {str(e)}")
        return 0

def list_unused_datasets(min_days_unused=30, min_rows=100):
    """List datasets that haven't been accessed in a specified time period"""
    try:
        # Get Trino credentials from environment config
        trino_creds = get_trino_credentials()
        
        # Connect to Trino
        conn = connect(
            host=trino_creds['host'],
            port=int(trino_creds['port']),
            user=trino_creds['user'],
            catalog=trino_creds['catalog']
        )
        
        cursor = conn.cursor()
        
        # Calculate cutoff date
        cutoff_date = (datetime.now() - timedelta(days=min_days_unused)).strftime("%Y-%m-%d")
        
        # Query for potentially unused datasets
        cursor.execute(f"""
            SELECT 
                r.dataset_id, 
                r.schema_name,
                r.table_name,
                r.dataset_title,
                r.row_count,
                r.etl_timestamp,
                r.last_updated
            FROM iceberg.nycdata.dataset_registry r
            WHERE r.row_count <= {min_rows}
               OR r.etl_timestamp < timestamp '{cutoff_date}'
            ORDER BY r.row_count
        """)
        
        results = cursor.fetchall()
        
        # Format results
        unused_datasets = []
        for row in results:
            unused_datasets.append({
                "dataset_id": row[0],
                "schema_name": row[1],
                "table_name": row[2],
                "title": row[3],
                "row_count": row[4],
                "etl_timestamp": row[5],
                "last_updated": row[6]
            })
            
        logger.info(f"Found {len(unused_datasets)} potentially unused datasets")
        return unused_datasets
        
    except Exception as e:
        logger.error(f"Error listing unused datasets: {str(e)}")
        return []

def remove_dataset(dataset_id, schema_name, table_name):
    """Remove a dataset from Trino and MinIO"""
    try:
        logger.info(f"Removing dataset {dataset_id} from {schema_name}.{table_name}")
        
        # Get Trino credentials from environment config
        trino_creds = get_trino_credentials()
        
        # Connect to Trino
        conn = connect(
            host=trino_creds['host'],
            port=int(trino_creds['port']),
            user=trino_creds['user'],
            catalog=trino_creds['catalog']
        )
        
        cursor = conn.cursor()
        
        # 1. Drop the table
        try:
            cursor.execute(f"DROP TABLE IF EXISTS iceberg.{schema_name}.{table_name}")
            logger.info(f"Dropped table iceberg.{schema_name}.{table_name}")
        except Exception as e:
            logger.error(f"Error dropping table: {str(e)}")
        
        # 2. Remove from registry
        try:
            cursor.execute(f"""
                DELETE FROM iceberg.nycdata.dataset_registry 
                WHERE dataset_id = '{dataset_id}'
            """)
            logger.info(f"Removed dataset {dataset_id} from registry")
        except Exception as e:
            logger.error(f"Error removing from registry: {str(e)}")
        
        # 3. Remove files from MinIO
        try:
            # Get MinIO credentials from environment config
            minio_creds = get_minio_credentials()
            
            minio_client = Minio(
                minio_creds['endpoint'],
                access_key=minio_creds['access_key'],
                secret_key=minio_creds['secret_key'],
                secure=False
            )
            
            # List objects with prefix matching this dataset
            bucket_name = "iceberg"
            prefix = f"data/{schema_name}/{table_name}/"
            
            objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
            
            # Delete each object
            for obj in objects:
                minio_client.remove_object(bucket_name, obj.object_name)
                logger.info(f"Removed object {obj.object_name} from MinIO")
                
            logger.info(f"Removed dataset files from MinIO")
            
        except Exception as e:
            logger.error(f"Error removing files from MinIO: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error removing dataset {dataset_id}: {str(e)}")
        return False 