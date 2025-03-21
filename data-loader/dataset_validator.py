import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from logger_config import setup_logger
from trino.dbapi import connect
from env_config import get_trino_credentials
from minio import Minio
from env_config import get_minio_credentials

logger = setup_logger(__name__)

class DatasetValidator:
    """Unified dataset validation and verification functionality"""
    
    def __init__(self):
        # Get credentials from environment
        self.trino_creds = get_trino_credentials()
        self.minio_creds = get_minio_credentials()
    
    def validate_dataset_structure(self, dataset_path: str) -> Dict[str, Any]:
        """Validate a dataset's file structure and format"""
        try:
            path = Path(dataset_path)
            
            if not path.exists():
                return {
                    "valid": False,
                    "error": f"Dataset path {dataset_path} does not exist"
                }
                
            if path.suffix.lower() != '.parquet':
                return {
                    "valid": False,
                    "error": f"Dataset file {dataset_path} is not a Parquet file"
                }
                
            # Check if file is readable as Parquet
            try:
                import pyarrow.parquet as pq
                table = pq.read_metadata(dataset_path)
                
                return {
                    "valid": True,
                    "row_groups": table.num_row_groups,
                    "schema": str(table.schema),
                    "size_bytes": os.path.getsize(dataset_path)
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Invalid Parquet format: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error validating dataset structure: {str(e)}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def verify_trino_table(self, schema_name: str, table_name: str) -> Dict[str, Any]:
        """Verify if a table exists in Trino and check its properties"""
        try:
            # Connect to Trino
            conn = connect(
                host=self.trino_creds['host'],
                port=int(self.trino_creds['port']),
                user=self.trino_creds['user'],
                catalog=self.trino_creds['catalog']
            )
            
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f"""
                SELECT table_name, table_type
                FROM {self.trino_creds['catalog']}.information_schema.tables 
                WHERE table_schema = '{schema_name}'
                AND table_name = '{table_name}'
            """)
            
            table_info = cursor.fetchone()
            
            if not table_info:
                return {
                    "exists": False,
                    "error": f"Table {schema_name}.{table_name} does not exist in Trino"
                }
                
            # Get column information
            cursor.execute(f"""
                SELECT column_name, data_type
                FROM {self.trino_creds['catalog']}.information_schema.columns
                WHERE table_schema = '{schema_name}'
                AND table_name = '{table_name}'
            """)
            
            columns = [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]
            
            # Get row count (might be expensive for large tables)
            cursor.execute(f"SELECT count(*) FROM {schema_name}.{table_name}")
            row_count = cursor.fetchone()[0]
            
            return {
                "exists": True,
                "type": table_info[1],
                "row_count": row_count,
                "columns": columns,
                "column_count": len(columns)
            }
            
        except Exception as e:
            logger.error(f"Error verifying Trino table: {str(e)}")
            return {
                "exists": False,
                "error": str(e)
            }
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def verify_minio_objects(self, bucket: str, prefix: str) -> Dict[str, Any]:
        """Verify if objects exist in MinIO for a given dataset"""
        try:
            # Initialize MinIO client
            minio_client = Minio(
                self.minio_creds['endpoint'],
                access_key=self.minio_creds['access_key'],
                secret_key=self.minio_creds['secret_key'],
                secure=self.minio_creds['secure']
            )
            
            # Check if bucket exists
            if not minio_client.bucket_exists(bucket):
                return {
                    "exists": False,
                    "error": f"Bucket {bucket} does not exist in MinIO"
                }
                
            # List objects with the given prefix
            objects = list(minio_client.list_objects(bucket, prefix=prefix, recursive=True))
            
            if not objects:
                return {
                    "exists": False,
                    "error": f"No objects found with prefix {prefix} in bucket {bucket}"
                }
                
            total_size = sum(obj.size for obj in objects)
            
            return {
                "exists": True,
                "object_count": len(objects),
                "total_size_bytes": total_size,
                "objects": [{"name": obj.object_name, "size": obj.size} for obj in objects[:10]]  # List only first 10
            }
            
        except Exception as e:
            logger.error(f"Error verifying MinIO objects: {str(e)}")
            return {
                "exists": False,
                "error": str(e)
            }
    
    def run_comprehensive_check(self, dataset_id: str, schema_name: str, table_name: str) -> Dict[str, Any]:
        """Run a comprehensive validation check on a dataset across all systems"""
        from cache_manager import DatasetCacheManager
        
        results = {
            "dataset_id": dataset_id,
            "schema_name": schema_name,
            "table_name": table_name,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 1. Check cache
        cache_manager = DatasetCacheManager()
        cache_info = cache_manager.get_dataset_info(dataset_id)
        results["cache"] = {
            "exists": cache_info is not None
        }
        if cache_info:
            results["cache"].update({
                "file_path": cache_info.get("file_path"),
                "row_count": cache_info.get("row_count"),
                "last_updated": cache_info.get("updated_at")
            })
            
            # Validate the parquet file structure
            if "file_path" in cache_info and os.path.exists(cache_info["file_path"]):
                results["parquet_validation"] = self.validate_dataset_structure(cache_info["file_path"])
        
        # 2. Check Trino table
        results["trino"] = self.verify_trino_table(schema_name, table_name)
        
        # 3. Check MinIO storage
        # Assuming a standard pattern for MinIO objects
        minio_prefix = f"{schema_name}/{table_name}"
        results["minio"] = self.verify_minio_objects("iceberg", minio_prefix)
        
        # 4. Overall validation result
        results["is_valid"] = (
            results.get("cache", {}).get("exists", False) and
            results.get("trino", {}).get("exists", False) and
            results.get("minio", {}).get("exists", False) and
            results.get("parquet_validation", {}).get("valid", False)
        )
        
        return results

# Create a singleton instance
_validator = None

def get_validator() -> DatasetValidator:
    """Get the singleton DatasetValidator instance"""
    global _validator
    if _validator is None:
        _validator = DatasetValidator()
    return _validator 