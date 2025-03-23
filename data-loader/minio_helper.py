import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from app_config import get_config
from logger_config import setup_logger

logger = setup_logger(__name__)
config = get_config()

class MinioManager:
    """Manages operations with MinIO storage"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MinioManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the MinIO manager with configuration"""
        self.credentials = config.get_minio_credentials()
        self.client = self._create_client()
        self.bucket_name = self.credentials.get('bucket', 'iceberg')
        
        # Ensure the bucket exists
        self.ensure_bucket_exists()
    
    def _create_client(self) -> Minio:
        """Create a MinIO client using configuration"""
        try:
            client = Minio(
                endpoint=self.credentials['endpoint'],
                access_key=self.credentials['access_key'],
                secret_key=self.credentials['secret_key'],
                secure=self.credentials['secure']
            )
            logger.debug(f"MinIO client created for endpoint {self.credentials['endpoint']}")
            return client
        except Exception as e:
            logger.error(f"Error creating MinIO client: {e}")
            raise
    
    def ensure_bucket_exists(self) -> bool:
        """Ensure the configured bucket exists"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                logger.info(f"Creating MinIO bucket: {self.bucket_name}")
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Successfully created bucket: {self.bucket_name}")
                return True
            
            logger.debug(f"Bucket {self.bucket_name} already exists")
            return True
        except Exception as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            raise
    
    def upload_file(self, local_path: str, object_name: str = None) -> bool:
        """
        Upload a file to MinIO
        
        Args:
            local_path: Path to the local file
            object_name: Name to use in the MinIO bucket (default: filename from local_path)
            
        Returns:
            bool: True if upload was successful
        """
        if not object_name:
            object_name = os.path.basename(local_path)
            
        try:
            logger.info(f"Uploading file {local_path} to MinIO as {object_name}")
            
            # Get file stats for progress info
            file_size = os.path.getsize(local_path)
            
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=local_path
            )
            
            logger.info(f"Successfully uploaded {file_size} bytes to {self.bucket_name}/{object_name}")
            return True
        except Exception as e:
            logger.error(f"Error uploading file to MinIO: {e}")
            return False
    
    def download_file(self, object_name: str, local_path: str) -> bool:
        """
        Download a file from MinIO
        
        Args:
            object_name: Name of the object in MinIO
            local_path: Path to save the file locally
            
        Returns:
            bool: True if download was successful
        """
        try:
            logger.info(f"Downloading file {object_name} from MinIO to {local_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=local_path
            )
            
            logger.info(f"Successfully downloaded {self.bucket_name}/{object_name} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading file from MinIO: {e}")
            return False
    
    def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in MinIO"""
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except Exception as e:
            return False
    
    def list_files(self, prefix: str = "", recursive: bool = True) -> List[str]:
        """
        List files in the MinIO bucket
        
        Args:
            prefix: Prefix to filter results
            recursive: Whether to list files recursively
            
        Returns:
            List of object names
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=recursive
            )
            
            return [obj.object_name for obj in objects]
        except Exception as e:
            logger.error(f"Error listing files in MinIO: {e}")
            return []
    
    def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from MinIO
        
        Args:
            object_name: Name of the object to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            logger.info(f"Deleting file {object_name} from MinIO")
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Successfully deleted {self.bucket_name}/{object_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from MinIO: {e}")
            return False
    
    def clean_bucket(self, prefix: str = "") -> int:
        """
        Clean the MinIO bucket by removing objects
        
        Args:
            prefix: Prefix to filter objects to delete
            
        Returns:
            int: Number of objects deleted
        """
        try:
            logger.info(f"Cleaning MinIO bucket with prefix: {prefix}")
            
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            
            object_count = 0
            
            for obj in objects:
                self.client.remove_object(self.bucket_name, obj.object_name)
                object_count += 1
                
                if object_count % 100 == 0:
                    logger.info(f"Deleted {object_count} objects so far")
            
            logger.info(f"Successfully cleaned MinIO bucket. Deleted {object_count} objects.")
            return object_count
        except Exception as e:
            logger.error(f"Error cleaning MinIO bucket: {e}")
            raise

# Singleton instance for global access
def get_minio_manager():
    """Get the singleton MinioManager instance"""
    return MinioManager() 