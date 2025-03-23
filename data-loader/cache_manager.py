import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

class DatasetCacheManager:
    """Simplified cache manager that only tracks dataset metadata"""
    
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize database
        self.db_path = os.path.join(cache_dir, "dataset_registry.db")
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Create the registry table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_registry (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    last_updated TEXT,
                    cached_at TEXT,
                    row_count INTEGER,
                    file_size INTEGER,
                    file_path TEXT,
                    metadata_path TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def connect(self):
        """Connect to the SQLite database"""
        return sqlite3.connect(self.db_path)
    
    def update_dataset_cache(self, dataset_id: str, name: str, description: str, 
                           row_count: int, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """Update or create a dataset cache entry"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Save metadata if provided
            metadata_path = None
            if metadata:
                metadata_path = os.path.join(self.cache_dir, f"{dataset_id}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Get file size
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Insert or update dataset record
            cursor.execute("""
                INSERT OR REPLACE INTO dataset_registry
                (dataset_id, name, description, last_updated, cached_at, row_count, file_size, file_path, metadata_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                name,
                description,
                metadata.get("last_updated", datetime.now().isoformat()) if metadata else datetime.now().isoformat(),
                datetime.now().isoformat(),
                row_count,
                file_size,
                file_path,
                metadata_path
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated cache for dataset {dataset_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating cache for {dataset_id}: {e}")
            return False
    
    def is_dataset_cached(self, dataset_id: str) -> bool:
        """Check if a dataset is cached"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("SELECT file_path FROM dataset_registry WHERE dataset_id = ?", (dataset_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] and os.path.exists(result[0]):
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking if dataset {dataset_id} is cached: {e}")
            return False
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a cached dataset"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dataset_id, name, description, last_updated, cached_at, 
                       row_count, file_size, file_path, metadata_path 
                FROM dataset_registry
                WHERE dataset_id = ?
            """, (dataset_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return None
                
            return {
                "dataset_id": result[0],
                "name": result[1],
                "description": result[2],
                "last_updated": result[3],
                "cached_at": result[4],
                "row_count": result[5],
                "file_size": result[6],
                "file_path": result[7],
                "metadata_path": result[8]
            }
        except Exception as e:
            logger.error(f"Error getting dataset info for {dataset_id}: {e}")
            return None
    
    def get_all_cached_datasets(self) -> List[Dict[str, Any]]:
        """Get information about all cached datasets"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dataset_id, name, description, last_updated, cached_at, 
                       row_count, file_size, file_path, metadata_path 
                FROM dataset_registry
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            datasets = []
            for row in results:
                datasets.append({
                    "dataset_id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "last_updated": row[3],
                    "cached_at": row[4],
                    "row_count": row[5],
                    "file_size": row[6],
                    "file_path": row[7],
                    "metadata_path": row[8]
                })
                
            return datasets
        except Exception as e:
            logger.error(f"Error getting all cached datasets: {e}")
            return []
            
    def remove_dataset(self, dataset_id: str) -> bool:
        """Remove a dataset from the cache
        
        Args:
            dataset_id: The Socrata dataset ID to remove
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get dataset info first
            dataset_info = self.get_dataset_info(dataset_id)
            if not dataset_info:
                logger.warning(f"Dataset {dataset_id} not found in cache")
                return False
                
            # Remove metadata file if it exists
            metadata_path = dataset_info.get("metadata_path")
            if metadata_path and os.path.exists(metadata_path):
                try:
                    os.remove(metadata_path)
                    logger.info(f"Removed metadata file: {metadata_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove metadata file {metadata_path}: {e}")
            
            # Remove dataset file if it exists
            file_path = dataset_info.get("file_path")
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Removed dataset file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove dataset file {file_path}: {e}")
            
            # Remove from database
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dataset_registry WHERE dataset_id = ?", (dataset_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully removed dataset {dataset_id} from cache")
            return True
            
        except Exception as e:
            logger.error(f"Error removing dataset {dataset_id} from cache: {e}")
            return False
    
    def update_dataset_table_info(self, dataset_id: str, schema_name: str, table_name: str) -> bool:
        """Update the table information for a dataset
        
        Args:
            dataset_id: The Socrata dataset ID
            schema_name: The Trino schema name
            table_name: The Trino table name
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Add the schema_name and table_name columns if they don't exist
            try:
                cursor.execute("ALTER TABLE dataset_registry ADD COLUMN schema_name TEXT")
                cursor.execute("ALTER TABLE dataset_registry ADD COLUMN table_name TEXT")
                logger.info("Added schema_name and table_name columns to dataset_registry table")
            except sqlite3.OperationalError:
                # Columns likely already exist
                pass
            
            # Update the schema and table info
            cursor.execute("""
                UPDATE dataset_registry 
                SET schema_name = ?, table_name = ? 
                WHERE dataset_id = ?
            """, (schema_name, table_name, dataset_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated table info for dataset {dataset_id}: {schema_name}.{table_name}")
            return True
        except Exception as e:
            logger.error(f"Error updating table info for dataset {dataset_id}: {e}")
            return False
            
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a cached dataset
        
        Args:
            dataset_id: The Socrata dataset ID
            
        Returns:
            Dictionary with dataset metadata or empty dict if not found
        """
        try:
            # Get dataset info to find metadata path
            dataset_info = self.get_dataset_info(dataset_id)
            if not dataset_info:
                logger.warning(f"Dataset {dataset_id} not found in cache")
                return {}
                
            # Check if metadata file exists
            metadata_path = dataset_info.get("metadata_path")
            if not metadata_path or not os.path.exists(metadata_path):
                logger.warning(f"No metadata file found for dataset {dataset_id}")
                
                # Return a minimal metadata object with information from dataset_info
                return {
                    "dataset_id": dataset_id,
                    "name": dataset_info.get("name", f"Dataset {dataset_id}"),
                    "description": dataset_info.get("description", ""),
                    "row_count": dataset_info.get("row_count", 0),
                }
            
            # Read metadata from file
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.debug(f"Retrieved metadata for dataset {dataset_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting metadata for dataset {dataset_id}: {e}")
            # Return minimal metadata to avoid breaking the client code
            return {
                "dataset_id": dataset_id,
                "name": f"Dataset {dataset_id}",
                "description": "",
                "error": str(e)
            } 