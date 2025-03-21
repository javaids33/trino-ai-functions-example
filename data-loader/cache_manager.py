import os
import json
import hashlib
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

class DatasetCacheManager:
    """Manages local caching of datasets to minimize API calls and provide persistence"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Base directory for cached data
        """
        self.cache_dir = Path(cache_dir)
        self.registry_file = self.cache_dir / "registry.json"
        self.datasets_dir = self.cache_dir / "datasets"
        
        # Create required directories
        self.cache_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Initialize or load registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load the dataset registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                return {"datasets": {}, "last_updated": datetime.now().isoformat()}
        else:
            # Create new registry
            registry = {"datasets": {}, "last_updated": datetime.now().isoformat()}
            self._save_registry(registry)
            return registry
    
    def _save_registry(self, registry: Dict[str, Any] = None) -> None:
        """Save the dataset registry to disk"""
        if registry is None:
            registry = self.registry
        
        try:
            registry["last_updated"] = datetime.now().isoformat()
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def get_dataset_path(self, dataset_id: str) -> Path:
        """Get the path for a dataset's cache directory"""
        return self.datasets_dir / dataset_id
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a cached dataset"""
        return self.registry["datasets"].get(dataset_id)
    
    def is_dataset_cached(self, dataset_id: str) -> bool:
        """Check if a dataset is cached locally"""
        # Check registry
        if dataset_id not in self.registry["datasets"]:
            return False
        
        # Check if files exist
        dataset_path = self.get_dataset_path(dataset_id)
        parquet_file = dataset_path / f"{dataset_id}.parquet"
        metadata_file = dataset_path / "metadata.json"
        
        return dataset_path.exists() and parquet_file.exists() and metadata_file.exists()
    
    def update_dataset_cache(self, dataset_id: str, 
                            parquet_file_path: str, 
                            metadata: Dict[str, Any],
                            row_count: int) -> Dict[str, Any]:
        """
        Update the cache with a new dataset file and metadata
        
        Args:
            dataset_id: The Socrata dataset ID
            parquet_file_path: Path to the Parquet file
            metadata: Dataset metadata
            row_count: Number of rows in the dataset
            
        Returns:
            Information about the cached dataset
        """
        # Create dataset directory
        dataset_path = self.get_dataset_path(dataset_id)
        dataset_path.mkdir(exist_ok=True)
        
        # Copy parquet file to cache
        dest_file = dataset_path / f"{dataset_id}.parquet"
        shutil.copy2(parquet_file_path, dest_file)
        
        # Save metadata
        with open(dataset_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        cache_info = {
            "dataset_id": dataset_id,
            "cached_at": datetime.now().isoformat(),
            "last_updated": metadata.get("updatedAt", datetime.now().isoformat()),
            "row_count": row_count,
            "file_size": os.path.getsize(dest_file),
            "file_path": str(dest_file)
        }
        
        self.registry["datasets"][dataset_id] = cache_info
        self._save_registry()
        
        return cache_info
    
    def is_update_needed(self, dataset_id: str, remote_update_time: str, remote_row_count: int) -> bool:
        """
        Check if a dataset needs to be updated based on remote information
        
        Args:
            dataset_id: The Socrata dataset ID
            remote_update_time: Last update time from Socrata
            remote_row_count: Row count from Socrata
            
        Returns:
            True if update is needed, False otherwise
        """
        if not self.is_dataset_cached(dataset_id):
            return True
            
        dataset_info = self.get_dataset_info(dataset_id)
        
        # Convert string timestamps to datetime for comparison
        try:
            remote_time = datetime.fromisoformat(remote_update_time.replace('Z', '+00:00'))
            local_time = datetime.fromisoformat(dataset_info["last_updated"].replace('Z', '+00:00'))
            
            # Update if remote is newer or has more rows
            if remote_time > local_time or remote_row_count > dataset_info["row_count"]:
                return True
        except Exception as e:
            logger.error(f"Error comparing update times: {e}")
            return True
            
        return False
    
    def get_all_cached_datasets(self) -> List[Dict[str, Any]]:
        """Get information about all cached datasets"""
        return list(self.registry["datasets"].values())
    
    def create_dataset_archive(self, output_path: str, dataset_ids: List[str] = None) -> str:
        """
        Create a zip archive of selected datasets for portability
        
        Args:
            output_path: Path to save the archive
            dataset_ids: List of dataset IDs to include, or None for all
            
        Returns:
            Path to the created archive
        """
        if dataset_ids is None:
            # Include all datasets
            dataset_ids = list(self.registry["datasets"].keys())
        
        # Create a temporary directory with the registry and selected datasets
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create registry subset
            subset_registry = {
                "datasets": {k: self.registry["datasets"][k] for k in dataset_ids if k in self.registry["datasets"]},
                "created_at": datetime.now().isoformat(),
                "source_machine": os.uname().nodename
            }
            
            # Save registry
            with open(os.path.join(temp_dir, "registry.json"), 'w') as f:
                json.dump(subset_registry, f, indent=2)
                
            # Copy dataset files
            for dataset_id in dataset_ids:
                if self.is_dataset_cached(dataset_id):
                    src_dir = self.get_dataset_path(dataset_id)
                    dest_dir = os.path.join(temp_dir, "datasets", dataset_id)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Copy files
                    for item in os.listdir(src_dir):
                        src_path = os.path.join(src_dir, item)
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, os.path.join(dest_dir, item))
            
            # Create archive
            archive_path = f"{output_path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            shutil.make_archive(
                output_path,
                'zip',
                temp_dir
            )
            
            logger.info(f"Created dataset archive at {archive_path}")
            return archive_path
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def import_dataset_archive(self, archive_path: str, overwrite: bool = False) -> List[str]:
        """
        Import datasets from an archive
        
        Args:
            archive_path: Path to the archive
            overwrite: Whether to overwrite existing datasets
            
        Returns:
            List of imported dataset IDs
        """
        import tempfile
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        imported_datasets = []
        
        try:
            # Extract archive
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Load registry
            try:
                with open(os.path.join(temp_dir, "registry.json"), 'r') as f:
                    archive_registry = json.load(f)
            except Exception as e:
                logger.error(f"Error loading archive registry: {e}")
                return []
                
            # Import datasets
            for dataset_id, info in archive_registry["datasets"].items():
                if not overwrite and self.is_dataset_cached(dataset_id):
                    logger.info(f"Skipping existing dataset: {dataset_id}")
                    continue
                    
                src_dir = os.path.join(temp_dir, "datasets", dataset_id)
                if not os.path.exists(src_dir):
                    logger.error(f"Dataset directory not found in archive: {dataset_id}")
                    continue
                    
                # Copy dataset
                dest_dir = self.get_dataset_path(dataset_id)
                os.makedirs(dest_dir, exist_ok=True)
                
                for item in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, item)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, os.path.join(dest_dir, item))
                
                # Update registry
                self.registry["datasets"][dataset_id] = info
                # Update the file path to match local path
                self.registry["datasets"][dataset_id]["file_path"] = str(dest_dir / f"{dataset_id}.parquet")
                
                imported_datasets.append(dataset_id)
            
            self._save_registry()
            logger.info(f"Imported {len(imported_datasets)} datasets from archive")
            return imported_datasets
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)

    # Add new method for batch operations
    def batch_process_datasets(self, dataset_ids: List[str], 
                               process_function, 
                               max_concurrency: int = 3) -> Dict[str, Any]:
        """
        Process multiple datasets concurrently with controlled concurrency
        
        Args:
            dataset_ids: List of dataset IDs to process
            process_function: Function to call for each dataset
            max_concurrency: Maximum number of concurrent processes
            
        Returns:
            Dictionary with results for each dataset
        """
        import concurrent.futures
        
        logger.info(f"Batch processing {len(dataset_ids)} datasets with max concurrency {max_concurrency}")
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_dataset = {
                executor.submit(process_function, dataset_id): dataset_id 
                for dataset_id in dataset_ids
            }
            
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_id = future_to_dataset[future]
                try:
                    results[dataset_id] = future.result()
                    logger.info(f"Successfully processed dataset {dataset_id}")
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
                    results[dataset_id] = {"success": False, "error": str(e)}
        
        return results 

    def get_dataset_metadata(self, dataset_id):
        """Get metadata for a cached dataset"""
        metadata_path = os.path.join(self.cache_dir, f"{dataset_id}_metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    import json
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading metadata for {dataset_id}: {e}")
                return None
        else:
            logger.info(f"No cached metadata found for {dataset_id}")
            return None
        
    def save_dataset_metadata(self, dataset_id, metadata):
        """Save metadata for a dataset"""
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        metadata_path = os.path.join(self.cache_dir, f"{dataset_id}_metadata.json")
        
        try:
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata for {dataset_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata for {dataset_id}: {e}")
            return False 