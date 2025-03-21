import os
import json
import time
from datetime import datetime
from threading import Lock
from typing import Dict, List, Any, Optional
from logger_config import setup_logger

logger = setup_logger(__name__)

class ETLJobTracker:
    """Tracks ETL job status with thread-safe operations"""
    
    def __init__(self, storage_path="data_cache/etl_jobs.json"):
        self.storage_path = storage_path
        self.lock = Lock()
        self._ensure_storage_file()
        
    def _ensure_storage_file(self):
        """Ensure the storage file exists"""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "active_jobs": [],
                    "completed_jobs": [],
                    "failed_jobs": []
                }, f)
    
    def register_job(self, dataset_id: str, job_type: str) -> str:
        """Register a new ETL job and return its ID"""
        job_id = f"{job_type}_{dataset_id}_{int(time.time())}"
        job_info = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "job_type": job_type,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "details": {}
        }
        
        with self.lock:
            data = self._read_storage()
            data["active_jobs"].append(job_info)
            self._write_storage(data)
            
        logger.info(f"Registered ETL job: {job_id} for dataset {dataset_id}")
        return job_id
    
    def complete_job(self, job_id: str, details: Dict[str, Any] = None) -> bool:
        """Mark a job as completed"""
        with self.lock:
            data = self._read_storage()
            
            # Find the job in active jobs
            job_index = None
            job_info = None
            for i, job in enumerate(data["active_jobs"]):
                if job["job_id"] == job_id:
                    job_index = i
                    job_info = job
                    break
            
            if job_index is None:
                logger.warning(f"Job {job_id} not found in active jobs")
                return False
            
            # Update job info
            job_info["status"] = "completed"
            job_info["end_time"] = datetime.now().isoformat()
            if details:
                job_info["details"] = details
            
            # Move from active to completed
            data["active_jobs"].pop(job_index)
            data["completed_jobs"].insert(0, job_info)  # Most recent first
            
            # Trim completed jobs list if it gets too long
            if len(data["completed_jobs"]) > 100:
                data["completed_jobs"] = data["completed_jobs"][:100]
                
            self._write_storage(data)
            
        logger.info(f"Marked ETL job {job_id} as completed")
        return True
    
    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark a job as failed"""
        with self.lock:
            data = self._read_storage()
            
            # Find the job in active jobs
            job_index = None
            job_info = None
            for i, job in enumerate(data["active_jobs"]):
                if job["job_id"] == job_id:
                    job_index = i
                    job_info = job
                    break
            
            if job_index is None:
                logger.warning(f"Job {job_id} not found in active jobs")
                return False
            
            # Update job info
            job_info["status"] = "failed"
            job_info["end_time"] = datetime.now().isoformat()
            job_info["error"] = error
            
            # Move from active to failed
            data["active_jobs"].pop(job_index)
            data["failed_jobs"].insert(0, job_info)  # Most recent first
            
            # Trim failed jobs list if it gets too long
            if len(data["failed_jobs"]) > 100:
                data["failed_jobs"] = data["failed_jobs"][:100]
                
            self._write_storage(data)
            
        logger.error(f"Marked ETL job {job_id} as failed: {error}")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of all ETL jobs"""
        with self.lock:
            data = self._read_storage()
            
        return {
            "active_jobs": len(data["active_jobs"]),
            "completed_jobs": len(data["completed_jobs"]),
            "failed_jobs": len(data["failed_jobs"]),
            "details": {
                "active": data["active_jobs"],
                "completed": data["completed_jobs"][:10],  # Just return the 10 most recent
                "failed": data["failed_jobs"][:10]
            }
        }
    
    def _read_storage(self) -> Dict[str, List[Dict[str, Any]]]:
        """Read job data from storage file"""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading ETL job storage: {e}")
            return {"active_jobs": [], "completed_jobs": [], "failed_jobs": []}
    
    def _write_storage(self, data: Dict[str, List[Dict[str, Any]]]):
        """Write job data to storage file"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing ETL job storage: {e}")

# Singleton instance for global access
_job_tracker = None

def get_job_tracker() -> ETLJobTracker:
    """Get the singleton ETLJobTracker instance"""
    global _job_tracker
    if _job_tracker is None:
        _job_tracker = ETLJobTracker()
    return _job_tracker 