import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import threading
from colorama import Fore

logger = logging.getLogger(__name__)

class MonitoringService:
    """Central service for monitoring Trino AI activity"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of monitoring service"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """Initialize monitoring service"""
        self.query_history = []
        self.agent_activity = {}
        self.error_log = []
        self.performance_metrics = {}
        self.system_health = {
            "startup_time": datetime.now().isoformat(),
            "status": "healthy"
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        logger.info(f"{Fore.GREEN}Monitoring service initialized{Fore.RESET}")
    
    def log_query(self, query_data: Dict[str, Any]):
        """Log a new query"""
        query_id = query_data.get("query_id", f"q-{int(time.time())}")
        timestamp = datetime.now().isoformat()
        
        entry = {
            "query_id": query_id,
            "timestamp": timestamp,
            "natural_language": query_data.get("natural_language", ""),
            "generated_sql": query_data.get("generated_sql", ""),
            "status": query_data.get("status", "submitted"),
            "user": query_data.get("user", "anonymous"),
            "execution_time_ms": query_data.get("execution_time_ms", 0),
            "result_count": query_data.get("result_count", 0)
        }
        
        self.query_history.append(entry)
        self._write_to_log("query_history.jsonl", entry)
        logger.debug(f"Logged query: {query_id}")
        return query_id
    
    def update_query_status(self, query_id: str, status: str, **kwargs):
        """Update the status of a query"""
        for query in self.query_history:
            if query["query_id"] == query_id:
                query["status"] = status
                query["last_updated"] = datetime.now().isoformat()
                
                # Update any additional fields
                for key, value in kwargs.items():
                    query[key] = value
                
                self._write_to_log("query_updates.jsonl", {
                    "query_id": query_id,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                    **kwargs
                })
                return True
        return False
    
    def log_agent_activity(self, agent_name: str, activity_data: Dict[str, Any]):
        """Log activity from an agent"""
        timestamp = datetime.now().isoformat()
        
        if agent_name not in self.agent_activity:
            self.agent_activity[agent_name] = []
        
        entry = {
            "timestamp": timestamp,
            "agent": agent_name,
            "action": activity_data.get("action", "unknown"),
            "details": activity_data.get("details", {}),
            "duration_ms": activity_data.get("duration_ms", 0),
            "query_id": activity_data.get("query_id", "unknown")
        }
        
        self.agent_activity[agent_name].append(entry)
        self._write_to_log("agent_activity.jsonl", entry)
        return True
    
    def log_error(self, error_data: Dict[str, Any]):
        """Log an error"""
        timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "error_type": error_data.get("error_type", "unknown"),
            "message": error_data.get("message", "No message provided"),
            "source": error_data.get("source", "unknown"),
            "query_id": error_data.get("query_id", "unknown"),
            "stack_trace": error_data.get("stack_trace", None)
        }
        
        self.error_log.append(entry)
        self._write_to_log("error_log.jsonl", entry)
        
        # Update system health if critical error
        if error_data.get("is_critical", False):
            self.system_health["status"] = "degraded"
            self.system_health["last_critical_error"] = timestamp
        
        return True
    
    def update_performance_metric(self, metric_name: str, value: Any):
        """Update a performance metric"""
        timestamp = datetime.now().isoformat()
        
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        entry = {
            "timestamp": timestamp,
            "value": value
        }
        
        self.performance_metrics[metric_name].append(entry)
        self._write_to_log("performance_metrics.jsonl", {
            "metric": metric_name,
            "timestamp": timestamp,
            "value": value
        })
        return True
    
    def get_recent_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent queries"""
        return sorted(
            self.query_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def get_agent_activity(self, agent_name: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get agent activity"""
        if agent_name:
            if agent_name in self.agent_activity:
                return sorted(
                    self.agent_activity[agent_name],
                    key=lambda x: x["timestamp"],
                    reverse=True
                )[:limit]
            return []
        
        # Combine all agent activity and sort
        all_activity = []
        for activities in self.agent_activity.values():
            all_activity.extend(activities)
        
        return sorted(
            all_activity,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def get_error_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the error log"""
        return sorted(
            self.error_log,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def get_performance_metrics(self, metric_name: Optional[str] = None,
                               limit: int = 100) -> Dict[str, Any]:
        """Get performance metrics"""
        if metric_name:
            if metric_name in self.performance_metrics:
                return {
                    metric_name: sorted(
                        self.performance_metrics[metric_name],
                        key=lambda x: x["timestamp"],
                        reverse=True
                    )[:limit]
                }
            return {metric_name: []}
        
        # Return all metrics with limited history
        result = {}
        for name, metrics in self.performance_metrics.items():
            result[name] = sorted(
                metrics,
                key=lambda x: x["timestamp"],
                reverse=True
            )[:limit]
        
        return result
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information"""
        # Update with current timestamp
        self.system_health["current_time"] = datetime.now().isoformat()
        self.system_health["uptime_seconds"] = (
            datetime.now() - 
            datetime.fromisoformat(self.system_health["startup_time"])
        ).total_seconds()
        
        return self.system_health
    
    def _write_to_log(self, filename: str, data: Dict[str, Any]):
        """Write an entry to the specified log file"""
        try:
            with open(f"logs/{filename}", "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Error writing to log file {filename}: {str(e)}")

# Create global instance
monitoring_service = MonitoringService.get_instance() 