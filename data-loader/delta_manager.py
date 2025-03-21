import pandas as pd
import datetime
from trino.dbapi import connect
from sodapy import Socrata
from typing import Dict, Any, Optional, List
from logger_config import setup_logger
from env_config import get_socrata_credentials, get_trino_credentials

# Set up logger
logger = setup_logger(__name__)

class DeltaManager:
    """Manages delta loading for NYC datasets to minimize data transfer and storage"""
    
    def __init__(self):
        # Get Trino credentials from environment config
        trino_creds = get_trino_credentials()
        self.trino_host = trino_creds['host']
        self.trino_port = int(trino_creds['port'])
        self.trino_user = trino_creds['user']
        self.trino_catalog = trino_creds['catalog']
        self.registry_schema = "nycdata"
        self.registry_table = "dataset_registry"
        
        # Get Socrata credentials from environment config
        socrata_creds = get_socrata_credentials()
        self.domain = socrata_creds['domain']
        self.app_token = socrata_creds['app_token']
        self.api_key_id = socrata_creds['api_key_id']
        self.api_key_secret = socrata_creds['api_key_secret']
        
        # Initialize Socrata client
        self.client = self._init_socrata_client()
        
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
    
    def _get_trino_connection(self):
        """Get a connection to Trino"""
        try:
            conn = connect(
                host=self.trino_host,
                port=self.trino_port,
                user=self.trino_user,
                catalog=self.trino_catalog
            )
            return conn
        except Exception as e:
            logger.error(f"Error connecting to Trino: {e}")
            raise
    
    def get_dataset_status(self, dataset_id: str) -> Dict[str, Any]:
        """Get the status of a dataset to determine if it needs to be refreshed"""
        try:
            # Get dataset info from registry
            conn = self._get_trino_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute(f"""
                    SELECT 
                        dataset_id, schema_name, table_name, row_count, 
                        etl_timestamp, last_updated
                    FROM {self.trino_catalog}.{self.registry_schema}.{self.registry_table}
                    WHERE dataset_id = '{dataset_id}'
                """)
                
                result = cursor.fetchone()
                
                if not result:
                    return {
                        "dataset_id": dataset_id,
                        "exists_in_registry": False,
                        "needs_refresh": True,
                        "reason": "Dataset not found in registry"
                    }
                
                # Get dataset info from Socrata
                try:
                    # Get metadata to check for update date
                    metadata = self.client.get_metadata(dataset_id)
                    
                    # Get row count from Socrata
                    source_row_count = int(metadata.get('rowsUpdatedAt', 0))
                    
                    # Get last updated date
                    last_source_update = None
                    if 'rowsUpdatedAt' in metadata:
                        try:
                            last_source_update = datetime.datetime.fromtimestamp(metadata['rowsUpdatedAt'])
                        except (ValueError, TypeError):
                            pass
                    
                    # Get dataset info from registry
                    dataset_info = {
                        "dataset_id": result[0],
                        "schema_name": result[1],
                        "table_name": result[2],
                        "current_row_count": result[3],
                        "last_etl_run": result[4],
                        "last_updated": result[5],
                        "source_row_count": source_row_count,
                        "last_source_update": last_source_update,
                        "exists_in_registry": True
                    }
                    
                    # Check if refresh is needed
                    refresh_info = self._check_if_refresh_needed(dataset_info)
                    dataset_info.update(refresh_info)
                    
                    return dataset_info
                    
                except Exception as e:
                    logger.error(f"Error getting dataset info from Socrata: {e}")
                    return {
                        "dataset_id": dataset_id,
                        "exists_in_registry": True,
                        "needs_refresh": True,
                        "reason": f"Error checking source: {str(e)}"
                    }
                    
            except Exception as e:
                logger.error(f"Error checking dataset status: {e}")
                return {
                    "dataset_id": dataset_id,
                    "exists_in_registry": False,
                    "needs_refresh": True,
                    "reason": f"Error checking registry: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error checking dataset status: {e}")
            return {
                "dataset_id": dataset_id,
                "exists_in_registry": False,
                "needs_refresh": True,
                "reason": f"Error: {str(e)}"
            }
    
    def _check_if_refresh_needed(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a dataset needs to be refreshed based on its status"""
        needs_refresh = False
        reason = "No refresh needed"
        
        # Check if source has more rows than current
        if dataset_info.get("source_row_count", 0) > dataset_info.get("current_row_count", 0):
            needs_refresh = True
            reason = f"Source has more rows ({dataset_info['source_row_count']}) than current ({dataset_info['current_row_count']})"
        
        # Check if source was updated after last ETL run
        elif dataset_info.get("last_source_update") and dataset_info.get("last_etl_run"):
            if dataset_info["last_source_update"] > dataset_info["last_etl_run"]:
                needs_refresh = True
                reason = f"Source was updated ({dataset_info['last_source_update']}) after last ETL run ({dataset_info['last_etl_run']})"
        
        return {
            "needs_refresh": needs_refresh,
            "reason": reason
        }
    
    def _get_dataset_columns(self, dataset_id: str) -> List[str]:
        """Get the columns for a dataset from Socrata"""
        try:
            # Get a sample row to determine columns
            sample = self.client.get(dataset_id, limit=1)
            if sample and len(sample) > 0:
                return list(sample[0].keys())
            return []
        except Exception as e:
            logger.error(f"Error getting dataset columns: {e}")
            return []
    
    def get_delta_query_params(self, dataset_id: str, days_back: int = 7) -> Dict[str, Any]:
        """Get query parameters for delta loading based on dataset structure"""
        try:
            # Get dataset columns
            columns = self._get_dataset_columns(dataset_id)
            
            # Check for common update date columns
            update_date_columns = [
                'updated_at', 'last_updated', 'update_date', 'modified_date', 
                'last_modified', 'modified_at', 'update_timestamp'
            ]
            
            # Find a suitable update date column
            update_column = None
            for col in update_date_columns:
                if col in columns:
                    update_column = col
                    break
            
            if update_column:
                # Calculate the date threshold
                threshold_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                # Create a query parameter for filtering by update date
                return {
                    '$where': f"{update_column} > '{threshold_date}'"
                }
            
            # If no update column is found, check for created date columns
            created_date_columns = [
                'created_at', 'creation_date', 'created_date', 'date_created',
                'create_date', 'created'
            ]
            
            for col in created_date_columns:
                if col in columns:
                    # Calculate the date threshold
                    threshold_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
                    
                    # Create a query parameter for filtering by creation date
                    return {
                        '$where': f"{col} > '{threshold_date}'"
                    }
            
            # If no suitable column is found, return None to indicate full load is needed
            logger.info(f"No suitable date column found for delta loading of dataset {dataset_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting delta query parameters: {e}")
            return None

    def batch_process_datasets(self, dataset_ids: List[str], 
                              process_function, 
                              max_concurrency: int = 3) -> Dict[str, Any]:
        # Current implementation might not handle concurrent requests optimally
        # Consider using asyncio or ThreadPoolExecutor with proper resource management 