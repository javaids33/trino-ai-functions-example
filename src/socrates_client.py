import requests
import logging
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SocratesClient:
    def __init__(self, base_url: str = "https://data.cityofnewyork.us"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'X-App-Token': os.getenv('SOCRATA_API_TOKEN', '')
        })
        logger.info("SocratesClient initialized with base URL: %s", base_url)

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a dataset using the public data API."""
        try:
            url = f"{self.base_url}/api/views/{dataset_id}"
            logger.info("Fetching metadata for dataset %s", dataset_id)
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error fetching dataset metadata: %s", str(e))
            raise

    def get_dataset_schema(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get schema information for a dataset using the public data API."""
        try:
            url = f"{self.base_url}/api/views/{dataset_id}/columns"
            logger.info("Fetching schema for dataset %s", dataset_id)
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error fetching dataset schema: %s", str(e))
            raise

    def fetch_data(
        self,
        dataset_id: str,
        limit: int = 1000,
        offset: int = 0,
        where: Optional[str] = None,
        order: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch data from a dataset using the public data API."""
        try:
            url = f"{self.base_url}/resource/{dataset_id}.json"
            params = {
                '$limit': limit,
                '$offset': offset
            }
            
            if where:
                params['$where'] = where
            if order:
                params['$order'] = order
                
            logger.info("Fetching data from dataset %s with params: %s", dataset_id, params)
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error("Error fetching data: %s", str(e))
            raise

    def fetch_all_data(
        self,
        dataset_id: str,
        batch_size: int = 1000,
        where: Optional[str] = None,
        order: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all data from a dataset in batches."""
        all_data = []
        offset = 0
        
        while True:
            try:
                logger.info("Fetching batch %d for dataset %s", offset // batch_size + 1, dataset_id)
                batch = self.fetch_data(
                    dataset_id=dataset_id,
                    limit=batch_size,
                    offset=offset,
                    where=where,
                    order=order
                )
                
                if not batch:
                    break
                    
                all_data.extend(batch)
                offset += len(batch)
                
                # Add a small delay between requests to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error("Error in batch %d: %s", offset // batch_size + 1, str(e))
                break
                
        logger.info("Fetched %d total records for dataset %s", len(all_data), dataset_id)
        return all_data 