import requests
import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import io
import time
import json

logger = logging.getLogger(__name__)

class SocratesClient:
    def __init__(self):
        self.base_url = "https://data.cityofnewyork.us"
        self.api_key = os.getenv('SOCRATES_API_KEY')
        self.api_secret = os.getenv('SOCRATES_API_SECRET')
        self.session = requests.Session()
        self.session.headers.update({
            'X-App-Token': self.api_key,
            'Accept': 'application/json'
        })
        logger.info(f"Initialized Socrates client with API key: {self.api_key[:5]}...")
    
    def get_catalog(self) -> List[Dict[str, Any]]:
        """Get the complete catalog of NYC Open Data datasets"""
        try:
            logger.info("Fetching complete catalog from NYC Open Data API")
            response = self.session.get(f"{self.base_url}/api/catalog/v1")
            response.raise_for_status()
            
            results = response.json().get('results', [])
            logger.info(f"Successfully retrieved {len(results)} datasets from catalog")
            
            catalog = [{
                'id': result.get('resource', {}).get('id'),
                'name': result.get('resource', {}).get('name'),
                'description': result.get('resource', {}).get('description'),
                'category': result.get('classification', {}).get('domain_category'),
                'last_updated': result.get('resource', {}).get('updatedAt'),
                'row_count': result.get('resource', {}).get('rows_count')
            } for result in results if result.get('resource', {}).get('id')]
            
            logger.info(f"Processed {len(catalog)} valid datasets from catalog")
            return catalog
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching catalog: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a dataset"""
        try:
            url = f"{self.base_url}/api/views/{dataset_id}"
            logger.info(f"Fetching metadata from {url}")
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching metadata for dataset {dataset_id}: {str(e)}")
            raise
    
    def get_dataset_count(self, dataset_id: str) -> int:
        """Get total count of records in a dataset"""
        try:
            url = f"{self.base_url}/resource/{dataset_id}.json?$select=count(*)"
            logger.info(f"Fetching count from {url}")
            response = self.session.get(url)
            response.raise_for_status()
            return int(response.json()[0]['count'])
        except Exception as e:
            logger.error(f"Error fetching count for dataset {dataset_id}: {str(e)}")
            raise

    def get_dataset_data(self, dataset_id: str, limit: int = 10000, offset: int = 0) -> List[Dict[str, Any]]:
        """Get data from a dataset with pagination"""
        try:
            url = f"{self.base_url}/resource/{dataset_id}.json?$limit={limit}&$offset={offset}"
            logger.info(f"Fetching data from {url}")
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching data for dataset {dataset_id}: {str(e)}")
            raise

    def get_all_dataset_data(self, dataset_id: str) -> pd.DataFrame:
        """Get all data from a dataset"""
        try:
            total_count = self.get_dataset_count(dataset_id)
            logger.info(f"Total records to fetch: {total_count}")
            
            all_data = []
            batch_size = 10000
            for offset in range(0, total_count, batch_size):
                logger.info(f"Fetching batch {offset//batch_size + 1} of {(total_count + batch_size - 1)//batch_size}")
                batch_data = self.get_dataset_data(dataset_id, batch_size, offset)
                all_data.extend(batch_data)
                logger.info(f"Fetched {len(batch_data)} records in this batch")
            
            logger.info(f"Successfully fetched all {len(all_data)} records")
            return pd.DataFrame(all_data)
        except Exception as e:
            logger.error(f"Error fetching all data for dataset {dataset_id}: {str(e)}")
            raise
    
    def get_column_metadata(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get detailed column metadata including descriptions and data types"""
        try:
            logger.info(f"Fetching column metadata for dataset {dataset_id}")
            response = self.session.get(f"{self.base_url}/api/views/{dataset_id}/columns.json")
            response.raise_for_status()
            
            columns = response.json()
            column_metadata = [{
                'name': col.get('fieldName'),
                'description': col.get('description'),
                'type': col.get('dataTypeName'),
                'format': col.get('format')
            } for col in columns]
            
            logger.info(f"Successfully retrieved metadata for {len(column_metadata)} columns in dataset {dataset_id}")
            return column_metadata
        except Exception as e:
            logger.error(f"Error fetching column metadata for {dataset_id}: {str(e)}")
            raise

    def get_dataset_schema(self, dataset_id: str) -> List[Dict[str, str]]:
        """Get schema information for a dataset"""
        try:
            metadata = self.get_dataset_metadata(dataset_id)
            columns = metadata.get('columns', [])
            return [{'name': col.get('name', ''), 'type': col.get('dataTypeName', '')} for col in columns]
        except Exception as e:
            logger.error(f"Error fetching schema for dataset {dataset_id}: {str(e)}")
            raise 