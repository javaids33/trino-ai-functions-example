# socrata_discovery.py
import requests
import json
from typing import List, Dict, Any
from logger_config import setup_logger
from app_config import get_config

logger = setup_logger(__name__)
config = get_config()

def find_popular_datasets(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Find the most popular datasets from NYC Open Data
    
    Args:
        limit: Number of datasets to return
        
    Returns:
        List of dataset metadata dictionaries
    """
    logger.info(f"Finding {limit} most popular datasets from data.cityofnewyork.us")
    
    # Get Socrata credentials
    socrata_creds = config.get_socrata_credentials()
    app_token = socrata_creds.get('app_token', '')
    
    # API endpoint
    url = "https://api.us.socrata.com/api/catalog/v1"
    
    # Set up parameters to find popular datasets
    params = {
        'domains': 'data.cityofnewyork.us',
        'limit': 100,  # Request more to filter out non-datasets
        'only': 'dataset',
        'order': 'page_views_last_month'
    }
    
    # Set up headers
    headers = {}
    if app_token:
        headers['X-App-Token'] = app_token
    
    logger.info(f"Making request to {url} with params: {params}")
    
    try:
        # Make the request
        response = requests.get(url, params=params, headers=headers)
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Error response: {response.text}")
            return []
        
        # Parse the response
        results = response.json().get('results', [])
        logger.info(f"Received {len(results)} results from API")
        
        # Filter to ensure we only get actual datasets
        valid_datasets = []
        for item in results:
            # Skip if not a dataset
            if item.get('resource', {}).get('type') != 'dataset':
                continue
                
            # Extract relevant metadata
            metadata = item.get('resource', {})
            dataset_info = {
                'id': metadata.get('id', ''),
                'name': metadata.get('name', ''),
                'description': metadata.get('description', ''),
                'category': metadata.get('category', ''),
                'domain': metadata.get('domain', ''),
                'updated_at': metadata.get('updatedAt', ''),
                'created_at': metadata.get('createdAt', ''),
                'permalink': metadata.get('permalink', '')
            }
            
            # Only add if it has an ID
            if dataset_info['id']:
                valid_datasets.append(dataset_info)
                
            # If we have enough, stop
            if len(valid_datasets) >= limit:
                break
        
        logger.info(f"Found {len(valid_datasets)} valid datasets after filtering")
        return valid_datasets[:limit]
        
    except Exception as e:
        logger.error(f"Error finding popular datasets: {str(e)}")
        return []