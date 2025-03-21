import os
import requests
import logging
from typing import List, Dict, Any, Optional
from logger_config import setup_logger
from env_config import get_socrata_credentials, DEFAULT_DOMAIN
from sodapy import Socrata

# Set up logger
logger = setup_logger(__name__)

class SocrataDiscovery:
    """Class to discover datasets from Socrata"""
    
    def __init__(self):
        # Get Socrata credentials
        socrata_creds = get_socrata_credentials()
        api_key_id = socrata_creds.get('key_id', '')
        api_key_secret = socrata_creds.get('key_secret', '')
        self.domain = socrata_creds.get('domain', DEFAULT_DOMAIN)
        
        # Initialize Socrata client
        if api_key_id and api_key_secret:
            # Pass empty string as the required app_token positional parameter
            self.client = Socrata(
                domain=self.domain,
                app_token="",  # Required positional parameter
                username=api_key_id,
                password=api_key_secret
            )
            logger.info("Initialized Socrata client with API key authentication")
        else:
            # Still need to pass the empty string for app_token
            self.client = Socrata(domain=self.domain, app_token="")
            logger.warning("Initialized Socrata client without authentication - rate limits will apply")
        
        # Setup headers for API calls
        self.headers = {}  # Empty headers, will rely on client authentication
    
    def search_datasets(self, query: str = None, domain: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for datasets in Socrata"""
        try:
            logger.info(f"Searching for datasets with query: {query}, domain: {domain}, limit: {limit}")
            
            # Build search parameters
            params = {}
            if query:
                params['q'] = query
            if domain:
                params['domains'] = domain
                
            # Search for datasets
            results = self.client.datasets(**params, limit=limit)
            
            # Format results
            datasets = []
            for result in results:
                dataset = {
                    'dataset_id': result.get('resource', {}).get('id'),
                    'name': result.get('resource', {}).get('name'),
                    'description': result.get('resource', {}).get('description'),
                    'domain': result.get('metadata', {}).get('domain'),
                    'category': result.get('classification', {}).get('domain_category'),
                    'row_count': result.get('resource', {}).get('rows_count'),
                    'view_count': result.get('resource', {}).get('view_count'),
                    'download_count': 0,  # Not typically available in API
                    'last_updated': result.get('resource', {}).get('updatedAt')
                }
                datasets.append(dataset)
                
            logger.info(f"Found {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"Error searching for datasets: {str(e)}")
            return []
            
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset"""
        try:
            logger.info(f"Getting metadata for dataset {dataset_id}")
            
            # Get dataset metadata
            metadata = self.client.get_metadata(dataset_id)
            
            # Format result
            result = {
                'dataset_id': dataset_id,
                'name': metadata.get('name'),
                'description': metadata.get('description'),
                'domain': metadata.get('domain'),
                'category': metadata.get('category'),
                'row_count': metadata.get('rowsCount'),
                'view_count': metadata.get('viewsCount'),
                'download_count': 0,  # Not typically available
                'last_updated': metadata.get('rowsUpdatedAt'),
                'columns': metadata.get('columns', [])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting dataset metadata: {str(e)}")
            return {'dataset_id': dataset_id, 'error': str(e)}

    def get_popular_datasets(self, limit: int = 20, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the most popular datasets from Socrata Open Data API
        
        Args:
            limit: Maximum number of datasets to return
            domain: Domain to search within (overrides instance domain if provided)
            
        Returns:
            List of dataset information dictionaries
        """
        try:
            # Use the provided domain or the instance domain
            search_domain = domain or self.domain
            logger.info(f"Finding {limit} most popular datasets from {search_domain}")
            
            # Use the Socrata Discovery API to find popular datasets
            url = "https://api.us.socrata.com/api/catalog/v1"
            params = {
                "domains": search_domain,
                "limit": 100,  # Get more than we need to filter
                "only": "dataset",
                "order": "page_views_last_month"  # Sort by popularity
            }
            
            logger.info(f"Making request to {url} with params: {params}")
            response = requests.get(url, headers=self.headers, params=params)
            logger.info(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            results = response.json().get('results', [])
            logger.info(f"Received {len(results)} results from API")
            
            # Filter for datasets that are valid
            valid_datasets = []
            for dataset in results:
                resource = dataset.get('resource', {})
                dataset_id = resource.get('id')
                
                # Only require that it's a dataset with an ID
                if dataset_id and resource.get('type') == 'dataset':
                    # Extract additional metadata
                    metadata = {
                        'dataset_id': dataset_id,
                        'name': resource.get('name', 'Unnamed Dataset'),
                        'description': resource.get('description', 'No description available'),
                        'domain': search_domain,
                        'category': dataset.get('classification', {}).get('domain_category', 'Uncategorized'),
                        'row_count': resource.get('rows_size', 0),
                        'view_count': resource.get('page_views_last_month', 0),
                        'download_count': resource.get('download_count', 0),
                        'last_updated': resource.get('updatedAt', 'Unknown')
                    }
                    valid_datasets.append(metadata)
                    
                if len(valid_datasets) >= limit:
                    break
                    
            logger.info(f"Found {len(valid_datasets)} valid datasets after filtering")
            return valid_datasets[:limit]
            
        except Exception as e:
            logger.error(f"Error getting popular datasets: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific dataset
        
        Args:
            dataset_id: The Socrata dataset ID
            
        Returns:
            Dictionary with dataset details
        """
        try:
            logger.info(f"Getting details for dataset {dataset_id}")
            
            # Use the Socrata Discovery API to get dataset details
            url = f"https://api.us.socrata.com/api/catalog/v1/datasets/{dataset_id}"
            
            logger.info(f"Making request to {url}")
            response = requests.get(url, headers=self.headers)
            logger.info(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            resource = result.get('resource', {})
            
            # Format the response
            details = {
                'dataset_id': dataset_id,
                'name': resource.get('name', 'Unnamed Dataset'),
                'description': resource.get('description', 'No description available'),
                'domain': resource.get('domain', self.domain),
                'category': result.get('classification', {}).get('domain_category', 'Uncategorized'),
                'row_count': resource.get('rows_size', 0),
                'view_count': resource.get('page_views_last_month', 0),
                'download_count': resource.get('download_count', 0),
                'last_updated': resource.get('updatedAt', 'Unknown'),
                'columns': resource.get('columns_field_name', []),
                'attribution': resource.get('attribution', 'Unknown'),
                'license': resource.get('license', 'Unknown'),
                'metadata': result
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting dataset details for {dataset_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {} 