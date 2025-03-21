import os
import requests
import logging
from typing import List, Dict, Any, Optional
from logger_config import setup_logger
from env_config import get_socrata_credentials

# Set up logger
logger = setup_logger(__name__)

class SocrataDiscovery:
    """Client for discovering datasets from Socrata Open Data API"""
    
    def __init__(self, app_token=None, domain=None):
        """
        Initialize the Socrata Discovery client
        
        Args:
            app_token: Socrata app token for authentication
            domain: Domain to search within (e.g., data.cityofnewyork.us)
        """
        # Get credentials from environment if not provided
        if app_token is None or domain is None:
            creds = get_socrata_credentials()
            self.app_token = app_token or creds.get('app_token')
            self.domain = domain or creds.get('domain')
        else:
            self.app_token = app_token
            self.domain = domain
            
        logger.info(f"Initialized SocrataDiscovery for domain: {self.domain}")
        
        # Set up headers for API requests
        self.headers = {}
        if self.app_token:
            self.headers["X-App-Token"] = self.app_token
    
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
    
    def search_datasets(self, query: str, limit: int = 20, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for datasets in Socrata Open Data API
        
        Args:
            query: Search query string
            limit: Maximum number of datasets to return
            domain: Domain to search within (overrides instance domain if provided)
            
        Returns:
            List of dataset information dictionaries
        """
        try:
            # Use the provided domain or the instance domain
            search_domain = domain or self.domain
            logger.info(f"Searching for '{query}' in {search_domain} (limit: {limit})")
            
            # Use the Socrata Discovery API to search datasets
            url = "https://api.us.socrata.com/api/catalog/v1"
            params = {
                "domains": search_domain,
                "limit": 100,  # Get more than we need to filter
                "only": "dataset",
                "q": query
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
            logger.error(f"Error searching datasets: {str(e)}")
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