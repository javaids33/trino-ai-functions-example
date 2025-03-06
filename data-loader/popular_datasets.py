import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger_config import setup_logger
from env_config import get_socrata_credentials
from socrata_loader import SocrataToTrinoETL

# Set up logger
logger = setup_logger(__name__)

def get_popular_datasets(domain=None, limit=20):
    """Get the most popular datasets from NYC Open Data"""
    try:
        # Get domain from environment config if not provided
        if domain is None:
            socrata_creds = get_socrata_credentials()
            domain = socrata_creds['domain']
            
        logger.info(f"Finding {limit} most popular datasets from {domain}")
        
        # Use the Socrata Discovery API to find popular datasets
        url = f"https://api.us.socrata.com/api/catalog/v1"
        params = {
            "domains": domain,
            "limit": 100,  # Get more than we need to filter
            "only": "dataset",
            "order": "page_views_last_month"  # Sort by popularity
        }
        
        logger.info(f"Making request to {url} with params: {params}")
        response = requests.get(url, params=params)
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
                    'id': dataset_id,
                    'name': resource.get('name', 'Unnamed Dataset'),
                    'category': dataset.get('classification', {}).get('domain_category', 'Uncategorized'),
                    'rows': resource.get('rows_size', 0),
                    'views_last_month': resource.get('page_views_last_month', 0),
                    'updated_at': resource.get('updatedAt', 'Unknown'),
                    'description': resource.get('description', 'No description available')
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

def load_popular_datasets(limit=20, concurrency=5):
    """Load the most popular datasets into Trino"""
    try:
        # Get the popular datasets
        datasets = get_popular_datasets(limit=limit)
        
        if not datasets:
            logger.error("No popular datasets found")
            return False
            
        logger.info(f"Found {len(datasets)} popular datasets")
        for idx, dataset in enumerate(datasets):
            logger.info(f"{idx+1}. {dataset['name']} ({dataset['id']}) - {dataset['rows']} rows, {dataset['views_last_month']} views last month")
        
        # Get credentials from environment config
        socrata_creds = get_socrata_credentials()
        
        # Initialize ETL pipeline
        etl = SocrataToTrinoETL(
            app_token=socrata_creds['app_token'],
            api_key_id=socrata_creds['api_key_id'],
            api_key_secret=socrata_creds['api_key_secret']
        )
        
        # Load datasets with concurrent processing
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks
            future_to_dataset = {
                executor.submit(etl.create_trino_table_from_dataset, dataset['id']): dataset
                for dataset in datasets
            }
            
            # Process as they complete
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Successfully loaded dataset {dataset['id']} ({dataset['name']})")
                        results.append({**dataset, 'success': True})
                    else:
                        logger.error(f"Failed to load dataset {dataset['id']} ({dataset['name']})")
                        results.append({**dataset, 'success': False})
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset['id']}: {str(e)}")
                    results.append({**dataset, 'success': False, 'error': str(e)})
        
        # Report results
        success_count = sum(1 for r in results if r.get('success', False))
        logger.info(f"Successfully loaded {success_count} of {len(datasets)} datasets")
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading popular datasets: {str(e)}")
        return []

if __name__ == "__main__":
    load_popular_datasets(limit=20, concurrency=5) 