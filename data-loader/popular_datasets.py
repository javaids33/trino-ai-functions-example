import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger_config import setup_logger
from env_config import get_socrata_credentials
from socrata_loader import SocrataToTrinoETL
from typing import List, Dict, Any, Optional
from socrata_discovery import SocrataDiscovery
import concurrent.futures

# Set up logger
logger = setup_logger(__name__)

def get_popular_datasets(limit: int = 20, domain: str = None, category: str = None) -> List[Dict[str, Any]]:
    """Get a list of popular datasets from Socrata"""
    try:
        logger.info(f"Getting popular datasets (limit: {limit}, domain: {domain}, category: {category})")
        
        # Initialize discovery class
        discovery = SocrataDiscovery()
        
        # Search for datasets
        datasets = discovery.search_datasets(
            query=None,
            domain=domain,
            limit=limit * 2  # Get more than we need so we can filter
        )
        
        # Sort by popularity (view_count)
        datasets.sort(key=lambda x: x.get('view_count', 0), reverse=True)
        
        # Filter by category if specified
        if category:
            datasets = [d for d in datasets if d.get('category') == category]
            
        # Return limited results
        return datasets[:limit]
        
    except Exception as e:
        logger.error(f"Error getting popular datasets: {str(e)}")
        return []

def load_popular_datasets(limit: int = 5, concurrency: int = 3) -> Dict[str, Any]:
    """Load popular datasets into Trino"""
    try:
        logger.info(f"Loading popular datasets (limit: {limit}, concurrency: {concurrency})")
        
        # Get popular datasets
        datasets = get_popular_datasets(limit=limit)
        
        if not datasets:
            return {
                "success": False,
                "datasets_loaded": 0,
                "datasets_attempted": 0,
                "details": [],
                "error": "No datasets found"
            }
            
        # Initialize ETL class
        etl = SocrataToTrinoETL()
        
        # Load datasets concurrently
        results = []
        successful_loads = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all load tasks
            future_to_dataset = {
                executor.submit(etl.load_dataset, dataset['dataset_id']): dataset
                for dataset in datasets[:limit]
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.get('success', False):
                        successful_loads += 1
                        logger.info(f"Successfully loaded dataset {dataset['dataset_id']}")
                    else:
                        logger.error(f"Failed to load dataset {dataset['dataset_id']}: {result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Error loading dataset {dataset['dataset_id']}: {str(e)}")
                    results.append({
                        "success": False,
                        "dataset_id": dataset['dataset_id'],
                        "error": str(e)
                    })
        
        return {
            "success": successful_loads > 0,
            "datasets_loaded": successful_loads,
            "datasets_attempted": len(datasets[:limit]),
            "details": results
        }
        
    except Exception as e:
        logger.error(f"Error loading popular datasets: {str(e)}")
        return {
            "success": False,
            "datasets_loaded": 0,
            "datasets_attempted": 0,
            "details": [],
            "error": str(e)
        }

if __name__ == "__main__":
    load_popular_datasets(limit=20, concurrency=5) 