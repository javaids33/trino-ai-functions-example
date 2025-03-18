import requests
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class NYCOpenDataClient:
    """Client for interacting with NYC Open Data API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the NYC Open Data client"""
        self.base_url = "https://data.cityofnewyork.us"
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["X-App-Token"] = api_key
    
    def get_popular_datasets(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get popular datasets from NYC Open Data"""
        try:
            # Using the Socrata Discovery API to get datasets
            url = f"{self.base_url}/api/views"
            params = {
                "limit": limit,
                "order": "popularity"
            }
            logger.info(f"Fetching popular datasets from {url}")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            # Transform the response to match our expected format
            results = response.json()
            formatted_results = []
            
            for dataset in results:
                formatted_results.append({
                    "resource": {
                        "id": dataset.get("id"),
                        "name": dataset.get("name"),
                        "description": dataset.get("description"),
                        "updatedAt": dataset.get("viewLastModified"),
                        "estimated_row_count": dataset.get("rowsUpdatedAt", 0),
                        "columns_field_name": dataset.get("columns", []),
                        "page_views_last_month": dataset.get("popularity", 0)
                    },
                    "classification": {
                        "domain_category": dataset.get("category", "")
                    }
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error fetching popular datasets: {str(e)}")
            return []
    
    def search_datasets(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search datasets in NYC Open Data"""
        try:
            url = f"{self.base_url}/api/views"
            params = {
                "query": query,
                "limit": limit
            }
            logger.info(f"Searching datasets with query '{query}'")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            # Transform the response to match our expected format
            results = response.json()
            formatted_results = []
            
            for dataset in results:
                formatted_results.append({
                    "resource": {
                        "id": dataset.get("id"),
                        "name": dataset.get("name"),
                        "description": dataset.get("description"),
                        "updatedAt": dataset.get("viewLastModified"),
                        "estimated_row_count": dataset.get("rowsUpdatedAt", 0),
                        "columns_field_name": dataset.get("columns", []),
                        "page_views_last_month": dataset.get("popularity", 0)
                    },
                    "classification": {
                        "domain_category": dataset.get("category", "")
                    }
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching datasets: {str(e)}")
            return []
    
    def get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """Get details for a specific dataset"""
        try:
            url = f"{self.base_url}/api/views/{dataset_id}"
            logger.info(f"Fetching dataset details for {dataset_id}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Transform the response to match our expected format
            dataset = response.json()
            formatted_result = {
                "resource": {
                    "id": dataset.get("id"),
                    "name": dataset.get("name"),
                    "description": dataset.get("description"),
                    "updatedAt": dataset.get("viewLastModified"),
                    "estimated_row_count": dataset.get("rowsUpdatedAt", 0),
                    "columns_field_name": dataset.get("columns", []),
                    "page_views_last_month": dataset.get("popularity", 0)
                },
                "classification": {
                    "domain_category": dataset.get("category", "")
                }
            }
            
            return formatted_result
        except Exception as e:
            logger.error(f"Error fetching dataset details: {str(e)}")
            return {"error": str(e)}
    
    def get_dataset_columns(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get column information for a dataset"""
        try:
            details = self.get_dataset_details(dataset_id)
            columns = details.get("resource", {}).get("columns_field_name", [])
            
            # Format columns to include name, type, and description
            formatted_columns = []
            for column in columns:
                formatted_columns.append({
                    "name": column.get("fieldName", ""),
                    "type": column.get("dataTypeName", ""),
                    "description": column.get("description", "")
                })
            
            logger.info(f"Retrieved {len(formatted_columns)} columns for dataset {dataset_id}")
            return formatted_columns
        except Exception as e:
            logger.error(f"Error fetching dataset columns: {str(e)}")
            return [] 