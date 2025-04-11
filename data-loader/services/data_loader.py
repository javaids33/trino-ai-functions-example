import logging
import os
import sys
from typing import Dict, List, Any
from services.socrates_client import SocratesClient
from services.trino_client import TrinoClient
import threading
import re
import duckdb
import tempfile
import shutil
import pandas as pd

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, socrates_client: SocratesClient, trino_client: TrinoClient):
        """Initialize DataLoader with Socrates and Trino clients"""
        self.socrates_client = socrates_client
        self.trino_client = trino_client
        logger.info("Initializing DataLoader with Socrates and Trino clients")
        self._lock = threading.Lock()
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        """Clean up temporary directory"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def get_catalog(self) -> List[Dict[str, Any]]:
        """Get the complete catalog of NYC Open Data datasets"""
        try:
            logger.info("Starting catalog retrieval")
            catalog = self.socrates_client.get_catalog()
            logger.info(f"Retrieved {len(catalog)} datasets from catalog")
            return catalog
        except Exception as e:
            logger.error(f"Error retrieving catalog: {str(e)}")
            raise

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name to be compatible with Trino"""
        # Replace special characters with underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Convert to lowercase
        return name.lower()

    def _get_table_name(self, dataset_id: str, metadata: Dict[str, Any]) -> str:
        """Generate a proper table name from dataset metadata"""
        try:
            # Get the department name from metadata
            dept = metadata.get('attribution', '').strip()
            if not dept:
                dept = 'NYC'
            
            # Clean department name
            dept = dept.replace('NYC ', '')  # Remove NYC prefix if exists
            dept = ''.join(word[0] for word in dept.split())  # Get initials
            dept = dept.upper()
            
            # Get the dataset name
            name = metadata.get('name', dataset_id).strip()
            
            # Clean dataset name
            name = name.replace(' ', '_')
            name = name.replace('-', '_')
            name = name.replace('(', '')
            name = name.replace(')', '')
            name = name.replace('.', '')
            name = name.replace(',', '')
            name = name.lower()
            
            # Combine department and name
            table_name = f"{dept}.{name}"
            logger.info(f"Generated table name: {table_name}")
            return table_name
            
        except Exception as e:
            logger.error(f"Error generating table name: {str(e)}")
            # Fallback to a simple table name
            return f"dataset_{dataset_id.replace('-', '_')}"

    def load_dataset(self, dataset_id: str) -> None:
        """Load a dataset from Socrates API into Trino"""
        try:
            logger.info(f"Fetching metadata for dataset {dataset_id}")
            metadata = self.socrates_client.get_dataset_metadata(dataset_id)
            
            # Generate table name
            table_name = self._get_table_name(dataset_id, metadata)
            logger.info(f"Generated table name: {table_name}")
            
            # Fetch all data
            logger.info(f"Fetching all data for dataset {dataset_id}")
            df = self.socrates_client.get_all_data(dataset_id)
            logger.info(f"Retrieved {len(df)} rows of data")
            
            # Save to temporary parquet file
            with tempfile.TemporaryDirectory() as temp_dir:
                parquet_path = os.path.join(temp_dir, f"{dataset_id}.parquet")
                logger.info(f"Saving data to parquet file: {parquet_path}")
                df.to_parquet(parquet_path, index=False)
                logger.info(f"Successfully wrote {len(df)} rows to parquet")
                
                # Load data into Trino
                logger.info(f"Loading data into Trino table {table_name}")
                self.trino_client.load_data(table_name, parquet_path)
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            raise

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        try:
            logger.info(f"Fetching table info for {table_name}")
            return self.trino_client.get_table_info(table_name)
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            raise

    def list_tables(self) -> List[str]:
        """List all tables in the schema"""
        try:
            logger.info("Listing all tables")
            return self.trino_client.list_tables()
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            raise 