import logging
import pandas as pd
from services.socrates_client import SocratesClient
from services.trino_client import TrinoClient
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize clients
        logger.info("Initializing clients...")
        socrates = SocratesClient()
        trino = TrinoClient()
        
        # Get dataset metadata
        logger.info("Fetching dataset metadata...")
        dataset_id = os.getenv('DATASET_ID')
        if not dataset_id:
            raise ValueError("DATASET_ID environment variable not set")
        
        # Download dataset
        logger.info(f"Downloading dataset {dataset_id}...")
        df = socrates.download_dataset(dataset_id, limit=1000)  # Adjust limit as needed
        logger.info(f"Downloaded {len(df)} rows")
        
        # Clean table name
        table_name = f"nyc_data_{dataset_id.lower().replace('-', '_')}"
        
        # Create table
        logger.info(f"Creating table {table_name}...")
        trino.create_table(table_name, df)
        
        # Load data
        logger.info(f"Loading data into {table_name}...")
        trino.load_data(table_name, df)
        
        # Verify data
        table_info = trino.get_table_info(table_name)
        logger.info(f"Table info: {table_info}")
        
        logger.info("Data loading completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data loading process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 