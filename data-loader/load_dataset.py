import os
import sys
import logging
from dotenv import load_dotenv
from socrata_loader import SocrataToTrinoETL

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Load a specific dataset from Socrata"""
    try:
        # Check if dataset ID is provided
        if len(sys.argv) < 2:
            logger.error("No dataset ID provided. Usage: python load_dataset.py <dataset_id>")
            logger.error("Example: python load_dataset.py vx8i-nprf (NYC Civil Service List)")
            return False
            
        dataset_id = sys.argv[1]
        logger.info(f"Starting Socrata to Trino ETL process for dataset {dataset_id}")
        
        # Get credentials from environment
        app_token = os.environ.get('SOCRATA_APP_TOKEN')
        api_key_id = os.environ.get('SOCRATA_API_KEY_ID')
        api_key_secret = os.environ.get('SOCRATA_API_KEY_SECRET')
        
        if api_key_id and api_key_secret:
            logger.info("Using Socrata API authentication")
        else:
            logger.warning("No Socrata API credentials found - rate limits will apply")
        
        # Initialize ETL pipeline
        etl = SocrataToTrinoETL(
            app_token=app_token,
            api_key_id=api_key_id,
            api_key_secret=api_key_secret
        )
        
        # Process the specified dataset
        result = etl.create_trino_table_from_dataset(dataset_id)
        
        if result:
            logger.info(f"Successfully loaded dataset {dataset_id}")
            logger.info(f"Dataset details: {result}")
            
            # Print SQL queries for the user
            schema_name = result.get('schema_name', 'general')
            table_name = result.get('table_name', 'unknown')
            
            print("\nTo query the imported data in Trino, use the following SQL:")
            print("\n-- List all available schemas")
            print("SHOW SCHEMAS FROM iceberg;")
            print("\n-- List all tables in a schema")
            print(f"SHOW TABLES FROM iceberg.{schema_name};")
            print("\n-- Query the metadata registry")
            print(f"SELECT * FROM iceberg.metadata.dataset_registry WHERE dataset_id = '{dataset_id}';")
            print("\n-- Query the dataset")
            print(f"SELECT * FROM iceberg.{schema_name}.{table_name} LIMIT 10;")
            
            return True
        else:
            logger.error(f"Failed to load dataset {dataset_id}")
            return False
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 