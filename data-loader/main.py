import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from socrata_loader import SocrataToTrinoETL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'data_loader.log'))
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

def load_dataset(dataset_id):
    """Load a specific dataset from Socrata"""
    try:
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
            print(f"SELECT * FROM iceberg.nycdata.dataset_registry WHERE dataset_id = '{dataset_id}';")
            print("\n-- Query the dataset")
            print(f"SELECT * FROM iceberg.{schema_name}.{table_name} LIMIT 10;")
            
            return True
        else:
            logger.error(f"Failed to load dataset {dataset_id}")
            return False
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        return False

def discover_and_load_datasets(domain="data.cityofnewyork.us", limit=5):
    """Discover and load datasets from a Socrata domain"""
    try:
        logger.info(f"Starting dataset discovery from {domain} (limit: {limit})")
        
        # Get credentials from environment
        app_token = os.environ.get('SOCRATA_APP_TOKEN')
        api_key_id = os.environ.get('SOCRATA_API_KEY_ID')
        api_key_secret = os.environ.get('SOCRATA_API_KEY_SECRET')
        
        # Initialize ETL pipeline
        etl = SocrataToTrinoETL(
            app_token=app_token,
            api_key_id=api_key_id,
            api_key_secret=api_key_secret
        )
        
        # Discover datasets
        datasets = etl.discover_datasets(domain=domain, limit=limit)
        
        if not datasets:
            logger.error(f"No datasets found for domain {domain}")
            return False
            
        logger.info(f"Found {len(datasets)} datasets")
        
        # Process each dataset
        success_count = 0
        for i, dataset in enumerate(datasets):
            dataset_id = dataset.get('resource', {}).get('id')
            if not dataset_id:
                logger.warning(f"Dataset {i+1}/{len(datasets)} has no ID, skipping")
                continue
                
            dataset_name = dataset.get('resource', {}).get('name', 'Unknown')
            logger.info(f"Processing dataset {i+1}/{len(datasets)}: {dataset_id} - {dataset_name}")
            
            try:
                result = etl.create_trino_table_from_dataset(dataset_id)
                if result:
                    success_count += 1
                    logger.info(f"Successfully processed dataset {dataset_id}")
                else:
                    logger.error(f"Failed to process dataset {dataset_id}")
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {str(e)}")
        
        logger.info(f"ETL process completed. Successfully processed {success_count}/{len(datasets)} datasets")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error in discover_and_load_datasets: {str(e)}")
        return False

def main():
    """Main entry point for the data loader"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load datasets from Socrata to Trino')
    parser.add_argument('--dataset', type=str, help='Specific dataset ID to load')
    parser.add_argument('--domain', type=str, default='data.cityofnewyork.us', help='Socrata domain to discover datasets from')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of datasets to discover and load')
    
    args = parser.parse_args()
    
    logger.info("Starting Socrata to Trino ETL process")
    
    # If a specific dataset ID is provided, load it
    if args.dataset:
        success = load_dataset(args.dataset)
    else:
        # Otherwise, discover and load datasets
        success = discover_and_load_datasets(args.domain, args.limit)
    
    if success:
        logger.info("ETL process completed successfully")
        return 0
    else:
        logger.error("ETL process failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 