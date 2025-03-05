#!/bin/bash

# Script to run the full Socrata ETL process
# Usage: ./scripts/run_socrata_etl.sh [domain] [limit]
# Example: ./scripts/run_socrata_etl.sh data.cityofnewyork.us 10

# Set up logging
LOG_FILE="logs/socrata_etl.log"
mkdir -p logs

echo "$(date) - Starting Socrata ETL process" | tee -a $LOG_FILE

# Default values
DOMAIN=${1:-"data.cityofnewyork.us"}
LIMIT=${2:-5}

echo "Using domain: $DOMAIN" | tee -a $LOG_FILE
echo "Dataset limit: $LIMIT" | tee -a $LOG_FILE

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first." | tee -a $LOG_FILE
    exit 1
fi

# Check if Trino is running
echo "Checking if Trino is running..." | tee -a $LOG_FILE
if ! docker-compose ps trino | grep -q "(healthy)"; then
    echo "ERROR: Trino is not running or not healthy. Please start Trino first." | tee -a $LOG_FILE
    exit 1
fi

# Check if MinIO is running
echo "Checking if MinIO is running..." | tee -a $LOG_FILE
if ! docker-compose ps minio | grep -q "Up"; then
    echo "ERROR: MinIO is not running. Please start MinIO first." | tee -a $LOG_FILE
    exit 1
fi

# Check if .env file exists
if [ ! -f "data-loader/.env" ]; then
    echo "WARNING: data-loader/.env file not found. Please create it with your Socrata API credentials." | tee -a $LOG_FILE
    echo "You can copy data-loader/.env.example to data-loader/.env and update it with your credentials." | tee -a $LOG_FILE
    echo "Continuing without authentication - rate limits will apply." | tee -a $LOG_FILE
fi

# Create a temporary Python script to run the ETL process
TEMP_SCRIPT="data-loader/run_etl.py"
echo "Creating temporary script to run ETL process..." | tee -a $LOG_FILE

cat > $TEMP_SCRIPT << EOF
import os
import logging
from dotenv import load_dotenv
from socrata_loader import SocrataToTrinoETL

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the full Socrata ETL process"""
    try:
        logger.info("Starting Socrata to Trino ETL process")
        
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
        
        # Discover and process datasets
        domain = "$DOMAIN"
        limit = $LIMIT
        
        logger.info(f"Discovering datasets from {domain} (limit: {limit})")
        datasets = etl.discover_datasets(domain=domain, limit=limit)
        
        if not datasets:
            logger.error(f"No datasets found for domain {domain}")
            return
            
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
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
EOF

# Build the data-loader container if needed
echo "Building the data-loader container..." | tee -a $LOG_FILE
docker-compose build data-loader

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build the data-loader container. Check the logs for details." | tee -a $LOG_FILE
    exit 1
fi

# Run the data-loader container with the temporary script
echo "Running Socrata ETL process..." | tee -a $LOG_FILE
docker-compose run --rm data-loader python run_etl.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to run the ETL process. Check the logs for details." | tee -a $LOG_FILE
    exit 1
fi

# Clean up the temporary script
rm $TEMP_SCRIPT

echo "$(date) - ETL process completed" | tee -a $LOG_FILE

# Provide instructions for querying the data
echo ""
echo "To query the imported data in Trino, use the following SQL:"
echo ""
echo "-- List all available schemas"
echo "SHOW SCHEMAS FROM iceberg;"
echo ""
echo "-- List all tables in a schema"
echo "SHOW TABLES FROM iceberg.general;"
echo ""
echo "-- Query the metadata registry"
echo "SELECT * FROM iceberg.nycdata.dataset_registry;"
echo ""
echo "-- Get the schema and table name for datasets"
echo "SELECT dataset_id, schema_name, table_name FROM iceberg.nycdata.dataset_registry;"
echo ""

exit 0 