#!/bin/bash

# Script to load a specific NYC dataset from Socrata
# Usage: ./scripts/load_nyc_dataset.sh <dataset_id>
# Example: ./scripts/load_nyc_dataset.sh vx8i-nprf

# Set up logging
LOG_FILE="logs/load_nyc_dataset.log"
mkdir -p logs

echo "$(date) - Starting NYC dataset loading process" | tee -a $LOG_FILE

# Check if dataset ID is provided
if [ -z "$1" ]; then
    echo "ERROR: No dataset ID provided. Usage: ./scripts/load_nyc_dataset.sh <dataset_id>" | tee -a $LOG_FILE
    echo "Example: ./scripts/load_nyc_dataset.sh vx8i-nprf (NYC Civil Service List)" | tee -a $LOG_FILE
    exit 1
fi

DATASET_ID=$1

# Check if Docker is running
if ! docker-compose ps | grep -q "Up"; then
    echo "ERROR: No Docker containers are running. Please start Docker and the required services first." | tee -a $LOG_FILE
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

# Create a temporary Python script to load the dataset
TEMP_SCRIPT="data-loader/load_dataset.py"
echo "Creating temporary script to load dataset $DATASET_ID..." | tee -a $LOG_FILE

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
    """Load a specific dataset from Socrata"""
    try:
        logger.info("Starting Socrata to Trino ETL process for dataset $DATASET_ID")
        
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
        result = etl.create_trino_table_from_dataset("$DATASET_ID")
        
        if result:
            logger.info(f"Successfully loaded dataset $DATASET_ID")
            logger.info(f"Dataset details: {result}")
        else:
            logger.error(f"Failed to load dataset $DATASET_ID")
        
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
echo "Loading dataset $DATASET_ID..." | tee -a $LOG_FILE
docker-compose run --rm data-loader python load_dataset.py

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to load dataset $DATASET_ID. Check the logs for details." | tee -a $LOG_FILE
    exit 1
fi

# Clean up the temporary script
rm $TEMP_SCRIPT

echo "$(date) - Dataset loading process completed" | tee -a $LOG_FILE

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
echo "SELECT * FROM iceberg.nycdata.dataset_registry WHERE dataset_id = '$DATASET_ID';"
echo ""
echo "-- Get the schema and table name for the dataset"
echo "SELECT schema_name, table_name FROM iceberg.nycdata.dataset_registry WHERE dataset_id = '$DATASET_ID';"
echo ""

exit 0 