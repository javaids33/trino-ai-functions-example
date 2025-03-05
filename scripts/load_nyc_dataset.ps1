# PowerShell script to load a specific NYC dataset from Socrata
# Usage: .\scripts\load_nyc_dataset.ps1 <dataset_id>
# Example: .\scripts\load_nyc_dataset.ps1 vx8i-nprf

# Set up logging
$LOG_FILE = "logs\load_nyc_dataset.log"
New-Item -Path "logs" -ItemType Directory -Force | Out-Null

function Log-Message {
    param (
        [string]$Message
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Tee-Object -FilePath $LOG_FILE -Append
}

Log-Message "Starting NYC dataset loading process"

# Check if dataset ID is provided
if (-not $args[0]) {
    Log-Message "ERROR: No dataset ID provided. Usage: .\scripts\load_nyc_dataset.ps1 <dataset_id>"
    Log-Message "Example: .\scripts\load_nyc_dataset.ps1 vx8i-nprf (NYC Civil Service List)"
    exit 1
}

$DATASET_ID = $args[0]

# Check if Docker containers are running
Log-Message "Checking if Docker containers are running..."
$containersRunning = docker-compose ps | Select-String "Up"
if (-not $containersRunning) {
    Log-Message "ERROR: No Docker containers are running. Please start Docker and the required services first."
    exit 1
}

# Check if Trino is running
Log-Message "Checking if Trino is running..."
$trinoRunning = docker-compose ps trino | Select-String "(healthy)"
if (-not $trinoRunning) {
    Log-Message "ERROR: Trino is not running or not healthy. Please start Trino first."
    exit 1
}

# Check if MinIO is running
Log-Message "Checking if MinIO is running..."
$minioRunning = docker-compose ps minio | Select-String "Up"
if (-not $minioRunning) {
    Log-Message "ERROR: MinIO is not running. Please start MinIO first."
    exit 1
}

# Check if .env file exists
if (-not (Test-Path "data-loader\.env")) {
    Log-Message "WARNING: data-loader\.env file not found. Please create it with your Socrata API credentials."
    Log-Message "You can copy data-loader\.env.example to data-loader\.env and update it with your credentials."
    Log-Message "Continuing without authentication - rate limits will apply."
}

# Create a temporary Python script to load the dataset
$TEMP_SCRIPT = "data-loader\load_dataset.py"
Log-Message "Creating temporary script to load dataset $DATASET_ID..."

@"
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
"@ | Out-File -FilePath $TEMP_SCRIPT -Encoding utf8

# Build the data-loader container if needed
Log-Message "Building the data-loader container..."
docker-compose build data-loader

if ($LASTEXITCODE -ne 0) {
    Log-Message "ERROR: Failed to build the data-loader container. Check the logs for details."
    exit 1
}

# Run the data-loader container with the temporary script
Log-Message "Loading dataset $DATASET_ID..."
docker-compose run --rm data-loader python load_dataset.py

if ($LASTEXITCODE -ne 0) {
    Log-Message "ERROR: Failed to load dataset $DATASET_ID. Check the logs for details."
    exit 1
}

# Clean up the temporary script
Remove-Item $TEMP_SCRIPT

Log-Message "Dataset loading process completed"

# Provide instructions for querying the data
Write-Host ""
Write-Host "To query the imported data in Trino, use the following SQL:"
Write-Host ""
Write-Host "-- List all available schemas"
Write-Host "SHOW SCHEMAS FROM iceberg;"
Write-Host ""
Write-Host "-- List all tables in a schema"
Write-Host "SHOW TABLES FROM iceberg.general;"
Write-Host ""
Write-Host "-- Query the metadata registry"
Write-Host "SELECT * FROM iceberg.metadata.dataset_registry WHERE dataset_id = '$DATASET_ID';"
Write-Host ""
Write-Host "-- Get the schema and table name for the dataset"
Write-Host "SELECT schema_name, table_name FROM iceberg.metadata.dataset_registry WHERE dataset_id = '$DATASET_ID';"
Write-Host ""

exit 0 