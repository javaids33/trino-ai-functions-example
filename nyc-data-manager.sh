#!/bin/bash

# NYC Open Data Manager - Shell Script
# Usage: ./nyc-data-manager.sh [command] [options]

echo "NYC Open Data Manager"

# Set up logging
LOG_FILE="logs/nyc_data_manager.log"
mkdir -p logs

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first." | tee -a $LOG_FILE
    exit 1
fi

# Check if a command is provided
if [ -z "$1" ]; then
    echo "Usage: ./nyc-data-manager.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  list-popular   List popular datasets"
    echo "  load-popular   Load popular datasets"
    echo "  load-dataset   Load a specific dataset"
    echo "  load-pool      Load a diverse pool of datasets"
    echo "  report         Generate a report on all datasets"
    echo ""
    echo "Examples:"
    echo "  ./nyc-data-manager.sh list-popular --limit 10"
    echo "  ./nyc-data-manager.sh load-popular --limit 5 --concurrency 3"
    echo "  ./nyc-data-manager.sh load-dataset --dataset-id vx8i-nprf"
    echo "  ./nyc-data-manager.sh load-pool --pool-size 5"
    echo "  ./nyc-data-manager.sh report"
    exit 0
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

# Build the command string
CMD="python nyc_data_manager.py $@"

# Run the command in the data-loader container
echo "Running command: $CMD" | tee -a $LOG_FILE
docker-compose run --rm data-loader $CMD

echo ""
echo "Command completed."
exit 0 