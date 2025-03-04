#!/bin/bash

# Script to apply the ai_translate UDF to Trino
# This script connects to Trino and executes the SQL script to create the UDF

# Set up logging
LOG_FILE="logs/apply_ai_translate_udf.log"
mkdir -p logs

echo "$(date) - Starting UDF application" | tee -a $LOG_FILE

# Check if Trino is running
echo "Checking if Trino is running..." | tee -a $LOG_FILE
if ! docker-compose ps trino | grep -q "(healthy)"; then
    echo "ERROR: Trino is not running or not healthy. Please start Trino first." | tee -a $LOG_FILE
    exit 1
fi

# Check if trino-ai is running
echo "Checking if trino-ai is running..." | tee -a $LOG_FILE
if ! docker-compose ps trino-ai | grep -q "Up"; then
    echo "ERROR: trino-ai is not running. Please start trino-ai first." | tee -a $LOG_FILE
    exit 1
fi

# Apply the UDF
echo "Applying the ai_translate UDF to Trino..." | tee -a $LOG_FILE
docker-compose exec trino trino --catalog="ai-functions" --schema=ai --file=/etc/trino/ai_translate_udf.sql

if [ $? -eq 0 ]; then
    echo "UDF applied successfully!" | tee -a $LOG_FILE
else
    echo "ERROR: Failed to apply UDF. Check the logs for details." | tee -a $LOG_FILE
    exit 1
fi

# Test the UDF
echo "Testing the ai_translate UDF..." | tee -a $LOG_FILE
docker-compose exec trino trino --catalog="ai-functions" --schema=ai --execute="SELECT ai_translate('Show me the top 5 customers', 'sql')"

if [ $? -eq 0 ]; then
    echo "UDF test successful!" | tee -a $LOG_FILE
else
    echo "ERROR: UDF test failed. Check the logs for details." | tee -a $LOG_FILE
    exit 1
fi

echo "$(date) - UDF application completed successfully" | tee -a $LOG_FILE

# Provide instructions for viewing the workflow
echo ""
echo "To view the agent workflow, visit: http://localhost:5001/workflow-viewer"
echo ""

exit 0 