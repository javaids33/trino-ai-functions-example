#!/bin/bash
# ai_functions.sh - Script to execute AI functions through Trino CLI

# Log file for debugging
LOG_FILE="ai_function_calls.log"
echo "Starting AI function calls at $(date)" > $LOG_FILE

# Function to call Trino CLI with proper parameters
function run_trino_query() {
    local query="$1"
    echo "Executing query: $query" | tee -a $LOG_FILE
    
    # Use docker exec to run Trino CLI with proper headers
    docker exec -it trino trino --execute="$query" | tee -a $LOG_FILE
    
    echo "Query completed at $(date)" | tee -a $LOG_FILE
    echo "-----------------------------------" | tee -a $LOG_FILE
}

# AI function examples
echo "1. Running AI text generation function..." | tee -a $LOG_FILE
run_trino_query "SELECT \"ai-functions\".ai.ai_gen('Write a short marketing description for a smartphone that costs \$999')"

echo "2. Running AI masking function correctly..." | tee -a $LOG_FILE
run_trino_query "SELECT \"ai-functions\".ai.ai_mask('Customer John Doe with email john.doe@example.com and phone 555-123-4567', ARRAY['email', 'phone number'])"

echo "3. Running AI sentiment analysis function..." | tee -a $LOG_FILE
run_trino_query "SELECT \"ai-functions\".ai.ai_analyze_sentiment('I love this product, it is amazing!')"

echo "4. Running AI extraction function..." | tee -a $LOG_FILE
run_trino_query "SELECT \"ai-functions\".ai.ai_extract('Customer John Smith lives in New York and has loyalty tier Gold', ARRAY['city', 'loyalty level'])"

echo "5. Running AI classification function..." | tee -a $LOG_FILE
run_trino_query "SELECT \"ai-functions\".ai.ai_classify('This smartphone has excellent battery life and a stunning display', ARRAY['battery', 'display', 'camera', 'performance'])"

echo "6. Running improved AI masking function with instructions..." | tee -a $LOG_FILE
run_trino_query "SELECT \"ai-functions\".ai.ai_mask('Customer Information:
Name: John Smith
Email: john.smith@example.com
Phone: (555) 123-4567
SSN: 123-45-6789
Address: 123 Main St, Anytown, USA', ARRAY['email', 'phone', 'ssn'])"

# Make the script executable
chmod +x ai_functions.sh 