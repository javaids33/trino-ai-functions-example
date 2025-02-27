#!/bin/bash
# monitor_agent.sh - Monitor agent thinking and responses in real-time

# Set up logging directory
LOGS_DIR="agent_monitoring"
mkdir -p $LOGS_DIR

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting agent monitoring at $(date)"
echo "Logs will be saved in the $LOGS_DIR directory"

# Function to monitor logs while running a query
function monitor_query() {
    local query="$1"
    local description="$2"
    
    echo "Running query: $description"
    echo "Query: $query"
    
    # Start monitoring logs in background
    docker logs -f trino-ai > "$LOGS_DIR/trino_ai_logs_$TIMESTAMP.log" &
    LOG_PID=$!
    
    # Execute the query
    echo "Executing query at $(date)"
    docker exec -it trino trino --execute="$query" > "$LOGS_DIR/query_result_$TIMESTAMP.log"
    
    # Give some time for logs to be captured
    sleep 5
    
    # Stop log monitoring
    kill $LOG_PID
    
    # Check the logs for agent thinking
    echo "Extracting agent thinking from logs..."
    grep -A 10 "TRINO-AI → OLLAMA" "$LOGS_DIR/trino_ai_logs_$TIMESTAMP.log" > "$LOGS_DIR/agent_thinking_$TIMESTAMP.log"
    
    # Check the logs for agent responses
    echo "Extracting agent responses from logs..."
    grep -A 10 "OLLAMA → TRINO-AI" "$LOGS_DIR/trino_ai_logs_$TIMESTAMP.log" > "$LOGS_DIR/agent_responses_$TIMESTAMP.log"
    
    echo "Query execution completed at $(date)"
    echo "Results saved to $LOGS_DIR/query_result_$TIMESTAMP.log"
    echo "Agent thinking saved to $LOGS_DIR/agent_thinking_$TIMESTAMP.log"
    echo "Agent responses saved to $LOGS_DIR/agent_responses_$TIMESTAMP.log"
    echo "-----------------------------------"
}

# Run different AI function queries and monitor agent thinking
echo "1. Monitoring AI text generation function..."
monitor_query "SELECT \"ai-functions\".ai.ai_gen('Write a short marketing description for a smartphone that costs \$999')" "AI Text Generation"

echo "2. Monitoring AI masking function..."
monitor_query "SELECT \"ai-functions\".ai.ai_mask('Customer John Doe with email john.doe@example.com and phone 555-123-4567', ARRAY['email', 'phone number'])" "AI Masking"

echo "3. Monitoring AI sentiment analysis function..."
monitor_query "SELECT \"ai-functions\".ai.ai_analyze_sentiment('I love this product, it is amazing!')" "AI Sentiment Analysis"

echo "4. Monitoring AI extraction function..."
monitor_query "SELECT \"ai-functions\".ai.ai_extract('Customer John Smith lives in New York and has loyalty tier Gold', ARRAY['city', 'loyalty level'])" "AI Extraction"

echo "5. Monitoring AI classification function..."
monitor_query "SELECT \"ai-functions\".ai.ai_classify('This smartphone has excellent battery life and a stunning display', ARRAY['battery', 'display', 'camera', 'performance'])" "AI Classification"

echo "All monitoring tasks completed at $(date)"

# Make the script executable
chmod +x monitor_agent.sh 