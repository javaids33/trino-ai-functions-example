#!/bin/bash
# view_agent_thinking.sh - Script to extract and view agent thinking from log files

# Default log directory
LOG_DIR="logs"

# Function to display usage information
function show_usage() {
    echo "Usage: $0 [options] [log_file]"
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -a, --all      Show all messages (thinking and responses)"
    echo "  -t, --thinking Show only agent thinking (TRINO-AI → OLLAMA)"
    echo "  -r, --response Show only agent responses (OLLAMA → TRINO-AI)"
    echo "  -l, --list     List available conversation log files"
    echo ""
    echo "If no log file is specified, the most recent log file will be used."
}

# Function to list available log files
function list_logs() {
    echo "Available conversation log files:"
    ls -lt $LOG_DIR/conversation-conv-*.log 2>/dev/null | head -10
}

# Function to extract and display agent thinking
function extract_thinking() {
    local log_file="$1"
    echo "Extracting agent thinking (TRINO-AI → OLLAMA) from $log_file..."
    grep -A 20 "TRINO-AI → OLLAMA" "$log_file" | less
}

# Function to extract and display agent responses
function extract_responses() {
    local log_file="$1"
    echo "Extracting agent responses (OLLAMA → TRINO-AI) from $log_file..."
    grep -A 20 "OLLAMA → TRINO-AI" "$log_file" | less
}

# Function to extract and display all messages
function extract_all() {
    local log_file="$1"
    echo "Extracting all agent messages from $log_file..."
    grep -A 20 -E "TRINO-AI → OLLAMA|OLLAMA → TRINO-AI" "$log_file" | less
}

# Parse command line arguments
SHOW_THINKING=false
SHOW_RESPONSES=false
SHOW_ALL=false
LIST_LOGS=false
LOG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -a|--all)
            SHOW_ALL=true
            shift
            ;;
        -t|--thinking)
            SHOW_THINKING=true
            shift
            ;;
        -r|--response)
            SHOW_RESPONSES=true
            shift
            ;;
        -l|--list)
            LIST_LOGS=true
            shift
            ;;
        *)
            LOG_FILE="$1"
            shift
            ;;
    esac
done

# List logs if requested
if $LIST_LOGS; then
    list_logs
    exit 0
fi

# If no log file specified, use the most recent one
if [ -z "$LOG_FILE" ]; then
    LOG_FILE=$(ls -t $LOG_DIR/conversation-conv-*.log 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo "Error: No log files found in $LOG_DIR"
        exit 1
    fi
    echo "Using most recent log file: $LOG_FILE"
fi

# Check if the log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file '$LOG_FILE' not found"
    exit 1
fi

# If no specific option is selected, show all by default
if ! $SHOW_THINKING && ! $SHOW_RESPONSES && ! $SHOW_ALL; then
    SHOW_ALL=true
fi

# Extract and display the requested information
if $SHOW_ALL; then
    extract_all "$LOG_FILE"
elif $SHOW_THINKING; then
    extract_thinking "$LOG_FILE"
elif $SHOW_RESPONSES; then
    extract_responses "$LOG_FILE"
fi

# Make the script executable
chmod +x view_agent_thinking.sh 