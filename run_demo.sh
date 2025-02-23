#!/bin/bash

# Function to check if a port is in use
check_port() {
    lsof -i :"$1" >/dev/null 2>&1
    return $?
}

# Kill process on port if needed
kill_port() {
    lsof -ti :"$1" | xargs kill -9 2>/dev/null
}

# Check and kill if needed
echo "Checking ports..."
if check_port 11434; then
    echo "Port 11434 in use, killing process..."
    kill_port 11434
fi

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Pull the model if not present
if ! ollama list | grep -q "llama2"; then
    echo "Pulling llama2 model..."
    ollama pull llama2
fi

# Start the services
echo "Starting Docker services..."
docker-compose down
docker-compose up -d

# Wait for Trino to be ready
echo "Waiting for Trino to be ready..."
until curl -s http://localhost:8080/v1/info/state > /dev/null; do
    echo "Waiting for Trino..."
    sleep 5
done

# Run the test script
echo "Running AI functions test..."
python test_ai_functions.py 