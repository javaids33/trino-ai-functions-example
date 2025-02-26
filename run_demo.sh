#!/bin/bash
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================================================"
echo "Checking for GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    nvidia-smi | tee logs/gpu_info.log
    export USE_GPU=true
    echo "Setting USE_GPU=true for GPU acceleration"
else
    echo "No NVIDIA GPU detected, running in CPU mode"
    export USE_GPU=auto
    echo "Setting USE_GPU=auto for automatic detection"
fi

echo "========================================================================"
echo "Checking for any existing services using our ports..."
# Check for netstat command differences between Mac and Windows/Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OS
    netstat -an | grep "LISTEN" | grep -E "8080|9000|9001|19120|11434|5001" || echo "No conflicting ports found"
else
    # Windows/Linux - try both commands
    if command -v findstr &> /dev/null; then
        netstat -ano | findstr "8080 9000 9001 19120 11434 5001" || echo "No conflicting ports found"
    else
        netstat -tulpn 2>/dev/null | grep -E "8080|9000|9001|19120|11434|5001" || echo "No conflicting ports found"
    fi
fi

echo "========================================================================"
echo "Stopping any existing containers and cleaning volumes..."
docker compose down --volumes || docker-compose down --volumes

echo "========================================================================"
echo "Rebuilding the trino-ai service..."
docker compose build trino-ai || docker-compose build trino-ai

echo "========================================================================"
echo "Starting core services: MinIO, Nessie, and Trino..."
docker compose up -d minio nessie trino || docker-compose up -d minio nessie trino

echo "Waiting for Trino to be healthy..."
# Cross-platform compatible timeout
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OS doesn't have timeout command
    end=$((SECONDS+60))
    while [ $SECONDS -lt $end ]; do
        if docker compose ps trino | grep -q "(healthy)" || docker-compose ps trino | grep -q "(healthy)"; then
            echo "Trino is healthy!"
            break
        fi
        echo "Waiting for Trino to be healthy... ($(($end-SECONDS))s remaining)"
        sleep 5
    done
else
    # Linux/Windows
    timeout 60 bash -c 'until docker compose ps trino | grep -q "(healthy)" || docker-compose ps trino | grep -q "(healthy)"; do echo "Waiting for Trino..."; sleep 5; done'
fi

echo "========================================================================"
echo "Starting data loader..."
docker compose up -d data-loader || docker-compose up -d data-loader

echo "Monitoring data loader logs in the background..."
(docker compose logs -f data-loader || docker-compose logs -f data-loader) &
DATA_LOADER_PID=$!

echo "========================================================================"
echo "Starting Ollama service..."
docker compose up -d ollama || docker-compose up -d ollama

echo "Monitoring Ollama logs (model download and setup) in the background..."
(docker compose logs -f ollama || docker-compose logs -f ollama) &
OLLAMA_PID=$!

echo "========================================================================"
echo "Starting Trino AI service..."
docker compose up -d trino-ai || docker-compose up -d trino-ai

echo "========================================================================"
echo "Checking service status..."
docker compose ps || docker-compose ps

echo "Waiting 10 seconds for Trino AI service to fully initialize..."
sleep 10

# Kill background processes to avoid log clutter
kill $DATA_LOADER_PID $OLLAMA_PID 2>/dev/null || true

echo -e "\n========================================================================"
echo "Displaying trino-ai logs..."
docker compose logs -f trino-ai || docker-compose logs -f trino-ai