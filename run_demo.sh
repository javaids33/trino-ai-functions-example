#!/bin/bash
set -e

echo "========================================================================"
echo "Checking for any existing services using our ports..."
netstat -ano | findstr "8080 9000 9001 19120 11434 5001" | tee logs/ports_check.log

echo "========================================================================"
echo "Stopping any existing containers and cleaning volumes..."
docker-compose down --volumes | tee -a logs/docker_down.log

echo "========================================================================"
echo "Rebuilding the trino-ai service..."
docker-compose build trino-ai | tee -a logs/build_trino_ai.log

echo "========================================================================"
echo "Starting core services: MinIO, Nessie, and Trino..."
docker-compose up -d minio nessie trino | tee -a logs/start_core.log

echo "Waiting for Trino to be healthy..."
timeout 60 bash -c 'until docker-compose ps trino | grep -q "(healthy)"; do sleep 5; done'

echo "========================================================================"
echo "Starting data loader..."
docker-compose up -d data-loader | tee -a logs/start_data_loader.log

echo "Monitoring data loader logs in the background..."
docker-compose logs -f data-loader &

echo "========================================================================"
echo "Starting Ollama service..."
docker-compose up -d ollama | tee -a logs/start_ollama.log

echo "Monitoring Ollama logs (model download and setup) in the background..."
docker-compose logs -f ollama &

echo "========================================================================"
echo "Starting Trino AI service..."
docker-compose up -d trino-ai | tee -a logs/start_trino_ai.log

echo "========================================================================"
echo "Checking service status..."
docker-compose ps | tee -a logs/service_status.log

echo "Waiting 10 seconds for Trino AI service to fully initialize..."
sleep 10

# echo "========================================================================"
# echo "Testing the NL2SQL endpoint..."
# curl -X POST http://localhost:5001/utility/nl2sql \
#   -H "Content-Type: application/json" \
#   -d '{"query": "find top customers we sold products to"}' | tee -a logs/nl2sql_test.log

echo -e "\n========================================================================"
echo "Displaying trino-ai logs..."
docker-compose logs -f trino-ai