# Check for any existing containers using our ports
echo "Checking for existing services..."
netstat -ano | findstr "8080 9000 9001 19120 11434 5001"

# Stop any existing containers
docker-compose down --volumes

# Start the core services first (MinIO, Nessie, and Trino)
echo "Starting core services..."
docker-compose up -d minio nessie trino

# Wait for Trino to be healthy before proceeding
echo "Waiting for Trino to be healthy..."
timeout 60 bash -c 'until docker-compose ps trino | grep -q "(healthy)"; do sleep 5; done'

# Start data loader
echo "Starting data loader..."
docker-compose up -d data-loader

# Follow data-loader logs to monitor progress
echo "Monitoring data loading progress..."
docker-compose logs -f data-loader &

# Start Ollama service
echo "Starting Ollama service..."
docker-compose up -d ollama

# Monitor Ollama logs for model download
echo "Monitoring Ollama model setup..."
docker-compose logs -f ollama &

# Once Ollama is ready, start the AI service
echo "Starting Trino AI service..."
docker-compose up -d trino-ai

# Display all service statuses
echo "Checking service status..."
docker-compose ps

# Monitor all logs
echo "Displaying all service logs..."
docker-compose logs -f