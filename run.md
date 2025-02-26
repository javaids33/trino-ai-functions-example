# Trino AI Functions Setup Guide

This guide documents the steps to set up and run the Trino AI Functions environment.

## Environment Setup

### 1. Check Running Containers

First, check if any containers are already running:

```bash
docker ps
```

### 2. Stop Any Existing Containers

If needed, stop and remove any existing containers:

```bash
docker-compose down --volumes
```

### 3. Start Core Services

Start the essential services (MinIO, Nessie, and Trino):

```bash
docker-compose up -d minio nessie trino
```

### 4. Wait for Trino to be Healthy

```bash
# Wait for Trino to be healthy before proceeding
timeout 60 bash -c 'until docker-compose ps trino | grep -q "(healthy)"; do sleep 5; done'
```

### 5. Start Data Loader

```bash
docker-compose up -d data-loader
```

### 6. Start Ollama Service

```bash
docker-compose up -d ollama
```

### 7. Start Trino AI Service

```bash
docker-compose up -d trino-ai
```

### 8. Check Service Status

```bash
docker-compose ps
```

## Testing AI Functions

Once all services are running, you can test the AI functions:

```sql
SELECT "ai-functions".ai.ai_translate("find top customers we sold products to", "sql")
```

## Monitoring Logs

To monitor the Trino AI service logs:

```bash
docker-compose logs -f trino-ai
```

To monitor the Trino service logs:

```bash
docker-compose logs -f trino
```
