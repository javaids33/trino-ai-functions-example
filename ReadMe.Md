# Trino AI Functions Example

A demonstration project showing how to integrate Large Language Models (LLM) with Trino ai functions for natural language querying of data. This project uses Ollama for LLM capabilities and integrates with Trino using Iceberg tables.

## Architecture

The project consists of several key services:

- **Trino**: SQL query engine
- **MinIO**: S3-compatible object storage for Iceberg tables
- **Nessie**: Version control for Iceberg tables
- **Ollama**: Local LLM service
- **Trino-AI**: Flask service for natural language to SQL conversion
- **Data-Loader**: Service to populate sample data

## Features

- Natural language to SQL conversion
- Schema-aware query generation
- Vector-based metadata search using ChromaDB
- Sample dataset with customers, products, and sales tables
- Automatic schema context retrieval
- Query result explanations

## Prerequisites

- Docker with GPU support (for Ollama)
- Windows 10 or later
- NVIDIA GPU (recommended)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd trino-ai-functions-example
```

2. Start the services:
```bash
docker-compose up -d
```

This will:
- Start Trino, MinIO, and Nessie
- Pull and run Ollama with the llama3.2 model
- Build and start the Trino-AI service
- Load sample data into Iceberg tables

## Usage

Send natural language queries to the Trino-AI service:

```bash
curl -X POST http://localhost:5001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me total sales by region"}'
```

Example response:
```json
{
  "sql": "SELECT region, SUM(net_amount) as total_sales FROM iceberg.iceberg.sales GROUP BY region",
  "results": [...],
  "explanation": "This query shows the total sales aggregated by region...",
  "context": "Schema context used for query generation..."
}
```

## Available Endpoints

- `/query`: Main endpoint for natural language queries
- `/health`: Service health check
- `/embed`: Generate embeddings for text
- `/refresh_metadata`: Manually trigger metadata refresh

## Sample Data

The project includes sample tables:

- `customers`: Customer information including region and loyalty tier
- `products`: Product catalog with categories and pricing
- `sales`: Transaction records with amounts and payment methods

## Environment Variables

Key configuration options (from docker-compose.yml):

```ini
TRINO_HOST=trino
TRINO_PORT=8080
TRINO_USER=admin
TRINO_CATALOG=iceberg
TRINO_SCHEMA=iceberg
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://ollama:11434
```

## Monitoring

Check service logs:
```bash
docker-compose logs trino-ai
```

## Development

The main components are:

- `app.py`: Flask application handling requests
- `ollama_client.py`: LLM integration for SQL generation
- `embeddings.py`: Schema metadata management
- `data_loader.py`: Sample data population

## Limitations

- Requires GPU for optimal LLM performance
- Limited to predefined schema context
- Sample data size is restricted for demonstration purposes

## Security Note

This is a demonstration project and includes default credentials. For production use:
- Change default passwords
- Implement proper authentication
- Use secure connection strings
- Follow security best practices
