# NYC Open Data ETL Service

This service provides an API to load NYC Open Data datasets into a Trino instance using the Socrates API. The service maintains a one-to-one copy of the datasets in Iceberg format with proper metadata and schema organization.

## Features

- Fetches complete catalog of NYC Open Data datasets
- Loads specific datasets into Trino with proper schema and metadata
- Organizes datasets by category into schemas
- Preserves all metadata including table and column descriptions
- Efficient data loading using Parquet format
- Comprehensive logging for monitoring and debugging

## API Endpoints

### GET /api/catalog
Returns the complete catalog of NYC Open Data datasets.

### POST /api/dataset/<dataset_id>
Loads a specific dataset into Trino. The dataset will be organized into a schema based on its category.

## Environment Variables

- `TRINO_HOST`: Hostname of the Trino server
- `TRINO_PORT`: Port of the Trino server
- `TRINO_USER`: Username for Trino authentication
- `TRINO_CATALOG`: Catalog name in Trino
- `TRINO_SCHEMA`: Default schema in Trino
- `PORT`: Port for the Flask application
- `MINIO_ENDPOINT`: MinIO server endpoint
- `AWS_ACCESS_KEY_ID`: MinIO access key
- `AWS_SECRET_ACCESS_KEY`: MinIO secret key

## Running the Service

The service is designed to run in a Docker container as part of the larger Trino ecosystem. To run it:

```bash
docker-compose up -d data-loader-api
```

## Data Organization

- Datasets are organized into schemas based on their categories
- Each table includes comprehensive metadata
- Data is stored in Parquet format for efficient querying
- All metadata is preserved for AI/LLM applications

## Logging

Logs are stored in the `/app/logs` directory with daily rotation. Each log entry includes:
- Timestamp
- Log level
- Module name
- Message
- Error details (if applicable) 