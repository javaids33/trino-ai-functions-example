# NYC Open Data ETL with Socrata and Trino

This module provides a comprehensive ETL pipeline for loading data from NYC Open Data (powered by Socrata) into Trino.

## Features

1. **Dataset Discovery**:
   - Discovers datasets from NYC Open Data API
   - Supports filtering by category

2. **Metadata Extraction**:
   - Extracts comprehensive metadata for each dataset
   - Preserves descriptions, tags, column information

3. **Schema Organization**:
   - Maps dataset categories to Trino schemas
   - Creates dataset-specific tables with appropriate types

4. **Smart Partitioning**:
   - Analyzes column types to determine appropriate partitioning
   - Prioritizes categorical, geographical, and date columns

5. **Metadata Registry**:
   - Tracks all dataset information in a central registry
   - Records original and converted size statistics
   - Maintains lineage information for each table

6. **Batch Processing**:
   - Handles large datasets in manageable batches
   - Prevents memory issues with very large files

7. **Error Handling**:
   - Comprehensive error handling and logging
   - Continues processing if one dataset fails

## Usage

### Running the ETL Process

Use the provided script to run the ETL process:

```bash
./scripts/run_socrata_etl.sh
```

This script will:
1. Check if Trino and MinIO are running
2. Build the data-loader container
3. Run the ETL process
4. Follow the logs

### Setting Up API Credentials

To use your Socrata API credentials:

1. Copy the example environment file:
   ```bash
   cp data-loader/.env.example data-loader/.env
   ```

2. Edit the `.env` file with your credentials:
   ```
   SOCRATA_APP_TOKEN=your_app_token
   SOCRATA_API_KEY_ID=your_api_key_id
   SOCRATA_API_KEY_SECRET=your_api_key_secret
   ```

3. The `.env` file is excluded from Git, so your credentials will remain local and won't be committed to the repository.

### Customizing the ETL Process

To customize the datasets to be loaded, edit the `main()` function in `socrata_loader.py`:

```python
# Process specific datasets
dataset_ids = ["vx8i-nprf", "another-dataset-id"]
results = etl.process_multiple_datasets(dataset_ids=dataset_ids)

# Or discover datasets by category
categories = ["Transportation", "Environment", "Health"]
for category in categories:
    etl.process_multiple_datasets(category=category, limit=5)
```

### Using an App Token

For better performance and to avoid rate limiting, you can provide a Socrata App Token:

```bash
docker-compose run -e SOCRATA_APP_TOKEN=your_app_token data-loader
```

Or edit the Dockerfile to include your token:

```dockerfile
ENV SOCRATA_APP_TOKEN="your_app_token"
```

## Querying the Data

After the ETL process completes, you can query the data in Trino:

```sql
-- List all available schemas
SHOW SCHEMAS FROM iceberg;

-- List all tables in a schema
SHOW TABLES FROM iceberg.general;

-- Query the metadata registry
SELECT * FROM iceberg.metadata.dataset_registry;

-- Query a specific dataset
SELECT * FROM iceberg.general.civil_service_list LIMIT 10;
```

## Metadata Registry

The ETL process creates a metadata registry table that contains information about all loaded datasets:

```sql
SELECT 
    dataset_id, 
    dataset_title, 
    schema_name, 
    table_name, 
    row_count, 
    column_count, 
    partitioning_columns,
    last_updated, 
    etl_timestamp
FROM iceberg.metadata.dataset_registry;
```

## Troubleshooting

If you encounter issues:

1. Check the logs:
   ```bash
   docker-compose logs data-loader
   ```

2. Ensure Trino and MinIO are running:
   ```bash
   docker-compose ps
   ```

3. Check the Trino logs:
   ```bash
   docker-compose logs trino
   ```

4. Verify the MinIO bucket:
   ```bash
   docker-compose exec minio mc ls local/iceberg
   ``` 