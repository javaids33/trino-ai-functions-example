# NYC Open Data ETL for Trino AI

A comprehensive ETL tool for loading NYC Open Data into Trino with optimized metadata for Trino AI, featuring local caching, dataset portability, and memory-efficient processing.

## Features

1. **Smart Caching System**:
   - Locally caches datasets to minimize API calls
   - Only downloads datasets when they've been updated at the source
   - Portable dataset archives for moving between environments

2. **Memory-Efficient Processing**:
   - Uses DuckDB for processing large datasets with minimal memory footprint
   - Handles datasets of any size through chunked processing

3. **Optimized Metadata for Trino AI**:
   - Preserves rich metadata for AI-powered query understanding
   - Maintains column descriptions and dataset context
   - Tracks dataset lineage and update history

4. **Smart Partitioning**:
   - Analyzes column types to determine appropriate partitioning
   - Optimizes for query performance with Trino

5. **Dataset Management**:
   - Discover popular datasets from NYC Open Data
   - Export and import datasets between environments
   - Generate reports on dataset statistics

## Installation

### Prerequisites

- Python 3.8+
- Trino server
- MinIO or S3-compatible object storage
- Socrata API credentials (optional, for better performance)

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## Usage

### Loading Datasets

```bash
# Load a specific dataset (with automatic caching)
python nyc_data_manager.py load-dataset --dataset-id kxp8-n2sj

# Load popular datasets
python nyc_data_manager.py load-popular --limit 5 --concurrency 3

# Load a diverse pool of datasets
python nyc_data_manager.py load-pool --pool-size 5
```

### Dataset Portability

```bash
# Export datasets to a portable archive
python nyc_data_manager.py export-datasets --output nyc_data_export

# Export specific datasets
python nyc_data_manager.py export-datasets --datasets vx8i-nprf kxp8-n2sj

# Import datasets from an archive
python nyc_data_manager.py import-datasets --archive nyc_data_export_20240306_152530.zip

# Force overwrite of existing datasets
python nyc_data_manager.py import-datasets --archive nyc_data_export_20240306_152530.zip --overwrite
```

### Dataset Management

```bash
# Generate a report on loaded datasets
python nyc_data_manager.py report

# List popular datasets without loading them
python nyc_data_manager.py list-popular --limit 20

# Check for dataset updates
python nyc_data_manager.py check-updates

# Clean up temporary files
python nyc_data_manager.py cleanup-temp

# List unused datasets
python nyc_data_manager.py list-unused

# Remove a dataset
python nyc_data_manager.py remove-dataset --dataset-id kxp8-n2sj
```

## Local Dataset Cache

All datasets are automatically cached in the `data_cache` directory. The cache includes:

- Parquet files containing the actual data
- Metadata files with information about the dataset
- A registry file tracking all cached datasets

The cache manager automatically checks if a dataset needs to be updated based on:
- Last update time at the source
- Row count changes
- Manual override flags

## Querying the Data

After loading datasets, you can query them in Trino:

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

## Benefits for Trino AI

The optimized metadata and caching system provides several benefits for Trino AI:

1. **Rich Context**: Preserves dataset descriptions, column names, and relationships
2. **Consistent Data**: Ensures the same dataset version is used across environments
3. **Efficient Processing**: Minimizes memory usage and API calls
4. **Portable Datasets**: Easily move datasets between development and production

## Troubleshooting

If you encounter issues:

1. Check the logs in the `logs` directory
2. Ensure the cache directory (`data_cache`) is writable
3. For import failures, verify the archive is valid and contains the expected datasets
4. Use the `--overwrite` flag if you want to replace existing datasets during import

## Implementation Details

The system consists of these key components:

- `nyc_data_manager.py`: Main entry point with all commands
- `socrata_loader.py`: Core ETL functionality
- `cache_manager.py`: Manages the local dataset cache
- `dataset_portability.py`: Handles export and import operations
- `duckdb_processor.py`: Memory-efficient data processing 