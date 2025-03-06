# NYC Open Data Manager

A comprehensive tool for managing NYC Open Data datasets in Trino, focusing on finding popular datasets and generating reports on dataset statistics.

## Features

- **Popular Dataset Discovery**: Find the most viewed datasets from NYC Open Data
- **Diverse Dataset Pool**: Select datasets from different categories to ensure variety
- **Concurrent Loading**: Speed up data ingestion by loading multiple datasets in parallel
- **Comprehensive Reporting**: Generate detailed reports with visualizations on dataset sizes and other statistics
- **Simple Command-line Interface**: Easy-to-use commands for all dataset management tasks

## Installation

The NYC Open Data Manager is included in the data-loader container. No additional installation is required if you're already using the Trino AI environment.

## Prerequisites

- Docker and Docker Compose
- Running Trino and MinIO instances

## Usage

### Windows

```batch
nyc-data-manager.bat [command] [options]
```

### Unix/Linux/macOS

```bash
./nyc-data-manager.sh [command] [options]
```

### Available Commands

1. **List Popular Datasets**

   ```bash
   ./nyc-data-manager.sh list-popular --limit 20
   ```

   This command lists the most popular datasets from NYC Open Data, showing details like category, row count, and view count.

2. **Load Popular Datasets**

   ```bash
   ./nyc-data-manager.sh load-popular --limit 10 --concurrency 5
   ```

   This command loads the most popular datasets into Trino. You can specify how many datasets to load and how many to load concurrently.

3. **Load a Specific Dataset**

   ```bash
   ./nyc-data-manager.sh load-dataset --dataset-id vx8i-nprf
   ```

   This command loads a specific dataset by its ID.

4. **Load a Diverse Pool of Datasets**

   ```bash
   ./nyc-data-manager.sh load-pool --pool-size 5
   ```

   This command selects and loads a diverse pool of datasets from different categories.

5. **Generate a Dataset Report**

   ```bash
   ./nyc-data-manager.sh report
   ```

   This command generates a comprehensive report on all datasets in the registry, including visualizations.

## Report Features

The dataset report includes:

- Summary statistics (total datasets, rows, size)
- Top 10 largest datasets by row count
- Top 10 largest datasets by size
- Dataset distribution by category
- Visualizations:
  - Top datasets by row count
  - Dataset size by category
  - Original vs. Parquet size comparison

Reports are saved in the `reports` directory as CSV files and PNG images.

## Examples

### Example 1: Find and load the 5 most popular datasets

```bash
# First, list the popular datasets
./nyc-data-manager.sh list-popular --limit 5

# Then, load them into Trino
./nyc-data-manager.sh load-popular --limit 5 --concurrency 3
```

### Example 2: Generate a report after loading datasets

```bash
# Load a diverse pool of datasets
./nyc-data-manager.sh load-pool --pool-size 10

# Generate a report
./nyc-data-manager.sh report
```

### Example 3: Load a specific dataset and check its details

```bash
# Load the NYC Civil Service List dataset
./nyc-data-manager.sh load-dataset --dataset-id vx8i-nprf

# Generate a report to see its details
./nyc-data-manager.sh report
```

## Troubleshooting

If you encounter issues:

1. Ensure Trino and MinIO are running:
   ```bash
   docker-compose ps
   ```

2. Check the logs:
   ```bash
   docker-compose logs data-loader
   ```

3. Verify your Socrata API credentials in the `.env` file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 