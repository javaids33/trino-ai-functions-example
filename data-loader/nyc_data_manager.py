import os
import sys
import argparse
import datetime
from logger_config import setup_logger
from env_config import get_socrata_credentials, get_minio_credentials, get_trino_credentials
from popular_datasets import get_popular_datasets, load_popular_datasets
from dataset_report import generate_dataset_report
from socrata_loader import SocrataToTrinoETL
from delta_manager import DeltaManager
from cleanup_utility import cleanup_temp_files, list_unused_datasets, remove_dataset
from dataset_portability import export_datasets, import_datasets
from init_duckdb import init_duckdb_database

# Set up logger
logger = setup_logger(__name__)

def display_sql_examples(result, dataset_id):
    """Display example SQL queries for the loaded dataset"""
    schema_name = result.get('schema_name', 'general')
    table_name = result.get('table_name', 'unknown')
    
    print("\nTo query the imported data in Trino, use the following SQL:")
    print("\n-- List all available schemas")
    print("SHOW SCHEMAS FROM iceberg;")
    print("\n-- List all tables in a schema")
    print(f"SHOW TABLES FROM iceberg.{schema_name};")
    print("\n-- Query the metadata registry")
    print(f"SELECT * FROM iceberg.metadata.dataset_registry WHERE dataset_id = '{dataset_id}';")
    print("\n-- Query the dataset")
    print(f"SELECT * FROM iceberg.{schema_name}.{table_name} LIMIT 10;")

def main():
    parser = argparse.ArgumentParser(description="NYC Open Data Manager")
    
    # Command options
    parser.add_argument("command", choices=[
        "load-popular", "report", "list-popular", "load-dataset", "load-pool",
        "check-updates", "delta-load", "cleanup-temp", "list-unused", "remove-dataset",
        "export-datasets", "import-datasets", "init-duckdb"
    ], help="Command to run")
    
    # Options for load-popular
    parser.add_argument("--limit", type=int, default=20, help="Number of popular datasets to load")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent loads")
    
    # Options for load-dataset
    parser.add_argument("--dataset-id", help="Dataset ID to load")
    
    # Options for load-pool
    parser.add_argument("--pool-size", type=int, default=5, help="Number of datasets in the pool")
    
    # Options for delta-load
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back for changes")
    parser.add_argument("--force", action="store_true", help="Force reload even if no changes detected")
    
    # Options for list-unused
    parser.add_argument("--min-days", type=int, default=30, help="Minimum days since last update to consider unused")
    parser.add_argument("--min-rows", type=int, default=100, help="Minimum row count to consider for removal")
    
    # Options for remove-dataset
    parser.add_argument("--schema-name", help="Schema name for dataset removal")
    parser.add_argument("--table-name", help="Table name for dataset removal")
    
    # Options for export-datasets
    parser.add_argument("--output", type=str, help="Output path for export archive")
    parser.add_argument("--datasets", type=str, nargs="+", help="Dataset IDs to export")
    
    # Options for import-datasets
    parser.add_argument("--archive", type=str, help="Archive file to import")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing datasets when importing")
    
    # Options for init-duckdb
    parser.add_argument("--db-path", type=str, default="/data/duckdb/nyc_data.duckdb", 
                      help="Path to DuckDB database file")
    
    # Handle case where script is called with dataset_id as first argument (legacy support)
    if len(sys.argv) > 1 and sys.argv[1] not in [
        "load-popular", "report", "list-popular", "load-dataset", "load-pool",
        "check-updates", "delta-load", "cleanup-temp", "list-unused", "remove-dataset",
        "export-datasets", "import-datasets", "init-duckdb"
    ]:
        # Assume it's a dataset ID for backward compatibility
        dataset_id = sys.argv[1]
        logger.info(f"Legacy mode: Loading dataset {dataset_id}")
        sys.argv = [sys.argv[0], "load-dataset", "--dataset-id", dataset_id] + sys.argv[2:]
    
    args = parser.parse_args()
    
    # Execute the chosen command
    if args.command == "load-popular":
        logger.info(f"Loading {args.limit} popular datasets with concurrency {args.concurrency}")
        load_popular_datasets(limit=args.limit, concurrency=args.concurrency)
        
    elif args.command == "report":
        logger.info("Generating dataset report")
        generate_dataset_report()
        
    elif args.command == "list-popular":
        logger.info(f"Listing {args.limit} popular datasets")
        datasets = get_popular_datasets(limit=args.limit)
        
        if not datasets:
            print("\nNo datasets found. Please check your connection and try again.")
            return
            
        print("\n===== POPULAR NYC OPEN DATA DATASETS =====")
        print(f"Found {len(datasets)} popular datasets from NYC Open Data\n")
        
        for idx, dataset in enumerate(datasets):
            print(f"{idx+1}. {dataset['name']} ({dataset['id']})")
            print(f"   Category: {dataset['category']}")
            print(f"   Rows: {dataset['rows']:,}")
            print(f"   Views last month: {dataset['views_last_month']:,}")
            print(f"   Last updated: {dataset['updated_at']}")
            
            # Add a short description if available
            description = dataset.get('description', '')
            if description:
                # Truncate long descriptions
                if len(description) > 100:
                    description = description[:97] + "..."
                print(f"   Description: {description}")
                
            print()
            
    elif args.command == "load-dataset":
        if not args.dataset_id:
            logger.error("Dataset ID is required")
            return
            
        logger.info(f"Loading dataset {args.dataset_id}")
        
        # Get credentials from environment config
        socrata_creds = get_socrata_credentials()
        minio_creds = get_minio_credentials()
        
        etl = SocrataToTrinoETL(
            app_token=socrata_creds['app_token'],
            api_key_id=socrata_creds['api_key_id'],
            api_key_secret=socrata_creds['api_key_secret'],
            minio_endpoint=minio_creds['endpoint'],
            minio_access_key=minio_creds['access_key'],
            minio_secret_key=minio_creds['secret_key']
        )
        
        result = etl.create_trino_table_from_dataset(args.dataset_id)
        
        if result:
            logger.info(f"Successfully loaded dataset {args.dataset_id}")
            logger.info(f"Dataset details: {result}")
            display_sql_examples(result, args.dataset_id)
        else:
            logger.error(f"Failed to load dataset {args.dataset_id}")
            
    elif args.command == "load-pool":
        logger.info(f"Loading a pool of {args.pool_size} datasets")
        # Get the popular datasets and select a diverse pool
        all_datasets = get_popular_datasets(limit=50)  # Get more to choose from
        
        # Group by category
        categories = {}
        for dataset in all_datasets:
            category = dataset.get('category', 'Uncategorized')
            if category not in categories:
                categories[category] = []
            categories[category].append(dataset)
            
        # Select one from each category until we reach pool_size
        pool = []
        category_keys = list(categories.keys())
        idx = 0
        while len(pool) < args.pool_size and idx < len(category_keys):
            category = category_keys[idx]
            if categories[category]:
                pool.append(categories[category].pop(0))
            idx = (idx + 1) % len(category_keys)
            if all(not datasets for datasets in categories.values()):
                break
                
        # Load the pool
        logger.info(f"Selected a pool of {len(pool)} datasets across {len(set(d['category'] for d in pool))} categories")
        for idx, dataset in enumerate(pool):
            print(f"{idx+1}. {dataset['name']} ({dataset['id']}) - {dataset['category']}")
        
        # Get credentials from environment config
        socrata_creds = get_socrata_credentials()
        
        etl = SocrataToTrinoETL(
            app_token=socrata_creds['app_token'],
            api_key_id=socrata_creds['api_key_id'],
            api_key_secret=socrata_creds['api_key_secret']
        )
        
        for dataset in pool:
            logger.info(f"Loading dataset {dataset['id']} ({dataset['name']})")
            result = etl.create_trino_table_from_dataset(dataset['id'])
            if result:
                logger.info(f"Successfully loaded dataset {dataset['id']}")
            else:
                logger.error(f"Failed to load dataset {dataset['id']}")
                
    elif args.command == "check-updates":
        if not args.dataset_id:
            logger.error("Dataset ID is required")
            return
            
        logger.info(f"Checking for updates to dataset {args.dataset_id}")
        delta_manager = DeltaManager()
        status = delta_manager.get_dataset_status(args.dataset_id)
        
        print(f"\n===== UPDATE STATUS FOR DATASET {args.dataset_id} =====")
        print(f"Last ETL run: {status.get('last_etl_run')}")
        print(f"Last source update: {status.get('last_source_update')}")
        print(f"Current row count: {status.get('current_row_count')}")
        print(f"Source row count: {status.get('source_row_count')}")
        print(f"Needs refresh: {status.get('needs_refresh')}")
        print(f"Reason: {status.get('reason')}")
        
    elif args.command == "delta-load":
        if not args.dataset_id:
            logger.error("Dataset ID is required")
            return
            
        logger.info(f"Performing delta load for dataset {args.dataset_id}")
        
        # Initialize the delta manager
        delta_manager = DeltaManager()
        
        # Check if dataset needs refresh
        status = delta_manager.get_dataset_status(args.dataset_id)
        
        if status.get('needs_refresh') or args.force:
            logger.info(f"Dataset {args.dataset_id} needs refresh: {status.get('reason')}")
            
            # Get delta query parameters
            query_params = delta_manager.get_delta_query_params(args.dataset_id, args.days)
            
            if query_params:
                logger.info(f"Using delta query parameters: {query_params}")
                
                # Get credentials from environment config
                socrata_creds = get_socrata_credentials()
                
                # Initialize ETL and load with delta parameters
                etl = SocrataToTrinoETL(
                    app_token=socrata_creds['app_token'],
                    api_key_id=socrata_creds['api_key_id'],
                    api_key_secret=socrata_creds['api_key_secret']
                )
                
                result = etl.create_trino_table_from_dataset(args.dataset_id, query_params=query_params)
                
                if result:
                    logger.info(f"Successfully performed delta load for dataset {args.dataset_id}")
                    logger.info(f"Loaded {result.get('row_count')} rows")
                else:
                    logger.error(f"Failed to perform delta load for dataset {args.dataset_id}")
            else:
                logger.info("No delta parameters available, performing full load")
                
                # Get credentials from environment config
                socrata_creds = get_socrata_credentials()
                
                etl = SocrataToTrinoETL(
                    app_token=socrata_creds['app_token'],
                    api_key_id=socrata_creds['api_key_id'],
                    api_key_secret=socrata_creds['api_key_secret']
                )
                
                result = etl.create_trino_table_from_dataset(args.dataset_id)
                
                if result:
                    logger.info(f"Successfully performed full load for dataset {args.dataset_id}")
                else:
                    logger.error(f"Failed to perform full load for dataset {args.dataset_id}")
        else:
            logger.info(f"Dataset {args.dataset_id} does not need refresh: {status.get('reason')}")
            
    elif args.command == "cleanup-temp":
        logger.info("Cleaning up temporary files")
        count = cleanup_temp_files()
        logger.info(f"Cleaned up {count} temporary directories")
        
    elif args.command == "list-unused":
        logger.info(f"Listing datasets unused for at least {args.min_days} days with fewer than {args.min_rows} rows")
        unused_datasets = list_unused_datasets(min_days_unused=args.min_days, min_rows=args.min_rows)
        
        if not unused_datasets:
            print("\nNo unused datasets found.")
            return
            
        print(f"\n===== UNUSED DATASETS ({len(unused_datasets)}) =====")
        for idx, dataset in enumerate(unused_datasets):
            print(f"{idx+1}. {dataset['title']} ({dataset['dataset_id']})")
            print(f"   Schema/Table: {dataset['schema_name']}.{dataset['table_name']}")
            print(f"   Row count: {dataset['row_count']}")
            print(f"   Last ETL: {dataset['etl_timestamp']}")
            print(f"   Last updated: {dataset['last_updated']}")
            print()
            
    elif args.command == "remove-dataset":
        if not args.dataset_id or not args.schema_name or not args.table_name:
            logger.error("Dataset ID, schema name, and table name are all required")
            return
            
        logger.info(f"Removing dataset {args.dataset_id} ({args.schema_name}.{args.table_name})")
        
        # Confirm removal
        confirm = input(f"Are you sure you want to remove dataset {args.dataset_id}? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Dataset removal cancelled")
            return
            
        result = remove_dataset(args.dataset_id, args.schema_name, args.table_name)
        
        if result:
            logger.info(f"Successfully removed dataset {args.dataset_id}")
        else:
            logger.error(f"Failed to remove dataset {args.dataset_id}")

    elif args.command == "export-datasets":
        logger.info("Exporting datasets")
        export_datasets(args.output, args.datasets)
        
    elif args.command == "import-datasets":
        if not args.archive:
            parser.error("--archive is required for import-datasets command")
        logger.info(f"Importing datasets from {args.archive}")
        import_datasets(args.archive, args.overwrite)

    elif args.command == "init-duckdb":
        logger.info(f"Initializing DuckDB database at {args.db_path}")
        success = init_duckdb_database(args.db_path)
        if success:
            print("DuckDB database initialized successfully!")
        else:
            print("Failed to initialize DuckDB database. Check logs for details.")

if __name__ == "__main__":
    main() 