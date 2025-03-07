#!/usr/bin/env python3
"""
Initialize DuckDB with sample data for Trino connector testing
"""

import os
import json
import re
import duckdb
import logging
from pathlib import Path
from logger_config import setup_logger
from cache_manager import DatasetCacheManager
from minio import Minio

# Setup logger
logger = setup_logger(__name__)

def init_duckdb_database(db_path='/data/duckdb/nyc_data.duckdb', cache_dir='data_cache'):
    """Initialize DuckDB database with sample data from cache"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to the DuckDB database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to DuckDB at {db_path}")
        
        # Initialize cache manager to get dataset info
        cache_manager = DatasetCacheManager(cache_dir=cache_dir)
        cached_datasets = cache_manager.get_all_cached_datasets()
        
        if not cached_datasets:
            logger.warning("No cached datasets found to import into DuckDB")
            return False
            
        # Get MinIO credentials
        from env_config import get_minio_credentials
        minio_creds = get_minio_credentials()
        
        # Initialize MinIO client
        minio_client = Minio(
            minio_creds['endpoint'],
            access_key=minio_creds['access_key'],
            secret_key=minio_creds['secret_key'],
            secure=False
        )
        
        # Ensure iceberg bucket exists
        if not minio_client.bucket_exists("iceberg"):
            minio_client.make_bucket("iceberg")
            logger.info("Created 'iceberg' bucket in MinIO")
        
        # Process each cached dataset
        dataset_registry = []
            
        # Import each cached dataset
        for dataset_info in cached_datasets:
            dataset_id = dataset_info.get("dataset_id")
            file_path = dataset_info.get("file_path")
            
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"Skipping dataset {dataset_id}: file not found at {file_path}")
                continue
                
            # Create a schema for the dataset if it doesn't exist
            schema_name = f"nyc_data_{dataset_id.replace('-', '_')}"
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            
            # Create a table from the Parquet file
            table_name = "data"
            logger.info(f"Importing {file_path} into {schema_name}.{table_name}")
            
            try:
                conn.execute(f"""
                CREATE OR REPLACE TABLE {schema_name}.{table_name} AS 
                SELECT * FROM read_parquet('{file_path}')
                """)
                
                # Get row count
                result = conn.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}").fetchone()
                row_count = result[0] if result else 0
                
                logger.info(f"Imported {row_count} rows into {schema_name}.{table_name}")
                
                # Determine schema and table name for Iceberg
                metadata_file = os.path.join(os.path.dirname(file_path), 'metadata.json')
                iceberg_schema_name = "general"
                iceberg_table_name = dataset_id.replace("-", "_")
                
                try:
                    # Load metadata if available
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        if 'classification' in metadata and 'domain_category' in metadata['classification']:
                            category = metadata['classification']['domain_category']
                            iceberg_schema_name = re.sub(r'[^a-zA-Z0-9_]', '_', category.lower())
                    
                    # Upload to MinIO with proper directory structure for Iceberg
                    s3_path = f"iceberg/{iceberg_schema_name}/{iceberg_table_name}"
                    
                    # Use DuckDB's COPY TO for proper Parquet formatting
                    tmp_parquet = f"/tmp/{dataset_id}.parquet"
                    conn.execute(f"""
                        COPY {schema_name}.{table_name} TO '{tmp_parquet}' (FORMAT PARQUET, CODEC 'SNAPPY')
                    """)
                    
                    # Upload to MinIO
                    minio_client.fput_object(
                        "iceberg", 
                        f"{iceberg_schema_name}/{iceberg_table_name}/data.parquet", 
                        tmp_parquet
                    )
                    
                    logger.info(f"Uploaded {dataset_id} to MinIO at s3://iceberg/{iceberg_schema_name}/{iceberg_table_name}")
                    
                    # Clean up temporary file
                    if os.path.exists(tmp_parquet):
                        os.remove(tmp_parquet)
                    
                    # Add to registry for Trino table creation
                    dataset_registry.append({
                        "dataset_id": dataset_id,
                        "schema_name": iceberg_schema_name,
                        "table_name": iceberg_table_name,
                        "s3_path": s3_path,
                        "row_count": row_count
                    })
                    
                    logger.info(f"Registered {row_count} rows for {iceberg_schema_name}.{iceberg_table_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_id} for Iceberg: {e}")
                
            except Exception as e:
                logger.error(f"Error importing dataset {dataset_id}: {e}")
                
        # Create metadata view
        conn.execute("""
        CREATE OR REPLACE VIEW main.datasets AS
        SELECT 
            schema_name, 
            table_name,
            SUM(row_count) as total_rows
        FROM (
            SELECT 
                schema_name, 
                table_name,
                COUNT(*) as row_count
            FROM 
                information_schema.tables t
            JOIN 
                information_schema.schemata s ON t.table_schema = s.schema_name
            CROSS JOIN 
                (SELECT * FROM t LIMIT 1) r
            WHERE 
                t.table_schema LIKE 'nyc_data_%'
                AND t.table_type = 'BASE TABLE'
            GROUP BY 
                schema_name, table_name
        ) sub
        GROUP BY 
            schema_name, table_name;
        """)
        
        logger.info("Created metadata view main.datasets")
        
        # Instead of directly creating tables in Trino, we'll use the register_iceberg_tables.py script
        logger.info("Tables have been prepared in MinIO. Use register_iceberg_tables.py to register them in Trino.")
        
        return True
            
    except Exception as e:
        logger.error(f"Error initializing DuckDB database: {e}")
        return False

def main():
    """Main function for initializing DuckDB database"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize DuckDB with sample data")
    
    parser.add_argument("--db-path", type=str, default="/data/duckdb/nyc_data.duckdb", 
                        help="Path to the DuckDB database file")
    parser.add_argument("--cache-dir", type=str, default="data_cache", 
                        help="Path to the dataset cache directory")
    
    args = parser.parse_args()
    
    # Initialize DuckDB database
    success = init_duckdb_database(args.db_path, args.cache_dir)
    
    if success:
        print("\nDuckDB initialization complete!")
        print("\nTo register tables in Trino, run:")
        print("python register_iceberg_tables.py")
        return 0
    else:
        print("\nFailed to initialize DuckDB database")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 