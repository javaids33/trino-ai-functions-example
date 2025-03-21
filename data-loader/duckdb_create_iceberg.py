#!/usr/bin/env python3
"""
Create Iceberg Tables with DuckDB

This script uses DuckDB's Iceberg extension to create proper Iceberg tables in MinIO.
"""

import os
import sys
import argparse
import logging
import duckdb
import json
from pathlib import Path
from logger_config import setup_logger
from env_config import get_minio_credentials

# Set up logger
logger = setup_logger(__name__)

def create_iceberg_tables(db_path='/data/duckdb/nyc_data.duckdb', cache_dir='data_cache'):
    """Create Iceberg tables using DuckDB's Iceberg extension"""
    try:
        # Check if DuckDB file exists
        if not os.path.exists(db_path):
            logger.error(f"DuckDB database file not found at {db_path}")
            return False
        
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to DuckDB at {db_path}")
        
        # Load Iceberg extension
        try:
            conn.execute("INSTALL iceberg")
            conn.execute("LOAD iceberg")
            logger.info("Loaded Iceberg extension in DuckDB")
        except Exception as e:
            logger.error(f"Error loading Iceberg extension: {e}")
            return False
        
        # Get MinIO credentials
        minio_creds = get_minio_credentials()
        
        # Configure S3 connection
        conn.execute(f"""
        SET s3_region='us-east-1';
        SET s3_endpoint='{minio_creds['endpoint']}';
        SET s3_access_key_id='{minio_creds['access_key']}';
        SET s3_secret_access_key='{minio_creds['secret_key']}';
        SET s3_url_style='path';
        SET s3_use_ssl='false';
        """)
        logger.info("Configured S3 connection in DuckDB")
        
        # Get list of datasets from cache directory
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            logger.error(f"Cache directory not found at {cache_dir}")
            return False
        
        # Find all metadata.json files in the cache directory
        metadata_files = list(cache_path.glob("**/metadata.json"))
        
        if not metadata_files:
            logger.warning("No metadata files found in cache directory")
            return False
        
        # Process each dataset
        for metadata_file in metadata_files:
            try:
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                dataset_id = metadata.get('dataset_id')
                if not dataset_id:
                    logger.warning(f"No dataset_id found in {metadata_file}")
                    continue
                
                # Determine schema and table name for Iceberg
                iceberg_schema_name = "general"
                if 'classification' in metadata and 'domain_category' in metadata['classification']:
                    category = metadata['classification']['domain_category']
                    iceberg_schema_name = category.lower().replace(' ', '_').replace('-', '_')
                
                iceberg_table_name = dataset_id.replace("-", "_")
                
                # Get the parquet file path
                parquet_file = metadata_file.parent / f"{dataset_id}.parquet"
                if not parquet_file.exists():
                    logger.warning(f"Parquet file not found at {parquet_file}")
                    continue
                
                # Create schema if it doesn't exist
                schema_name = f"nyc_data_{dataset_id.replace('-', '_')}"
                conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
                
                # Create a table from the Parquet file if it doesn't exist
                table_name = "data"
                try:
                    conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} AS 
                    SELECT * FROM read_parquet('{parquet_file}')
                    """)
                    
                    # Get row count
                    result = conn.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}").fetchone()
                    row_count = result[0] if result else 0
                    
                    logger.info(f"Imported {row_count} rows into {schema_name}.{table_name}")
                except Exception as e:
                    logger.error(f"Error creating table from Parquet file: {e}")
                    continue
                
                # Create Iceberg table
                iceberg_location = f"s3://iceberg/{iceberg_schema_name}/{iceberg_table_name}"
                
                try:
                    # Create Iceberg catalog with unique table locations enabled
                    conn.execute(f"""
                    CREATE OR REPLACE ICEBERG CATALOG iceberg_catalog
                    WITH (
                        location = 's3://iceberg',
                        compression = 'gzip',
                        iceberg.unique-table-location = 'true'
                    )
                    """)
                    
                    # Create Iceberg schema
                    conn.execute(f"""
                    CREATE SCHEMA IF NOT EXISTS iceberg_catalog.{iceberg_schema_name}
                    """)
                    
                    # Create Iceberg table
                    conn.execute(f"""
                    CREATE OR REPLACE TABLE iceberg_catalog.{iceberg_schema_name}.{iceberg_table_name} AS
                    SELECT * FROM {schema_name}.{table_name}
                    """)
                    
                    logger.info(f"Created Iceberg table iceberg_catalog.{iceberg_schema_name}.{iceberg_table_name}")
                    
                    # Export Iceberg metadata
                    conn.execute(f"""
                    COPY iceberg_catalog.{iceberg_schema_name}.{iceberg_table_name} TO '{iceberg_location}'
                    (FORMAT ICEBERG)
                    """)
                    
                    logger.info(f"Exported Iceberg metadata to {iceberg_location}")
                    
                except Exception as e:
                    logger.error(f"Error creating Iceberg table: {e}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {e}")
        
        # Close connection
        conn.close()
        
        logger.info("Successfully created Iceberg tables in MinIO")
        return True
        
    except Exception as e:
        logger.error(f"Error creating Iceberg tables: {e}")
        return False

def main():
    """Main function for creating Iceberg tables"""
    parser = argparse.ArgumentParser(description="Create Iceberg Tables with DuckDB")
    
    parser.add_argument("--db-path", type=str, default="/data/duckdb/nyc_data.duckdb", 
                        help="Path to the DuckDB database file")
    parser.add_argument("--cache-dir", type=str, default="data_cache", 
                        help="Path to the dataset cache directory")
    
    args = parser.parse_args()
    
    # Create Iceberg tables
    success = create_iceberg_tables(args.db_path, args.cache_dir)
    
    if success:
        logger.info("Successfully created Iceberg tables in MinIO")
        print("\nIceberg tables creation complete!")
        print("\nTo register tables in Trino, use the following SQL commands:")
        print("\n-- List all available schemas")
        print("SHOW SCHEMAS FROM iceberg;")
        print("\n-- Create a schema if it doesn't exist")
        print("CREATE SCHEMA IF NOT EXISTS iceberg.transportation;")
        print("\n-- Register a table")
        print("CALL iceberg.system.register_table(")
        print("  schema_name => 'transportation',")
        print("  table_name => 'yellow_taxi_2020',")
        print("  table_location => 's3://iceberg/transportation/tbl_2020_yellow_taxi_trip_data'")
        print(");")
        return 0
    else:
        logger.error("Failed to create Iceberg tables in MinIO")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 