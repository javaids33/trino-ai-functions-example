#!/usr/bin/env python3
"""
Register Iceberg Tables in Trino

This script registers Iceberg tables in Trino using the register_table procedure.
It connects to DuckDB to get the table information and then registers the tables in Trino.
"""

import os
import sys
import argparse
import logging
import duckdb
import json
from pathlib import Path
from trino.dbapi import connect
from minio import Minio
from logger_config import setup_logger
from env_config import get_minio_credentials, get_trino_credentials

# Set up logger
logger = setup_logger(__name__)

def register_iceberg_tables(db_path='/data/duckdb/nyc_data.duckdb', cache_dir='data_cache'):
    """Register Iceberg tables in Trino using the register_table procedure"""
    try:
        # Check if DuckDB file exists
        if not os.path.exists(db_path):
            logger.error(f"DuckDB database file not found at {db_path}")
            return False
        
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to DuckDB at {db_path}")
        
        # Get MinIO credentials
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
            logger.error("Iceberg bucket does not exist in MinIO")
            return False
        
        # Get Trino credentials
        trino_creds = get_trino_credentials()
        
        # Connect to Trino
        trino_conn = connect(
            host=trino_creds['host'],
            port=int(trino_creds['port']),
            user=trino_creds['user'],
            catalog=trino_creds['catalog']
        )
        cursor = trino_conn.cursor()
        
        # Check if register_table procedure is available
        try:
            cursor.execute("SHOW PROCEDURES IN iceberg.system")
            procedures = cursor.fetchall()
            register_table_available = any("register_table" in proc[0] for proc in procedures)
            
            if not register_table_available:
                logger.error("register_table procedure is not available in Trino")
                return False
                
            logger.info("register_table procedure is available in Trino")
        except Exception as e:
            logger.error(f"Error checking for register_table procedure: {e}")
            return False
        
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
                
                # Check if the data file exists in MinIO
                s3_path = f"iceberg/{iceberg_schema_name}/{iceberg_table_name}"
                data_path = f"{iceberg_schema_name}/{iceberg_table_name}/data.parquet"
                
                try:
                    minio_client.stat_object("iceberg", data_path)
                    logger.info(f"Found data file in MinIO: s3://iceberg/{data_path}")
                except Exception as e:
                    logger.warning(f"Data file not found in MinIO for {dataset_id}: {e}")
                    continue
                
                # Create schema if it doesn't exist
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS iceberg.{iceberg_schema_name}")
                logger.info(f"Created schema iceberg.{iceberg_schema_name}")
                
                # Register the table using the register_table procedure
                table_location = f"s3://iceberg/{iceberg_schema_name}/{iceberg_table_name}"
                
                register_sql = f"""
                CALL iceberg.system.register_table(
                    schema_name => '{iceberg_schema_name}',
                    table_name => '{iceberg_table_name}',
                    table_location => '{table_location}'
                )
                """
                
                cursor.execute(register_sql)
                logger.info(f"Registered table iceberg.{iceberg_schema_name}.{iceberg_table_name} from {table_location}")
                
            except Exception as e:
                logger.error(f"Error registering table for dataset {dataset_id}: {e}")
        
        # Close connections
        cursor.close()
        trino_conn.close()
        conn.close()
        
        logger.info("Successfully registered Iceberg tables in Trino")
        return True
        
    except Exception as e:
        logger.error(f"Error registering Iceberg tables: {e}")
        return False

def main():
    """Main function for registering Iceberg tables"""
    parser = argparse.ArgumentParser(description="Register Iceberg Tables in Trino")
    
    parser.add_argument("--db-path", type=str, default="/data/duckdb/nyc_data.duckdb", 
                        help="Path to the DuckDB database file")
    parser.add_argument("--cache-dir", type=str, default="data_cache", 
                        help="Path to the dataset cache directory")
    
    args = parser.parse_args()
    
    # Register Iceberg tables in Trino
    success = register_iceberg_tables(args.db_path, args.cache_dir)
    
    if success:
        logger.info("Successfully registered Iceberg tables in Trino")
        print("\nIceberg tables registration complete!")
        print("\nTo query the data in Trino, use the following SQL commands:")
        print("\n-- List all available schemas")
        print("SHOW SCHEMAS FROM iceberg;")
        print("\n-- List all tables in a schema")
        print("SHOW TABLES FROM iceberg.general;")
        print("\n-- Query a specific table")
        print("SELECT * FROM iceberg.general.<table_name> LIMIT 10;")
        return 0
    else:
        logger.error("Failed to register Iceberg tables in Trino")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 