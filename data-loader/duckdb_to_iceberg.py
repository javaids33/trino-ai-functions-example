#!/usr/bin/env python3
"""
DuckDB to Iceberg Bridge

This script utilizes DuckDB to process datasets and store them in MinIO in a format 
that can be used by Trino as Iceberg tables.
"""

import os
import sys
import argparse
import logging
from logger_config import setup_logger
from init_duckdb import init_duckdb_database
from duckdb_create_iceberg import create_iceberg_tables

# Set up logger
logger = setup_logger(__name__)

def main():
    """Main function for DuckDB to Iceberg conversion"""
    parser = argparse.ArgumentParser(description="DuckDB to Iceberg Bridge")
    
    parser.add_argument("--db-path", type=str, default="/data/duckdb/nyc_data.duckdb", 
                        help="Path to the DuckDB database file")
    parser.add_argument("--cache-dir", type=str, default="data_cache", 
                        help="Path to the dataset cache directory")
    
    args = parser.parse_args()
    
    # Initialize DuckDB and create Iceberg tables in MinIO
    try:
        logger.info(f"Initializing DuckDB database at {args.db_path}")
        duckdb_success = init_duckdb_database(args.db_path, args.cache_dir)
        
        if duckdb_success:
            logger.info("Successfully initialized DuckDB database")
            
            # Create Iceberg tables
            logger.info("Creating Iceberg tables in MinIO")
            iceberg_success = create_iceberg_tables(args.db_path, args.cache_dir)
            
            if iceberg_success:
                logger.info("Successfully created Iceberg tables in MinIO")
                print("\nDuckDB to Iceberg conversion complete!")
                print("\nTo query the data in Trino, use the following SQL commands:")
                print("\n-- List all available schemas")
                print("SHOW SCHEMAS FROM iceberg;")
                print("\n-- List all tables in a schema")
                print("SHOW TABLES FROM iceberg.transportation;")
                print("\n-- Query a specific table")
                print("SELECT * FROM iceberg.transportation.tbl_2020_yellow_taxi_trip_data LIMIT 10;")
                return 0
            else:
                logger.error("Failed to create Iceberg tables in MinIO")
                return 1
        else:
            logger.error("Failed to initialize DuckDB database")
            return 1
            
    except Exception as e:
        logger.error(f"Error in DuckDB to Iceberg conversion: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 