#!/usr/bin/env python3

import trino
import os
import sys
import logging
from logger_config import setup_logger

logger = setup_logger(__name__)

def test_trino_duckdb_connection():
    """Test connection from Trino to DuckDB"""
    try:
        # Connect to Trino
        conn = trino.dbapi.connect(
            host="localhost",
            port=8080,
            user="trino",
            catalog="duckdb",
            schema="main"
        )
        
        cursor = conn.cursor()
        
        # Test query to show schemas
        logger.info("Querying available schemas in DuckDB catalog")
        cursor.execute("SHOW SCHEMAS")
        schemas = cursor.fetchall()
        
        print("\nAvailable schemas in DuckDB catalog:")
        for schema in schemas:
            print(f"  - {schema[0]}")
            
        # Test query for main.datasets view
        try:
            logger.info("Querying datasets metadata")
            cursor.execute("SELECT * FROM main.datasets")
            datasets = cursor.fetchall()
            
            if datasets:
                print("\nDatasets in DuckDB:")
                for dataset in datasets:
                    print(f"  - {dataset[0]}.{dataset[1]}: {dataset[2]} rows")
            else:
                print("\nNo datasets found in DuckDB")
                
        except Exception as e:
            logger.warning(f"Could not query datasets view: {e}")
            
        # Close connection
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Trino-DuckDB connection: {e}")
        print(f"Failed to connect: {e}")
        return False

if __name__ == "__main__":
    print("Testing Trino connection to DuckDB...")
    success = test_trino_duckdb_connection()
    sys.exit(0 if success else 1) 