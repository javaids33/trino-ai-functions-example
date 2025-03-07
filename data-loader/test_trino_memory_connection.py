#!/usr/bin/env python3

import trino
import os
import sys
import logging
from logger_config import setup_logger

logger = setup_logger(__name__)

def test_trino_memory_connection():
    """Test connection to Trino's memory connector and create sample data"""
    try:
        # Connect to Trino
        conn = trino.dbapi.connect(
            host="localhost",
            port=8080,
            user="trino",
            catalog="memory",
            schema="default"
        )
        
        cursor = conn.cursor()
        
        # Create a schema for NYC data
        logger.info("Creating schema for NYC data")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS memory.nyc_data")
        
        # Create a sample table with taxi data
        logger.info("Creating sample table with taxi data")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory.nyc_data.taxi (
            id INTEGER,
            type VARCHAR,
            pickup_datetime TIMESTAMP,
            dropoff_datetime TIMESTAMP,
            passenger_count INTEGER,
            trip_distance DOUBLE,
            fare_amount DOUBLE,
            tip_amount DOUBLE,
            total_amount DOUBLE
        )
        """)
        
        # Insert sample data
        logger.info("Inserting sample data")
        cursor.execute("""
        INSERT INTO memory.nyc_data.taxi VALUES
        (1, 'Yellow', TIMESTAMP '2020-01-01 12:00:00', TIMESTAMP '2020-01-01 12:30:00', 1, 2.5, 10.0, 2.0, 12.0),
        (2, 'Green', TIMESTAMP '2020-01-01 13:00:00', TIMESTAMP '2020-01-01 13:15:00', 2, 1.5, 7.0, 1.5, 8.5),
        (3, 'Yellow', TIMESTAMP '2020-01-01 14:00:00', TIMESTAMP '2020-01-01 14:45:00', 3, 3.0, 15.0, 3.0, 18.0)
        """)
        
        # Create a metadata view
        logger.info("Creating metadata view")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory.default.datasets (
            schema_name VARCHAR,
            table_name VARCHAR,
            total_rows INTEGER
        )
        """)
        
        cursor.execute("""
        INSERT INTO memory.default.datasets VALUES
        ('nyc_data', 'taxi', 3)
        """)
        
        # Test query to show schemas
        logger.info("Querying available schemas")
        cursor.execute("SHOW SCHEMAS FROM memory")
        schemas = cursor.fetchall()
        
        print("\nAvailable schemas in memory catalog:")
        for schema in schemas:
            print(f"  - {schema[0]}")
            
        # Test query for datasets view
        logger.info("Querying datasets metadata")
        cursor.execute("SELECT * FROM memory.default.datasets")
        datasets = cursor.fetchall()
        
        print("\nDatasets in memory:")
        for dataset in datasets:
            print(f"  - {dataset[0]}.{dataset[1]}: {dataset[2]} rows")
            
        # Test query for taxi data
        logger.info("Querying taxi data")
        cursor.execute("SELECT * FROM memory.nyc_data.taxi")
        taxi_data = cursor.fetchall()
        
        print("\nTaxi data:")
        for row in taxi_data:
            print(f"  - ID: {row[0]}, Type: {row[1]}, Fare: ${row[6]:.2f}, Total: ${row[8]:.2f}")
            
        # Close connection
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Trino connection: {e}")
        print(f"Failed to connect: {e}")
        return False

if __name__ == "__main__":
    print("Testing Trino connection to memory catalog...")
    success = test_trino_memory_connection()
    sys.exit(0 if success else 1) 