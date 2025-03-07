#!/usr/bin/env python3
"""
Load Parquet Files to Trino

This script loads data from parquet files in MinIO into Trino tables.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import pyarrow.parquet as pq
from trino.dbapi import connect
from minio import Minio
from logger_config import setup_logger
from env_config import get_trino_credentials, get_minio_credentials

# Set up logger
logger = setup_logger(__name__)

def load_parquet_to_trino(dataset_id, schema_name, table_name):
    """Load a parquet file from MinIO into a Trino table"""
    try:
        # Get MinIO credentials
        minio_creds = get_minio_credentials()
        
        # Override the MinIO endpoint to use localhost
        minio_endpoint = "localhost:9000"
        logger.info(f"Using MinIO endpoint: {minio_endpoint}")
        
        # Initialize MinIO client
        minio_client = Minio(
            minio_endpoint,
            access_key=minio_creds['access_key'],
            secret_key=minio_creds['secret_key'],
            secure=False
        )
        
        # Check if the parquet file exists in MinIO
        bucket_name = "nyc-etl"
        object_path = f"{schema_name}/{table_name}/{dataset_id}.parquet"
        
        try:
            minio_client.stat_object(bucket_name, object_path)
            logger.info(f"Found parquet file in MinIO: s3://{bucket_name}/{object_path}")
        except Exception as e:
            logger.error(f"Parquet file not found in MinIO: {e}")
            return False
        
        # Download the parquet file locally
        local_path = f"/tmp/{dataset_id}.parquet"
        minio_client.fget_object(bucket_name, object_path, local_path)
        logger.info(f"Downloaded parquet file to {local_path}")
        
        # Read the parquet file schema
        parquet_schema = pq.read_schema(local_path)
        logger.info(f"Parquet schema: {parquet_schema}")
        
        # Connect to Trino
        trino_creds = get_trino_credentials()
        # Override the Trino host to use localhost
        trino_host = "localhost"
        logger.info(f"Using Trino host: {trino_host}")
        
        conn = connect(
            host=trino_host,
            port=int(trino_creds['port']),
            user=trino_creds['user'],
            catalog=trino_creds['catalog']
        )
        cursor = conn.cursor()
        
        # Create schema if it doesn't exist
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS iceberg.{schema_name}")
        logger.info(f"Created schema iceberg.{schema_name}")
        
        # Drop the table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS iceberg.{schema_name}.{table_name}")
        logger.info(f"Dropped table iceberg.{schema_name}.{table_name}")
        
        # Create the table with the correct schema
        columns = []
        for field in parquet_schema:
            col_name = field.name
            # Map PyArrow type to Trino type
            if 'string' in str(field.type).lower():
                col_type = 'VARCHAR'
            elif 'int' in str(field.type).lower():
                col_type = 'INTEGER'
            elif 'double' in str(field.type).lower() or 'float' in str(field.type).lower():
                col_type = 'DOUBLE'
            elif 'timestamp' in str(field.type).lower():
                col_type = 'TIMESTAMP'
            elif 'date' in str(field.type).lower():
                col_type = 'DATE'
            elif 'bool' in str(field.type).lower():
                col_type = 'BOOLEAN'
            else:
                col_type = 'VARCHAR'
            
            columns.append(f"{col_name} {col_type}")
        
        create_table_sql = f"""
        CREATE TABLE iceberg.{schema_name}.{table_name} (
            {', '.join(columns)}
        )
        """
        
        cursor.execute(create_table_sql)
        logger.info(f"Created table iceberg.{schema_name}.{table_name}")
        
        # Read the parquet file in chunks and insert into Trino
        df = pd.read_parquet(local_path)
        total_rows = len(df)
        logger.info(f"Read {total_rows} rows from parquet file")
        
        # Convert all columns to strings to avoid type issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Insert data in chunks
        chunk_size = 1000
        for i in range(0, total_rows, chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            
            # Create VALUES clause for INSERT
            values_list = []
            for _, row in chunk.iterrows():
                # Use regular string formatting instead of f-string
                formatted_values = []
                for val in row:
                    # Replace single quotes with double single quotes for SQL
                    escaped_val = val.replace("'", "''")
                    formatted_values.append(f"'{escaped_val}'")
                values = ", ".join(formatted_values)
                values_list.append(f"({values})")
            
            insert_sql = f"""
            INSERT INTO iceberg.{schema_name}.{table_name}
            VALUES {', '.join(values_list)}
            """
            
            cursor.execute(insert_sql)
            logger.info(f"Inserted chunk {i//chunk_size + 1} ({len(chunk)} rows)")
        
        # Verify the data was inserted
        cursor.execute(f"SELECT COUNT(*) FROM iceberg.{schema_name}.{table_name}")
        count = cursor.fetchone()[0]
        logger.info(f"Total rows in table: {count}")
        
        # Clean up
        cursor.close()
        conn.close()
        os.remove(local_path)
        logger.info(f"Cleaned up temporary file {local_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading parquet to Trino: {e}")
        return False

def main():
    """Main function for loading parquet files to Trino"""
    parser = argparse.ArgumentParser(description="Load Parquet Files to Trino")
    
    parser.add_argument("--dataset-id", type=str, required=True,
                        help="Dataset ID (e.g., tg4x-b46p)")
    parser.add_argument("--schema-name", type=str, required=True,
                        help="Schema name in Trino (e.g., city_government)")
    parser.add_argument("--table-name", type=str, required=True,
                        help="Table name in Trino (e.g., film_permits)")
    
    args = parser.parse_args()
    
    # Load parquet file to Trino
    success = load_parquet_to_trino(args.dataset_id, args.schema_name, args.table_name)
    
    if success:
        logger.info(f"Successfully loaded {args.dataset_id} to Trino table {args.schema_name}.{args.table_name}")
        print(f"\nSuccessfully loaded {args.dataset_id} to Trino table {args.schema_name}.{args.table_name}")
        print("\nTo query the data in Trino, use the following SQL:")
        print(f"\nSELECT * FROM iceberg.{args.schema_name}.{args.table_name} LIMIT 10;")
        return 0
    else:
        logger.error(f"Failed to load {args.dataset_id} to Trino")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 