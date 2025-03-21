import pyarrow.parquet as pq
from trino.dbapi import connect
import os
from datetime import date, datetime
import decimal
import logging
import re
from minio import Minio
import time
import requests
from requests.exceptions import RequestException
from logger_config import setup_logger

# Replace manual logging configuration with the standard setup_logger
logger = setup_logger(__name__)

def wait_for_trino_ready(max_retries=30, delay=10):
    """Wait for Trino to be ready by checking its health endpoint"""
    logger.info("Waiting for Trino to be ready...")
    
    for attempt in range(max_retries):
        try:
            # Try to connect to Trino's health endpoint
            response = requests.get('http://trino:8080/v1/info/state')
            if response.status_code == 200 and response.text.strip('"') == "ACTIVE":
                logger.info("Trino is ready!")
                
                # Additional wait to ensure full initialization
                logger.info("Waiting additional 30 seconds for full initialization...")
                time.sleep(30)
                return True
                
            logger.debug(f"Trino not ready yet (Attempt {attempt + 1}/{max_retries}). Status: {response.text}")
        except RequestException as e:
            logger.debug(f"Connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
        
        logger.info(f"Waiting {delay} seconds before next attempt...")
        time.sleep(delay)
    
    raise Exception("Trino failed to become ready within the timeout period")

def reset_minio_storage():
    """Reset MinIO storage by removing and recreating the iceberg bucket"""
    try:
        logger.info("Resetting MinIO storage...")
        
        # Initialize MinIO client
        minio_client = Minio(
            "minio:9000",
            access_key="admin",
            secret_key="password",
            secure=False
        )
        
        bucket_name = "iceberg"
        
        # Check if bucket exists
        if minio_client.bucket_exists(bucket_name):
            logger.info(f"Removing existing bucket: {bucket_name}")
            
            # List all objects in the bucket
            objects = minio_client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                logger.debug(f"Removing object: {obj.object_name}")
                minio_client.remove_object(bucket_name, obj.object_name)
            
            # Remove the bucket itself
            minio_client.remove_bucket(bucket_name)
            logger.info(f"Successfully removed bucket: {bucket_name}")
        
        # Create a fresh bucket
        logger.info(f"Creating fresh MinIO bucket: {bucket_name}")
        minio_client.make_bucket(bucket_name)
        logger.info(f"Successfully created bucket: {bucket_name}")
            
    except Exception as e:
        logger.error(f"Error resetting MinIO storage: {str(e)}")
        raise

def ensure_minio_bucket():
    """Ensure MinIO bucket exists"""
    try:
        # Initialize MinIO client
        minio_client = Minio(
            "minio:9000",
            access_key="admin",
            secret_key="password",
            secure=False
        )
        
        bucket_name = "iceberg"
        
        # Check if bucket exists
        if not minio_client.bucket_exists(bucket_name):
            logger.info(f"Creating MinIO bucket: {bucket_name}")
            minio_client.make_bucket(bucket_name)
            logger.info(f"Successfully created bucket: {bucket_name}")
        else:
            logger.info(f"Bucket {bucket_name} already exists")
            
        return minio_client, bucket_name
            
    except Exception as e:
        logger.error(f"Error ensuring MinIO bucket: {str(e)}")
        raise

def clean_minio_bucket():
    """Clean up the MinIO bucket by removing all objects"""
    try:
        logger.info("Cleaning up MinIO bucket...")
        
        # Initialize MinIO client
        minio_client, bucket_name = ensure_minio_bucket()
        
        # List and delete all objects in the bucket
        objects = minio_client.list_objects(bucket_name, recursive=True)
        object_count = 0
        
        for obj in objects:
            minio_client.remove_object(bucket_name, obj.object_name)
            object_count += 1
            if object_count % 100 == 0:
                logger.info(f"Deleted {object_count} objects from MinIO bucket")
        
        logger.info(f"Successfully cleaned up MinIO bucket. Deleted {object_count} objects.")
        
    except Exception as e:
        logger.error(f"Error cleaning MinIO bucket: {str(e)}")
        raise

def drop_tables(tables):
    """Drop existing tables to ensure clean state"""
    try:
        logger.info("Dropping existing tables...")
        
        conn = connect(
            host="trino",
            port=8080,
            user="admin",
            catalog="iceberg",
            schema="iceberg"
        )
        cursor = conn.cursor()
        
        for table_name in tables:
            try:
                logger.info(f"Dropping table {table_name}...")
                cursor.execute(f"DROP TABLE IF EXISTS iceberg.iceberg.{table_name}")
                logger.info(f"Table {table_name} dropped successfully")
            except Exception as e:
                logger.warning(f"Error dropping table {table_name}: {str(e)}")
        
        cursor.close()
        conn.close()
        logger.info("Finished dropping tables")
        
    except Exception as e:
        logger.error(f"Error dropping tables: {str(e)}")
        raise

def convert_value(val, col_name=None):
    """Convert values to appropriate Python types for Trino"""
    try:
        if val is None:
            return 'NULL'
        if isinstance(val, (date, datetime)):
            # Format date as 'YYYY-MM-DD' without time component
            return f"DATE '{val.strftime('%Y-%m-%d')}'"
        if isinstance(val, bool):
            return str(val).lower()
        if isinstance(val, int):
            return str(val)
        if isinstance(val, (float, decimal.Decimal)):
            # Format with exactly 2 decimal places for price/amount columns
            if col_name and any(term in col_name.lower() for term in ['price', 'amount', 'cost', 'discount']):
                return f"DECIMAL '{val:.2f}'"
            return str(val)
        # Escape single quotes in string values
        if isinstance(val, str):
            escaped_val = val.replace("'", "''")
            return f"'{escaped_val}'"
        return f"'{str(val)}'"
    except Exception as e:
        logger.error(f"Error converting value {val} for column {col_name}: {str(e)}")
        raise

def split_sql_statements(sql_content):
    """Split SQL content into individual statements, handling parentheses and semicolons properly"""
    statements = []
    current_statement = []
    paren_count = 0
    in_string = False
    string_char = None
    
    for line in sql_content.split('\n'):
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('--'):
            continue
            
        # Process each character to handle strings and parentheses
        for i, char in enumerate(line):
            if char in ["'", '"'] and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif string_char == char:
                    in_string = False
                    string_char = None
            
            if not in_string:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == ';' and paren_count == 0:
                    current_statement.append(line[:i+1])
                    statements.append('\n'.join(current_statement))
                    current_statement = []
                    break
        else:
            current_statement.append(line)
    
    if current_statement:
        statements.append('\n'.join(current_statement))
    
    return [stmt.strip() for stmt in statements if stmt.strip()]

def load_parquet_to_trino(file_path, table_name):
    """Load a parquet file into a Trino table"""
    logger.info(f"\nProcessing {file_path}...")
    
    try:
        # Read parquet file
        table = pq.read_table(file_path)
        data = table.to_pylist()
        logger.info(f"Read {len(data)} rows from {file_path}")
        logger.debug(f"Schema: {table.schema}")
        
        # Connect to Trino
        conn = connect(
            host="trino",  # Use container name since we're in Docker
            port=8080,
            user="admin",
            catalog="iceberg",
            schema="iceberg"
        )
        
        # Prepare the INSERT statement
        columns = table.schema.names
        column_names = ", ".join(columns)
        
        # Execute in batches
        batch_size = 500  # Increased batch size for faster loading
        cursor = conn.cursor()
        
        logger.info(f"Loading {len(data)} rows into {table_name}...")
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Create VALUES clause
            values_list = []
            for row_idx, row in enumerate(batch):
                try:
                    values = [convert_value(row[col], col) for col in columns]
                    values_list.append(f"({', '.join(values)})")
                except Exception as e:
                    logger.error(f"Error converting row {i + row_idx}: {str(e)}")
                    logger.error(f"Row data: {row}")
                    continue
            
            if not values_list:
                logger.warning(f"No valid values in batch starting at row {i}")
                continue
                
            values_clause = ",\n".join(values_list)
            sql = f"INSERT INTO iceberg.iceberg.{table_name} ({column_names}) VALUES {values_clause}"
            
            try:
                cursor.execute(sql)
                logger.info(f"Loaded {min(i + batch_size, len(data))} rows...")
            except Exception as e:
                logger.error(f"Error loading batch starting at row {i}: {str(e)}")
                logger.error(f"First row in batch: {batch[0]}")
                logger.error(f"SQL: {sql[:1000]}..." if len(sql) > 1000 else sql)
                raise
        
        logger.info(f"Finished loading {table_name}")
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        raise

def main():
    try:
        # Get absolute paths for data files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        # Verify data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Reset MinIO storage for a clean start
        reset_minio_storage()
        
        # Wait for Trino to be ready before proceeding
        wait_for_trino_ready()
        
        # Define tables
        tables = {
            'customers': """
                CREATE TABLE IF NOT EXISTS iceberg.iceberg.customers (
                    customer_id INT,
                    name VARCHAR,
                    email VARCHAR,
                    phone VARCHAR,
                    address VARCHAR,
                    city VARCHAR,
                    region VARCHAR,
                    signup_date DATE,
                    loyalty_tier VARCHAR
                ) WITH (
                    format = 'PARQUET',
                    partitioning = ARRAY['region'],
                    location = 's3://iceberg/customers'
                )
            """,
            'products': """
                CREATE TABLE IF NOT EXISTS iceberg.iceberg.products (
                    product_id INT,
                    name VARCHAR,
                    category VARCHAR,
                    subcategory VARCHAR,
                    price DECIMAL(10,2),
                    cost DECIMAL(10,2),
                    in_stock BOOLEAN,
                    min_stock INT,
                    max_stock INT,
                    supplier VARCHAR
                ) WITH (
                    format = 'PARQUET',
                    partitioning = ARRAY['category'],
                    location = 's3://iceberg/products'
                )
            """,
            'sales': """
                CREATE TABLE IF NOT EXISTS iceberg.iceberg.sales (
                    order_id INT,
                    order_date DATE,
                    customer_id INT,
                    product_id INT,
                    quantity INT,
                    unit_price DECIMAL(10,2),
                    gross_amount DECIMAL(10,2),
                    discount DECIMAL(10,2),
                    net_amount DECIMAL(10,2),
                    region VARCHAR,
                    payment_method VARCHAR
                ) WITH (
                    format = 'PARQUET',
                    partitioning = ARRAY['region', 'payment_method'],
                    location = 's3://iceberg/sales'
                )
            """
        }
        
        # Clean up existing data
        # 1. Drop existing tables
        drop_tables(tables.keys())
        
        # 2. Clean MinIO bucket
        clean_minio_bucket()
        
        # Create schema first
        conn = connect(
            host="trino",
            port=8080,
            user="admin",
            catalog="iceberg",
            schema="iceberg"
        )
        cursor = conn.cursor()
        
        try:
            logger.info("Creating schema...")
            cursor.execute("CREATE SCHEMA IF NOT EXISTS iceberg.iceberg")
            logger.info("Schema created successfully")
        except Exception as e:
            logger.error(f"Error creating schema: {str(e)}")
            raise
        
        # Create tables
        for table_name, create_sql in tables.items():
            try:
                logger.info(f"Creating table {table_name}...")
                cursor.execute(create_sql)
                logger.info(f"Table {table_name} created successfully")
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {str(e)}")
                raise
        
        cursor.close()
        conn.close()
        
        # Load each table
        for table in tables.keys():
            file_path = os.path.join(data_dir, f"{table}.parquet")
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                continue
            load_parquet_to_trino(file_path, table)
        
        logger.info("Data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 