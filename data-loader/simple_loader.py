#!/usr/bin/env python3
"""Simplified loader to get data directly from Socrata to Trino"""

import os
import pandas as pd
import requests
import json
import time
from logger_config import setup_logger
from trino_connector import get_trino_connection_manager
from app_config import get_config

logger = setup_logger(__name__)
config = get_config()

def load_dataset(dataset_id, force_reload=False):
    """Load dataset directly from Socrata to Trino"""
    start_time = time.time()
    
    logger.info(f"Loading dataset {dataset_id}")
    
    # 1. Fetch data from Socrata API (no auth required)
    socrata_url = f"https://data.cityofnewyork.us/resource/{dataset_id}.json"
    logger.info(f"Fetching data from {socrata_url}")
    
    response = requests.get(socrata_url, params={"$limit": 10000})
    
    if response.status_code != 200:
        logger.error(f"Error fetching data: {response.status_code} - {response.text}")
        return {"success": False, "error": f"API request failed: {response.status_code}"}
    
    data = response.json()
    logger.info(f"Fetched {len(data)} records")
    
    if not data:
        logger.error("No data returned from API")
        return {"success": False, "error": "No data returned from API"}
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(data)
    logger.info(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    
    # 3. Save to parquet for caching
    os.makedirs("data_cache", exist_ok=True)
    parquet_path = f"data_cache/{dataset_id}.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved data to {parquet_path}")
    
    # 4. Connect to Trino and create table
    try:
        trino = get_trino_connection_manager()
        conn = trino.get_connection()
        cursor = conn.cursor()
        
        # Create schema if it doesn't exist
        schema_name = "nyc_data"
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS iceberg.{schema_name}")
        
        # Create table with appropriate data types
        table_name = f"dataset_{dataset_id.replace('-', '_')}"
        
        # Drop table if it exists and force_reload is True
        if force_reload:
            cursor.execute(f"DROP TABLE IF EXISTS iceberg.{schema_name}.{table_name}")
        
        # Generate column definitions
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if 'int' in dtype:
                col_type = 'INTEGER'
            elif 'float' in dtype:
                col_type = 'DOUBLE'
            elif 'datetime' in dtype:
                col_type = 'TIMESTAMP'
            else:
                col_type = 'VARCHAR'
            
            # Clean column name (remove special chars)
            clean_col = ''.join(c if c.isalnum() else '_' for c in col)
            columns.append(f'"{clean_col}" {col_type}')
        
        # Create table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS iceberg.{schema_name}.{table_name} (
            {", ".join(columns)}
        )
        """
        cursor.execute(create_table_sql)
        logger.info(f"Created table iceberg.{schema_name}.{table_name}")
        
        # Insert data in batches
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare INSERT statement
            cols = [f'"{c}"' for c in batch.columns]
            cols_str = ", ".join(cols)
            
            values_list = []
            for _, row in batch.iterrows():
                values = []
                for val in row:
                    if pd.isna(val):
                        values.append('NULL')
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    else:
                        # Escape single quotes
                        val_str = str(val).replace("'", "''")
                        values.append(f"'{val_str}'")
                values_list.append(f"({', '.join(values)})")
            
            insert_sql = f"""
            INSERT INTO iceberg.{schema_name}.{table_name} ({cols_str})
            VALUES {', '.join(values_list)}
            """
            cursor.execute(insert_sql)
            logger.info(f"Inserted batch of {len(batch)} rows")
        
        # Verify the data was inserted
        cursor.execute(f"SELECT COUNT(*) FROM iceberg.{schema_name}.{table_name}")
        count = cursor.fetchone()[0]
        
        end_time = time.time()
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "schema_name": schema_name,
            "table_name": table_name, 
            "row_count": count,
            "load_time_seconds": round(end_time - start_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Error loading data to Trino: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Test with NYC For Hire Vehicles dataset
    result = load_dataset("8wbx-tsch")
    print(json.dumps(result, indent=2)) 