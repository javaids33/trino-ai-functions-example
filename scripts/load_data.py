import pyarrow.parquet as pq
from trino.dbapi import connect
import os
from datetime import date, datetime
import decimal
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            sql = f"INSERT INTO {table_name} ({column_names}) VALUES {values_clause}"
            
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
        data_dir = os.path.join(script_dir, '..', 'data')
        
        # Verify data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # First, create tables if they don't exist
        conn = connect(
            host="trino",  # Use container name since we're in Docker
            port=8080,
            user="admin",
            catalog="iceberg",
            schema="iceberg"
        )
        cursor = conn.cursor()
        
        # Read and execute create tables SQL
        with open(os.path.join(script_dir, 'create_tables.sql'), 'r') as f:
            create_tables_sql = f.read()
            for statement in create_tables_sql.split(';'):
                if statement.strip():
                    try:
                        cursor.execute(statement)
                        logger.info("Executed table creation SQL")
                    except Exception as e:
                        logger.error(f"Error executing table creation SQL: {str(e)}")
        
        cursor.close()
        conn.close()
        
        # Clear existing data
        conn = connect(
            host="trino",  # Use container name since we're in Docker
            port=8080,
            user="admin",
            catalog="iceberg",
            schema="iceberg"
        )
        cursor = conn.cursor()
        
        tables = ['customers', 'products', 'sales']
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
                logger.info(f"Cleared existing data from {table}")
            except Exception as e:
                logger.error(f"Error clearing data from {table}: {str(e)}")
        
        cursor.close()
        conn.close()
        
        # Load each table
        for table in tables:
            file_path = os.path.join(data_dir, f"{table}.parquet")
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                continue
            load_parquet_to_trino(file_path, table)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 