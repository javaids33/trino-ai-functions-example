import logging
from typing import Dict, List, Any
import trino
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import boto3
from botocore.client import Config
import json
from trino.exceptions import TrinoUserError
from contextlib import contextmanager
import duckdb
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class TrinoClient:
    def __init__(self):
        """Initialize Trino client with configuration from environment variables"""
        logger.info(f"Initializing Trino client with host={os.getenv('TRINO_HOST')}, port={os.getenv('TRINO_PORT')}, user={os.getenv('TRINO_USER')}, catalog={os.getenv('TRINO_CATALOG')}, schema={os.getenv('TRINO_SCHEMA')}")
        
        self.host = os.getenv('TRINO_HOST', 'localhost')
        self.port = int(os.getenv('TRINO_PORT', '8080'))
        self.user = os.getenv('TRINO_USER', 'admin')
        self.catalog = os.getenv('TRINO_CATALOG', 'iceberg')
        self.schema = os.getenv('TRINO_SCHEMA', 'iceberg')
        
        self.conn = trino.dbapi.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            catalog=self.catalog,
            schema=self.schema,
            http_scheme='http'
        )
        self._test_connection()
        self._ensure_schema_exists()
        
    @contextmanager
    def _get_cursor(self):
        """Get a cursor with proper context management"""
        cursor = self.conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _test_connection(self):
        """Test connection to Trino server"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchall()
            logger.info("Successfully connected to Trino server")
        except Exception as e:
            logger.error(f"Failed to connect to Trino server: {str(e)}")
            raise

    def _ensure_schema_exists(self):
        """Ensure the schema exists"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            logger.info(f"Schema {self.schema} created or already exists")
        except Exception as e:
            logger.error(f"Failed to create schema: {str(e)}")
            raise

    def create_schema(self, schema_name: str):
        """Create a schema if it doesn't exist"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{schema_name}")
            logger.info(f"Schema {schema_name} created or already exists")
        except Exception as e:
            logger.error(f"Error creating schema {schema_name}: {str(e)}")
            raise

    def create_table_from_parquet(self, parquet_file: str, table_name: str, schema_name: str = None) -> None:
        """Create a table from a parquet file"""
        if schema_name is None:
            schema_name = self.schema
            
        try:
            # Create schema if it doesn't exist
            self.create_schema(schema_name)
            
            # Read parquet file schema using duckdb
            con = duckdb.connect()
            df = con.execute(f"SELECT * FROM parquet_scan('{parquet_file}') LIMIT 0").df()
            
            # Generate column definitions
            columns = []
            for col_name, dtype in df.dtypes.items():
                trino_type = self._map_pandas_type_to_trino(dtype)
                # Escape column names that contain special characters
                escaped_col_name = col_name.replace('"', '""')
                columns.append(f'"{escaped_col_name}" {trino_type}')
                
            columns_str = ",\n    ".join(columns)
            
            # Drop table if exists
            drop_table_sql = f'DROP TABLE IF EXISTS {self.catalog}.{schema_name}.{table_name}'
            logger.debug(f"Executing SQL: {drop_table_sql}")
            with self._get_cursor() as cursor:
                cursor.execute(drop_table_sql)
            
            # Create table
            create_table_sql = f'''
            CREATE TABLE {self.catalog}.{schema_name}.{table_name} (
                {columns_str}
            )
            WITH (
                format = 'PARQUET'
            )
            '''
            logger.debug(f"Executing SQL: {create_table_sql}")
            with self._get_cursor() as cursor:
                cursor.execute(create_table_sql)
            
            # Load data
            load_data_sql = f'''
            INSERT INTO {self.catalog}.{schema_name}.{table_name}
            SELECT *
            FROM parquet_scan('{parquet_file}')
            '''
            logger.debug(f"Executing SQL: {load_data_sql}")
            with self._get_cursor() as cursor:
                cursor.execute(load_data_sql)
            
        except Exception as e:
            logger.error(f"Error loading data from parquet: {str(e)}")
            raise

    def _map_pandas_type_to_trino(self, pandas_type: str) -> str:
        """Map pandas dtype to Trino type"""
        type_map = {
            'object': 'VARCHAR',
            'int64': 'BIGINT',
            'float64': 'DOUBLE',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'timedelta64[ns]': 'INTERVAL DAY TO SECOND'
        }
        return type_map.get(str(pandas_type), 'VARCHAR')

    def _execute_with_retry(self, sql: str, params: List[Any] = None, max_retries: int = 3):
        """Execute a SQL statement with retry logic"""
        last_error = None
        for attempt in range(max_retries):
            conn = self.conn
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                return cursor.fetchall()
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            finally:
                cursor.close()
                conn.close()
            
        raise last_error

    def sanitize_table_name(self, table_name: str) -> str:
        """Sanitize table name to be compatible with Trino"""
        return table_name.replace('-', '_').replace(' ', '_').lower()

    def check_connection(self):
        """Check connection to Trino server"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT 1')
        cursor.fetchone()
        cursor.close()

    def _create_schema_if_not_exists(self):
        """Create schema if it doesn't exist"""
        cursor = self.conn.cursor()
        cursor.execute(f'CREATE SCHEMA IF NOT EXISTS {self.schema}')
        cursor.close()

    def _create_bucket_if_not_exists(self):
        """Create MinIO bucket if it doesn't exist"""
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT schema_name FROM information_schema.schemata WHERE schema_name = \'{self.schema}\'')
        if not cursor.fetchone():
            cursor.execute(f'CREATE SCHEMA {self.schema}')
        cursor.close()

    def _get_column_type(self, dtype: str, sample_value: Any = None) -> str:
        """Map pandas dtype to Trino type"""
        if pd.isna(sample_value):
            sample_value = None

        if dtype.startswith('int'):
            return 'integer'
        elif dtype.startswith('float'):
            return 'double'
        elif dtype.startswith('bool'):
            return 'boolean'
        elif dtype.startswith('datetime'):
            return 'timestamp'
        elif isinstance(sample_value, dict):
            return 'json'
        else:
            # For string columns, determine the max length
            if sample_value is not None:
                try:
                    max_length = len(str(sample_value))
                    return f'varchar({max_length * 2})'  # Double the length to be safe
                except:
                    pass
            return 'varchar'

    def _create_table(self, table_name: str, df: pd.DataFrame):
        """Create an Iceberg table with the correct schema"""
        try:
            # Split table name into schema and table if needed
            if '.' in table_name:
                schema, table = table_name.split('.')
            else:
                schema = self.schema
                table = table_name

            # Generate column definitions
            columns = []
            for col in df.columns:
                # Get a non-null sample value for better type inference
                sample_value = df[col].dropna().iloc[0] if not df[col].isna().all() else None
                col_type = self._get_column_type(str(df[col].dtype), sample_value)
                columns.append(f"{col} {col_type}")

            # Create schema if it doesn't exist
            self._execute_with_retry(f"CREATE SCHEMA IF NOT EXISTS {schema}")

            # Create table
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                {', '.join(columns)}
            )
            WITH (
                format = 'PARQUET'
            )
            """
            
            self._execute_with_retry(create_table_sql)
            logger.info(f"Created table {schema}.{table}")

        except Exception as e:
            logger.error(f"Error creating table {table_name}: {str(e)}")
            raise

    def load_data(self, table_name: str, parquet_file: str) -> None:
        """Load data from parquet file into Trino table"""
        try:
            # Read parquet file metadata to get total row count
            parquet_file = pq.ParquetFile(parquet_file)
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"Loading {total_rows} rows from {parquet_file} into {table_name}")

            # Create table if it doesn't exist using the parquet schema
            df = pd.read_parquet(parquet_file, engine='pyarrow')
            create_table_sql = self._get_create_table_sql(table_name, df)
            
            with self._get_cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                cursor.execute(create_table_sql)
                logger.info(f"Created table {table_name}")

            # Load data in chunks
            chunk_size = 10000
            for i in range(0, len(df), chunk_size):
                chunk = df[i:i + chunk_size]
                insert_sql = self._get_insert_sql(table_name, chunk)
                with self._get_cursor() as cursor:
                    cursor.execute(insert_sql)
                logger.info(f"Loaded chunk {i//chunk_size + 1} of {(len(df) + chunk_size - 1)//chunk_size} ({len(chunk)} rows)")

            logger.info(f"Successfully loaded {total_rows} rows into {table_name}")

        except Exception as e:
            logger.error(f"Failed to load data into table {table_name}: {str(e)}")
            raise

    def _get_create_table_sql(self, table_name: str, df: pd.DataFrame) -> str:
        """Generate CREATE TABLE SQL statement from DataFrame schema"""
        type_mapping = {
            'object': 'VARCHAR',
            'int64': 'BIGINT',
            'float64': 'DOUBLE',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'category': 'VARCHAR'
        }
        
        columns = []
        for col_name, dtype in df.dtypes.items():
            trino_type = type_mapping.get(str(dtype), 'VARCHAR')
            columns.append(f'"{col_name}" {trino_type}')
        
        return f"""
        CREATE TABLE {table_name} (
            {','.join(columns)}
        )
        WITH (
            format = 'PARQUET'
        )
        """

    def _get_insert_sql(self, table_name: str, df: pd.DataFrame) -> str:
        """Generate INSERT SQL statement from DataFrame"""
        columns = [f'"{col}"' for col in df.columns]
        values = []
        
        for _, row in df.iterrows():
            row_values = []
            for val in row:
                if pd.isna(val):
                    row_values.append('NULL')
                elif isinstance(val, str):
                    # Escape single quotes and wrap in single quotes
                    val = val.replace("'", "''")
                    row_values.append(f"'{val}'")
                elif isinstance(val, (int, float)):
                    row_values.append(str(val))
                elif isinstance(val, bool):
                    row_values.append(str(val).lower())
                else:
                    row_values.append(f"'{str(val)}'")
            values.append(f"({','.join(row_values)})")
        
        return f"""
        INSERT INTO {table_name} ({','.join(columns)})
        VALUES {','.join(values)}
        """

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        try:
            # Split table name into schema and table if needed
            if '.' in table_name:
                schema, table = table_name.split('.')
            else:
                schema = self.schema
                table = table_name

            # Get column information
            columns = []
            for row in self._execute_with_retry(f"DESCRIBE {schema}.{table}"):
                columns.append({
                    'name': row[0],
                    'type': row[1]
                })

            # Get row count
            row_count = self._execute_with_retry(f"SELECT COUNT(*) FROM {schema}.{table}")[0][0]

            return {
                'columns': columns,
                'row_count': row_count,
                'table_name': f"{schema}.{table}"
            }

        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            raise

    def list_tables(self) -> List[str]:
        """List all tables in the schema"""
        try:
            logger.info(f"Listing tables in schema {self.schema}")
            rows = self._execute_with_retry(f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = '{self.schema}'
            """)
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            raise

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results as a list of dictionaries"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor:
                    results.append(dict(zip(columns, row)))
                return results
        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            raise 