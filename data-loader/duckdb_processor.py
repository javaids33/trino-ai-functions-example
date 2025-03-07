import os
import logging
import tempfile
import shutil
import pandas as pd
import duckdb
from typing import Dict, List, Any, Iterator, Optional
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/duckdb_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DuckDBProcessor:
    """
    A class to handle data processing using DuckDB for memory-efficient operations
    and optimized Parquet file generation.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the DuckDB processor.
        
        Args:
            temp_dir: Optional temporary directory to use for processing
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="duckdb_processor_")
        logger.info(f"Initialized DuckDB processor with temp directory: {self.temp_dir}")
        
        # Create a DuckDB connection
        self.db_path = os.path.join(self.temp_dir, "processing.duckdb")
        self.conn = duckdb.connect(database=self.db_path)
        logger.info(f"Connected to DuckDB at {self.db_path}")
    
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                logger.info("Closed DuckDB connection")
            
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def process_dataframe_chunks(self, 
                               chunk_generator: Iterator[pd.DataFrame], 
                               table_name: str = "dataset") -> Dict[str, Any]:
        """
        Process dataframe chunks using DuckDB for memory efficiency.
        
        Args:
            chunk_generator: Iterator yielding pandas DataFrames
            table_name: Name of the table to create in DuckDB
            
        Returns:
            Dictionary with processing statistics
        """
        try:
            # Initialize counters
            total_rows = 0
            chunk_count = 0
            columns = []
            
            # Process the first chunk to create the table
            try:
                first_chunk = next(chunk_generator)
                
                # Replace None values with appropriate defaults
                for col in first_chunk.columns:
                    if pd.api.types.is_string_dtype(first_chunk[col]):
                        first_chunk[col] = first_chunk[col].fillna('')
                    elif pd.api.types.is_numeric_dtype(first_chunk[col]):
                        first_chunk[col] = first_chunk[col].fillna(0)
                    elif pd.api.types.is_datetime64_dtype(first_chunk[col]):
                        first_chunk[col] = first_chunk[col].fillna(pd.Timestamp('1970-01-01'))
                    elif pd.api.types.is_bool_dtype(first_chunk[col]):
                        first_chunk[col] = first_chunk[col].fillna(False)
                
                chunk_count += 1
                total_rows += len(first_chunk)
                columns = list(first_chunk.columns)
                
                # Register the first chunk and create the table
                self.conn.register("first_chunk", first_chunk)
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM first_chunk")
                logger.info(f"Created table {table_name} with {len(first_chunk)} rows and {len(columns)} columns")
                
                # Free memory
                del first_chunk
                gc.collect()
                
            except StopIteration:
                logger.warning("No data chunks provided")
                return {"total_rows": 0, "columns": [], "chunk_count": 0}
            
            # Process remaining chunks
            for i, df_chunk in enumerate(chunk_generator, 1):
                # Replace None values with appropriate defaults
                for col in df_chunk.columns:
                    if pd.api.types.is_string_dtype(df_chunk[col]):
                        df_chunk[col] = df_chunk[col].fillna('')
                    elif pd.api.types.is_numeric_dtype(df_chunk[col]):
                        df_chunk[col] = df_chunk[col].fillna(0)
                    elif pd.api.types.is_datetime64_dtype(df_chunk[col]):
                        df_chunk[col] = df_chunk[col].fillna(pd.Timestamp('1970-01-01'))
                    elif pd.api.types.is_bool_dtype(df_chunk[col]):
                        df_chunk[col] = df_chunk[col].fillna(False)
                
                chunk_count += 1
                chunk_rows = len(df_chunk)
                total_rows += chunk_rows
                
                # Register the chunk and insert into the table
                chunk_name = f"chunk_{i}"
                self.conn.register(chunk_name, df_chunk)
                self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM {chunk_name}")
                logger.info(f"Processed chunk {i} with {chunk_rows} rows (total: {total_rows} rows)")
                
                # Free memory
                del df_chunk
                gc.collect()
            
            return {
                "total_rows": total_rows,
                "columns": columns,
                "chunk_count": chunk_count
            }
            
        except Exception as e:
            logger.error(f"Error processing dataframe chunks: {e}")
            return {"total_rows": 0, "columns": [], "chunk_count": 0, "error": str(e)}
    
    def save_to_parquet(self, 
                      table_name: str, 
                      output_path: str,
                      partitioning_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Save a DuckDB table to a Parquet file with optional partitioning.
        
        Args:
            table_name: Name of the table in DuckDB
            output_path: Path to save the Parquet file
            partitioning_columns: Optional list of columns to partition by
            
        Returns:
            Dictionary with file statistics
        """
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get table statistics
            table_stats = self.conn.execute(f"SELECT COUNT(*) as row_count FROM {table_name}").fetchone()
            row_count = table_stats[0] if table_stats else 0
            
            # Get column information
            columns_info = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            column_names = [col[0] for col in columns_info]
            column_types = [col[1] for col in columns_info]
            
            # Check if partitioning columns exist in the table
            valid_partition_cols = []
            if partitioning_columns:
                valid_partition_cols = [col for col in partitioning_columns if col in column_names]
                if valid_partition_cols:
                    logger.info(f"Using partitioning columns: {valid_partition_cols}")
                else:
                    logger.warning(f"None of the specified partitioning columns {partitioning_columns} exist in the table")
            
            # Write to Parquet with optimized settings
            if valid_partition_cols:
                # Create a temporary directory for partitioned output
                partition_dir = os.path.join(os.path.dirname(output_path), "partitioned")
                os.makedirs(partition_dir, exist_ok=True)
                
                # Export with partitioning
                self.conn.execute(f"""
                    COPY {table_name} TO '{partition_dir}' 
                    (FORMAT 'PARQUET', COMPRESSION 'ZSTD', ROW_GROUP_SIZE 100000, PARTITION_BY ({', '.join(valid_partition_cols)}))
                """)
                
                # Combine partitioned files into a single file
                self.conn.execute(f"""
                    COPY (SELECT * FROM read_parquet('{partition_dir}/*/*.parquet')) 
                    TO '{output_path}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD', ROW_GROUP_SIZE 100000)
                """)
                
                # Clean up partition directory
                shutil.rmtree(partition_dir)
            else:
                # Export without partitioning
                self.conn.execute(f"""
                    COPY {table_name} TO '{output_path}' 
                    (FORMAT 'PARQUET', COMPRESSION 'ZSTD', ROW_GROUP_SIZE 100000)
                """)
            
            # Get file size
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            logger.info(f"Saved {row_count} rows to {output_path} ({file_size/1024/1024:.2f} MB)")
            
            return {
                "row_count": row_count,
                "column_count": len(column_names),
                "file_size": file_size,
                "columns": column_names,
                "column_types": column_types
            }
            
        except Exception as e:
            logger.error(f"Error saving to Parquet: {e}")
            return {"row_count": 0, "column_count": 0, "file_size": 0, "error": str(e)}
    
    def optimize_parquet(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Optimize an existing Parquet file using DuckDB's optimized writer.
        
        Args:
            input_path: Path to the input Parquet file
            output_path: Path to save the optimized Parquet file
            
        Returns:
            Dictionary with optimization statistics
        """
        try:
            # Load the Parquet file into a temporary table
            temp_table = "temp_optimize"
            self.conn.execute(f"CREATE TABLE {temp_table} AS SELECT * FROM read_parquet('{input_path}')")
            
            # Get original statistics
            original_stats = self.conn.execute(f"SELECT COUNT(*) as row_count FROM {temp_table}").fetchone()
            original_row_count = original_stats[0] if original_stats else 0
            original_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
            
            # Write optimized Parquet
            self.conn.execute(f"""
                COPY {temp_table} TO '{output_path}' 
                (FORMAT 'PARQUET', COMPRESSION 'ZSTD', ROW_GROUP_SIZE 100000)
            """)
            
            # Get optimized file size
            optimized_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            # Calculate compression ratio
            compression_ratio = original_size / optimized_size if optimized_size > 0 else 1.0
            
            logger.info(f"Optimized Parquet file: {input_path} -> {output_path}")
            logger.info(f"Original size: {original_size/1024/1024:.2f} MB, Optimized size: {optimized_size/1024/1024:.2f} MB")
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            
            return {
                "row_count": original_row_count,
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": compression_ratio
            }
            
        except Exception as e:
            logger.error(f"Error optimizing Parquet file: {e}")
            return {"row_count": 0, "original_size": 0, "optimized_size": 0, "compression_ratio": 1.0, "error": str(e)}
        finally:
            # Clean up the temporary table
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS temp_optimize")
            except:
                pass
    
    def analyze_dataset(self, table_name: str) -> Dict[str, Any]:
        """
        Analyze a dataset to get statistics and insights.
        
        Args:
            table_name: Name of the table in DuckDB
            
        Returns:
            Dictionary with dataset statistics and insights
        """
        try:
            # Get basic statistics
            stats = self.conn.execute(f"""
                SELECT 
                    COUNT(*) as row_count,
                    (SELECT COUNT(*) FROM pragma_table_info('{table_name}')) as column_count
                FROM {table_name}
            """).fetchone()
            
            row_count = stats[0] if stats else 0
            column_count = stats[1] if stats else 0
            
            # Get column statistics
            column_stats = []
            columns = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            
            for col in columns:
                col_name = col[0]
                col_type = col[1]
                
                # Get basic column statistics
                col_stats = self.conn.execute(f"""
                    SELECT 
                        COUNT(*) as count,
                        COUNT(DISTINCT "{col_name}") as distinct_count,
                        COUNT(*) - COUNT("{col_name}") as null_count
                    FROM {table_name}
                """).fetchone()
                
                # Calculate null percentage
                null_percentage = (col_stats[2] / row_count * 100) if row_count > 0 else 0
                
                # Calculate cardinality ratio (distinct values / total values)
                cardinality_ratio = (col_stats[1] / row_count) if row_count > 0 else 0
                
                column_stats.append({
                    "name": col_name,
                    "type": col_type,
                    "null_count": col_stats[2],
                    "null_percentage": null_percentage,
                    "distinct_count": col_stats[1],
                    "cardinality_ratio": cardinality_ratio
                })
            
            return {
                "row_count": row_count,
                "column_count": column_count,
                "column_stats": column_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            return {"row_count": 0, "column_count": 0, "column_stats": [], "error": str(e)} 