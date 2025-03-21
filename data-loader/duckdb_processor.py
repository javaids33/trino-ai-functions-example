import os
import logging
import tempfile
import shutil
import pandas as pd
import duckdb
from typing import Dict, List, Any, Iterator, Optional
import gc
from logger_config import setup_logger

# Configure logging
logger = setup_logger(__name__)

class DuckDBProcessor:
    """
    A class to handle data processing using DuckDB for memory-efficient operations
    and optimized Parquet file generation.
    """
    
    def __init__(self, db_path=":memory:"):
        """
        Initialize the DuckDB processor.
        
        Args:
            db_path: Path to the DuckDB database
        """
        self.conn = duckdb.connect(db_path)
        self.table_created = False
        self.table_name = "dataset_table"
        logger.info(f"Initialized DuckDBProcessor with database at {db_path}")
    
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
    
    def process_dataframe_chunks(self, chunks, table_name=None):
        """Process a list of DataFrame chunks and store in DuckDB.
        
        Args:
            chunks: List of DataFrame objects to process
            table_name: Optional name for the table
        """
        if not chunks or len(chunks) == 0:
            logger.warning("No data chunks provided")
            return False
        
        try:
            logger.info(f"Processing {len(chunks)} chunks with table_name: {table_name}")
            # Take the FIRST chunk to create the table schema
            first_chunk = chunks[0]
            logger.info(f"Creating table nyc_data with {len(first_chunk)} rows and {len(first_chunk.columns)} columns")
            
            # Create the table with the schema from the first chunk
            self.cursor.execute(f"CREATE TABLE nyc_data AS SELECT * FROM first_chunk")
            
            # For subsequent chunks, validate that the schema matches before inserting
            for i, chunk in enumerate(chunks[1:], 1):
                # Ensure the chunk has the same columns as the table
                if set(chunk.columns) != set(first_chunk.columns):
                    # Handle schema mismatch by aligning columns
                    logger.warning(f"Schema mismatch in chunk {i+1}. Aligning columns...")
                    chunk = chunk[first_chunk.columns]
                
                # Insert the chunk into the table
                self.cursor.execute(f"INSERT INTO nyc_data SELECT * FROM chunk")
                logger.info(f"Processed chunk {i+1} with {len(chunk)} rows (total: {(i+1)*len(chunk)} rows)")
            
            return True
        except Exception as e:
            logger.error(f"Error processing dataframe chunks: {str(e)}")
            return False
    
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

    def add_chunk(self, chunk):
        """Add a pandas DataFrame chunk to DuckDB with schema alignment"""
        if not isinstance(chunk, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            return False
            
        if chunk.empty:
            logger.warning("Empty chunk provided, skipping")
            return False
            
        try:
            if not self.table_created:
                # Create the table with the first chunk
                self.conn.register("temp_df", chunk)
                self.conn.execute(f"CREATE TABLE {self.table_name} AS SELECT * FROM temp_df")
                self.table_created = True
                # Keep reference to original columns for schema validation
                self.columns = list(chunk.columns)
                logger.info(f"Created table {self.table_name} with {len(chunk)} rows and {len(chunk.columns)} columns")
            else:
                # Check for schema drift
                if set(chunk.columns) != set(self.columns):
                    logger.warning(f"Schema mismatch detected: table has {len(self.columns)} columns, chunk has {len(chunk.columns)} columns")
                    
                    # Ensure chunk has all columns from original schema
                    for col in self.columns:
                        if col not in chunk.columns:
                            logger.warning(f"Adding missing column '{col}' to chunk")
                            chunk[col] = None
                    
                    # Select only columns that exist in the original schema
                    aligned_chunk = chunk[self.columns]
                    logger.info(f"Aligned chunk to match table schema with {len(self.columns)} columns")
                    
                    # Append the aligned chunk
                    self.conn.register("temp_df", aligned_chunk)
                    self.conn.execute(f"INSERT INTO {self.table_name} SELECT * FROM temp_df")
                else:
                    # Schemas match, append directly
                    self.conn.register("temp_df", chunk)
                    self.conn.execute(f"INSERT INTO {self.table_name} SELECT * FROM temp_df")
                    
                logger.info(f"Appended {len(chunk)} rows to {self.table_name}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunk to DuckDB: {str(e)}")
            return False
            
    def to_parquet(self, output_path):
        """Export DuckDB table to Parquet file"""
        try:
            if not self.table_created:
                logger.error("No data has been added to the processor")
                return None
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export to Parquet
            self.conn.execute(f"COPY {self.table_name} TO '{output_path}' (FORMAT 'PARQUET')")
            logger.info(f"Exported data to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to Parquet: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up resources used by the processor"""
        try:
            # Close the DuckDB connection
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                logger.info("Closed DuckDB connection")
            
            # Remove temporary directory if it was created by this instance
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Removed temporary directory: {self.temp_dir}")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def _clean_column_name(self, name):
        """Clean column name to be compatible with SQL"""
        import re
        
        # Replace spaces and special characters with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', str(name).lower())
        
        # Remove consecutive underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        
        # Remove leading and trailing underscores
        clean_name = clean_name.strip('_')
        
        # Ensure it doesn't start with a number
        if clean_name and clean_name[0].isdigit():
            clean_name = 'col_' + clean_name
            
        # If empty, use a default name
        if not clean_name:
            clean_name = 'column'
            
        return clean_name 