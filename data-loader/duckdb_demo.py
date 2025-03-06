#!/usr/bin/env python3
"""
DuckDB Integration Demo Script

This script demonstrates the DuckDB integration with the NYC Open Data ETL pipeline.
It shows how DuckDB can be used for memory-efficient data processing and optimized
Parquet file generation.
"""

import os
import logging
import pandas as pd
import tempfile
import argparse
from dotenv import load_dotenv
from duckdb_processor import DuckDBProcessor
from socrata_loader import SocrataToTrinoETL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def demo_duckdb_processing(dataset_id: str = "kxp8-n2sj"):
    """
    Demonstrate DuckDB processing with a NYC Open Data dataset.
    
    Args:
        dataset_id: Dataset ID to use for the demo (default: NYC Popular Baby Names)
    """
    logger.info(f"Starting DuckDB processing demo with dataset {dataset_id}")
    
    # Initialize ETL pipeline
    etl = SocrataToTrinoETL(
        app_token=os.environ.get('SOCRATA_APP_TOKEN'),
        api_key_id=os.environ.get('SOCRATA_API_KEY_ID'),
        api_key_secret=os.environ.get('SOCRATA_API_KEY_SECRET')
    )
    
    # Create a temporary directory for the demo
    temp_dir = tempfile.mkdtemp(prefix="duckdb_demo_")
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Fetch data in chunks
        logger.info(f"Fetching data for dataset {dataset_id} in chunks")
        chunk_generator = etl._fetch_data_in_chunks(dataset_id, chunk_size=5000)
        
        # Initialize DuckDB processor
        duckdb_processor = DuckDBProcessor(temp_dir=temp_dir)
        
        # Process chunks with DuckDB
        logger.info("Processing chunks with DuckDB")
        processing_result = duckdb_processor.process_dataframe_chunks(chunk_generator, table_name="nyc_data")
        
        logger.info(f"Processed {processing_result.get('total_rows', 0)} rows in {processing_result.get('chunk_count', 0)} chunks")
        
        # Run some SQL queries on the data
        logger.info("Running SQL queries on the data")
        
        # Get column names
        columns = duckdb_processor.conn.execute("DESCRIBE nyc_data").fetchall()
        logger.info(f"Columns in the dataset: {[col[0] for col in columns]}")
        
        # Get row count
        row_count = duckdb_processor.conn.execute("SELECT COUNT(*) FROM nyc_data").fetchone()[0]
        logger.info(f"Total rows in the dataset: {row_count}")
        
        # Run a sample query
        logger.info("Running a sample query")
        sample_query_result = duckdb_processor.conn.execute("""
            SELECT * FROM nyc_data LIMIT 5
        """).fetchall()
        logger.info(f"Sample data: {sample_query_result}")
        
        # Save to Parquet
        logger.info("Saving data to Parquet")
        output_path = os.path.join(temp_dir, f"{dataset_id}.parquet")
        save_result = duckdb_processor.save_to_parquet(
            table_name="nyc_data",
            output_path=output_path
        )
        
        logger.info(f"Saved {save_result.get('row_count', 0)} rows to {output_path}")
        logger.info(f"File size: {save_result.get('file_size', 0) / 1024 / 1024:.2f} MB")
        
        # Analyze the dataset
        logger.info("Analyzing the dataset")
        analysis_result = duckdb_processor.analyze_dataset("nyc_data")
        
        logger.info(f"Dataset analysis: {analysis_result.get('row_count', 0)} rows, {analysis_result.get('column_count', 0)} columns")
        
        # Print column statistics
        logger.info("Column statistics:")
        for col_stat in analysis_result.get('column_stats', []):
            logger.info(f"  {col_stat['name']} ({col_stat['type']}): {col_stat['null_percentage']:.2f}% null, {col_stat['distinct_count']} distinct values")
        
        logger.info("DuckDB processing demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in DuckDB processing demo: {e}")
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

def demo_duckdb_optimization(input_file: str = None):
    """
    Demonstrate DuckDB optimization of an existing Parquet file.
    
    Args:
        input_file: Path to an existing Parquet file to optimize
    """
    if not input_file:
        logger.error("No input file provided for optimization demo")
        return
    
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return
    
    logger.info(f"Starting DuckDB optimization demo with file {input_file}")
    
    # Create a temporary directory for the demo
    temp_dir = tempfile.mkdtemp(prefix="duckdb_optimize_demo_")
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Initialize DuckDB processor
        duckdb_processor = DuckDBProcessor(temp_dir=temp_dir)
        
        # Optimize the Parquet file
        output_path = os.path.join(temp_dir, "optimized.parquet")
        logger.info(f"Optimizing Parquet file: {input_file} -> {output_path}")
        
        optimization_result = duckdb_processor.optimize_parquet(input_file, output_path)
        
        logger.info(f"Optimization results:")
        logger.info(f"  Original size: {optimization_result.get('original_size', 0) / 1024 / 1024:.2f} MB")
        logger.info(f"  Optimized size: {optimization_result.get('optimized_size', 0) / 1024 / 1024:.2f} MB")
        logger.info(f"  Compression ratio: {optimization_result.get('compression_ratio', 1.0):.2f}x")
        
        logger.info("DuckDB optimization demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in DuckDB optimization demo: {e}")
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

def main():
    """Main function to run the DuckDB integration demo"""
    parser = argparse.ArgumentParser(description="DuckDB Integration Demo")
    
    # Command options
    parser.add_argument("command", choices=["process", "optimize"], help="Command to run")
    
    # Options for process
    parser.add_argument("--dataset-id", default="kxp8-n2sj", help="Dataset ID to process (default: NYC Popular Baby Names)")
    
    # Options for optimize
    parser.add_argument("--input-file", help="Path to an existing Parquet file to optimize")
    
    args = parser.parse_args()
    
    # Execute the chosen command
    if args.command == "process":
        demo_duckdb_processing(args.dataset_id)
    elif args.command == "optimize":
        demo_duckdb_optimization(args.input_file)

if __name__ == "__main__":
    main() 