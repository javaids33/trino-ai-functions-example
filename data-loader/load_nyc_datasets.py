#!/usr/bin/env python3
import os
import sys
import argparse
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from tabulate import tabulate
import concurrent.futures

from socrata_loader import SocrataToTrinoETL
from logger_config import setup_logger
from duckdb_processor import DuckDBProcessor
from duckdb_to_iceberg import convert_duckdb_to_iceberg

# Set up logger
logger = setup_logger(__name__)

def load_datasets(dataset_ids: List[str], validate: bool = True, quiet: bool = False, 
                 parallel: bool = False, max_workers: int = 3) -> Dict[str, Any]:
    """
    Load specified NYC Open Data datasets by ID and validate the ETL process
    
    Args:
        dataset_ids: List of NYC Open Data dataset IDs to load
        validate: Whether to validate the datasets after loading
        quiet: Reduce log output
        parallel: Whether to load datasets in parallel
        max_workers: Maximum number of concurrent workers for parallel loading
        
    Returns:
        Dictionary with results of the loading process
    """
    logger.info(f"Starting ETL process for {len(dataset_ids)} datasets")
    
    # Temporarily reduce log level if quiet mode is enabled
    if quiet:
        logging.getLogger("socrata_loader").setLevel(logging.WARNING)
        logging.getLogger("duckdb_processor").setLevel(logging.WARNING)
        logging.getLogger("cache_manager").setLevel(logging.WARNING)
    
    etl = SocrataToTrinoETL()
    
    results = {}
    
    if parallel and len(dataset_ids) > 1:
        logger.info(f"Using parallel processing with {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary mapping futures to dataset_ids for tracking
            future_to_dataset = {
                executor.submit(etl.create_trino_table_from_dataset, dataset_id): dataset_id 
                for dataset_id in dataset_ids
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_id = future_to_dataset[future]
                try:
                    start_time = time.time()
                    result = future.result()
                    end_time = time.time()
                    
                    if result.get("success", False):
                        logger.info(f"Successfully loaded dataset {dataset_id} in {end_time - start_time:.2f} seconds")
                        results[dataset_id] = {
                            "success": True,
                            "schema_name": result.get("schema_name"),
                            "table_name": result.get("table_name"),
                            "row_count": result.get("row_count"),
                            "time_taken": f"{end_time - start_time:.2f}s"
                        }
                    else:
                        error_message = result.get("error", "Unknown error")
                        logger.error(f"Failed to load dataset {dataset_id}: {error_message}")
                        results[dataset_id] = {
                            "success": False,
                            "error": error_message
                        }
                except Exception as e:
                    logger.error(f"Exception loading dataset {dataset_id}: {str(e)}")
                    results[dataset_id] = {
                        "success": False,
                        "error": str(e)
                    }
    else:
        # Sequential processing
        for i, dataset_id in enumerate(dataset_ids):
            logger.info(f"Processing dataset {dataset_id} ({i+1}/{len(dataset_ids)})")
            try:
                start_time = time.time()
                result = etl.create_trino_table_from_dataset(dataset_id)
                end_time = time.time()
                
                if result.get("success", False):
                    logger.info(f"Successfully loaded dataset {dataset_id} in {end_time - start_time:.2f} seconds")
                    results[dataset_id] = {
                        "success": True,
                        "schema_name": result.get("schema_name"),
                        "table_name": result.get("table_name"),
                        "row_count": result.get("row_count"),
                        "time_taken": f"{end_time - start_time:.2f}s"
                    }
                else:
                    error_message = result.get("error", "Unknown error")
                    logger.error(f"Failed to load dataset {dataset_id}: {error_message}")
                    results[dataset_id] = {
                        "success": False,
                        "error": error_message
                    }
            except Exception as e:
                logger.error(f"Exception loading dataset {dataset_id}: {str(e)}")
                results[dataset_id] = {
                    "success": False,
                    "error": str(e)
                }
    
    # Restore log level
    if quiet:
        logging.getLogger("socrata_loader").setLevel(logging.INFO)
        logging.getLogger("duckdb_processor").setLevel(logging.INFO)
        logging.getLogger("cache_manager").setLevel(logging.INFO)
    
    return results

def load_datasets_to_duckdb(dataset_ids: List[str], quiet: bool = False, 
                          parallel: bool = False, max_workers: int = 3) -> Dict[str, Any]:
    """
    Load specified NYC Open Data datasets into DuckDB
    
    Args:
        dataset_ids: List of NYC Open Data dataset IDs to load
        quiet: Reduce log output
        parallel: Whether to load datasets in parallel
        max_workers: Maximum number of concurrent workers for parallel loading
        
    Returns:
        Dictionary with results of the loading process
    """
    logger.info(f"Starting DuckDB ETL process for {len(dataset_ids)} datasets")
    
    # Temporarily reduce log level if quiet mode is enabled
    if quiet:
        logging.getLogger("socrata_loader").setLevel(logging.WARNING)
        logging.getLogger("duckdb_processor").setLevel(logging.WARNING)
        logging.getLogger("cache_manager").setLevel(logging.WARNING)
    
    etl = SocrataToTrinoETL()
    duckdb_processor = DuckDBProcessor()
    
    results = {}
    
    def load_to_duckdb(dataset_id):
        try:
            start_time = time.time()
            
            # First fetch data from Socrata (this will cache the dataset)
            # Check if we need to download the dataset first
            dataset_metadata = etl.get_dataset_metadata(dataset_id)
            dataset_title = dataset_metadata.get('resource', {}).get('name', 'Unknown')
            
            # Fetch data from Socrata (this will cache the dataset)
            fetch_result = etl.fetch_dataset(dataset_id)
            
            if not fetch_result.get("success", False):
                error_message = fetch_result.get("error", "Failed to fetch dataset")
                logger.error(f"Failed to fetch dataset {dataset_id}: {error_message}")
                return {
                    "success": False,
                    "error": error_message
                }
            
            # Load the dataset into DuckDB from cache
            cache_file = os.path.join(etl.cache_dir, f"{dataset_id}.parquet")
            if not os.path.exists(cache_file):
                error_message = f"Cache file for dataset {dataset_id} not found"
                logger.error(error_message)
                return {
                    "success": False,
                    "error": error_message
                }
            
            # Load data into DuckDB
            duckdb_processor.load_parquet_to_table(cache_file, dataset_id)
            
            # Get table metadata
            table_info = duckdb_processor.get_table_info(dataset_id)
            row_count = table_info.get('row_count', 0)
            columns = table_info.get('columns', [])
            
            end_time = time.time()
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "title": dataset_title,
                "duckdb_table": dataset_id,
                "row_count": row_count,
                "columns": columns,
                "time_taken": f"{end_time - start_time:.2f}s"
            }
        except Exception as e:
            logger.error(f"Exception loading dataset {dataset_id} to DuckDB: {str(e)}")
            return {
                "success": False,
                "dataset_id": dataset_id,
                "error": str(e)
            }
    
    if parallel and len(dataset_ids) > 1:
        logger.info(f"Using parallel processing with {max_workers} workers for DuckDB loading")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary mapping futures to dataset_ids for tracking
            future_to_dataset = {
                executor.submit(load_to_duckdb, dataset_id): dataset_id 
                for dataset_id in dataset_ids
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_id = future_to_dataset[future]
                try:
                    result = future.result()
                    results[dataset_id] = result
                except Exception as e:
                    logger.error(f"Exception processing future for dataset {dataset_id}: {str(e)}")
                    results[dataset_id] = {
                        "success": False,
                        "dataset_id": dataset_id,
                        "error": str(e)
                    }
    else:
        # Sequential processing
        for i, dataset_id in enumerate(dataset_ids):
            logger.info(f"Processing dataset {dataset_id} for DuckDB ({i+1}/{len(dataset_ids)})")
            result = load_to_duckdb(dataset_id)
            results[dataset_id] = result
    
    # Restore log level
    if quiet:
        logging.getLogger("socrata_loader").setLevel(logging.INFO)
        logging.getLogger("duckdb_processor").setLevel(logging.INFO)
        logging.getLogger("cache_manager").setLevel(logging.INFO)
    
    return results

def validate_datasets(etl: SocrataToTrinoETL, dataset_ids: List[str]) -> Dict[str, Any]:
    """
    Validate that datasets were properly loaded by checking the registry and row counts
    
    Args:
        etl: SocrataToTrinoETL instance
        dataset_ids: List of dataset IDs to validate
        
    Returns:
        Validation results for each dataset
    """
    logger.info(f"Validating {len(dataset_ids)} datasets")
    
    validation_results = {}
    for i, dataset_id in enumerate(dataset_ids):
        logger.info(f"Validating dataset {dataset_id} ({i+1}/{len(dataset_ids)})")
        try:
            # Check if dataset exists in registry
            registry_entry = etl._get_dataset_from_registry(dataset_id)
            
            if not registry_entry:
                validation_results[dataset_id] = {
                    "validated": False,
                    "error": "Dataset not found in registry"
                }
                continue
            
            # Get source metadata
            source_metadata = etl.get_dataset_metadata(dataset_id)
            source_row_count = int(source_metadata.get("resource", {}).get("rows_size", 0))
            
            # Get loaded row count
            loaded_row_count = registry_entry.get("row_count", 0)
            
            # Check if the row counts match (allow for small differences due to updates)
            row_count_diff = abs(source_row_count - loaded_row_count)
            row_count_diff_pct = (row_count_diff / source_row_count * 100) if source_row_count > 0 else 0
            
            validation_results[dataset_id] = {
                "validated": True,
                "source_row_count": source_row_count,
                "loaded_row_count": loaded_row_count,
                "row_count_diff": row_count_diff,
                "row_count_diff_pct": f"{row_count_diff_pct:.2f}%",
                "schema_name": registry_entry.get("schema_name"),
                "table_name": registry_entry.get("table_name"),
                "dataset_title": registry_entry.get("dataset_title"),
                "etl_timestamp": registry_entry.get("etl_timestamp")
            }
        except Exception as e:
            logger.error(f"Exception validating dataset {dataset_id}: {str(e)}")
            validation_results[dataset_id] = {
                "validated": False,
                "error": str(e)
            }
    
    return validation_results

def display_results(load_results: Dict[str, Any], validation_results: Dict[str, Any] = None):
    """
    Display results of loading and validation in a readable format
    """
    # Load results table
    load_table_data = []
    for dataset_id, result in load_results.items():
        if result.get("success", False):
            load_table_data.append([
                dataset_id, 
                "✅ Success", 
                result.get("schema_name", ""), 
                result.get("table_name", ""), 
                result.get("row_count", ""), 
                result.get("time_taken", "")
            ])
        else:
            load_table_data.append([
                dataset_id, 
                "❌ Failed", 
                "", 
                "", 
                "", 
                result.get("error", "Unknown error")
            ])
    
    print("\n=== DATASET LOADING RESULTS ===")
    print(tabulate(
        load_table_data, 
        headers=["Dataset ID", "Status", "Schema", "Table", "Row Count", "Time/Error"],
        tablefmt="grid"
    ))
    
    # Validation results table
    if validation_results:
        validation_table_data = []
        for dataset_id, result in validation_results.items():
            if result.get("validated", False):
                validation_table_data.append([
                    dataset_id,
                    result.get("dataset_title", "")[:30],
                    "✅ Available" if result.get("validated") else "❌ Not Available",
                    result.get("source_row_count", ""),
                    result.get("loaded_row_count", ""),
                    result.get("row_count_diff_pct", ""),
                    result.get("etl_timestamp", "")
                ])
            else:
                validation_table_data.append([
                    dataset_id,
                    "",
                    "❌ Not Available",
                    "",
                    "",
                    "",
                    result.get("error", "")
                ])
        
        print("\n=== DATASET VALIDATION RESULTS ===")
        print(tabulate(
            validation_table_data, 
            headers=["Dataset ID", "Title", "Status", "Source Rows", "Loaded Rows", "Diff %", "ETL Timestamp/Error"],
            tablefmt="grid"
        ))
        
        # Summary
        total_datasets = len(validation_results)
        successful_datasets = sum(1 for result in validation_results.values() if result.get("validated", False))
        print(f"\nSUMMARY: Successfully validated {successful_datasets} out of {total_datasets} datasets ({successful_datasets/total_datasets*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Load and validate NYC Open Data datasets")
    parser.add_argument("--dataset-ids", nargs="+", help="List of dataset IDs to load", required=False)
    parser.add_argument("--from-file", help="File containing dataset IDs (one per line)", required=False)
    parser.add_argument("--validate-only", action="store_true", help="Only validate datasets, don't load them")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce log output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Load datasets in parallel")
    parser.add_argument("--max-workers", "-w", type=int, default=3, help="Maximum number of concurrent workers for parallel loading")
    parser.add_argument("--to-duckdb", action="store_true", help="Load datasets into DuckDB instead of directly to Trino")
    parser.add_argument("--to-trino", action="store_true", help="Convert DuckDB tables to Trino Iceberg format after loading to DuckDB")
    
    args = parser.parse_args()
    
    # Get dataset IDs from command line or file
    dataset_ids = []
    if args.dataset_ids:
        dataset_ids = args.dataset_ids
    elif args.from_file:
        with open(args.from_file, 'r') as f:
            dataset_ids = [line.strip() for line in f if line.strip()]
    else:
        # Default dataset IDs if none provided
        dataset_ids = [
            "fm4z-qud6",
            "n7f2-dyvt",
            "dwrg-kzni",
            "2c5m-rke8",
            "2w2g-fk3i",
            "h3ce-uahi",
            "df32-vzax",
            "eqxx-j5y8",
            "jknp-skuy",
            "i642-2fxq"
        ]
    
    etl = SocrataToTrinoETL()
    
    if args.validate_only:
        # Only perform validation
        logger.info("Validating datasets without loading")
        validation_results = validate_datasets(etl, dataset_ids)
        display_results({dataset_id: {"success": True} for dataset_id in dataset_ids}, validation_results)
    elif args.to_duckdb:
        # Load datasets into DuckDB
        logger.info(f"Loading {len(dataset_ids)} datasets into DuckDB")
        duckdb_results = load_datasets_to_duckdb(
            dataset_ids, 
            quiet=args.quiet,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        
        # Display DuckDB results
        duckdb_table_data = []
        for dataset_id, result in duckdb_results.items():
            if result.get("success", False):
                duckdb_table_data.append([
                    dataset_id, 
                    "✅ Success", 
                    result.get("title", "")[:30],
                    result.get("duckdb_table", ""), 
                    result.get("row_count", ""), 
                    result.get("time_taken", "")
                ])
            else:
                duckdb_table_data.append([
                    dataset_id, 
                    "❌ Failed", 
                    "", 
                    "", 
                    "", 
                    result.get("error", "Unknown error")
                ])
        
        print("\n=== DUCKDB LOADING RESULTS ===")
        print(tabulate(
            duckdb_table_data, 
            headers=["Dataset ID", "Status", "Title", "DuckDB Table", "Row Count", "Time/Error"],
            tablefmt="grid"
        ))
        
        # Convert to Trino Iceberg if requested
        if args.to_trino:
            logger.info("Converting DuckDB tables to Trino Iceberg format")
            
            # Convert each successful DuckDB table to Trino
            iceberg_results = {}
            for dataset_id, result in duckdb_results.items():
                if result.get("success", False):
                    try:
                        logger.info(f"Converting {dataset_id} to Trino Iceberg format")
                        start_time = time.time()
                        
                        # Use the dataset_id as both the DuckDB table name and the Iceberg table name
                        iceberg_result = convert_duckdb_to_iceberg(
                            duckdb_table_name=dataset_id,
                            iceberg_schema="nyc_open_data",
                            iceberg_table=dataset_id
                        )
                        
                        end_time = time.time()
                        
                        if iceberg_result.get("success", False):
                            iceberg_results[dataset_id] = {
                                "success": True,
                                "duckdb_table": dataset_id,
                                "iceberg_schema": "nyc_open_data",
                                "iceberg_table": dataset_id,
                                "time_taken": f"{end_time - start_time:.2f}s"
                            }
                        else:
                            iceberg_results[dataset_id] = {
                                "success": False,
                                "error": iceberg_result.get("error", "Unknown error during conversion")
                            }
                    except Exception as e:
                        logger.error(f"Error converting {dataset_id} to Iceberg: {str(e)}")
                        iceberg_results[dataset_id] = {
                            "success": False,
                            "error": str(e)
                        }
            
            # Display Iceberg conversion results
            iceberg_table_data = []
            for dataset_id, result in iceberg_results.items():
                if result.get("success", False):
                    iceberg_table_data.append([
                        dataset_id, 
                        "✅ Success", 
                        result.get("duckdb_table", ""),
                        result.get("iceberg_schema", ""),
                        result.get("iceberg_table", ""),
                        result.get("time_taken", "")
                    ])
                else:
                    iceberg_table_data.append([
                        dataset_id, 
                        "❌ Failed", 
                        "", 
                        "", 
                        "", 
                        result.get("error", "Unknown error")
                    ])
            
            print("\n=== ICEBERG CONVERSION RESULTS ===")
            print(tabulate(
                iceberg_table_data, 
                headers=["Dataset ID", "Status", "DuckDB Table", "Iceberg Schema", "Iceberg Table", "Time/Error"],
                tablefmt="grid"
            ))
    else:
        # Load datasets and validate
        logger.info(f"Loading and validating {len(dataset_ids)} datasets")
        load_results = load_datasets(
            dataset_ids, 
            quiet=args.quiet,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        validation_results = validate_datasets(etl, dataset_ids)
        display_results(load_results, validation_results)
    
    logger.info("Process completed")

if __name__ == "__main__":
    main() 