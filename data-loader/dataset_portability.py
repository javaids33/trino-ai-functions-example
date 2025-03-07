#!/usr/bin/env python3
"""
Dataset Portability Tool

This script provides functionality to export and import datasets between machines.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from logger_config import setup_logger
from cache_manager import DatasetCacheManager
from socrata_loader import SocrataToTrinoETL

# Set up logger
logger = setup_logger(__name__)

def export_datasets(output_path: str = None, dataset_ids: list = None):
    """Export datasets to a portable archive"""
    try:
        if not output_path:
            output_path = f"nyc_data_export_{datetime.now().strftime('%Y%m%d')}"
            
        # Initialize cache manager
        cache_manager = DatasetCacheManager()
        
        # Get list of datasets to export
        if not dataset_ids:
            cached_datasets = cache_manager.get_all_cached_datasets()
            dataset_ids = [d["dataset_id"] for d in cached_datasets]
            
        if not dataset_ids:
            logger.error("No datasets to export")
            return False
            
        logger.info(f"Exporting {len(dataset_ids)} datasets to {output_path}")
        
        # Create the archive
        archive_path = cache_manager.create_dataset_archive(output_path, dataset_ids)
        
        logger.info(f"Successfully exported datasets to {archive_path}")
        print(f"\nDatasets exported to: {archive_path}")
        print(f"Total datasets: {len(dataset_ids)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error exporting datasets: {e}")
        return False

def import_datasets(archive_path: str, overwrite: bool = False):
    """Import datasets from a portable archive"""
    try:
        if not os.path.exists(archive_path):
            logger.error(f"Archive not found: {archive_path}")
            return False
            
        # Initialize cache manager
        cache_manager = DatasetCacheManager()
        
        logger.info(f"Importing datasets from {archive_path}")
        
        # Import the archive
        imported_datasets = cache_manager.import_dataset_archive(archive_path, overwrite)
        
        if not imported_datasets:
            logger.error("No datasets were imported")
            return False
            
        logger.info(f"Successfully imported {len(imported_datasets)} datasets")
        print(f"\nImported {len(imported_datasets)} datasets from {archive_path}")
        
        # Initialize ETL to update Trino tables for imported datasets
        etl = SocrataToTrinoETL()
        
        print("\nUpdating Trino tables for imported datasets:")
        for dataset_id in imported_datasets:
            logger.info(f"Creating Trino table for dataset {dataset_id}")
            result = etl.create_trino_table_from_dataset(dataset_id)
            
            if result.get("success", False):
                print(f"  ✓ {dataset_id}: {result.get('name', dataset_id)}")
            else:
                print(f"  ✗ {dataset_id}: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error importing datasets: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dataset Portability Tool")
    
    # Command options
    parser.add_argument("command", choices=["export", "import"], help="Command to run")
    
    # Export options
    parser.add_argument("--output", type=str, help="Output path for export archive")
    parser.add_argument("--datasets", type=str, nargs="+", help="Dataset IDs to export")
    
    # Import options
    parser.add_argument("--archive", type=str, help="Archive file to import")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing datasets when importing")
    
    args = parser.parse_args()
    
    if args.command == "export":
        success = export_datasets(args.output, args.datasets)
    elif args.command == "import":
        if not args.archive:
            parser.error("--archive is required for import command")
        success = import_datasets(args.archive, args.overwrite)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 