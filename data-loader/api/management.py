from flask import request
from flask_restx import Namespace, Resource, fields
from cleanup_utility import cleanup_temp_files, list_unused_datasets, remove_dataset
from cache_manager import DatasetCacheManager
from logger_config import setup_logger

# Make sure we import these directly
import trino_connector
import minio_helper
import os
import time

# Set up logger
logger = setup_logger(__name__)

# Initialize namespace
api = Namespace('management', description='Dataset management operations')

# Models for request/response
cleanup_response = api.model('CleanupResponse', {
    'success': fields.Boolean(description='Whether the cleanup succeeded', example=True),
    'removed_count': fields.Integer(description='Number of temporary files removed', example=15)
})

unused_dataset = api.model('UnusedDataset', {
    'dataset_id': fields.String(description='Dataset ID', example='abc1-xyz2'),
    'schema_name': fields.String(description='Schema name', example='nyc_open_data'),
    'table_name': fields.String(description='Table name', example='parking_violations'),
    'row_count': fields.Integer(description='Number of rows', example=1250000),
    'etl_timestamp': fields.String(description='When the dataset was loaded', 
                                   example='2023-06-15T14:30:22'),
    'last_updated': fields.String(description='When the dataset was last updated', 
                                 example='2023-06-01T09:15:47')
})

unused_response = api.model('UnusedResponse', {
    'datasets': fields.List(fields.Nested(unused_dataset), 
                           description='List of unused datasets'),
    'total': fields.Integer(description='Total number of unused datasets', example=5)
})

remove_request = api.model('RemoveRequest', {
    'dataset_id': fields.String(required=True, description='Dataset ID', example='abc1-xyz2'),
    'schema_name': fields.String(required=True, description='Schema name', example='nyc_open_data'),
    'table_name': fields.String(required=True, description='Table name', example='parking_violations')
})

export_request = api.model('ExportRequest', {
    'output_path': fields.String(required=True, 
                                description='Output path for export archive', 
                                example='./exports/nyc_datasets_2023-08-15.zip'),
    'dataset_ids': fields.List(fields.String, 
                              description='List of dataset IDs to export', 
                              example=['abc1-xyz2', 'def3-uvw4', 'ghi5-rst6'])
})

import_request = api.model('ImportRequest', {
    'archive_path': fields.String(required=True, 
                                 description='Path to the archive', 
                                 example='./exports/nyc_datasets_2023-08-15.zip'),
    'overwrite': fields.Boolean(description='Whether to overwrite existing datasets', 
                               example=False)
})

# Add this model for the reset response
reset_response = api.model('ResetResponse', {
    'success': fields.Boolean(description='Whether the reset operation succeeded', example=True),
    'trino_tables_removed': fields.Integer(description='Number of Trino tables removed', example=12),
    'minio_objects_removed': fields.Integer(description='Number of MinIO objects removed', example=45),
    'cache_entries_removed': fields.Integer(description='Number of cache entries cleared', example=10),
    'message': fields.String(description='Additional details about the reset operation', 
                            example='System successfully reset')
})

# Add your API endpoints below
@api.route('/cleanup')
class CleanupTemp(Resource):
    @api.doc('cleanup_temp_files')
    @api.marshal_with(cleanup_response)
    def post(self):
        """Clean up temporary files created during ETL process"""
        try:
            removed_count = cleanup_temp_files()
            return {
                'success': True,
                'removed_count': removed_count
            }, 200
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
            return {
                'success': False,
                'removed_count': 0,
                'error': str(e)
            }, 500 

@api.route('/reset-system')
class ResetSystem(Resource):
    @api.doc('reset_system', 
             description='WARNING: This will delete ALL data in Trino and MinIO. This operation cannot be undone.')
    @api.marshal_with(reset_response)
    def post(self):
        """Reset the entire system by deleting all Trino tables and MinIO objects"""
        logger.warning("Initiating complete system reset - wiping all Trino tables and MinIO objects")
        
        result = {
            'success': True,
            'trino_tables_removed': 0,
            'minio_objects_removed': 0,
            'cache_entries_removed': 0,
            'message': 'System reset completed successfully'
        }
        
        try:
            # 1. Reset Trino tables
            try:
                trino = trino_connector.get_trino_client()
                removed_tables = trino.drop_all_tables()
                result['trino_tables_removed'] = removed_tables
                logger.info(f"Removed {removed_tables} tables from Trino")
            except Exception as e:
                logger.error(f"Error dropping Trino tables: {str(e)}")
                result['message'] += f" (Trino reset failed: {str(e)})"
                # Continue with other cleanup steps even if Trino fails
            
            # 2. Reset MinIO buckets
            try:
                minio = minio_helper.get_minio_client()
                removed_objects = minio.remove_all_objects()
                result['minio_objects_removed'] = removed_objects
                logger.info(f"Removed {removed_objects} objects from MinIO")
            except Exception as e:
                logger.error(f"Error removing MinIO objects: {str(e)}")
                result['message'] += f" (MinIO reset failed: {str(e)})"
            
            # 3. Clear the dataset cache
            try:
                cache_manager = DatasetCacheManager()
                cache_entries = len(cache_manager.get_all_cached_datasets())
                cache_manager.clear_cache()
                result['cache_entries_removed'] = cache_entries
                logger.info(f"Cleared {cache_entries} entries from cache")
            except Exception as e:
                logger.error(f"Error clearing cache: {str(e)}")
                result['message'] += f" (Cache reset failed: {str(e)})"
            
            # 4. Clean up any temporary files
            try:
                cleanup_temp_files()
                logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.error(f"Error cleaning temp files: {str(e)}")
            
            # Set success based on whether all operations completed
            if "failed" in result['message']:
                result['success'] = False
            
            return result, 200
            
        except Exception as e:
            logger.error(f"Error during system reset: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            result['success'] = False
            result['message'] = f"Reset failed: {str(e)}"
            return result, 500 