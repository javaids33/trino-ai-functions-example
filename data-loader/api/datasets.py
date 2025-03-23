from flask import request
from flask_restx import Namespace, Resource, fields
from socrata_loader import SocrataLoader
from app_config import get_config
from logger_config import setup_logger
import os
import time
from trino_connector import get_trino_connection_manager
from cache_manager import DatasetCacheManager
from data_loader import DataLoader
from simple_loader import load_dataset

# Set up logger
logger = setup_logger(__name__)
config = get_config()

# Initialize namespace BEFORE defining models
api = Namespace('datasets', description='Dataset operations')

# Models for request/response
dataset_request = api.model('DatasetRequest', {
    'dataset_id': fields.String(required=True, description='Socrata dataset ID', 
                               example='2bnn-yakx', 
                               pattern='^[a-z0-9]{4}-[a-z0-9]{4}$'),
    'force_reload': fields.Boolean(required=False, default=False, 
                                  description='Force reload even if dataset is already cached',
                                  example=False)
})

dataset_response = api.model('DatasetResponse', {
    'success': fields.Boolean(description='Operation success status', example=True),
    'dataset_id': fields.String(description='Socrata dataset ID', example='2bnn-yakx'),
    'name': fields.String(description='Dataset name', example='NYC 311 Service Requests'),
    'row_count': fields.Integer(description='Number of rows loaded', example=15230),
    'file_size': fields.Integer(description='Parquet file size in bytes', example=2560834),
    'schema_name': fields.String(description='Schema name in Trino', example='nyc_open_data'),
    'table_name': fields.String(description='Table name in Trino', example='service_requests_311'),
    'columns': fields.List(fields.String, description='Column names in the dataset', 
                          example=['request_id', 'created_date', 'status', 'agency', 'location']),
    'message': fields.String(description='Additional information', example='Dataset loaded successfully'),
    'load_time_seconds': fields.Float(description='Time taken to load the dataset in seconds', example=12.45)
})

@api.route('/load')
class LoadDataset(Resource):
    @api.expect(dataset_request)
    @api.marshal_with(dataset_response, code=200, description='Dataset loaded successfully')
    @api.response(400, 'Invalid request parameters')
    @api.response(500, 'Error loading dataset')
    def post(self):
        """Load a dataset from Socrata to Trino with Iceberg"""
        # Parse request parameters
        dataset_id = request.json.get('dataset_id')
        force_reload = request.json.get('force_reload', False)
        
        if not dataset_id:
            return {"success": False, "message": "Missing dataset_id"}, 400
            
        logger.info(f"API request to load dataset {dataset_id}, force_reload={force_reload}")
        
        # Use the simplified loader
        result = load_dataset(dataset_id, force_reload)
        
        if result.get('success', False):
            return result, 200
        else:
            return result, 500

@api.route('/list')
class ListDatasets(Resource):
    @api.doc('list_datasets',
             params={'limit': 'Number of datasets to return (default: 10)',
                    'category': 'Category to filter by',
                    'offset': 'Offset for pagination (default: 0)'})
    def get(self):
        """List available datasets from Socrata"""
        limit = request.args.get('limit', 10, type=int)
        category = request.args.get('category', None)
        offset = request.args.get('offset', 0, type=int)
        
        try:
            # Initialize Socrata loader
            loader = SocrataLoader()
            
            # Discover datasets
            datasets = loader.discover_datasets(
                limit=limit,
                offset=offset,
                domain_category=category
            )
            
            # Format the response
            result = []
            for dataset in datasets:
                result.append({
                    'dataset_id': dataset.get('resource', {}).get('id'),
                    'name': dataset.get('resource', {}).get('name'),
                    'category': dataset.get('classification', {}).get('domain_category'),
                    'description': dataset.get('resource', {}).get('description'),
                    'created_at': dataset.get('resource', {}).get('createdAt'),
                    'updated_at': dataset.get('resource', {}).get('updatedAt'),
                    'metadata': {
                        'row_count': dataset.get('resource', {}).get('estimated_row_count', 0),
                        'columns': len(dataset.get('resource', {}).get('columns_field_name', [])),
                        'domain': dataset.get('metadata', {}).get('domain')
                    }
                })
            
            return {
                'success': True,
                'total': len(result),
                'datasets': result
            }
            
        except Exception as e:
            logger.error(f"Error listing datasets: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }, 500
