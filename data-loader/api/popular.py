# api/popular.py
from flask import request
from flask_restx import Namespace, Resource, fields
from socrata_discovery import find_popular_datasets
from data_loader import DataLoader, load_dataset
from logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Create namespace
api = Namespace('popular', description='Popular datasets operations')

# Models for request/response
dataset_load_request = api.model('PopularLoadRequest', {
    'count': fields.Integer(description='Number of popular datasets to load', default=1),
    'force_reload': fields.Boolean(description='Force reload even if dataset is cached', default=False)
})

dataset_response = api.model('PopularDatasetResponse', {
    'dataset_id': fields.String(description='Dataset ID', example='8wbx-tsch'),
    'name': fields.String(description='Dataset name', example='For Hire Vehicles (FHV) - Active'),
    'success': fields.Boolean(description='Whether the dataset was loaded successfully', example=True),
    'row_count': fields.Integer(description='Number of rows loaded', example=15230),
    'message': fields.String(description='Additional information', example='Dataset loaded successfully'),
    'load_time_seconds': fields.Float(description='Time taken to load the dataset in seconds', example=12.45),
    'errors': fields.List(fields.String, description='Any errors that occurred during loading')
})

popular_load_response = api.model('PopularLoadResponse', {
    'datasets': fields.List(fields.Nested(dataset_response), description='List of loaded datasets'),
    'total': fields.Integer(description='Total number of datasets processed'),
    'successful': fields.Integer(description='Number of datasets loaded successfully'),
    'message': fields.String(description='Overall operation status'),
})

@api.route('/load')
class LoadPopularDatasets(Resource):
    @api.expect(dataset_load_request)
    @api.marshal_with(popular_load_response, code=202, description='Datasets loading started')
    def post(self):
        """Load the most popular NYC Open Data datasets into Trino"""
        # Get request parameters
        count = request.json.get('count', 1)
        force_reload = request.json.get('force_reload', False)
        
        # Find the most popular datasets
        logger.info(f"Finding {count} most popular datasets")
        popular_datasets = find_popular_datasets(limit=count)
        
        # Initialize results structure
        results = {
            'datasets': [],
            'total': len(popular_datasets),
            'successful': 0,
            'message': f"Loading {len(popular_datasets)} popular datasets"
        }
        
        # Start loading each dataset
        for dataset in popular_datasets:
            dataset_id = dataset['id']
            dataset_name = dataset['name']
            logger.info(f"Loading dataset {dataset_id}: {dataset_name}")
            
            # Use the modern loader function from data_loader.py
            result = load_dataset(dataset_id, force_reload)
            
            # Add result to the datasets list
            dataset_result = {
                'dataset_id': dataset_id,
                'name': dataset_name,
                'success': result.get('success', False),
                'row_count': result.get('row_count', 0),
                'message': result.get('message', ''),
                'load_time_seconds': result.get('load_time_seconds', 0),
                'errors': result.get('errors', [])
            }
            
            results['datasets'].append(dataset_result)
            if dataset_result['success']:
                results['successful'] += 1
        
        return results, 202

@api.route('/list')
class ListPopularDatasets(Resource):
    @api.doc('list_popular_datasets')
    def get(self):
        """List the most popular NYC Open Data datasets"""
        count = request.args.get('count', 10, type=int)
        
        try:
            popular_datasets = find_popular_datasets(limit=count)
            return {
                'datasets': popular_datasets,
                'total': len(popular_datasets)
            }, 200
        except Exception as e:
            logger.error(f"Error listing popular datasets: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }, 500