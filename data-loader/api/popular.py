from flask import request
from flask_restx import Namespace, Resource, fields
from socrata_discovery import SocrataDiscovery
from socrata_loader import SocrataToTrinoETL
from logger_config import setup_logger
from popular_datasets import get_popular_datasets, load_popular_datasets

# Set up logger
logger = setup_logger(__name__)

# Initialize namespace
api = Namespace('popular', description='Popular dataset operations')

# Models for request/response
popular_params = api.model('PopularParams', {
    'limit': fields.Integer(description='Maximum number of datasets to return', 
                           example=10, 
                           min=1, 
                           max=100),
    'domain': fields.String(description='Domain to search within', 
                           example='data.cityofnewyork.us',
                           pattern='^[a-zA-Z0-9.-]+$')
})

dataset_info = api.model('DatasetInfo', {
    'dataset_id': fields.String(description='Socrata dataset ID', example='erm2-nwe9'),
    'name': fields.String(description='Dataset name', example='311 Service Requests from 2010 to Present'),
    'description': fields.String(description='Dataset description', 
                               example='All 311 Service Requests from 2010 to the present. This information is automatically updated daily.'),
    'domain': fields.String(description='Domain of the dataset', example='data.cityofnewyork.us'),
    'category': fields.String(description='Dataset category', example='Social Services'),
    'row_count': fields.Integer(description='Number of rows', example=28924452),
    'view_count': fields.Integer(description='Number of views', example=5842157),
    'download_count': fields.Integer(description='Number of downloads', example=234567),
    'last_updated': fields.String(description='Last updated date', example='2023-07-15')
})

popular_response = api.model('PopularResponse', {
    'datasets': fields.List(fields.Nested(dataset_info), description='List of popular datasets'),
    'total': fields.Integer(description='Total number of datasets returned', example=10)
})

load_popular_params = api.model('LoadPopularParams', {
    'limit': fields.Integer(description='Number of popular datasets to load', 
                           example=5, 
                           min=1, 
                           max=20),
    'concurrency': fields.Integer(description='Number of concurrent loads', 
                                 example=3, 
                                 min=1, 
                                 max=10)
})

load_popular_response = api.model('LoadPopularResponse', {
    'success': fields.Boolean(description='Overall operation success status', example=True),
    'datasets_loaded': fields.Integer(description='Number of datasets successfully loaded', example=3),
    'datasets_attempted': fields.Integer(description='Total number of datasets attempted', example=5),
    'details': fields.List(fields.Raw, description='Details for each dataset load attempt')
})

etl_status_response = api.model('ETLStatusResponse', {
    'active_jobs': fields.Integer(description='Number of active ETL jobs', example=2),
    'completed_jobs': fields.Integer(description='Number of completed ETL jobs', example=10),
    'failed_jobs': fields.Integer(description='Number of failed ETL jobs', example=1),
    'details': fields.List(fields.Raw, description='Details of currently running and recent jobs')
})

@api.route('')
class PopularDatasets(Resource):
    @api.doc('get_popular_datasets')
    @api.expect(popular_params, validate=False)
    @api.response(200, 'Success', popular_response)
    @api.response(400, 'Bad Request')
    @api.response(500, 'Internal Server Error')
    def get(self):
        """Get a list of popular datasets from Socrata"""
        try:
            # Parse parameters
            limit = request.args.get('limit', default=20, type=int)
            domain = request.args.get('domain')
            
            # Validate limit
            limit = max(1, min(limit, 100))
            
            logger.info(f"Fetching popular datasets (limit: {limit}, domain: {domain})")
            
            # Use the SocrataDiscovery class to get popular datasets
            discovery = SocrataDiscovery()
            datasets = discovery.get_popular_datasets(limit=limit, domain=domain)
            
            return {
                'datasets': datasets,
                'total': len(datasets)
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting popular datasets: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}, 500

@api.route('/load')
class LoadPopularDatasets(Resource):
    @api.doc('load_popular_datasets')
    @api.expect(load_popular_params)
    @api.response(202, 'Loading Process Started', load_popular_response)
    @api.response(400, 'Bad Request')
    @api.response(500, 'Internal Server Error')
    def post(self):
        """Load popular datasets into DuckDB and Trino"""
        try:
            # Get parameters from request
            data = request.json
            limit = data.get('limit', 5)
            concurrency = data.get('concurrency', 3)
            
            # Validate parameters
            limit = max(1, min(limit, 20))
            concurrency = max(1, min(concurrency, 10))
            
            logger.info(f"Loading {limit} popular datasets with concurrency {concurrency}")
            
            # Initialize ETL process
            etl = SocrataToTrinoETL()
            
            # Get popular datasets
            discovery = SocrataDiscovery()
            datasets = discovery.get_popular_datasets(limit=limit)
            
            if not datasets:
                return {
                    'success': False,
                    'message': 'No popular datasets found',
                    'datasets_loaded': 0,
                    'datasets_attempted': 0,
                    'details': []
                }, 400
            
            # Start the loading process for each dataset
            results = []
            for dataset in datasets:
                dataset_id = dataset.get('dataset_id')
                try:
                    logger.info(f"Loading dataset {dataset_id}: {dataset.get('name')}")
                    result = etl.create_trino_table_from_dataset(dataset_id)
                    
                    # Add result with dataset info
                    success = result.get('success', False)
                    results.append({
                        'dataset_id': dataset_id,
                        'name': dataset.get('name'),
                        'success': success,
                        'row_count': result.get('row_count') if success else None,
                        'schema_name': result.get('schema_name') if success else None,
                        'table_name': result.get('table_name') if success else None,
                        'message': result.get('message', '') if success else result.get('error', 'Unknown error')
                    })
                    
                except Exception as e:
                    logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
                    results.append({
                        'dataset_id': dataset_id,
                        'name': dataset.get('name'),
                        'success': False,
                        'message': str(e)
                    })
            
            # Count successful loads
            successful_loads = sum(1 for r in results if r.get('success', False))
            
            return {
                'success': successful_loads > 0,
                'datasets_loaded': successful_loads,
                'datasets_attempted': len(datasets),
                'details': results
            }, 202
            
        except Exception as e:
            logger.error(f"Error loading popular datasets: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}, 500

@api.route('/status')
class ETLStatus(Resource):
    @api.doc('get_etl_status')
    @api.response(200, 'Success', etl_status_response)
    @api.response(500, 'Internal Server Error')
    def get(self):
        """Get the status of current and recent ETL jobs"""
        try:
            # Get ETL status (mock implementation - replace with actual status tracking)
            # In a real implementation, you would track this in a database or in-memory store
            
            # Mock data
            active_jobs = []
            completed_jobs = []
            failed_jobs = []
            
            # Get status from registry table if possible
            try:
                etl = SocrataToTrinoETL()
                registry_data = etl._get_all_datasets_from_registry()
                
                for entry in registry_data:
                    dataset_info = {
                        'dataset_id': entry.get('dataset_id'),
                        'name': entry.get('dataset_title'),
                        'schema_name': entry.get('schema_name'),
                        'table_name': entry.get('table_name'),
                        'row_count': entry.get('row_count'),
                        'status': 'completed',
                        'timestamp': entry.get('etl_timestamp')
                    }
                    completed_jobs.append(dataset_info)
            except Exception as e:
                logger.error(f"Error getting registry data: {str(e)}")
                # Fall back to empty list if registry access fails
                completed_jobs = []
            
            return {
                'active_jobs': len(active_jobs),
                'completed_jobs': len(completed_jobs),
                'failed_jobs': len(failed_jobs),
                'details': {
                    'active': active_jobs,
                    'completed': completed_jobs,
                    'failed': failed_jobs
                }
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting ETL status: {str(e)}")
            return {'error': str(e)}, 500 