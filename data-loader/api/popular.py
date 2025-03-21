from flask import request
from flask_restx import Namespace, Resource, fields
from socrata_discovery import SocrataDiscovery
from socrata_loader import SocrataToTrinoETL
from logger_config import setup_logger

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