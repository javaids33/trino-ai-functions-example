from flask import request, send_file
from flask_restx import Namespace, Resource, fields
from dataset_report import generate_dataset_report
from cache_manager import DatasetCacheManager
import pandas as pd
import json
import os
from io import BytesIO
from logger_config import setup_logger

logger = setup_logger(__name__)

# Initialize namespace
api = Namespace('metadata', description='Dataset metadata operations')

# Models for request/response
report_params = api.model('ReportParams', {
    'save_csv': fields.Boolean(description='Save report as CSV'),
    'save_plots': fields.Boolean(description='Generate visualization plots')
})

cache_info = api.model('CacheInfo', {
    'dataset_id': fields.String(description='Dataset ID'),
    'dataset_title': fields.String(description='Dataset title'),
    'row_count': fields.Integer(description='Number of rows'),
    'cached_at': fields.String(description='Cache timestamp'),
    'file_size': fields.Integer(description='File size in bytes'),
    'file_path': fields.String(description='Path to cached file')
})

cache_response = api.model('CacheResponse', {
    'datasets': fields.List(fields.Nested(cache_info)),
    'total': fields.Integer(description='Total number of cached datasets')
})

@api.route('/report')
class DatasetReport(Resource):
    @api.doc('generate_report')
    @api.expect(report_params)
    def get(self):
        """Generate a comprehensive report on all datasets"""
        args = request.args
        save_csv = args.get('save_csv', 'true').lower() == 'true'
        save_plots = args.get('save_plots', 'true').lower() == 'true'
        
        # Generate the report
        report_df = generate_dataset_report(save_csv=save_csv, save_plots=save_plots)
        
        # Convert report to JSON for API response
        report_json = json.loads(report_df.to_json(orient='records'))
        
        return {
            'report': report_json,
            'total_datasets': len(report_json)
        }, 200

@api.route('/report/csv')
class DatasetReportCSV(Resource):
    @api.doc('download_report_csv')
    def get(self):
        """Download dataset report as CSV"""
        # Generate report without saving to disk
        report_df = generate_dataset_report(save_csv=False, save_plots=False)
        
        # Create in-memory CSV
        csv_data = BytesIO()
        report_df.to_csv(csv_data, index=False)
        csv_data.seek(0)
        
        return send_file(
            csv_data,
            mimetype='text/csv',
            as_attachment=True,
            download_name='dataset_report.csv'
        )

@api.route('/cache')
class CachedDatasets(Resource):
    @api.doc('get_cached_datasets')
    @api.marshal_with(cache_response)
    def get(self):
        """Get information about all cached datasets"""
        cache_manager = DatasetCacheManager()
        datasets = cache_manager.get_all_cached_datasets()
        
        return {
            'datasets': datasets,
            'total': len(datasets)
        }, 200 