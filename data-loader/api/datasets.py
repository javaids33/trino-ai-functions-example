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
    'table_type': fields.String(description='Type of table (Iceberg/Hive)', example='ICEBERG'),
    'message': fields.String(description='Additional information', example='Dataset loaded successfully'),
    'load_time_seconds': fields.Float(description='Time taken to load the dataset in seconds', example=12.45)
}) 