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
                                 example='2023-03-01T09:15:45'),
    'last_access': fields.String(description='When the dataset was last accessed', 
                                example='2023-03-05T10:22:18'),
    'days_since_access': fields.Integer(description='Days since last access', example=128)
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