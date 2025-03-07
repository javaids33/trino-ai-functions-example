-- Check available schemas
SELECT schema_name 
FROM duckdb.information_schema.schemata;

-- View our datasets metadata
SELECT * FROM duckdb.main.datasets;

-- Query one of the datasets (assumes at least one dataset exists)
SELECT * 
FROM duckdb.information_schema.tables 
WHERE table_schema LIKE 'nyc_data_%'
LIMIT 1; 