def _load_to_trino(self, dataset_id, parquet_path, column_definitions, description=''):
    # Create the destination table
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS iceberg.socrata_{dataset_id} (
            {', '.join([f'"{col}" {col_type}' for col, col_type in column_definitions])}
        )
        COMMENT '{description}'
        """
    self.trino.execute_query(create_sql)
    
    # Create temporary external table pointing to the Parquet file
    create_temp_sql = f"""
        CREATE TABLE IF NOT EXISTS iceberg.temp_{dataset_id} (
            {', '.join([f'"{col}" {col_type}' for col, col_type in column_definitions])}
        )
        WITH (
            external_location = '{parquet_path}',
            format = 'PARQUET'
        )
        """
    try:
        self.trino.execute_query(create_temp_sql)
        
        # Insert from temporary table into the destination table
        insert_sql = f"""
            INSERT INTO iceberg.socrata_{dataset_id}
            SELECT * FROM iceberg.temp_{dataset_id}
            """
        self.trino.execute_query(insert_sql)
        
        # Clean up temporary table
        self.trino.execute_query(f"DROP TABLE iceberg.temp_{dataset_id}")
        
    except Exception as e:
        self.logger.error(f"Error loading dataset {dataset_id} to Trino: {e}")
        raise 