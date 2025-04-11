import os
import json
import logging
import yaml
from typing import Dict, Any, List
from services.socrates_client import SocratesClient
from services.trino_client import TrinoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DbtModelGenerator:
    def __init__(self):
        self.base_url = "https://data.cityofnewyork.us"
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dbt_nyc_data", "models")
        self.staging_dir = os.path.join(self.models_dir, "staging")
        self.marts_dir = os.path.join(self.models_dir, "marts")
        self.sources_dir = os.path.join(self.models_dir, "staging")
        
        # Create directories if they don't exist
        for directory in [self.staging_dir, self.marts_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a dataset from Socrates API"""
        socrates_client = SocratesClient()
        return socrates_client.get_dataset_metadata(dataset_id)

    def generate_staging_model(self, dataset_id: str, metadata: Dict[str, Any]) -> str:
        """Generate a staging model for a dataset"""
        table_name = f"dataset_{dataset_id.replace('-', '_')}"
        columns = []
        
        for col in metadata.get('columns', []):
            col_name = col['name']
            col_type = col.get('type', 'VARCHAR')
            columns.append(f"{{{{ sanitize_column_name('{col_name}') }}}} as {col_name}")
        
        model_content = f"""with source as (
    select * from {{{{ source('nyc_data', '{table_name}') }}}}
),

renamed as (
    select
        {',\n        '.join(columns)}
    from source
)

select * from renamed"""

        return model_content

    def generate_mart_model(self, dataset_id: str, metadata: Dict[str, Any]) -> str:
        """Generate a mart model for a dataset"""
        table_name = f"dataset_{dataset_id.replace('-', '_')}"
        staging_model = f"stg_{table_name}"
        
        # Basic aggregation model
        model_content = f"""with source as (
    select * from {{{{ ref('{staging_model}') }}}}
),

aggregated as (
    select
        count(*) as total_records,
        current_timestamp as last_updated
    from source
)

select * from aggregated"""

        return model_content

    def update_sources_yml(self, dataset_id: str, metadata: Dict[str, Any]):
        """Update the sources.yml file with new dataset information"""
        sources_file = os.path.join(self.sources_dir, "sources.yml")
        
        # Read existing sources.yml if it exists
        if os.path.exists(sources_file):
            with open(sources_file, 'r') as f:
                sources_config = yaml.safe_load(f)
        else:
            sources_config = {
                'version': 2,
                'sources': [{
                    'name': 'nyc_data',
                    'database': 'iceberg',
                    'schema': 'iceberg',
                    'tables': []
                }]
            }

        # Add or update table information
        table_name = f"dataset_{dataset_id.replace('-', '_')}"
        table_info = {
            'name': table_name,
            'description': metadata.get('description', ''),
            'columns': []
        }

        for col in metadata.get('columns', []):
            table_info['columns'].append({
                'name': col['name'],
                'description': col.get('description', '')
            })

        # Check if table already exists in sources
        source = next((s for s in sources_config['sources'] if s['name'] == 'nyc_data'), None)
        if source:
            table = next((t for t in source['tables'] if t['name'] == table_name), None)
            if table:
                # Update existing table
                table.update(table_info)
            else:
                # Add new table
                source['tables'].append(table_info)

        # Write updated sources.yml
        with open(sources_file, 'w') as f:
            yaml.dump(sources_config, f, sort_keys=False)

    def generate_models(self, dataset_id: str):
        """Generate dbt models for a dataset"""
        try:
            logger.info(f"Generating dbt models for dataset {dataset_id}")
            
            # Get dataset metadata
            metadata = self.get_dataset_metadata(dataset_id)
            
            # Generate staging model
            staging_model = self.generate_staging_model(dataset_id, metadata)
            staging_file = os.path.join(self.staging_dir, f"stg_dataset_{dataset_id.replace('-', '_')}.sql")
            with open(staging_file, 'w') as f:
                f.write(staging_model)
            
            # Generate mart model
            mart_model = self.generate_mart_model(dataset_id, metadata)
            mart_file = os.path.join(self.marts_dir, f"mart_dataset_{dataset_id.replace('-', '_')}.sql")
            with open(mart_file, 'w') as f:
                f.write(mart_model)
            
            # Update sources.yml
            self.update_sources_yml(dataset_id, metadata)
            
            logger.info(f"Successfully generated dbt models for dataset {dataset_id}")
            
        except Exception as e:
            logger.error(f"Error generating dbt models for dataset {dataset_id}: {str(e)}")
            raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_dbt_models.py <dataset_id>")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    generator = DbtModelGenerator()
    generator.generate_models(dataset_id) 