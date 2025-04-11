import os
import logging
import sys
from services.socrates_client import SocratesClient
from services.trino_client import TrinoClient
from services.data_loader import DataLoader
from scripts.generate_dbt_models import DbtModelGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_transform(dataset_id: str):
    """Load a dataset and generate dbt models"""
    try:
        # Initialize clients
        socrates_client = SocratesClient()
        trino_client = TrinoClient()
        data_loader = DataLoader(socrates_client, trino_client)
        
        # Load dataset
        logger.info(f"Loading dataset {dataset_id}")
        data_loader.load_dataset(dataset_id)
        
        # Generate dbt models
        logger.info(f"Generating dbt models for dataset {dataset_id}")
        model_generator = DbtModelGenerator()
        model_generator.generate_models(dataset_id)
        
        logger.info(f"Successfully completed loading and transformation for dataset {dataset_id}")
        
    except Exception as e:
        logger.error(f"Error in load_and_transform for dataset {dataset_id}: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_and_transform.py <dataset_id>")
        sys.exit(1)
    
    dataset_id = sys.argv[1]
    load_and_transform(dataset_id) 