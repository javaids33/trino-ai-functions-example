import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # API settings
    RESTX_MASK_SWAGGER = False
    SWAGGER_UI_DOC_EXPANSION = 'list'
    
    # Cache settings
    CACHE_DIR = os.environ.get('CACHE_DIR', 'data_cache')
    
    # Database paths
    DUCKDB_PATH = os.environ.get('DUCKDB_PATH', '/data/duckdb/nyc_data.duckdb') 