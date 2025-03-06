import os
from dotenv import load_dotenv
from logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# Socrata API credentials
SOCRATA_APP_TOKEN = os.environ.get('SOCRATA_APP_TOKEN')
SOCRATA_API_KEY_ID = os.environ.get('SOCRATA_API_KEY_ID')
SOCRATA_API_KEY_SECRET = os.environ.get('SOCRATA_API_KEY_SECRET')

# MinIO credentials
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'admin')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', 'password')

# Trino connection
TRINO_HOST = os.environ.get('TRINO_HOST', 'trino')
TRINO_PORT = os.environ.get('TRINO_PORT', '8080')
TRINO_USER = os.environ.get('TRINO_USER', 'admin')
TRINO_CATALOG = os.environ.get('TRINO_CATALOG', 'iceberg')
TRINO_SCHEMA = os.environ.get('TRINO_SCHEMA', 'iceberg')

# Dataset configuration
DEFAULT_DATASET_ID = os.environ.get('DATASET_ID', 'vx8i-nprf')
DEFAULT_DOMAIN = os.environ.get('SOCRATA_DOMAIN', 'data.cityofnewyork.us')

# Logging configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Check if credentials are available
if SOCRATA_API_KEY_ID and SOCRATA_API_KEY_SECRET:
    logger.info("Socrata API credentials found")
else:
    logger.warning("No Socrata API credentials found - rate limits will apply")

def get_socrata_credentials():
    """Get Socrata API credentials"""
    return {
        'app_token': SOCRATA_APP_TOKEN,
        'api_key_id': SOCRATA_API_KEY_ID,
        'api_key_secret': SOCRATA_API_KEY_SECRET
    }

def get_minio_credentials():
    """Get MinIO credentials"""
    return {
        'endpoint': MINIO_ENDPOINT,
        'access_key': MINIO_ACCESS_KEY,
        'secret_key': MINIO_SECRET_KEY
    }

def get_trino_credentials():
    """Get Trino connection details"""
    return {
        'host': TRINO_HOST,
        'port': TRINO_PORT,
        'user': TRINO_USER,
        'catalog': TRINO_CATALOG,
        'schema': TRINO_SCHEMA
    } 