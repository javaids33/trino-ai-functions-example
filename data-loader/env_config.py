import os
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Default Socrata domain for NYC Open Data
DEFAULT_DOMAIN = "data.cityofnewyork.us"

def get_minio_credentials():
    """Get MinIO connection credentials from environment variables."""
    config = {
        'endpoint': os.getenv('MINIO_ENDPOINT', 'minio:9000'),  # Use service name instead of localhost
        'access_key': os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
        'secret_key': os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
        'secure': os.getenv('MINIO_SECURE', 'false').lower() == 'true',
        'location': os.getenv('MINIO_LOCATION', '')
    }
    return config

def get_socrata_credentials():
    """Get Socrata API credentials from environment variables."""
    app_token = os.getenv('SOCRATA_APP_TOKEN', '')
    username = os.getenv('SOCRATA_USERNAME', '')
    password = os.getenv('SOCRATA_PASSWORD', '')
    
    # Log whether credentials were found (without exposing sensitive values)
    if app_token or (username and password):
        logger.info("Socrata API credentials found")
    else:
        logger.warning("No Socrata API credentials found, using anonymous access (rate limited)")
    
    return {
        'app_token': app_token,
        'username': username,
        'password': password
    }

def get_trino_credentials():
    """Get Trino connection configuration from environment variables."""
    config = {
        'host': os.getenv('TRINO_HOST', 'trino'),  # Use service name instead of localhost
        'port': int(os.getenv('TRINO_PORT', '8080')),
        'user': os.getenv('TRINO_USER', 'trino'),
        'catalog': os.getenv('TRINO_CATALOG', 'iceberg'),
        'schema': os.getenv('TRINO_SCHEMA', 'nyc_data')
    }
    return config