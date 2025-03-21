import os
from env_config import get_socrata_credentials, get_minio_credentials, get_trino_credentials
from logger_config import setup_logger

logger = setup_logger(__name__)

def check_environment():
    """Check if environment variables are loaded properly"""
    # Check Socrata credentials
    socrata_creds = get_socrata_credentials()
    logger.info(f"Socrata API Key ID: {socrata_creds['key_id'][:5]}..." if socrata_creds['key_id'] else "Socrata API Key ID: Not set")
    logger.info(f"Socrata API Key Secret: {'SET' if socrata_creds['key_secret'] else 'NOT SET'}")
    logger.info(f"Socrata Domain: {socrata_creds['domain']}")
    
    # Check MinIO credentials
    minio_creds = get_minio_credentials()
    logger.info(f"MinIO Endpoint: {minio_creds['endpoint']}")
    logger.info(f"MinIO Access Key: {'SET' if minio_creds['access_key'] else 'NOT SET'}")
    
    # Check Trino credentials
    trino_creds = get_trino_credentials()
    logger.info(f"Trino Host: {trino_creds['host']}:{trino_creds['port']}")
    logger.info(f"Trino Catalog: {trino_creds['catalog']}")
    
    # Check other key environment variables
    logger.info(f"PORT: {os.getenv('PORT', 'Not set')}")
    logger.info(f"DEBUG: {os.getenv('DEBUG', 'Not set')}")
    logger.info(f"CACHE_DIR: {os.getenv('CACHE_DIR', 'Not set')}")
    logger.info(f"DUCKDB_PATH: {os.getenv('DUCKDB_PATH', 'Not set')}")

if __name__ == "__main__":
    check_environment() 