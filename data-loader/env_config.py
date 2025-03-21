import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Configure logger first
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Get the directory this file is in
current_dir = Path(__file__).parent.absolute()

# Try multiple possible .env file locations
env_paths = [
    current_dir / '.env',
    current_dir.parent / '.env',
    Path('/app/.env'),
    Path('.env')
]

# Load environment variables from .env file with explicit paths
env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        logger.info(f"Loading environment from: {env_path}")
        load_dotenv(dotenv_path=str(env_path))
        env_loaded = True
        break

if not env_loaded:
    logger.warning(f"No .env file found in any of these locations: {[str(p) for p in env_paths]}")

# Default Socrata domain for NYC Open Data
DEFAULT_DOMAIN = "data.cityofnewyork.us"

def get_minio_credentials():
    """Get MinIO configuration from environment variables."""
    config = {
        'endpoint': os.getenv('MINIO_ENDPOINT', 'minio:9000'),
        'access_key': os.getenv('MINIO_ACCESS_KEY', 'admin'),
        'secret_key': os.getenv('MINIO_SECRET_KEY', 'password'),
        'secure': os.getenv('MINIO_SECURE', 'False').lower() == 'true'
    }
    logger.debug(f"MinIO endpoint: {config['endpoint']}")
    return config

def get_socrata_credentials():
    """Get Socrata API credentials from environment variables."""
    api_key_id = os.getenv('SOCRATA_API_KEY_ID', '')
    api_key_secret = os.getenv('SOCRATA_API_KEY_SECRET', '')
    domain = os.getenv('SOCRATA_DOMAIN', DEFAULT_DOMAIN)
    
    # Log whether credentials were found (without exposing sensitive values)
    if api_key_id and api_key_secret:
        logger.info("Socrata API credentials found")
    else:
        logger.warning("No Socrata API credentials found, using anonymous access (rate limited)")
        logger.debug(f"SOCRATA_API_KEY_ID env var: {'SET' if os.environ.get('SOCRATA_API_KEY_ID') else 'NOT SET'}")
        logger.debug(f"SOCRATA_API_KEY_SECRET env var: {'SET' if os.environ.get('SOCRATA_API_KEY_SECRET') else 'NOT SET'}")
    
    return {
        'domain': domain,
        'key_id': api_key_id,
        'key_secret': api_key_secret
    }

def get_trino_credentials():
    """Get Trino connection configuration from environment variables."""
    config = {
        'host': os.getenv('TRINO_HOST', 'trino'),
        'port': int(os.getenv('TRINO_PORT', '8080')),
        'user': os.getenv('TRINO_USER', 'trino'),
        'catalog': os.getenv('TRINO_CATALOG', 'iceberg'),
        'schema': os.getenv('TRINO_SCHEMA', 'nyc_data')
    }
    logger.debug(f"Trino host: {config['host']}:{config['port']}")
    return config