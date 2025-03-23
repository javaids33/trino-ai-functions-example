import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from logger_config import setup_logger

logger = setup_logger(__name__)

class AppConfig:
    """Centralized application configuration for the data loader application"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._load_configuration()
        return cls._instance
    
    def _load_configuration(self):
        """Load application configuration from environment variables"""
        # Try multiple possible .env file locations
        env_paths = [
            Path('.env'),
            Path(__file__).parent / '.env',
            Path(__file__).parent.parent / '.env',
            Path('/app/.env')
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
            # Continue anyway, will use environment variables or defaults
        
        # Application settings
        self.debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        self.port = int(os.environ.get('PORT', 5000))
        self.log_level = os.environ.get('LOG_LEVEL', 'INFO')
        self.env = os.environ.get('ENVIRONMENT', 'development')
        
        # Data paths
        self.cache_dir = os.environ.get('CACHE_DIR', 'data_cache')
        self.temp_dir = os.environ.get('TEMP_DIR', 'temp')
        self.data_dir = os.environ.get('DATA_DIR', 'data')
        self.logs_dir = os.environ.get('LOGS_DIR', 'logs')
        
        # Create necessary directories
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Socrata API
        self.socrata_app_token = os.environ.get('SOCRATA_APP_TOKEN', '')
        self.socrata_api_key_id = os.environ.get('SOCRATA_API_KEY_ID', '')
        self.socrata_api_key_secret = os.environ.get('SOCRATA_API_KEY_SECRET', '')
        self.socrata_domain = os.environ.get('SOCRATA_DOMAIN', 'data.cityofnewyork.us')
        self.socrata_timeout = int(os.environ.get('SOCRATA_TIMEOUT', 60))
        self.socrata_batch_size = int(os.environ.get('SOCRATA_BATCH_SIZE', 10000))
        
        # MinIO configuration
        self.minio_endpoint = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
        self.minio_access_key = os.environ.get('MINIO_ACCESS_KEY', 'admin') 
        self.minio_secret_key = os.environ.get('MINIO_SECRET_KEY', 'password')
        self.minio_secure = os.environ.get('MINIO_SECURE', 'False').lower() == 'true'
        self.minio_bucket = os.environ.get('MINIO_BUCKET', 'iceberg')
        
        # Trino configuration
        self.trino_host = os.environ.get('TRINO_HOST', 'trino')
        self.trino_port = int(os.environ.get('TRINO_PORT', 8080))
        self.trino_user = os.environ.get('TRINO_USER', 'admin')
        self.trino_password = os.environ.get('TRINO_PASSWORD', '')
        self.trino_catalog = os.environ.get('TRINO_CATALOG', 'iceberg')
        self.trino_schema = os.environ.get('TRINO_SCHEMA', 'iceberg')
        self.trino_session_props = os.environ.get('TRINO_SESSION_PROPS', '{}')
        
        # Rate limiting
        self.rate_limit_enabled = os.environ.get('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
        self.rate_limit_per_minute = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 60))
        
        # ETL Parameters
        self.max_concurrent_loads = int(os.environ.get('MAX_CONCURRENT_LOADS', 3))
        self.retry_attempts = int(os.environ.get('RETRY_ATTEMPTS', 3))
        self.retry_delay = int(os.environ.get('RETRY_DELAY', 5))
        
        # Performance tuning
        self.chunk_size = int(os.environ.get('CHUNK_SIZE', 100000))
        self.connection_pool_size = int(os.environ.get('CONNECTION_POOL_SIZE', 5))
        
        # Log configuration summary
        logger.info(f"Application configured with cache_dir={self.cache_dir}, "
                   f"env={self.env}, debug={self.debug}")
    
    def get_socrata_credentials(self):
        """Get Socrata API credentials"""
        return {
            'app_token': self.socrata_app_token,
            'key_id': self.socrata_api_key_id,
            'key_secret': self.socrata_api_key_secret,
            'domain': self.socrata_domain,
            'timeout': self.socrata_timeout
        }
    
    def get_minio_credentials(self):
        """Get MinIO configuration"""
        return {
            'endpoint': self.minio_endpoint,
            'access_key': self.minio_access_key,
            'secret_key': self.minio_secret_key,
            'secure': self.minio_secure,
            'bucket': self.minio_bucket
        }
    
    def get_trino_credentials(self):
        """Get Trino connection configuration"""
        return {
            'host': self.trino_host,
            'port': self.trino_port,
            'user': self.trino_user,
            'password': self.trino_password,
            'catalog': self.trino_catalog,
            'schema': self.trino_schema
        }
    
    def validate_required_configs(self):
        """Validate configuration values (simplified)"""
        missing_vars = []
        
        # Only check essential connection params
        if not self.trino_host:
            missing_vars.append('TRINO_HOST')
        if not self.minio_endpoint:
            missing_vars.append('MINIO_ENDPOINT')
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        return True

# Singleton instance for global access
def get_config() -> AppConfig:
    """Get the singleton AppConfig instance"""
    return AppConfig() 