import os
from pathlib import Path
from dotenv import load_dotenv
from logger_config import setup_logger

logger = setup_logger(__name__)

class AppConfig:
    """Centralized application configuration"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._load_configuration()
        return cls._instance
    
    def _load_configuration(self):
        """Load application configuration from environment variables"""
        # Load environment variables from .env file
        load_dotenv()
        
        # Application settings
        self.debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        self.port = int(os.environ.get('PORT', 5000))
        self.log_level = os.environ.get('LOG_LEVEL', 'INFO')
        
        # Data paths
        self.cache_dir = os.environ.get('CACHE_DIR', 'data_cache')
        self.temp_dir = os.environ.get('TEMP_DIR', 'temp')
        self.duckdb_path = os.environ.get('DUCKDB_PATH', '/data/duckdb/nyc_data.duckdb')
        
        # Create necessary directories
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.duckdb_path), exist_ok=True)
        
        # Socrata API
        self.socrata_app_token = os.environ.get('SOCRATA_APP_TOKEN', '')
        self.socrata_api_key_id = os.environ.get('SOCRATA_API_KEY_ID', '')
        self.socrata_api_key_secret = os.environ.get('SOCRATA_API_KEY_SECRET', '')
        self.socrata_domain = os.environ.get('SOCRATA_DOMAIN', 'data.cityofnewyork.us')
        
        # MinIO configuration
        self.minio_endpoint = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
        self.minio_access_key = os.environ.get('MINIO_ACCESS_KEY', 'admin') 
        self.minio_secret_key = os.environ.get('MINIO_SECRET_KEY', 'password')
        self.minio_secure = os.environ.get('MINIO_SECURE', 'False').lower() == 'true'
        
        # Trino configuration
        self.trino_host = os.environ.get('TRINO_HOST', 'trino')
        self.trino_port = int(os.environ.get('TRINO_PORT', 8080))
        self.trino_user = os.environ.get('TRINO_USER', 'admin')
        self.trino_catalog = os.environ.get('TRINO_CATALOG', 'iceberg')
        self.trino_schema = os.environ.get('TRINO_SCHEMA', 'iceberg')
        
        # Log configuration summary
        logger.info(f"Application configured with cache_dir={self.cache_dir}, debug={self.debug}")
    
    # Convenience methods to get credentials in the format expected by existing code
    def get_socrata_credentials(self):
        """Get Socrata API credentials"""
        return {
            'app_token': self.socrata_app_token,
            'key_id': self.socrata_api_key_id,
            'key_secret': self.socrata_api_key_secret,
            'domain': self.socrata_domain
        }
    
    def get_minio_credentials(self):
        """Get MinIO configuration"""
        return {
            'endpoint': self.minio_endpoint,
            'access_key': self.minio_access_key,
            'secret_key': self.minio_secret_key,
            'secure': self.minio_secure
        }
    
    def get_trino_credentials(self):
        """Get Trino connection configuration"""
        return {
            'host': self.trino_host,
            'port': self.trino_port,
            'user': self.trino_user,
            'catalog': self.trino_catalog,
            'schema': self.trino_schema
        }

# Singleton instance for global access
def get_config() -> AppConfig:
    """Get the singleton AppConfig instance"""
    return AppConfig() 