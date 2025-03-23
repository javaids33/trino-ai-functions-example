"""
DEPRECATED: Backward compatibility module for env_config
This module imports from app_config and provides the old interface
to maintain compatibility with existing code.

Please update your imports to use app_config.get_config() directly.
"""

from app_config import get_config
import warnings

# Emit a deprecation warning
warnings.warn(
    "env_config is deprecated. Please use app_config.get_config() instead.",
    DeprecationWarning,
    stacklevel=2
)

# Constants for backward compatibility
DEFAULT_DOMAIN = 'data.cityofnewyork.us'

def get_socrata_credentials():
    """Get Socrata API credentials - DEPRECATED"""
    config = get_config()
    return config.get_socrata_credentials()

def get_minio_credentials():
    """Get MinIO configuration - DEPRECATED"""
    config = get_config()
    return config.get_minio_credentials()

def get_trino_credentials():
    """Get Trino connection configuration - DEPRECATED"""
    config = get_config()
    return config.get_trino_credentials()
    
def get_duckdb_path():
    """Get DuckDB path - DEPRECATED"""
    # This is deprecated but included for backward compatibility
    config = get_config()
    return config.data_dir + "/duckdb/nyc_data.duckdb"
    
def check_required_env_vars():
    """Check required environment variables - DEPRECATED"""
    config = get_config()
    return config.validate_required_configs() 