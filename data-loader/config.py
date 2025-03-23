import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration for the Flask application"""
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # API settings
    RESTX_MASK_SWAGGER = False
    SWAGGER_UI_DOC_EXPANSION = 'list'
    RESTX_VALIDATE = True
    
    # Cache settings
    CACHE_DIR = os.environ.get('CACHE_DIR', 'data_cache')
    
    # CORS settings
    CORS_HEADERS = 'Content-Type'
    
    # Rate limiting
    RATELIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
    RATELIMIT_DEFAULT = os.environ.get('RATE_LIMIT_PER_MINUTE', '60')
    
    # Application settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload size 