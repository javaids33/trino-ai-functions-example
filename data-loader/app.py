import os
from pathlib import Path
import sys
from flask import Flask, Blueprint, jsonify, request
from flask_restx import Api
from flask_cors import CORS
try:
    # Import the namespaces directly from their respective modules
    from api.datasets import api as datasets_ns
    from api.popular import api as popular_ns
    from api.metadata import api as metadata_ns
    from api.management import api as management_ns
except ImportError as e:
    print(f"Error importing API modules: {e}")
    sys.exit(1)

from logger_config import setup_logger
from app_config import get_config
from trino_connector import get_trino_connection_manager
from minio_helper import get_minio_manager
from datetime import datetime

# Set up logger
logger = setup_logger(__name__)
logger.info("Starting NYC Data Loader application")

# Initialize Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Load application configuration
try:
    config = get_config()
    logger.info(f"Loaded configuration for environment: {config.env}")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    sys.exit(1)

# Validate environment variables
if not config.validate_required_configs():
    logger.error("Missing required environment variables. Check the logs for details.")
    logger.error("The application may not function correctly.")

# Configure application
app.config.from_object('config.Config')

# Create Blueprint for API
blueprint = Blueprint('api', __name__, url_prefix='/api')

# Initialize API with Swagger UI configuration
api = Api(blueprint, 
    version='1.0', 
    title='NYC Data Loader API',
    description='API for loading NYC Open Data into Trino/Iceberg',
    doc='/swagger',
    validate=True,
    # Enhanced Swagger UI configuration
    authorizations={
        'apikey': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-KEY'
        }
    },
    # Improve default formatting
    default_swagger_ui_config={
        'persistAuthorization': True,
        'displayRequestDuration': True,
        'docExpansion': 'list',
        'defaultModelsExpandDepth': 3,
        'defaultModelExpandDepth': 3,
        'tryItOutEnabled': True,
    }
)

app.register_blueprint(blueprint)

# Register namespaces
try:
    api.add_namespace(datasets_ns, path='/datasets')
    api.add_namespace(popular_ns, path='/popular')
    api.add_namespace(metadata_ns, path='/metadata')
    api.add_namespace(management_ns, path='/management')
    logger.info("Registered all API namespaces")
except Exception as e:
    logger.error(f"Error registering API namespaces: {e}")

@app.route('/health')
def health():
    """Basic health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/system-status')
def system_status():
    """Comprehensive system status endpoint"""
    try:
        # Get connection managers
        trino = get_trino_connection_manager()
        minio = get_minio_manager()
        
        # Check database connections
        trino_status = {"connected": False}
        minio_status = {"connected": False}
        
        # Check Trino connection
        try:
            # Try a simple query
            result = trino.execute_query("SELECT 1")
            trino_status = {
                "connected": True,
                "host": config.trino_host,
                "port": config.trino_port,
                "catalog": config.trino_catalog,
                "schema": config.trino_schema
            }
        except Exception as e:
            trino_status["error"] = str(e)
            logger.error(f"Failed to connect to Trino: {e}")
        
        # Check MinIO connection
        try:
            # Check if bucket exists
            bucket_exists = minio.ensure_bucket_exists()
            minio_status = {
                "connected": True,
                "endpoint": config.minio_endpoint,
                "bucket": minio.bucket_name,
                "bucket_exists": bucket_exists
            }
        except Exception as e:
            minio_status["error"] = str(e)
            logger.error(f"Failed to connect to MinIO: {e}")
        
        # Get cached datasets info
        from cache_manager import DatasetCacheManager
        cache_manager = DatasetCacheManager(config.cache_dir)
        cached_datasets = cache_manager.get_all_cached_datasets()
        
        # Assemble status response
        status = {
            "status": "operational" if trino_status["connected"] and minio_status["connected"] else "degraded",
            "timestamp": datetime.now().isoformat(),
            "environment": config.env,
            "version": "1.0.0",
            "connections": {
                "trino": trino_status,
                "minio": minio_status
            },
            "cache": {
                "dataset_count": len(cached_datasets),
                "cache_dir": config.cache_dir
            },
            "configuration": {
                "log_level": config.log_level,
                "debug": config.debug
            }
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/')
def index():
    """Root endpoint with basic API info"""
    return jsonify({
        "name": "NYC Data Loader API",
        "version": "1.0.0",
        "docs": "/api/swagger",
        "health": "/health",
        "status": "/system-status"
    })

def main():
    """Main entry point for the application"""
    port = config.port
    debug = config.debug
    
    logger.info(f"Starting application on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)

if __name__ == '__main__':
    main() 