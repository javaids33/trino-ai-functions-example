import os
from pathlib import Path
from dotenv import load_dotenv
import sys
from flask import Flask, Blueprint, jsonify, request
from flask_restx import Api
# Import the namespaces directly from their respective modules
from api.datasets import api as datasets_ns
from api.popular import api as popular_ns
from api.metadata import api as metadata_ns
from api.management import api as management_ns
from logger_config import setup_logger
from check_env import check_environment
from datetime import datetime

# Set up logger
logger = setup_logger(__name__)

# Set up proper path for imports and .env loading
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Explicitly load .env before any other imports
env_path = current_dir / '.env'
if env_path.exists():
    print(f"Loading environment from: {env_path}")
    load_dotenv(dotenv_path=str(env_path))
else:
    print(f"Warning: No .env file found at {env_path}")
    # Try parent directory
    parent_env_path = current_dir.parent / '.env'
    if parent_env_path.exists():
        print(f"Loading environment from parent directory: {parent_env_path}")
        load_dotenv(dotenv_path=str(parent_env_path))

# Print some debug info about environment variables
print(f"SOCRATA_API_KEY_ID: {'SET' if os.environ.get('SOCRATA_API_KEY_ID') else 'NOT SET'}")
print(f"MINIO_ENDPOINT: {os.environ.get('MINIO_ENDPOINT', 'NOT SET')}")

# Initialize Flask app
app = Flask(__name__)

# Configure application
app.config.from_object('config.Config')

# Create Blueprint for API
blueprint = Blueprint('api', __name__, url_prefix='')

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
api.add_namespace(datasets_ns, path='/api/datasets')
api.add_namespace(popular_ns, path='/api/popular')
api.add_namespace(metadata_ns, path='/api/metadata')
api.add_namespace(management_ns, path='/api/management')

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/system-status')
def system_status():
    """Comprehensive system status endpoint"""
    try:
        from etl_tracker import get_job_tracker
        
        # Check database connections
        trino_status = {"connected": False}
        minio_status = {"connected": False}
        duckdb_status = {"connected": False}
        
        # Check Trino connection
        try:
            from env_config import get_trino_credentials
            from trino.dbapi import connect
            
            trino_creds = get_trino_credentials()
            conn = connect(
                host=trino_creds['host'],
                port=int(trino_creds['port']),
                user=trino_creds['user'],
                catalog=trino_creds['catalog']
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            trino_status = {
                "connected": True,
                "host": trino_creds['host'],
                "port": trino_creds['port']
            }
            
            cursor.close()
            conn.close()
        except Exception as e:
            trino_status["error"] = str(e)
        
        # Check MinIO connection
        try:
            from env_config import get_minio_credentials
            from minio import Minio
            
            minio_creds = get_minio_credentials()
            minio_client = Minio(
                minio_creds['endpoint'],
                access_key=minio_creds['access_key'],
                secret_key=minio_creds['secret_key'],
                secure=minio_creds['secure']
            )
            
            # List buckets as a connectivity test
            buckets = minio_client.list_buckets()
            
            minio_status = {
                "connected": True,
                "endpoint": minio_creds['endpoint'],
                "buckets": [bucket.name for bucket in buckets]
            }
        except Exception as e:
            minio_status["error"] = str(e)
        
        # Check DuckDB connection
        try:
            import os
            import duckdb
            
            db_path = os.environ.get('DUCKDB_PATH', '/data/duckdb/nyc_data.duckdb')
            
            if os.path.exists(db_path):
                conn = duckdb.connect(db_path)
                conn.execute("SELECT 1")
                
                duckdb_status = {
                    "connected": True,
                    "path": db_path,
                    "exists": True
                }
                
                conn.close()
            else:
                duckdb_status = {
                    "connected": False,
                    "path": db_path,
                    "exists": False
                }
        except Exception as e:
            duckdb_status["error"] = str(e)
        
        # Get ETL job status
        etl_status = get_job_tracker().get_status()
        
        # Get cache stats
        from cache_manager import DatasetCacheManager
        cache_manager = DatasetCacheManager()
        cached_datasets = cache_manager.get_all_cached_datasets()
        
        # Return comprehensive status
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "connections": {
                "trino": trino_status,
                "minio": minio_status,
                "duckdb": duckdb_status
            },
            "etl": {
                "active_jobs": etl_status["active_jobs"],
                "completed_jobs": etl_status["completed_jobs"],
                "failed_jobs": etl_status["failed_jobs"]
            },
            "cache": {
                "datasets": len(cached_datasets)
            }
        })
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "error": str(e)
        }), 500

# Run environment check at startup
check_environment()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug) 