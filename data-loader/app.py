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

# Run environment check at startup
check_environment()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug) 