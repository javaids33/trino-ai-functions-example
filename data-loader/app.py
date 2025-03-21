import os
from flask import Flask, Blueprint
from flask_restx import Api
# Import the namespaces directly from their respective modules
from api.datasets import api as datasets_ns
from api.popular import api as popular_ns
from api.metadata import api as metadata_ns
from api.management import api as management_ns
from logger_config import setup_logger

# Set up logger
logger = setup_logger(__name__)

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
def health_check():
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 