# API package initialization
from flask_restx import Api

# Create API instance
api = Api(
    version='1.0',
    title='NYC Open Data Loader API',
    description='API for loading, managing, and querying NYC Open Data datasets',
    doc='/docs'
)

# Import namespaces to ensure they're registered
from api.datasets import api as datasets_ns
from api.popular import api as popular_ns
from api.metadata import api as metadata_ns
from api.management import api as management_ns 