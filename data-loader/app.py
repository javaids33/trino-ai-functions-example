import os
import logging
from flask import Flask, jsonify, request
from datetime import datetime
from dotenv import load_dotenv
from services.socrates_client import SocratesClient
from services.trino_client import TrinoClient
from services.data_loader import DataLoader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'data_loader_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize clients
try:
    logger.info("Initializing service clients")
    socrates_client = SocratesClient()
    trino_client = TrinoClient()
    data_loader = DataLoader(socrates_client, trino_client)
    logger.info("Service clients initialized successfully")
except Exception as e:
    logger.error(f"Error initializing service clients: {str(e)}")
    raise

@app.before_request
def log_request_info():
    """Log request information before processing"""
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    if request.is_json:
        logger.info(f"Body: {request.get_json()}")

@app.after_request
def log_response_info(response):
    """Log response information after processing"""
    logger.info(f"Response: {response.status} {response.status_code}")
    if response.is_json:
        logger.info(f"Response Body: {response.get_json()}")
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for container orchestration"""
    try:
        logger.info("Health check requested")
        # Check if Trino connection is alive
        trino_client.list_tables()
        logger.info("Health check passed successfully")
        return jsonify({"status": "healthy", "services": {"trino": "connected"}}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/catalog', methods=['GET'])
def get_catalog():
    """Get the complete catalog of datasets"""
    try:
        logger.info("Catalog request received")
        catalog = data_loader.get_catalog()
        logger.info(f"Returning catalog with {len(catalog)} datasets")
        return jsonify({"status": "success", "data": catalog})
    except Exception as e:
        logger.error(f"Error fetching catalog: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/dataset/<dataset_id>', methods=['POST'])
def load_dataset(dataset_id):
    """Load a specific dataset into Trino"""
    try:
        logger.info(f"Dataset load request received for {dataset_id}")
        result = data_loader.load_dataset(dataset_id)
        logger.info(f"Successfully loaded dataset {dataset_id}")
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/tables', methods=['GET'])
def list_tables():
    """List all tables in the schema"""
    try:
        logger.info("Table list request received")
        tables = data_loader.list_tables()
        logger.info(f"Returning list of {len(tables)} tables")
        return jsonify({"status": "success", "data": tables})
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/tables/<table_name>', methods=['GET'])
def get_table_info(table_name):
    """Get information about a specific table"""
    try:
        logger.info(f"Table info request received for {table_name}")
        info = data_loader.get_table_info(table_name)
        logger.info(f"Returning info for table {table_name}")
        return jsonify({"status": "success", "data": info})
    except Exception as e:
        logger.error(f"Error getting table info for {table_name}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port) 