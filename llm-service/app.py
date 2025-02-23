from flask import Flask, request, jsonify, Response, stream_with_context
import os
import json
from dotenv import load_dotenv
from trino.dbapi import connect
from embeddings import embedding_service
from ollama_client import OllamaClient
import metadata_sync  # This will start the background sync
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Ollama client
ollama = OllamaClient(model=os.getenv("OLLAMA_MODEL", "llama3.2"))

def get_trino_conn():
    """Get a new Trino connection with error handling"""
    try:
        host = os.getenv("TRINO_HOST", "trino")
        port = int(os.getenv("TRINO_PORT", "8080"))
        user = os.getenv("TRINO_USER", "admin")
        catalog = os.getenv("TRINO_CATALOG", "iceberg")
        schema = os.getenv("TRINO_SCHEMA", "iceberg")
        
        logger.info(f"Connecting to Trino at {host}:{port} with catalog={catalog}, schema={schema}")
        
        return connect(
            host=host,
            port=port,
            user=user,
            catalog=catalog,
            schema=schema,
            http_scheme="http"
        )
    except Exception as e:
        logger.error(f"Failed to connect to Trino: {str(e)}", exc_info=True)
        raise

def get_schema_context(query: str) -> str:
    """Get relevant schema elements using vector search"""
    try:
        # First try to get context from vector search
        results = embedding_service.query_metadata(query)
        context = "\n".join(results['documents']) if results['documents'] else ""
        
        # If no context found, provide basic schema information
        if not context:
            logger.warning("No schema context found from vector search, using basic schema info")
            conn = get_trino_conn()
            cur = conn.cursor()
            
            # Get table schemas
            tables = ['customers', 'products', 'sales']
            schema_info = []
            
            for table in tables:
                try:
                    cur.execute(f"DESCRIBE iceberg.iceberg.{table}")
                    columns = cur.fetchall()
                    schema_info.append(f"Table: {table}")
                    schema_info.append("Columns:")
                    for col in columns:
                        schema_info.append(f"  - {col[0]} ({col[1]})")
                    schema_info.append("")
                except Exception as e:
                    logger.error(f"Error getting schema for table {table}: {str(e)}")
            
            context = "\n".join(schema_info)
            
        logger.debug(f"Schema context: {context}")
        return context
        
    except Exception as e:
        logger.error(f"Error getting schema context: {str(e)}", exc_info=True)
        return ""

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        if not request.is_json:
            logger.error("Request Content-Type is not application/json")
            return jsonify({"error": "Content-Type must be application/json"}), 415
            
        data = request.json
        if not data or 'query' not in data:
            logger.error("Missing 'query' field in request")
            return jsonify({"error": "Missing 'query' field"}), 400
            
        stream = data.get('stream', False)
        model = data.get('model')
        logger.info(f"Received query: {data['query']}")
        
        # Get schema context
        context = get_schema_context(data['query'])
        if not context:
            logger.warning("No schema context available")
        
        try:
            # Generate SQL
            sql = ollama.generate_sql(context, data['query'], model=model)
            logger.info(f"Generated SQL: {sql}")
            
            # Execute query
            conn = get_trino_conn()
            cur = conn.cursor()
            cur.execute(sql)
            results = cur.fetchall()
            logger.debug(f"Query results: {results}")
            
            if stream:
                def generate():
                    yield json.dumps({
                        "sql": sql,
                        "results": results
                    }) + "\n"
                    
                    for chunk in ollama.stream_explanation(sql, results, model=model):
                        yield chunk
                
                return Response(stream_with_context(generate()), 
                              mimetype='application/json')
            
            # Get explanation
            explanation = ollama.explain_results(sql, results, model=model)
            logger.debug(f"Generated explanation: {explanation}")
            
            return jsonify({
                "sql": sql,
                "results": results,
                "explanation": explanation,
                "context": context
            })
            
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}", exc_info=True)
            return jsonify({"error": f"SQL execution failed: {str(e)}"}), 400
            
    except Exception as e:
        logger.critical(f"Unexpected error in handle_query: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/embed', methods=['POST'])
def embed():
    """Generate embeddings for text"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
            
        embeddings = embedding_service.embed(data['text'])
        
        return jsonify({
            "embedding": embeddings,
            "dimensions": len(embeddings)
        })
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/refresh_metadata', methods=['POST'])
def refresh_metadata():
    """Manually trigger metadata refresh"""
    try:
        embedding_service.refresh_embeddings()
        return jsonify({"status": "metadata refreshed"})
    except Exception as e:
        logger.error(f"Metadata refresh error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Enhanced health check with metadata status"""
    status = {
        "status": "ready",
        "components": {
            "trino": "unknown",
            "ollama": "unknown",
            "embedding_service": "unknown"
        }
    }
    
    try:
        # Test Trino connection
        conn = get_trino_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        conn.close()
        status["components"]["trino"] = "healthy"
    except Exception as e:
        status["components"]["trino"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    try:
        # Test Ollama
        ollama.generate_sql("test", "SELECT 1")
        status["components"]["ollama"] = "healthy"
    except Exception as e:
        status["components"]["ollama"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    try:
        # Test embedding service
        embedding_service.embed("test")
        status["components"]["embedding_service"] = "healthy"
    except Exception as e:
        status["components"]["embedding_service"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True) 