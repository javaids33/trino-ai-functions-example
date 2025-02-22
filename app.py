from flask import Flask, request, jsonify, Response, stream_with_context
import os
import json
from dotenv import load_dotenv
from trino.dbapi import connect, DatabaseError
from trino.auth import BasicAuthentication
from embeddings import embedding_service
from ollama_client import OllamaClient
import metadata_sync  # This will start the background sync
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Ollama client
ollama = OllamaClient(model=os.getenv("OLLAMA_MODEL", "llama3:8b"))

def get_trino_conn():
    """Get a new Trino connection with error handling"""
    try:
        return connect(
            host=os.getenv("TRINO_HOST"),
            port=int(os.getenv("TRINO_PORT", "8080")),
            user=os.getenv("TRINO_USER"),
            catalog=os.getenv("TRINO_CATALOG"),
            schema=os.getenv("TRINO_SCHEMA"),
            auth=BasicAuthentication(os.getenv("TRINO_USER", ""))
        )
    except Exception as e:
        logger.error(f"Failed to connect to Trino: {str(e)}")
        raise

def get_schema_context(query: str) -> str:
    """Get relevant schema elements using vector search"""
    try:
        results = embedding_service.query_metadata(query)
        return "\n".join(results['documents']) if results['documents'] else ""
    except Exception as e:
        logger.error(f"Error getting schema context: {str(e)}")
        return ""

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        stream = data.get('stream', False)
        model = data.get('model')
        logger.debug(f"Received query: {data['query']}")
        
        # Get schema context
        context = get_schema_context(data['query'])
        
        try:
            # Generate SQL
            sql = ollama.generate_sql(context, data['query'], model=model)
            logger.debug(f"Generated SQL: {sql}")
            
            # Execute query
            conn = get_trino_conn()
            cur = conn.cursor()
            cur.execute(sql)
            results = cur.fetchall()
            
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
            
            return jsonify({
                "sql": sql,
                "results": results,
                "explanation": explanation,
                "context": context
            })
            
        except DatabaseError as e:
            logger.error(f"Trino error: {str(e)}")
            return jsonify({"error": f"SQL execution failed: {str(e)}"}), 400
            
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
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
        with get_trino_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
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