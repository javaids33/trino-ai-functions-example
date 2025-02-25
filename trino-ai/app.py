from flask import Flask, request, jsonify, Response, stream_with_context
import os
import json
from dotenv import load_dotenv
from trino.dbapi import connect
from embeddings import embedding_service
from ollama_client import OllamaClient
import metadata_sync  # This will start the background sync
import logging
from flask_restx import Api, Resource, fields, Namespace
import time
import uuid

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
# Set all loggers to DEBUG level
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('werkzeug').setLevel(logging.DEBUG)
logging.getLogger('trino').setLevel(logging.DEBUG)
logging.getLogger('requests').setLevel(logging.DEBUG)
logging.getLogger('urllib3').setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Swagger documentation with Flask-RESTX
api = Api(
    app,
    version='1.0',
    title='Trino AI API',
    description='API for Trino AI functions with OpenAI-compatible endpoints',
    doc='/swagger',
    default='Trino AI',
    default_label='Trino AI API Endpoints'
)

# Create namespaces for different API groups
openai_ns = Namespace('OpenAI Compatible', description='OpenAI-compatible API endpoints')
legacy_ns = Namespace('Legacy', description='Legacy API endpoints')
utility_ns = Namespace('Utility', description='Utility endpoints')

api.add_namespace(openai_ns, path='/v1')
api.add_namespace(legacy_ns, path='')
api.add_namespace(utility_ns, path='')

# Define models for request/response documentation
chat_message = api.model('ChatMessage', {
    'role': fields.String(required=True, description='Role of the message sender (system, user, assistant)', enum=['system', 'user', 'assistant']),
    'content': fields.String(required=True, description='Content of the message')
})

chat_request = api.model('ChatCompletionRequest', {
    'model': fields.String(description='Model to use for completion'),
    'messages': fields.List(fields.Nested(chat_message), required=True, description='Messages to generate chat completion for'),
    'stream': fields.Boolean(description='Whether to stream the response', default=False)
})

chat_response = api.model('ChatCompletionResponse', {
    'id': fields.String(description='Unique identifier for the completion'),
    'object': fields.String(description='Object type', default='chat.completion'),
    'created': fields.Integer(description='Unix timestamp of when the completion was created'),
    'model': fields.String(description='Model used for completion'),
    'choices': fields.List(fields.Raw(description='Completion choices')),
    'usage': fields.Raw(description='Token usage information')
})

query_request = api.model('QueryRequest', {
    'query': fields.String(required=True, description='Natural language query to convert to SQL'),
    'stream': fields.Boolean(description='Whether to stream the response', default=False),
    'model': fields.String(description='Model to use for completion')
})

query_response = api.model('QueryResponse', {
    'sql': fields.String(description='Generated SQL query'),
    'results': fields.List(fields.Raw(description='Query results')),
    'explanation': fields.String(description='Explanation of the results'),
    'context': fields.String(description='Schema context used for query generation')
})

embed_request = api.model('EmbedRequest', {
    'text': fields.String(required=True, description='Text to generate embeddings for')
})

embed_response = api.model('EmbedResponse', {
    'embedding': fields.List(fields.Float, description='Vector embedding of the input text'),
    'dimensions': fields.Integer(description='Number of dimensions in the embedding')
})

health_response = api.model('HealthResponse', {
    'status': fields.String(description='Overall service status'),
    'components': fields.Raw(description='Status of individual components')
})

models_response = api.model('ModelsResponse', {
    'data': fields.List(fields.Raw(description='Available models'))
})

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
    """Get relevant schema context for the question using vector search."""
    try:
        # Query metadata embeddings to retrieve context documents
        results = embedding_service.query_metadata(query)
        if results and results.get('documents'):
            # Flatten and deduplicate documents
            flattened_docs = []
            seen = set()
            for doc in results['documents']:
                if isinstance(doc, list):
                    # Handle nested lists
                    for d in doc:
                        if str(d) not in seen:
                            flattened_docs.append(str(d))
                            seen.add(str(d))
                else:
                    if str(doc) not in seen:
                        flattened_docs.append(str(doc))
                        seen.add(str(doc))
            
            context = "\n".join(flattened_docs)
            logger.info(f"Found schema context: {context}")
            return context
        else:
            logger.warning("No schema context available from vector search, using basic schema info")
            # If no context found, get basic schema info from Trino
            conn = get_trino_conn()
            cur = conn.cursor()
            tables = ['customers', 'products', 'sales']
            schema_info = []
            for table in tables:
                try:
                    cur.execute(f"DESCRIBE iceberg.iceberg.{table}")
                    columns = cur.fetchall()
                    schema_info.append(f"Table iceberg.iceberg.{table}:")
                    for col in columns:
                        schema_info.append(f"  - {col[0]} ({col[1]})")
                except Exception as e:
                    logger.error(f"Error describing table {table}: {e}")
            cur.close()
            conn.close()
            return "\n".join(schema_info)
    except Exception as e:
        logger.error(f"Error getting schema context: {e}")
        return ""

@legacy_ns.route('/query')
class Query(Resource):
    @legacy_ns.doc('query')
    @legacy_ns.expect(query_request)
    @legacy_ns.response(200, 'Success', query_response)
    @legacy_ns.response(400, 'Bad Request')
    @legacy_ns.response(415, 'Unsupported Media Type')
    @legacy_ns.response(500, 'Internal Server Error')
    def post(self):
        """Execute a natural language query and return SQL results"""
        try:
            if not request.is_json:
                logger.error("Request Content-Type is not application/json")
                return {"error": "Content-Type must be application/json"}, 415
                
            data = request.json
            if not data or 'query' not in data:
                logger.error("Missing 'query' field in request")
                return {"error": "Missing 'query' field"}, 400
                
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
                    
                    return Response(
                        stream_with_context(generate()),
                        mimetype='application/json'
                    )
                
                # Get explanation
                explanation = ollama.explain_results(sql, results, model=model)
                logger.debug(f"Generated explanation: {explanation}")
                
                return {
                    "sql": sql,
                    "results": results,
                    "explanation": explanation,
                    "context": context
                }
                
            except Exception as e:
                logger.error(f"SQL execution failed: {str(e)}", exc_info=True)
                return {"error": f"SQL execution failed: {str(e)}"}, 400
                
        except Exception as e:
            logger.critical(f"Unexpected error in handle_query: {str(e)}", exc_info=True)
            return {"error": "Internal server error"}, 500

@openai_ns.route('/chat/completions')
class ChatCompletions(Resource):
    @openai_ns.doc('chat_completions')
    @openai_ns.expect(chat_request)
    @openai_ns.response(200, 'Success', chat_response)
    @openai_ns.response(400, 'Bad Request')
    @openai_ns.response(415, 'Unsupported Media Type')
    @openai_ns.response(500, 'Internal Server Error')
    def post(self):
        """Generate chat completions using the OpenAI-compatible API"""
        try:
            if not request.is_json:
                return {"error": {"message": "Content-Type must be application/json"}}, 415
                
            data = request.json
            if not data or 'messages' not in data:
                return {"error": {"message": "Missing 'messages' field"}}, 400
            
            stream = data.get('stream', False)
            model = data.get('model', ollama.model)
            messages = data.get('messages', [])
            
            # Extract the last user message
            last_message = next((m for m in reversed(messages) if m['role'] == 'user'), None)
            if not last_message:
                return {"error": {"message": "No user message found"}}, 400
                
            query = last_message['content']
            logger.info(f"Processing chat completion for query: {query}")
            
            # Enhanced logging - input
            logger.info("==== AI FUNCTION REQUEST ====")
            logger.info(f"Input messages: {json.dumps(messages, indent=2)}")
            
            # Always treat requests from Trino AI functions as AI function requests
            # This is because Trino AI functions use the chat/completions endpoint
            # for all AI function calls
            logger.info(f"Treating as AI function request: {query}")
            
            try:
                # For AI functions, directly use Ollama without SQL generation
                completion = ollama._generate_completion(messages, model=model)
                logger.info(f"Generated AI function response: {completion}")
                
                # Enhanced logging - output
                logger.info("==== AI FUNCTION RESPONSE ====")
                logger.info(f"Output content: {completion}")
                
                if stream:
                    def generate():
                        response = {
                            "id": f"chatcmpl-{str(uuid.uuid4())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": completion},
                                "finish_reason": "stop"
                            }]
                        }
                        yield json.dumps(response) + "\n"
                        yield "data: [DONE]\n\n"
                    
                    return Response(
                        stream_with_context(generate()),
                        mimetype='text/event-stream'
                    )
                
                return {
                    "id": f"chatcmpl-{str(uuid.uuid4())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": completion},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": -1,  # We don't track tokens
                        "completion_tokens": -1,
                        "total_tokens": -1
                    }
                }
                
            except Exception as e:
                logger.error(f"Query execution failed: {str(e)}", exc_info=True)
                return {
                    "error": {
                        "message": f"Query execution failed: {str(e)}",
                        "type": "invalid_request_error"
                    }
                }, 400
                
        except Exception as e:
            logger.critical(f"Unexpected error in chat completions: {str(e)}", exc_info=True)
            return {
                "error": {
                    "message": "Internal server error",
                    "type": "server_error"
                }
            }, 500

@utility_ns.route('/embed')
class Embed(Resource):
    @utility_ns.doc('embed')
    @utility_ns.expect(embed_request)
    @utility_ns.response(200, 'Success', embed_response)
    @utility_ns.response(400, 'Bad Request')
    @utility_ns.response(500, 'Internal Server Error')
    def post(self):
        """Generate embeddings for text"""
        try:
            data = request.json
            if not data or 'text' not in data:
                return {"error": "Missing 'text' field"}, 400
                
            embeddings = embedding_service.embed(data['text'])
            
            return {
                "embedding": embeddings,
                "dimensions": len(embeddings)
            }
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return {"error": str(e)}, 500

@utility_ns.route('/refresh_metadata')
class RefreshMetadata(Resource):
    @utility_ns.doc('refresh_metadata')
    @utility_ns.response(200, 'Success')
    @utility_ns.response(500, 'Internal Server Error')
    def post(self):
        """Manually trigger metadata refresh"""
        try:
            embedding_service.refresh_embeddings()
            return {"status": "metadata refreshed"}
        except Exception as e:
            logger.error(f"Metadata refresh error: {str(e)}")
            return {"error": str(e)}, 500

@utility_ns.route('/health')
class Health(Resource):
    @utility_ns.doc('health')
    @utility_ns.response(200, 'Success', health_response)
    def get(self):
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
        
        return status

@openai_ns.route('/models')
class Models(Resource):
    @openai_ns.doc('list_models')
    @openai_ns.response(200, 'Success', models_response)
    def get(self):
        """List available models in OpenAI format"""
        return {
            "data": [{
                "id": ollama.model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama"
            }]
        }

# Add completions endpoint for AI functions
@openai_ns.route('/completions')
class Completions(Resource):
    @openai_ns.doc('completions')
    @openai_ns.response(200, 'Success')
    @openai_ns.response(400, 'Bad Request')
    @openai_ns.response(500, 'Internal Server Error')
    def post(self):
        """Generate completions using the OpenAI-compatible API"""
        try:
            logger.debug(f"Received completions request: {request.headers}")
            logger.debug(f"Request data: {request.get_data(as_text=True)}")
            
            if not request.is_json:
                logger.error(f"Request is not JSON. Content-Type: {request.headers.get('Content-Type')}")
                return {"error": {"message": "Content-Type must be application/json"}}, 415
                
            data = request.json
            logger.debug(f"Parsed JSON data: {data}")
            
            if not data or 'prompt' not in data:
                logger.error("Missing 'prompt' field in request")
                return {"error": {"message": "Missing 'prompt' field"}}, 400
            
            model = data.get('model', ollama.model)
            prompt = data.get('prompt', '')
            max_tokens = data.get('max_tokens', 100)
            
            logger.info(f"Processing completion for prompt: {prompt}")
            logger.debug(f"Using model: {model}, max_tokens: {max_tokens}")
            
            # Enhanced logging - input
            logger.info("==== AI FUNCTION REQUEST (completions) ====")
            logger.info(f"Input prompt: {prompt}")
            logger.info(f"Model: {model}, max_tokens: {max_tokens}")
            
            # For AI functions, the prompt will typically be in a specific format
            # that indicates which AI function to use
            messages = [{"role": "user", "content": prompt}]
            
            try:
                logger.debug(f"Sending messages to Ollama: {messages}")
                completion = ollama._generate_completion(messages, model=model)
                logger.debug(f"Received completion from Ollama: {completion}")
                
                # Enhanced logging - output
                logger.info("==== AI FUNCTION RESPONSE (completions) ====")
                logger.info(f"Output content: {completion}")
                
                response = {
                    "id": f"cmpl-{str(uuid.uuid4())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "text": completion,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": -1,  # We don't track tokens
                        "completion_tokens": -1,
                        "total_tokens": -1
                    }
                }
                logger.debug(f"Sending response: {response}")
                return response
                
            except Exception as e:
                logger.error(f"Completion generation failed: {str(e)}", exc_info=True)
                return {
                    "error": {
                        "message": f"Completion generation failed: {str(e)}",
                        "type": "invalid_request_error"
                    }
                }, 400
                
        except Exception as e:
            logger.critical(f"Unexpected error in completions: {str(e)}", exc_info=True)
            return {
                "error": {
                    "message": "Internal server error",
                    "type": "server_error"
                }
            }, 500

@utility_ns.route('/metadata')
class Metadata(Resource):
    @utility_ns.doc('get_metadata')
    @utility_ns.response(200, 'Success')
    def get(self):
        """Get all metadata collected from Trino schema"""
        try:
            # Get metadata from ChromaDB
            collection_data = embedding_service.collection.get()
            
            # Format the response
            metadata_response = {
                "tables": [],
                "count": 0,
                "last_updated": int(time.time())
            }
            
            if collection_data and collection_data.get('metadatas'):
                metadata_response["count"] = len(collection_data.get('ids', []))
                
                # Group by table
                for idx, meta in enumerate(collection_data.get('metadatas', [])):
                    table_info = {
                        "id": collection_data.get('ids', [])[idx] if idx < len(collection_data.get('ids', [])) else None,
                        "catalog": meta.get('catalog'),
                        "schema": meta.get('schema'),
                        "table": meta.get('table'),
                        "columns": meta.get('columns', '').split(', '),
                        "document": collection_data.get('documents', [])[idx] if idx < len(collection_data.get('documents', [])) else None
                    }
                    metadata_response["tables"].append(table_info)
            
            return metadata_response
            
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}", exc_info=True)
            return {"error": f"Failed to retrieve metadata: {str(e)}"}, 500

    @utility_ns.doc('refresh_metadata')
    @utility_ns.response(200, 'Success')
    def post(self):
        """Force refresh of Trino metadata"""
        try:
            # Trigger metadata refresh
            embedding_service.refresh_embeddings()
            return {"status": "success", "message": "Metadata refresh triggered successfully"}
        except Exception as e:
            logger.error(f"Error refreshing metadata: {str(e)}", exc_info=True)
            return {"error": f"Failed to refresh metadata: {str(e)}"}, 500

@utility_ns.route('/logs')
class Logs(Resource):
    @utility_ns.doc('get_logs')
    @utility_ns.response(200, 'Success')
    def get(self):
        """Get recent application logs"""
        try:
            # Get the last 100 lines from the log file
            with open('logs/app.log', 'r') as log_file:
                lines = log_file.readlines()
                last_lines = lines[-100:] if len(lines) > 100 else lines
            
            log_html = "<html><head><title>Trino AI Logs</title>"
            log_html += "<style>body{font-family:monospace;background:#f5f5f5;margin:20px}"
            log_html += "pre{background:#fff;padding:15px;border-radius:5px;overflow:auto;max-height:80vh}"
            log_html += ".error{color:red}.warning{color:orange}.info{color:blue}"
            log_html += "</style></head><body>"
            log_html += "<h1>Trino AI - Recent Logs</h1>"
            log_html += "<pre>"
            
            for line in last_lines:
                if "ERROR" in line:
                    log_html += f'<span class="error">{line}</span>'
                elif "WARNING" in line:
                    log_html += f'<span class="warning">{line}</span>'
                elif "INFO" in line:
                    log_html += f'<span class="info">{line}</span>'
                else:
                    log_html += line
            
            log_html += "</pre></body></html>"
            
            return Response(log_html, mimetype='text/html')
            
        except Exception as e:
            return {"error": f"Failed to retrieve logs: {str(e)}"}, 500

# Add a redirect from root to Swagger UI
@app.route('/')
def index():
    return Response(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trino AI API</title>
            <meta http-equiv="refresh" content="0; url=/swagger" />
        </head>
        <body>
            <p>Redirecting to <a href="/swagger">Swagger UI</a>...</p>
        </body>
        </html>
        """,
        mimetype='text/html'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True) 