from flask import Flask, request, jsonify, Response, stream_with_context
import os
import json
import re
from dotenv import load_dotenv
from trino.dbapi import connect
from embeddings import embedding_service
from ollama_client import OllamaClient
import metadata_sync  # This will start the background sync
import logging
from flask_restx import Api, Resource, fields, Namespace
import time
import uuid
from typing import Tuple
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from trino_client import TrinoClient
import colorama
from colorama import Fore, Back, Style
from agent_orchestrator import AgentOrchestrator
from conversation_logger import conversation_logger
from tools.metadata_tools import GetSchemaContextTool, RefreshMetadataTool
from tools.sql_tools import ValidateSQLTool, ExecuteSQLTool
from ai_translate_handler import AITranslateHandler

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# Configure logging with a cleaner format for Docker Desktop
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)

# Create custom loggers with colors
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

# Custom logger functions with cleaner formatting
def log_ai_function_request(function_name, content):
    """Log AI function request with clean formatting"""
    logger.info(f"AI FUNCTION REQUEST: {function_name}")
    logger.info(f"Content: {content}")
    conversation_logger.log_trino_request(function_name, content)

def log_ai_function_response(function_name, content):
    """Log AI function response with clean formatting"""
    logger.info(f"AI FUNCTION RESPONSE: {function_name}")
    content_preview = content[:200] + "..." if len(content) > 200 else content
    logger.info(f"Content: {content_preview}")
    conversation_logger.log_trino_ai_to_trino(function_name, content[:500] + "..." if len(content) > 500 else content)

def log_nl2sql_conversion(nl_query, sql_query):
    """Log NL to SQL conversion with clean formatting"""
    logger.info(f"NL2SQL CONVERSION:")
    logger.info(f"NL Query: {nl_query}")
    logger.info(f"SQL Query: {sql_query}")
    conversation_logger.log_trino_ai_processing("nl2sql_conversion", {
        "nl_query": nl_query,
        "sql_query": sql_query
    })

def log_error(message, error=None):
    """Log error with clean formatting"""
    logger.error(f"ERROR: {message}")
    if error:
        logger.error(f"Exception: {str(error)}")
    conversation_logger.log_error("trino-ai", message, error)

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
api.add_namespace(utility_ns, path='/utility')

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

# Define the request/response models for our new NL2SQL endpoint
nl2sql_request = api.model('NL2SQLRequest', {
    'query': fields.String(required=True, description='Natural language query')
})

nl2sql_response = api.model('NL2SQLResponse', {
    'natural_language_query': fields.String(description='Original query'),
    'sql_query': fields.String(description='Generated SQL query'),
    'explanation': fields.String(description='Explanation of the SQL query'),
    'context_used': fields.String(description='Schema context used'),
    'refinement_steps': fields.Integer(description='Number of refinement steps')
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
ollama = OllamaClient(
    base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
    model=os.getenv("OLLAMA_MODEL", "llama3.2")
)

# Initialize Trino client
trino_client = TrinoClient(
    host=os.getenv("TRINO_HOST", "trino"),
    port=int(os.getenv("TRINO_PORT", "8080")),
    user=os.getenv("TRINO_USER", "admin"),
    catalog=os.getenv("TRINO_CATALOG", "iceberg"),
    schema=os.getenv("TRINO_SCHEMA", "iceberg")
)

# Initialize tools
logger.info("Initializing tools...")
tools = {
    "get_schema_context": GetSchemaContextTool(),
    "refresh_metadata": RefreshMetadataTool(),
    "validate_sql": ValidateSQLTool(trino_client=trino_client),
    "execute_sql": ExecuteSQLTool()
}

# Initialize agent orchestrator
agent_orchestrator = AgentOrchestrator(ollama_client=ollama)
logger.info(f"{Fore.GREEN}Agent orchestrator initialized{Fore.RESET}")

def validate_sql(sql_query: str) -> Tuple[bool, str]:
    """Validate SQL query against Trino without executing it."""
    try:
        logger.info(f"Validating SQL query: {sql_query}")
        
        # Clean up the SQL query before validation
        # Remove any "Query**" or similar text at the beginning
        sql_query = re.sub(r'^Query\*\*\s*', '', sql_query)
        
        # Remove any trailing semicolons
        sql_query = sql_query.strip()
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1].strip()
            logger.info(f"Removed trailing semicolon from SQL query")
        
        # Connect to Trino
        with connect(
            host=os.getenv('TRINO_HOST', 'trino'),
            port=int(os.getenv('TRINO_PORT', '8080')),
            user=os.getenv('TRINO_USER', 'admin'),
            catalog=os.getenv('TRINO_CATALOG', 'iceberg'),
            schema=os.getenv('TRINO_SCHEMA', 'iceberg')
        ) as conn:
            cur = conn.cursor()
            
            # Use EXPLAIN to validate without execution
            explain_query = f"EXPLAIN {sql_query}"
            logger.debug(f"Executing query: {explain_query}")
            cur.execute(explain_query)
            
            # If we get here, query is valid
            cur.fetchall()  # Consume results
            logger.info("SQL validation successful")
            return True, ""
            
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"SQL validation failed: {error_msg}")
        
        # Provide more detailed error messages for common issues
        if "mismatched input 'Query'" in error_msg:
            error_msg = "SQL contains 'Query' text at the beginning. This needs to be removed."
        elif "mismatched input ';'" in error_msg:
            error_msg = "SQL contains a semicolon at the end. This needs to be removed."
        elif "mismatched input 'OFFSET'" in error_msg:
            error_msg = "SQL contains OFFSET syntax which is not supported. Use LIMIT instead."
        elif "mismatched input 'FETCH'" in error_msg:
            error_msg = "SQL contains FETCH FIRST syntax which is not supported. Use LIMIT instead."
        elif "MISSING_SCHEMA_NAME" in error_msg:
            error_msg = "SQL is missing schema name. Use fully qualified table names (iceberg.iceberg.table_name)."
        elif "MISSING_CATALOG_NAME" in error_msg:
            error_msg = "SQL is missing catalog name. Use fully qualified table names (iceberg.iceberg.table_name)."
        elif "TABLE_NOT_FOUND" in error_msg:
            error_msg = "Table not found. Ensure you're using fully qualified table names (iceberg.iceberg.table_name)."
        
        return False, error_msg

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

@utility_ns.route('/nl2sql')
class NaturalLanguageToSQL(Resource):
    @utility_ns.doc('nl_to_sql')
    @utility_ns.expect(nl2sql_request)
    @utility_ns.response(200, 'Success', nl2sql_response)
    @utility_ns.response(400, 'Bad Request')
    @utility_ns.response(500, 'Internal Server Error')
    def post(self):
        """Convert natural language query to SQL"""
        try:
            if not request.is_json:
                return {"error": {"message": "Content-Type must be application/json"}}, 415
                
            data = request.json
            if 'query' not in data:
                return {"error": {"message": "Missing 'query' field"}}, 400
            
            nl_query = data['query']
            logger.info(f"Processing NL2SQL request: {nl_query}")
            
            # Get relevant schema context from vector DB
            context = embedding_service.get_context_for_query(nl_query)
            logger.info(f"Retrieved context with {context.count('Table:')} tables")
            
            # Generate initial SQL query
            sql_query, explanation = ollama.generate_sql_with_explanation(context, nl_query)
            logger.info(f"Initial SQL query generated: {sql_query}")
            
            # Validate and refine the SQL query if needed
            refinement_steps = 0
            max_refinements = 2  # Limit refinement attempts
            
            while refinement_steps < max_refinements:
                valid, error_message = validate_sql(sql_query)
                
                if valid:
                    logger.info("SQL query validation successful")
                    break
                    
                logger.info(f"SQL validation failed: {error_message}")
                refinement_steps += 1
                
                # Refine the query based on the error
                sql_query, explanation = ollama.refine_sql(
                    context, 
                    nl_query, 
                    sql_query, 
                    error_message
                )
                logger.info(f"Refined SQL (step {refinement_steps}): {sql_query}")
            
            # Create response with full context
            response = {
                "natural_language_query": nl_query,
                "sql_query": sql_query,
                "explanation": explanation,
                "context_used": context,
                "refinement_steps": refinement_steps
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing NL2SQL request: {str(e)}", exc_info=True)
            return {"error": {"message": f"Failed to process query: {str(e)}"}}, 500

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
                log_error("Request Content-Type is not application/json")
                return {"error": {"message": "Content-Type must be application/json"}}, 415
                
            data = request.json
            if not data or 'messages' not in data:
                log_error("Missing 'messages' field in request")
                return {"error": {"message": "Missing 'messages' field"}}, 400
            
            stream = data.get('stream', False)
            model = data.get('model', ollama.model)
            messages = data.get('messages', [])
            
            # Extract the last user message
            last_message = next((m for m in reversed(messages) if m['role'] == 'user'), None)
            if not last_message:
                log_error("No user message found")
                return {"error": {"message": "No user message found"}}, 400
                
            query = last_message['content']
            
            # Check if this is an ai_gen function call
            if "ai_gen" in query.lower():
                try:
                    # Extract the prompt from the query
                    prompt = query.split("ai_gen(")[1].split(")")[0].strip("'\"")
                    
                    # Generate text using the ollama client
                    response = ollama.generate_response(prompt)
                    
                    # Return the response
                    if stream:
                        def generate():
                            yield json.dumps({
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": response
                                        },
                                        "finish_reason": "stop"
                                    }
                                ]
                            }) + "\n"
                            
                        return Response(generate(), mimetype='text/event-stream')
                    else:
                        return {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": response
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(query),
                                "completion_tokens": len(response),
                                "total_tokens": len(query) + len(response)
                            }
                        }
                except Exception as e:
                    log_error("Error processing ai_gen as NL2SQL", e)
            
            # Check if this is an ai_translate function call
            if "Language: \"sql\"" in query:
                try:
                    # Extract the query text
                    query_text = query.split("=====\n")[-1].strip()
                    
                    # Get the AI translate handler
                    handler = AITranslateHandler()
                    
                    # Translate to SQL
                    result = handler.translate_to_sql(query_text)
                    
                    # Return the response
                    if stream:
                        def generate():
                            yield json.dumps({
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": result.get("sql", f"Error: {result.get('error', 'Unknown error')}")
                                        },
                                        "finish_reason": "stop"
                                    }
                                ]
                            }) + "\n"
                            
                        return Response(generate(), mimetype='text/event-stream')
                    else:
                        return {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": result.get("sql", f"Error: {result.get('error', 'Unknown error')}")
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(query),
                                "completion_tokens": len(result.get("sql", "")),
                                "total_tokens": len(query) + len(result.get("sql", ""))
                            }
                        }
                except Exception as e:
                    log_error("Error processing ai_translate", e)
            
            # Process as a regular NL2SQL query
            # For non-ai_gen functions or fallback, process normally
            if not "ai_gen" in query.lower():
                # Identify which AI function is being called
                function_name = "unknown"
                for func in ["ai_analyze_sentiment", "ai_classify", "ai_extract", "ai_fix_grammar", 
                             "ai_mask", "ai_translate"]:
                    if func in query:
                        function_name = func
                        break
                
                log_ai_function_request(function_name, query)
                
                # For AI functions, directly use Ollama without SQL generation
                completion = ollama._generate_completion(messages, model=model)
                log_ai_function_response(function_name, completion)
                
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
            log_error("Unexpected error in chat completions", e)
            return {
                "error": {
                    "message": f"Internal server error: {str(e)}",
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
    @openai_ns.expect(chat_request)
    @openai_ns.response(200, 'Success', chat_response)
    @openai_ns.response(400, 'Bad Request')
    @openai_ns.response(500, 'Internal Server Error')
    def post(self):
        """Generate chat completions using the OpenAI-compatible API"""
        try:
            if not request.is_json:
                log_error("Request Content-Type is not application/json")
                return {"error": {"message": "Content-Type must be application/json"}}, 415
                
            data = request.json
            if not data or 'messages' not in data:
                log_error("Missing 'messages' field in request")
                return {"error": {"message": "Missing 'messages' field"}}, 400
            
            stream = data.get('stream', False)
            model = data.get('model', ollama.model)
            messages = data.get('messages', [])
            
            # Extract the last user message
            last_message = next((m for m in reversed(messages) if m['role'] == 'user'), None)
            if not last_message:
                log_error("No user message found")
                return {"error": {"message": "No user message found"}}, 400
                
            query = last_message['content']
            
            # Check if this is an ai_gen function call
            if "ai_gen" in query.lower():
                try:
                    # Extract the prompt from the query
                    prompt = query.split("ai_gen(")[1].split(")")[0].strip("'\"")
                    
                    # Generate text using the ollama client
                    response = ollama.generate_response(prompt)
                    
                    # Return the response
                    if stream:
                        def generate():
                            yield json.dumps({
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": response
                                        },
                                        "finish_reason": "stop"
                                    }
                                ]
                            }) + "\n"
                            
                        return Response(generate(), mimetype='text/event-stream')
                    else:
                        return {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": response
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(query),
                                "completion_tokens": len(response),
                                "total_tokens": len(query) + len(response)
                            }
                        }
                except Exception as e:
                    log_error("Error processing ai_gen as NL2SQL", e)
            
            # Check if this is an ai_translate function call
            if "Language: \"sql\"" in query:
                try:
                    # Extract the query text
                    query_text = query.split("=====\n")[-1].strip()
                    
                    # Get the AI translate handler
                    handler = AITranslateHandler()
                    
                    # Translate to SQL
                    result = handler.translate_to_sql(query_text)
                    
                    # Return the response
                    if stream:
                        def generate():
                            yield json.dumps({
                                "id": f"chatcmpl-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": result.get("sql", f"Error: {result.get('error', 'Unknown error')}")
                                        },
                                        "finish_reason": "stop"
                                    }
                                ]
                            }) + "\n"
                            
                        return Response(generate(), mimetype='text/event-stream')
                    else:
                        return {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": result.get("sql", f"Error: {result.get('error', 'Unknown error')}")
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(query),
                                "completion_tokens": len(result.get("sql", "")),
                                "total_tokens": len(query) + len(result.get("sql", ""))
                            }
                        }
                except Exception as e:
                    log_error("Error processing ai_translate", e)
            
            # Process as a regular NL2SQL query
            # For non-ai_gen functions or fallback, process normally
            if not "ai_gen" in query.lower():
                # Identify which AI function is being called
                function_name = "unknown"
                for func in ["ai_analyze_sentiment", "ai_classify", "ai_extract", "ai_fix_grammar", 
                             "ai_mask", "ai_translate"]:
                    if func in query:
                        function_name = func
                        break
                
                log_ai_function_request(function_name, query)
                
                # For AI functions, directly use Ollama without SQL generation
                completion = ollama._generate_completion(messages, model=model)
                log_ai_function_response(function_name, completion)
                
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
            log_error("Unexpected error in chat completions", e)
            return {
                "error": {
                    "message": f"Internal server error: {str(e)}",
                    "type": "server_error"
                }
            }, 500

@utility_ns.route('/metadata')
class MetadataExplorer(Resource):
    @utility_ns.doc('get_metadata')
    @utility_ns.response(200, 'Success')
    def get(self):
        """Get all metadata collected from Trino schema"""
        try:
            # Get all data from the collection
            collection_data = embedding_service.collection.get()
            
            metadata_response = {
                "tables": [],
                "count": 0,
                "last_updated": int(time.time())
            }
            
            if collection_data and 'metadatas' in collection_data:
                metadata_response["count"] = len(collection_data['ids'])
                
                # Process each metadata entry
                for i, metadata in enumerate(collection_data['metadatas']):
                    table_info = {
                        "id": collection_data['ids'][i],
                        "catalog": metadata.get('catalog', ''),
                        "schema": metadata.get('schema', ''),
                        "table": metadata.get('table', ''),
                        "columns": metadata.get('columns', '').split(', '),
                        "has_samples": metadata.get('has_samples', False),
                        "sample_count": metadata.get('sample_count', 0)
                    }
                    metadata_response["tables"].append(table_info)
            
            return metadata_response
            
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}", exc_info=True)
            return {"error": {"message": f"Failed to retrieve metadata: {str(e)}"}}, 500
            
    @utility_ns.doc('refresh_metadata')
    @utility_ns.response(200, 'Success')
    def post(self):
        """Force refresh of Trino metadata"""
        try:
            embedding_service.refresh_embeddings()
            return {"status": "success", "message": "Metadata refresh initiated"}
        except Exception as e:
            logger.error(f"Error refreshing metadata: {str(e)}", exc_info=True)
            return {"error": {"message": f"Failed to refresh metadata: {str(e)}"}}, 500

@utility_ns.route('/execute_query')
class ExecuteQuery(Resource):
    @utility_ns.doc('execute_query')
    @utility_ns.expect(api.model('ExecuteQueryRequest', {
        'query': fields.String(required=True, description='SQL query to execute')
    }))
    @utility_ns.response(200, 'Success')
    @utility_ns.response(400, 'Bad Request')
    @utility_ns.response(500, 'Internal Server Error')
    def post(self):
        """Execute a SQL query against Trino"""
        try:
            data = request.json
            query = data.get('query', '')
            
            if not query:
                return {'error': 'Query is required'}, 400
            
            logger.info(f"Executing SQL query: {query}")
            
            # Connect to Trino
            with connect(
                host=os.getenv('TRINO_HOST', 'trino'),
                port=int(os.getenv('TRINO_PORT', '8080')),
                user=os.getenv('TRINO_USER', 'admin'),
                catalog=os.getenv('TRINO_CATALOG', 'iceberg'),
                schema=os.getenv('TRINO_SCHEMA', 'iceberg')
            ) as conn:
                cur = conn.cursor()
                
                # Execute the query
                cur.execute(query)
                
                # Fetch column names
                columns = [desc[0] for desc in cur.description] if cur.description else []
                
                # Fetch results (limit to 100 rows for safety)
                rows = []
                for i, row in enumerate(cur.fetchall()):
                    if i >= 100:
                        break
                    rows.append(row)
                
                return {
                    'success': True,
                    'columns': columns,
                    'rows': rows,
                    'row_count': len(rows)
                }
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing query: {error_msg}")
            return {'error': error_msg}, 500

@utility_ns.route('/logs')
class LogViewer(Resource):
    @utility_ns.doc('view_logs')
    @utility_ns.response(200, 'Success')
    def get(self):
        """Get conversation logs"""
        try:
            # Get the conversation ID from the conversation logger
            conversation_id = conversation_logger.conversation_id
            log_file = f"logs/conversation-{conversation_id}.log"
            
            # Check if the specific log file exists
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()
                
                # Parse the logs into a structured format
                logs = self._parse_logs(content)
                
                return {
                    'success': True,
                    'conversation_id': conversation_id,
                    'logs': logs
                }
            else:
                # If the specific log file doesn't exist, try to find the most recent log file
                logger.warning(f"Log file not found: {log_file}, looking for most recent log file")
                
                # Check if logs directory exists
                if not os.path.exists("logs"):
                    logger.error("Logs directory not found")
                    return {
                        'success': False,
                        'error': {"message": "Log file not found"}
                    }
                
                # Get all conversation log files
                log_files = [f for f in os.listdir("logs") if f.startswith("conversation-") and f.endswith(".log")]
                
                if not log_files:
                    logger.error("No conversation log files found")
                    return {
                        'success': False,
                        'error': {"message": "No conversation log files found"}
                    }
                
                # Sort by modification time (most recent first)
                log_files.sort(key=lambda x: os.path.getmtime(os.path.join("logs", x)), reverse=True)
                most_recent_log = os.path.join("logs", log_files[0])
                
                logger.info(f"Using most recent log file: {most_recent_log}")
                
                with open(most_recent_log, "r") as f:
                    content = f.read()
                
                # Extract conversation ID from filename
                recent_conv_id = log_files[0].replace("conversation-", "").replace(".log", "")
                
                # Parse the logs into a structured format
                logs = self._parse_logs(content)
                
                return {
                    'success': True,
                    'conversation_id': recent_conv_id,
                    'logs': logs,
                    'note': f"Using most recent log file instead of current conversation ({conversation_id})"
                }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error retrieving logs: {error_msg}")
            return {'error': {"message": error_msg}}, 500
    
    def _parse_logs(self, content):
        """Parse log content into structured format"""
        lines = content.split('\n')
        logs = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if this is a conversation header or system message
            if line.startswith("=== Conversation"):
                logs.append({
                    'type': 'system',
                    'from': 'SYSTEM',
                    'to': 'SYSTEM',
                    'message': line,
                    'timestamp': datetime.now().isoformat()
                })
                i += 1
                continue
            
            # Check if this is a log entry
            if line.startswith("[") and "]" in line:
                # Extract timestamp and message type
                parts = line.split("]", 1)
                if len(parts) < 2:
                    i += 1
                    continue
                
                timestamp = parts[0].strip("[")
                message_type = parts[1].strip()
                
                # Extract from and to
                if "→" in message_type:
                    from_to_parts = message_type.split(":")
                    if len(from_to_parts) < 2:
                        i += 1
                        continue
                    
                    from_to = from_to_parts[0].strip()
                    from_to_parts = from_to.split("→")
                    
                    if len(from_to_parts) == 2:
                        from_entity = from_to_parts[0].strip()
                        to_entity = from_to_parts[1].strip()
                        
                        # Get the message content from the next line
                        message_content = ""
                        j = i + 1
                        while j < len(lines) and not lines[j].strip().startswith("["):
                            message_content += lines[j] + "\n"
                            j += 1
                        
                        message_content = message_content.strip()
                        
                        # Add to logs
                        logs.append({
                            'from': from_entity,
                            'to': to_entity,
                            'message': message_content,
                            'timestamp': timestamp
                        })
                        
                        i = j
                        continue
            
            i += 1
        
        return logs
    
    @utility_ns.doc('clear_logs')
    @utility_ns.response(200, 'Success')
    def delete(self):
        """Clear conversation logs"""
        try:
            # Get the conversation ID from the conversation logger
            conversation_id = conversation_logger.conversation_id
            log_file = f"logs/conversation-{conversation_id}.log"
            
            if os.path.exists(log_file):
                # Clear the log file
                with open(log_file, "w") as f:
                    f.write(f"=== Conversation {conversation_id} cleared at {datetime.now().isoformat()} ===\n\n")
                
                return {
                    'success': True,
                    'message': f"Conversation logs cleared for {conversation_id}"
                }
            else:
                # If the specific log file doesn't exist, try to find the most recent log file
                logger.warning(f"Log file not found: {log_file}, looking for most recent log file")
                
                # Check if logs directory exists
                if not os.path.exists("logs"):
                    logger.error("Logs directory not found")
                    return {
                        'success': False,
                        'error': "Logs directory not found"
                    }
                
                # Get all conversation log files
                log_files = [f for f in os.listdir("logs") if f.startswith("conversation-") and f.endswith(".log")]
                
                if not log_files:
                    logger.error("No conversation log files found")
                    return {
                        'success': False,
                        'error': "No conversation log files found"
                    }
                
                # Sort by modification time (most recent first)
                log_files.sort(key=lambda x: os.path.getmtime(os.path.join("logs", x)), reverse=True)
                most_recent_log = os.path.join("logs", log_files[0])
                
                logger.info(f"Clearing most recent log file: {most_recent_log}")
                
                # Extract conversation ID from filename
                recent_conv_id = log_files[0].replace("conversation-", "").replace(".log", "")
                
                # Clear the log file
                with open(most_recent_log, "w") as f:
                    f.write(f"=== Conversation {recent_conv_id} cleared at {datetime.now().isoformat()} ===\n\n")
                
                return {
                    'success': True,
                    'message': f"Cleared most recent log file for conversation {recent_conv_id}",
                    'note': f"Using most recent log file instead of current conversation ({conversation_id})"
                }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error clearing logs: {error_msg}")
            return {'error': error_msg}, 500

@utility_ns.route('/conversations')
class Conversations(Resource):
    @utility_ns.doc('get_conversations')
    @utility_ns.response(200, 'Success')
    def get(self):
        """Get a list of all conversations"""
        try:
            conversations = conversation_logger.get_all_conversations()
            return {"conversations": conversations}, 200
        except Exception as e:
            logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
            return {"error": f"Error getting conversations: {str(e)}"}, 500

# Add a route for the workflow viewer
@app.route('/workflow-viewer')
def workflow_viewer():
    """Serve the workflow viewer HTML page"""
    return Response(
        open('static/workflow-viewer.html').read(),
        mimetype='text/html'
    )

@app.route('/workflow-viewer/<conversation_id>')
def workflow_viewer_with_id(conversation_id):
    """Serve the workflow viewer HTML page with a specific conversation ID"""
    return Response(
        open('static/workflow-viewer.html').read(),
        mimetype='text/html'
    )

# Add a route to view the workflow for a specific conversation
@app.route('/utility/workflow/<conversation_id>', methods=['GET'])
def get_workflow(conversation_id):
    """Get the workflow details for a specific conversation"""
    try:
        # Get the workflow details from the conversation logger
        workflow = conversation_logger.get_workflow(conversation_id)
        
        if not workflow:
            return jsonify({
                "error": f"No workflow found for conversation ID: {conversation_id}",
                "status": "error"
            }), 404
            
        return jsonify({
            "conversation_id": conversation_id,
            "workflow": workflow,
            "status": "success"
        })
    except Exception as e:
        logger.exception(f"Error retrieving workflow: {str(e)}")
        return jsonify({
            "error": f"Error retrieving workflow: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/ai_translate', methods=['POST'])
def handle_ai_translate():
    """Handle AI translate requests"""
    try:
        # Check if the request is JSON
        if not request.is_json:
            logger.error("Request content type is not application/json")
            return jsonify({
                "error": "Request content type must be application/json",
                "status": "error"
            }), 400
            
        # Get the request data
        data = request.get_json()
        
        # Check if the query is provided
        if "query" not in data:
            logger.error("No query provided in request")
            return jsonify({
                "error": "No query provided in request",
                "status": "error"
            }), 400
            
        # Get the query and optional parameters
        query = data["query"]
        execute = data.get("execute", True)  # Default to executing the query
        model = data.get("model", None)
        
        logger.info(f"Received AI translate request: {query}")
        
        # Initialize the AI translate handler
        handler = AITranslateHandler()
        
        # Handle the request
        result = handler.handle_translate_request(data, execute)
        
        # Check if there was an error
        if "error" in result:
            logger.error(f"Error in AI translate: {result['error']}")
            return jsonify(result), 400
            
        # Return the result
        return jsonify(result)
    except Exception as e:
        logger.exception(f"Error handling AI translate request: {str(e)}")
        return jsonify({
            "error": f"Error handling AI translate request: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/workflow/current', methods=['GET'])
def get_current_workflow():
    """Get workflow details for the current conversation"""
    try:
        # Get the current conversation ID
        conversation_id = conversation_logger.get_current_conversation_id()
        
        if not conversation_id:
            return jsonify({
                "error": "No active conversation found",
                "status": "error"
            }), 404
            
        # Get the workflow details
        workflow = conversation_logger.get_workflow(conversation_id)
        
        if not workflow:
            return jsonify({
                "error": "No workflow found for the current conversation",
                "status": "error"
            }), 404
            
        return jsonify({
            "conversation_id": conversation_id,
            "workflow": workflow,
            "status": "success"
        })
    except Exception as e:
        logger.exception(f"Error retrieving current workflow: {str(e)}")
        return jsonify({
            "error": f"Error retrieving current workflow: {str(e)}",
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True) 