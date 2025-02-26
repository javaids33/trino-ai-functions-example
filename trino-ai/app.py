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

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# Configure logging with colorama colors
logging.basicConfig(
    level=logging.DEBUG,
    format=f'{Fore.GREEN}%(asctime)s{Fore.RESET} - {Fore.CYAN}%(levelname)s{Fore.RESET} - {Fore.WHITE}%(message)s{Fore.RESET}',
    handlers=[
        logging.StreamHandler()
    ]
)

# Create custom loggers with colors
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger(__name__)

# Custom logger functions with colors
def log_ai_function_request(function_name, content):
    """Log AI function request with color coding"""
    logger.info(f"{Fore.CYAN}==== AI FUNCTION REQUEST: {function_name} ===={Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}Content: {content}{Style.RESET_ALL}")
    conversation_logger.log_trino_request(function_name, content)

def log_ai_function_response(function_name, content):
    """Log AI function response with color coding"""
    logger.info(f"{Fore.GREEN}==== AI FUNCTION RESPONSE: {function_name} ===={Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}Content: {content[:200]}...{Style.RESET_ALL}" if len(content) > 200 else f"{Fore.GREEN}Content: {content}{Style.RESET_ALL}")
    conversation_logger.log_trino_ai_to_trino(function_name, content[:500] + "..." if len(content) > 500 else content)

def log_nl2sql_conversion(nl_query, sql_query):
    """Log NL2SQL conversion with color coding"""
    logger.info(f"{Fore.YELLOW}==== NL2SQL CONVERSION ===={Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}NL Query: {nl_query}{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}SQL Query: {sql_query}{Style.RESET_ALL}")
    conversation_logger.log_trino_ai_processing("nl2sql_conversion", {
        "nl_query": nl_query,
        "sql_query": sql_query
    })

def log_error(message, error=None):
    """Log error with color coding"""
    logger.error(f"{Fore.RED}ERROR: {message}{Style.RESET_ALL}")
    if error:
        logger.error(f"{Fore.RED}Exception: {str(error)}{Style.RESET_ALL}")
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
            is_ai_gen = False
            ai_gen_pattern = r'(?:ai-functions\.ai\.ai_gen|ai_gen)\s*\(\s*[\'"](.+?)[\'"]\s*\)'
            ai_gen_match = re.search(ai_gen_pattern, query, re.IGNORECASE)
            
            if ai_gen_match:
                # This is an ai_gen function call
                is_ai_gen = True
                nl_query = ai_gen_match.group(1)
                log_ai_function_request("ai_gen", nl_query)
                
                # Check if it looks like a natural language query for SQL
                # We'll assume all ai_gen calls are for NL2SQL conversion
                try:
                    # Process the natural language query using the agent orchestrator
                    logger.info(f"{Fore.BLUE}Processing NL2SQL request using agent orchestrator: {nl_query}{Fore.RESET}")
                    result = agent_orchestrator.process_natural_language_query(nl_query, model=model)
                    
                    if "error" in result:
                        error_message = result["error"]
                        error_stage = result.get("stage", "unknown")
                        logger.error(f"{Fore.RED}Error in {error_stage} stage: {error_message}{Fore.RESET}")
                        completion = f"""Error processing your query: {error_message}

The error occurred during the {error_stage} stage of processing.

Please try rephrasing your question or providing more context."""
                    else:
                        # Extract the results
                        sql_query = result["sql_query"]
                        explanation = result["explanation"]
                        is_valid = result.get("is_valid", True)
                        refinement_steps = result.get("refinement_steps", 0)
                        processing_time = result.get("processing_time", 0)
                        
                        log_nl2sql_conversion(nl_query, sql_query)
                        logger.info(f"{Fore.GREEN}NL2SQL conversion completed in {processing_time:.2f}s with {refinement_steps} refinement steps{Fore.RESET}")
                        
                        # Format the response
                        completion = f"""SQL Query for: "{nl_query}"

```sql
{sql_query}
```

Explanation:
{explanation}

This query was generated based on the database schema and the natural language question you provided."""
                        
                        if not is_valid:
                            completion += f"""

⚠️ Warning: This SQL query may have issues. Error: {result.get('error_message', 'Unknown error')}"""
                    
                    log_ai_function_response("ai_gen (NL2SQL)", completion)
                    
                    # Return the response in the expected format
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
                            "prompt_tokens": -1,
                            "completion_tokens": -1,
                            "total_tokens": -1
                        }
                    }
                    
                except Exception as e:
                    log_error("Error processing ai_gen as NL2SQL", e)
                    # Fall back to regular completion if NL2SQL processing fails
                    logger.info("Falling back to regular completion for ai_gen")
                    is_ai_gen = False
            
            # For non-ai_gen functions or fallback, process normally
            if not is_ai_gen:
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
            is_ai_gen = False
            ai_gen_pattern = r'(?:ai-functions\.ai\.ai_gen|ai_gen)\s*\(\s*[\'"](.+?)[\'"]\s*\)'
            ai_gen_match = re.search(ai_gen_pattern, query, re.IGNORECASE)
            
            if ai_gen_match:
                # This is an ai_gen function call
                is_ai_gen = True
                nl_query = ai_gen_match.group(1)
                log_ai_function_request("ai_gen", nl_query)
                
                # Check if it looks like a natural language query for SQL
                # We'll assume all ai_gen calls are for NL2SQL conversion
                try:
                    # Process the natural language query using the agent orchestrator
                    logger.info(f"{Fore.BLUE}Processing NL2SQL request using agent orchestrator: {nl_query}{Fore.RESET}")
                    result = agent_orchestrator.process_natural_language_query(nl_query, model=model)
                    
                    if "error" in result:
                        error_message = result["error"]
                        error_stage = result.get("stage", "unknown")
                        logger.error(f"{Fore.RED}Error in {error_stage} stage: {error_message}{Fore.RESET}")
                        completion = f"""Error processing your query: {error_message}

The error occurred during the {error_stage} stage of processing.

Please try rephrasing your question or providing more context."""
                    else:
                        # Extract the results
                        sql_query = result["sql_query"]
                        explanation = result["explanation"]
                        is_valid = result.get("is_valid", True)
                        refinement_steps = result.get("refinement_steps", 0)
                        processing_time = result.get("processing_time", 0)
                        
                        log_nl2sql_conversion(nl_query, sql_query)
                        logger.info(f"{Fore.GREEN}NL2SQL conversion completed in {processing_time:.2f}s with {refinement_steps} refinement steps{Fore.RESET}")
                        
                        # Format the response
                        completion = f"""SQL Query for: "{nl_query}"

```sql
{sql_query}
```

Explanation:
{explanation}

This query was generated based on the database schema and the natural language question you provided."""
                        
                        if not is_valid:
                            completion += f"""

⚠️ Warning: This SQL query may have issues. Error: {result.get('error_message', 'Unknown error')}"""
                    
                    log_ai_function_response("ai_gen (NL2SQL)", completion)
                    
                    # Return the response in the expected format
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
                            "prompt_tokens": -1,
                            "completion_tokens": -1,
                            "total_tokens": -1
                        }
                    }
                    
                except Exception as e:
                    log_error("Error processing ai_gen as NL2SQL", e)
                    # Fall back to regular completion if NL2SQL processing fails
                    logger.info("Falling back to regular completion for ai_gen")
                    is_ai_gen = False
            
            # For non-ai_gen functions or fallback, process normally
            if not is_ai_gen:
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

@utility_ns.route('/logs')
class LogViewer(Resource):
    @utility_ns.doc('view_logs')
    @utility_ns.response(200, 'Success')
    def get(self):
        """View recent application logs"""
        try:
            log_file_path = os.path.join(os.getcwd(), 'logs/app.log')
            if not os.path.exists(log_file_path):
                return {"error": {"message": "Log file not found"}}, 404
                
            # Read last 200 lines
            with open(log_file_path, 'r') as f:
                lines = f.readlines()
                last_lines = lines[-200:] if len(lines) > 200 else lines
            
            # Format as HTML for browser viewing
            log_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trino AI Logs</title>
                <style>
                    body { font-family: monospace; padding: 20px; background: #f8f9fa; }
                    .log-container { background: white; border-radius: 5px; padding: 15px; overflow: auto; height: 80vh; }
                    .error { color: #dc3545; }
                    .warning { color: #ffc107; }
                    .info { color: #17a2b8; }
                    .debug { color: #6c757d; }
                    h1 { color: #343a40; }
                    .timestamp { color: #6c757d; }
                    .refresh { margin-bottom: 10px; }
                </style>
            </head>
            <body>
                <h1>Trino AI - Log Viewer</h1>
                <div class="refresh">
                    <button onclick="location.reload()">Refresh Logs</button>
                </div>
                <div class="log-container">
            """
            
            for line in last_lines:
                line_class = "debug"
                if "ERROR" in line:
                    line_class = "error"
                elif "WARNING" in line:
                    line_class = "warning"
                elif "INFO" in line:
                    line_class = "info"
                
                # Escape HTML characters
                line = line.replace("<", "&lt;").replace(">", "&gt;")
                log_html += f'<div class="{line_class}">{line}</div>'
            
            log_html += """
                </div>
            </body>
            </html>
            """
            
            return Response(log_html, mimetype='text/html')
            
        except Exception as e:
            logger.error(f"Error reading logs: {str(e)}", exc_info=True)
            return {"error": {"message": f"Failed to read logs: {str(e)}"}}, 500

# Add a redirect from root to Swagger UI
@app.route('/')
def index():
    try:
        with open('static/index.html', 'r') as f:
            return Response(f.read(), mimetype='text/html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}", exc_info=True)
        return Response(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trino AI API</title>
                <meta http-equiv="refresh" content="0; url=/swagger" />
            </head>
            <body>
                <p>Web UI not found. Redirecting to <a href="/swagger">Swagger UI</a>...</p>
            </body>
            </html>
            """,
            mimetype='text/html'
        )

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    try:
        with open(f'static/{path}', 'r') as f:
            content_type = 'text/css' if path.endswith('.css') else 'text/javascript' if path.endswith('.js') else 'text/html'
            return Response(f.read(), mimetype=content_type)
    except Exception as e:
        logger.error(f"Error serving static file {path}: {str(e)}", exc_info=True)
        return {"error": "File not found"}, 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True) 