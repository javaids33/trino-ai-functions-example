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
logger.info("Agent orchestrator initialized")

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
        """Convert natural language to SQL using the agent orchestrator"""
        try:
            data = request.json
            nl_query = data.get('query', '')
            model = data.get('model', None)
            
            if not nl_query:
                return {"error": "No query provided"}, 400
            
            # Log the request
            log_nl2sql_conversion(nl_query, "")
            
            # Initialize the agent orchestrator if not already done
            if not hasattr(app, 'agent_orchestrator'):
                logger.info("Initializing agent orchestrator")
                app.agent_orchestrator = AgentOrchestrator(ollama_client=ollama)
            
            # Process the query
            logger.info(f"Processing NL2SQL request using agent orchestrator: {nl_query}")
            result = app.agent_orchestrator.process_natural_language_query(nl_query, model=model)
            
            # Check for errors
            if "error" in result:
                error_message = result["error"]
                error_stage = result.get("stage", "unknown")
                logger.error(f"Error in {error_stage} stage: {error_message}")
                return {"error": f"Error processing your query: {error_message}. The error occurred during the {error_stage} stage of processing."}, 500
            
            # Format the response based on whether it's a data query or knowledge query
            if "response" in result:
                # Knowledge query
                processing_time = time.time() - start_time if 'start_time' in locals() else 0
                logger.info(f"Knowledge query processed in {processing_time:.2f}s")
                
                return {
                    "query": nl_query,
                    "response": result["response"],
                    "is_data_query": False
                }
            else:
                # Data query
                sql_query = result.get("sql", "")
                explanation = result.get("explanation", "")
                refinement_steps = result.get("refinement_steps", 0)
                processing_time = time.time() - start_time if 'start_time' in locals() else 0
                
                # Log the conversion
                log_nl2sql_conversion(nl_query, sql_query)
                
                logger.info(f"NL2SQL conversion completed in {processing_time:.2f}s with {refinement_steps} refinement steps")
                
                return {
                    "query": nl_query,
                    "sql": sql_query,
                    "explanation": explanation,
                    "is_data_query": True
                }
                
        except Exception as e:
            logger.exception(f"Error processing NL2SQL request: {str(e)}")
            return {"error": f"Error processing your query: {str(e)}"}, 500

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
                        # Check if this is a data query or a knowledge query
                        is_data_query = result.get("is_data_query", True)
                        
                        if not is_data_query:
                            # This is a knowledge query
                            response = result.get("response", "No response available")
                            processing_time = result.get("processing_time", 0)
                            
                            logger.info(f"{Fore.GREEN}Knowledge query processed in {processing_time:.2f}s{Fore.RESET}")
                            
                            # Format the response
                            completion = f"""Response to: "{nl_query}"

{response}

This response was generated based on the general knowledge available to the AI model."""
                        else:
                            # This is a data query
                            sql_query = result["sql_query"]
                            explanation = result.get("explanation", "")
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
                        # Check if this is a data query or a knowledge query
                        is_data_query = result.get("is_data_query", True)
                        
                        if not is_data_query:
                            # This is a knowledge query
                            response = result.get("response", "No response available")
                            processing_time = result.get("processing_time", 0)
                            
                            logger.info(f"{Fore.GREEN}Knowledge query processed in {processing_time:.2f}s{Fore.RESET}")
                            
                            # Format the response
                            completion = f"""Response to: "{nl_query}"

{response}

This response was generated based on the general knowledge available to the AI model."""
                        else:
                            # This is a data query
                            sql_query = result["sql_query"]
                            explanation = result.get("explanation", "")
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

# Add a route for the workflow viewer
@app.route('/workflow-viewer')
def workflow_viewer():
    try:
        with open('static/workflow-viewer.html', 'r') as f:
            return Response(f.read(), mimetype='text/html')    
    except Exception as e:
        logger.error(f"Error serving workflow-viewer.html: {str(e)}", exc_info=True)
        return Response(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trino AI API</title>
                <meta http-equiv="refresh" content="0; url=/swagger" />
            </head>
            <body>
                <p>Workflow Viewer not found. Redirecting to <a href="/swagger">Swagger UI</a>...</p>
            </body>
            </html>
            """,
            mimetype='text/html'
        )

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
            content_type='text/html'
        )

# Add a route for the conversation viewer
@app.route('/conversation-viewer')
def conversation_viewer():
    try:
        with open('static/conversation-viewer.html', 'r') as f:
            return Response(f.read(), mimetype='text/html')    
    except Exception as e:
        logger.error(f"Error serving conversation-viewer.html: {str(e)}", exc_info=True)
        return Response(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trino AI API</title>
                <meta http-equiv="refresh" content="0; url=/swagger" />
            </head>
            <body>
                <p>Conversation Viewer not found. Redirecting to <a href="/swagger">Swagger UI</a>...</p>
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

# Add this after the other route definitions
@app.route('/api/query', methods=['POST'])
def api_query():
    """
    API endpoint to handle both data queries and knowledge queries
    """
    try:
        if not request.is_json:
            logger.error("Request Content-Type is not application/json")
            return jsonify({"error": "Content-Type must be application/json"}), 415
            
        data = request.json
        if not data or 'query' not in data:
            logger.error("Missing 'query' field in request")
            return jsonify({"error": "Missing 'query' field"}), 400
            
        query = data['query']
        model = data.get('model')
        logger.info(f"Received query: {query}")
        
        # Process the query using the agent orchestrator
        try:
            # Initialize the agent orchestrator if not already done
            if not hasattr(app, 'agent_orchestrator'):
                logger.info("Initializing agent orchestrator")
                app.agent_orchestrator = AgentOrchestrator(ollama_client=ollama)
            
            # Process the query
            result = app.agent_orchestrator.process_natural_language_query(query, model=model)
            
            if "error" in result:
                error_message = result["error"]
                error_stage = result.get("stage", "unknown")
                logger.error(f"Error in {error_stage} stage: {error_message}")
                return jsonify({
                    "error": f"Error processing your query: {error_message}. The error occurred during the {error_stage} stage of processing.",
                    "agent_reasoning": result.get("agent_reasoning", [])
                }), 400
            
            # Return the result with schema context and agent reasoning
            response = {
                "query": query,
                "is_data_query": result.get("is_data_query", result.get("query_type") == "sql"),
                "schema_context": result.get("schema_context", ""),
                "agent_reasoning": result.get("agent_reasoning", []),
                "explanation": result.get("explanation", "")
            }
            
            # Add SQL-specific fields if this is a data query
            if response["is_data_query"]:
                response["sql_query"] = result.get("sql_query", "")
                response["result_table"] = result.get("execution_results", {})
            else:
                # Add knowledge-specific fields
                response["response"] = result.get("knowledge", result.get("response", ""))
            
            # Return the result
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            return jsonify({"error": f"Query processing failed: {str(e)}"}), 400
            
    except Exception as e:
        logger.critical(f"Unexpected error in api_query: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# Add this after the other route definitions
@app.route('/api/ai_translate', methods=['POST'])
def handle_ai_translate():
    """Handle AI translate function calls from Trino"""
    try:
        # Parse the request
        data = request.json
        query = data.get('query', '')
        target_format = data.get('target_format', 'sql')
        model = data.get('model', None)
        execute = data.get('execute', True)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Initialize the AI translate handler if not already done
        if not hasattr(app, 'ai_translate_handler'):
            logger.info("Initializing AI translate handler")
            app.ai_translate_handler = AITranslateHandler(ollama_client=ollama)
        
        # Process the request
        result = app.ai_translate_handler.handle_translate_request(
            query=query,
            target_format=target_format,
            model=model,
            execute=execute
        )
        
        # Check for errors
        if "error" in result:
            error_message = result["error"]
            logger.error(f"Error in AI translate: {error_message}")
            return jsonify({"error": error_message}), 500
        
        # Return the result
        return jsonify(result)
        
    except Exception as e:
        logger.exception(f"Error handling AI translate request: {str(e)}")
        return jsonify({"error": f"Error processing your request: {str(e)}"}), 500

# Add a route to view the workflow for a specific conversation
@app.route('/utility/workflow/<conversation_id>', methods=['GET'])
def get_workflow(conversation_id):
    """Get the workflow details for a specific conversation"""
    try:
        workflow = conversation_logger.get_workflow(conversation_id)
        return jsonify(workflow)
    except Exception as e:
        logger.error(f"Error retrieving workflow: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error retrieving workflow: {str(e)}"}), 500

# Add a route to view the workflow for the current conversation
@app.route('/utility/workflow', methods=['GET'])
def get_current_workflow():
    """Get the workflow details for the current conversation"""
    try:
        workflow = conversation_logger.get_workflow()
        return jsonify(workflow)
    except Exception as e:
        logger.error(f"Error retrieving workflow: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error retrieving workflow: {str(e)}"}), 500

# Add a utility endpoint for executing a specific SQL query
@utility_ns.route('/execute_sql')
class ExecuteSQL(Resource):
    @utility_ns.doc('execute_sql')
    @utility_ns.expect(api.model('ExecuteSQLRequest', {
        'query': fields.String(required=True, description='SQL query to execute')
    }))
    @utility_ns.response(200, 'Success')
    @utility_ns.response(400, 'Bad Request')
    @utility_ns.response(500, 'Internal Server Error')
    def post(self):
        """Execute a specific SQL query"""
        try:
            data = request.json
            query = data.get('query')
            
            if not query:
                return {"error": "No query provided"}, 400
            
            # Execute the query
            from trino_executor import TrinoExecutor
            executor = TrinoExecutor()
            result = executor.execute_query(query)
            
            return result
        except Exception as e:
            logger.error("Error executing query: %s", str(e), exc_info=True)
            return {"error": f"Error executing query: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True) 