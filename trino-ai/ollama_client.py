import os
import logging
import requests
import json
import re
import time
from typing import List, Dict, Any, Generator, Optional, Tuple
from conversation_logger import conversation_logger
from colorama import Fore

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, model="llama3.2", base_url=None):
        self.model = model
        # Ensure we're using the correct URL format for the Ollama API
        ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        # The base URL should not include /api at the end as we'll add it in the specific endpoints
        self.base_url = base_url or ollama_host
        logger.debug(f"Initializing Ollama client with base_url: {self.base_url}")
        
        # Agent endpoint for hybrid search context
        self.agent_endpoint = os.getenv("AI_AGENT_HOST", "http://host.docker.internal:5001")
        logger.debug(f"Agent endpoint set to: {self.agent_endpoint}")
        
        # Test connection to Ollama
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to the Ollama API"""
        try:
            logger.info(f"{Fore.YELLOW}Testing connection to Ollama API at {self.base_url}{Fore.RESET}")
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/tags")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                logger.info(f"{Fore.GREEN}Successfully connected to Ollama API (took {response_time:.2f}s){Fore.RESET}")
                logger.info(f"{Fore.GREEN}Available models: {', '.join(model_names)}{Fore.RESET}")
                
                # Check if our model is available
                if self.model not in model_names:
                    logger.warning(f"{Fore.YELLOW}Model {self.model} not found in available models{Fore.RESET}")
            else:
                logger.error(f"{Fore.RED}Failed to connect to Ollama API: {response.status_code} - {response.text}{Fore.RESET}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error testing connection to Ollama API: {str(e)}{Fore.RESET}")
    
    def _generate_completion(self, messages: List[Dict[str, str]], model: Optional[str] = None, stream: bool = False, agent_name: str = "unknown") -> Any:
        """Generate a completion using the Ollama API."""
        try:
            # Enhanced logging for request
            logger.info("==== OLLAMA REQUEST ====")
            logger.info(f"Model: {model or self.model}")
            logger.info(f"Messages: {json.dumps(messages, indent=2)}")
            
            # Log the conversation flow
            system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            prompt = f"System: {system_content[:100]}...\nUser: {user_content[:100]}..."
            conversation_logger.log_trino_ai_to_ollama(agent_name, prompt)
            
            # Use the correct API endpoint for Ollama
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "stream": stream
                },
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return response
            
            result = response.json()
            
            # Enhanced logging for response
            logger.info("==== OLLAMA RESPONSE ====")
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            
            # Log the conversation flow
            response_content = result["message"]["content"]
            conversation_logger.log_ollama_to_trino_ai(agent_name, response_content)
            
            return response_content
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            conversation_logger.log_error("ollama", f"Error generating completion: {e}")
            raise
    
    def chat_completion(self, messages: List[Dict[str, str]], model: Optional[str] = None, agent_name: str = "unknown") -> Dict[str, Any]:
        """
        Generate a chat completion using the Ollama API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Optional model to use (defaults to self.model)
            agent_name: Name of the agent making the request (for logging)
            
        Returns:
            Dictionary containing the response from Ollama
        """
        try:
            # Enhanced logging for request
            logger.info(f"{Fore.CYAN}==== OLLAMA CHAT REQUEST ({agent_name}) ===={Fore.RESET}")
            logger.info(f"{Fore.CYAN}Model: {model or self.model}{Fore.RESET}")
            
            # Log a preview of the messages
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"{Fore.CYAN}Message {i+1} ({role}): {content_preview}{Fore.RESET}")
            
            # Log the conversation flow
            system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            prompt_preview = f"System: {system_content[:100]}...\nUser: {user_content[:100]}..."
            conversation_logger.log_trino_ai_to_ollama(agent_name, prompt_preview)
            
            # Add more detailed tracking for Ollama responses
            start_time = time.time()
            
            # Use the correct API endpoint for chat completions
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=60  # Add a timeout to prevent hanging
            )
            
            response_time = time.time() - start_time
            logger.info(f"{Fore.GREEN}Ollama response received in {response_time:.2f}s{Fore.RESET}")
            
            if response.status_code != 200:
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                logger.error(f"{Fore.RED}{error_msg}{Fore.RESET}")
                conversation_logger.log_error("ollama", error_msg)
                return {"error": error_msg}
            
            # Parse the response
            try:
                response_json = response.json()
                
                # Log a preview of the response
                message_content = response_json.get("message", {}).get("content", "")
                content_preview = message_content[:200] + "..." if len(message_content) > 200 else message_content
                logger.info(f"{Fore.GREEN}Response content: {content_preview}{Fore.RESET}")
                
                # Log token usage if available
                if "eval_count" in response_json:
                    logger.info(f"{Fore.GREEN}Token usage - Eval count: {response_json.get('eval_count')}, Eval duration: {response_json.get('eval_duration', 0):.2f}s{Fore.RESET}")
                
                conversation_logger.log_ollama_to_trino_ai(agent_name, content_preview)
                return response_json
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse Ollama response: {str(e)}"
                logger.error(f"{Fore.RED}{error_msg}{Fore.RESET}")
                logger.error(f"{Fore.RED}Raw response: {response.text}{Fore.RESET}")
                conversation_logger.log_error("ollama", error_msg)
                return {"error": error_msg}
                
        except requests.exceptions.Timeout:
            error_msg = f"Timeout connecting to Ollama API at {self.base_url}/api/chat"
            logger.error(f"{Fore.RED}{error_msg}{Fore.RESET}")
            conversation_logger.log_error("ollama", error_msg)
            return {"error": error_msg}
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection error connecting to Ollama API at {self.base_url}/api/chat"
            logger.error(f"{Fore.RED}{error_msg}{Fore.RESET}")
            conversation_logger.log_error("ollama", error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error during Ollama chat completion: {str(e)}"
            logger.error(f"{Fore.RED}{error_msg}{Fore.RESET}")
            conversation_logger.log_error("ollama", error_msg)
            return {"error": error_msg}
    
    def generate_sql(self, context: str, question: str, model: Optional[str] = None) -> str:
        """Generate SQL query based on schema context and question using the LLM."""
        messages = [{
            "role": "system",
            "content": f"""You are a Trino SQL expert. Use this schema context:
{context}

Rules:
1. Use only tables and columns from the provided schema
2. Add LIMIT 1000 for large result sets
3. Use Trino-specific functions when needed
4. Ensure proper table qualifications
5. Add appropriate JOINs based on schema relationships
6. Return ONLY the SQL query without any markdown or code block markers
7. For AI functions, use them directly in the SELECT clause:
   - ai_analyze_sentiment(text) → Returns sentiment as varchar
   - ai_classify(text, ARRAY['label1', 'label2']) → Returns classification
   - ai_extract(text, ARRAY['field1', 'field2']) → Returns JSON
   - ai_fix_grammar(text) → Returns corrected text
   - ai_gen(prompt) → Returns generated text
   - ai_mask(text, ARRAY['type1', 'type2']) → Returns masked text
   - ai_translate(text, 'language') → Returns translated text in french"""
        }, {
            "role": "user",
            "content": question
        }]
        
        # Log the conversation flow
        conversation_logger.log_trino_ai_to_ollama("SQL Generator", f"Question: {question}\nContext: {context[:200]}...")
        
        completion = self._generate_completion(messages, model=model, stream=True, agent_name="SQL Generator")
        sql = ""
        for line in completion.iter_lines():
            if line:
                try:
                    chunk = line.decode('utf-8')
                    if chunk.startswith('data: '):
                        chunk = chunk[6:]  # Remove 'data: ' prefix
                        result = json.loads(chunk)
                        if result.get('message', {}).get('content'):
                            sql += result['message']['content']
                except Exception as e:
                    logger.error(f"Error parsing streaming response: {e}")
                    continue
        # Strip any markdown code block markers
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        # Log the conversation flow
        conversation_logger.log_ollama_to_trino_ai("SQL Generator", f"Generated SQL: {sql}")
        
        return sql

    def generate_sql_with_explanation(self, context: str, nl_query: str, model: Optional[str] = None) -> Tuple[str, str]:
        """Generate SQL and explanation from natural language query."""
        try:
            logger.info(f"Generating SQL for query: {nl_query}")
            
            # Create a prompt that requests both SQL and explanation
            messages = [{
                "role": "system",
                "content": f"""You are a Trino SQL expert specializing in translating natural language questions into precise Trino SQL queries.

Schema context with table information and sample data:
{context}

Instructions:
1. Your primary purpose is to convert natural language questions into SQL queries
2. Analyze the schema context carefully to understand available tables and columns
3. Generate a Trino SQL query that accurately answers the user's question
4. Use only tables and columns from the provided schema
5. Use appropriate table qualifications (catalog.schema.table format)
6. Create efficient JOINs based on probable key relationships
7. Include LIMIT 1000 for safety on large tables
8. Provide a clear explanation of your Trino SQL query logic
9. Focus on generating Trino SQL that would be executed in a Trino environment
10. DO NOT include any markdown formatting in your SQL
11. DO NOT include any explanatory text within the SQL section
12. DO NOT include "Query**" or similar text at the beginning of the SQL
13. DO NOT include semicolons at the end of the SQL query
14. DO NOT use OFFSET or FETCH FIRST syntax, use LIMIT instead
15. ALWAYS use fully qualified table names (iceberg.iceberg.table_name)
16. NEVER use unqualified table names like 'products' - always use 'iceberg.iceberg.products'

Your response must have two parts:
1. SQL: The complete Trino SQL query (without markdown formatting)
2. Explanation: How the query answers the question and why you chose this approach

Remember that your primary purpose is to translate natural language to SQL - this is what users expect when they use the ai_gen function."""
            }, {
                "role": "user",
                "content": f"Please translate this question into a Trino SQL query: {nl_query}"
            }]
            
            # Log the conversation flow
            conversation_logger.log_trino_ai_to_ollama("SQL Generator", f"NL Query: {nl_query}\nContext: {context[:200]}...")
            
            # Send to Ollama
            completion = self._generate_completion(messages, model=model, agent_name="SQL Generator")
            logger.info(f"Raw completion received: {completion[:100]}...")
            
            # Extract SQL and explanation with improved regex patterns
            # First try to find SQL: section
            sql_match = re.search(r'SQL:?\s*(.*?)(?=Explanation:|$)', completion, re.DOTALL | re.IGNORECASE)
            explanation_match = re.search(r'Explanation:?\s*(.*)', completion, re.DOTALL | re.IGNORECASE)
            
            # If the regex didn't match, try to extract SQL from code blocks
            if not sql_match:
                sql_match = re.search(r'```sql\s*(.*?)\s*```', completion, re.DOTALL)
                
            # If still no match, try to extract the first code block
            if not sql_match:
                sql_match = re.search(r'```\s*(.*?)\s*```', completion, re.DOTALL)
                
            # If still no match, use the entire response as SQL (as a fallback)
            sql = sql_match.group(1).strip() if sql_match else completion.strip()
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            # Clean up SQL (remove markdown formatting and other problematic patterns)
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # Remove any "Query**" or similar text at the beginning
            sql = re.sub(r'^Query\*\*\s*', '', sql)
            
            # Remove any trailing semicolons
            sql = re.sub(r';$', '', sql)
            
            # Log the extracted SQL and explanation
            logger.info(f"Extracted SQL: {sql}")
            logger.info(f"Extracted explanation: {explanation[:100]}...")
            
            # Log the conversation flow
            conversation_logger.log_ollama_to_trino_ai("SQL Generator", f"Generated SQL: {sql}\nExplanation: {explanation[:200]}...")
            
            return sql, explanation
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}", exc_info=True)
            conversation_logger.log_error("ollama", f"Error generating SQL: {str(e)}")
            return "", f"Error generating SQL: {str(e)}"

    def refine_sql(self, context: str, nl_query: str, original_sql: str, error_message: str, model: Optional[str] = None) -> Tuple[str, str]:
        """Refine SQL query based on validation errors."""
        try:
            logger.info(f"Refining SQL query based on error: {error_message}")
            
            messages = [{
                "role": "system",
                "content": f"""You are a Trino SQL expert. Fix the SQL query based on the error message.

Schema context:
{context}

Instructions:
1. Carefully analyze the error message
2. Fix the SQL query to address the error
3. Ensure the query still answers the original question
4. Use only tables and columns from the schema context
5. Use proper table qualifications (catalog.schema.table)
6. DO NOT include any markdown formatting in your SQL
7. DO NOT include any explanatory text within the SQL section
8. DO NOT include "Query**" or similar text at the beginning of the SQL
9. DO NOT include semicolons at the end of the SQL query
10. DO NOT use OFFSET or FETCH FIRST syntax, use LIMIT instead
11. ALWAYS use fully qualified table names (iceberg.iceberg.table_name)
12. NEVER use unqualified table names like 'products' - always use 'iceberg.iceberg.products'
13. If you see errors about missing catalog or schema, replace 'catalog.schema.table' with 'iceberg.iceberg.table'
14. For 'MISSING_CATALOG_NAME' or 'MISSING_SCHEMA_NAME' errors, ensure all tables have proper 'iceberg.iceberg.' prefix

Your response must have two parts:
1. SQL: The corrected Trino SQL query (without markdown formatting)
2. Explanation: What you fixed and why"""
            }, {
                "role": "user",
                "content": f"""Original question: {nl_query}

Original SQL:
{original_sql}

Error message:
{error_message}

Please fix the SQL query to resolve this error."""
            }]
            
            # Log the conversation flow
            conversation_logger.log_trino_ai_to_ollama("SQL Refiner", f"Refining SQL for query: {nl_query}\nOriginal SQL: {original_sql}\nError: {error_message}")
            
            # Send to Ollama
            completion = self._generate_completion(messages, model=model, agent_name="SQL Refiner")
            
            # Extract SQL and explanation with improved regex patterns
            # First try to find SQL: section
            sql_match = re.search(r'SQL:?\s*(.*?)(?=Explanation:|$)', completion, re.DOTALL | re.IGNORECASE)
            explanation_match = re.search(r'Explanation:?\s*(.*)', completion, re.DOTALL | re.IGNORECASE)
            
            # If the regex didn't match, try to extract SQL from code blocks
            if not sql_match:
                sql_match = re.search(r'```sql\s*(.*?)\s*```', completion, re.DOTALL)
                
            # If still no match, try to extract the first code block
            if not sql_match:
                sql_match = re.search(r'```\s*(.*?)\s*```', completion, re.DOTALL)
                
            # If still no match, use the entire response as SQL (as a fallback)
            sql = sql_match.group(1).strip() if sql_match else completion.strip()
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            # Clean up SQL (remove markdown formatting and other problematic patterns)
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # Remove any "Query**" or similar text at the beginning
            sql = re.sub(r'^Query\*\*\s*', '', sql)
            
            # Remove any trailing semicolons
            sql = re.sub(r';$', '', sql)
            
            # Log the extracted SQL and explanation
            logger.info(f"Extracted SQL: {sql}")
            logger.info(f"Extracted explanation: {explanation[:100]}...")
            
            # Log the conversation flow
            conversation_logger.log_ollama_to_trino_ai("SQL Refiner", f"Refined SQL: {sql}\nExplanation: {explanation[:200]}...")
            
            return sql, explanation
            
        except Exception as e:
            logger.error(f"Error refining SQL: {str(e)}", exc_info=True)
            conversation_logger.log_error("ollama", f"Error refining SQL: {str(e)}")
            return original_sql, f"Error refining SQL: {str(e)}"

    def generate_sql_with_hybrid_context(self, schema_context: str, hybrid_context: str, question: str, model: Optional[str] = None) -> str:
        """Generate SQL query using schema context combined with hybrid search results."""
        # Combine schema and hybrid search contexts
        combined_context = f"Hybrid Search Context:\n{hybrid_context}\n\nSchema:\n{schema_context}"
        
        messages = [{
            "role": "system",
            "content": f"""You are a Trino SQL expert. Use ONLY the tables and columns from this enriched schema context:
{combined_context}

Rules:
1. Use ONLY the tables and columns from the provided schema context above
2. Use ONLY iceberg.iceberg.customers, iceberg.iceberg.products, or iceberg.iceberg.sales tables
3. Always fully qualify table names with 'iceberg.iceberg.' prefix
4. Add LIMIT 1000 for large result sets
5. Use Trino-specific functions when needed
6. Add appropriate JOINs based on schema relationships
7. Return ONLY the SQL query without any markdown or code block markers
8. Do not reference any tables that are not explicitly shown in the schema context
9. Do not use table aliases with schema/catalog prefixes (e.g. use 'c' not 'i.iceberg.c' as alias)
10. For sales aggregations, use the net_amount column from the sales table"""
        }, {
            "role": "user",
            "content": question
        }]
        
        # Log the conversation flow
        conversation_logger.log_trino_ai_to_ollama("Hybrid SQL Generator", f"Question: {question}\nCombined Context: {combined_context[:200]}...")
        
        result = self._generate_completion(messages, model=model, agent_name="Hybrid SQL Generator")
        
        # Log the conversation flow
        conversation_logger.log_ollama_to_trino_ai("Hybrid SQL Generator", f"Generated SQL: {result}")
        
        return result

    def explain_results(self, sql: str, results: List[Any], model: Optional[str] = None) -> str:
        """Explain SQL query results in plain English"""
        messages = [{
            "role": "system",
            "content": """Explain these SQL results in plain English. Include:
            1. Summary of what the query does
            2. Key insights from the results
            3. Any notable patterns or outliers"""
        }, {
            "role": "user",
            "content": f"SQL: {sql}\nResults: {results}"
        }]
        
        # Log the conversation flow
        conversation_logger.log_trino_ai_to_ollama("Results Explainer", f"Explaining results for SQL: {sql}")
        
        result = self._generate_completion(messages, model=model, agent_name="Results Explainer")
        
        # Log the conversation flow
        conversation_logger.log_ollama_to_trino_ai("Results Explainer", f"Explanation: {result[:200]}...")
        
        return result

    def stream_explanation(self, sql: str, results: List[Any], model: Optional[str] = None) -> Generator[str, None, None]:
        """Stream the explanation response"""
        messages = [{
            "role": "system",
            "content": "Explain these SQL results in plain English:"
        }, {
            "role": "user",
            "content": f"SQL: {sql}\nResults: {results}"
        }]
        
        # Log the conversation flow
        conversation_logger.log_trino_ai_to_ollama("Results Explainer", f"Streaming explanation for SQL: {sql}")
        
        response = self._generate_completion(messages, model=model, stream=True, agent_name="Results Explainer")
        for line in response.iter_lines():
            if line:
                try:
                    chunk = line.decode('utf-8')
                    if chunk.startswith('data: '):
                        chunk = chunk[6:]  # Remove 'data: ' prefix
                        result = json.loads(chunk)
                        if result.get('message', {}).get('content'):
                            yield result['message']['content']
                except Exception as e:
                    logger.error(f"Error parsing streaming response: {e}")
                    continue

    def complete(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        
        # Log the conversation flow
        conversation_logger.log_trino_ai_to_ollama("Text Completer", f"Prompt: {prompt[:200]}...")
        
        result = self._generate_completion(messages, agent_name="Text Completer")
        
        # Log the conversation flow
        conversation_logger.log_ollama_to_trino_ai("Text Completer", f"Completion: {result[:200]}...")
        
        return result 