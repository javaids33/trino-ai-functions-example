import os
import logging
import requests
import json
from typing import List, Dict, Any, Generator, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, model="llama3.2"):
        self.model = model
        self.base_url = f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api"
        logger.debug(f"Initializing Ollama client with base_url: {self.base_url}")
        
        # Agent endpoint for hybrid search context
        self.agent_endpoint = os.getenv("AI_AGENT_HOST", "http://host.docker.internal:5001")
        logger.debug(f"Agent endpoint set to: {self.agent_endpoint}")
    
    def _generate_completion(self, messages: List[Dict[str, str]], model: Optional[str] = None, stream: bool = False) -> Any:
        """Generate a completion using the Ollama API."""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
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
            return result["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    def generate_sql(self, context: str, question: str, model: Optional[str] = None) -> str:
        """Generate SQL query based on schema context and question using the LLM."""
        messages = [{
            "role": "system",
            "content": f"""You are a Trino SQL expert. Use ONLY the tables and columns from this schema context:
{context}

Rules:
1. Use ONLY the tables and columns from the provided schema context above
2. Use ONLY iceberg.iceberg.customers, iceberg.iceberg.products, or iceberg.iceberg.sales tables
3. Always fully qualify table names with 'iceberg.iceberg.' prefix
4. Add LIMIT 1000 for large result sets
5. Use Trino-specific functions when needed
6. Add appropriate JOINs based on schema relationships
7. Return ONLY the SQL query without any markdown or code block markers
8. Do not reference any tables that are not explicitly shown in the schema context
9. Do not use schema/catalog prefixes in table aliases (e.g. use 'c' not 'i.iceberg.c' as alias)
10. For sales aggregations, use the net_amount column from the sales table
11. When using table aliases, use simple letters like 'c', 's', 'p' for clarity"""
        }, {
            "role": "user",
            "content": question
        }]
        
        sql = self._generate_completion(messages, model=model)
        # Strip any markdown code block markers
        sql = sql.replace('```sql', '').replace('```', '').strip()
        return sql

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
        
        return self._generate_completion(messages, model=model)

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
        
        return self._generate_completion(messages, model=model)

    def stream_explanation(self, sql: str, results: List[Any], model: Optional[str] = None) -> Generator[str, None, None]:
        """Stream the explanation response"""
        messages = [{
            "role": "system",
            "content": "Explain these SQL results in plain English:"
        }, {
            "role": "user",
            "content": f"SQL: {sql}\nResults: {results}"
        }]
        
        response = self._generate_completion(messages, model=model, stream=True)
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
        return self._generate_completion(messages) 