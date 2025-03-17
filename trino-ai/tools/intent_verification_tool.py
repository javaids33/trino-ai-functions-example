import logging
import json
from typing import Dict, Any, List, Optional
import re

from tools.base_tool import BaseTool
from conversation_logger import conversation_logger

logger = logging.getLogger(__name__)

class IntentVerificationTool(BaseTool):
    """Tool for verifying that SQL matches the user's intent"""
    
    def __init__(self, name: str = "Intent Verification Tool", 
                 description: str = "Verifies that SQL matches the user's intent",
                 ollama_client=None):
        """
        Initialize the intent verification tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            ollama_client: An Ollama client for LLM calls
        """
        super().__init__(name, description)
        self.ollama_client = ollama_client
        
        if self.ollama_client is None:
            logger.warning("IntentVerificationTool initialized without an Ollama client. Client must be provided before execution.")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the intent verification tool
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - sql: The SQL to verify
                - schema_context: Optional schema context
                
        Returns:
            Dictionary containing:
                - matches_intent: Boolean indicating if SQL matches intent
                - missing_aspects: List of aspects missing from the SQL
                - intent_score: Score from 0-10 indicating how well SQL matches intent
        """
        query = inputs.get("query", "")
        sql = inputs.get("sql", "")
        schema_context = inputs.get("schema_context", "")
        
        if not query or not sql:
            logger.error("Query and SQL must be provided")
            return {
                "matches_intent": False,
                "missing_aspects": ["No query or SQL provided"],
                "intent_score": 0
            }
        
        # If we have an Ollama client, use it to verify intent
        if self.ollama_client:
            try:
                # Create a system prompt for intent verification
                system_prompt = """
                You are an expert SQL reviewer. Your task is to verify that a SQL query correctly addresses a user's natural language query.
                
                Analyze both the natural language query and the SQL query carefully. Determine if the SQL query correctly addresses all aspects of the user's intent.
                
                Return a JSON object with the following fields:
                - matches_intent: Boolean indicating if the SQL fully matches the user's intent
                - missing_aspects: List of aspects from the user's query that are not addressed in the SQL
                - intent_score: Score from 0-10 indicating how well the SQL matches the user's intent
                """
                
                # Create a variable for schema context to avoid backslash in the f-string expression
                schema_context_str = f"Schema Context:\n{schema_context}" if schema_context else ""
                
                # Create a user prompt for intent verification
                user_prompt = f"""
                Natural Language Query: {query}
                
                SQL Query:
                ```sql
                {sql}
                ```
                
                {schema_context_str}
                
                Does this SQL query correctly address all aspects of the user's natural language query?
                Provide your analysis in JSON format.
                """
                
                # Call the LLM
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.ollama_client.chat_completion(messages, agent_name="intent_verification")
                
                if "error" in response:
                    logger.error(f"Error from LLM: {response['error']}")
                    return {
                        "matches_intent": False,
                        "missing_aspects": [f"Error from LLM: {response['error']}"],
                        "intent_score": 0
                    }
                
                response_text = response.get("message", {}).get("content", "")
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON in the response (with markdown)
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to find JSON without markdown
                        json_match = re.search(r'(\{[\s\S]*\})', response_text)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = response_text
                    
                    result = json.loads(json_str)
                    
                    # Ensure all required fields are present
                    if "matches_intent" not in result:
                        result["matches_intent"] = False
                    if "missing_aspects" not in result:
                        result["missing_aspects"] = []
                    if "intent_score" not in result:
                        result["intent_score"] = 0
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error parsing LLM response: {str(e)}")
                    return {
                        "matches_intent": False,
                        "missing_aspects": [f"Error parsing LLM response: {str(e)}"],
                        "intent_score": 0
                    }
                
            except Exception as e:
                logger.error(f"Error during intent verification: {str(e)}")
                return {
                    "matches_intent": False,
                    "missing_aspects": [f"Error during intent verification: {str(e)}"],
                    "intent_score": 0
                }
        
        # If we don't have an Ollama client, return a default response
        logger.warning("No Ollama client provided, returning default response")
        return {
            "matches_intent": True,
            "missing_aspects": [],
            "intent_score": 10
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query"
                },
                "sql": {
                    "type": "string",
                    "description": "The SQL to verify"
                },
                "schema_context": {
                    "type": "string",
                    "description": "The schema context"
                }
            },
            "required": ["query", "sql"]
        }
