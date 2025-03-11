import logging
from typing import Dict, Any, List, Optional
import json
from colorama import Fore
from tools.base_tool import Tool
from ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class QueryIntentVerificationTool(Tool):
    """Tool to verify that the generated SQL fulfills the original query intent"""
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        super().__init__(
            name="Query Intent Verification Tool",
            description="Verifies that generated SQL fulfills the original query intent"
        )
        self.ollama_client = ollama_client
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that the generated SQL fulfills the original query intent
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - sql: The generated SQL
                - schema_context: The schema context
                - dba_analysis: The DBA analysis
                
        Returns:
            Dictionary containing the verification results
        """
        query = inputs.get("query", "")
        sql = inputs.get("sql", "")
        schema_context = inputs.get("schema_context", "")
        dba_analysis = inputs.get("dba_analysis", {})
        
        logger.info(f"{Fore.CYAN}Verifying SQL intent for query: {query}{Fore.RESET}")
        
        if not query or not sql:
            return {
                "matches_intent": False,
                "missing_aspects": ["Empty query or SQL"],
                "suggestions": ["Provide both a query and SQL to verify"]
            }
        
        # Create a prompt for the LLM to verify intent
        system_prompt = """
        You are a query auditor specializing in verifying that SQL queries match their original natural language intent.
        Your task is to analyze a natural language query and its corresponding SQL translation to determine if the SQL
        correctly addresses all aspects of the original question.
        
        Perform a thorough analysis that considers:
        1. Tables - Are all necessary tables included?
        2. Columns - Are all required columns selected or used in filters?
        3. Filters - Are all filters from the question applied?
        4. Joins - Are tables joined correctly based on logical relationships?
        5. Aggregations - Are the correct aggregation functions used?
        6. Sorting - Is data ordered appropriately if the question implies ordering?
        
        Return your analysis in JSON format with these fields:
        - matches_intent: boolean (true if the SQL fulfills the query intent, false otherwise)
        - missing_aspects: array of strings (aspects missing from the SQL)
        - suggestions: array of strings (suggestions to improve the SQL)
        """
        
        user_prompt = f"""
        Natural Language Query: {query}
        
        Generated SQL: 
        {sql}
        
        Schema Context:
        {schema_context}
        
        DBA Analysis:
        {json.dumps(dba_analysis, indent=2)}
        
        Evaluate whether the SQL correctly answers the natural language query and return your analysis in JSON format.
        """
        
        try:
            # Get response from LLM
            response = self.ollama_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                agent_name="Intent Verifier"
            )
            
            content = response.get("message", {}).get("content", "")
            
            # Parse the JSON response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_content = content[json_start:json_end]
                verification_result = json.loads(json_content)
                
                logger.info(f"{Fore.GREEN}Intent verification complete: {verification_result.get('matches_intent', False)}{Fore.RESET}")
                
                return verification_result
            else:
                # Extract information using regex if JSON parsing fails
                import re
                
                matches_intent = "true" in content.lower() and "matches_intent" in content.lower()
                
                missing_aspects = []
                missing_match = re.search(r'missing_aspects.*?\[(.*?)\]', content, re.DOTALL)
                if missing_match:
                    missing_text = missing_match.group(1)
                    missing_aspects = [item.strip(' "\'') for item in missing_text.split(',') if item.strip()]
                
                suggestions = []
                suggestions_match = re.search(r'suggestions.*?\[(.*?)\]', content, re.DOTALL)
                if suggestions_match:
                    suggestions_text = suggestions_match.group(1)
                    suggestions = [item.strip(' "\'') for item in suggestions_text.split(',') if item.strip()]
                
                return {
                    "matches_intent": matches_intent,
                    "missing_aspects": missing_aspects,
                    "suggestions": suggestions,
                    "raw_response": content
                }
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error during intent verification: {str(e)}{Fore.RESET}")
            return {
                "matches_intent": False,
                "missing_aspects": ["Error during verification"],
                "suggestions": ["Try regenerating the SQL"],
                "error": str(e)
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
                    "description": "The generated SQL"
                },
                "schema_context": {
                    "type": "string",
                    "description": "The schema context"
                },
                "dba_analysis": {
                    "type": "object",
                    "description": "The DBA analysis"
                }
            },
            "required": ["query", "sql"]
        } 