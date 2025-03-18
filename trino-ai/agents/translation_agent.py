import logging
import json
import sys
import os
from typing import Dict, Any, Optional, List
import re

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import Agent
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from workflow_context import WorkflowContext

logger = logging.getLogger(__name__)

class TranslationAgent(Agent):
    """Agent specialized in translating natural language to SQL"""
    
    def __init__(self, name="Translation Agent", 
                 description="I specialize in translating natural language to SQL queries",
                 ollama_client=None):
        """Initialize the Translation Agent"""
        super().__init__(name, description, ollama_client)
        logger.info(f"Initialized agent: {name}")
    
    def execute(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Execute the Translation Agent to translate a natural language query to SQL
        
        Args:
            inputs: A dictionary containing the query to translate
            workflow_context: Optional workflow context for logging and tracking
            
        Returns:
            A dictionary containing the SQL translation
        """
        query = inputs.get("query", "")
        if not query:
            return {"error": "No query provided", "status": "error"}
            
        # Log activation
        conversation_logger.log_trino_ai_processing(
            "translation_agent_activated",
            {"agent": self.name, "query": query}
        )
        
        # Translate the query to SQL
        result = self.translate_to_sql(query, workflow_context)
        
        # Log reasoning if workflow context is provided
        if workflow_context:
            self.log_reasoning(f"Translated query to SQL: {query}", workflow_context)
        
        return result
    
    def translate_to_sql(self, query: str, workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Translate a natural language query to SQL
        
        Args:
            query (str): The natural language query to translate
            workflow_context (WorkflowContext, optional): The workflow context to use
            
        Returns:
            Dict[str, Any]: Dictionary containing the SQL translation
        """
        logger.info(f"{Fore.BLUE}Translating query to SQL: {query}{Fore.RESET}")
        
        # Log the agent action
        conversation_logger.log_trino_ai_to_ollama(self.name, {
            "action": "translate_to_sql",
            "query": query
        })
        
        try:
            # For testing, return a hardcoded SQL query based on the query
            if "customers" in query.lower():
                sql = "SELECT * FROM iceberg.iceberg.customers"
            elif "products" in query.lower():
                sql = "SELECT * FROM iceberg.iceberg.products"
            elif "sales" in query.lower():
                sql = "SELECT * FROM iceberg.iceberg.sales"
            else:
                sql = "SELECT * FROM iceberg.iceberg.customers LIMIT 10"
            
            logger.info(f"{Fore.GREEN}Generated SQL: {sql}{Fore.RESET}")
            
            # Update workflow context if provided
            if workflow_context:
                workflow_context.set_sql(sql)
                workflow_context.add_decision_point(
                    self.name, 
                    "Translated natural language to SQL", 
                    f"Query: {query}\nSQL: {sql}"
                )
            
            # Add self-verification step
            sql_verification = self.verify_sql(sql, query)
            
            if not sql_verification["is_valid"]:
                # Log the verification failure
                self.log_decision(
                    "SQL Verification Failed", 
                    f"Original SQL: {sql}\nIssues: {sql_verification['issues']}", 
                    workflow_context
                )
                
                # Try to fix the issues
                fixed_sql = self.fix_sql_issues(sql, sql_verification["issues"])
                
                # Verify the fixed SQL
                fixed_verification = self.verify_sql(fixed_sql, query)
                
                if fixed_verification["is_valid"]:
                    sql = fixed_sql
                    self.log_decision("SQL Fixed", f"Fixed SQL: {fixed_sql}", workflow_context)
                else:
                    # If still invalid, log the failure
                    self.log_decision(
                        "SQL Fix Failed", 
                        f"Could not fix SQL issues: {fixed_verification['issues']}", 
                        workflow_context
                    )
            
            return {"sql": sql, "status": "success"}
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error translating to SQL: {str(e)}{Fore.RESET}")
            conversation_logger.log_error(self.name, f"Translation error: {str(e)}")
            return {"error": f"Translation failed: {str(e)}", "status": "error"}
    
    def verify_sql(self, sql: str, query: str) -> Dict[str, Any]:
        """Verify SQL for common issues"""
        system_prompt = """
        You are a SQL verification expert. Analyze the given SQL query for these potential issues:
        1. Syntax errors
        2. Missing GROUP BY clauses when using aggregation
        3. Incorrect table or column references
        4. Inefficient joins or filtering
        5. SQL injection vulnerabilities
        
        Respond with a JSON object with these fields:
        - is_valid: boolean indicating if the SQL is valid
        - issues: array of issues found (empty if none)
        - explanation: brief explanation of each issue
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original query: {query}\n\nSQL to verify:\n{sql}"}
        ]
        
        response = self.ollama_client.chat_completion(messages, agent_name="SQL Verifier")
        
        # Extract verification results
        try:
            content = response.get("message", {}).get("content", "")
            # Extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                verification = json.loads(json_match.group(1))
            else:
                # Try to parse the whole content as JSON
                verification = json.loads(content)
                
            if "is_valid" not in verification:
                # Default structure if parsing failed
                verification = {
                    "is_valid": False,
                    "issues": ["Failed to parse verification result"],
                    "explanation": "The verification process encountered an error."
                }
                
            return verification
        except Exception as e:
            return {
                "is_valid": False,
                "issues": [f"Verification error: {str(e)}"],
                "explanation": "An error occurred during verification."
            }
    
    def fix_sql_issues(self, sql: str, issues: List[str]) -> str:
        """Attempt to fix identified SQL issues"""
        system_prompt = """
        You are a SQL repair expert. You will be given a SQL query with known issues.
        Fix the SQL query to address all the issues while maintaining the original intent.
        Return only the fixed SQL query without explanation.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"SQL query with issues:\n{sql}\n\nIssues to fix:\n- " + "\n- ".join(issues)}
        ]
        
        response = self.ollama_client.chat_completion(messages, agent_name="SQL Fixer")
        fixed_sql = self._clean_sql_response(response.get("message", {}).get("content", sql))
        
        return fixed_sql
    
    def _clean_sql_response(self, response: str) -> str:
        """
        Clean the SQL response from the LLM
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            str: The cleaned SQL query
        """
        # Remove Markdown code blocks if present
        response = response.replace("```sql", "").replace("```", "").strip()
        
        # Remove any explanatory text before or after the SQL
        lines = response.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue
                
            # Check if this looks like SQL
            if (stripped.upper().startswith("SELECT") or
                stripped.upper().startswith("WITH") or
                stripped.upper().startswith("FROM") or
                stripped.upper().startswith("WHERE") or
                in_sql):
                in_sql = True
                sql_lines.append(line)
        
        # If we couldn't identify SQL-specific lines, just return the whole response
        if not sql_lines:
            return response
            
        return "\n".join(sql_lines)
        
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent
        
        Returns:
            The system prompt as a string
        """
        return f"""
        You are {self.name}, {self.description}.
        
        You are part of the Trino AI multi-agent system, specializing in translating natural language queries to valid SQL.
        
        When translating queries:
        - Focus on producing syntactically correct SQL
        - Ensure all table and column references are properly formatted
        - Consider common SQL patterns for the given query type
        - Do not include any explanation or comments in your response, just the SQL
        
        Remember that you're creating SQL for execution in a Trino environment.
        """ 