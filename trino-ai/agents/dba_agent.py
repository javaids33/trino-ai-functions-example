import logging
import json
import re
from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import Agent
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from workflow_context import WorkflowContext

logger = logging.getLogger(__name__)

class DBAAgent(Agent):
    """Agent that analyzes natural language queries to identify relevant tables and columns"""
    
    def __init__(self, name: str = "DBA Agent", description: str = "Analyzes natural language queries to identify relevant tables and columns", 
                 ollama_client: Optional[OllamaClient] = None, tools: Optional[Dict[str, Any]] = None):
        super().__init__(name, description, ollama_client)
        self.tools = tools or {}
        logger.info(f"{Fore.CYAN}DBA Agent initialized with {len(self.tools)} tools{Fore.RESET}")
    
    def execute(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Analyze a natural language query to identify necessary tables, columns, joins, etc.
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query to analyze
                - schema_context: Optional schema context (if not provided, will be retrieved)
            workflow_context: Optional workflow context for logging and tracking
                
        Returns:
            Dictionary with analysis results including tables, columns, joins, etc.
        """
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        
        logger.info(f"{Fore.CYAN}DBA Agent analyzing query: {query}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("dba_agent_analysis_start", {
            "query": query,
            "schema_context_length": len(schema_context),
            "schema_context_preview": schema_context[:200] + "..." if len(schema_context) > 200 else schema_context
        })
        
        # If no schema context was provided, retrieve it
        if not schema_context and "get_schema_context" in self.tools:
            logger.info(f"{Fore.YELLOW}Retrieving schema context for query{Fore.RESET}")
            context_tool = self.tools["get_schema_context"]
            context_result = context_tool.execute({"query": query})
            schema_context = context_result.get("schema_context", "")
            logger.info(f"{Fore.GREEN}Retrieved schema context ({len(schema_context)} chars){Fore.RESET}")
            conversation_logger.log_trino_ai_processing("dba_agent_schema_context_retrieved", {
                "schema_context_length": len(schema_context),
                "schema_context_preview": schema_context[:200] + "..." if len(schema_context) > 200 else schema_context
            })
            
            # Update workflow context with schema context if provided
            if workflow_context:
                workflow_context.set_schema_context(schema_context)
        
        # Get conversation context if available
        conversation_context = ""
        if workflow_context:
            conversation_context = self.get_conversation_context(workflow_context)
        
        # Prepare the prompt for the LLM
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
        Natural Language Query: {query}
        
        Schema Context:
        {schema_context}
        
        {f"Conversation Context:\n{conversation_context}" if conversation_context else ""}
        
        Analyze the natural language query and identify the following:
        1. Tables needed to answer the query (with justification for why each table is needed)
        2. Columns needed from each table (with purpose - filter, join, display, aggregate)
        3. Any joins required between tables (with EXPLICIT join columns and join type)
        4. Any filters or conditions (with operators and values)
        5. Any aggregations or groupings (with function and purpose)
        
        For joins, be extremely explicit about which tables are being joined and on which columns.
        For each join specify:
        - Left table name
        - Left column name
        - Join type (INNER, LEFT, RIGHT)
        - Right table name
        - Right column name
        - Reason for the join
        
        Provide your analysis in JSON format.
        """
        
        # Get the analysis from the LLM
        logger.info(f"{Fore.YELLOW}Sending query to LLM for analysis{Fore.RESET}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="dba_agent")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("dba_agent", f"LLM error: {response['error']}")
                return {"error": response["error"]}
            
            analysis_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received analysis from LLM ({len(analysis_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("dba_agent", analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text)
            
            # Extract JSON from the response
            try:
                # Try to find JSON in the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', analysis_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find any JSON-like structure
                    json_match = re.search(r'({[\s\S]*})', analysis_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Just try to parse the whole thing
                        json_str = analysis_text
                
                # Clean up common formatting issues before parsing
                # Remove spaces between dots in table names (e.g., "iceberg". "table" -> "iceberg.table")
                json_str = re.sub(r'(["\']iceberg["\'])\s*\.\s*(["\']?[a-zA-Z_]+["\']?)', r'\1.\2', json_str)
                json_str = re.sub(r'(["\']iceberg["\'])\s*\.\s*(["\']?iceberg["\']?)\s*\.\s*(["\']?[a-zA-Z_]+["\']?)', r'\1.\2.\3', json_str)
                
                # Fix missing quotes around property names
                json_str = re.sub(r'([{,]\s*)([a-zA-Z_]+)(\s*:)', r'\1"\2"\3', json_str)
                
                # Fix missing commas in arrays
                json_str = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1,\n\2', json_str)
                
                logger.debug(f"Cleaned JSON string: {json_str}")
                
                # Parse the cleaned JSON
                analysis_json = json.loads(json_str)
                
                logger.info(f"{Fore.GREEN}Successfully parsed analysis JSON{Fore.RESET}")
                logger.info(f"{Fore.CYAN}Analysis: {json.dumps(analysis_json, indent=2)}{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("dba_agent_analysis_complete", {
                    "tables": analysis_json.get("tables", []),
                    "columns": analysis_json.get("columns", []),
                    "joins": analysis_json.get("joins", []),
                    "filters": analysis_json.get("filters", []),
                    "aggregations": analysis_json.get("aggregations", [])
                })
                
                return {
                    "tables": analysis_json.get("tables", []),
                    "columns": analysis_json.get("columns", []),
                    "joins": analysis_json.get("joins", []),
                    "filters": analysis_json.get("filters", []),
                    "aggregations": analysis_json.get("aggregations", []),
                    "schema_context": schema_context
                }
                
            except Exception as e:
                logger.error(f"{Fore.RED}Error parsing analysis JSON: {str(e)}{Fore.RESET}")
                logger.error(f"{Fore.RED}Raw analysis text: {analysis_text}{Fore.RESET}")
                conversation_logger.log_error("dba_agent", f"JSON parsing error: {str(e)}")
                
                # Return a best-effort analysis
                return {
                    "error": f"Failed to parse analysis: {str(e)}",
                    "raw_analysis": analysis_text,
                    "schema_context": schema_context
                }
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error during DBA analysis: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("dba_agent", f"Execution error: {str(e)}")
            return {"error": f"DBA analysis failed: {str(e)}"}
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the DBA agent"""
        return """
        You are an expert Database Administrator and Data Analyst. Your task is to analyze natural language queries
        and identify the database objects (tables, columns, etc.) needed to answer them.
        
        You have deep knowledge of SQL, database schema design, and query optimization. You can identify:
        - Which tables are relevant to a query
        - Which columns are needed from each table
        - How tables should be joined
        - What filters should be applied
        - What aggregations or calculations are needed
        
        Your analysis should be thorough, precise, and focused on only the elements needed to answer the query.
        
        IMPORTANT: Always provide your analysis in the exact JSON format requested. Follow these strict formatting rules:
        1. Use proper JSON syntax with no extra spaces in property names or values
        2. For table names, use the format "iceberg.iceberg.table_name" (no spaces between dots)
        3. Make sure all strings are properly quoted
        4. Ensure all arrays and objects have matching brackets and braces
        5. Use commas correctly to separate items in arrays and properties in objects
        """

    def analyze_query(self, nlq: str, schema_context: str) -> Dict[str, Any]:
        """
        Analyze a natural language query to identify relevant tables, columns, etc.
        
        Args:
            nlq: The natural language query to analyze
            schema_context: The schema context to use for analysis
            
        Returns:
            A dictionary containing the analysis results
        """
        logger.info(f"{Fore.BLUE}DBA Agent analyzing query: {nlq}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}DBA Agent using schema context: {schema_context[:100]}...{Fore.RESET}")
        
        # Log the agent action
        conversation_logger.log_trino_ai_to_ollama(self.name, {
            "action": "analyze_query",
            "nlq": nlq,
            "schema_context": schema_context[:200] + "..." if len(schema_context) > 200 else schema_context
        })
        
        # Log the reasoning process in more detail
        logger.info(f"{Fore.BLUE}DBA Agent reasoning process:{Fore.RESET}")
        logger.info(f"{Fore.BLUE}1. Identifying query intent: {nlq}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}2. Scanning available schema for relevant tables{Fore.RESET}")
        logger.info(f"{Fore.BLUE}3. Determining required columns and relationships{Fore.RESET}")
        logger.info(f"{Fore.BLUE}4. Identifying filters, aggregations and sort criteria{Fore.RESET}")
        
        # Build the prompt for the LLM
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
        Natural Language Query: {nlq}
        
        Schema Context:
        {schema_context}
        
        Analyze the natural language query and identify the following:
        1. Tables needed to answer the query
        2. Columns needed from each table
        3. Any joins required between tables
        4. Any filters or conditions
        5. Any aggregations or groupings
        
        Provide your analysis in JSON format.
        """
        
        # Get the analysis from the LLM
        logger.info(f"{Fore.YELLOW}Sending query to LLM for analysis{Fore.RESET}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="dba_agent")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("dba_agent", f"LLM error: {response['error']}")
                return {"error": response["error"]}
            
            analysis_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received analysis from LLM ({len(analysis_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("dba_agent", analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text)
            
            # Extract JSON from the response
            try:
                # Try to find JSON in the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', analysis_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find any JSON-like structure
                    json_match = re.search(r'({[\s\S]*})', analysis_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Just try to parse the whole thing
                        json_str = analysis_text
                
                # Clean up common formatting issues before parsing
                # Remove spaces between dots in table names (e.g., "iceberg". "table" -> "iceberg.table")
                json_str = re.sub(r'(["\']iceberg["\'])\s*\.\s*(["\']?[a-zA-Z_]+["\']?)', r'\1.\2', json_str)
                json_str = re.sub(r'(["\']iceberg["\'])\s*\.\s*(["\']?iceberg["\']?)\s*\.\s*(["\']?[a-zA-Z_]+["\']?)', r'\1.\2.\3', json_str)
                
                # Fix missing quotes around property names
                json_str = re.sub(r'([{,]\s*)([a-zA-Z_]+)(\s*:)', r'\1"\2"\3', json_str)
                
                # Fix missing commas in arrays
                json_str = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1,\n\2', json_str)
                
                logger.debug(f"Cleaned JSON string: {json_str}")
                
                # Parse the cleaned JSON
                analysis_json = json.loads(json_str)
                
                logger.info(f"{Fore.GREEN}Successfully parsed analysis JSON{Fore.RESET}")
                logger.info(f"{Fore.CYAN}Analysis: {json.dumps(analysis_json, indent=2)}{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("dba_agent_analysis_complete", {
                    "tables": analysis_json.get("tables", []),
                    "columns": analysis_json.get("columns", []),
                    "joins": analysis_json.get("joins", []),
                    "filters": analysis_json.get("filters", []),
                    "aggregations": analysis_json.get("aggregations", [])
                })
                
                return {
                    "tables": analysis_json.get("tables", []),
                    "columns": analysis_json.get("columns", []),
                    "joins": analysis_json.get("joins", []),
                    "filters": analysis_json.get("filters", []),
                    "aggregations": analysis_json.get("aggregations", []),
                    "schema_context": schema_context
                }
                
            except Exception as e:
                logger.error(f"{Fore.RED}Error parsing analysis JSON: {str(e)}{Fore.RESET}")
                logger.error(f"{Fore.RED}Raw analysis text: {analysis_text}{Fore.RESET}")
                conversation_logger.log_error("dba_agent", f"JSON parsing error: {str(e)}")
                
                # Return a best-effort analysis
                return {
                    "error": f"Failed to parse analysis: {str(e)}",
                    "raw_analysis": analysis_text,
                    "schema_context": schema_context
                }
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error during DBA analysis: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("dba_agent", f"Execution error: {str(e)}")
            return {"error": f"DBA analysis failed: {str(e)}"} 