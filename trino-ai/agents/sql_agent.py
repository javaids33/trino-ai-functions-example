import logging
import json
import re
from typing import Dict, Any, List, Optional
import sys
import os
import time

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import Agent
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from workflow_context import WorkflowContext

logger = logging.getLogger(__name__)

class SQLAgent(Agent):
    """Agent that generates and refines SQL queries"""
    
    def __init__(self, name: str = "SQL Agent", description: str = "Generates accurate, efficient, and valid SQL queries", 
                 ollama_client: Optional[OllamaClient] = None, tools: Optional[Dict[str, Any]] = None):
        super().__init__(name, description, ollama_client)
        self.tools = tools or {}
        logger.info(f"{Fore.CYAN}SQL Agent initialized with {len(self.tools)} tools{Fore.RESET}")
    
    def execute(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Generate SQL from natural language query and DBA analysis
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - schema_context: The schema context
                - dba_analysis: The DBA analysis results
            workflow_context: Optional workflow context for logging and tracking
                
        Returns:
            Dictionary containing the generated SQL and explanation
        """
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        dba_analysis = inputs.get("dba_analysis", {})
        
        logger.info(f"{Fore.CYAN}SQL Agent generating SQL for query: {query}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("sql_agent_generation_start", {
            "query": query,
            "schema_context_length": len(schema_context),
            "dba_analysis_tables": dba_analysis.get("tables", []),
            "dba_analysis_joins": dba_analysis.get("joins", [])
        })
        
        # Get conversation context if available
        conversation_context = ""
        if workflow_context:
            conversation_context = self.get_conversation_context(workflow_context)
        
        # Prepare the prompt for the LLM
        system_prompt = self.get_system_prompt()
        
        # Create a detailed prompt with the DBA analysis
        tables_info = ""
        if isinstance(dba_analysis.get("tables"), list):
            for table in dba_analysis.get("tables", []):
                if isinstance(table, dict):
                    tables_info += f"- {table.get('name', 'Unknown')}"
                    if table.get("alias"):
                        tables_info += f" (alias: {table.get('alias')})"
                    if table.get("reason"):
                        tables_info += f": {table.get('reason')}"
                    tables_info += "\n"
                elif isinstance(table, str):
                    tables_info += f"- {table}\n"
        
        columns_info = ""
        if isinstance(dba_analysis.get("columns"), list):
            for column in dba_analysis.get("columns", []):
                if isinstance(column, dict):
                    columns_info += f"- {column.get('table', 'Unknown')}.{column.get('name', 'Unknown')}"
                    if column.get("purpose"):
                        columns_info += f": {column.get('purpose')}"
                    columns_info += "\n"
                elif isinstance(column, str):
                    columns_info += f"- {column}\n"
        
        joins_info = ""
        if isinstance(dba_analysis.get("joins"), list):
            for join in dba_analysis.get("joins", []):
                if isinstance(join, dict):
                    joins_info += f"- {join.get('left_table', 'Unknown')}.{join.get('left_column', 'Unknown')} {join.get('join_type', 'JOIN')} {join.get('right_table', 'Unknown')}.{join.get('right_column', 'Unknown')}"
                    if join.get("reason"):
                        joins_info += f": {join.get('reason')}"
                    joins_info += "\n"
                elif isinstance(join, str):
                    joins_info += f"- {join}\n"
        
        filters_info = ""
        if isinstance(dba_analysis.get("filters"), list):
            for filter_item in dba_analysis.get("filters", []):
                if isinstance(filter_item, dict):
                    filters_info += f"- {filter_item.get('table', 'Unknown')}.{filter_item.get('column', 'Unknown')} {filter_item.get('operator', '=')} {filter_item.get('value', '')}"
                    if filter_item.get("reason"):
                        filters_info += f": {filter_item.get('reason')}"
                    filters_info += "\n"
                elif isinstance(filter_item, str):
                    filters_info += f"- {filter_item}\n"
        
        aggregations_info = ""
        if isinstance(dba_analysis.get("aggregations"), list):
            for agg in dba_analysis.get("aggregations", []):
                if isinstance(agg, dict):
                    aggregations_info += f"- {agg.get('function', 'Unknown')}({agg.get('column', 'Unknown')})"
                    if agg.get("purpose"):
                        aggregations_info += f": {agg.get('purpose')}"
                    aggregations_info += "\n"
                elif isinstance(agg, str):
                    aggregations_info += f"- {agg}\n"
        
        user_prompt = f"""
        Natural Language Query: {query}
        
        Schema Context:
        {schema_context}
        
        {f"Conversation Context:\n{conversation_context}" if conversation_context else ""}
        
        DBA Analysis:
        
        Tables:
        {tables_info if tables_info else "No tables identified"}
        
        Columns:
        {columns_info if columns_info else "No columns identified"}
        
        Joins:
        {joins_info if joins_info else "No joins identified"}
        
        Filters:
        {filters_info if filters_info else "No filters identified"}
        
        Aggregations:
        {aggregations_info if aggregations_info else "No aggregations identified"}
        
        Generate a valid Trino SQL query that answers the natural language query based on the DBA analysis.
        Return only the SQL query without any explanations or markdown formatting.
        """
        
        # Get the SQL from the LLM
        logger.info(f"{Fore.YELLOW}Sending query to LLM for SQL generation{Fore.RESET}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("sql_agent", f"LLM error: {response['error']}")
                return {"error": response["error"]}
            
            response_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received SQL from LLM ({len(response_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("sql_agent", response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
            # Extract SQL from the response
            sql = self._extract_sql(response_text)
            
            if not sql:
                logger.warning(f"{Fore.YELLOW}No SQL found in response, using full response{Fore.RESET}")
                sql = response_text
            
            # Validate the SQL
            is_valid = True
            error_message = ""
            
            if "validate_sql" in self.tools:
                logger.info(f"{Fore.YELLOW}Validating SQL{Fore.RESET}")
                validate_tool = self.tools["validate_sql"]
                validation_result = validate_tool.execute({"sql": sql})
                
                is_valid = validation_result.get("is_valid", False)
                error_message = validation_result.get("error_message", "")
                
                if is_valid:
                    logger.info(f"{Fore.GREEN}SQL validation successful{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_validation_success", {
                        "sql": sql
                    })
                else:
                    logger.warning(f"{Fore.YELLOW}SQL validation failed: {error_message}{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_validation_failed", {
                        "sql": sql,
                        "error_message": error_message
                    })
                    
                    # Try to refine the SQL
                    refinement_result = self.refine_sql({
                        "query": query,
                        "sql": sql,
                        "error_message": error_message,
                        "schema_context": schema_context,
                        "dba_analysis": dba_analysis
                    }, workflow_context)
                    
                    sql = refinement_result.get("sql", sql)
                    is_valid = refinement_result.get("is_valid", is_valid)
                    error_message = refinement_result.get("error_message", error_message)
            
            # Update workflow context if provided
            if workflow_context:
                workflow_context.set_sql(sql)
                workflow_context.add_metadata("sql_generation", {
                    "is_valid": is_valid,
                    "error_message": error_message if not is_valid else ""
                })
            
            return {
                "sql": sql,
                "is_valid": is_valid,
                "error_message": error_message if not is_valid else ""
            }
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error during SQL generation: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("sql_agent", f"Error during SQL generation: {str(e)}")
            return {"error": str(e)}
    
    def _extract_sql(self, text: str) -> str:
        """
        Extract SQL from text, removing markdown code blocks if present
        
        Args:
            text: The text containing SQL
            
        Returns:
            The extracted SQL
        """
        # Try to find SQL in code blocks
        sql_match = re.search(r'```(?:sql)?\s*([\s\S]*?)\s*```', text)
        
        if sql_match:
            sql = sql_match.group(1).strip()
            logger.info(f"{Fore.GREEN}Extracted SQL from code block{Fore.RESET}")
        else:
            # Just use the whole text
            sql = text.strip()
            logger.info(f"{Fore.YELLOW}No code block found, using entire text as SQL{Fore.RESET}")
        
        return sql
    
    def refine_sql(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Refine a SQL query based on error message or intent feedback
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - sql: The SQL to refine
                - error_message: Error message from validation
                - schema_context: The schema context
                - dba_analysis: The DBA analysis results 
                - intent_feedback: Optional feedback from intent verification
            workflow_context: Optional workflow context for logging and tracking
                
        Returns:
            Dictionary containing the refined SQL and validation status
        """
        query = inputs.get("query", "")
        sql = inputs.get("sql", "")
        error_message = inputs.get("error_message", "")
        schema_context = inputs.get("schema_context", "")
        dba_analysis = inputs.get("dba_analysis", {})
        intent_feedback = inputs.get("intent_feedback", {})
        
        logger.info(f"{Fore.CYAN}SQL Agent refining SQL for query: {query}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("sql_agent_refinement_start", {
            "query": query,
            "sql": sql[:200] + "..." if len(sql) > 200 else sql,
            "error_message": error_message,
            "has_intent_feedback": bool(intent_feedback)
        })
        
        # Get conversation context if available
        conversation_context = ""
        if workflow_context:
            conversation_context = self.get_conversation_context(workflow_context)
        
        # Determine if this is refinement for SQL syntax or for intent mismatch
        is_intent_refinement = "intent_feedback" in inputs and intent_feedback
        
        # Choose appropriate system prompt
        system_prompt = self.get_intent_refinement_prompt() if is_intent_refinement else self.get_error_refinement_prompt()
        
        # Prepare feedback information
        feedback_info = ""
        if is_intent_refinement:
            missing_aspects = intent_feedback.get("missing_aspects", [])
            suggestions = intent_feedback.get("suggestions", [])
            
            feedback_info = f"""
            Intent Analysis Feedback:
            - Missing aspects: {', '.join(missing_aspects)}
            - Suggestions: {', '.join(suggestions)}
            """
        
        # Format DBA analysis for the prompt
        dba_analysis_formatted = self._format_dba_analysis(dba_analysis)
        
        # Create the user prompt
        user_prompt = f"""
        Natural Language Query: {query}
        
        Current SQL:
        {sql}
        
        {'Error Message:' if not is_intent_refinement else 'Intent Mismatch:'}
        {error_message}
        
        {feedback_info if is_intent_refinement else ''}
        
        Schema Context:
        {schema_context}
        
        DBA Analysis:
        {dba_analysis_formatted}
        
        {f"Conversation Context:\n{conversation_context}" if conversation_context else ""}
        
        Please refine the SQL query to {'fully address the original query intent' if is_intent_refinement else 'fix the error'}.
        Return only the refined SQL without any explanations or markdown formatting.
        """
        
        # Get the refined SQL from the LLM
        logger.info(f"{Fore.YELLOW}Sending refinement request to LLM{Fore.RESET}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent_refine")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM during refinement: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("sql_agent", f"LLM refinement error: {response['error']}")
                return {
                    "sql": sql,
                    "is_valid": False,
                    "error_message": f"Refinement failed: {response['error']}"
                }
            
            response_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received refined SQL from LLM ({len(response_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("sql_agent_refine", response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
            # Extract SQL from the response
            refined_sql = self._extract_sql(response_text)
            
            if not refined_sql:
                logger.warning(f"{Fore.YELLOW}No SQL found in refinement response, using full response{Fore.RESET}")
                refined_sql = response_text
            
            # Validate the refined SQL
            is_valid = True
            new_error_message = ""
            
            if "validate_sql" in self.tools:
                logger.info(f"{Fore.YELLOW}Validating refined SQL{Fore.RESET}")
                validation_result = self.tools["validate_sql"].execute({"sql": refined_sql})
                
                is_valid = validation_result.get("is_valid", False)
                new_error_message = validation_result.get("error_message", "")
                
                if is_valid:
                    logger.info(f"{Fore.GREEN}Refined SQL validation successful{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_refinement_success", {
                        "sql": refined_sql[:200] + "..." if len(refined_sql) > 200 else refined_sql
                    })
                else:
                    logger.warning(f"{Fore.YELLOW}Refined SQL validation failed: {new_error_message}{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_refinement_failed", {
                        "error_message": new_error_message,
                        "sql": refined_sql[:200] + "..." if len(refined_sql) > 200 else refined_sql
                    })
            
            # Update workflow context if provided
            if workflow_context:
                workflow_context.set_sql(refined_sql)
                workflow_context.add_metadata("sql_refinement", {
                    "is_valid": is_valid,
                    "error_message": new_error_message if not is_valid else "",
                    "refinement_type": "intent" if is_intent_refinement else "error"
                })
            
            return {
                "sql": refined_sql,
                "is_valid": is_valid,
                "error_message": new_error_message if not is_valid else ""
            }
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error during SQL refinement: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("sql_agent", f"Refinement error: {str(e)}")
            return {
                "sql": sql,
                "is_valid": False,
                "error_message": f"Refinement failed: {str(e)}"
            }
    
    def _format_dba_analysis(self, dba_analysis: Dict[str, Any]) -> str:
        """Format DBA analysis for inclusion in prompts"""
        formatted = ""
        
        # Format tables
        if "tables" in dba_analysis and dba_analysis["tables"]:
            formatted += "Tables:\n"
            for table in dba_analysis["tables"]:
                if isinstance(table, dict):
                    formatted += f"- {table.get('name', 'Unknown')}"
                    if "reason" in table:
                        formatted += f": {table['reason']}"
                    formatted += "\n"
                else:
                    formatted += f"- {table}\n"
            formatted += "\n"
        
        # Format columns
        if "columns" in dba_analysis and dba_analysis["columns"]:
            formatted += "Columns:\n"
            for column in dba_analysis["columns"]:
                if isinstance(column, dict):
                    formatted += f"- {column.get('table', 'Unknown')}.{column.get('name', 'Unknown')}"
                    if "purpose" in column:
                        formatted += f": {column['purpose']}"
                    formatted += "\n"
                else:
                    formatted += f"- {column}\n"
            formatted += "\n"
        
        # Format joins
        if "joins" in dba_analysis and dba_analysis["joins"]:
            formatted += "Joins:\n"
            for join in dba_analysis["joins"]:
                if isinstance(join, dict):
                    formatted += f"- {join.get('left_table', 'Unknown')}.{join.get('left_column', 'Unknown')} {join.get('join_type', 'JOIN')} {join.get('right_table', 'Unknown')}.{join.get('right_column', 'Unknown')}"
                    if "reason" in join:
                        formatted += f": {join['reason']}"
                    formatted += "\n"
                else:
                    formatted += f"- {join}\n"
            formatted += "\n"
        
        return formatted
            
    def get_intent_refinement_prompt(self) -> str:
        """Get the system prompt for intent-based SQL refinement"""
        return """
        You are an expert SQL developer specializing in Trino SQL. Your task is to refine a SQL query
        to better match the original query intent based on feedback from intent analysis.
        
        The SQL query you're refining has correct syntax but doesn't fully address all aspects of
        the original natural language query. Your job is to modify the SQL to ensure it captures
        ALL requirements from the original question.
        
        Common issues to fix:
        1. Missing tables that contain relevant information
        2. Missing columns needed to provide complete answers
        3. Incomplete filter conditions (WHERE clauses)
        4. Incorrect or missing JOIN conditions
        5. Missing or incorrect aggregations
        6. Missing or incorrect grouping
        7. Missing or incorrect ordering
        
        When refining the SQL:
        - Carefully analyze what aspects of the query are missing
        - Ensure your solution addresses ALL the requirements
        - Maintain correct Trino SQL syntax
        - Keep the solution as efficient as possible
        - Ensure all column references are properly qualified
        
        Return ONLY the refined SQL without explanations or markdown formatting.
        """
    
    def get_error_refinement_prompt(self) -> str:
        """Get the system prompt for error-based SQL refinement"""
        return """
        You are an expert SQL developer specializing in Trino SQL. Your task is to fix a SQL query
        that has syntax or semantic errors based on the provided error message.
        
        The SQL query you're fixing has errors that prevent it from executing. Your job is to analyze
        the error message and make the necessary corrections while ensuring the query still addresses
        the original question.
        
        Common errors to fix:
        1. Missing or misspelled table/column names
        2. Incorrect table qualifications (catalog.schema.table)
        3. Type mismatches in comparisons or joins
        4. Ambiguous column references
        5. Missing GROUP BY columns for aggregations
        6. Invalid function calls or syntax
        
        When fixing the SQL:
        - Focus on addressing the specific error in the message
        - Maintain the original query intent
        - Ensure all table and column references are valid
        - Keep the solution as clean and efficient as possible
        
        Return ONLY the fixed SQL without explanations or markdown formatting.
        """

    def generate_sql(self, nlq: str, dba_analysis: Dict[str, Any]) -> str:
        """
        Generate SQL for a natural language query based on DBA analysis
        
        Args:
            nlq: The natural language query to convert to SQL
            dba_analysis: The analysis from the DBA agent
            
        Returns:
            The generated SQL query
        """
        logger.info(f"{Fore.BLUE}SQL Agent generating SQL for: {nlq}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}Using DBA analysis: {str(dba_analysis)[:100]}...{Fore.RESET}")
        
        # Log detailed reasoning
        logger.info(f"{Fore.BLUE}SQL Agent reasoning process:{Fore.RESET}")
        logger.info(f"{Fore.BLUE}1. Reviewing tables identified by DBA: {dba_analysis.get('tables', [])}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}2. Planning SQL structure based on required columns: {dba_analysis.get('columns', [])}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}3. Implementing joins for identified relationships{Fore.RESET}")
        logger.info(f"{Fore.BLUE}4. Adding filters, aggregations and ordering{Fore.RESET}")
        
        # Log the agent action
        conversation_logger.log_trino_ai_to_ollama(self.name, {
            "action": "generate_sql",
            "nlq": nlq,
            "dba_analysis": {k: v for k, v in dba_analysis.items() if k != "schema_context"}
        })
        
        # Build the prompt for the LLM
        # ... rest of the existing code ... 