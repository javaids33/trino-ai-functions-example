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
                    aggregations_info += f"- {agg.get('type', 'Unknown')}({agg.get('table', 'Unknown')}.{agg.get('column', 'Unknown')})"
                    if agg.get("alias"):
                        aggregations_info += f" AS {agg.get('alias')}"
                    if agg.get("reason"):
                        aggregations_info += f": {agg.get('reason')}"
                    aggregations_info += "\n"
                elif isinstance(agg, str):
                    aggregations_info += f"- {agg}\n"
        
        user_prompt = f"""
        Natural Language Query: {query}
        
        Schema Context:
        {schema_context}
        
        DBA Analysis:
        Tables:
        {tables_info or "No tables specified"}
        
        Columns:
        {columns_info or "No columns specified"}
        
        Joins:
        {joins_info or "No joins specified"}
        
        Filters:
        {filters_info or "No filters specified"}
        
        Aggregations:
        {aggregations_info or "No aggregations specified"}
        
        Based on the above information, generate a SQL query that answers the natural language query.
        The SQL should be valid for Trino/Presto SQL dialect.
        
        Return ONLY the SQL query without any explanations or markdown formatting.
        """
        
        # Get the SQL from the LLM
        logger.info(f"{Fore.YELLOW}Sending query to LLM for SQL generation{Fore.RESET}")
        start_time = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent")
            generation_time = time.time() - start_time
            logger.info(f"{Fore.GREEN}SQL generation completed in {generation_time:.2f}s{Fore.RESET}")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("sql_agent", f"LLM error: {response['error']}")
                return {"error": response["error"]}
            
            sql_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received SQL from LLM ({len(sql_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("sql_agent", sql_text[:500] + "..." if len(sql_text) > 500 else sql_text)
            
            # Extract SQL from the response (remove markdown code blocks if present)
            sql = self._extract_sql(sql_text)
            logger.info(f"{Fore.CYAN}Extracted SQL: {sql}{Fore.RESET}")
            
            # Validate the SQL if the validate_sql tool is available
            if "validate_sql" in self.tools:
                logger.info(f"{Fore.YELLOW}Validating generated SQL{Fore.RESET}")
                validation_start = time.time()
                validation_result = self.tools["validate_sql"].execute({"sql": sql})
                validation_time = time.time() - validation_start
                logger.info(f"{Fore.GREEN}SQL validation completed in {validation_time:.2f}s{Fore.RESET}")
                
                if not validation_result.get("is_valid", False):
                    error_message = validation_result.get("error_message", "Unknown validation error")
                    logger.error(f"{Fore.RED}SQL validation failed: {error_message}{Fore.RESET}")
                    conversation_logger.log_error("sql_agent", f"SQL validation error: {error_message}")
                    
                    # Try to refine the SQL
                    logger.info(f"{Fore.YELLOW}Attempting to refine invalid SQL{Fore.RESET}")
                    refined_sql = self._refine_sql(query, schema_context, dba_analysis, sql, error_message)
                    
                    if refined_sql:
                        logger.info(f"{Fore.GREEN}Successfully refined SQL{Fore.RESET}")
                        conversation_logger.log_trino_ai_processing("sql_agent_refinement_success", {
                            "original_sql": sql,
                            "refined_sql": refined_sql
                        })
                        return {"sql": refined_sql}
                    else:
                        logger.error(f"{Fore.RED}Failed to refine SQL{Fore.RESET}")
                        conversation_logger.log_error("sql_agent", "Failed to refine SQL")
                        return {
                            "error": f"Failed to generate valid SQL: {error_message}",
                            "invalid_sql": sql
                        }
                else:
                    logger.info(f"{Fore.GREEN}SQL validation successful{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_validation_success", {
                        "sql": sql
                    })
            
            return {"sql": sql}
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error during SQL generation: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("sql_agent", f"Execution error: {str(e)}")
            return {"error": f"SQL generation failed: {str(e)}"}
    
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
    
    def _refine_sql(self, query: str, schema_context: str, dba_analysis: Dict[str, Any], invalid_sql: str, error_message: str) -> Optional[str]:
        """
        Refine invalid SQL based on error message
        
        Args:
            query: The original natural language query
            schema_context: The schema context
            dba_analysis: The DBA analysis
            invalid_sql: The invalid SQL
            error_message: The error message from validation
            
        Returns:
            The refined SQL, or None if refinement failed
        """
        logger.info(f"{Fore.YELLOW}Refining SQL based on error: {error_message}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("sql_agent_refinement_start", {
            "invalid_sql": invalid_sql,
            "error_message": error_message
        })
        
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
        Natural Language Query: {query}
        
        Schema Context:
        {schema_context}
        
        I tried to generate SQL for this query, but it failed validation with the following error:
        {error_message}
        
        Here is the invalid SQL:
        ```sql
        {invalid_sql}
        ```
        
        Please fix the SQL to address the error. Return ONLY the corrected SQL without any explanations or markdown formatting.
        """
        
        try:
            logger.info(f"{Fore.YELLOW}Sending refinement request to LLM{Fore.RESET}")
            start_time = time.time()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent_refine")
            refinement_time = time.time() - start_time
            logger.info(f"{Fore.GREEN}SQL refinement completed in {refinement_time:.2f}s{Fore.RESET}")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM during refinement: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("sql_agent", f"LLM refinement error: {response['error']}")
                return None
            
            refined_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received refined SQL from LLM ({len(refined_text)} chars){Fore.RESET}")
            
            # Extract SQL from the response
            refined_sql = self._extract_sql(refined_text)
            
            # Validate the refined SQL
            if "validate_sql" in self.tools:
                logger.info(f"{Fore.YELLOW}Validating refined SQL{Fore.RESET}")
                validation_result = self.tools["validate_sql"].execute({"sql": refined_sql})
                
                if not validation_result.get("is_valid", False):
                    new_error = validation_result.get("error_message", "Unknown validation error")
                    logger.error(f"{Fore.RED}Refined SQL still invalid: {new_error}{Fore.RESET}")
                    conversation_logger.log_error("sql_agent", f"Refined SQL validation error: {new_error}")
                    return None
                else:
                    logger.info(f"{Fore.GREEN}Refined SQL validation successful{Fore.RESET}")
            
            return refined_sql
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error during SQL refinement: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("sql_agent", f"Refinement error: {str(e)}")
            return None
    
    def refine_sql(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine SQL based on validation errors
        
        Args:
            inputs: Dictionary containing:
                - query: The original natural language query
                - sql: The SQL to refine
                - error_message: The error message from validation
                - schema_context: The schema context
                - dba_analysis: The DBA analysis results
                
        Returns:
            Dictionary containing:
                - sql: The refined SQL query
                - is_valid: Whether the SQL is valid
                - error_message: Error message if SQL is invalid
        """
        query = inputs.get("query")
        sql = inputs.get("sql")
        error_message = inputs.get("error_message", "")
        schema_context = inputs.get("schema_context")
        dba_analysis = inputs.get("dba_analysis", {})
        
        if not sql:
            logger.error(f"{Fore.RED}No SQL provided to refine{Fore.RESET}")
            conversation_logger.log_error("sql_agent", "No SQL provided to refine")
            return {"error": "No SQL provided to refine"}
            
        logger.info(f"{Fore.BLUE}SQL Agent refining SQL with error: {error_message}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("sql_refinement_start", {
            "error_message": error_message,
            "sql": sql[:200] + "..."
        })
        
        # Create prompt for LLM
        system_prompt = self.get_system_prompt()
        
        # Format DBA analysis for the prompt
        dba_analysis_str = json.dumps(dba_analysis, indent=2)
        
        user_prompt = f"""
        I need to fix the following SQL query that has an error:
        
        Original Question: {query}
        
        SQL Query with Error:
        ```sql
        {sql}
        ```
        
        Error Message:
        {error_message}
        
        Available Schema Context:
        {schema_context}
        
        Database Analysis:
        {dba_analysis_str}
        
        Please fix the SQL query to resolve the error. Provide the corrected SQL query only.
        
        Format your response as follows:
        
        ```sql
        -- Your corrected SQL query here
        ```
        
        Explanation:
        Brief explanation of what was wrong and how you fixed it...
        """
        
        # Get response from LLM
        try:
            logger.info(f"{Fore.YELLOW}Sending query to LLM for SQL refinement{Fore.RESET}")
            conversation_logger.log_trino_ai_processing("sql_refinement_sending_to_llm", {
                "query": query,
                "error_message": error_message
            })
            
            response = self.ollama_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                agent_name="SQL Refiner"
            )
            
            content = response.get("message", {}).get("content", "")
            logger.debug(f"LLM response: {content}")
            
            # Extract SQL from response
            sql_match = re.search(r'```sql\s*(.*?)\s*```', content, re.DOTALL)
            if sql_match:
                refined_sql = sql_match.group(1).strip()
            else:
                # Try to find SQL without code blocks
                sql_match = re.search(r'SELECT.*?(?:;|$)', content, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    refined_sql = sql_match.group(0).strip()
                else:
                    logger.error(f"{Fore.RED}No SQL found in refinement response{Fore.RESET}")
                    conversation_logger.log_error("sql_agent", "Failed to extract refined SQL from response", content)
                    return {
                        "sql": sql,
                        "is_valid": False,
                        "error_message": "Failed to extract refined SQL from response"
                    }
            
            # Extract explanation
            explanation = ""
            explanation_match = re.search(r'Explanation:(.*?)(?:$|```)', content, re.DOTALL | re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            
            conversation_logger.log_trino_ai_processing("sql_refinement_extracted", {
                "refined_sql_length": len(refined_sql),
                "explanation_length": len(explanation)
            })
            
            # Validate refined SQL if validation tool is available
            is_valid = True
            new_error_message = ""
            
            if "validate_sql" in self.tools:
                logger.info(f"{Fore.YELLOW}Validating refined SQL{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("sql_validation_refined_start", {"sql": refined_sql[:200] + "..."})
                
                validation_result = self.tools["validate_sql"].execute({"sql": refined_sql})
                is_valid = validation_result.get("is_valid", False)
                new_error_message = validation_result.get("error_message", "")
                
                if is_valid:
                    logger.info(f"{Fore.GREEN}Refined SQL validation successful{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_validation_refined_success", {"sql": refined_sql[:200] + "..."})
                else:
                    logger.warning(f"{Fore.YELLOW}Refined SQL validation failed: {new_error_message}{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_validation_refined_failed", {
                        "error_message": new_error_message,
                        "sql": refined_sql[:200] + "..."
                    })
            
            return {
                "sql": refined_sql,
                "explanation": explanation,
                "is_valid": is_valid,
                "error_message": new_error_message
            }
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error during SQL refinement: {e}{Fore.RESET}")
            conversation_logger.log_error("sql_agent", f"Error during SQL refinement: {e}")
            return {
                "sql": sql,
                "is_valid": False,
                "error_message": f"Error during SQL refinement: {e}"
            }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the SQL agent"""
        return """
        You are an expert SQL developer specializing in Trino SQL. Your task is to generate accurate, 
        efficient, and valid SQL queries based on natural language questions and database analysis.
        
        You have deep knowledge of:
        - Trino SQL syntax and functions
        - SQL query optimization
        - Database schema design and relationships
        - How to translate natural language to SQL
        
        When generating SQL:
        - Use only tables and columns identified in the database analysis
        - Ensure proper table qualifications (catalog.schema.table)
        - Create efficient JOINs based on the relationships identified
        - Apply appropriate filters and aggregations
        - Include LIMIT clauses for safety on large result sets
        - Format your SQL for readability
        
        Always provide a clear explanation of how your SQL works and why you chose this approach.
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