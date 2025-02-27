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
from conversation_logger import conversation_logger
from trino_executor import TrinoExecutor

logger = logging.getLogger(__name__)

class SQLAgent(Agent):
    """
    Agent specialized in generating SQL queries from natural language.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize the SQL Agent
        
        Args:
            ollama_client: The Ollama client to use for completions
        """
        super().__init__(
            name="SQL Agent",
            description="A specialized agent for generating SQL queries from natural language"
        )
        self.ollama_client = ollama_client
        self.trino_executor = TrinoExecutor()
        self.logger.info("SQL Agent initialized")
    
    def execute(self, inputs: dict) -> dict:
        """
        Execute the SQL Agent to generate a SQL query from natural language
        
        Args:
            inputs: A dictionary containing the query and schema context
            
        Returns:
            A dictionary containing the generated SQL query
        """
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        
        if not query:
            return {"error": "No query provided"}
            
        if not schema_context:
            self.logger.warning("No schema context provided, SQL generation may be less accurate")
        
        sql_query = self.generate_sql(query, schema_context)
        
        # Extract explanation if provided
        explanation = ""
        if "EXPLANATION:" in sql_query:
            parts = sql_query.split("EXPLANATION:", 1)
            sql_query = parts[0].strip()
            if len(parts) > 1:
                explanation = parts[1].strip()
        
        # Clean up the SQL query
        sql_query = self._clean_sql(sql_query)
        
        return {
            "query": query,
            "sql": sql_query,
            "explanation": explanation
        }
    
    def generate_sql(self, nlq: str, dba_analysis: Dict[str, Any]) -> str:
        """
        Generate SQL for a natural language query based on DBA analysis
        
        Args:
            nlq: The natural language query to convert to SQL
            dba_analysis: The analysis from the DBA agent
            
        Returns:
            The generated SQL query
        """
        logger.info(f"SQL Agent generating SQL for: {nlq}")
        
        # Log detailed reasoning
        reasoning_steps = [
            {"description": f"Reviewing tables identified by DBA: {dba_analysis.get('tables', [])}"},
            {"description": f"Planning SQL structure based on required columns: {dba_analysis.get('columns', [])}"},
            {"description": f"Implementing joins for identified relationships: {dba_analysis.get('joins', [])}"},
            {"description": f"Adding filters: {dba_analysis.get('filters', [])}"},
            {"description": f"Adding aggregations: {dba_analysis.get('aggregations', [])}"}
        ]
        
        # Log the reasoning steps
        conversation_logger.log_agent_reasoning(self.name, reasoning_steps)
        
        # Build the prompt with explicit schema information
        schema_info = ""
        for table in dba_analysis.get('tables', []):
            schema_info += f"Table: {table}\n"
            # Get columns for this table if available
            table_columns = [col for col in dba_analysis.get('columns', []) if col.get('table') == table]
            if table_columns:
                schema_info += "Columns: " + ", ".join([col.get('name') for col in table_columns]) + "\n"
        
        # Create a more structured prompt with examples matching the schema
        prompt = f"""
        Generate a SQL query for the following natural language request:
        "{nlq}"
        
        Available schema information:
        {schema_info}
        
        Required joins:
        {', '.join([f"{join.get('left_table')}.{join.get('left_column')} = {join.get('right_table')}.{join.get('right_column')}" for join in dba_analysis.get('joins', [])])}
        
        Filters to apply:
        {', '.join([f"{filter.get('table')}.{filter.get('column')} {filter.get('operator')} {filter.get('value')}" for filter in dba_analysis.get('filters', [])])}
        
        Aggregations needed:
        {', '.join([f"{agg.get('type')}({agg.get('table')}.{agg.get('column')})" for agg in dba_analysis.get('aggregations', [])])}
        
        Generate ONLY the SQL query without any additional text or explanations.
        """
        
        # Generate the SQL using the LLM
        response = self.ollama_client.generate(prompt)
        
        # Extract the SQL from the response
        sql = self._extract_sql(response)
        
        # Clean up the SQL
        sql = self._clean_sql(sql)
        
        logger.info(f"Generated SQL: {sql}")
        
        return sql
    
    def get_system_prompt(self, schema_context: str) -> str:
        """
        Get the system prompt for the agent
        
        Args:
            schema_context: The database schema context
            
        Returns:
            The system prompt as a string
        """
        return f"""
        You are {self.name}, {self.description}.
        
        You are part of the Trino AI multi-agent system, specializing in translating natural language queries into SQL.
        
        Your task is to generate a SQL query that answers the user's question based on the database schema provided below.
        
        Database Schema:
        {schema_context}
        
        Guidelines:
        1. Generate ONLY the SQL query without any additional text, explanations, or markdown formatting.
        2. Use standard SQL syntax compatible with Trino.
        3. Include appropriate JOINs based on the schema relationships.
        4. Use column aliases for clarity when needed.
        5. Limit result sets to a reasonable number (e.g., TOP 100) unless specified otherwise.
        6. If you need to explain your reasoning, add it AFTER the SQL query with the prefix "EXPLANATION:".
        
        Example:
        User: "Show me the top 5 customers by total sales"
        
        Your response:
        SELECT c.customer_name, SUM(o.total_amount) as total_sales
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_name
        ORDER BY total_sales DESC
        LIMIT 5
        EXPLANATION: This query joins the customers and orders tables, calculates the sum of order amounts for each customer, and returns the top 5 customers by total sales.
        """
    
    def _clean_sql(self, sql: str) -> str:
        """
        Clean up the SQL query by removing markdown formatting and other artifacts
        
        Args:
            sql: The SQL query to clean
            
        Returns:
            The cleaned SQL query
        """
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove "SQL:" prefix if present
        sql = re.sub(r'^SQL:\s*', '', sql)
        
        # Remove any "Query:" prefix if present
        sql = re.sub(r'^Query:\s*', '', sql)
        
        return sql.strip()

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SQL from natural language query and DBA analysis
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - schema_context: The schema context
                - dba_analysis: The DBA analysis results
                
        Returns:
            Dictionary containing:
                - sql: The generated SQL query
        """
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        dba_analysis = inputs.get("dba_analysis", {})
        
        logger.info(f"SQL Agent generating SQL for query: {query}")
        conversation_logger.log_trino_ai_processing("sql_agent_generation_start", {
            "query": query,
            "schema_context_length": len(schema_context),
            "dba_analysis_tables": dba_analysis.get("tables", []),
            "dba_analysis_joins": dba_analysis.get("joins", [])
        })
        
        # Prepare the prompt for the LLM
        system_prompt = self.get_system_prompt(schema_context)
        
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
        logger.info(f"Sending query to LLM for SQL generation")
        start_time = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent")
            generation_time = time.time() - start_time
            logger.info(f"SQL generation completed in {generation_time:.2f}s")
            
            if "error" in response:
                logger.error(f"Error from LLM: {response['error']}")
                conversation_logger.log_error("sql_agent", f"LLM error: {response['error']}")
                return {"error": response["error"]}
            
            sql_text = response.get("message", {}).get("content", "")
            logger.info(f"Received SQL from LLM ({len(sql_text)} chars)")
            conversation_logger.log_ollama_to_trino_ai("sql_agent", sql_text[:500] + "..." if len(sql_text) > 500 else sql_text)
            
            # Extract SQL from the response (remove markdown code blocks if present)
            sql = self._extract_sql(sql_text)
            logger.info(f"Extracted SQL: {sql}")
            
            # Validate the SQL if the validate_sql tool is available
            if "validate_sql" in self.tools:
                logger.info(f"Validating generated SQL")
                validation_start = time.time()
                validation_result = self.tools["validate_sql"].execute({"sql": sql})
                validation_time = time.time() - validation_start
                logger.info(f"SQL validation completed in {validation_time:.2f}s")
                
                if not validation_result.get("is_valid", False):
                    error_message = validation_result.get("error_message", "Unknown validation error")
                    logger.error(f"SQL validation failed: {error_message}")
                    conversation_logger.log_error("sql_agent", f"SQL validation error: {error_message}")
                    
                    # Try to refine the SQL
                    logger.info(f"Attempting to refine invalid SQL")
                    refined_sql = self._refine_sql(query, schema_context, dba_analysis, sql, error_message)
                    
                    if refined_sql:
                        logger.info(f"Successfully refined SQL")
                        conversation_logger.log_trino_ai_processing("sql_agent_refinement_success", {
                            "original_sql": sql,
                            "refined_sql": refined_sql
                        })
                        return {"sql": refined_sql}
                    else:
                        logger.error(f"Failed to refine SQL")
                        conversation_logger.log_error("sql_agent", "Failed to refine SQL")
                        return {
                            "error": f"Failed to generate valid SQL: {error_message}",
                            "invalid_sql": sql
                        }
                else:
                    logger.info(f"SQL validation successful")
                    conversation_logger.log_trino_ai_processing("sql_agent_validation_success", {
                        "sql": sql
                    })
            
            return {"sql": sql}
            
        except Exception as e:
            logger.error(f"Error during SQL generation: {str(e)}")
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
            logger.info(f"Extracted SQL from code block")
        else:
            # Just use the whole text
            sql = text.strip()
            logger.info(f"No code block found, using entire text as SQL")
        
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
        logger.info(f"Refining SQL based on error: {error_message}")
        conversation_logger.log_trino_ai_processing("sql_agent_refinement_start", {
            "invalid_sql": invalid_sql,
            "error_message": error_message
        })
        
        system_prompt = self.get_system_prompt(schema_context)
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
            logger.info(f"Sending refinement request to LLM")
            start_time = time.time()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent_refine")
            refinement_time = time.time() - start_time
            logger.info(f"SQL refinement completed in {refinement_time:.2f}s")
            
            if "error" in response:
                logger.error(f"Error from LLM during refinement: {response['error']}")
                conversation_logger.log_error("sql_agent", f"LLM refinement error: {response['error']}")
                return None
            
            refined_text = response.get("message", {}).get("content", "")
            logger.info(f"Received refined SQL from LLM ({len(refined_text)} chars)")
            
            # Extract SQL from the response
            refined_sql = self._extract_sql(refined_text)
            
            # Validate the refined SQL
            if "validate_sql" in self.tools:
                logger.info(f"Validating refined SQL")
                validation_result = self.tools["validate_sql"].execute({"sql": refined_sql})
                
                if not validation_result.get("is_valid", False):
                    new_error = validation_result.get("error_message", "Unknown validation error")
                    logger.error(f"Refined SQL still invalid: {new_error}")
                    conversation_logger.log_error("sql_agent", f"Refined SQL validation error: {new_error}")
                    return None
                else:
                    logger.info(f"Refined SQL validation successful")
            
            return refined_sql
            
        except Exception as e:
            logger.error(f"Error during SQL refinement: {str(e)}")
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
            logger.error(f"No SQL provided to refine")
            conversation_logger.log_error("sql_agent", "No SQL provided to refine")
            return {"error": "No SQL provided to refine"}
            
        logger.info(f"SQL Agent refining SQL with error: {error_message}")
        conversation_logger.log_trino_ai_processing("sql_refinement_start", {
            "error_message": error_message,
            "sql": sql[:200] + "..."
        })
        
        # Create prompt for LLM
        system_prompt = self.get_system_prompt(schema_context)
        
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
            logger.info(f"Sending query to LLM for SQL refinement")
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
                    logger.error(f"No SQL found in refinement response")
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
                logger.info(f"Validating refined SQL")
                conversation_logger.log_trino_ai_processing("sql_validation_refined_start", {"sql": refined_sql[:200] + "..."})
                
                validation_result = self.tools["validate_sql"].execute({"sql": refined_sql})
                is_valid = validation_result.get("is_valid", False)
                new_error_message = validation_result.get("error_message", "")
                
                if is_valid:
                    logger.info(f"Refined SQL validation successful")
                    conversation_logger.log_trino_ai_processing("sql_validation_refined_success", {"sql": refined_sql[:200] + "..."})
                else:
                    logger.warning(f"Refined SQL validation failed: {new_error_message}")
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
            logger.error(f"Error during SQL refinement: {e}")
            conversation_logger.log_error("sql_agent", f"Error during SQL refinement: {e}")
            return {
                "sql": sql,
                "is_valid": False,
                "error_message": f"Error during SQL refinement: {e}"
            }
    
    def get_system_prompt(self, schema_context: str) -> str:
        """
        Get the system prompt for the agent
        
        Args:
            schema_context: The database schema context
            
        Returns:
            The system prompt as a string
        """
        return f"""
        You are {self.name}, {self.description}.
        
        You are part of the Trino AI multi-agent system, specializing in translating natural language queries into SQL.
        
        Your task is to generate a SQL query that answers the user's question based on the database schema provided below.
        
        Database Schema:
        {schema_context}
        
        Guidelines:
        1. Generate ONLY the SQL query without any additional text, explanations, or markdown formatting.
        2. Use standard SQL syntax compatible with Trino.
        3. Include appropriate JOINs based on the schema relationships.
        4. Use column aliases for clarity when needed.
        5. Limit result sets to a reasonable number (e.g., TOP 100) unless specified otherwise.
        6. If you need to explain your reasoning, add it AFTER the SQL query with the prefix "EXPLANATION:".
        
        Example:
        User: "Show me the top 5 customers by total sales"
        
        Your response:
        SELECT c.customer_name, SUM(o.total_amount) as total_sales
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_name
        ORDER BY total_sales DESC
        LIMIT 5
        EXPLANATION: This query joins the customers and orders tables, calculates the sum of order amounts for each customer, and returns the top 5 customers by total sales.
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
        logger.info(f"SQL Agent generating SQL for: {nlq}")
        logger.info(f"Using DBA analysis: {str(dba_analysis)[:100]}...")
        
        # Log detailed reasoning
        logger.info(f"SQL Agent reasoning process:")
        logger.info(f"1. Reviewing tables identified by DBA: {dba_analysis.get('tables', [])}")
        logger.info(f"2. Planning SQL structure based on required columns: {dba_analysis.get('columns', [])}")
        logger.info(f"3. Implementing joins for identified relationships")
        logger.info(f"4. Adding filters, aggregations and ordering")
        
        # Log the agent action
        conversation_logger.log_trino_ai_to_ollama(self.name, {
            "action": "generate_sql",
            "nlq": nlq,
            "dba_analysis": {k: v for k, v in dba_analysis.items() if k != "schema_context"}
        })
        
        # Build the prompt for the LLM
        # ... rest of the existing code ... 