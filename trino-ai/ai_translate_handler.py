import logging
import json
import time
import os
import re
from typing import Dict, Any, Optional

from agent_orchestrator import AgentOrchestrator
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from trino_executor import TrinoExecutor
from context_manager import WorkflowContext
from agents.translation_agent import TranslationAgent

logger = logging.getLogger(__name__)

class AITranslateHandler:
    """
    Handler for intercepting and processing ai_translate function calls from Trino
    """
    
    def __init__(self):
        """
        Initialize the AI Translate Handler
        """
        # Initialize the Ollama client
        self.ollama_client = OllamaClient(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_HOST", "http://ollama:11434"),
        )
        
        # Initialize the agent orchestrator
        self.agent_orchestrator = AgentOrchestrator(ollama_client=self.ollama_client)
        
        # Initialize the Trino executor for running SQL queries
        self.trino_executor = TrinoExecutor(
            host=os.getenv("TRINO_HOST", "trino"),
            port=int(os.getenv("TRINO_PORT", "8080")),
            user=os.getenv("TRINO_USER", "admin"),
            catalog=os.getenv("TRINO_CATALOG", "iceberg"),
            schema=os.getenv("TRINO_SCHEMA", "iceberg")
        )
        
        # Initialize the translation agent
        self.translation_agent = TranslationAgent(ollama_client=self.ollama_client)
        
        logger.info("AI translate handler initialized")
    
    def translate_to_sql(self, query: str) -> Dict[str, Any]:
        """
        Translate a natural language query to SQL using the Translation Agent
        
        Args:
            query (str): The natural language query to translate
            
        Returns:
            Dict[str, Any]: Dictionary containing the SQL translation
        """
        logger.info(f"Translating query to SQL: {query}")
        
        # Log the query for conversation tracking
        conversation_logger.log_trino_to_trino_ai("query", {"query": query})
        
        # Initialize a workflow context for this translation
        workflow_context = WorkflowContext()
        workflow_context.set_query(query)
        workflow_context.set_data_query_status(True)
        
        # Get the translation
        result = self.translation_agent.execute({"query": query}, workflow_context)
        
        # Log the translation
        conversation_logger.log_trino_ai_to_trino("translation", {
            "action": "translate_to_sql",
            "query": query,
            "result": result
        })
        
        if "error" in result:
            return {
                "error": result["error"],
                "status": "error"
            }
        
        return {
            "sql": result.get("sql", ""),
            "status": "success"
        }
    
    def handle_translate_request(self, request_data: Dict[str, Any], execute: bool = True) -> Dict[str, Any]:
        """
        Handle an AI translate request
        
        Args:
            request_data: The request data
            execute: Whether to execute the translated SQL query
            
        Returns:
            The response data
        """
        # Start a new conversation for each request
        conversation_logger.start_conversation()
        
        start_time = time.time()
        
        # Start a new conversation
        conversation_logger.start_conversation()
        
        # Extract the query from the request
        query = request_data.get("query", "")
        target_format = request_data.get("target_format", "sql").lower()
        model = request_data.get("model", None)
        
        if not query:
            logger.error("No query provided in request")
            return {
                "error": "No query provided in request",
                "status": "error"
            }
            
        logger.info(f"Processing AI translate request: {query}")
        
        # If the target format is SQL, use our specialized translation agent
        if target_format == "sql":
            translation_result = self.translate_to_sql(query)
            
            if "error" in translation_result:
                return translation_result
                
            sql_query = translation_result.get("sql", "")
            
            # Execute the SQL query if requested
            execution_results = None
            if execute:
                try:
                    logger.info(f"Executing SQL query: {sql_query}")
                    execution_results = self.trino_executor.execute_query(sql_query)
                except Exception as e:
                    logger.error(f"Error executing SQL query: {str(e)}")
                    execution_results = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Generate a simple reasoning explanation
            reasoning = self._generate_simple_explanation(query, sql_query, execution_results)
            
            # Extract basic metadata
            metadata = self._extract_basic_metadata(sql_query)
            
            # Calculate the execution time
            execution_time = time.time() - start_time
            
            return {
                "query": query,
                "sql": sql_query,
                "execution": execution_results,
                "execution_time": f"{execution_time:.2f}s",
                "reasoning": reasoning,
                "metadata_used": metadata,
                "status": "success"
            }
        
        # For other target formats, use the agent orchestrator
        result = self.agent_orchestrator.process_natural_language_query(query, model)
        
        # Check if there was an error
        if "error" in result:
            logger.error(f"Error processing query: {result['error']}")
            return {
                "error": result["error"],
                "status": "error",
                "stage": result.get("stage", "unknown"),
                "workflow_context": result.get("workflow_context", {})
            }
            
        # Extract the workflow context
        workflow_context = result.get("workflow_context", {})
        
        # If this is a data query and we have a valid SQL query, execute it if requested
        execution_results = None
        if result.get("is_data_query", False) and result.get("is_valid", False) and execute:
            sql_query = result.get("sql_query", "")
            
            if sql_query:
                logger.info(f"Executing SQL query: {sql_query}")
                
                # Create a decision point for executing the SQL
                if isinstance(workflow_context, dict) and "decision_points" in workflow_context:
                    workflow_context["decision_points"].append({
                        "agent": "ai_translate_handler",
                        "decision": "execute_sql",
                        "explanation": f"Executing the generated SQL query"
                    })
                
                try:
                    # Execute the SQL query
                    execution_start = time.time()
                    execution_results = self.trino_executor.execute_query(sql_query)
                    execution_time = time.time() - execution_start
                    
                    # Add execution results to the workflow context
                    if isinstance(workflow_context, dict):
                        if "metadata" not in workflow_context:
                            workflow_context["metadata"] = {}
                        workflow_context["metadata"]["execution_results"] = {
                            "success": execution_results.get("success", False),
                            "row_count": len(execution_results.get("rows", [])),
                            "execution_time": f"{execution_time:.2f}s"
                        }
                        
                        if "decision_points" in workflow_context:
                            workflow_context["decision_points"].append({
                                "agent": "ai_translate_handler",
                                "decision": "sql_execution_complete",
                                "explanation": f"SQL execution completed in {execution_time:.2f}s with {len(execution_results.get('rows', []))} rows returned"
                            })
                    
                    logger.info(f"SQL execution completed in {execution_time:.2f}s with {len(execution_results.get('rows', []))} rows")
                except Exception as e:
                    error_msg = f"Error executing SQL query: {str(e)}"
                    logger.error(error_msg)
                    
                    # Add execution error to the workflow context
                    if isinstance(workflow_context, dict):
                        if "metadata" not in workflow_context:
                            workflow_context["metadata"] = {}
                        workflow_context["metadata"]["execution_error"] = {
                            "error": error_msg
                        }
                        
                        if "decision_points" in workflow_context:
                            workflow_context["decision_points"].append({
                                "agent": "ai_translate_handler",
                                "decision": "sql_execution_error",
                                "explanation": error_msg
                            })
                    
                    execution_results = {
                        "success": False,
                        "error": error_msg
                    }
        
        # Prepare the response
        response = {
            "query": query,
            "status": "success"
        }
        
        # Add the appropriate fields based on the query type
        if result.get("is_data_query", False):
            response.update({
                "sql": result.get("sql_query", ""),
                "explanation": result.get("explanation", ""),
                "is_valid": result.get("is_valid", False),
                "refinement_steps": result.get("refinement_steps", 0),
                "agent_reasoning": result.get("agent_reasoning", ""),
                "metadata_used": result.get("metadata", {})
            })
            
            # Add execution results if available
            if execution_results:
                response["execution"] = {
                    "success": execution_results.get("success", False),
                    "row_count": len(execution_results.get("rows", [])),
                    "execution_time": execution_results.get("execution_time", ""),
                    "columns": execution_results.get("columns", []),
                    "rows": execution_results.get("rows", [])
                }
                
                if not execution_results.get("success", False):
                    response["execution"]["error"] = execution_results.get("error", "")
        else:
            # For knowledge queries
            response.update({
                "response": result.get("response", ""),
                "agent_reasoning": result.get("agent_reasoning", ""),
                "metadata_used": result.get("metadata", {})
            })
        
        logger.info(f"AI translate request processed in {elapsed_time:.2f}s")
        
        return response
    
    def _generate_simple_explanation(self, nl_query: str, sql_query: str, execution_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a simple explanation of the SQL query
        
        Args:
            nl_query: The natural language query
            sql_query: The generated SQL query
            execution_results: The execution results, if available
            
        Returns:
            A simple explanation of the query
        """
        try:
            # Create a prompt that asks for an explanation
            prompt = f"""
            Natural language query: {nl_query}
            
            SQL query:
            ```
            {sql_query}
            ```
            
            Briefly explain what this SQL query does in simple terms that a non-technical person would understand.
            Keep your explanation to 2-3 sentences maximum.
            """
            
            # Generate the explanation
            response = self.ollama_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert at explaining SQL queries in simple terms."},
                    {"role": "user", "content": prompt}
                ],
                agent_name="SQL Explainer"
            )
            
            explanation = response.get("message", {}).get("content", "No explanation available.")
            
            # If we have results, add a simple summary
            if execution_results and "rows" in execution_results:
                num_rows = len(execution_results["rows"])
                if num_rows == 0:
                    explanation += " The query returned no results."
                else:
                    explanation += f" The query returned {num_rows} rows of data."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating simple explanation: {str(e)}")
            return f"This query translates '{nl_query}' into SQL. Unable to provide detailed explanation due to an error."
    
    def _generate_enhanced_explanation(self, nl_query: str, sql_query: str, execution_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate an enhanced explanation of the SQL query and results"""
        
        system_prompt = """
        You are an expert SQL educator. Your task is to explain SQL queries and their results in a way that's accessible to non-technical users.
        
        Provide an explanation with these components:
        1. Query Intent: A plain English description of what the query is trying to achieve
        2. Query Breakdown: An explanation of each part of the SQL query (SELECT, FROM, WHERE, etc.)
        3. Results Explanation: An interpretation of the results (if provided)
        4. Key Insights: Important observations about the data
        
        Use simple language and avoid technical jargon when possible.
        """
        
        # Prepare results sample for explanation
        results_sample = "No results available"
        if execution_results and "rows" in execution_results:
            rows = execution_results.get("rows", [])
            columns = execution_results.get("columns", [])
            
            if rows and columns:
                # Format a sample of up to 5 rows
                sample_rows = rows[:5]
                results_sample = "Column names: " + ", ".join(columns) + "\n\n"
                results_sample += "Sample results:\n"
                
                for row in sample_rows:
                    results_sample += str(row) + "\n"
                
                if len(rows) > 5:
                    results_sample += f"\n... and {len(rows) - 5} more rows"
        
        # Create the prompt
        user_prompt = f"""
        Natural language query: {nl_query}
        
        SQL query:
        ```sql
        {sql_query}
        ```
        
        Query results:
        {results_sample}
        
        Please provide:
        1. A clear explanation of what this SQL query does
        2. A breakdown of the SQL query parts
        3. An interpretation of the results in plain English
        4. Key insights from the data (if applicable)
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.ollama_client.chat_completion(messages, agent_name="SQL Explainer")
        
        if "error" in response:
            return {
                "simple_explanation": f"I couldn't generate a detailed explanation. {response['error']}",
                "detailed_explanation": None
            }
        
        explanation = response.get("message", {}).get("content", "")
        
        # Create a simpler version for basic needs
        simple_version = self._generate_simple_explanation(nl_query, sql_query, execution_results)
        
        return {
            "simple_explanation": simple_version,
            "detailed_explanation": explanation
        }
    
    def _extract_basic_metadata(self, sql_query: str) -> Dict[str, Any]:
        """
        Extract basic metadata from the SQL query
        
        Args:
            sql_query: The SQL query
            
        Returns:
            A dictionary containing basic metadata
        """
        try:
            # Extract table names from the SQL query
            table_pattern = r'FROM\s+([a-zA-Z0-9_."]+(?:\.[a-zA-Z0-9_."]+)?(?:\.[a-zA-Z0-9_."]+)?)'
            join_pattern = r'JOIN\s+([a-zA-Z0-9_."]+(?:\.[a-zA-Z0-9_."]+)?(?:\.[a-zA-Z0-9_."]+)?)'
            
            tables = []
            
            # Find tables in FROM clauses
            from_matches = re.findall(table_pattern, sql_query, re.IGNORECASE)
            if from_matches:
                tables.extend(from_matches)
                
            # Find tables in JOIN clauses
            join_matches = re.findall(join_pattern, sql_query, re.IGNORECASE)
            if join_matches:
                tables.extend(join_matches)
                
            # Clean up table names (remove quotes, etc.)
            tables = [table.strip('"').strip() for table in tables]
            
            # Extract column names from the SQL query
            column_pattern = r'SELECT\s+(.*?)\s+FROM'
            column_matches = re.findall(column_pattern, sql_query, re.IGNORECASE | re.DOTALL)
            
            columns = []
            if column_matches:
                column_list = column_matches[0]
                columns = [col.strip() for col in column_list.split(',')]
            
            # Determine query type
            query_type = "SIMPLE_SELECT"
            if "GROUP BY" in sql_query.upper():
                query_type = "AGGREGATION"
            elif "JOIN" in sql_query.upper():
                query_type = "JOIN"
            elif "WHERE" in sql_query.upper():
                query_type = "FILTERED_SELECT"
            
            # Compile the metadata
            metadata = {
                "tables_used": tables,
                "columns_referenced": columns,
                "query_type": query_type
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting basic metadata: {str(e)}")
            return {"error": f"Unable to extract metadata: {str(e)}"}
    
    def _extract_agent_workflow(self) -> Dict[str, Any]:
        """
        Extract the agent workflow from the conversation logger
        
        Returns:
            The agent workflow
        """
        # Get the current conversation ID
        conversation_id = conversation_logger.get_current_conversation_id()
        
        if not conversation_id:
            return {
                "error": "No active conversation found",
                "status": "error"
            }
            
        # Get the workflow details
        workflow = conversation_logger.get_workflow(conversation_id)
        
        if not workflow:
            return {
                "error": "No workflow found for the current conversation",
                "status": "error"
            }
            
        return {
            "conversation_id": conversation_id,
            "workflow": workflow,
            "status": "success"
        } 