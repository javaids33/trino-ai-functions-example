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
                
                # Log the execution in the workflow context if available
                if isinstance(workflow_context, dict) and "decision_points" in workflow_context:
                    workflow_context["decision_points"].append({
                        "agent": "ai_translate_handler",
                        "decision": "execute_sql",
                        "explanation": f"Executing the generated SQL query"
                    })
                
                # Execute the query
                execution_start_time = time.time()
                execution_results = self.trino_executor.execute_query(sql_query)
                execution_time = time.time() - execution_start_time
                
                logger.info(f"SQL execution completed in {execution_time:.2f}s, success: {execution_results.get('success', False)}")
                
                # Log the execution results in the workflow context if available
                if isinstance(workflow_context, dict):
                    if "metadata" not in workflow_context:
                        workflow_context["metadata"] = {}
                    
                    workflow_context["metadata"]["sql_execution"] = {
                        "success": execution_results.get("success", False),
                        "execution_time": f"{execution_time:.2f}s",
                        "row_count": len(execution_results.get("rows", [])),
                        "error": execution_results.get("error", "")
                    }
                    
                    if "decision_points" in workflow_context:
                        workflow_context["decision_points"].append({
                            "agent": "ai_translate_handler",
                            "decision": "sql_execution_complete",
                            "explanation": f"SQL execution completed in {execution_time:.2f}s, success: {execution_results.get('success', False)}"
                        })
        
        # Prepare the response
        elapsed_time = time.time() - start_time
        
        response = {
            "query": query,
            "status": "success",
            "processing_time": f"{elapsed_time:.2f}s",
            "workflow_context": workflow_context
        }
        
        # Add data query specific fields
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
                    "rows": execution_results.get("rows", []),
                    "columns": execution_results.get("columns", []),
                    "row_count": len(execution_results.get("rows", [])),
                    "execution_time": execution_results.get("execution_time", ""),
                    "error": execution_results.get("error", "")
                }
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
            # Extract table names from the SQL query
            table_pattern = r'FROM\s+([a-zA-Z0-9_."]+(?:\.[a-zA-Z0-9_."]+)?(?:\.[a-zA-Z0-9_."]+)?)'
            tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
            tables = [table.strip('"').strip() for table in tables]
            
            # Extract column names from the SQL query
            column_pattern = r'SELECT\s+(.*?)\s+FROM'
            column_matches = re.findall(column_pattern, sql_query, re.IGNORECASE | re.DOTALL)
            
            columns = []
            if column_matches:
                column_list = column_matches[0]
                columns = [col.strip() for col in column_list.split(',')]
            
            # Determine query type
            query_type = "SELECT"
            if "GROUP BY" in sql_query.upper():
                query_type = "AGGREGATION"
            elif "JOIN" in sql_query.upper():
                query_type = "JOIN"
            elif "WHERE" in sql_query.upper():
                query_type = "FILTERED SELECT"
            
            # Build the explanation
            explanation = f"This query translates '{nl_query}' into SQL by "
            
            if query_type == "SELECT":
                explanation += f"selecting {len(columns)} columns from the {', '.join(tables)} table."
            elif query_type == "AGGREGATION":
                explanation += f"aggregating data from the {', '.join(tables)} table with a GROUP BY clause."
            elif query_type == "JOIN":
                explanation += f"joining data from multiple tables: {', '.join(tables)}."
            elif query_type == "FILTERED SELECT":
                explanation += f"filtering data from the {', '.join(tables)} table using a WHERE clause."
            
            # Add execution information if available
            if execution_results:
                if execution_results.get("success", False):
                    row_count = len(execution_results.get("rows", []))
                    explanation += f"\n\nThe query executed successfully and returned {row_count} rows."
                    
                    if row_count == 0:
                        explanation += " No data was found matching your criteria."
                else:
                    explanation += f"\n\nThe query failed to execute with error: {execution_results.get('error', 'Unknown error')}"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating simple explanation: {str(e)}")
            return f"This query translates '{nl_query}' into SQL. Unable to provide detailed explanation due to an error."
    
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
        Extract the agent workflow from the conversation log
        
        Returns:
            A dictionary containing the agent workflow information
        """
        # This would extract and format the workflow data from the conversation logger
        workflow_steps = []
        
        # Get the last 100 entries from the conversation log
        for entry in conversation_logger.conversation_log[-100:]:
            if entry["type"] in ["trino_ai_processing", "trino_ai_to_ollama", "ollama_to_trino_ai"]:
                # Format the entry for display
                step = {
                    "timestamp": entry["timestamp"],
                    "agent": entry.get("agent", "system"),
                    "action": entry.get("action", entry["type"]),
                    "details": entry.get("data", {})
                }
                
                # Clean up and limit the size of large data fields
                if "schema_context" in step["details"]:
                    step["details"]["schema_context"] = step["details"]["schema_context"][:200] + "..." \
                        if len(step["details"]["schema_context"]) > 200 else step["details"]["schema_context"]
                        
                if "response" in step["details"] and isinstance(step["details"]["response"], str):
                    step["details"]["response"] = step["details"]["response"][:200] + "..." \
                        if len(step["details"]["response"]) > 200 else step["details"]["response"]
                
                workflow_steps.append(step)
        
        return {
            "workflow_steps": workflow_steps[-50:] if workflow_steps else []  # Return the most recent 50 steps
        } 