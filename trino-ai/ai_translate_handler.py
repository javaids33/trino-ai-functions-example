import logging
import json
import time
import os
from typing import Dict, Any, Optional

from agent_orchestrator import AgentOrchestrator
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from trino_executor import TrinoExecutor
from context_manager import WorkflowContext

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
            host=os.getenv("OLLAMA_HOST", "ollama"),
            port=int(os.getenv("OLLAMA_PORT", "11434"))
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
        
        logger.info("AI translate handler initialized")
    
    def handle_translate_request(self, request_data: Dict[str, Any], execute: bool = True) -> Dict[str, Any]:
        """
        Handle an AI translate request
        
        Args:
            request_data: The request data
            execute: Whether to execute the translated SQL query
            
        Returns:
            The response data
        """
        start_time = time.time()
        
        # Start a new conversation
        conversation_logger.start_conversation()
        
        # Extract the query from the request
        query = request_data.get("query", "")
        model = request_data.get("model", None)
        
        if not query:
            logger.error("No query provided in request")
            return {
                "error": "No query provided in request",
                "status": "error"
            }
            
        logger.info(f"Processing AI translate request: {query}")
        
        # Process the query using the agent orchestrator
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
            response["sql"] = result.get("sql_query", "")
            response["explanation"] = result.get("explanation", "")
            response["is_data_query"] = True
            
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
            response["response"] = result.get("response", "")
            response["is_data_query"] = False
        
        # Add metadata and workflow context
        response["metadata"] = result.get("metadata", {})
        response["workflow_context"] = workflow_context
        
        # Add agent reasoning if available
        if "agent_reasoning" in result:
            response["agent_reasoning"] = result["agent_reasoning"]
        
        # Calculate total processing time
        total_time = time.time() - start_time
        response["processing_time"] = f"{total_time:.2f}s"
        
        # Update the workflow context in the conversation logger
        conversation_logger.update_workflow_context(workflow_context)
        
        return response
    
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