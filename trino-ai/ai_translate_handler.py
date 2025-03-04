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
                "agent_reasoning": result.get("agent_reasoning", "")
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
                "agent_reasoning": result.get("agent_reasoning", "")
            })
        
        logger.info(f"AI translate request processed in {elapsed_time:.2f}s")
        
        return response
    
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