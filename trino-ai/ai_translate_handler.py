import logging
import json
import time
from typing import Dict, Any, Optional

from agent_orchestrator import AgentOrchestrator
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from trino_executor import TrinoExecutor

logger = logging.getLogger(__name__)

class AITranslateHandler:
    """
    Handler for intercepting and processing ai_translate function calls from Trino
    """
    
    def __init__(self, ollama_client: OllamaClient = None):
        """
        Initialize the AI Translate Handler
        
        Args:
            ollama_client: The Ollama client to use for LLM interactions
        """
        self.ollama_client = ollama_client
        self.orchestrator = None  # Lazy initialization
        self.trino_executor = TrinoExecutor()
        logger.info("AI Translate Handler initialized")
    
    def handle_translate_request(self, query: str, target_format: str = "sql", execute: bool = True, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle an AI translate request
        
        Args:
            query: The natural language query to translate
            target_format: The target format (e.g. "sql")
            execute: Whether to execute the translated SQL query
            model: Optional model to use for translation
            
        Returns:
            A dictionary containing the translated query and metadata about the process
        """
        start_time = time.time()
        
        # Only support SQL translation for now
        if target_format.lower() != "sql":
            logger.warning(f"Unsupported target format: {target_format}. Only 'sql' is supported.")
            return {"error": f"Unsupported target format: {target_format}. Only 'sql' is supported."}
        
        logger.info(f"Handling AI translate request: {query} -> {target_format}")
        conversation_logger.log_trino_ai_processing("ai_translate_request", {
            "query": query,
            "target_format": target_format,
            "model": model,
            "execute": execute
        })
        
        # Initialize orchestrator if not already done
        if self.orchestrator is None:
            logger.info("Initializing agent orchestrator for AI Translate")
            self.orchestrator = AgentOrchestrator(ollama_client=self.ollama_client)
        
        # Process the request through our multi-agent system
        result = self.orchestrator.process_natural_language_query(query, model=model)
        processing_time = time.time() - start_time
        
        logger.info(f"AI translate request processed in {processing_time:.2f}s")
        
        # Check for errors
        if "error" in result:
            error_message = result["error"]
            error_stage = result.get("stage", "unknown")
            logger.error(f"Error in {error_stage} stage: {error_message}")
            return {
                "error": f"Error processing your query: {error_message}. The error occurred during the {error_stage} stage of processing."
            }
        
        # Format the response to match ai_translate's expected output
        if "sql" in result or result.get("sql_query"):
            # Data query result
            sql = result.get("sql", result.get("sql_query", ""))
            response = {
                "query": query,
                "sql": sql,
                "processing_time_seconds": processing_time,
                "agent_workflow": self._extract_agent_workflow(),
                "explanation": result.get("explanation", ""),
                "is_data_query": True
            }
            
            # Execute the SQL if requested and if a valid SQL was generated
            if execute and sql and result.get("is_valid", True):
                logger.info(f"Executing translated SQL: {sql}")
                
                # Execute the query
                execution_result = self.trino_executor.execute_query(sql)
                
                # Add the execution results to the response
                response["execution_result"] = execution_result
                
                if execution_result["success"]:
                    logger.info(f"SQL execution successful: {execution_result['row_count']} rows returned")
                else:
                    logger.warning(f"SQL execution failed: {execution_result.get('error', 'Unknown error')}")
                    
                # Log the execution results
                conversation_logger.log_trino_ai_processing("sql_execution", {
                    "success": execution_result["success"],
                    "row_count": execution_result.get("row_count", 0),
                    "execution_time": execution_result.get("execution_time", 0),
                    "error": execution_result.get("error", "")
                })
        else:
            # Knowledge query result
            response = {
                "query": query,
                "sql": "", # Empty SQL for knowledge queries
                "processing_time_seconds": processing_time,
                "agent_workflow": self._extract_agent_workflow(),
                "explanation": result.get("response", ""),
                "is_data_query": False
            }
        
        conversation_logger.log_trino_ai_processing("ai_translate_response", {
            "sql_length": len(response["sql"]),
            "processing_time_seconds": processing_time,
            "is_data_query": response["is_data_query"],
            "has_execution_result": "execution_result" in response
        })
        
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