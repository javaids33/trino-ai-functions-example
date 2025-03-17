import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import sys
import os
import json
import time
import traceback

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from monitoring.monitoring_service import monitoring_service

# Import the WorkflowContext
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow_context import WorkflowContext

logger = logging.getLogger(__name__)

class Agent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, description: str, ollama_client: Optional[OllamaClient] = None, tools: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent
        
        Args:
            name: The name of the agent
            description: A description of what the agent does
            ollama_client: An optional OllamaClient instance for LLM interactions
            tools: An optional dictionary of tools available to the agent
        """
        self.name = name
        self.description = description
        self.ollama_client = ollama_client
        self.tools = tools or {}
        logger.info(f"{Fore.CYAN}Initialized agent: {name}{Fore.RESET}")
    
    def execute(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Execute the agent's task
        
        Args:
            inputs: A dictionary of inputs for the agent
            workflow_context: Optional workflow context for tracking agent decisions
            
        Returns:
            A dictionary containing the results of the agent's execution
        """
        start_time = time.time()
        query_id = workflow_context.get_metadata("query_id", "unknown") if workflow_context else "unknown"
        
        try:
            # Log the start of execution
            logger.info(f"{Fore.CYAN}Executing agent {self.name} with inputs: {json.dumps({k: str(v)[:200] + '...' if isinstance(v, str) and len(v) > 200 else v for k, v in inputs.items() if k != 'schema_context'}, default=str)}{Fore.RESET}")
            conversation_logger.log_trino_ai_processing(f"{self.__class__.__name__.lower()}_execute_start", {
                "agent": self.name,
                "input_keys": list(inputs.keys())
            })
        
            # Log agent activation in workflow context if provided
            if workflow_context:
                workflow_context.add_decision_point(
                    self.name,
                    "agent_activated",
                    f"Agent {self.name} activated to process inputs: {list(inputs.keys())}"
                )
            
                # Log to monitoring service
                monitoring_service.log_agent_activity(self.name, {
                    "action": "execution_started",
                    "details": {"inputs": self._sanitize_inputs(inputs)},
                    "query_id": query_id
                })
            
            # Execute the agent-specific logic
            result = self._execute_impl(inputs, workflow_context)
            
            # Log the completion
            execution_time = time.time() - start_time
            monitoring_service.log_agent_activity(self.name, {
                "action": "execution_completed",
                "details": {"execution_time": execution_time},
                "duration_ms": int(execution_time * 1000),
                "query_id": query_id
            })
            
            # Update performance metrics
            monitoring_service.update_performance_metric(
                f"agent_{self.name.lower().replace(' ', '_')}_execution_time",
                execution_time
            )
            
            return result
            
        except Exception as e:
            # Log the error
            error_message = str(e)
            logger.error(f"{Fore.RED}Error in {self.name}: {error_message}{Fore.RESET}")
            conversation_logger.log_error(self.name, error_message)
            
            # Log to monitoring service
            monitoring_service.log_error({
                "error_type": type(e).__name__,
                "message": error_message,
                "source": f"agent_{self.name.lower().replace(' ', '_')}",
                "query_id": query_id,
                "stack_trace": traceback.format_exc()
            })
            
            # Log the failed execution
            execution_time = time.time() - start_time
            monitoring_service.log_agent_activity(self.name, {
                "action": "execution_failed",
                "details": {"error": error_message},
                "duration_ms": int(execution_time * 1000),
                "query_id": query_id
            })
            
            if workflow_context:
                workflow_context.add_decision_point(
                    self.name,
                    "error_occurred",
                    error_message
                )
                
                # Add error to metadata
                workflow_context.add_metadata(f"{self.name.lower()}_error", {
                    "message": str(e),
                    "type": type(e).__name__
                })
            
            return {"error": f"Error in {self.name}: {error_message}", "status": "error"}
    
    def _execute_impl(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Implement agent-specific execution logic
        
        Args:
            inputs: A dictionary of inputs for the agent
            workflow_context: Optional workflow context for tracking
            
        Returns:
            A dictionary containing the results of the agent's execution
        """
        # This method should be overridden by subclasses
        raise NotImplementedError("Agent subclasses must implement _execute_impl")
    
    def _sanitize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize inputs for logging (remove sensitive data, truncate large values)
        
        Args:
            inputs: The inputs to sanitize
            
        Returns:
            Sanitized inputs
        """
        sanitized = {}
        for key, value in inputs.items():
            if key == "schema_context" and isinstance(value, str):
                sanitized[key] = f"{value[:100]}... ({len(value)} chars)"
            elif isinstance(value, str) and len(value) > 200:
                sanitized[key] = f"{value[:200]}... ({len(value)} chars)"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_inputs(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def handle_error(self, error, workflow_context=None):
        """
        Standardized error handling for all agents
        
        Args:
            error: The error that occurred
            workflow_context: Optional workflow context for tracking
            
        Returns:
            Dictionary with error information
        """
        error_message = f"Error in {self.name}: {str(error)}"
        logger.error(f"{Fore.RED}{error_message}{Fore.RESET}")
        conversation_logger.log_error(self.name, error_message)
        
        # Log to monitoring service
        query_id = workflow_context.get_metadata("query_id", "unknown") if workflow_context else "unknown"
        monitoring_service.log_error({
            "error_type": type(error).__name__,
            "message": str(error),
            "source": f"agent_{self.name.lower().replace(' ', '_')}",
            "query_id": query_id
        })
        
        if workflow_context:
            workflow_context.add_decision_point(
                self.name,
                "error_occurred",
                error_message
            )
            
            # Add error to metadata
            workflow_context.add_metadata(f"{self.name.lower()}_error", {
                "message": str(error),
                "type": type(error).__name__
            })
        
        return {"error": error_message, "status": "error"}
    
    def validate_inputs(self, inputs, required_keys):
        """
        Validate that required inputs are present
        
        Args:
            inputs: Dictionary of inputs to validate
            required_keys: List of required keys
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing = [key for key in required_keys if key not in inputs]
        if missing:
            error_message = f"Missing required inputs: {', '.join(missing)}"
            logger.error(f"{Fore.RED}{error_message}{Fore.RESET}")
            conversation_logger.log_error(self.name, f"Input validation error: {error_message}")
            return False, error_message
        return True, ""
    
    def get_conversation_context(self, workflow_context: WorkflowContext) -> str:
        """
        Get the conversation context formatted for the agent's prompt
        
        Args:
            workflow_context: The workflow context
            
        Returns:
            Formatted conversation context
        """
        # Get relevant conversation history for this agent
        history = workflow_context.get_conversation_for_agent(self.name)
        
        # Format the conversation history for inclusion in prompts
        formatted_history = ""
        for entry in history[-10:]:  # Limit to last 10 entries to avoid token limits
            sender = entry["sender"]
            message = entry["message"]
            
            # Format the message based on its type and content
            message_content = ""
            if isinstance(message, dict):
                if "content" in message:
                    message_content = message["content"]
                elif "query" in message:
                    message_content = message["query"]
                else:
                    message_content = str(message)
            else:
                message_content = str(message)
            
            formatted_history += f"{sender}: {message_content}\n\n"
        
        return formatted_history
    
    def log_reasoning(self, reasoning: str, workflow_context: Optional[WorkflowContext] = None):
        """
        Log agent reasoning process
        
        Args:
            reasoning: The reasoning to log
            workflow_context: Optional workflow context to update
        """
        logger.info(f"{Fore.CYAN}[{self.name}] Reasoning: {reasoning}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing(f"{self.name.lower()}_reasoning", {
            "agent": self.name,
            "reasoning": reasoning
        })
        
        # Log to monitoring service
        query_id = workflow_context.get_metadata("query_id", "unknown") if workflow_context else "unknown"
        monitoring_service.log_agent_activity(self.name, {
            "action": "reasoning",
            "details": {"reasoning": reasoning[:500] + "..." if len(reasoning) > 500 else reasoning},
            "query_id": query_id
        })
        
        # Update workflow context if provided
        if workflow_context:
            workflow_context.add_agent_reasoning(self.name, reasoning)
    
    def log_metadata_usage(self, metadata_keys: List[str], workflow_context: Optional[WorkflowContext] = None):
        """
        Log which metadata was used in decision making
        
        Args:
            metadata_keys: List of metadata keys that were used
            workflow_context: Optional workflow context to update
        """
        logger.info(f"{Fore.CYAN}[{self.name}] Used metadata: {metadata_keys}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing(f"{self.name.lower()}_metadata_usage", {
            "agent": self.name,
            "metadata_keys": metadata_keys
        })
        
        # Update workflow context if provided
        if workflow_context:
            workflow_context.mark_metadata_used(self.name, metadata_keys)
    
    def log_decision(self, decision: str, rationale: str, workflow_context: Optional[WorkflowContext] = None):
        """
        Log a key decision made by the agent
        
        Args:
            decision: The decision that was made
            rationale: The rationale for the decision
            workflow_context: Optional workflow context to update
        """
        logger.info(f"{Fore.YELLOW}[{self.name}] Decision: {decision} - {rationale}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing(f"{self.name.lower()}_decision", {
            "agent": self.name,
            "decision": decision,
            "rationale": rationale
        })
        
        # Log to monitoring service
        query_id = workflow_context.get_metadata("query_id", "unknown") if workflow_context else "unknown"
        monitoring_service.log_agent_activity(self.name, {
            "action": "decision",
            "details": {
                "decision": decision,
                "rationale": rationale
            },
            "query_id": query_id
        })
        
        # Update workflow context if provided
        if workflow_context:
            workflow_context.add_decision_point(self.name, decision, rationale)
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent
        
        Returns:
            The system prompt as a string
        """
        return f"""
        You are {self.name}, {self.description}.
        
        Respond with accurate, helpful information based on your expertise.
        """ 