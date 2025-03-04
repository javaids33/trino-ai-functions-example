import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import sys
import os
import json

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger

# Import the WorkflowContext
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from context_manager import WorkflowContext

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
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Execute the agent's task
        
        Args:
            inputs: A dictionary of inputs for the agent
            workflow_context: Optional workflow context for tracking agent decisions
            
        Returns:
            A dictionary containing the results of the agent's execution
        """
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
        
        pass
    
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