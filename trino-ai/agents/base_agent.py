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
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task
        
        Args:
            inputs: A dictionary of inputs for the agent
            
        Returns:
            A dictionary containing the results of the agent's execution
        """
        logger.info(f"{Fore.CYAN}Executing agent {self.name} with inputs: {json.dumps({k: str(v)[:200] + '...' if isinstance(v, str) and len(v) > 200 else v for k, v in inputs.items() if k != 'schema_context'}, default=str)}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing(f"{self.__class__.__name__.lower()}_execute_start", {
            "agent": self.name,
            "input_keys": list(inputs.keys())
        })
        pass
    
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