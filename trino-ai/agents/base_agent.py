import logging
import time
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from conversation_logger import conversation_logger

logger = logging.getLogger(__name__)

class Agent(ABC):
    """
    Base class for all agents.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the agent
        
        Args:
            name: The name of the agent
            description: A description of what the agent does
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Agent {name} initialized")
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the agent with the given inputs
        
        Args:
            inputs: The inputs to the agent
            
        Returns:
            The outputs from the agent
        """
        start_time = time.time()
        
        self.logger.info(f"Executing agent: {self.name}")
        conversation_logger.log_trino_ai_processing("agent_execution_start", {
            "agent_name": self.name,
            "inputs": {k: v for k, v in inputs.items() if k != "password" and k != "secret" and k != "token"}
        })
        
        try:
            # Execute the agent
            result = self.execute(inputs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log the result
            self.logger.info(f"Agent {self.name} executed successfully in {execution_time:.2f}s")
            conversation_logger.log_trino_ai_processing("agent_execution_success", {
                "agent_name": self.name,
                "execution_time": execution_time
            })
            
            return result
            
        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log the error
            self.logger.error(f"Error executing agent {self.name}: {str(e)}")
            conversation_logger.log_error("agent_execution", f"Error executing agent {self.name}: {str(e)}")
            
            # Return an error result
            return {
                "error": f"Error executing agent {self.name}: {str(e)}"
            }
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent with the given inputs
        
        Args:
            inputs: The inputs to the agent
            
        Returns:
            The outputs from the agent
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for this agent
        
        Returns:
            The schema for this agent
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the parameters schema for this agent
        
        Returns:
            The parameters schema for this agent
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

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