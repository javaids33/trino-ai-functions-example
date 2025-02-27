import logging
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from conversation_logger import conversation_logger

logger = logging.getLogger(__name__)

class Tool(ABC):
    """
    Base class for all tools.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the tool with the given inputs
        
        Args:
            inputs: The inputs to the tool
            
        Returns:
            The outputs from the tool
        """
        start_time = time.time()
        
        self.logger.info(f"Executing tool: {self.name}")
        conversation_logger.log_trino_ai_processing("tool_execution_start", {
            "tool_name": self.name,
            "inputs": {k: v for k, v in inputs.items() if k != "password" and k != "secret" and k != "token"}
        })
        
        try:
            # Execute the tool
            result = self.execute(inputs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log the result
            self.logger.info(f"Tool {self.name} executed successfully in {execution_time:.2f}s")
            conversation_logger.log_trino_ai_processing("tool_execution_success", {
                "tool_name": self.name,
                "execution_time": execution_time
            })
            
            return result
            
        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log the error
            self.logger.error(f"Error executing tool {self.name}: {str(e)}")
            conversation_logger.log_error("tool_execution", f"Error executing tool {self.name}: {str(e)}")
            
            # Return an error result
            return {
                "error": f"Error executing tool {self.name}: {str(e)}"
            }
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with the given inputs
        
        Args:
            inputs: The inputs to the tool
            
        Returns:
            The outputs from the tool
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for this tool
        
        Returns:
            The schema for this tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the parameters schema for this tool
        
        Returns:
            The parameters schema for this tool
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        } 