import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        self.name = name
        self.description = description
        logger.info(f"Initializing tool: {name}")
    
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
    
    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the schema for the tool's parameters
        
        Returns:
            A dictionary describing the parameters schema
        """
        pass
    
    def get_name(self) -> str:
        """Get the name of the tool"""
        return self.name
    
    def get_description(self) -> str:
        """Get the description of the tool"""
        return self.description

    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for this tool"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        } 