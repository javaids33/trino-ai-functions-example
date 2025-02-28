import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class Tool(ABC):
    """Base class for all tools that agents can use"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"tool.{name}")
        
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given inputs"""
        pass
        
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for this tool"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }
        
    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool"""
        pass 