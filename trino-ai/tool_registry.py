import logging
from typing import Dict, List, Any, Optional
from tools.base_tool import Tool

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Registry to manage the tools available to agents"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        logger.info("Initialized tool registry")
        
    def register_tool(self, tool: Tool) -> None:
        """Register a new tool"""
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' already registered, replacing")
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
        
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())
        
    def get_all_tools(self) -> Dict[str, Tool]:
        """Get all registered tools"""
        return self.tools 