from .base_tool import BaseCustomTool, ToolParameter, ToolRegistry
from .rag_tool import RAGSearchTool
from .web_search_tool import WebSearchTool
from .weather_tool import WeatherTool

__all__ = [
    'BaseCustomTool',
    'ToolParameter', 
    'ToolRegistry',
    'RAGSearchTool',
    'WebSearchTool',
    'WeatherTool'
]

def create_tool_registry() -> ToolRegistry:
    """Create and populate the tool registry with all available tools"""
    registry = ToolRegistry()
    
    # Register all tools
    registry.register(RAGSearchTool())
    registry.register(WebSearchTool())
    registry.register(WeatherTool())
    
    return registry