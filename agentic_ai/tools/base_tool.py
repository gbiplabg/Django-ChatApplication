from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class ToolParameter(BaseModel):
    """Represents a parameter for a tool"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class BaseCustomTool(BaseTool, ABC):
    """Base class for all custom tools in the agentic AI system"""
    
    # Tool metadata
    tool_name: str
    tool_description: str
    parameters: List[ToolParameter]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = self.tool_name
        self.description = self.tool_description
    
    @abstractmethod
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def _arun(self, **kwargs) -> Dict[str, Any]:
        """Async execution of the tool"""
        pass
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the parameter schema for this tool"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean parameters"""
        validated = {}
        
        for param in self.parameters:
            if param.required and param.name not in params:
                raise ValueError(f"Required parameter '{param.name}' is missing")
            
            if param.name in params:
                validated[param.name] = params[param.name]
            elif param.default is not None:
                validated[param.name] = param.default
        
        return validated
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get comprehensive tool information"""
        return {
            "name": self.tool_name,
            "description": self.tool_description,
            "parameters": [param.dict() for param in self.parameters],
            "schema": self.get_parameter_schema()
        }


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseCustomTool] = {}
    
    def register(self, tool: BaseCustomTool):
        """Register a new tool"""
        self.tools[tool.tool_name] = tool
    
    def get_tool(self, name: str) -> Optional[BaseCustomTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> Dict[str, BaseCustomTool]:
        """Get all registered tools"""
        return self.tools.copy()
    
    def get_tool_names(self) -> List[str]:
        """Get names of all registered tools"""
        return list(self.tools.keys())
    
    def get_tools_info(self) -> List[Dict[str, Any]]:
        """Get information about all tools"""
        return [tool.get_tool_info() for tool in self.tools.values()]