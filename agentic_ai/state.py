from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from datetime import datetime

class AgentState(TypedDict):
    """State schema for the agentic AI system"""
    
    # Message history
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Current user query
    current_query: str
    
    # Tool execution history
    tool_calls: List[Dict[str, Any]]
    
    # Collected parameters for tools
    collected_params: Dict[str, Any]
    
    # Context and memory
    context: Dict[str, Any]
    
    # Current step in the workflow
    current_step: str
    
    # Available tools that can be called
    available_tools: List[str]
    
    # Tool routing decision
    selected_tool: Optional[str]
    
    # Intermediate results
    intermediate_results: Dict[str, Any]
    
    # Session metadata
    session_id: str
    timestamp: datetime

class StateManager:
    """Manages agent state and provides utility methods"""
    
    def __init__(self):
        self.default_tools = ["rag_search", "web_search", "weather_info"]
    
    def create_initial_state(self, query: str, session_id: str = None) -> AgentState:
        """Create initial state for a new conversation"""
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return AgentState(
            messages=[HumanMessage(content=query)],
            current_query=query,
            tool_calls=[],
            collected_params={},
            context={},
            current_step="analyze_query",
            available_tools=self.default_tools,
            selected_tool=None,
            intermediate_results={},
            session_id=session_id,
            timestamp=datetime.now()
        )
    
    def update_state(self, state: AgentState, **updates) -> AgentState:
        """Update state with new values"""
        new_state = state.copy()
        for key, value in updates.items():
            if key in new_state:
                new_state[key] = value
        return new_state
    
    def add_message(self, state: AgentState, message: BaseMessage) -> AgentState:
        """Add a message to the state"""
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [message]
        return new_state
    
    def add_tool_call(self, state: AgentState, tool_name: str, params: Dict[str, Any], result: Any) -> AgentState:
        """Record a tool call in the state"""
        new_state = state.copy()
        tool_call = {
            "tool": tool_name,
            "params": params,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        new_state["tool_calls"] = state["tool_calls"] + [tool_call]
        return new_state
    
    def get_conversation_history(self, state: AgentState) -> str:
        """Get formatted conversation history"""
        history = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                history.append(f"Assistant: {msg.content}")
        return "\n".join(history)
    
    def get_tool_history(self, state: AgentState) -> str:
        """Get formatted tool execution history"""
        if not state["tool_calls"]:
            return "No tools have been called yet."
        
        history = []
        for call in state["tool_calls"]:
            history.append(f"Tool: {call['tool']}, Params: {call['params']}")
        return "\n".join(history)
    
    def extract_context_for_tool(self, state: AgentState, tool_name: str) -> Dict[str, Any]:
        """Extract relevant context for a specific tool"""
        context = {
            "query": state["current_query"],
            "conversation_history": self.get_conversation_history(state),
            "previous_results": state["intermediate_results"]
        }
        
        # Add tool-specific context
        if tool_name == "weather_info":
            context["location_hints"] = self._extract_location_hints(state)
        elif tool_name == "rag_search":
            context["search_hints"] = self._extract_search_hints(state)
        elif tool_name == "web_search":
            context["search_terms"] = self._extract_search_terms(state)
        
        return context
    
    def _extract_location_hints(self, state: AgentState) -> List[str]:
        """Extract potential location references from conversation"""
        import re
        
        text = state["current_query"] + " " + self.get_conversation_history(state)
        
        # Simple location extraction (can be enhanced with NER)
        location_patterns = [
            r'\bin\s+([A-Z][a-zA-Z\s]+)',
            r'\bat\s+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+weather',
            r'weather\s+in\s+([A-Z][a-zA-Z\s]+)'
        ]
        
        locations = []
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            locations.extend(matches)
        
        return list(set(locations))
    
    def _extract_search_hints(self, state: AgentState) -> List[str]:
        """Extract search terms for RAG"""
        # Extract key terms from the query
        import re
        
        query = state["current_query"]
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why'}
        words = re.findall(r'\b\w+\b', query.lower())
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return meaningful_words
    
    def _extract_search_terms(self, state: AgentState) -> List[str]:
        """Extract search terms for web search"""
        return [state["current_query"]]