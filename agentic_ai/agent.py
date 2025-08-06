from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from state import AgentState, StateManager
from routing import QueryRouter
from tools import create_tool_registry
from config import Config


class AgenticAI:
    """Main agentic AI system using LangGraph for state management and tool orchestration"""
    
    def __init__(self):
        # Initialize components
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.3
        )
        self.state_manager = StateManager()
        self.router = QueryRouter()
        self.tool_registry = create_tool_registry()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("route_to_tool", self._route_to_tool_node)
        workflow.add_node("execute_tool", self._execute_tool_node)
        workflow.add_node("synthesize_response", self._synthesize_response_node)
        workflow.add_node("check_completion", self._check_completion_node)
        
        # Define the flow
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_edge("analyze_query", "route_to_tool")
        workflow.add_edge("route_to_tool", "execute_tool")
        workflow.add_edge("execute_tool", "synthesize_response")
        workflow.add_edge("synthesize_response", "check_completion")
        
        # Add conditional edge from check_completion
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "route_to_tool",
                "end": END
            }
        )
        
        return workflow
    
    def _analyze_query_node(self, state: AgentState) -> AgentState:
        """Analyze the user query and update state"""
        print(f"🔍 Analyzing query: {state['current_query']}")
        
        # Update state with analysis step
        new_state = self.state_manager.update_state(
            state,
            current_step="analyze_query",
            context={
                **state.get("context", {}),
                "analysis_complete": True,
                "query_intent": "pending_routing"
            }
        )
        
        return new_state
    
    def _route_to_tool_node(self, state: AgentState) -> AgentState:
        """Route the query to the appropriate tool"""
        print("🧭 Routing query to appropriate tool...")
        
        # Use the router to determine the best tool
        tool_name, parameters, confidence = self.router.route_query(
            state["current_query"], 
            state
        )
        
        print(f"📍 Selected tool: {tool_name} (confidence: {confidence:.2f})")
        print(f"📋 Parameters: {parameters}")
        
        # Update state with routing decision
        new_state = self.state_manager.update_state(
            state,
            current_step="route_to_tool",
            selected_tool=tool_name,
            collected_params=parameters,
            context={
                **state.get("context", {}),
                "routing_confidence": confidence,
                "routing_complete": True
            }
        )
        
        return new_state
    
    def _execute_tool_node(self, state: AgentState) -> AgentState:
        """Execute the selected tool with collected parameters"""
        tool_name = state["selected_tool"]
        parameters = state["collected_params"]
        
        print(f"⚡ Executing {tool_name} with parameters: {parameters}")
        
        # Get the tool
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            error_result = {
                "success": False,
                "error": f"Tool {tool_name} not found"
            }
            return self._handle_tool_error(state, tool_name, parameters, error_result)
        
        try:
            # Execute the tool
            result = tool._run(**parameters)
            
            print(f"✅ Tool execution completed successfully")
            
            # Update state with tool results
            new_state = self.state_manager.add_tool_call(
                state, tool_name, parameters, result
            )
            
            new_state = self.state_manager.update_state(
                new_state,
                current_step="execute_tool",
                intermediate_results={
                    **state.get("intermediate_results", {}),
                    tool_name: result
                },
                context={
                    **state.get("context", {}),
                    "tool_execution_complete": True,
                    "last_tool_success": result.get("success", True)
                }
            )
            
            return new_state
            
        except Exception as e:
            print(f"❌ Tool execution failed: {str(e)}")
            error_result = {
                "success": False,
                "error": str(e)
            }
            return self._handle_tool_error(state, tool_name, parameters, error_result)
    
    def _handle_tool_error(self, state: AgentState, tool_name: str, parameters: Dict, error_result: Dict) -> AgentState:
        """Handle tool execution errors"""
        new_state = self.state_manager.add_tool_call(
            state, tool_name, parameters, error_result
        )
        
        new_state = self.state_manager.update_state(
            new_state,
            current_step="execute_tool",
            intermediate_results={
                **state.get("intermediate_results", {}),
                tool_name: error_result
            },
            context={
                **state.get("context", {}),
                "tool_execution_complete": True,
                "last_tool_success": False,
                "error": error_result["error"]
            }
        )
        
        return new_state
    
    def _synthesize_response_node(self, state: AgentState) -> AgentState:
        """Synthesize the final response based on tool results"""
        print("🧠 Synthesizing response from tool results...")
        
        # Get the tool results
        tool_results = state.get("intermediate_results", {})
        conversation_history = self.state_manager.get_conversation_history(state)
        
        # Create synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(
            state["current_query"],
            tool_results,
            conversation_history
        )
        
        try:
            # Generate response using LLM
            messages = [
                SystemMessage(content=synthesis_prompt),
                HumanMessage(content=f"User query: {state['current_query']}")
            ]
            
            response = self.llm.invoke(messages)
            synthesized_response = response.content
            
            print("✨ Response synthesized successfully")
            
            # Add the AI response to messages
            ai_message = AIMessage(content=synthesized_response)
            new_state = self.state_manager.add_message(state, ai_message)
            
            new_state = self.state_manager.update_state(
                new_state,
                current_step="synthesize_response",
                context={
                    **state.get("context", {}),
                    "response_synthesized": True,
                    "final_response": synthesized_response
                }
            )
            
            return new_state
            
        except Exception as e:
            print(f"❌ Response synthesis failed: {str(e)}")
            
            # Fallback response
            fallback_response = self._create_fallback_response(state, str(e))
            ai_message = AIMessage(content=fallback_response)
            new_state = self.state_manager.add_message(state, ai_message)
            
            new_state = self.state_manager.update_state(
                new_state,
                current_step="synthesize_response",
                context={
                    **state.get("context", {}),
                    "response_synthesized": True,
                    "final_response": fallback_response,
                    "synthesis_error": str(e)
                }
            )
            
            return new_state
    
    def _check_completion_node(self, state: AgentState) -> AgentState:
        """Check if the query has been fully addressed or if more tools are needed"""
        print("🎯 Checking if query is fully addressed...")
        
        # Simple completion check - in a more advanced system, this could use LLM to determine
        # if the query has been fully addressed or if additional tools are needed
        
        context = state.get("context", {})
        is_complete = (
            context.get("response_synthesized", False) and
            context.get("tool_execution_complete", False)
        )
        
        new_state = self.state_manager.update_state(
            state,
            current_step="check_completion",
            context={
                **context,
                "completion_check_done": True,
                "is_complete": is_complete
            }
        )
        
        if is_complete:
            print("✅ Query processing complete")
        else:
            print("🔄 Additional processing needed")
        
        return new_state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the workflow should continue or end"""
        context = state.get("context", {})
        
        # End if we have a complete response
        if context.get("is_complete", False):
            return "end"
        
        # Continue if we need more processing (this is a simple implementation)
        return "end"  # For now, always end after one iteration
    
    def _create_synthesis_prompt(self, query: str, tool_results: Dict, conversation_history: str) -> str:
        """Create a prompt for synthesizing the final response"""
        
        # Format tool results
        results_summary = []
        for tool_name, result in tool_results.items():
            if isinstance(result, dict):
                if result.get("success", True):
                    if tool_name == "weather_info" and "summary" in result:
                        results_summary.append(f"Weather Information:\n{result['summary']}")
                    elif tool_name == "web_search" and "results" in result:
                        search_results = []
                        for item in result["results"][:3]:  # Top 3 results
                            search_results.append(f"- {item.get('title', 'N/A')}: {item.get('snippet', 'N/A')}")
                        results_summary.append(f"Web Search Results:\n" + "\n".join(search_results))
                    elif tool_name == "rag_search" and "results" in result:
                        rag_results = []
                        for item in result["results"][:3]:  # Top 3 results
                            rag_results.append(f"- {item.get('content', 'N/A')[:200]}...")
                        results_summary.append(f"Knowledge Base Results:\n" + "\n".join(rag_results))
                    else:
                        results_summary.append(f"{tool_name}: {str(result)}")
                else:
                    results_summary.append(f"{tool_name} Error: {result.get('error', 'Unknown error')}")
        
        results_text = "\n\n".join(results_summary) if results_summary else "No tool results available."
        
        return f"""You are an intelligent assistant that synthesizes information from various tools to answer user queries.

User Query: {query}

Tool Results:
{results_text}

Conversation History:
{conversation_history}

Instructions:
1. Provide a comprehensive and helpful answer based on the tool results
2. If there are errors in tool results, acknowledge them and provide what information you can
3. Be conversational and natural in your response
4. If the information is incomplete, suggest what the user might do next
5. Maintain context from the conversation history

Please synthesize a response that directly addresses the user's query using the available information."""
    
    def _create_fallback_response(self, state: AgentState, error: str) -> str:
        """Create a fallback response when synthesis fails"""
        return f"""I apologize, but I encountered an issue while processing your query: "{state['current_query']}". 

Error: {error}

I was able to gather some information using my tools, but had trouble synthesizing a complete response. Please try rephrasing your question or ask for specific information that might be easier to process."""
    
    def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a user query through the complete workflow"""
        print(f"\n🚀 Processing query: {query}")
        print("=" * 50)
        
        try:
            # Create initial state
            initial_state = self.state_manager.create_initial_state(query, session_id)
            
            # Run the workflow
            final_state = self.app.invoke(initial_state)
            
            # Extract the response
            messages = final_state["messages"]
            if messages and isinstance(messages[-1], AIMessage):
                response = messages[-1].content
            else:
                response = "I apologize, but I couldn't generate a response to your query."
            
            print("=" * 50)
            print(f"✅ Query processing completed\n")
            
            return {
                "success": True,
                "query": query,
                "response": response,
                "session_id": final_state["session_id"],
                "tools_used": [call["tool"] for call in final_state["tool_calls"]],
                "state": final_state
            }
            
        except Exception as e:
            print(f"❌ Query processing failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "response": f"I apologize, but I encountered an error while processing your query: {str(e)}"
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get information about available tools"""
        return self.tool_registry.get_tools_info()
    
    def add_documents_to_rag(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to the RAG tool's knowledge base"""
        rag_tool = self.tool_registry.get_tool("rag_search")
        if rag_tool:
            rag_tool.add_documents(documents, metadatas)
            print(f"Added {len(documents)} documents to RAG knowledge base")
        else:
            print("RAG tool not available")