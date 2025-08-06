import re
from typing import Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState, StateManager
from tools import create_tool_registry
from config import Config


class QueryRouter:
    """Intelligent routing system to determine which tool to use based on user queries"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.1
        )
        self.tool_registry = create_tool_registry()
        self.state_manager = StateManager()
        
        # Define routing patterns and keywords
        self.routing_patterns = {
            "weather_info": {
                "keywords": [
                    "weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy",
                    "humidity", "wind", "climate", "hot", "cold", "warm", "cool",
                    "storm", "precipitation", "degrees", "celsius", "fahrenheit"
                ],
                "patterns": [
                    r"weather\s+in\s+\w+",
                    r"temperature\s+in\s+\w+",
                    r"forecast\s+for\s+\w+",
                    r"how\s+hot\s+is\s+it",
                    r"how\s+cold\s+is\s+it",
                    r"will\s+it\s+rain",
                    r"is\s+it\s+going\s+to\s+rain"
                ]
            },
            "web_search": {
                "keywords": [
                    "search", "find", "latest", "current", "recent", "news", "today",
                    "what happened", "update", "breaking", "trending", "online",
                    "internet", "website", "article", "blog", "information about"
                ],
                "patterns": [
                    r"search\s+for\s+.+",
                    r"find\s+information\s+about\s+.+",
                    r"what\s+is\s+the\s+latest\s+.+",
                    r"current\s+news\s+about\s+.+",
                    r"recent\s+developments\s+in\s+.+",
                    r"what\s+happened\s+to\s+.+"
                ]
            },
            "rag_search": {
                "keywords": [
                    "explain", "definition", "what is", "how does", "tell me about",
                    "describe", "concept", "theory", "principle", "understand",
                    "learn", "knowledge", "information", "details", "overview"
                ],
                "patterns": [
                    r"what\s+is\s+.+",
                    r"explain\s+.+",
                    r"tell\s+me\s+about\s+.+",
                    r"how\s+does\s+.+\s+work",
                    r"describe\s+.+",
                    r"definition\s+of\s+.+"
                ]
            }
        }
    
    def analyze_query_intent(self, query: str, state: AgentState) -> Dict[str, any]:
        """Analyze the user query to determine intent and extract parameters"""
        query_lower = query.lower().strip()
        
        # Calculate scores for each tool
        tool_scores = {}
        extracted_params = {}
        
        for tool_name, patterns in self.routing_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in query_lower:
                    score += 2
            
            # Check regex patterns
            for pattern in patterns["patterns"]:
                if re.search(pattern, query_lower):
                    score += 5
            
            tool_scores[tool_name] = score
            
            # Extract parameters based on tool type
            if tool_name == "weather_info" and score > 0:
                extracted_params[tool_name] = self._extract_weather_params(query)
            elif tool_name == "web_search" and score > 0:
                extracted_params[tool_name] = self._extract_search_params(query)
            elif tool_name == "rag_search" and score > 0:
                extracted_params[tool_name] = self._extract_rag_params(query)
        
        # Determine the best tool
        best_tool = max(tool_scores.items(), key=lambda x: x[1])
        
        # If no clear winner, use LLM to decide
        if best_tool[1] == 0 or len([score for score in tool_scores.values() if score == best_tool[1]]) > 1:
            llm_decision = self._llm_route_decision(query, state)
            return llm_decision
        
        return {
            "selected_tool": best_tool[0],
            "confidence": min(best_tool[1] / 10, 1.0),  # Normalize to 0-1
            "parameters": extracted_params.get(best_tool[0], {}),
            "reasoning": f"Pattern matching selected {best_tool[0]} with score {best_tool[1]}"
        }
    
    def _extract_weather_params(self, query: str) -> Dict[str, any]:
        """Extract weather-specific parameters from query"""
        params = {}
        
        # Extract location
        location_patterns = [
            r"weather\s+in\s+([^?\n]+)",
            r"temperature\s+in\s+([^?\n]+)",
            r"forecast\s+for\s+([^?\n]+)",
            r"in\s+([A-Z][a-zA-Z\s,]+)",
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip().rstrip('?.,!').strip()
                if len(location) > 2:
                    params["location"] = location
                    break
        
        # Extract forecast days
        forecast_patterns = [
            r"(\d+)\s*days?\s+forecast",
            r"forecast\s+for\s+(\d+)\s*days?",
            r"next\s+(\d+)\s*days?",
        ]
        
        for pattern in forecast_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["forecast_days"] = min(int(match.group(1)), 5)
                break
        
        # Extract units
        if "celsius" in query.lower() or "°c" in query.lower():
            params["units"] = "metric"
        elif "fahrenheit" in query.lower() or "°f" in query.lower():
            params["units"] = "imperial"
        
        return params
    
    def _extract_search_params(self, query: str) -> Dict[str, any]:
        """Extract web search parameters from query"""
        params = {"query": query}
        
        # Extract number of results
        num_patterns = [
            r"(\d+)\s+results?",
            r"top\s+(\d+)",
            r"first\s+(\d+)",
        ]
        
        for pattern in num_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["num_results"] = min(int(match.group(1)), 10)
                break
        
        # Check if they want detailed content
        if any(word in query.lower() for word in ["detailed", "full", "complete", "content"]):
            params["include_snippets"] = True
        
        return params
    
    def _extract_rag_params(self, query: str) -> Dict[str, any]:
        """Extract RAG search parameters from query"""
        params = {"query": query}
        
        # Extract number of results
        num_patterns = [
            r"(\d+)\s+examples?",
            r"(\d+)\s+results?",
            r"top\s+(\d+)",
        ]
        
        for pattern in num_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["top_k"] = min(int(match.group(1)), 10)
                break
        
        return params
    
    def _llm_route_decision(self, query: str, state: AgentState) -> Dict[str, any]:
        """Use LLM to make routing decision when patterns are unclear"""
        
        # Get available tools info
        tools_info = self.tool_registry.get_tools_info()
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tools_info
        ])
        
        # Get conversation context
        context = self.state_manager.get_conversation_history(state)
        
        system_prompt = f"""You are a query routing assistant. Your job is to determine which tool should be used to answer a user's query.

Available tools:
{tools_description}

Guidelines:
1. weather_info: Use for weather-related queries, forecasts, temperature questions
2. web_search: Use for current events, recent information, news, or when you need up-to-date information
3. rag_search: Use for general knowledge questions, explanations, definitions, or educational content

Consider the conversation context and choose the most appropriate tool.

Respond with a JSON object containing:
- "selected_tool": the name of the tool to use
- "confidence": a number between 0 and 1 indicating your confidence
- "parameters": a dictionary of parameters to extract from the query
- "reasoning": a brief explanation of your decision

Query: "{query}"

Context: {context}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Route this query: {query}")
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the JSON response
            import json
            try:
                decision = json.loads(response.content)
                
                # Validate the decision
                if decision.get("selected_tool") in self.tool_registry.get_tool_names():
                    return decision
                
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            print(f"LLM routing error: {e}")
        
        # Fallback to web search for unknown queries
        return {
            "selected_tool": "web_search",
            "confidence": 0.5,
            "parameters": {"query": query},
            "reasoning": "Fallback to web search due to routing uncertainty"
        }
    
    def route_query(self, query: str, state: AgentState) -> Tuple[str, Dict[str, any], float]:
        """Main routing method that returns tool name, parameters, and confidence"""
        
        # Analyze the query
        routing_decision = self.analyze_query_intent(query, state)
        
        tool_name = routing_decision["selected_tool"]
        parameters = routing_decision["parameters"]
        confidence = routing_decision["confidence"]
        
        # Get the tool and validate parameters
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            try:
                # Validate and complete parameters
                validated_params = self._complete_parameters(tool, parameters, query, state)
                return tool_name, validated_params, confidence
            except Exception as e:
                print(f"Parameter validation error: {e}")
                # Return basic parameters
                return tool_name, parameters, confidence
        
        # Fallback
        return "web_search", {"query": query}, 0.5
    
    def _complete_parameters(self, tool, provided_params: Dict, query: str, state: AgentState) -> Dict[str, any]:
        """Complete missing parameters using context and defaults"""
        completed_params = provided_params.copy()
        
        # Get tool parameter requirements
        required_params = [param for param in tool.parameters if param.required]
        
        for param in required_params:
            if param.name not in completed_params:
                # Try to extract from context or use intelligent defaults
                if param.name == "location" and tool.tool_name == "weather_info":
                    # Try to extract location from conversation history or use a default
                    location_hints = self.state_manager.extract_context_for_tool(state, "weather_info").get("location_hints", [])
                    if location_hints:
                        completed_params["location"] = location_hints[0]
                    else:
                        # Ask user or use default - for now, we'll extract from query
                        completed_params["location"] = self._extract_location_from_query(query)
                elif param.name == "query":
                    completed_params["query"] = query
        
        return completed_params
    
    def _extract_location_from_query(self, query: str) -> str:
        """Extract location from query or return default"""
        # Simple location extraction
        words = query.split()
        
        # Look for capitalized words that might be locations
        potential_locations = []
        for i, word in enumerate(words):
            if word[0].isupper() and word.lower() not in ["I", "The", "A", "An", "This", "That", "What", "Where", "When", "How", "Why"]:
                # Check if it's part of a location phrase
                location_phrase = word
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    location_phrase += " " + words[i + 1]
                    if i + 2 < len(words) and words[i + 2][0].isupper():
                        location_phrase += " " + words[i + 2]
                potential_locations.append(location_phrase)
        
        if potential_locations:
            return potential_locations[0]
        
        # Default location if none found
        return "New York, NY, USA"