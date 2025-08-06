#!/usr/bin/env python3
"""
Agentic AI - Main Application Interface

A comprehensive agentic AI system using LangGraph and LangChain with custom tools
including RAG-based search, web search, and weather API integration.
"""

import os
import sys
import json
from typing import Dict, Any
import argparse

from agent import AgenticAI
from config import Config


class AgenticAIApp:
    """Main application class for the Agentic AI system"""
    
    def __init__(self):
        print("🤖 Initializing Agentic AI System...")
        print("=" * 60)
        
        # Validate configuration
        try:
            Config.validate()
            print("✅ Configuration validated")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("Please check your environment variables in .env file")
            sys.exit(1)
        
        # Initialize the agent
        try:
            self.agent = AgenticAI()
            print("✅ Agent initialized successfully")
            print("✅ Tools loaded:", [tool["name"] for tool in self.agent.get_available_tools()])
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            sys.exit(1)
        
        print("=" * 60)
        print("🚀 Agentic AI System Ready!")
        print()
    
    def interactive_mode(self):
        """Run the application in interactive mode"""
        print("Welcome to Agentic AI Interactive Mode!")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 40)
        
        session_id = None
        
        while True:
            try:
                # Get user input
                query = input("\n🤔 You: ").strip()
                
                if not query:
                    continue
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'help':
                    self.show_help()
                    continue
                elif query.lower() == 'tools':
                    self.show_tools()
                    continue
                elif query.lower().startswith('add_doc '):
                    self.add_document(query[8:])
                    continue
                elif query.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                # Process the query
                print(f"\n🤖 Assistant: Processing your query...")
                result = self.agent.process_query(query, session_id)
                
                if result["success"]:
                    print(f"\n🤖 Assistant: {result['response']}")
                    
                    # Update session ID for continuity
                    session_id = result["session_id"]
                    
                    # Show tools used (optional debug info)
                    if result["tools_used"]:
                        print(f"\n🔧 Tools used: {', '.join(result['tools_used'])}")
                else:
                    print(f"\n❌ Error: {result['error']}")
                    print(f"🤖 Assistant: {result['response']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
    
    def process_single_query(self, query: str) -> Dict[str, Any]:
        """Process a single query and return the result"""
        return self.agent.process_query(query)
    
    def show_help(self):
        """Show help information"""
        help_text = """
🤖 Agentic AI Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 General Commands:
  help              Show this help message
  tools             Show available tools and their descriptions
  clear             Clear the screen
  quit/exit/q       Exit the application

📚 Document Management:
  add_doc <text>    Add a document to the RAG knowledge base

💡 Example Queries:
  • "What's the weather in New York?"
  • "Search for latest news about artificial intelligence"
  • "Explain machine learning concepts"
  • "What's the forecast for London for 3 days?"
  • "Find information about climate change"
  • "Tell me about Python programming"

🔧 Available Tools:
  • RAG Search: Search through your knowledge base
  • Web Search: Find current information on the internet
  • Weather Info: Get weather data and forecasts

💡 Tips:
  • The agent automatically selects the best tool for your query
  • You can ask follow-up questions in the same session
  • Be specific in your queries for better results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        print(help_text)
    
    def show_tools(self):
        """Show available tools and their descriptions"""
        tools = self.agent.get_available_tools()
        
        print("\n🔧 Available Tools:")
        print("━" * 50)
        
        for tool in tools:
            print(f"\n📋 {tool['name'].upper()}")
            print(f"   Description: {tool['description']}")
            print(f"   Parameters:")
            for param in tool['parameters']:
                required = "Required" if param['required'] else "Optional"
                default = f" (default: {param['default']})" if param.get('default') is not None else ""
                print(f"     • {param['name']} ({param['type']}) - {required}{default}")
                print(f"       {param['description']}")
        
        print("━" * 50)
    
    def add_document(self, document_text: str):
        """Add a document to the RAG knowledge base"""
        if not document_text.strip():
            print("❌ Please provide document text")
            return
        
        try:
            self.agent.add_documents_to_rag([document_text])
            print("✅ Document added to knowledge base")
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
    
    def run_examples(self):
        """Run example queries to demonstrate the system"""
        examples = [
            {
                "query": "What is artificial intelligence?",
                "description": "RAG Search Example - Knowledge base query"
            },
            {
                "query": "What's the weather in London?",
                "description": "Weather Tool Example - Current weather query"
            },
            {
                "query": "Search for latest news about OpenAI",
                "description": "Web Search Example - Current information query"
            },
            {
                "query": "Explain machine learning algorithms",
                "description": "RAG Search Example - Educational content"
            },
            {
                "query": "Weather forecast for Tokyo for 2 days",
                "description": "Weather Tool Example - Forecast query"
            }
        ]
        
        print("🎯 Running Example Queries:")
        print("=" * 60)
        
        for i, example in enumerate(examples, 1):
            print(f"\n📝 Example {i}: {example['description']}")
            print(f"Query: {example['query']}")
            print("-" * 40)
            
            result = self.agent.process_query(example['query'])
            
            if result["success"]:
                print(f"Response: {result['response'][:300]}...")
                print(f"Tools used: {', '.join(result['tools_used'])}")
            else:
                print(f"Error: {result['error']}")
            
            print("-" * 40)
            
            # Wait for user input to continue
            input("Press Enter to continue to next example...")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Agentic AI - Intelligent agent with RAG, web search, and weather tools"
    )
    parser.add_argument(
        "--mode", 
        choices=["interactive", "examples", "query"],
        default="interactive",
        help="Application mode"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (for query mode)"
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Initialize the application
    app = AgenticAIApp()
    
    if args.mode == "interactive":
        app.interactive_mode()
    
    elif args.mode == "examples":
        app.run_examples()
    
    elif args.mode == "query":
        if not args.query:
            print("❌ Query mode requires --query parameter")
            sys.exit(1)
        
        result = app.process_single_query(args.query)
        
        if args.output == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            if result["success"]:
                print(f"Query: {result['query']}")
                print(f"Response: {result['response']}")
                print(f"Tools used: {', '.join(result['tools_used'])}")
            else:
                print(f"Error: {result['error']}")
                print(f"Response: {result['response']}")


if __name__ == "__main__":
    main()