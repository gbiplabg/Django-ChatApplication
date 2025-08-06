#!/usr/bin/env python3
"""
Agentic AI Demo Script

This script demonstrates the capabilities of the Agentic AI system
by running various example queries and showing the agent's responses.
"""

import time
import json
from agent import AgenticAI
from config import Config


class AgenticAIDemo:
    """Demo class to showcase the Agentic AI capabilities"""
    
    def __init__(self):
        print("🎬 Initializing Agentic AI Demo...")
        try:
            Config.validate()
            self.agent = AgenticAI()
            print("✅ Demo ready!")
        except Exception as e:
            print(f"❌ Demo initialization failed: {e}")
            print("Please make sure you have set up your API keys in .env file")
            raise
    
    def run_demo(self):
        """Run the complete demonstration"""
        print("\n" + "=" * 80)
        print("🎯 AGENTIC AI DEMONSTRATION")
        print("=" * 80)
        print("\nThis demo showcases the intelligent routing and tool execution")
        print("capabilities of our agentic AI system.")
        print("\n" + "=" * 80)
        
        # Demo scenarios
        scenarios = [
            {
                "title": "🌤️  Weather Intelligence",
                "description": "Demonstrating weather information retrieval with location extraction",
                "queries": [
                    "What's the weather like in Tokyo?",
                    "Will it rain in London tomorrow?",
                    "Show me the 3-day forecast for New York"
                ]
            },
            {
                "title": "🔍 Web Search Intelligence", 
                "description": "Demonstrating real-time web search capabilities",
                "queries": [
                    "Search for latest news about artificial intelligence",
                    "Find information about recent developments in quantum computing",
                    "What happened with OpenAI recently?"
                ]
            },
            {
                "title": "📚 Knowledge Base Intelligence",
                "description": "Demonstrating RAG-based search through local knowledge",
                "queries": [
                    "What is machine learning?",
                    "Explain the concept of artificial intelligence",
                    "Tell me about data science processes"
                ]
            },
            {
                "title": "🧠 Multi-Tool Intelligence",
                "description": "Demonstrating complex queries that might use multiple tools",
                "queries": [
                    "I'm planning a trip to Paris. What's the weather like and what are the latest travel updates?",
                    "Explain climate change and tell me about current environmental news",
                    "What is Python programming and find me the latest Python news"
                ]
            }
        ]
        
        for scenario in scenarios:
            self.run_scenario(scenario)
            self.pause_for_user()
        
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED!")
        print("=" * 80)
        print("\nThe Agentic AI system has demonstrated:")
        print("✅ Intelligent query routing")
        print("✅ Automatic parameter extraction")
        print("✅ Dynamic tool execution")
        print("✅ Context-aware responses")
        print("✅ Error handling and fallbacks")
        print("\nTry the interactive mode: python main.py")
        print("=" * 80)
    
    def run_scenario(self, scenario):
        """Run a specific demo scenario"""
        print(f"\n🎬 {scenario['title']}")
        print("-" * 60)
        print(f"📝 {scenario['description']}")
        print("-" * 60)
        
        for i, query in enumerate(scenario['queries'], 1):
            print(f"\n💭 Query {i}: {query}")
            print("⏳ Processing...")
            
            start_time = time.time()
            result = self.agent.process_query(query)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if result["success"]:
                print(f"\n🤖 Response:")
                print(f"{result['response']}")
                print(f"\n📊 Metrics:")
                print(f"   • Processing time: {processing_time:.2f}s")
                print(f"   • Tools used: {', '.join(result['tools_used'])}")
                print(f"   • Session ID: {result['session_id']}")
            else:
                print(f"\n❌ Error: {result['error']}")
                print(f"🤖 Fallback response: {result['response']}")
            
            print("-" * 40)
    
    def pause_for_user(self):
        """Pause and wait for user input"""
        try:
            input("\n⏸️  Press Enter to continue to next scenario (or Ctrl+C to exit)...")
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted by user. Goodbye!")
            exit(0)
    
    def quick_demo(self):
        """Run a quick demo with just one example from each tool"""
        print("\n🚀 Quick Demo - One example from each tool")
        print("=" * 50)
        
        quick_queries = [
            ("Weather Tool", "What's the weather in San Francisco?"),
            ("Web Search Tool", "Latest AI news"),
            ("RAG Tool", "What is artificial intelligence?")
        ]
        
        for tool_name, query in quick_queries:
            print(f"\n🔧 Testing {tool_name}")
            print(f"Query: {query}")
            
            result = self.agent.process_query(query)
            
            if result["success"]:
                # Truncate long responses for quick demo
                response = result['response']
                if len(response) > 200:
                    response = response[:200] + "..."
                print(f"Response: {response}")
                print(f"Tools used: {', '.join(result['tools_used'])}")
            else:
                print(f"Error: {result['error']}")
            
            print("-" * 30)
    
    def interactive_demo(self):
        """Run an interactive demo where user can input queries"""
        print("\n🎮 Interactive Demo Mode")
        print("=" * 40)
        print("Enter queries to see the agent in action!")
        print("Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            try:
                query = input("\n💭 Your query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                print("⏳ Processing...")
                result = self.agent.process_query(query)
                
                if result["success"]:
                    print(f"\n🤖 {result['response']}")
                    print(f"\n📊 Tools used: {', '.join(result['tools_used'])}")
                else:
                    print(f"\n❌ Error: {result['error']}")
                    print(f"🤖 {result['response']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic AI Demo")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "interactive"],
        default="full",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    try:
        demo = AgenticAIDemo()
        
        if args.mode == "full":
            demo.run_demo()
        elif args.mode == "quick":
            demo.quick_demo()
        elif args.mode == "interactive":
            demo.interactive_demo()
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nPlease ensure:")
        print("1. You have set up your .env file with API keys")
        print("2. All dependencies are installed (run: python setup.py)")
        print("3. Your internet connection is working")


if __name__ == "__main__":
    main()