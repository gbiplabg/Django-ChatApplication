# 🤖 Agentic AI - Intelligent Multi-Tool Agent

A comprehensive agentic AI system built with **LangGraph** and **LangChain** that intelligently routes user queries to specialized tools including RAG-based search, web search, and weather APIs. The agent maintains conversation state, remembers context, and dynamically calls tools based on user intent.

## 🌟 Features

- **🧠 Intelligent Query Routing**: Automatically determines the best tool for each query using pattern matching and LLM-based decision making
- **📚 RAG-Based Search**: Semantic search through your own document collection using vector embeddings
- **🌐 Web Search**: Real-time web search using DuckDuckGo for current information
- **🌤️ Weather Information**: Current weather and forecasts using OpenWeatherMap API
- **💾 State Management**: Maintains conversation context and memory across interactions
- **🔄 Dynamic Tool Calling**: Collects parameters automatically and executes tools based on user queries
- **📊 LangGraph Workflow**: Structured workflow with proper state flow and error handling

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic AI System                        │
├─────────────────────────────────────────────────────────────┤
│  📝 User Query                                              │
│           ↓                                                 │
│  🔍 Query Analysis (LangGraph Node)                         │
│           ↓                                                 │
│  🧭 Intelligent Routing (Pattern + LLM)                     │
│           ↓                                                 │
│  ⚡ Tool Execution (Dynamic Parameter Collection)           │
│           ↓                                                 │
│  🧠 Response Synthesis (LLM-based)                          │
│           ↓                                                 │
│  💬 Final Response                                          │
└─────────────────────────────────────────────────────────────┘
```

### 🔧 Available Tools

1. **RAG Search Tool** (`rag_search`)
   - Semantic search through document collections
   - Uses HuggingFace embeddings and Chroma vector store
   - Automatically chunks and indexes documents

2. **Web Search Tool** (`web_search`)
   - Real-time web search using DuckDuckGo
   - Content extraction and summarization
   - No API key required

3. **Weather Tool** (`weather_info`)
   - Current weather conditions
   - Multi-day forecasts
   - Location-based queries with geocoding

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (required)
- OpenWeatherMap API key (optional, for weather features)

### Installation

1. **Clone and navigate to the project:**
```bash
cd /workspace/agentic_ai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
WEATHER_API_KEY=your_weather_api_key_here  # Optional
```

4. **Run the application:**
```bash
python main.py
```

## 📖 Usage

### Interactive Mode (Default)

```bash
python main.py
```

This starts an interactive chat session where you can:
- Ask questions naturally
- Get help with `help` command
- View available tools with `tools` command
- Add documents with `add_doc <text>`

### Single Query Mode

```bash
python main.py --mode query --query "What's the weather in Tokyo?"
```

### Example Demonstrations

```bash
python main.py --mode examples
```

### Command Line Options

```bash
python main.py --help
```

## 💡 Example Queries

### Weather Queries
```
"What's the weather in New York?"
"Weather forecast for London for 3 days"
"How hot is it in Tokyo?"
"Will it rain in Paris tomorrow?"
```

### Web Search Queries
```
"Search for latest news about artificial intelligence"
"Find information about climate change"
"What happened to OpenAI recently?"
"Current developments in quantum computing"
```

### RAG Search Queries
```
"What is machine learning?"
"Explain artificial intelligence concepts"
"Tell me about Python programming"
"How does data science work?"
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM operations |
| `WEATHER_API_KEY` | No | OpenWeatherMap API key for weather data |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing (true/false) |
| `LANGCHAIN_API_KEY` | No | LangSmith API key for tracing |

### Customization

#### Adding Documents to RAG

You can add your own documents to the RAG system:

1. **Via Interactive Mode:**
```
add_doc Your document content here...
```

2. **Via Code:**
```python
from agent import AgenticAI

agent = AgenticAI()
documents = ["Document 1 content", "Document 2 content"]
agent.add_documents_to_rag(documents)
```

3. **Via File Directory:**
Place `.txt` files in `./data/documents/` directory before starting the application.

#### Customizing Tools

Add new tools by extending the `BaseCustomTool` class:

```python
from tools.base_tool import BaseCustomTool, ToolParameter

class MyCustomTool(BaseCustomTool):
    tool_name = "my_tool"
    tool_description = "Description of what my tool does"
    parameters = [
        ToolParameter(
            name="param1",
            type="string", 
            description="Parameter description",
            required=True
        )
    ]
    
    def _run(self, param1: str):
        # Your tool logic here
        return {"result": "success"}
```

## 🏗️ Project Structure

```
agentic_ai/
├── agent.py              # Main LangGraph agent implementation
├── config.py             # Configuration management
├── main.py               # CLI application interface
├── routing.py            # Intelligent query routing system
├── state.py              # State management for conversations
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── tools/                # Custom tools directory
│   ├── __init__.py
│   ├── base_tool.py      # Base tool class
│   ├── rag_tool.py       # RAG search implementation
│   ├── web_search_tool.py # Web search implementation
│   └── weather_tool.py   # Weather API integration
└── data/                 # Data storage (auto-created)
    ├── documents/        # RAG documents
    └── vector_store/     # Vector database
```

## 🔄 How It Works

### 1. Query Analysis
The agent receives a user query and analyzes it to understand the intent and extract potential parameters.

### 2. Intelligent Routing
Using a combination of pattern matching and LLM-based routing, the system determines which tool is most appropriate:

- **Weather patterns**: "weather", "temperature", "forecast", location names
- **Search patterns**: "search", "find", "latest", "current", "news"
- **Knowledge patterns**: "what is", "explain", "tell me about", "how does"

### 3. Parameter Collection
The agent automatically extracts and validates parameters needed for the selected tool:

- **Weather**: Location, forecast days, temperature units
- **Web Search**: Search terms, number of results
- **RAG Search**: Query terms, result count

### 4. Tool Execution
The selected tool is executed with the collected parameters, handling errors gracefully.

### 5. Response Synthesis
The LLM synthesizes the tool results into a natural, conversational response that directly addresses the user's query.

## 🛠️ Development

### Adding New Tools

1. Create a new tool class inheriting from `BaseCustomTool`
2. Define tool metadata and parameters
3. Implement `_run` and `_arun` methods
4. Register the tool in `tools/__init__.py`
5. Add routing patterns in `routing.py`

### Extending Routing Logic

Modify `routing.py` to add new routing patterns:

```python
self.routing_patterns["new_tool"] = {
    "keywords": ["keyword1", "keyword2"],
    "patterns": [r"pattern1", r"pattern2"]
}
```

### State Management

The agent maintains rich state including:
- Conversation history
- Tool execution history
- Collected parameters
- Context and metadata
- Session information

## 📊 API Integration

### OpenAI Integration
- Uses GPT-4 for intelligent routing and response synthesis
- Configurable model and temperature settings
- Error handling and fallback responses

### Vector Database
- Chroma vector store for RAG functionality
- HuggingFace embeddings for semantic search
- Persistent storage with automatic indexing

### External APIs
- OpenWeatherMap for weather data
- DuckDuckGo for web search (no API key required)
- Extensible architecture for additional APIs

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   Error: OPENAI_API_KEY is required
   ```
   Solution: Set your OpenAI API key in the `.env` file

2. **Weather Tool Not Working**
   ```
   Warning: WEATHER_API_KEY not set
   ```
   Solution: Get a free API key from OpenWeatherMap

3. **Vector Store Issues**
   ```
   Error initializing vector store
   ```
   Solution: Delete `./data/vector_store/` and restart

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain'
   ```
   Solution: Install dependencies with `pip install -r requirements.txt`

### Debug Mode

Enable verbose logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **LangChain** for the foundational framework
- **LangGraph** for state management and workflows
- **OpenAI** for powerful language models
- **HuggingFace** for embeddings and transformers
- **Chroma** for vector storage

---

**Happy Coding! 🚀**

For questions or support, please open an issue in the repository.