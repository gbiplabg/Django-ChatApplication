import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the agentic AI project"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4-turbo-preview"
    
    # Weather API Configuration
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    # Search Configuration
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
    
    # LangSmith Configuration (optional)
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "agentic-ai-project")
    
    # RAG Configuration
    RAG_CHUNK_SIZE = 1000
    RAG_CHUNK_OVERLAP = 200
    RAG_TOP_K = 5
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = "./data/vector_store"
    DOCUMENTS_PATH = "./data/documents"
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        if not cls.WEATHER_API_KEY:
            print("Warning: WEATHER_API_KEY not set. Weather tool will not work.")
        
        return True