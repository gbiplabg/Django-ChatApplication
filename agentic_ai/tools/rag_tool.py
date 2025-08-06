import os
import json
from typing import Dict, Any, List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .base_tool import BaseCustomTool, ToolParameter
from config import Config


class RAGSearchTool(BaseCustomTool):
    """RAG-based search tool for retrieving relevant information from documents"""
    
    tool_name = "rag_search"
    tool_description = "Search through uploaded documents using semantic similarity to find relevant information"
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query to find relevant documents",
            required=True
        ),
        ToolParameter(
            name="top_k",
            type="integer",
            description="Number of top results to return",
            required=False,
            default=5
        )
    ]
    
    def __init__(self):
        super().__init__()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.RAG_CHUNK_SIZE,
            chunk_overlap=Config.RAG_CHUNK_OVERLAP
        )
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load the vector store"""
        try:
            # Try to load existing vector store
            if os.path.exists(Config.VECTOR_STORE_PATH):
                self.vector_store = Chroma(
                    persist_directory=Config.VECTOR_STORE_PATH,
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing vector store with {self.vector_store._collection.count()} documents")
            else:
                # Create new vector store
                self._create_vector_store()
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self._create_sample_documents()
    
    def _create_vector_store(self):
        """Create a new vector store from documents"""
        documents_path = Config.DOCUMENTS_PATH
        
        if os.path.exists(documents_path):
            # Load documents from directory
            loader = DirectoryLoader(
                documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
        else:
            # Create sample documents if no documents directory exists
            documents = self._create_sample_documents()
        
        if documents:
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Create vector store
            os.makedirs(Config.VECTOR_STORE_PATH, exist_ok=True)
            self.vector_store = Chroma.from_documents(
                texts,
                self.embeddings,
                persist_directory=Config.VECTOR_STORE_PATH
            )
            self.vector_store.persist()
            print(f"Created vector store with {len(texts)} document chunks")
        else:
            print("No documents found to create vector store")
    
    def _create_sample_documents(self) -> List[Document]:
        """Create sample documents for demonstration"""
        sample_docs = [
            {
                "content": """
                Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
                that can perform tasks that typically require human intelligence. These tasks include learning, 
                reasoning, problem-solving, perception, and language understanding. AI systems can be categorized 
                into narrow AI (designed for specific tasks) and general AI (human-level intelligence across all domains).
                
                Machine Learning is a subset of AI that enables computers to learn and improve from experience 
                without being explicitly programmed. It uses algorithms to parse data, learn from it, and make 
                predictions or decisions. Common types include supervised learning, unsupervised learning, and 
                reinforcement learning.
                """,
                "metadata": {"source": "ai_basics.txt", "topic": "artificial_intelligence"}
            },
            {
                "content": """
                Python is a high-level, interpreted programming language known for its simplicity and readability. 
                It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes 
                code readability and a syntax that allows programmers to express concepts in fewer lines of code.
                
                Python is widely used in web development, data science, artificial intelligence, automation, 
                and scientific computing. Popular frameworks include Django and Flask for web development, 
                NumPy and Pandas for data analysis, and TensorFlow and PyTorch for machine learning.
                """,
                "metadata": {"source": "python_intro.txt", "topic": "programming"}
            },
            {
                "content": """
                Climate change refers to long-term shifts in global temperatures and weather patterns. While climate 
                variations occur naturally, human activities have been the main driver of climate change since the 1800s, 
                primarily through burning fossil fuels like coal, oil, and gas.
                
                The effects of climate change include rising temperatures, melting ice caps, rising sea levels, 
                extreme weather events, and ecosystem disruption. Mitigation strategies include renewable energy 
                adoption, energy efficiency improvements, and carbon capture technologies.
                """,
                "metadata": {"source": "climate_change.txt", "topic": "environment"}
            },
            {
                "content": """
                Data Science is an interdisciplinary field that combines statistics, computer science, and domain 
                expertise to extract insights from structured and unstructured data. The data science process 
                typically involves data collection, cleaning, exploration, modeling, and interpretation.
                
                Key skills for data scientists include programming (Python, R), statistics, machine learning, 
                data visualization, and domain knowledge. Common tools include Jupyter notebooks, pandas, 
                scikit-learn, matplotlib, and SQL databases.
                """,
                "metadata": {"source": "data_science.txt", "topic": "data_science"}
            }
        ]
        
        documents = []
        os.makedirs(Config.DOCUMENTS_PATH, exist_ok=True)
        
        for i, doc in enumerate(sample_docs):
            # Save sample document to file
            file_path = os.path.join(Config.DOCUMENTS_PATH, f"sample_doc_{i+1}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
            
            # Create Document object
            documents.append(Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            ))
        
        return documents
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add new documents to the vector store"""
        if not self.vector_store:
            self._create_vector_store()
        
        # Create Document objects
        docs = []
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            docs.append(Document(page_content=doc, metadata=metadata))
        
        # Split and add to vector store
        texts = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(texts)
        self.vector_store.persist()
    
    def _run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute RAG search"""
        try:
            params = self.validate_parameters({"query": query, "top_k": top_k})
            
            if not self.vector_store:
                return {
                    "success": False,
                    "error": "Vector store not initialized",
                    "results": []
                }
            
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                params["query"], 
                k=params["top_k"]
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            return {
                "success": True,
                "query": params["query"],
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def _arun(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Async version of RAG search"""
        return self._run(query, top_k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection"""
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        try:
            count = self.vector_store._collection.count()
            return {
                "total_documents": count,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": Config.RAG_CHUNK_SIZE,
                "chunk_overlap": Config.RAG_CHUNK_OVERLAP
            }
        except Exception as e:
            return {"error": str(e)}