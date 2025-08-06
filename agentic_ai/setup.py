#!/usr/bin/env python3
"""
Setup script for Agentic AI project
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    print("✅ Dependencies installed successfully")
    return True


def setup_environment():
    """Set up environment variables"""
    print("🔐 Setting up environment...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print("❌ .env.example file not found")
        return False
    
    if not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys")
    else:
        print("✅ .env file already exists")
    
    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "data/documents", 
        "data/vector_store"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")
    return True


def download_models():
    """Download required models"""
    print("🤖 Downloading embedding models...")
    
    try:
        # Import here to avoid import errors during setup
        from sentence_transformers import SentenceTransformer
        
        # Download the embedding model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embedding model downloaded")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not download embedding model: {e}")
        print("   The model will be downloaded on first use")
        return True


def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import langchain
        import langgraph
        import openai
        import chromadb
        import sentence_transformers
        print("✅ All imports successful")
        
        # Test configuration
        from config import Config
        print("✅ Configuration module loaded")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Agentic AI Setup")
    print("=" * 50)
    
    steps = [
        ("Check Python version", check_python_version),
        ("Install dependencies", install_dependencies),
        ("Setup environment", setup_environment),
        ("Create directories", create_directories),
        ("Download models", download_models),
        ("Run tests", run_tests)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_function():
            failed_steps.append(step_name)
    
    print("\n" + "=" * 50)
    
    if failed_steps:
        print(f"❌ Setup completed with errors in: {', '.join(failed_steps)}")
        print("\nPlease resolve the errors above and run setup again.")
        return False
    else:
        print("✅ Setup completed successfully!")
        print("\n🎉 Your Agentic AI system is ready!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python main.py")
        print("3. Type 'help' for available commands")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)