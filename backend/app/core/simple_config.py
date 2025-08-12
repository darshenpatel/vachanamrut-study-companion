import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simple configuration without external dependencies
class SimpleSettings:
    """Simple settings for the spiritual guidance system"""
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vachanamrut.db")
    
    # AI Service Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    AI_MODEL = os.getenv("AI_MODEL", "gpt-3.5-turbo")
    
    # Vector Database Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vachanamrut-embeddings")
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Performance Settings
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Global settings instance
settings = SimpleSettings()