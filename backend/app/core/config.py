import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = int(os.getenv("PORT", 8000))  # Render uses PORT env var
    DEBUG: bool = os.getenv("ENVIRONMENT", "development") != "production"
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./vachanamrut.db"
    
    # AI Service Configuration
    OPENAI_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None
    AI_MODEL: str = "gpt-3.5-turbo"
    
    # Vector Database Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-west1-gcp"
    PINECONE_INDEX_NAME: str = "vachanamrut-embeddings"
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Performance Settings
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    TOP_K_RESULTS: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env

settings = Settings()