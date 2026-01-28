# Development Guidelines - Vachanamrut Study Companion

## Frontend Development Guidelines (React + TypeScript)

### Component Architecture

#### Component Design Principles
- **Single Responsibility**: Each component has one clear purpose
- **Composition over Inheritance**: Build complex UIs from simple components
- **Props Interface**: Always define TypeScript interfaces for props
- **Minimal State**: Keep component state as minimal as possible

#### Component Structure Template
```tsx
// components/ui/Button.tsx - Using class-variance-authority for variants
import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/utils/formatting";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-gray-900 text-white hover:bg-gray-800",
        destructive: "bg-red-600 text-white hover:bg-red-700",
        outline: "border border-gray-200 bg-white text-gray-900 hover:bg-gray-50",
        secondary: "bg-gray-100 text-gray-900 hover:bg-gray-200",
        ghost: "hover:bg-gray-100 hover:text-gray-900",
        link: "text-gray-900 underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md gap-1.5 px-3",
        lg: "h-10 rounded-md px-6",
        icon: "h-9 w-9 rounded-md",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
```

### State Management

#### Context Pattern for Global State
```tsx
// context/ChatContext.tsx
import React, { createContext, useContext, useReducer } from 'react';

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  currentTheme?: string;
}

type ChatAction = 
  | { type: 'ADD_MESSAGE'; payload: Message }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_THEME'; payload: string };

const ChatContext = createContext<{
  state: ChatState;
  dispatch: React.Dispatch<ChatAction>;
} | null>(null);

export const useChatContext = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChatContext must be used within ChatProvider');
  }
  return context;
};
```

#### Custom Hooks for Logic
```tsx
// hooks/useChat.ts
import { useState, useCallback } from 'react';
import { chatService } from '@/services/chatService';

export const useChat = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = useCallback(async (content: string) => {
    setIsLoading(true);
    try {
      const response = await chatService.sendMessage(content);
      setMessages(prev => [...prev, response]);
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { messages, isLoading, sendMessage };
};
```

### Styling Guidelines

#### Tailwind CSS Best Practices
- Use semantic class combinations
- Create reusable utility classes in `index.css`
- Leverage Tailwind's responsive design utilities
- Maintain consistent spacing scale
- Use neutral gray palette (inspired by Ramp/Linear)

```css
/* src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --background: #fafafa;
  --foreground: #0f0f0f;
  --surface: #ffffff;
  --border: rgba(0, 0, 0, 0.1);
  --text-secondary: #737373;
  --radius: 0.625rem;
}

@layer components {
  /* Card styles */
  .card {
    @apply bg-white rounded-lg border border-gray-200;
  }
  
  .card-hover {
    @apply transition-all hover:border-gray-300 hover:shadow-sm;
  }
  
  /* Message bubbles - neutral gray theme */
  .message-bubble-user {
    @apply bg-gray-900 text-white rounded-lg px-4 py-3 max-w-2xl;
  }

  .message-bubble-assistant {
    @apply bg-white border border-gray-200 text-gray-900 rounded-lg p-5;
  }

  /* Sidebar navigation */
  .sidebar-nav-item {
    @apply w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors;
  }
  
  .sidebar-nav-item-active {
    @apply bg-gray-100 text-gray-900;
  }
  
  .sidebar-nav-item-inactive {
    @apply text-gray-600 hover:bg-gray-50 hover:text-gray-900;
  }
}
```

### Mobile-First Development

#### App Layout with Sidebar
```tsx
// Main App layout with sidebar navigation
function App() {
  const [viewMode, setViewMode] = useState<'home' | 'topics' | 'search' | 'topic-detail'>('home');
  const [showCommandPalette, setShowCommandPalette] = useState(false);

  return (
    <div className="h-screen bg-[#FAFAFA] flex">
      {/* Sidebar - 240px fixed width */}
      <aside className="w-60 border-r border-gray-200 bg-white flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-gray-900 rounded-md flex items-center justify-center">
              <BookOpen className="w-4 h-4 text-white" />
            </div>
            <span className="font-semibold text-sm text-gray-900">Vachanamrut</span>
          </div>
        </div>

        <nav className="flex-1 p-3">
          {/* Navigation items */}
          <button className="sidebar-nav-item sidebar-nav-item-active">
            <Home className="w-4 h-4" />
            Home
          </button>
          {/* Quick Access section */}
        </nav>

        <div className="p-3 border-t border-gray-200">
          <button onClick={() => setShowCommandPalette(true)}>
            <Command className="w-4 h-4" />
            Quick search
            <kbd>⌘K</kbd>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Views: Home, Topics, Topic Detail, Search */}
      </div>
    </div>
  );
}
```

#### Command Palette (⌘K)
```tsx
// Keyboard shortcut for quick navigation
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      setShowCommandPalette(prev => !prev);
    }
    if (e.key === 'Escape') {
      setShowCommandPalette(false);
    }
  };
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, []);
```

### API Integration

#### Service Layer Pattern
```tsx
// services/chatService.ts
import { apiClient } from './api';

export interface ChatRequest {
  message: string;
  theme?: string;
  context?: string[];
}

export interface ChatResponse {
  response: string;
  citations: Citation[];
  relatedThemes: string[];
  timestamp: string;
}

class ChatService {
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const { data } = await apiClient.post<ChatResponse>('/chat', request);
    return data;
  }

  async getThemes(): Promise<string[]> {
    const { data } = await apiClient.get<string[]>('/themes');
    return data;
  }
}

export const chatService = new ChatService();
```

### Testing Guidelines

#### Component Testing
```tsx
// tests/components/MessageBubble.test.tsx
import { render, screen } from '@testing-library/react';
import { MessageBubble } from '@/components/chat/MessageBubble';

describe('MessageBubble', () => {
  it('renders user message correctly', () => {
    render(
      <MessageBubble
        message="Test message"
        isUser={true}
        timestamp={new Date('2023-01-01')}
      />
    );
    
    expect(screen.getByText('Test message')).toBeInTheDocument();
    expect(screen.getByText(/test message/i)).toHaveClass('bg-blue-500');
  });
});
```

## Backend Development Guidelines (FastAPI + Python)

### API Design Principles

#### RESTful Endpoints with Clear Patterns
```python
# app/api/endpoints/chat.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Send a message and receive AI response with citations."""
    try:
        response = await chat_service.process_message(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/themes", response_model=List[str])
async def get_available_themes() -> List[str]:
    """Get list of available scriptural themes."""
    return await chat_service.get_themes()
```

#### Pydantic Models for Data Validation
```python
# app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    theme: Optional[str] = None
    context: Optional[List[str]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "message": "How should I approach spiritual growth?",
                "theme": "devotion",
                "context": ["previous_conversation_id"]
            }
        }

class Citation(BaseModel):
    reference: str
    passage: str
    page_number: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    related_themes: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### Service Layer Architecture

#### Separation of Concerns
```python
# app/services/chat_service.py
from typing import List
import logging
from ..core.retrieval import DocumentRetriever
from ..core.ai_client import AIClient
from ..schemas.chat import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(
        self,
        retriever: DocumentRetriever,
        ai_client: AIClient,
        theme_service: ThemeService
    ):
        self.retriever = retriever
        self.ai_client = ai_client
        self.theme_service = theme_service

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process user message and generate AI response."""
        try:
            # Retrieve relevant documents
            relevant_docs = await self.retriever.search(
                query=request.message,
                theme_filter=request.theme
            )
            
            # Generate AI response
            ai_response = await self.ai_client.generate_response(
                message=request.message,
                context=relevant_docs,
                theme=request.theme
            )
            
            # Extract citations
            citations = self._extract_citations(relevant_docs)
            
            # Get related themes
            related_themes = await self.theme_service.get_related_themes(
                request.message
            )
            
            return ChatResponse(
                response=ai_response,
                citations=citations,
                related_themes=related_themes
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise
```

### Document Processing

#### PDF Processing and Chunking
```python
# app/core/pdf_processor.py
from typing import List, Dict
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF and create chunks with metadata."""
        doc = pymupdf.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Split text into chunks
            page_chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(page_chunks):
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        "page": page_num + 1,
                        "chunk_id": f"page_{page_num + 1}_chunk_{i}",
                        "source": "vachanamrut"
                    }
                })
        
        doc.close()
        return chunks

### Vector Database Integration

#### Embedding and Retrieval System
```python
# app/core/retrieval.py
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
from ..schemas.chat import Citation

class DocumentRetriever:
    def __init__(
        self,
        index_name: str,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(embedding_model)
        self.index = pinecone.Index(index_name)

    async def search(
        self,
        query: str,
        theme_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Search for relevant documents based on query."""
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].tolist()
        
        # Prepare filter
        filter_dict = {}
        if theme_filter:
            filter_dict["theme"] = theme_filter
        
        # Search vector database
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )
        
        # Format results
        documents = []
        for match in results.matches:
            documents.append({
                "content": match.metadata.get("content", ""),
                "score": match.score,
                "reference": match.metadata.get("reference", ""),
                "page": match.metadata.get("page"),
                "theme": match.metadata.get("theme")
            })
        
        return documents

    async def index_documents(self, documents: List[Dict]) -> None:
        """Index documents in vector database."""
        vectors = []
        for doc in documents:
            embedding = self.model.encode([doc["content"]])[0].tolist()
            vectors.append({
                "id": doc["metadata"]["chunk_id"],
                "values": embedding,
                "metadata": {
                    "content": doc["content"],
                    "page": doc["metadata"]["page"],
                    "reference": doc["metadata"].get("reference", ""),
                    "theme": doc["metadata"].get("theme", "")
                }
            })
        
        # Batch upsert to Pinecone
        self.index.upsert(vectors=vectors)
```

### AI Integration

#### LLM Client with Prompt Engineering
```python
# app/core/ai_client.py
from typing import List, Dict, Optional
import openai
from ..core.prompt_templates import CHAT_PROMPT_TEMPLATE

class AIClient:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    async def generate_response(
        self,
        message: str,
        context: List[Dict],
        theme: Optional[str] = None
    ) -> str:
        """Generate AI response with context from vachanamrut."""
        
        # Prepare context from retrieved documents
        context_text = self._format_context(context)
        
        # Build prompt
        prompt = CHAT_PROMPT_TEMPLATE.format(
            user_question=message,
            context=context_text,
            theme=theme or "general guidance"
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable guide helping users understand the Vachanamrut. Provide thoughtful, accurate responses based on the provided context."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            raise

    def _format_context(self, context: List[Dict]) -> str:
        """Format retrieved documents for prompt context."""
        formatted_context = []
        for doc in context:
            formatted_context.append(
                f"Reference: {doc.get('reference', 'Unknown')}\n"
                f"Content: {doc['content']}\n"
                f"Relevance Score: {doc.get('score', 0):.2f}\n"
            )
        return "\n---\n".join(formatted_context)
```

### Database Models and Operations

#### SQLAlchemy Models
```python
# app/models/document.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True)
    content = Column(Text, nullable=False)
    reference = Column(String, index=True)
    page_number = Column(Integer)
    theme = Column(String, index=True)
    embedding_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ThemeMapping(Base):
    __tablename__ = "theme_mappings"
    
    id = Column(Integer, primary_key=True, index=True)
    reference = Column(String, nullable=False, index=True)
    theme = Column(String, nullable=False, index=True)
    description = Column(Text)
    keywords = Column(Text)  # JSON string of related keywords
    relevance_score = Column(Float, default=1.0)
```

### Error Handling and Logging

#### Comprehensive Error Management
```python
# app/utils/exceptions.py
from fastapi import HTTPException
from typing import Optional

class VachanamrutException(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class DocumentProcessingError(VachanamrutException):
    """Raised when document processing fails."""
    pass

class AIServiceError(VachanamrutException):
    """Raised when AI service encounters an error."""
    pass

class ThemeNotFoundError(VachanamrutException):
    """Raised when requested theme is not found."""
    pass

# app/utils/logging.py
import logging
import sys
from typing import Dict, Any

def setup_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log")
        ]
    )

def log_api_call(endpoint: str, request_data: Dict[Any, Any]) -> None:
    """Log API call details for monitoring."""
    logger = logging.getLogger("api")
    logger.info(f"API call to {endpoint}", extra={
        "endpoint": endpoint,
        "request_size": len(str(request_data)),
        "timestamp": "%(asctime)s"
    })
```

### Testing Guidelines

#### Unit Testing with Pytest
```python
# tests/test_services/test_chat_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from app.services.chat_service import ChatService
from app.schemas.chat import ChatRequest, ChatResponse

@pytest.fixture
def mock_dependencies():
    return {
        "retriever": Mock(),
        "ai_client": Mock(),
        "theme_service": Mock()
    }

@pytest.mark.asyncio
async def test_process_message_success(mock_dependencies):
    # Arrange
    chat_service = ChatService(**mock_dependencies)
    request = ChatRequest(message="Test question")
    
    mock_dependencies["retriever"].search.return_value = [
        {"content": "Test content", "reference": "Test ref"}
    ]
    mock_dependencies["ai_client"].generate_response.return_value = "Test response"
    mock_dependencies["theme_service"].get_related_themes.return_value = ["devotion"]
    
    # Act
    response = await chat_service.process_message(request)
    
    # Assert
    assert isinstance(response, ChatResponse)
    assert response.response == "Test response"
    assert len(response.related_themes) == 1

# Integration testing
@pytest.mark.integration
async def test_full_chat_flow(test_client):
    response = await test_client.post("/chat/", json={
        "message": "How should I practice devotion?"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "citations" in data
```

### Performance Optimization

#### Caching and Async Operations
```python
# app/core/cache.py
from functools import wraps
from typing import Any, Callable
import redis
import json
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_response(expiration: int = 3600):
    """Decorator to cache function responses."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create cache key
            cache_key = hashlib.md5(
                f"{func.__name__}:{str(args)}:{str(kwargs)}".encode()
            ).hexdigest()
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(
                cache_key,
                expiration,
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

# Usage in service
class ChatService:
    @cache_response(expiration=1800)  # 30 minutes
    async def get_themes(self) -> List[str]:
        """Get available themes with caching."""
        return await self.theme_service.list_all_themes()
```

## Development Workflow Guidelines

### Code Quality Standards

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
```

#### TypeScript Configuration
```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### Environment Management

#### Docker Development Setup
```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/Vachanamrut
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./backend:/app
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8000

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=Vachanamrut
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

These guidelines provide a comprehensive foundation for building the Vachanamrut Study Companion with clean, maintainable, and scalable code. They emphasize separation of concerns, proper error handling, testing, and performance optimization while maintaining simplicity for rapid development with Claude Code.
```