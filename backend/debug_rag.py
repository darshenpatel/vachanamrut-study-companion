#!/usr/bin/env python3
"""
Debug the RAG service initialization and response generation
"""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.core.ai_rag_service import get_rag_service
from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService

async def debug_rag_service():
    """Debug the RAG service directly"""
    
    print("🔍 Debugging RAG Service")
    print("=" * 50)
    
    try:
        # Test RAG service initialization
        print("\n1. Testing RAG service initialization...")
        rag_service = await get_rag_service()
        print("✅ RAG service obtained")
        
        # Get system status
        print("\n2. Getting system status...")
        status = await rag_service.get_system_status()
        print(f"Status: {status}")
        
        # Test direct retrieval
        print("\n3. Testing direct retrieval...")
        retrieval_result = await rag_service.rag_retriever.retrieve_context(
            query="How should I remember God throughout the day?",
            theme_filter="devotion"
        )
        print(f"Retrieved passages: {len(retrieval_result.retrieved_passages)}")
        print(f"Context text length: {len(retrieval_result.context_text)}")
        print(f"Themes found: {retrieval_result.themes_found}")
        
        if retrieval_result.retrieved_passages:
            print("First passage:")
            doc = retrieval_result.retrieved_passages[0].document
            print(f"  Reference: {doc.reference}")
            print(f"  Content: {doc.content[:100]}...")
        
        # Test RAG response generation
        print("\n4. Testing RAG response generation...")
        response = await rag_service.generate_response(
            query="How should I remember God throughout the day?",
            theme="devotion"
        )
        print(f"Response length: {len(response.response)}")
        print(f"Citations: {len(response.citations)}")
        print(f"Model used: {response.model_used}")
        print(f"Response preview: {response.response[:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)
    
    # Test ChatService directly
    print("\n🗨️ Testing ChatService")
    print("=" * 50)
    
    try:
        chat_service = ChatService()
        
        request = ChatRequest(
            message="How should I remember God throughout the day?",
            theme="devotion"
        )
        
        print(f"Processing message: {request.message}")
        response = await chat_service.process_message(request)
        
        print(f"Response: {response.response[:200]}...")
        print(f"Citations: {len(response.citations)}")
        print(f"Related themes: {response.related_themes}")
        
        if response.citations:
            print("First citation:")
            citation = response.citations[0]
            print(f"  Reference: {citation.reference}")
            print(f"  Passage: {citation.passage[:100]}...")
        
    except Exception as e:
        print(f"❌ ChatService Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_rag_service())