#!/usr/bin/env python3
"""
Regenerate the embeddings store with proper theme detection
"""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.core.rag_retrieval import VachanamrutRAGRetriever

async def regenerate_store():
    """Regenerate the embeddings store from scratch"""
    
    print("🔄 Regenerating Embeddings Store")
    print("=" * 50)
    
    # Create a fresh RAG retriever (will not load existing store)
    retriever = VachanamrutRAGRetriever()
    
    # Initialize - this will process PDFs since no store exists
    await retriever.initialize()
    
    print("\n✅ Store regenerated!")
    
    # Check the results
    stats = retriever.embeddings_store.get_stats()
    print(f"Store stats: {stats}")
    
    # Check first few documents
    documents = retriever.embeddings_store.documents[:5]
    print(f"\nFirst 5 documents:")
    for i, doc in enumerate(documents):
        print(f"  {i+1}. {doc.reference} (themes: {doc.themes})")
        print(f"     Content: {doc.content[:80]}...")
    
    # Test search
    print("\n🔍 Testing search...")
    result = await retriever.retrieve_context("How should I practice devotion?")
    print(f"Retrieved {len(result.retrieved_passages)} passages")
    if result.retrieved_passages:
        doc = result.retrieved_passages[0].document
        print(f"First result: {doc.reference} (themes: {doc.themes})")

if __name__ == "__main__":
    asyncio.run(regenerate_store())