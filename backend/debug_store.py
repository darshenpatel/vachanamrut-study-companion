#!/usr/bin/env python3
"""
Debug the embeddings store to see what's actually stored
"""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.core.rag_retrieval import get_rag_retriever

async def debug_store():
    """Debug what's actually in the embeddings store"""
    
    print("🔍 Debugging Embeddings Store Content")
    print("=" * 60)
    
    retriever = await get_rag_retriever()
    
    # Get first few documents to examine their structure
    documents = retriever.embeddings_store.documents[:10]
    
    print(f"Total documents in store: {len(retriever.embeddings_store.documents)}")
    print(f"First 10 documents:")
    
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f"  Reference: {doc.reference}")
        print(f"  Page: {doc.page_number}")
        print(f"  Themes: {doc.themes}")
        print(f"  Content: {doc.content[:100]}...")
        
    # Check if there are any documents with themes
    docs_with_themes = [doc for doc in retriever.embeddings_store.documents if doc.themes]
    print(f"\nDocuments with themes: {len(docs_with_themes)}")
    
    if docs_with_themes:
        print("Sample documents with themes:")
        for doc in docs_with_themes[:3]:
            print(f"  {doc.reference}: {doc.themes}")
    
    # Check what get_themes returns
    themes = retriever.embeddings_store.get_themes()
    print(f"\nAll unique themes: {themes}")
    
    # Check stats
    stats = retriever.embeddings_store.get_stats()
    print(f"\nStore stats: {stats}")

if __name__ == "__main__":
    asyncio.run(debug_store())