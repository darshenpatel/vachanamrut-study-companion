#!/usr/bin/env python3
"""
Test the spiritual guidance system
"""

import asyncio
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.core.lightweight_retrieval import get_lightweight_retriever
from app.core.ai_service import get_ai_service
from app.services.chat_service import ChatService
from app.schemas.chat import ChatRequest


async def test_system():
    """Test the complete system"""

    print("🙏 Testing Vachanamrut Study Companion System")
    print("=" * 50)

    # Test retrieval system
    print("\n📚 Testing Document Retrieval System")
    retriever = get_lightweight_retriever()

    init_result = await retriever.initialize()
    print(f"✅ Initialization: {init_result['status']}")
    print(f"📖 Total documents: {init_result['total_documents']}")

    if 'theme_distribution' in init_result:
        print("\n🎯 Available themes:")
        for theme, count in init_result['theme_distribution'].items():
            print(f"   • {theme}: {count} passages")

    # Test search functionality
    print("\n🔍 Testing Search Functionality")
    test_queries = [
        "How should I practice devotion?",
        "What is the importance of faith?",
        "How to surrender to God?",
        "What is spiritual service?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = await retriever.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['reference']} (similarity: {result['similarity_score']:.3f})")
            print(f"      {result['content'][:80]}...")
            print(f"      Themes: {', '.join(result['themes'])}")

    # Test AI service
    print("\n🤖 Testing AI Service")
    ai_service = get_ai_service()

    test_questions = [
        ("How can I develop stronger faith?", "faith"),
        ("What does surrender mean in spiritual practice?", "surrender"),
        ("How do I serve God?", "service"),
    ]

    for question, theme in test_questions:
        print(f"\nQuestion: '{question}' (theme: {theme})")

        # Get context
        context = await retriever.search(question, theme_filter=theme, top_k=2)

        # Generate response
        response = await ai_service.generate_response(
            user_question=question,
            context_passages=context,
            theme=theme,
        )

        print("Response:")
        print(response[:200] + "..." if len(response) > 200 else response)

    # Test complete chat service
    print("\n💬 Testing Complete Chat Service")
    chat_service = ChatService()

    test_requests = [
        ChatRequest(message="How should I remember God throughout the day?", theme="devotion"),
        ChatRequest(message="I'm facing difficulties in life. What guidance do the scriptures offer?", theme="faith"),
        ChatRequest(message="What does it mean to be detached?", theme="detachment"),
    ]

    for request in test_requests:
        print(f"\nUser: {request.message}")
        response = await chat_service.process_message(request)

        print(f"AI: {response.response[:150]}...")
        print(f"Citations: {len(response.citations)} references")
        print(f"Related themes: {', '.join(response.related_themes)}")

        if response.citations:
            print("Top citation:")
            citation = response.citations[0]
            print(f"   📜 {citation.reference}: {citation.passage[:100]}...")

    # System status
    print("\n📊 System Status")
    status = retriever.get_system_status()
    for key, value in status.items():
        if isinstance(value, dict) and len(value) > 3:
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")

    print("\n✅ All tests completed successfully!")
    print("🚀 System is ready for spiritual guidance conversations!")


if __name__ == "__main__":
    asyncio.run(test_system())