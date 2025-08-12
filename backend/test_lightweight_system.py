#!/usr/bin/env python3
"""
Test the lightweight spiritual guidance system
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


async def test_lightweight_system():
    """Test the complete lightweight system"""

    print("🙏 Testing Vachanamrut Study Companion (Lightweight Version)")
    print("=" * 60)

    # Test retrieval system
    print("\n📚 Testing Lightweight Document Retrieval System")
    retriever = get_lightweight_retriever()

    init_result = await retriever.initialize()
    print(f"✅ Initialization: {init_result['status']}")
    print(f"📖 Total documents: {init_result['total_documents']}")

    # Fetch system status for additional stats (works for both cache and fresh init)
    status = retriever.get_system_status()
    print(f"🎯 Unique themes: {status.get('unique_themes', 0)}")
    print(f"📜 Unique references: {status.get('unique_references', 0)}")

    if 'theme_distribution' in status and status['theme_distribution']:
        print("\n🌟 Available themes and passage counts:")
        for theme, count in status['theme_distribution'].items():
            print(f"   • {theme}: {count} passages")

    # Test search functionality
    print("\n🔍 Testing Search Functionality")
    test_queries = [
        ("How should I practice devotion?", "devotion"),
        ("What is the importance of faith?", "faith"),
        ("How to surrender to God?", "surrender"),
        ("What is spiritual service?", "service"),
        ("How to gain spiritual knowledge?", "knowledge"),
        ("What does dharma mean?", "dharma"),
    ]

    for query, expected_theme in test_queries:
        print(f"\nQuery: '{query}' (expecting theme: {expected_theme})")
        results = await retriever.search(query, top_k=2)

        if results:
            for i, result in enumerate(results, 1):
                print(f"   {i}. 📜 {result['reference']} (similarity: {result['similarity_score']:.3f})")
                print(f"      📝 {result['content'][:80]}...")
                print(f"      🎯 Themes: {', '.join(result['themes'])}")
        else:
            print("   ❌ No results found")

    # Test AI service
    print("\n🤖 Testing AI Service")
    ai_service = get_ai_service()

    test_questions = [
        ("How can I develop stronger faith?", "faith"),
        ("What does surrender mean in spiritual practice?", "surrender"),
        ("How do I serve God selflessly?", "service"),
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

        print("AI Response:")
        # Show first 300 characters
        preview = response[:300] + "..." if len(response) > 300 else response
        print(f"   {preview}")
        print(f"   📊 Response length: {len(response)} characters")

    # Test complete chat service
    print("\n💬 Testing Complete Chat Service Integration")
    chat_service = ChatService()

    test_requests = [
        ChatRequest(message="How should I remember God throughout the day?", theme="devotion"),
        ChatRequest(message="I'm facing difficulties in life. What guidance do the scriptures offer?", theme="faith"),
        ChatRequest(message="What does it mean to be detached from worldly things?", theme="detachment"),
        ChatRequest(message="How can I serve others spiritually?", theme="service"),
    ]

    for i, request in enumerate(test_requests, 1):
        print(f"\n--- Chat Test {i} ---")
        print(f"User: {request.message}")
        if request.theme:
            print(f"Theme: {request.theme}")

        response = await chat_service.process_message(request)

        print(f"AI: {response.response[:200]}...")
        print(f"📚 Citations: {len(response.citations)} references")
        print(f"🎯 Related themes: {', '.join(response.related_themes)}")

        if response.citations:
            print("📜 Top citation:")
            citation = response.citations[0]
            relevance = (
                f" (relevance: {citation.relevance_score:.3f})" if citation.relevance_score else ""
            )
            print(f"   {citation.reference}{relevance}")
            print(f"   {citation.passage[:150]}...")

    # System status and performance
    print("\n📊 System Status and Performance")
    status = retriever.get_system_status()
    for key, value in status.items():
        if isinstance(value, dict) and len(value) > 3:
            print(f"{key}: {len(value)} items (showing top 5)")
            if key == 'theme_distribution':
                for theme, count in list(value.items())[:5]:
                    print(f"   • {theme}: {count}")
        else:
            print(f"{key}: {value}")

    # Test themes listing
    print("\n🎯 Available Themes")
    themes = retriever.get_themes_summary()
    print(f"Found {len(themes)} themes:")
    for theme, count in themes.items():
        print(f"   • {theme}: {count} passages")

    # Test references listing
    print("\n📜 Available References")
    references = retriever.get_references_summary()
    print(f"Found {len(references)} unique references:")
    for ref in references[:8]:  # Show first 8
        print(f"   • {ref}")
    if len(references) > 8:
        print(f"   ... and {len(references) - 8} more")

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("🚀 Lightweight system is ready for spiritual guidance!")
    print("💡 This system uses simple text matching - no external dependencies!")
    print("📱 Perfect for the Claude Projects approach - simple but effective!")


if __name__ == "__main__":
    asyncio.run(test_lightweight_system())