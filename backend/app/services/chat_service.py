from typing import List, Dict, Optional
import logging
from datetime import datetime

from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.core.simple_config import settings
from app.core.lightweight_retrieval import get_lightweight_retriever, LightweightRetriever
from app.core.ai_service import get_ai_service

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.retriever: Optional[LightweightRetriever] = None
        self.ai_service = get_ai_service()
        self._initialized = False

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process user message with lightweight system"""
        logger.info(f"Processing message: {request.message}")

        try:
            # Initialize retriever if needed
            if not self._initialized:
                await self._ensure_initialized()

            # Search for relevant passages
            search_results = await self.retriever.search(
                query=request.message,
                theme_filter=request.theme,
                top_k=3
            )

            # Generate AI response using retrieved context
            ai_response = await self.ai_service.generate_response(
                user_question=request.message,
                context_passages=search_results,
                theme=request.theme
            )

            # Convert search results to API citations
            citations: List[Citation] = []
            for result in search_results:
                citations.append(
                    Citation(
                        reference=result["reference"],
                        passage=result["content"],
                        page_number=result.get("page_number"),
                        relevance_score=result.get("similarity_score", 0.0),
                    )
                )

            # Extract themes from search results
            related_themes = []
            for result in search_results:
                if result.get("themes"):
                    related_themes.extend(result["themes"])
            
            # Remove duplicates and limit
            related_themes = list(set(related_themes))[:5]

            return ChatResponse(
                response=ai_response,
                citations=citations,
                related_themes=related_themes,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Fallback to simple response
            return await self._generate_fallback_response(request.message, request.theme)

    async def get_themes(self) -> List[str]:
        """Get list of available themes from lightweight retriever"""
        try:
            if not self._initialized:
                await self._ensure_initialized()
            
            themes_summary = self.retriever.get_themes_summary()
            themes = list(themes_summary.keys()) if themes_summary else []
            return themes if themes else ["devotion", "faith", "surrender", "service", "knowledge"]
        except Exception as e:
            logger.error(f"Error getting themes: {e}")
            return ["devotion", "faith", "surrender", "service", "knowledge"]

    async def _ensure_initialized(self):
        """Ensure lightweight retriever is initialized"""
        try:
            self.retriever = get_lightweight_retriever()
            await self.retriever.initialize()
            self._initialized = True
            logger.info("Lightweight retriever initialized successfully")
        except Exception as e:
            logger.error(f"Could not initialize lightweight retriever: {e}")
            raise

    async def _generate_fallback_response(self, query: str, theme: Optional[str] = None) -> ChatResponse:
        """Generate fallback response when RAG system fails"""
        theme_context = f" in the context of {theme}" if theme else ""

        fallback_text = f"""I understand you're seeking spiritual guidance{theme_context}. While I'm currently processing the complete Vachanamrut teachings, I can share these fundamental principles:

• Maintaining unwavering faith in God and His devotees
• Practicing constant remembrance of God in all activities  
• Following dharma and righteous conduct
• Seeking the company of true devotees
• Developing detachment from worldly matters

These timeless teachings from the Vachanamrut can guide you on your spiritual journey. Please feel free to ask more specific questions about these principles."""

        # Create basic citation
        citations = [
            Citation(
                reference="General Teaching",
                passage="The Vachanamrut emphasizes faith, devotion, and righteous conduct as foundations of spiritual life.",
                page_number=None,
                relevance_score=0.5,
            )
        ]

        return ChatResponse(
            response=fallback_text,
            citations=citations,
            related_themes=[theme] if theme else ["faith", "devotion", "dharma"],
            timestamp=datetime.utcnow(),
        )



# Dependency injection
def get_chat_service() -> ChatService:
    return ChatService()