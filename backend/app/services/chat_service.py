from typing import List, Dict, Optional, TYPE_CHECKING
import logging
from datetime import datetime

from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.core.config import settings
from app.core.factual_retrieval import get_factual_retriever, FactualRetriever
from app.core.ai_service import get_ai_service

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Avoid importing semantic retrieval (and heavy optional deps) at module import time.
    from app.core.semantic_retrieval import SemanticRetriever


class ChatService:
    """
    Main chat service orchestrating retrieval and AI response generation.
    Uses semantic search (sentence-transformers + FAISS) as primary retrieval method.
    Falls back to factual retrieval if semantic search fails.
    """
    
    def __init__(self):
        self.semantic_retriever: Optional["SemanticRetriever"] = None
        self.factual_retriever: Optional[FactualRetriever] = None
        self.ai_service = get_ai_service()
        self._initialized = False
        # Prefer semantic search when available, but keep it optional.
        self._use_semantic = True

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process user message with semantic or factual PDF retrieval"""
        logger.info(f"Processing message: {request.message[:50]}...")

        try:
            # Initialize retrievers if needed
            if not self._initialized:
                await self._ensure_initialized()

            # Try semantic search first (better quality)
            search_results = []
            if self._use_semantic and self.semantic_retriever:
                try:
                    search_results = await self.semantic_retriever.search(
                        query=request.message,
                        top_k=settings.TOP_K_RESULTS,
                        theme_filter=request.theme
                    )
                    logger.info(f"Semantic search returned {len(search_results)} results")
                except Exception as e:
                    logger.warning(f"Semantic search failed, falling back: {e}")
                    search_results = []
            
            # Fallback to factual retrieval if semantic search didn't work
            if not search_results and self.factual_retriever:
                search_results = await self.factual_retriever.search(
                    query=request.message,
                    top_k=settings.TOP_K_RESULTS
                )
                logger.info(f"Factual search returned {len(search_results)} results")

            # Generate AI response using retrieved context
            ai_response = await self.ai_service.generate_response(
                user_question=request.message,
                context_passages=search_results,
                theme=request.theme
            )

            # Convert search results to API citations
            citations: List[Citation] = []
            for result in search_results[:3]:  # Limit citations to top 3
                citations.append(
                    Citation(
                        reference=result.get("reference", "Unknown"),
                        passage=self._truncate_passage(result.get("content", "")),
                        page_number=result.get("page_number"),
                        relevance_score=result.get("similarity_score", result.get("relevance_score", 0.0)),
                    )
                )

            # Extract related themes from search results
            related_themes = self._extract_themes_from_results(search_results)

            return ChatResponse(
                response=ai_response,
                citations=citations,
                related_themes=related_themes,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return await self._generate_fallback_response(request.message, request.theme)

    async def get_themes(self) -> List[str]:
        """Get list of available spiritual themes"""
        # Return curated spiritual themes (will be enhanced in theme-system task)
        return [
            "devotion",
            "faith", 
            "surrender",
            "service",
            "knowledge",
            "detachment",
            "dharma",
            "meditation",
            "guru",
            "satsang"
        ]

    def _extract_themes_from_results(self, search_results: List[Dict]) -> List[str]:
        """Extract relevant themes from search results"""
        themes = set()
        
        # Check for themes in result metadata
        for result in search_results:
            result_themes = result.get('themes', result.get('spiritual_themes', []))
            if result_themes:
                themes.update(result_themes)
        
        # If no themes in metadata, extract from content
        if not themes:
            themes = self._extract_topics_from_content(search_results)
        
        return list(themes)[:5]

    def _extract_topics_from_content(self, search_results: List[Dict]) -> set:
        """Extract relevant topics from search results content"""
        topics = set()
        
        concept_keywords = {
            'devotion': ['devotion', 'bhakti', 'love', 'devotee'],
            'faith': ['faith', 'belief', 'trust', 'confidence'],
            'surrender': ['surrender', 'submission', 'humility'],
            'service': ['service', 'seva', 'serve'],
            'knowledge': ['knowledge', 'wisdom', 'understanding', 'jnan'],
            'meditation': ['meditation', 'dhyan', 'concentrate', 'focus'],
            'dharma': ['dharma', 'duty', 'righteousness', 'moral'],
            'guru': ['guru', 'master', 'teacher', 'sant'],
            'detachment': ['detachment', 'vairagya', 'renunciation'],
            'satsang': ['satsang', 'fellowship', 'association', 'company']
        }
        
        for result in search_results:
            content_lower = result.get('content', '').lower()
            for concept, keywords in concept_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topics.add(concept)
        
        return topics

    def _truncate_passage(self, passage: str, max_length: int = 300) -> str:
        """Truncate passage to reasonable length for citation"""
        if len(passage) <= max_length:
            return passage
        
        # Try to truncate at sentence boundary
        truncated = passage[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length // 2:
            return truncated[:last_period + 1]
        return truncated + "..."

    async def _ensure_initialized(self):
        """Initialize both retrieval systems"""
        try:
            # Initialize semantic retriever (primary) lazily so the backend can run
            # even when optional ML dependencies aren't installed.
            self._use_semantic = True
            try:
                from app.core.semantic_retrieval import get_semantic_retriever
                self.semantic_retriever = get_semantic_retriever()
                result = await self.semantic_retriever.initialize()

                if result.get('status') in ['initialized', 'loaded_from_cache']:
                    self._use_semantic = True
                    logger.info(f"Semantic retriever initialized: {result}")
                else:
                    logger.warning(f"Semantic retriever failed: {result}")
                    self._use_semantic = False
            except Exception as e:
                logger.warning(f"Semantic retriever unavailable, disabling semantic search: {e}")
                self.semantic_retriever = None
                self._use_semantic = False
            
            # Initialize factual retriever (fallback)
            self.factual_retriever = get_factual_retriever()
            await self.factual_retriever.initialize()
            
            self._initialized = True
            logger.info("Chat service initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing retrievers: {e}")
            # Try factual retriever as last resort
            try:
                self.factual_retriever = get_factual_retriever()
                await self.factual_retriever.initialize()
                self._use_semantic = False
                self._initialized = True
                logger.info("Initialized with factual retriever only")
            except Exception as e2:
                logger.error(f"All retrievers failed: {e2}")
                raise

    async def _generate_fallback_response(self, query: str, theme: Optional[str] = None) -> ChatResponse:
        """Generate fallback response when retrieval fails"""
        theme_context = f" in the context of {theme}" if theme else ""

        fallback_text = f"""I understand you're seeking spiritual guidance{theme_context}. Based on the Vachanamrut teachings, here are some fundamental principles:

• **Faith**: Maintain unwavering faith in God and His devotees
• **Devotion**: Practice constant remembrance of God in all activities  
• **Dharma**: Follow righteous conduct in all circumstances
• **Satsang**: Seek the company of true devotees
• **Detachment**: Develop detachment from worldly attachments

These timeless teachings from the Vachanamrut can guide you on your spiritual journey. Please feel free to ask more specific questions about these principles."""

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

    def get_system_status(self) -> Dict:
        """Get status of the chat service and its components"""
        status = {
            'initialized': self._initialized,
            'use_semantic': self._use_semantic
        }
        
        if self.semantic_retriever:
            status['semantic_retriever'] = self.semantic_retriever.get_system_status()
        
        if self.factual_retriever:
            status['factual_retriever'] = self.factual_retriever.get_system_status()
        
        return status


# Dependency injection
def get_chat_service() -> ChatService:
    return ChatService()
