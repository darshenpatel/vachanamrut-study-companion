"""
AI Service with RAG Integration

This module handles AI response generation using retrieved context from the Vachanamrut knowledge base.
Supports both OpenAI and Claude APIs with intelligent fallbacks.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import json

import openai
from app.core.config import settings
from app.core.rag_retrieval import VachanamrutRAGRetriever, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from the RAG-enhanced AI system"""
    response: str
    citations: List[Dict[str, Any]]
    related_themes: List[str]
    context_used: str
    confidence_score: Optional[float] = None
    model_used: Optional[str] = None

class VachanamrutRAGService:
    """
    AI service that combines retrieval with generation for Vachanamrut guidance.
    Similar to Claude Projects but with full PDF knowledge base.
    """
    
    def __init__(self):
        """Initialize the RAG service"""
        self.rag_retriever = VachanamrutRAGRetriever()
        
        # Initialize AI clients
        self.openai_client = None
        if settings.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized")
        
        # TODO: Add Claude client when available
        self.claude_client = None
        
        # System prompt for spiritual guidance
        self.system_prompt = self._create_system_prompt()
        
        logger.info("RAG service initialized")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for spiritual guidance"""
        return """You are a knowledgeable spiritual guide specializing in the Vachanamrut teachings. Your role is to provide thoughtful, authentic guidance based on the sacred texts.

Guidelines for responses:
1. Base your answers primarily on the provided context from the Vachanamrut
2. Maintain a respectful, devotional tone appropriate for spiritual discourse
3. Cite specific passages when relevant (format: "from [Reference]")
4. If the context doesn't fully answer the question, acknowledge this honestly
5. Provide practical spiritual guidance that applies the teachings to modern life
6. Never make up quotes or references not found in the provided context
7. Encourage sincere spiritual practice and devotion

Your responses should be:
- Authentic to the Vachanamrut teachings
- Practically applicable to spiritual seekers
- Respectful of the sacred nature of the texts
- Clear and accessible to modern practitioners

Context from Vachanamrut:
{context}

Question: {question}

Provide a thoughtful response that draws from the context while offering practical spiritual guidance."""
    
    async def generate_response(
        self,
        query: str,
        theme: Optional[str] = None
    ) -> RAGResponse:
        """
        Generate an AI response using RAG (Retrieval Augmented Generation).
        
        Args:
            query: User's question
            theme: Optional theme filter
            
        Returns:
            RAG response with citations and metadata
        """
        logger.info(f"Generating RAG response for query: '{query[:50]}...'")
        
        try:
            # 1. Retrieve relevant context
            retrieval_result = await self.rag_retriever.retrieve_context(
                query=query,
                theme_filter=theme
            )
            
            if not retrieval_result.retrieved_passages:
                return await self._generate_fallback_response(query, theme)
            
            # 2. Generate AI response using retrieved context
            ai_response = await self._call_ai_model(query, retrieval_result)
            
            # 3. Create citations from retrieved passages
            citations = self._create_citations(retrieval_result.retrieved_passages)
            
            # 4. Build final response
            rag_response = RAGResponse(
                response=ai_response,
                citations=citations,
                related_themes=retrieval_result.themes_found,
                context_used=retrieval_result.context_text,
                model_used=self._get_model_name()
            )
            
            logger.info(f"Generated response with {len(citations)} citations")
            return rag_response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return await self._generate_fallback_response(query, theme)
    
    async def _call_ai_model(self, query: str, retrieval_result: RetrievalResult) -> str:
        """Call the AI model to generate response"""
        
        # Prepare the prompt with context
        formatted_prompt = self.system_prompt.format(
            context=retrieval_result.context_text,
            question=query
        )
        
        if self.openai_client:
            return await self._call_openai(formatted_prompt)
        else:
            return await self._generate_rule_based_response(query, retrieval_result)
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API for response generation"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=settings.AI_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _generate_rule_based_response(
        self, 
        query: str, 
        retrieval_result: RetrievalResult
    ) -> str:
        """Generate response using rule-based approach (fallback)"""
        logger.info("Using rule-based response generation")
        
        # Create a structured response based on retrieved passages
        response_parts = [
            "Based on the Vachanamrut teachings, here is guidance for your question:",
            ""
        ]
        
        # Add relevant teachings
        for i, result in enumerate(retrieval_result.retrieved_passages[:3], 1):
            doc = result.document
            response_parts.append(f"{i}. {doc.content[:200]}... (from {doc.reference})")
            response_parts.append("")
        
        # Add practical guidance
        response_parts.extend([
            "Practical guidance:",
            "Cultivate these teachings through daily spiritual practice, devotion, and remembrance of God.",
            "",
            "These timeless teachings from the Vachanamrut can guide you on your spiritual journey.",
            "Remember that consistent practice and sincere devotion are key to spiritual progress."
        ])
        
        return "\n".join(response_parts)
    
    def _create_citations(self, search_results: List) -> List[Dict[str, Any]]:
        """Create citation objects from search results"""
        citations = []
        
        for result in search_results:
            doc = result.document
            citation = {
                "reference": doc.reference,
                "passage": doc.content,
                "page_number": doc.page_number,
                "relevance_score": result.similarity_score,
                "themes": doc.themes
            }
            citations.append(citation)
        
        return citations
    
    def _get_model_name(self) -> str:
        """Get the name of the model being used"""
        if self.openai_client:
            return settings.AI_MODEL
        else:
            return "rule_based"
    
    async def _generate_fallback_response(
        self, 
        query: str, 
        theme: Optional[str] = None
    ) -> RAGResponse:
        """Generate fallback response when retrieval fails"""
        logger.warning("Generating fallback response")
        
        fallback_text = f"""I apologize, but I couldn't find specific guidance from the Vachanamrut teachings for your question about "{query[:100]}...". 

However, I encourage you to:
1. Continue studying the Vachanamrut texts directly
2. Seek guidance from learned devotees and spiritual teachers
3. Maintain regular spiritual practices like prayer and meditation
4. Apply the general principles of devotion, faith, and surrender in your spiritual journey

The Vachanamrut contains profound wisdom for spiritual seekers. Please consider exploring the teachings more broadly or rephrasing your question."""
        
        return RAGResponse(
            response=fallback_text,
            citations=[],
            related_themes=[theme] if theme else [],
            context_used="",
            model_used="fallback"
        )
    
    async def get_themes(self) -> List[str]:
        """Get available themes from the knowledge base"""
        return self.rag_retriever.get_available_themes()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status including RAG components"""
        rag_status = self.rag_retriever.get_system_status()
        
        ai_status = {
            "openai_available": self.openai_client is not None,
            "claude_available": self.claude_client is not None,
            "model_name": self._get_model_name(),
            "has_api_key": bool(settings.OPENAI_API_KEY or settings.CLAUDE_API_KEY)
        }
        
        return {
            "rag_system": rag_status,
            "ai_system": ai_status,
            "status": "ready" if rag_status.get("is_initialized") else "initializing"
        }

# Global service instance
rag_service = VachanamrutRAGService()

async def get_rag_service() -> VachanamrutRAGService:
    """Get the global RAG service instance"""
    # Ensure RAG retriever is initialized
    await rag_service.rag_retriever.initialize()
    return rag_service