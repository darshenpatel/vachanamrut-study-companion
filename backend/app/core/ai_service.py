from typing import List, Dict, Optional, Any
import logging
import json
from datetime import datetime
import httpx
import asyncio

from app.core.simple_config import settings
from app.core.prompt_templates import SPIRITUAL_GUIDANCE_PROMPT, CONTEXT_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)

class AIService:
    """
    AI service for generating spiritual guidance responses
    Claude Projects approach: Simple, focused on spiritual guidance context
    """
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY or settings.CLAUDE_API_KEY
        self.model = settings.AI_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        
    async def generate_response(
        self,
        user_question: str,
        context_passages: List[Dict], 
        theme: Optional[str] = None
    ) -> str:
        """
        Generate AI response based on retrieved Vachanamrut passages
        Uses either OpenAI or Claude API based on configuration
        """
        
        # This is the base class - use specific implementations
        return self._generate_contextual_response(user_question, context_passages, theme)
    
    def _generate_contextual_response(
        self,
        user_question: str,
        context_passages: List[Dict],
        theme: Optional[str] = None
    ) -> str:
        """
        Generate contextual response based on retrieved passages
        Claude Projects approach: Intelligent synthesis without complex AI calls
        """
        
        if not context_passages:
            return self._generate_general_guidance(user_question, theme)
        
        # Analyze the context and user question
        key_concepts = self._extract_key_concepts(user_question)
        relevant_teachings = self._synthesize_teachings(context_passages, key_concepts)
        
        # Build structured response
        theme_context = f" regarding {theme}" if theme else ""
        
        response_parts = [
            f"Based on the Vachanamrut teachings{theme_context}, here is guidance for your question:\n"
        ]
        
        if relevant_teachings:
            response_parts.append("The scriptures teach us that:\n")
            for i, teaching in enumerate(relevant_teachings[:3], 1):
                response_parts.append(f"{i}. {teaching}\n")
        
        # Add practical guidance
        practical_guidance = self._generate_practical_guidance(key_concepts, theme)
        if practical_guidance:
            response_parts.append(f"\nPractical guidance:\n{practical_guidance}")
        
        response_parts.append(
            "\nThese timeless teachings from the Vachanamrut can guide you on your spiritual journey. "
            "Remember that consistent practice and sincere devotion are key to spiritual progress."
        )
        
        return "".join(response_parts)
    
    def _extract_key_concepts(self, question: str) -> List[str]:
        """Extract key spiritual concepts from user question"""
        question_lower = question.lower()
        
        concept_keywords = {
            'devotion': ['devotion', 'bhakti', 'love', 'worship', 'devotee'],
            'faith': ['faith', 'trust', 'belief', 'confidence', 'conviction'],
            'surrender': ['surrender', 'submission', 'ego', 'humility', 'let go'],
            'service': ['service', 'seva', 'help', 'serve', 'selfless'],
            'meditation': ['meditation', 'focus', 'concentrate', 'mind', 'thoughts'],
            'knowledge': ['knowledge', 'understand', 'wisdom', 'learn', 'realize'],
            'detachment': ['detachment', 'attachment', 'worldly', 'desires', 'renounce'],
            'dharma': ['dharma', 'duty', 'righteousness', 'moral', 'right'],
            'obstacles': ['problem', 'difficulty', 'challenge', 'obstacle', 'trouble'],
            'progress': ['progress', 'growth', 'develop', 'improve', 'advance'],
            'practice': ['practice', 'daily', 'routine', 'habit', 'regular'],
            'guidance': ['guidance', 'help', 'advice', 'direction', 'path']
        }
        
        found_concepts = []
        for concept, keywords in concept_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                found_concepts.append(concept)
        
        return found_concepts
    
    def _synthesize_teachings(self, passages: List[Dict], key_concepts: List[str]) -> List[str]:
        """Synthesize teachings from passages relevant to key concepts"""
        teachings = []
        
        for passage in passages[:3]:  # Focus on top 3 passages
            content = passage.get('content', '')
            reference = passage.get('reference', '')
            
            # Extract key teaching points
            teaching = self._extract_core_teaching(content, key_concepts)
            if teaching:
                teachings.append(f"{teaching} (from {reference})")
        
        return teachings
    
    def _extract_core_teaching(self, content: str, key_concepts: List[str]) -> Optional[str]:
        """Extract core teaching from passage content"""
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        # Look for sentences with teaching indicators
        teaching_indicators = [
            'should', 'must', 'never', 'always', 'one who', 'those who',
            'key', 'important', 'essential', 'necessary', 'leads to'
        ]
        
        for sentence in sentences[:3]:  # Check first 3 sentences
            sentence_lower = sentence.lower()
            
            # Check if sentence contains teaching indicators
            if any(indicator in sentence_lower for indicator in teaching_indicators):
                # Check if relevant to key concepts
                if not key_concepts or any(concept in sentence_lower for concept in key_concepts):
                    return sentence.capitalize()
        
        # Fallback to first meaningful sentence
        if sentences:
            return sentences[0].capitalize()
        
        return None
    
    def _generate_practical_guidance(self, key_concepts: List[str], theme: Optional[str]) -> str:
        """Generate practical guidance based on concepts and theme"""
        
        guidance_map = {
            'devotion': "Cultivate love for God through daily prayer, remembrance, and seeing the divine in all beings.",
            'faith': "Strengthen your faith through study of scriptures, association with devotees, and trusting in God's plan.",
            'surrender': "Practice surrender by offering all actions to God and accepting outcomes with equanimity.",
            'service': "Engage in selfless service without expecting rewards, serving God through serving others.",
            'meditation': "Establish a regular meditation practice, focusing the mind on God's form and attributes.",
            'knowledge': "Seek spiritual knowledge through study, contemplation, and guidance from realized souls.",
            'detachment': "Gradually reduce attachment to material things while increasing attachment to God.",
            'dharma': "Follow righteous conduct in all situations, guided by scriptural principles.",
            'obstacles': "View obstacles as opportunities for spiritual growth and tests of your faith.",
            'progress': "Measure progress by internal transformation rather than external achievements.",
            'practice': "Maintain consistency in spiritual practices, even if starting with small steps.",
            'guidance': "Seek guidance from authentic spiritual teachers and study revealed scriptures."
        }
        
        if theme and theme in guidance_map:
            return guidance_map[theme]
        
        if key_concepts:
            primary_concept = key_concepts[0]
            if primary_concept in guidance_map:
                return guidance_map[primary_concept]
        
        return "Focus on the fundamental practices: regular prayer, study of scriptures, selfless service, and maintaining the company of devotees."
    
    def _generate_general_guidance(self, question: str, theme: Optional[str]) -> str:
        """Generate general spiritual guidance when no specific context is found"""
        
        theme_responses = {
            'devotion': """Devotion (bhakti) is the heart of spiritual practice. The Vachanamrut teaches that true devotion involves:

• Loving God with complete sincerity and without any selfish motive
• Seeing God as the ultimate reality behind all experiences  
• Offering all actions and thoughts to God
• Finding joy in serving God and His devotees

Cultivate devotion through consistent prayer, remembrance of God's qualities, and seeing the divine presence in all situations.""",

            'faith': """Faith is the foundation of spiritual life. According to the Vachanamrut:

• True faith means complete trust in God's wisdom and love
• Faith grows through study of scriptures and association with devotees
• Those with firm faith remain peaceful even in difficult circumstances
• Faith should be supported by understanding, not blind belief

Strengthen your faith through regular spiritual study and reflecting on God's grace in your life.""",

            'surrender': """Surrender is the ultimate spiritual practice. The teachings emphasize:

• Complete surrender means offering your will to God's will
• Surrender is not passive but involves active service and practice
• True surrender leads to divine grace and protection
• Ego is the main obstacle to surrender

Practice surrender by accepting life's circumstances as God's plan while continuing to act righteously."""
        }
        
        if theme and theme in theme_responses:
            return theme_responses[theme]
        
        # General response
        return """The Vachanamrut provides timeless guidance for spiritual seekers. The key principles include:

• Developing unwavering faith in God and following His commandments
• Practicing constant remembrance of God in all activities
• Serving God and His devotees without selfish motives  
• Following dharma and maintaining righteous conduct
• Seeking the company of true devotees and avoiding negative influences

These practices, when followed sincerely, lead to spiritual realization and eternal happiness. Remember that spiritual progress requires patience, persistence, and God's grace."""

class OpenAIService(AIService):
    """Real OpenAI API service implementation"""
    
    def __init__(self):
        super().__init__()
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
    async def generate_response(
        self,
        user_question: str,
        context_passages: List[Dict], 
        theme: Optional[str] = None
    ) -> str:
        """Generate response using OpenAI API"""
        try:
            # Build context from passages
            context_text = self._build_context_text(context_passages)
            
            # Create prompt
            prompt = self._create_spiritual_guidance_prompt(
                user_question, context_text, theme
            )
            
            # Call OpenAI API
            response = await self._call_openai_api(prompt)
            return response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fallback to contextual response
            return self._generate_contextual_response(user_question, context_passages, theme)
    
    def _build_context_text(self, context_passages: List[Dict]) -> str:
        """Build context text from retrieved passages"""
        if not context_passages:
            return "No specific passages found."
        
        context_parts = []
        for i, passage in enumerate(context_passages[:3], 1):
            content = passage.get('content', '')
            reference = passage.get('reference', '')
            similarity = passage.get('similarity_score', 0)
            
            context_parts.append(
                f"Passage {i} (from {reference}, relevance: {similarity:.3f}):\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _create_spiritual_guidance_prompt(
        self, 
        user_question: str, 
        context_text: str, 
        theme: Optional[str] = None
    ) -> str:
        """Create structured prompt for spiritual guidance"""
        theme_context = f" focusing on the theme of {theme}" if theme else ""
        
        prompt = f"""You are a wise spiritual guide helping someone understand the Vachanamrut teachings. Your role is to provide compassionate, practical spiritual guidance based on the scriptural passages provided.

CONTEXT FROM VACHANAMRUT:
{context_text}

USER QUESTION: {user_question}

Please provide a thoughtful response{theme_context} that:
1. Directly addresses their question with wisdom from the passages
2. Offers practical spiritual guidance they can apply
3. Maintains a compassionate, encouraging tone
4. References the relevant passages naturally
5. Keeps the response concise but meaningful (aim for 2-3 paragraphs)

Your response should feel like guidance from a caring spiritual teacher, not an academic analysis."""

        return prompt
    
    async def _call_openai_api(self, prompt: str) -> str:
        """Make actual API call to OpenAI"""
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a compassionate spiritual guide specializing in Vachanamrut teachings. Provide wise, practical guidance that helps people apply these ancient teachings to their modern lives."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

class ClaudeService(AIService):
    """Anthropic Claude API service implementation"""
    
    def __init__(self):
        super().__init__()
        if not settings.CLAUDE_API_KEY:
            raise ValueError("CLAUDE_API_KEY not configured")
        
    async def generate_response(
        self,
        user_question: str,
        context_passages: List[Dict], 
        theme: Optional[str] = None
    ) -> str:
        """Generate response using Claude API"""
        try:
            # Build context from passages
            context_text = self._build_context_text(context_passages)
            
            # Create prompt
            prompt = self._create_spiritual_guidance_prompt(
                user_question, context_text, theme
            )
            
            # Call Claude API
            response = await self._call_claude_api(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            # Fallback to contextual response
            return self._generate_contextual_response(user_question, context_passages, theme)
    
    def _build_context_text(self, context_passages: List[Dict]) -> str:
        """Build context text from retrieved passages"""
        if not context_passages:
            return "No specific passages found."
        
        context_parts = []
        for i, passage in enumerate(context_passages[:3], 1):
            content = passage.get('content', '')
            reference = passage.get('reference', '')
            similarity = passage.get('similarity_score', 0)
            
            context_parts.append(
                f"Passage {i} (from {reference}, relevance: {similarity:.3f}):\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _create_spiritual_guidance_prompt(
        self, 
        user_question: str, 
        context_text: str, 
        theme: Optional[str] = None
    ) -> str:
        """Create structured prompt for spiritual guidance"""
        theme_context = f" focusing on the theme of {theme}" if theme else ""
        
        prompt = f"""You are a wise spiritual guide helping someone understand the Vachanamrut teachings. Your role is to provide compassionate, practical spiritual guidance based on the scriptural passages provided.

CONTEXT FROM VACHANAMRUT:
{context_text}

USER QUESTION: {user_question}

Please provide a thoughtful response{theme_context} that:
1. Directly addresses their question with wisdom from the passages
2. Offers practical spiritual guidance they can apply
3. Maintains a compassionate, encouraging tone
4. References the relevant passages naturally
5. Keeps the response concise but meaningful (aim for 2-3 paragraphs)

Your response should feel like guidance from a caring spiritual teacher, not an academic analysis."""

        return prompt
    
    async def _call_claude_api(self, prompt: str) -> str:
        """Make actual API call to Claude"""
        headers = {
            "x-api-key": settings.CLAUDE_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-haiku-20240307",  # Use a reliable Claude model
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Claude API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result["content"][0]["text"].strip()

# Mock implementation for development
class MockAIService(AIService):
    """Mock AI service for development without API dependencies"""
    
    async def generate_response(
        self,
        user_question: str,
        context_passages: List[Dict], 
        theme: Optional[str] = None
    ) -> str:
        """Generate mock response with spiritual guidance"""
        return self._generate_contextual_response(user_question, context_passages, theme)

def get_ai_service() -> AIService:
    """Factory function for AI service - chooses best available service"""
    
    # Try OpenAI first (since it's configured as the primary model)
    if settings.OPENAI_API_KEY:
        try:
            logger.info("Using OpenAI service for AI responses")
            return OpenAIService()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI service: {e}")
    
    # Fallback to Claude if available
    if settings.CLAUDE_API_KEY:
        try:
            logger.info("Using Claude service for AI responses")
            return ClaudeService()
        except Exception as e:
            logger.warning(f"Failed to initialize Claude service: {e}")
    
    # Final fallback to mock service
    logger.info("No API keys configured, using mock AI service")
    return MockAIService()