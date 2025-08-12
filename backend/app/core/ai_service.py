from typing import List, Dict, Optional, Any
import logging
import json
from datetime import datetime

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
        
        # For now, provide intelligent mock responses based on context
        # This can be easily upgraded to real API calls later
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
    """Factory function for AI service"""
    # For now, return mock service
    # Can be easily switched to real API service later
    return MockAIService()