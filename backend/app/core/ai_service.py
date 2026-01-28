from typing import List, Dict, Optional, Any, Union
import logging
import json
import time
from datetime import datetime, timedelta
import httpx
import asyncio
from functools import wraps
import os

from app.core.config import settings
from app.core.prompt_templates import SPIRITUAL_GUIDANCE_PROMPT, CONTEXT_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class AIServiceError(Exception):
    """Base exception for AI service errors"""
    pass

class APIConfigurationError(AIServiceError):
    """Raised when API configuration is invalid"""
    pass

class APIRateLimitError(AIServiceError):
    """Raised when API rate limit is exceeded"""
    pass

class APITimeoutError(AIServiceError):
    """Raised when API request times out"""
    pass

class APIQuotaExceededError(AIServiceError):
    """Raised when API quota is exceeded"""
    pass

# Rate limiting and cost control
class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls_per_minute: int = 50, max_tokens_per_minute: int = 40000):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.call_times = []
        self.token_usage = []
    
    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if request can be made within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        self.call_times = [t for t in self.call_times if t > minute_ago]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
        
        # Check limits
        current_calls = len(self.call_times)
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        
        return (current_calls < self.max_calls_per_minute and 
                current_tokens + estimated_tokens < self.max_tokens_per_minute)
    
    def record_request(self, tokens_used: int = 1000):
        """Record a successful request"""
        now = datetime.now()
        self.call_times.append(now)
        self.token_usage.append((now, tokens_used))

# Global rate limiter instance
_rate_limiter = RateLimiter()

def with_retry(max_retries: int = 3, backoff_multiplier: float = 2.0):
    """Decorator for retrying API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_multiplier ** attempt
                        logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise APITimeoutError(f"API request failed after {max_retries} retries: {e}")
                except Exception as e:
                    # Don't retry for other types of errors
                    raise e
            
            raise last_exception
        return wrapper
    return decorator

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
        self.rate_limiter = _rate_limiter
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate AI service configuration"""
        if not self.max_tokens or self.max_tokens <= 0:
            logger.warning("Invalid max_tokens, using default 500")
            self.max_tokens = 500
        
        if not (0.0 <= self.temperature <= 2.0):
            logger.warning("Invalid temperature, using default 0.7")
            self.temperature = 0.7
        
        if not self.model:
            logger.warning("No model specified, using default gpt-3.5-turbo")
            self.model = "gpt-3.5-turbo"
        
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
        try:
            # This is the base class - use specific implementations
            return self._generate_contextual_response(user_question, context_passages, theme)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(user_question, theme)
    
    def _generate_fallback_response(self, user_question: str, theme: Optional[str] = None) -> str:
        """Generate a safe fallback response when AI services fail"""
        theme_context = f" regarding {theme}" if theme else ""
        
        return f"""I understand you're seeking spiritual guidance{theme_context}. While I'm currently experiencing technical difficulties accessing the full AI capabilities, I can share these fundamental principles from the Vachanamrut:

• Maintain unwavering faith in God and follow His commandments
• Practice constant remembrance of God in all activities
• Serve God and His devotees with selfless love
• Follow dharma and maintain righteous conduct in all situations
• Seek the company of true devotees and avoid negative influences

These timeless teachings can guide you on your spiritual journey. Please try your question again in a moment, or feel free to ask about specific spiritual practices."""
    
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
            raise APIConfigurationError("OPENAI_API_KEY not configured")
        
        # Validate API key format
        if not settings.OPENAI_API_KEY.startswith('sk-'):
            raise APIConfigurationError("Invalid OpenAI API key format")
        
        logger.info("OpenAI service initialized successfully")
        
    async def generate_response(
        self,
        user_question: str,
        context_passages: List[Dict], 
        theme: Optional[str] = None
    ) -> str:
        """Generate response using OpenAI API"""
        try:
            # Check rate limits
            estimated_tokens = len(user_question) * 2 + self.max_tokens  # Rough estimate
            if not self.rate_limiter.can_make_request(estimated_tokens):
                logger.warning("Rate limit exceeded, using fallback response")
                return self._generate_fallback_response(user_question, theme)
            
            # Build context from passages
            context_text = self._build_context_text(context_passages)
            
            # Create prompt
            prompt = self._create_spiritual_guidance_prompt(
                user_question, context_text, theme
            )
            
            # Call OpenAI API with retry logic
            response = await self._call_openai_api(prompt)
            
            # Record successful request
            self.rate_limiter.record_request(estimated_tokens)
            
            return response
            
        except APIRateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {e}")
            return self._generate_fallback_response(user_question, theme)
        except APIQuotaExceededError as e:
            logger.error(f"OpenAI quota exceeded: {e}")
            return self._generate_fallback_response(user_question, theme)
        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {e}")
            return self._generate_fallback_response(user_question, theme)
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
    
    @with_retry(max_retries=3)
    async def _call_openai_api(self, prompt: str) -> str:
        """Make actual API call to OpenAI with error handling"""
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
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                logger.debug(f"Making OpenAI API request with model: {self.model}")
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 429:
                    retry_after = response.headers.get('retry-after', '60')
                    raise APIRateLimitError(f"Rate limit exceeded, retry after {retry_after} seconds")
                elif response.status_code == 402:
                    raise APIQuotaExceededError("OpenAI quota exceeded")
                elif response.status_code >= 500:
                    raise httpx.HTTPStatusError(f"OpenAI server error: {response.status_code}", request=response.request, response=response)
                elif response.status_code != 200:
                    error_text = response.text
                    raise Exception(f"OpenAI API error: {response.status_code} - {error_text}")
                
                result = response.json()
                
                # Validate response structure
                if 'choices' not in result or not result['choices']:
                    raise Exception("Invalid response structure from OpenAI API")
                
                content = result["choices"][0]["message"]["content"]
                if not content:
                    raise Exception("Empty response from OpenAI API")
                
                logger.debug(f"OpenAI API request successful, response length: {len(content)}")
                return content.strip()
                
        except httpx.TimeoutException:
            raise APITimeoutError("OpenAI API request timed out")
        except httpx.ConnectError as e:
            raise APITimeoutError(f"Failed to connect to OpenAI API: {e}")
        except Exception as e:
            if isinstance(e, (APIRateLimitError, APIQuotaExceededError, APITimeoutError)):
                raise
            logger.error(f"Unexpected error calling OpenAI API: {e}")
            raise

class ClaudeService(AIService):
    """Anthropic Claude API service implementation"""
    
    def __init__(self):
        super().__init__()
        if not settings.CLAUDE_API_KEY:
            raise APIConfigurationError("CLAUDE_API_KEY not configured")
        
        # Validate API key format (Claude keys usually start with 'sk-ant-')
        if not settings.CLAUDE_API_KEY.startswith('sk-ant-'):
            logger.warning("Claude API key doesn't match expected format")
        
        logger.info("Claude service initialized successfully")
        
    async def generate_response(
        self,
        user_question: str,
        context_passages: List[Dict], 
        theme: Optional[str] = None
    ) -> str:
        """Generate response using Claude API"""
        try:
            # Check rate limits
            estimated_tokens = len(user_question) * 2 + self.max_tokens  # Rough estimate
            if not self.rate_limiter.can_make_request(estimated_tokens):
                logger.warning("Rate limit exceeded, using fallback response")
                return self._generate_fallback_response(user_question, theme)
            
            # Build context from passages
            context_text = self._build_context_text(context_passages)
            
            # Create prompt
            prompt = self._create_spiritual_guidance_prompt(
                user_question, context_text, theme
            )
            
            # Call Claude API with retry logic
            response = await self._call_claude_api(prompt)
            
            # Record successful request
            self.rate_limiter.record_request(estimated_tokens)
            
            return response
            
        except APIRateLimitError as e:
            logger.warning(f"Claude rate limit exceeded: {e}")
            return self._generate_fallback_response(user_question, theme)
        except APIQuotaExceededError as e:
            logger.error(f"Claude quota exceeded: {e}")
            return self._generate_fallback_response(user_question, theme)
        except APITimeoutError as e:
            logger.error(f"Claude API timeout: {e}")
            return self._generate_fallback_response(user_question, theme)
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
    
    @with_retry(max_retries=3)
    async def _call_claude_api(self, prompt: str) -> str:
        """Make actual API call to Claude with error handling"""
        headers = {
            "x-api-key": settings.CLAUDE_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Use the appropriate Claude model based on configuration
        claude_model = "claude-3-haiku-20240307"  # Default to cost-effective model
        if self.model and "claude" in self.model.lower():
            claude_model = self.model
        
        payload = {
            "model": claude_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": "You are a compassionate spiritual guide specializing in Vachanamrut teachings. Provide wise, practical guidance that helps people apply these ancient teachings to their modern lives.",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                logger.debug(f"Making Claude API request with model: {claude_model}")
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 429:
                    retry_after = response.headers.get('retry-after', '60')
                    raise APIRateLimitError(f"Rate limit exceeded, retry after {retry_after} seconds")
                elif response.status_code == 402:
                    raise APIQuotaExceededError("Claude quota exceeded")
                elif response.status_code >= 500:
                    raise httpx.HTTPStatusError(f"Claude server error: {response.status_code}", request=response.request, response=response)
                elif response.status_code != 200:
                    error_text = response.text
                    raise Exception(f"Claude API error: {response.status_code} - {error_text}")
                
                result = response.json()
                
                # Validate response structure
                if 'content' not in result or not result['content']:
                    raise Exception("Invalid response structure from Claude API")
                
                content = result["content"][0]["text"]
                if not content:
                    raise Exception("Empty response from Claude API")
                
                logger.debug(f"Claude API request successful, response length: {len(content)}")
                return content.strip()
                
        except httpx.TimeoutException:
            raise APITimeoutError("Claude API request timed out")
        except httpx.ConnectError as e:
            raise APITimeoutError(f"Failed to connect to Claude API: {e}")
        except Exception as e:
            if isinstance(e, (APIRateLimitError, APIQuotaExceededError, APITimeoutError)):
                raise
            logger.error(f"Unexpected error calling Claude API: {e}")
            raise

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
    
    # Validate environment variables first
    _validate_environment()
    
    # Try OpenAI first (since it's configured as the primary model)
    if settings.OPENAI_API_KEY:
        try:
            logger.info("Initializing OpenAI service for AI responses")
            service = OpenAIService()
            logger.info(f"OpenAI service initialized successfully with model: {service.model}")
            return service
        except APIConfigurationError as e:
            logger.error(f"OpenAI configuration error: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI service: {e}")
    
    # Fallback to Claude if available
    if settings.CLAUDE_API_KEY:
        try:
            logger.info("Initializing Claude service for AI responses")
            service = ClaudeService()
            logger.info(f"Claude service initialized successfully")
            return service
        except APIConfigurationError as e:
            logger.error(f"Claude configuration error: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Claude service: {e}")
    
    # Final fallback to mock service
    logger.warning("No valid API keys configured, using mock AI service")
    return MockAIService()

def _validate_environment():
    """Validate environment variables and configuration"""
    # Check if any AI service is configured
    if not settings.OPENAI_API_KEY and not settings.CLAUDE_API_KEY:
        logger.warning(
            "No AI API keys configured. Set OPENAI_API_KEY or CLAUDE_API_KEY environment variables for full functionality."
        )
    
    # Validate model configuration
    if settings.AI_MODEL and settings.OPENAI_API_KEY:
        if not any(model in settings.AI_MODEL for model in ['gpt-3.5', 'gpt-4', 'gpt']):
            logger.warning(f"AI_MODEL '{settings.AI_MODEL}' may not be compatible with OpenAI API")
    
    # Check for development vs production settings
    if settings.ENVIRONMENT == "production":
        if settings.DEBUG:
            logger.warning("Debug mode is enabled in production environment")
        if not settings.OPENAI_API_KEY and not settings.CLAUDE_API_KEY:
            logger.error("No AI API keys configured in production environment")

def get_service_status() -> Dict[str, Any]:
    """Get status of AI services for monitoring"""
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
        "rate_limiter": {
            "calls_per_minute": len(_rate_limiter.call_times),
            "tokens_per_minute": sum(tokens for _, tokens in _rate_limiter.token_usage)
        }
    }
    
    # Check OpenAI service
    if settings.OPENAI_API_KEY:
        try:
            OpenAIService()  # Just test initialization
            status["services"]["openai"] = {"status": "available", "model": settings.AI_MODEL}
        except Exception as e:
            status["services"]["openai"] = {"status": "error", "error": str(e)}
    else:
        status["services"]["openai"] = {"status": "not_configured"}
    
    # Check Claude service
    if settings.CLAUDE_API_KEY:
        try:
            ClaudeService()  # Just test initialization
            status["services"]["claude"] = {"status": "available"}
        except Exception as e:
            status["services"]["claude"] = {"status": "error", "error": str(e)}
    else:
        status["services"]["claude"] = {"status": "not_configured"}
    
    return status