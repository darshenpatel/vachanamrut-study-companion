"""
Tests for AI service functionality
"""

import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
import httpx

from app.core.ai_service import (
    AIService, OpenAIService, ClaudeService, MockAIService,
    get_ai_service, APIConfigurationError, APIRateLimitError, 
    APITimeoutError, APIQuotaExceededError, RateLimiter
)


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.unit
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        limiter = RateLimiter(max_calls_per_minute=60, max_tokens_per_minute=40000)
        
        assert limiter.max_calls_per_minute == 60
        assert limiter.max_tokens_per_minute == 40000
        assert len(limiter.call_times) == 0
        assert len(limiter.token_usage) == 0
    
    @pytest.mark.unit
    def test_rate_limiter_allows_initial_requests(self):
        """Test that initial requests are allowed"""
        limiter = RateLimiter(max_calls_per_minute=10, max_tokens_per_minute=10000)
        
        assert limiter.can_make_request(1000)
        limiter.record_request(1000)
        
        assert limiter.can_make_request(1000)
    
    @pytest.mark.unit
    def test_rate_limiter_blocks_excessive_requests(self):
        """Test that excessive requests are blocked"""
        limiter = RateLimiter(max_calls_per_minute=2, max_tokens_per_minute=1000)
        
        # First two requests should be allowed
        assert limiter.can_make_request(400)
        limiter.record_request(400)
        
        assert limiter.can_make_request(400)
        limiter.record_request(400)
        
        # Third request should exceed call limit
        assert not limiter.can_make_request(400)
    
    @pytest.mark.unit
    def test_rate_limiter_token_limits(self):
        """Test token-based rate limiting"""
        limiter = RateLimiter(max_calls_per_minute=10, max_tokens_per_minute=1000)
        
        # Request that would exceed token limit
        assert not limiter.can_make_request(1500)
        
        # Smaller request should be allowed
        assert limiter.can_make_request(500)


class TestAIService:
    """Test base AI service functionality"""
    
    @pytest.mark.unit
    def test_ai_service_initialization(self):
        """Test AI service initialization"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            mock_settings.CLAUDE_API_KEY = None
            mock_settings.AI_MODEL = "gpt-3.5-turbo"
            mock_settings.MAX_TOKENS = 500
            mock_settings.TEMPERATURE = 0.7
            
            service = AIService()
            
            assert service.model == "gpt-3.5-turbo"
            assert service.max_tokens == 500
            assert service.temperature == 0.7
    
    @pytest.mark.unit
    def test_configuration_validation(self):
        """Test configuration validation"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            mock_settings.CLAUDE_API_KEY = None
            mock_settings.AI_MODEL = ""
            mock_settings.MAX_TOKENS = -1
            mock_settings.TEMPERATURE = 5.0
            
            service = AIService()
            
            # Should use defaults for invalid values
            assert service.max_tokens == 500
            assert service.temperature == 0.7
            assert service.model == "gpt-3.5-turbo"
    
    @pytest.mark.unit
    async def test_generate_response_with_fallback(self, sample_passages):
        """Test response generation with fallback"""
        service = AIService()
        
        response = await service.generate_response(
            user_question="How to practice devotion?",
            context_passages=[{'content': p.content, 'reference': p.reference} for p in sample_passages[:2]],
            theme="devotion"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "devotion" in response.lower() or "spiritual" in response.lower()
    
    @pytest.mark.unit
    def test_extract_key_concepts(self):
        """Test key concept extraction"""
        service = AIService()
        
        test_cases = [
            ("How to practice devotion?", ["devotion", "practice"]),
            ("I need faith and guidance", ["faith", "guidance"]), 
            ("Help with meditation and focus", ["meditation"]),
            ("What is surrender to God?", ["surrender"]),
        ]
        
        for question, expected_concepts in test_cases:
            concepts = service._extract_key_concepts(question)
            for concept in expected_concepts:
                assert concept in concepts
    
    @pytest.mark.unit
    def test_generate_practical_guidance(self):
        """Test practical guidance generation"""
        service = AIService()
        
        guidance = service._generate_practical_guidance(["devotion"], "devotion")
        
        assert isinstance(guidance, str)
        assert len(guidance) > 0
        assert "devotion" in guidance.lower() or "god" in guidance.lower()


class TestMockAIService:
    """Test mock AI service"""
    
    @pytest.mark.unit
    async def test_mock_service_response(self, sample_passages):
        """Test mock AI service generates reasonable responses"""
        service = MockAIService()
        
        response = await service.generate_response(
            user_question="How should I practice devotion?",
            context_passages=[{'content': p.content, 'reference': p.reference} for p in sample_passages],
            theme="devotion"
        )
        
        assert isinstance(response, str)
        assert len(response) > 50  # Should be substantial
        assert "devotion" in response.lower()


class TestOpenAIService:
    """Test OpenAI service functionality"""
    
    @pytest.mark.unit
    def test_openai_service_initialization_success(self):
        """Test successful OpenAI service initialization"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test123456789"
            mock_settings.AI_MODEL = "gpt-3.5-turbo"
            mock_settings.MAX_TOKENS = 500
            mock_settings.TEMPERATURE = 0.7
            
            service = OpenAIService()
            assert service.model == "gpt-3.5-turbo"
    
    @pytest.mark.unit
    def test_openai_service_initialization_failure(self):
        """Test OpenAI service initialization with invalid config"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            
            with pytest.raises(APIConfigurationError):
                OpenAIService()
        
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "invalid-key-format"
            
            with pytest.raises(APIConfigurationError):
                OpenAIService()
    
    @pytest.mark.unit
    def test_build_context_text(self):
        """Test context text building"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test123456789"
            
            service = OpenAIService()
            
            passages = [
                {'content': 'Test content 1', 'reference': 'Test-1', 'similarity_score': 0.9},
                {'content': 'Test content 2', 'reference': 'Test-2', 'similarity_score': 0.8}
            ]
            
            context = service._build_context_text(passages)
            
            assert "Test content 1" in context
            assert "Test content 2" in context
            assert "Test-1" in context
            assert "0.900" in context
    
    @pytest.mark.ai
    async def test_openai_api_call_success(self, mock_openai_response):
        """Test successful OpenAI API call"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test123456789"
            
            service = OpenAIService()
            
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_openai_response
                
                mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
                mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
                mock_client.return_value.post = AsyncMock(return_value=mock_response)
                
                result = await service._call_openai_api("Test prompt")
                
                assert isinstance(result, str)
                assert "devotion" in result.lower()
    
    @pytest.mark.ai
    async def test_openai_rate_limit_handling(self):
        """Test OpenAI rate limit error handling"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test123456789"
            
            service = OpenAIService()
            
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.headers = {'retry-after': '60'}
                
                mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
                mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
                mock_client.return_value.post = AsyncMock(return_value=mock_response)
                
                with pytest.raises(APIRateLimitError):
                    await service._call_openai_api("Test prompt")
    
    @pytest.mark.ai
    async def test_openai_timeout_handling(self):
        """Test OpenAI timeout error handling"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test123456789"
            
            service = OpenAIService()
            
            with patch('httpx.AsyncClient') as mock_client:
                mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
                mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
                mock_client.return_value.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
                
                with pytest.raises(APITimeoutError):
                    await service._call_openai_api("Test prompt")


class TestClaudeService:
    """Test Claude service functionality"""
    
    @pytest.mark.unit
    def test_claude_service_initialization(self):
        """Test Claude service initialization"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.CLAUDE_API_KEY = "sk-ant-test123456789"
            mock_settings.AI_MODEL = "claude-3-haiku-20240307"
            
            service = ClaudeService()
            assert service.model == "claude-3-haiku-20240307"
    
    @pytest.mark.unit
    def test_claude_service_initialization_failure(self):
        """Test Claude service initialization failure"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.CLAUDE_API_KEY = None
            
            with pytest.raises(APIConfigurationError):
                ClaudeService()
    
    @pytest.mark.ai
    async def test_claude_api_call_success(self, mock_claude_response):
        """Test successful Claude API call"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.CLAUDE_API_KEY = "sk-ant-test123456789"
            
            service = ClaudeService()
            
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_claude_response
                
                mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
                mock_client.return_value.__aexit__ = AsyncMock(return_None=None)
                mock_client.return_value.post = AsyncMock(return_value=mock_response)
                
                result = await service._call_claude_api("Test prompt")
                
                assert isinstance(result, str)
                assert "surrender" in result.lower()


class TestAIServiceFactory:
    """Test AI service factory function"""
    
    @pytest.mark.unit
    def test_get_ai_service_openai_priority(self):
        """Test that OpenAI is preferred when both APIs available"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test123456789"
            mock_settings.CLAUDE_API_KEY = "sk-ant-test123456789"
            
            service = get_ai_service()
            
            assert isinstance(service, OpenAIService)
    
    @pytest.mark.unit
    def test_get_ai_service_claude_fallback(self):
        """Test Claude fallback when OpenAI unavailable"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            mock_settings.CLAUDE_API_KEY = "sk-ant-test123456789"
            
            service = get_ai_service()
            
            assert isinstance(service, ClaudeService)
    
    @pytest.mark.unit
    def test_get_ai_service_mock_fallback(self):
        """Test mock service fallback when no APIs available"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            mock_settings.CLAUDE_API_KEY = None
            
            service = get_ai_service()
            
            assert isinstance(service, MockAIService)
    
    @pytest.mark.unit
    def test_get_ai_service_invalid_openai_config(self):
        """Test fallback when OpenAI config is invalid"""
        with patch('app.core.ai_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "invalid-key"
            mock_settings.CLAUDE_API_KEY = "sk-ant-test123456789"
            
            service = get_ai_service()
            
            # Should fallback to Claude
            assert isinstance(service, ClaudeService)


class TestAIServiceIntegration:
    """Integration tests for AI services"""
    
    @pytest.mark.integration
    async def test_full_response_generation_flow(self, sample_passages):
        """Test complete response generation flow"""
        service = get_ai_service()  # Will get mock service in test env
        
        context_passages = [
            {
                'content': passage.content,
                'reference': passage.reference,
                'similarity_score': 0.8
            }
            for passage in sample_passages[:2]
        ]
        
        response = await service.generate_response(
            user_question="How can I develop stronger faith?",
            context_passages=context_passages,
            theme="faith"
        )
        
        assert isinstance(response, str)
        assert len(response) > 100  # Should be substantial
        assert any(word in response.lower() for word in ["faith", "god", "spiritual", "practice"])
    
    @pytest.mark.integration
    async def test_error_resilience(self):
        """Test that service handles errors gracefully"""
        service = get_ai_service()
        
        # Test with problematic inputs
        test_cases = [
            ("", []),  # Empty question
            ("How to be spiritual?", None),  # None passages
            ("What is dharma?" * 1000, []),  # Very long question
        ]
        
        for question, passages in test_cases:
            try:
                response = await service.generate_response(
                    user_question=question,
                    context_passages=passages or [],
                    theme=None
                )
                # Should not crash and return some response
                assert isinstance(response, str)
            except Exception as e:
                pytest.fail(f"Service should handle edge cases gracefully: {e}")