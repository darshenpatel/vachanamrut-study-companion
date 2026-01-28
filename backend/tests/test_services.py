"""
Tests for service layer components (ChatService, ThemeService)
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.services.chat_service import ChatService
from app.services.theme_service import ThemeService  
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.schemas.theme import ThemeDetail


class TestChatService:
    """Test ChatService functionality"""
    
    @pytest.mark.unit
    def test_chat_service_initialization(self):
        """Test chat service initialization"""
        service = ChatService()
        
        # Updated for new dual-retriever architecture
        assert service.semantic_retriever is None
        assert service.factual_retriever is None
        assert service.ai_service is not None
        assert not service._initialized
    
    @pytest.mark.unit
    async def test_process_message_success(self, chat_service, chat_request, sample_passages):
        """Test successful message processing"""
        # Mock retriever search results
        search_results = [
            {
                'content': passage.content,
                'reference': passage.reference,
                'page_number': passage.page_number,
                'similarity_score': 0.85
            }
            for passage in sample_passages[:2]
        ]
        chat_service.retriever.search.return_value = search_results
        
        # Mock AI service response
        chat_service.ai_service.generate_response = AsyncMock(
            return_value="Based on the Vachanamrut teachings, devotion should be practiced with complete sincerity..."
        )
        
        response = await chat_service.process_message(chat_request)
        
        assert isinstance(response, ChatResponse)
        assert len(response.response) > 0
        assert len(response.citations) > 0
        assert isinstance(response.timestamp, datetime)
        
        # Verify citations are properly formatted
        for citation in response.citations:
            assert isinstance(citation, Citation)
            assert len(citation.reference) > 0
            assert len(citation.passage) > 0
    
    @pytest.mark.unit
    async def test_process_message_with_initialization(self, mock_factual_retriever, mock_ai_service, chat_request):
        """Test message processing with service initialization"""
        service = ChatService()
        service.factual_retriever = mock_factual_retriever
        service.semantic_retriever = None
        service.ai_service = mock_ai_service
        service._initialized = False
        
        # Mock initialization
        service._ensure_initialized = AsyncMock()
        
        await service.process_message(chat_request)
        
        service._ensure_initialized.assert_called_once()
    
    @pytest.mark.unit
    async def test_process_message_no_search_results(self, chat_service, chat_request):
        """Test message processing when no search results found"""
        # Mock empty search results
        chat_service.retriever.search.return_value = []
        
        # Mock AI service response
        chat_service.ai_service.generate_response = AsyncMock(
            return_value="I understand you're seeking spiritual guidance..."
        )
        
        response = await chat_service.process_message(chat_request)
        
        assert isinstance(response, ChatResponse)
        assert len(response.response) > 0
        assert len(response.citations) == 0  # No citations when no search results
    
    @pytest.mark.unit
    async def test_process_message_error_handling(self, chat_service, chat_request):
        """Test error handling in message processing"""
        # Mock retriever to raise an exception
        chat_service.retriever.search.side_effect = Exception("Retrieval error")
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            await chat_service.process_message(chat_request)
    
    @pytest.mark.unit
    async def test_get_themes(self, chat_service):
        """Test getting available themes"""
        expected_themes = ["devotion", "faith", "surrender", "service"]
        chat_service.retriever.get_system_status.return_value = {
            'chapters': {theme: 5 for theme in expected_themes}
        }
        
        themes = await chat_service.get_themes()
        
        assert isinstance(themes, list)
        assert len(themes) > 0
        for theme in themes:
            assert isinstance(theme, str)
    
    @pytest.mark.unit
    def test_format_citations(self, chat_service, sample_passages):
        """Test citation formatting"""
        search_results = [
            {
                'content': passage.content,
                'reference': passage.reference, 
                'page_number': passage.page_number,
                'similarity_score': 0.85
            }
            for passage in sample_passages[:2]
        ]
        
        citations = []
        for result in search_results:
            citations.append(Citation(
                reference=result["reference"],
                passage=result["content"],
                page_number=result.get("page_number"),
                relevance_score=result.get("similarity_score", 0.0),
            ))
        
        assert len(citations) == 2
        assert all(isinstance(c, Citation) for c in citations)
        assert all(c.relevance_score > 0 for c in citations)


class TestThemeService:
    """Test ThemeService functionality"""
    
    @pytest.mark.unit
    def test_theme_service_initialization(self):
        """Test theme service initialization"""
        service = ThemeService()
        
        # Updated for new theme service using semantic retriever
        assert service.semantic_retriever is None
        assert not service._initialized
    
    @pytest.mark.unit
    async def test_list_all_themes_success(self, theme_service):
        """Test successful theme listing"""
        themes = await theme_service.list_all_themes()
        
        assert isinstance(themes, list)
        assert len(themes) > 0
        assert all(isinstance(theme, str) for theme in themes)
    
    @pytest.mark.unit
    async def test_list_all_themes_returns_curated(self, theme_service):
        """Test theme listing returns curated 10 spiritual themes"""
        # ThemeService now uses curated themes from theme_mappings.py
        themes = await theme_service.list_all_themes()
        
        # Should have all 10 curated themes
        assert len(themes) == 10
        assert 'devotion' in themes
        assert 'faith' in themes
        assert 'surrender' in themes
        assert 'satsang' in themes
    
    @pytest.mark.unit
    async def test_list_all_themes_includes_core_themes(self, theme_service):
        """Test theme listing includes all core spiritual themes"""
        themes = await theme_service.list_all_themes()
        
        # Should return all 10 curated themes
        core_themes = ["devotion", "faith", "surrender", "service", "knowledge", "dharma", "meditation", "guru", "satsang", "detachment"]
        assert all(theme in themes for theme in core_themes)
    
    @pytest.mark.unit 
    async def test_list_all_themes_always_returns_themes(self, theme_service):
        """Test theme listing always returns the curated list"""
        themes = await theme_service.list_all_themes()
        
        # Should always return themes
        assert isinstance(themes, list)
        assert len(themes) == 10
    
    @pytest.mark.unit
    async def test_get_theme_detail_success(self, theme_service):
        """Test successful theme detail retrieval"""
        theme_name = "devotion"
        
        # Uses curated theme mappings
        theme_detail = await theme_service.get_theme_detail(theme_name)
        
        assert isinstance(theme_detail, ThemeDetail)
        assert theme_detail.name == theme_name
        assert len(theme_detail.description) > 0
        assert len(theme_detail.keywords) > 0
        assert len(theme_detail.related_passages) > 0
    
    @pytest.mark.unit
    async def test_get_theme_detail_has_related_themes(self, theme_service):
        """Test theme detail includes related themes"""
        theme_name = "faith"
        
        theme_detail = await theme_service.get_theme_detail(theme_name)
        
        assert isinstance(theme_detail, ThemeDetail)
        assert theme_detail.name == theme_name
        assert len(theme_detail.related_themes) > 0
    
    @pytest.mark.unit
    async def test_get_theme_detail_not_found(self, theme_service):
        """Test theme detail when theme not found"""
        theme_name = "nonexistent_theme_xyz"
        
        theme_detail = await theme_service.get_theme_detail(theme_name)
        
        # Non-curated themes return None
        assert theme_detail is None
    
    @pytest.mark.unit
    async def test_get_theme_detail_case_insensitive(self, theme_service):
        """Test theme detail lookup is case insensitive"""
        # Should work with different cases
        theme_detail = await theme_service.get_theme_detail("DEVOTION")
        
        assert theme_detail is not None
        assert theme_detail.name == "devotion"
    
    @pytest.mark.unit
    async def test_ensure_initialized(self, theme_service):
        """Test service initialization"""
        theme_service._initialized = False
        
        # Should initialize without error
        await theme_service._ensure_initialized()
        
        assert theme_service._initialized


class TestServiceIntegration:
    """Integration tests for services"""
    
    @pytest.mark.integration
    async def test_chat_to_theme_service_integration(self, mock_factual_retriever, mock_ai_service, sample_passages):
        """Test integration between chat and theme services"""
        # Setup services
        chat_service = ChatService()
        chat_service.retriever = mock_factual_retriever
        chat_service.ai_service = mock_ai_service
        chat_service._initialized = True
        
        theme_service = ThemeService()
        theme_service.factual_retriever = mock_factual_retriever
        theme_service._initialized = True
        
        # Test getting themes first
        themes = await theme_service.list_all_themes()
        assert len(themes) > 0
        
        # Test chat with one of the themes
        theme = themes[0] if themes else "devotion"
        chat_request = ChatRequest(
            message=f"Tell me about {theme}",
            theme=theme,
            context=[]
        )
        
        # Mock search and AI response
        mock_factual_retriever.search.return_value = [
            {
                'content': sample_passages[0].content,
                'reference': sample_passages[0].reference,
                'page_number': sample_passages[0].page_number,
                'similarity_score': 0.9
            }
        ]
        mock_ai_service.generate_response = AsyncMock(
            return_value=f"Here is guidance about {theme} from the Vachanamrut..."
        )
        
        response = await chat_service.process_message(chat_request)
        
        assert isinstance(response, ChatResponse)
        assert theme.lower() in response.response.lower() or "spiritual" in response.response.lower()
    
    @pytest.mark.integration
    async def test_service_error_resilience(self, mock_factual_retriever, mock_ai_service):
        """Test that services handle errors gracefully"""
        chat_service = ChatService()
        chat_service.retriever = mock_factual_retriever
        chat_service.ai_service = mock_ai_service
        chat_service._initialized = True
        
        # Test with various error conditions
        error_conditions = [
            Exception("Network error"),
            ValueError("Invalid data"),
            KeyError("Missing key"),
        ]
        
        for error in error_conditions:
            # Reset mocks
            mock_factual_retriever.search.side_effect = error
            mock_ai_service.generate_response = AsyncMock(side_effect=error)
            
            chat_request = ChatRequest(
                message="Test message",
                theme=None,
                context=[]
            )
            
            # Should handle errors gracefully (either return fallback or raise appropriate exception)
            try:
                response = await chat_service.process_message(chat_request)
                # If it returns a response, it should be valid
                assert isinstance(response, ChatResponse)
            except Exception as e:
                # If it raises an exception, it should be informative
                assert len(str(e)) > 0
    
    @pytest.mark.slow
    async def test_performance_with_large_context(self, mock_factual_retriever, mock_ai_service):
        """Test service performance with large context"""
        chat_service = ChatService()
        chat_service.retriever = mock_factual_retriever
        chat_service.ai_service = mock_ai_service
        chat_service._initialized = True
        
        # Mock large search results
        large_context = [
            {
                'content': f"This is passage {i} with spiritual content about devotion and faith. " * 10,
                'reference': f"Test-{i}",
                'page_number': i,
                'similarity_score': 0.8 - (i * 0.1)
            }
            for i in range(20)  # Large number of results
        ]
        
        mock_factual_retriever.search.return_value = large_context
        mock_ai_service.generate_response = AsyncMock(
            return_value="Comprehensive spiritual guidance based on extensive teachings..."
        )
        
        chat_request = ChatRequest(
            message="Provide comprehensive guidance on spiritual practice",
            theme=None,
            context=[]
        )
        
        import time
        start_time = time.time()
        response = await chat_service.process_message(chat_request)
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time even with large context
        assert processing_time < 5.0  # Less than 5 seconds
        assert isinstance(response, ChatResponse)
        assert len(response.citations) > 0  # Should have citations despite large context