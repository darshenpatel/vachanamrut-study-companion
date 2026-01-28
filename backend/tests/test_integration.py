"""
Integration tests for the complete Vachanamrut Study Companion system
"""

import pytest
import asyncio
import tempfile
import json
from unittest.mock import patch, Mock

from app.services.chat_service import ChatService
from app.services.theme_service import ThemeService
from app.core.pdf_retrieval import PDFVectorStore
from app.core.ai_service import get_ai_service
from app.schemas.chat import ChatRequest


class TestFullSystemIntegration:
    """Test complete system integration"""
    
    @pytest.mark.integration
    async def test_complete_chat_flow(self, sample_passages, temp_storage_path):
        """Test complete chat flow from request to response"""
        # Setup PDF store with sample data
        pdf_store = PDFVectorStore(storage_path=temp_storage_path)
        pdf_store.passages = sample_passages
        pdf_store._build_indexes()
        pdf_store.save()
        
        # Setup services
        chat_service = ChatService()
        ai_service = get_ai_service()  # Will get mock service in test env
        
        # Mock the retriever in chat service
        from app.core.factual_retrieval import FactualRetriever
        with patch('app.core.factual_retrieval.get_pdf_store', return_value=pdf_store):
            mock_retriever = FactualRetriever()
            mock_retriever.pdf_store = pdf_store
            mock_retriever.is_initialized = True
            
            chat_service.retriever = mock_retriever
            chat_service.ai_service = ai_service
            chat_service._initialized = True
            
            # Test various chat scenarios
            test_scenarios = [
                {
                    "request": ChatRequest(message="How should I practice devotion?", theme="devotion"),
                    "expected_concepts": ["devotion", "practice", "god"]
                },
                {
                    "request": ChatRequest(message="What is the meaning of surrender?", theme="surrender"),
                    "expected_concepts": ["surrender", "god", "ego"]
                },
                {
                    "request": ChatRequest(message="How can I strengthen my faith?"),
                    "expected_concepts": ["faith", "spiritual", "god"]
                }
            ]
            
            for scenario in test_scenarios:
                response = await chat_service.process_message(scenario["request"])
                
                # Verify response structure
                assert hasattr(response, 'response')
                assert hasattr(response, 'citations') 
                assert hasattr(response, 'related_themes')
                assert hasattr(response, 'timestamp')
                
                # Verify response content quality
                assert len(response.response) > 50
                
                # Check that response contains relevant concepts
                response_lower = response.response.lower()
                concept_found = any(concept in response_lower for concept in scenario["expected_concepts"])
                assert concept_found, f"Response should contain concepts: {scenario['expected_concepts']}"
    
    @pytest.mark.integration
    async def test_theme_service_integration(self, sample_passages, temp_storage_path):
        """Test theme service integration with real data"""
        # Setup PDF store
        pdf_store = PDFVectorStore(storage_path=temp_storage_path)
        pdf_store.passages = sample_passages
        pdf_store._build_indexes()
        pdf_store.save()
        
        # Setup theme service
        theme_service = ThemeService()
        
        from app.core.factual_retrieval import FactualRetriever
        with patch('app.core.factual_retrieval.get_pdf_store', return_value=pdf_store):
            mock_retriever = FactualRetriever()
            mock_retriever.pdf_store = pdf_store
            mock_retriever.is_initialized = True
            
            # Mock the get_system_status method
            mock_retriever.get_system_status = Mock(return_value={
                'total_passages': len(sample_passages),
                'chapters': {
                    'Gadhada I': 1,
                    'Sarangpur': 1, 
                    'Vadtal': 1,
                    'Ahmedabad': 1
                }
            })
            
            # Mock the get_chapter_content method
            def mock_get_chapter_content(chapter_name):
                return [p for p in sample_passages if p.chapter == chapter_name]
            
            mock_retriever.get_chapter_content = Mock(side_effect=mock_get_chapter_content)
            
            theme_service.factual_retriever = mock_retriever
            theme_service._initialized = True
            
            # Test theme listing
            themes = await theme_service.list_all_themes()
            assert isinstance(themes, list)
            assert len(themes) > 0
            
            # Test theme details for each available theme
            for theme in themes:
                theme_detail = await theme_service.get_theme_detail(theme)
                
                if theme_detail:  # Some themes might not have details
                    assert hasattr(theme_detail, 'name')
                    assert hasattr(theme_detail, 'description')
                    assert theme_detail.name == theme
                    assert len(theme_detail.description) > 0
    
    @pytest.mark.integration
    async def test_search_performance_integration(self, large_passage_dataset, temp_storage_path):
        """Test system performance with larger dataset"""
        # Setup PDF store with large dataset
        pdf_store = PDFVectorStore(storage_path=temp_storage_path)
        pdf_store.passages = large_passage_dataset
        pdf_store._build_indexes()
        
        import time
        
        # Test search performance
        search_queries = [
            "devotion to God",
            "spiritual practice",
            "faith and surrender", 
            "service to devotees",
            "knowledge and wisdom"
        ]
        
        for query in search_queries:
            start_time = time.time()
            results = pdf_store.search(query, top_k=10)
            search_time = time.time() - start_time
            
            # Should complete quickly even with large dataset
            assert search_time < 0.5  # Less than 500ms
            assert len(results) <= 10
            assert all(isinstance(result, tuple) for result in results)
            
            # Test caching - second search should be faster
            start_time = time.time()
            cached_results = pdf_store.search(query, top_k=10)
            cached_search_time = time.time() - start_time
            
            assert cached_search_time < search_time  # Should be faster due to caching
            assert cached_results == results  # Should return same results
    
    @pytest.mark.integration
    async def test_concurrent_requests(self, sample_passages, temp_storage_path):
        """Test system handling of concurrent requests"""
        # Setup services
        pdf_store = PDFVectorStore(storage_path=temp_storage_path)
        pdf_store.passages = sample_passages
        pdf_store._build_indexes()
        
        chat_service = ChatService()
        ai_service = get_ai_service()
        
        from app.core.factual_retrieval import FactualRetriever
        with patch('app.core.factual_retrieval.get_pdf_store', return_value=pdf_store):
            mock_retriever = FactualRetriever()
            mock_retriever.pdf_store = pdf_store
            mock_retriever.is_initialized = True
            
            chat_service.retriever = mock_retriever
            chat_service.ai_service = ai_service
            chat_service._initialized = True
            
            # Create multiple concurrent requests
            concurrent_requests = [
                ChatRequest(message=f"Question about devotion {i}", theme="devotion")
                for i in range(10)
            ]
            
            # Process requests concurrently
            tasks = [
                chat_service.process_message(request)
                for request in concurrent_requests
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All requests should complete successfully
            assert len(results) == 10
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 8  # At least 80% should succeed
            
            # Verify response quality
            for result in successful_results:
                assert hasattr(result, 'response')
                assert len(result.response) > 20


class TestErrorHandlingIntegration:
    """Test error handling across the integrated system"""
    
    @pytest.mark.integration
    async def test_missing_data_handling(self, temp_storage_path):
        """Test system behavior when data is missing"""
        # Setup with empty PDF store
        pdf_store = PDFVectorStore(storage_path=temp_storage_path)
        # Don't add any passages
        
        chat_service = ChatService()
        ai_service = get_ai_service()
        
        from app.core.factual_retrieval import FactualRetriever
        with patch('app.core.factual_retrieval.get_pdf_store', return_value=pdf_store):
            mock_retriever = FactualRetriever()
            mock_retriever.pdf_store = pdf_store
            mock_retriever.is_initialized = True
            
            chat_service.retriever = mock_retriever
            chat_service.ai_service = ai_service
            chat_service._initialized = True
            
            # Should handle empty data gracefully
            request = ChatRequest(message="How to practice devotion?")
            response = await chat_service.process_message(request)
            
            assert hasattr(response, 'response')
            assert len(response.response) > 0
            assert len(response.citations) == 0  # No citations when no data
    
    @pytest.mark.integration
    async def test_corrupted_data_handling(self, corrupt_data_file):
        """Test system behavior with corrupted data"""
        # Try to load corrupted data
        pdf_store = PDFVectorStore(storage_path=corrupt_data_file)
        load_result = pdf_store.load()
        
        # Should handle corruption gracefully
        assert load_result['status'] == 'error'
        assert len(pdf_store.passages) == 0  # Should not crash
    
    @pytest.mark.integration
    async def test_network_error_simulation(self, sample_passages, temp_storage_path):
        """Test system behavior under network errors"""
        # Setup services with real data but mock network failures
        pdf_store = PDFVectorStore(storage_path=temp_storage_path) 
        pdf_store.passages = sample_passages
        pdf_store._build_indexes()
        
        chat_service = ChatService()
        
        # Mock AI service to simulate network errors
        with patch('app.core.ai_service.get_ai_service') as mock_get_ai:
            mock_ai = Mock()
            mock_ai.generate_response = Mock(side_effect=ConnectionError("Network error"))
            mock_get_ai.return_value = mock_ai
            
            from app.core.factual_retrieval import FactualRetriever
            with patch('app.core.factual_retrieval.get_pdf_store', return_value=pdf_store):
                mock_retriever = FactualRetriever()
                mock_retriever.pdf_store = pdf_store
                mock_retriever.is_initialized = True
                
                chat_service.retriever = mock_retriever
                chat_service.ai_service = mock_ai
                chat_service._initialized = True
                
                request = ChatRequest(message="Test message")
                
                # Should handle network errors gracefully
                try:
                    response = await chat_service.process_message(request)
                    # If it returns a response, it should be a fallback
                    assert hasattr(response, 'response')
                    assert len(response.response) > 0
                except Exception as e:
                    # If it raises an exception, it should be informative
                    assert len(str(e)) > 0


class TestSystemConfiguration:
    """Test system configuration and environment handling"""
    
    @pytest.mark.integration
    def test_environment_detection(self):
        """Test that system correctly detects test environment"""
        import os
        
        # Should be in test environment
        env = os.getenv('ENVIRONMENT', 'development')
        assert env == 'test'
        
        debug = os.getenv('DEBUG', 'false').lower()
        assert debug == 'true'
    
    @pytest.mark.integration
    def test_service_initialization_in_test_env(self):
        """Test that services initialize correctly in test environment"""
        # AI service should default to mock in test environment
        ai_service = get_ai_service()
        
        from app.core.ai_service import MockAIService
        assert isinstance(ai_service, MockAIService)
    
    @pytest.mark.integration
    async def test_system_health_check(self):
        """Test overall system health"""
        from app.core.ai_service import get_service_status
        
        status = get_service_status()
        
        assert 'timestamp' in status
        assert 'services' in status
        assert 'rate_limiter' in status
        
        # Should report service statuses
        services = status['services']
        assert 'openai' in services
        assert 'claude' in services
        
        # Rate limiter should be functional
        rate_limiter = status['rate_limiter']
        assert 'calls_per_minute' in rate_limiter
        assert 'tokens_per_minute' in rate_limiter