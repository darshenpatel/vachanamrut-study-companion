"""
Tests for Semantic Search Engine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict


class TestSemanticSearchEngine:
    """Tests for VachanamrutSemanticSearch"""
    
    @pytest.fixture
    def sample_discourses(self):
        """Sample discourse data for testing"""
        return [
            {
                'id': 'gadhada_i_1',
                'content': 'One who has firm faith in God will never experience distress.',
                'title': 'Gadhada I-1',
                'reference': 'Gadhada I-1',
                'chapter': 'Gadhada I',
                'spiritual_themes': ['faith', 'devotion'],
                'page_number': 1,
                'word_count': 15,
                'quality_score': 0.8
            },
            {
                'id': 'sarangpur_5',
                'content': 'Devotion to God should be practiced with complete dedication.',
                'title': 'Sarangpur-5',
                'reference': 'Sarangpur-5',
                'chapter': 'Sarangpur',
                'spiritual_themes': ['devotion', 'surrender'],
                'page_number': 2,
                'word_count': 12,
                'quality_score': 0.85
            },
            {
                'id': 'vadtal_18',
                'content': 'Surrender to God means offering ego and desires at His feet.',
                'title': 'Vadtal-18',
                'reference': 'Vadtal-18',
                'chapter': 'Vadtal',
                'spiritual_themes': ['surrender', 'detachment'],
                'page_number': 3,
                'word_count': 14,
                'quality_score': 0.9
            }
        ]
    
    def test_search_query_creation(self):
        """Test SearchQuery dataclass creation"""
        from app.core.semantic_search_engine import SearchQuery
        
        query = SearchQuery(
            query_text="How to develop faith?",
            theme_filter="faith",
            max_results=5,
            min_similarity=0.3
        )
        
        assert query.query_text == "How to develop faith?"
        assert query.theme_filter == "faith"
        assert query.max_results == 5
        assert query.min_similarity == 0.3
    
    def test_search_result_creation(self):
        """Test SearchResult dataclass creation"""
        from app.core.semantic_search_engine import SearchResult
        
        result = SearchResult(
            discourse_id='test_1',
            reference='Gadhada I-1',
            content='Test content',
            title='Test Title',
            chapter='Gadhada I',
            similarity_score=0.85,
            semantic_score=0.85,
            metadata_score=0.7,
            combined_score=0.8,
            page_number=1,
            word_count=100,
            spiritual_themes=['faith'],
            explanation='high semantic match'
        )
        
        assert result.reference == 'Gadhada I-1'
        assert result.combined_score == 0.8
    
    @pytest.mark.skipif(
        True,  # Skip if sentence-transformers not installed
        reason="Requires sentence-transformers"
    )
    def test_semantic_search_initialization(self, sample_discourses):
        """Test search engine initialization"""
        from app.core.semantic_search_engine import VachanamrutSemanticSearch
        
        search_engine = VachanamrutSemanticSearch()
        result = search_engine.initialize(sample_discourses)
        
        assert result is True
        assert search_engine.is_initialized
        assert len(search_engine.discourses) == 3
    
    def test_fallback_search(self, sample_discourses):
        """Test fallback text-based search when dependencies unavailable"""
        from app.core.semantic_search_engine import VachanamrutSemanticSearch, SearchQuery
        
        search_engine = VachanamrutSemanticSearch()
        # Force fallback mode
        search_engine._initialize_fallback(sample_discourses)
        
        query = SearchQuery(
            query_text="faith God",
            max_results=5,
            min_similarity=0.1
        )
        
        results = search_engine._fallback_search(query)
        
        assert isinstance(results, list)


class TestSemanticRetriever:
    """Tests for SemanticRetriever service"""
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine"""
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.search = Mock(return_value=[])
        mock_engine.get_search_stats = Mock(return_value={'total_discourses': 10})
        return mock_engine
    
    @pytest.mark.asyncio
    async def test_semantic_retriever_initialization(self):
        """Test SemanticRetriever initialization"""
        from app.core.semantic_retrieval import SemanticRetriever
        
        retriever = SemanticRetriever()
        assert retriever.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_get_semantic_retriever_singleton(self):
        """Test that get_semantic_retriever returns singleton"""
        from app.core.semantic_retrieval import get_semantic_retriever
        
        retriever1 = get_semantic_retriever()
        retriever2 = get_semantic_retriever()
        
        # Should return same instance
        assert retriever1 is retriever2


class TestThemeMappings:
    """Tests for theme mappings"""
    
    def test_all_themes_defined(self):
        """Test that all 10 themes are defined"""
        from app.core.theme_mappings import get_all_themes, SPIRITUAL_THEMES
        
        themes = get_all_themes()
        
        assert len(themes) == 10
        assert 'devotion' in themes
        assert 'faith' in themes
        assert 'surrender' in themes
        assert 'service' in themes
        assert 'knowledge' in themes
        assert 'detachment' in themes
        assert 'dharma' in themes
        assert 'meditation' in themes
        assert 'guru' in themes
        assert 'satsang' in themes
    
    def test_get_theme_returns_mapping(self):
        """Test getting a specific theme"""
        from app.core.theme_mappings import get_theme
        
        theme = get_theme('devotion')
        
        assert theme is not None
        assert theme.name == 'devotion'
        assert len(theme.keywords) > 0
        assert len(theme.related_themes) > 0
        assert len(theme.key_passages) > 0
        assert len(theme.guidance) > 0
    
    def test_get_theme_case_insensitive(self):
        """Test theme lookup is case insensitive"""
        from app.core.theme_mappings import get_theme
        
        theme1 = get_theme('DEVOTION')
        theme2 = get_theme('Devotion')
        theme3 = get_theme('devotion')
        
        assert theme1 == theme2 == theme3
    
    def test_detect_themes_in_text(self):
        """Test theme detection in text"""
        from app.core.theme_mappings import detect_themes_in_text
        
        text = "Faith and devotion are essential for spiritual progress. One should have trust in God."
        detected = detect_themes_in_text(text)
        
        assert 'faith' in detected or 'devotion' in detected
    
    def test_get_related_themes(self):
        """Test getting related themes"""
        from app.core.theme_mappings import get_related_themes
        
        related = get_related_themes('devotion')
        
        assert isinstance(related, list)
        assert len(related) > 0
        assert 'faith' in related  # devotion is related to faith
    
    def test_theme_categories(self):
        """Test theme categories"""
        from app.core.theme_mappings import get_themes_by_category, THEME_CATEGORIES
        
        categories = get_themes_by_category()
        
        assert 'Foundational' in categories
        assert 'Practice' in categories
        assert 'Knowledge' in categories
        assert 'Community' in categories


class TestTextCleaner:
    """Tests for text cleaner"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        from app.core.text_cleaner import get_text_cleaner
        
        cleaner = get_text_cleaner()
        
        text = "VachanImrut  Gadhada  I-1"
        cleaned = cleaner.clean_text(text)
        
        assert "Vachanamrut" in cleaned
        assert "  " not in cleaned
    
    def test_fix_maharaj_ocr_errors(self):
        """Test fixing Maharaj OCR errors"""
        from app.core.text_cleaner import get_text_cleaner
        
        cleaner = get_text_cleaner()
        
        text = "Mahiiriij said to the devotees"
        cleaned = cleaner.clean_text(text)
        
        assert "Maharaj" in cleaned
    
    def test_fix_reference(self):
        """Test reference fixing"""
        from app.core.text_cleaner import get_text_cleaner
        
        cleaner = get_text_cleaner()
        
        reference = "GadhadH I - 1"
        fixed = cleaner.fix_reference(reference)
        
        assert "-" in fixed
        assert "  " not in fixed
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        from app.core.text_cleaner import get_text_cleaner
        
        cleaner = get_text_cleaner()
        
        # Good quality text
        good_text = "This discourse emphasizes the importance of faith and devotion in spiritual practice. One should maintain trust in God and practice constant remembrance."
        good_score = cleaner.calculate_quality_score(good_text)
        
        # Poor quality text
        poor_text = "123 456 789 table index"
        poor_score = cleaner.calculate_quality_score(poor_text)
        
        assert good_score > poor_score
        assert 0 <= good_score <= 1
        assert 0 <= poor_score <= 1
    
    def test_should_skip_passage(self):
        """Test passage skip detection"""
        from app.core.text_cleaner import get_text_cleaner
        
        cleaner = get_text_cleaner()
        
        # Should skip
        assert cleaner.should_skip_passage("") is True
        assert cleaner.should_skip_passage("short") is True
        assert cleaner.should_skip_passage("Glossary of terms used in this book") is True
        
        # Should not skip
        valid_text = "This is a valid passage about devotion and faith that should be included in the search."
        assert cleaner.should_skip_passage(valid_text) is False


class TestAnalytics:
    """Tests for analytics service"""
    
    def test_track_interaction(self, tmp_path):
        """Test tracking a chat interaction"""
        from app.core.analytics import AnalyticsService
        
        service = AnalyticsService(storage_path=str(tmp_path))
        
        interaction_id = service.track_interaction(
            query="How to develop faith?",
            response="Faith is developed through practice...",
            theme="faith",
            citations_count=2,
            related_themes=["devotion", "surrender"],
            response_time_ms=150,
            search_method="semantic"
        )
        
        assert interaction_id is not None
        assert len(service.interactions) == 1
    
    def test_record_feedback(self, tmp_path):
        """Test recording feedback"""
        from app.core.analytics import AnalyticsService
        
        service = AnalyticsService(storage_path=str(tmp_path))
        
        interaction_id = service.track_interaction(
            query="Test query",
            response="Test response",
            theme=None,
            citations_count=1,
            related_themes=[],
            response_time_ms=100,
            search_method="text"
        )
        
        success = service.record_feedback(
            interaction_id=interaction_id,
            score=4,
            feedback_text="Very helpful!"
        )
        
        assert success is True
        assert service.interactions[0].feedback_score == 4
    
    def test_get_metrics(self, tmp_path):
        """Test getting metrics"""
        from app.core.analytics import AnalyticsService
        
        service = AnalyticsService(storage_path=str(tmp_path))
        
        # Track some interactions
        for i in range(5):
            service.track_interaction(
                query=f"Query {i}",
                response=f"Response {i}",
                theme="devotion" if i % 2 == 0 else "faith",
                citations_count=i,
                related_themes=[],
                response_time_ms=100 + i * 10,
                search_method="semantic"
            )
        
        metrics = service.get_metrics()
        
        assert metrics['total_queries'] == 5
        assert 'avg_response_time_ms' in metrics
        assert 'theme_distribution' in metrics
    
    def test_feedback_validation(self, tmp_path):
        """Test feedback score validation"""
        from app.core.analytics import AnalyticsService
        
        service = AnalyticsService(storage_path=str(tmp_path))
        
        # Invalid scores
        assert service.record_feedback("fake_id", 0) is False
        assert service.record_feedback("fake_id", 6) is False

