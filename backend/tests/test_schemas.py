"""
Tests for data schemas and validation
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.schemas.theme import ThemeResponse, ThemeDetail
from app.schemas.response import HealthResponse


class TestChatSchemas:
    """Test chat-related schemas"""
    
    @pytest.mark.unit
    def test_chat_request_valid(self):
        """Test valid chat request creation"""
        request = ChatRequest(
            message="How should I practice devotion?",
            theme="devotion",
            context=["previous_message_1"]
        )
        
        assert request.message == "How should I practice devotion?"
        assert request.theme == "devotion"
        assert request.context == ["previous_message_1"]
    
    @pytest.mark.unit
    def test_chat_request_minimal(self):
        """Test minimal chat request"""
        request = ChatRequest(message="Hello")
        
        assert request.message == "Hello"
        assert request.theme is None
        assert request.context is None
    
    @pytest.mark.unit
    def test_chat_request_validation_empty_message(self):
        """Test chat request validation with empty message"""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="")
        
        assert "at least 1 character" in str(exc_info.value).lower()
    
    @pytest.mark.unit
    def test_chat_request_validation_too_long(self):
        """Test chat request validation with message too long"""
        long_message = "x" * 1001  # Exceeds 1000 character limit
        
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message=long_message)
        
        assert "at most 1000 characters" in str(exc_info.value).lower()
    
    @pytest.mark.unit
    def test_citation_creation(self):
        """Test citation creation"""
        citation = Citation(
            reference="Gadhada I-1",
            passage="One who has firm faith in God...",
            page_number=15,
            relevance_score=0.95
        )
        
        assert citation.reference == "Gadhada I-1"
        assert citation.passage == "One who has firm faith in God..."
        assert citation.page_number == 15
        assert citation.relevance_score == 0.95
    
    @pytest.mark.unit
    def test_citation_minimal(self):
        """Test minimal citation creation"""
        citation = Citation(
            reference="Test-1",
            passage="Test passage"
        )
        
        assert citation.reference == "Test-1"
        assert citation.passage == "Test passage"
        assert citation.page_number is None
        assert citation.relevance_score is None
    
    @pytest.mark.unit
    def test_chat_response_creation(self):
        """Test chat response creation"""
        citations = [
            Citation(
                reference="Gadhada I-1",
                passage="One who has firm faith...",
                relevance_score=0.95
            )
        ]
        
        response = ChatResponse(
            response="Based on the Vachanamrut teachings...",
            citations=citations,
            related_themes=["faith", "devotion"]
        )
        
        assert response.response == "Based on the Vachanamrut teachings..."
        assert len(response.citations) == 1
        assert "faith" in response.related_themes
        assert isinstance(response.timestamp, datetime)
    
    @pytest.mark.unit
    def test_chat_response_empty_lists(self):
        """Test chat response with empty lists"""
        response = ChatResponse(
            response="Simple response"
        )
        
        assert response.response == "Simple response"
        assert response.citations == []
        assert response.related_themes == []
        assert isinstance(response.timestamp, datetime)
    
    @pytest.mark.unit
    def test_chat_response_camel_case_conversion(self):
        """Test camel case conversion in response"""
        citations = [
            Citation(reference="Test-1", passage="Test", relevance_score=0.8)
        ]
        
        response = ChatResponse(
            response="Test response",
            citations=citations,
            related_themes=["test"]
        )
        
        # Test that the schema supports both snake_case and camelCase
        assert hasattr(response, 'related_themes')
        
        # Test JSON serialization with camelCase
        json_data = response.model_dump(by_alias=True)
        assert 'relatedThemes' in json_data
        assert 'relevanceScore' in json_data['citations'][0]


class TestThemeSchemas:
    """Test theme-related schemas"""
    
    @pytest.mark.unit
    def test_theme_response_creation(self):
        """Test theme response creation"""
        theme = ThemeResponse(
            name="devotion",
            description="The practice of loving dedication to God",
            keywords=["bhakti", "love", "surrender"]
        )
        
        assert theme.name == "devotion"
        assert theme.description == "The practice of loving dedication to God"
        assert "bhakti" in theme.keywords
    
    @pytest.mark.unit
    def test_theme_response_minimal(self):
        """Test minimal theme response"""
        theme = ThemeResponse(name="faith")
        
        assert theme.name == "faith"
        assert theme.description is None
        assert theme.keywords == []
    
    @pytest.mark.unit
    def test_theme_detail_creation(self):
        """Test theme detail creation"""
        detail = ThemeDetail(
            name="devotion",
            description="Detailed description of devotion",
            keywords=["bhakti", "love"],
            related_passages=["Gadhada I-1", "Sarangpur-5"],
            related_themes=["faith", "surrender"]
        )
        
        assert detail.name == "devotion"
        assert detail.description == "Detailed description of devotion"
        assert len(detail.keywords) == 2
        assert len(detail.related_passages) == 2
        assert len(detail.related_themes) == 2
    
    @pytest.mark.unit
    def test_theme_detail_empty_lists(self):
        """Test theme detail with empty lists"""
        detail = ThemeDetail(
            name="test",
            description="Test description"
        )
        
        assert detail.name == "test"
        assert detail.description == "Test description"
        assert detail.keywords == []
        assert detail.related_passages == []
        assert detail.related_themes == []
    
    @pytest.mark.unit
    def test_theme_detail_camel_case_conversion(self):
        """Test camel case conversion in theme detail"""
        detail = ThemeDetail(
            name="devotion",
            description="Test description",
            related_passages=["Test-1"],
            related_themes=["faith"]
        )
        
        # Test JSON serialization with camelCase
        json_data = detail.model_dump(by_alias=True)
        assert 'relatedPassages' in json_data
        assert 'relatedThemes' in json_data


class TestResponseSchemas:
    """Test response schemas"""
    
    @pytest.mark.unit
    def test_health_response_creation(self):
        """Test health response creation"""
        response = HealthResponse(
            status="healthy",
            message="API is running",
            version="1.0.0"
        )
        
        assert response.status == "healthy"
        assert response.message == "API is running"
        assert response.version == "1.0.0"


class TestSchemaValidation:
    """Test schema validation edge cases"""
    
    @pytest.mark.unit
    def test_chat_request_whitespace_message(self):
        """Test chat request with whitespace-only message"""
        with pytest.raises(ValidationError):
            ChatRequest(message="   ")  # Only whitespace
    
    @pytest.mark.unit
    def test_citation_empty_fields(self):
        """Test citation with empty required fields"""
        with pytest.raises(ValidationError):
            Citation(reference="", passage="Test")
        
        with pytest.raises(ValidationError):
            Citation(reference="Test", passage="")
    
    @pytest.mark.unit
    def test_relevance_score_bounds(self):
        """Test that relevance scores are within valid bounds"""
        # Valid scores
        citation1 = Citation(reference="Test", passage="Test", relevance_score=0.0)
        citation2 = Citation(reference="Test", passage="Test", relevance_score=1.0)
        citation3 = Citation(reference="Test", passage="Test", relevance_score=0.5)
        
        assert citation1.relevance_score == 0.0
        assert citation2.relevance_score == 1.0
        assert citation3.relevance_score == 0.5
        
        # Test that invalid scores are handled appropriately
        # Note: Pydantic doesn't enforce bounds by default, but we could add validators
    
    @pytest.mark.unit
    def test_unicode_content_handling(self):
        """Test handling of unicode content"""
        # Test with various unicode characters
        unicode_message = "આધ્યાત્મિક માર્ગદર્શન શું છે? 🙏"  # Gujarati + emoji
        sanskrit_passage = "ॐ शान्ति शान्ति शान्तिः"  # Sanskrit
        
        request = ChatRequest(message=unicode_message)
        assert request.message == unicode_message
        
        citation = Citation(
            reference="Sanskrit-1",
            passage=sanskrit_passage
        )
        assert citation.passage == sanskrit_passage
    
    @pytest.mark.unit
    def test_nested_validation(self):
        """Test nested schema validation"""
        # Invalid citation in response should raise validation error
        with pytest.raises(ValidationError):
            ChatResponse(
                response="Test response",
                citations=[
                    Citation(reference="", passage="Invalid")  # Empty reference
                ]
            )
    
    @pytest.mark.unit
    def test_extra_fields_handling(self):
        """Test handling of extra fields"""
        # Should ignore extra fields due to model configuration
        request_data = {
            "message": "Test message",
            "theme": "devotion",
            "extra_field": "should be ignored"
        }
        
        # This should work without raising an error
        request = ChatRequest(**request_data)
        assert request.message == "Test message"
        assert request.theme == "devotion"
        # extra_field should be ignored
    
    @pytest.mark.unit
    def test_json_serialization_deserialization(self):
        """Test JSON serialization and deserialization"""
        original_response = ChatResponse(
            response="Test response",
            citations=[
                Citation(
                    reference="Test-1",
                    passage="Test passage",
                    page_number=10,
                    relevance_score=0.85
                )
            ],
            related_themes=["test", "validation"]
        )
        
        # Serialize to JSON
        json_str = original_response.model_dump_json()
        assert isinstance(json_str, str)
        assert "Test response" in json_str
        
        # Deserialize from JSON
        json_data = original_response.model_dump()
        reconstructed = ChatResponse(**json_data)
        
        assert reconstructed.response == original_response.response
        assert len(reconstructed.citations) == len(original_response.citations)
        assert reconstructed.citations[0].reference == original_response.citations[0].reference
        assert reconstructed.related_themes == original_response.related_themes