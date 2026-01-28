"""
Pytest configuration and fixtures for Vachanamrut Study Companion tests
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add app to path for testing
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.core.pdf_retrieval import VachanamrutPassage, PDFVectorStore
from app.core.ai_service import AIService, MockAIService
from app.services.chat_service import ChatService
from app.services.theme_service import ThemeService
from app.schemas.chat import ChatRequest, ChatResponse
from app.core.text_cleaner import VachanamrutTextCleaner
from app.core.analytics import AnalyticsService

@pytest.fixture
def sample_passages():
    """Create sample Vachanamrut passages for testing"""
    return [
        VachanamrutPassage(
            id="gadhada_i_1_test",
            content="One who has firm faith in God and His saint will never experience distress in any situation. Such a person remains blissful and peaceful always.",
            reference="Gadhada I-1",
            page_number=1,
            chapter="Gadhada I"
        ),
        VachanamrutPassage(
            id="sarangpur_5_test",
            content="Devotion to God should be practiced with complete dedication and without any selfish motive. This leads to ultimate spiritual realization.",
            reference="Sarangpur-5",
            page_number=2,
            chapter="Sarangpur"
        ),
        VachanamrutPassage(
            id="vadtal_18_test", 
            content="Surrender to God means offering one's ego and desires at His feet. Through complete surrender, one attains divine grace.",
            reference="Vadtal-18",
            page_number=3,
            chapter="Vadtal"
        ),
        VachanamrutPassage(
            id="ahmedabad_3_test",
            content="Service to God and His devotees should be performed selflessly, without expecting any rewards or recognition.",
            reference="Ahmedabad-3",
            page_number=4,
            chapter="Ahmedabad"
        )
    ]

@pytest.fixture
def temp_storage_path():
    """Create temporary storage path for testing"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass

@pytest.fixture
def pdf_vector_store(sample_passages, temp_storage_path):
    """Create PDF vector store with sample data"""
    store = PDFVectorStore(storage_path=temp_storage_path)
    store.passages = sample_passages
    store._build_indexes()
    return store

@pytest.fixture
def mock_ai_service():
    """Create mock AI service for testing"""
    service = MockAIService()
    return service

@pytest.fixture
def chat_request():
    """Create sample chat request"""
    return ChatRequest(
        message="How should I practice devotion?",
        theme="devotion",
        context=[]
    )

@pytest.fixture
def mock_pdf_file():
    """Create mock PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        # Write minimal PDF content
        f.write(b'%PDF-1.4\n%Mock PDF content for testing\n')
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass

@pytest.fixture
def sample_themes():
    """Sample themes for testing"""
    return ["devotion", "faith", "surrender", "service", "knowledge", "dharma"]

@pytest.fixture 
def mock_factual_retriever(sample_passages):
    """Mock factual retriever with sample data"""
    mock_retriever = Mock()
    mock_retriever.search = Mock(return_value=[
        {
            'content': passage.content,
            'reference': passage.reference,
            'page_number': passage.page_number,
            'similarity_score': 0.85
        }
        for passage in sample_passages[:2]
    ])
    mock_retriever.initialize = Mock(return_value={'status': 'success'})
    mock_retriever.get_system_status = Mock(return_value={
        'total_passages': len(sample_passages),
        'chapters': {'Gadhada I': 1, 'Sarangpur': 1, 'Vadtal': 1, 'Ahmedabad': 1}
    })
    return mock_retriever

@pytest.fixture
def chat_service(mock_factual_retriever, mock_ai_service):
    """Create chat service with mocked dependencies"""
    service = ChatService()
    service.factual_retriever = mock_factual_retriever
    service.semantic_retriever = None  # No semantic retriever in test
    service._use_semantic = False
    service.ai_service = mock_ai_service
    service._initialized = True
    return service

@pytest.fixture
def theme_service(mock_semantic_retriever):
    """Create theme service with mocked dependencies"""
    service = ThemeService()
    service.semantic_retriever = mock_semantic_retriever
    service._initialized = True
    return service

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Mock environment variables
    os.environ.setdefault('ENVIRONMENT', 'test')
    os.environ.setdefault('DEBUG', 'true')
    
    yield
    
    # Cleanup test environment
    test_vars = ['ENVIRONMENT', 'DEBUG']
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "choices": [
            {
                "message": {
                    "content": "Based on the Vachanamrut teachings, devotion should be practiced with complete sincerity and love for God. True devotion involves offering all actions to God and maintaining constant remembrance of His divine qualities."
                }
            }
        ]
    }

@pytest.fixture
def mock_claude_response():
    """Mock Claude API response"""
    return {
        "content": [
            {
                "text": "The Vachanamrut teaches that surrender is the highest form of spiritual practice. It involves offering one's ego and personal desires at God's feet, trusting completely in His divine will and wisdom."
            }
        ]
    }

# Performance test fixtures
@pytest.fixture
def large_passage_dataset():
    """Create large dataset for performance testing"""
    passages = []
    for i in range(100):
        passages.append(VachanamrutPassage(
            id=f"test_passage_{i}",
            content=f"This is test passage number {i} with spiritual content about devotion and faith. " * 5,
            reference=f"Test-{i}",
            page_number=i + 1,
            chapter="Test Chapter"
        ))
    return passages

# Error testing fixtures
@pytest.fixture
def corrupt_data_file(temp_storage_path):
    """Create corrupt data file for error testing"""
    with open(temp_storage_path, 'w') as f:
        f.write("{ invalid json content")
    return temp_storage_path


@pytest.fixture
def text_cleaner():
    """Create text cleaner for testing"""
    return VachanamrutTextCleaner()


@pytest.fixture
def analytics_service(tmp_path):
    """Create analytics service with temp storage"""
    return AnalyticsService(storage_path=str(tmp_path / "analytics"))


@pytest.fixture
def sample_discourse_data():
    """Sample discourse data for semantic search testing"""
    return [
        {
            'id': 'test_1',
            'content': 'Faith and devotion are the foundations of spiritual life.',
            'title': 'Test Discourse 1',
            'reference': 'Test-1',
            'chapter': 'Test Chapter',
            'spiritual_themes': ['faith', 'devotion'],
            'page_number': 1,
            'word_count': 10
        },
        {
            'id': 'test_2',
            'content': 'Surrender to God brings peace and liberation.',
            'title': 'Test Discourse 2',
            'reference': 'Test-2',
            'chapter': 'Test Chapter',
            'spiritual_themes': ['surrender', 'detachment'],
            'page_number': 2,
            'word_count': 8
        }
    ]


@pytest.fixture
def mock_semantic_retriever():
    """Mock semantic retriever"""
    mock = Mock()
    mock.is_initialized = True
    mock.search = Mock(return_value=[
        {
            'content': 'Faith is essential for spiritual growth.',
            'reference': 'Gadhada I-1',
            'page_number': 1,
            'similarity_score': 0.85,
            'themes': ['faith']
        }
    ])
    mock.initialize = Mock(return_value={'status': 'initialized'})
    mock.get_system_status = Mock(return_value={'initialized': True})
    return mock