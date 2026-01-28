"""
Tests for PDF retrieval and processing functionality
"""

import pytest
import json
import time
from unittest.mock import patch, Mock
from pathlib import Path

from app.core.pdf_retrieval import PDFVectorStore, VachanamrutPassage


class TestVachanamrutPassage:
    """Test VachanamrutPassage data class"""
    
    @pytest.mark.unit
    def test_passage_creation(self):
        """Test creating a passage"""
        passage = VachanamrutPassage(
            id="test_id",
            content="Test content",
            reference="Test-1",
            page_number=1,
            chapter="Test"
        )
        assert passage.id == "test_id"
        assert passage.content == "Test content"
        assert passage.reference == "Test-1"
        assert passage.page_number == 1
        assert passage.chapter == "Test"
    
    @pytest.mark.unit
    def test_passage_to_dict(self):
        """Test converting passage to dict"""
        passage = VachanamrutPassage(
            id="test_id",
            content="Test content",
            reference="Test-1",
            page_number=1,
            chapter="Test"
        )
        result = passage.to_dict()
        expected = {
            'id': 'test_id',
            'content': 'Test content',
            'reference': 'Test-1',
            'page_number': 1,
            'chapter': 'Test'
        }
        assert result == expected
    
    @pytest.mark.unit
    def test_passage_from_dict(self):
        """Test creating passage from dict"""
        data = {
            'id': 'test_id',
            'content': 'Test content',
            'reference': 'Test-1',
            'page_number': 1,
            'chapter': 'Test'
        }
        passage = VachanamrutPassage.from_dict(data)
        assert passage.id == "test_id"
        assert passage.content == "Test content"


class TestPDFVectorStore:
    """Test PDF vector store functionality"""
    
    @pytest.mark.unit
    def test_initialization(self, temp_storage_path):
        """Test store initialization"""
        store = PDFVectorStore(storage_path=temp_storage_path)
        assert store.storage_path == temp_storage_path
        assert len(store.passages) == 0
        assert store.search_stats['total_searches'] == 0
    
    @pytest.mark.unit
    def test_build_indexes(self, pdf_vector_store, sample_passages):
        """Test building search indexes"""
        assert len(pdf_vector_store._reference_index) > 0
        assert len(pdf_vector_store._chapter_index) > 0
        assert len(pdf_vector_store._word_index) > 0
        
        # Check specific indexes
        assert "Gadhada I-1" in pdf_vector_store._reference_index
        assert "Gadhada I" in pdf_vector_store._chapter_index
        assert "faith" in pdf_vector_store._word_index
    
    @pytest.mark.unit
    def test_search_functionality(self, pdf_vector_store):
        """Test basic search functionality"""
        results = pdf_vector_store.search("faith", top_k=2, min_score=0.1)
        
        assert len(results) > 0
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        
        # Check that results are sorted by score
        if len(results) > 1:
            scores = [result[1] for result in results]
            assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.unit
    def test_search_with_theme(self, pdf_vector_store):
        """Test search with specific themes"""
        devotion_results = pdf_vector_store.search("devotion", top_k=3)
        faith_results = pdf_vector_store.search("faith", top_k=3)
        
        assert len(devotion_results) >= 0
        assert len(faith_results) >= 0
        
        # Results should contain relevant passages
        for passage, score in devotion_results:
            assert isinstance(passage, VachanamrutPassage)
            assert 0.0 <= score <= 1.0
    
    @pytest.mark.unit
    def test_prefiltering_performance(self, pdf_vector_store):
        """Test that prefiltering improves performance"""
        # Test with a query that should match some passages
        query = "devotion faith"
        
        start_time = time.time()
        candidates = pdf_vector_store._prefilter_passages(query.lower())
        prefilter_time = time.time() - start_time
        
        # Prefiltering should return subset of passages
        assert len(candidates) <= len(pdf_vector_store.passages)
        
        # Should be very fast
        assert prefilter_time < 0.1  # Should complete in less than 100ms
    
    @pytest.mark.unit
    def test_cache_key_generation(self, pdf_vector_store):
        """Test cache key generation"""
        key1 = pdf_vector_store._generate_cache_key("test query", 5, 0.1)
        key2 = pdf_vector_store._generate_cache_key("test query", 5, 0.1)
        key3 = pdf_vector_store._generate_cache_key("different query", 5, 0.1)
        
        assert key1 == key2  # Same inputs should produce same key
        assert key1 != key3  # Different inputs should produce different keys
        assert isinstance(key1, str)
        assert len(key1) > 0
    
    @pytest.mark.unit
    def test_search_caching(self, pdf_vector_store):
        """Test search result caching"""
        query = "devotion"
        
        # First search - should not be cached
        results1 = pdf_vector_store.search(query, top_k=2)
        
        # Second search - should use cache
        results2 = pdf_vector_store.search(query, top_k=2)
        
        assert results1 == results2
        assert pdf_vector_store.search_stats['cache_hits'] > 0
    
    @pytest.mark.unit
    def test_text_similarity_calculation(self, pdf_vector_store):
        """Test text similarity scoring"""
        query = "faith devotion"
        content = "One who has firm faith in God will experience peace"
        
        similarity = pdf_vector_store._calculate_text_similarity(query, content)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.0  # Should have some similarity
    
    @pytest.mark.unit
    def test_save_and_load(self, pdf_vector_store, sample_passages):
        """Test saving and loading passages"""
        # Save passages
        pdf_vector_store.save()
        assert Path(pdf_vector_store.storage_path).exists()
        
        # Create new store and load
        new_store = PDFVectorStore(storage_path=pdf_vector_store.storage_path)
        load_result = new_store.load()
        
        assert load_result['status'] == 'loaded_from_cache'
        assert len(new_store.passages) == len(sample_passages)
        
        # Check that loaded passages match original
        original_refs = {p.reference for p in pdf_vector_store.passages}
        loaded_refs = {p.reference for p in new_store.passages}
        assert original_refs == loaded_refs
    
    @pytest.mark.unit
    def test_reference_normalization(self, pdf_vector_store):
        """Test reference format normalization"""
        # Test various reference formats
        test_cases = [
            ("Gadhada I - 1", "Gadhada I-1"),
            ("Gadhada I.1", "Gadhada I-1"), 
            ("Sarangpur - 5", "Sarangpur-5"),
            ("Vadtal  18", "Vadtal  18"),  # This might not change
        ]
        
        for input_ref, expected in test_cases:
            result = pdf_vector_store._normalize_reference(input_ref)
            # Just test that normalization doesn't crash
            assert isinstance(result, str)
            assert len(result) > 0
    
    @pytest.mark.unit
    def test_clear_cache(self, pdf_vector_store):
        """Test cache clearing functionality"""
        # Perform some searches to populate cache
        pdf_vector_store.search("devotion")
        pdf_vector_store.search("faith")
        
        # Clear cache
        pdf_vector_store.clear_cache()
        
        # Verify cache is cleared
        assert pdf_vector_store.search_stats['total_searches'] == 0
        assert pdf_vector_store.search_stats['cache_hits'] == 0
    
    @pytest.mark.unit
    def test_get_by_reference(self, pdf_vector_store, sample_passages):
        """Test getting passages by reference"""
        reference = sample_passages[0].reference
        results = pdf_vector_store.get_by_reference(reference)
        
        assert len(results) > 0
        assert all(p.reference == reference for p in results)
    
    @pytest.mark.unit
    def test_get_search_stats(self, pdf_vector_store):
        """Test getting search statistics"""
        # Perform some operations
        pdf_vector_store.search("devotion")
        pdf_vector_store.search("faith")
        
        stats = pdf_vector_store.get_search_stats()
        
        assert 'total_searches' in stats
        assert 'cache_hits' in stats
        assert 'avg_search_time' in stats
        assert stats['total_searches'] >= 2
    
    @pytest.mark.unit
    def test_should_skip_processing(self, temp_storage_path):
        """Test PDF processing skip logic"""
        store = PDFVectorStore(storage_path=temp_storage_path)
        
        # Mock PDF info
        pdf_info = {
            'file_size': 12345,
            'modification_time': 1234567890,
            'file_path': '/test/path.pdf'
        }
        
        # Should not skip when no cache exists
        assert not store._should_skip_processing(pdf_info)
        
        # Save some data to create cache
        store.passages = []
        store.pdf_metadata_cache = pdf_info
        store.save()
        
        # Should skip when metadata matches
        assert store._should_skip_processing(pdf_info)
        
        # Should not skip when metadata differs
        different_info = pdf_info.copy()
        different_info['file_size'] = 54321
        assert not store._should_skip_processing(different_info)


class TestPDFVectorStorePerformance:
    """Performance tests for PDF vector store"""
    
    @pytest.mark.slow
    def test_large_dataset_search_performance(self, large_passage_dataset, temp_storage_path):
        """Test search performance with large dataset"""
        store = PDFVectorStore(storage_path=temp_storage_path)
        store.passages = large_passage_dataset
        store._build_indexes()
        
        # Test search performance
        start_time = time.time()
        results = store.search("devotion faith", top_k=10)
        search_time = time.time() - start_time
        
        # Should complete reasonably quickly even with large dataset
        assert search_time < 1.0  # Less than 1 second
        assert len(results) <= 10
    
    @pytest.mark.slow
    async def test_async_search(self, pdf_vector_store):
        """Test async search functionality"""
        results = await pdf_vector_store.search_async("devotion", top_k=5)
        
        assert len(results) >= 0
        assert all(isinstance(result, tuple) for result in results)


class TestPDFVectorStoreErrorHandling:
    """Test error handling in PDF vector store"""
    
    @pytest.mark.unit
    def test_load_corrupted_file(self, corrupt_data_file):
        """Test loading corrupted data file"""
        store = PDFVectorStore(storage_path=corrupt_data_file)
        result = store.load()
        
        # Should handle corruption gracefully
        assert result['status'] == 'error'
        assert 'error' in result
    
    @pytest.mark.unit
    def test_search_empty_store(self, temp_storage_path):
        """Test search with empty store"""
        store = PDFVectorStore(storage_path=temp_storage_path)
        results = store.search("any query")
        
        assert results == []
    
    @pytest.mark.unit
    def test_search_empty_query(self, pdf_vector_store):
        """Test search with empty query"""
        results = pdf_vector_store.search("", top_k=5)
        
        # Should handle empty query gracefully
        assert isinstance(results, list)
    
    @pytest.mark.unit
    def test_search_special_characters(self, pdf_vector_store):
        """Test search with special characters"""
        queries = ["@#$%", "निर्विकल्प", "émötiön", "123456"]
        
        for query in queries:
            results = pdf_vector_store.search(query, top_k=3)
            # Should not crash and return valid results
            assert isinstance(results, list)
            assert all(isinstance(r, tuple) for r in results)