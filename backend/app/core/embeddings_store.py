"""
Vector Embeddings and Storage System

This module handles embedding generation and similarity search for the RAG system.
Similar to Claude Projects approach but with full PDF support.
"""

import logging
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import hashlib

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import openai

from app.core.config import settings
from app.core.pdf_processor import VachanamrutDocument

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result from semantic search"""
    document: VachanamrutDocument
    similarity_score: float
    rank: int

class VachanamrutEmbeddingsStore:
    """
    Vector storage and similarity search for Vachanamrut documents.
    Supports both OpenAI embeddings and fallback TF-IDF similarity.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the embeddings store.
        
        Args:
            storage_path: Path to store embeddings and index
        """
        self.storage_path = storage_path or Path("data/processed/embeddings_store.pkl")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Document storage
        self.documents: List[VachanamrutDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Fallback TF-IDF system (works without OpenAI API)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = None
        
        # OpenAI client (if API key available)
        self.openai_client = None
        if settings.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for embeddings")
        else:
            logger.info("No OpenAI API key - using TF-IDF fallback")
    
    def add_documents(self, documents: List[VachanamrutDocument]) -> None:
        """
        Add documents to the store and generate embeddings.
        
        Args:
            documents: List of processed Vachanamrut documents
        """
        logger.info(f"Adding {len(documents)} documents to embeddings store")
        
        self.documents.extend(documents)
        
        # Generate embeddings
        if self.openai_client:
            self._generate_openai_embeddings(documents)
        else:
            self._generate_tfidf_embeddings()
        
        # Save to disk
        self.save_store()
        
        logger.info(f"Store now contains {len(self.documents)} documents")
    
    def _generate_openai_embeddings(self, new_documents: List[VachanamrutDocument]) -> None:
        """Generate OpenAI embeddings for documents"""
        logger.info("Generating OpenAI embeddings")
        
        # Get text content for new documents
        texts = [doc.content for doc in new_documents]
        
        try:
            # Generate embeddings in batches
            batch_size = 100  # OpenAI rate limits
            new_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"  # Latest embedding model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                new_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
            
            # Combine with existing embeddings
            new_embeddings_array = np.array(new_embeddings)
            
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
            else:
                self.embeddings = new_embeddings_array
                
            logger.info(f"Generated {len(new_embeddings)} OpenAI embeddings")
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            logger.info("Falling back to TF-IDF embeddings")
            self._generate_tfidf_embeddings()
    
    def _generate_tfidf_embeddings(self) -> None:
        """Generate TF-IDF based embeddings (fallback method)"""
        logger.info("Generating TF-IDF embeddings")
        
        if not self.documents:
            return
        
        # Extract text content
        texts = [doc.content for doc in self.documents]
        
        # Fit TF-IDF vectorizer and transform texts
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info(f"Generated TF-IDF matrix: {self.tfidf_matrix.shape}")
    
    def search_similar(
        self, 
        query: str, 
        top_k: int = 5, 
        min_similarity: float = 0.1
    ) -> List[SearchResult]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results ordered by similarity
        """
        if not self.documents:
            logger.warning("No documents in store")
            return []
        
        if self.embeddings is not None:
            return self._search_with_openai_embeddings(query, top_k, min_similarity)
        else:
            return self._search_with_tfidf(query, top_k, min_similarity)
    
    def _search_with_openai_embeddings(
        self, 
        query: str, 
        top_k: int, 
        min_similarity: float
    ) -> List[SearchResult]:
        """Search using OpenAI embeddings"""
        try:
            # Generate query embedding
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            
            query_embedding = np.array([response.data[0].embedding])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(top_indices):
                similarity = similarities[idx]
                if similarity >= min_similarity:
                    results.append(SearchResult(
                        document=self.documents[idx],
                        similarity_score=float(similarity),
                        rank=rank + 1
                    ))
            
            logger.info(f"Found {len(results)} results using OpenAI embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error in OpenAI search: {e}")
            return self._search_with_tfidf(query, top_k, min_similarity)
    
    def _search_with_tfidf(
        self, 
        query: str, 
        top_k: int, 
        min_similarity: float
    ) -> List[SearchResult]:
        """Search using TF-IDF similarity"""
        if self.tfidf_matrix is None:
            logger.error("TF-IDF matrix not initialized")
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            similarity = similarities[idx]
            if similarity >= min_similarity:
                results.append(SearchResult(
                    document=self.documents[idx],
                    similarity_score=float(similarity),
                    rank=rank + 1
                ))
        
        logger.info(f"Found {len(results)} results using TF-IDF")
        return results
    
    def get_themes(self) -> List[str]:
        """Get all unique themes from stored documents"""
        themes = set()
        for doc in self.documents:
            themes.update(doc.themes)
        return sorted(list(themes))
    
    def filter_by_theme(self, theme: str) -> List[VachanamrutDocument]:
        """Filter documents by theme"""
        return [doc for doc in self.documents if theme.lower() in [t.lower() for t in doc.themes]]
    
    def save_store(self) -> None:
        """Save the store to disk"""
        try:
            store_data = {
                'documents': [asdict(doc) for doc in self.documents],
                'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
                'tfidf_vocabulary': self.tfidf_vectorizer.vocabulary_ if hasattr(self.tfidf_vectorizer, 'vocabulary_') else None,
                'tfidf_idf': self.tfidf_vectorizer.idf_.tolist() if hasattr(self.tfidf_vectorizer, 'idf_') else None,
                'metadata': {
                    'total_documents': len(self.documents),
                    'embedding_model': 'text-embedding-3-small' if self.embeddings is not None else 'tfidf',
                    'themes': self.get_themes()
                }
            }
            
            with open(self.storage_path, 'wb') as f:
                pickle.dump(store_data, f)
            
            logger.info(f"Store saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving store: {e}")
    
    def load_store(self) -> bool:
        """
        Load the store from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if not self.storage_path.exists():
                logger.info("No existing store found")
                return False
            
            with open(self.storage_path, 'rb') as f:
                store_data = pickle.load(f)
            
            # Reconstruct documents
            self.documents = []
            for doc_data in store_data.get('documents', []):
                doc = VachanamrutDocument(
                    content=doc_data['content'],
                    reference=doc_data['reference'],
                    page_number=doc_data['page_number'],
                    chunk_index=doc_data['chunk_index'],
                    themes=doc_data['themes']
                )
                self.documents.append(doc)
            
            # Load embeddings
            if store_data.get('embeddings'):
                self.embeddings = np.array(store_data['embeddings'])
            
            # Restore TF-IDF if available
            if store_data.get('tfidf_vocabulary') and store_data.get('tfidf_idf'):
                self.tfidf_vectorizer.vocabulary_ = store_data['tfidf_vocabulary']
                self.tfidf_vectorizer.idf_ = np.array(store_data['tfidf_idf'])
                
                # Rebuild TF-IDF matrix
                texts = [doc.content for doc in self.documents]
                self.tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
            metadata = store_data.get('metadata', {})
            logger.info(f"Loaded store: {metadata.get('total_documents', 0)} documents, "
                       f"model: {metadata.get('embedding_model', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store"""
        themes_count = {}
        for doc in self.documents:
            for theme in doc.themes:
                themes_count[theme] = themes_count.get(theme, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'has_openai_embeddings': self.embeddings is not None,
            'has_tfidf_embeddings': self.tfidf_matrix is not None,
            'unique_themes': len(self.get_themes()),
            'themes_distribution': themes_count,
            'average_content_length': sum(len(doc.content) for doc in self.documents) // len(self.documents) if self.documents else 0
        }

# Global instance
embeddings_store = VachanamrutEmbeddingsStore()