from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

from app.core.config import settings
from app.core.pdf_processor import VachanamrutDocument

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Generate embeddings for semantic search
    Claude Projects approach: Simple, efficient embedding generation
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, documents: List[VachanamrutDocument]) -> List[Dict]:
        """
        Generate embeddings for all documents
        Returns list of dicts with document info and embedding
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        # Prepare texts for embedding
        texts = []
        doc_metadata = []
        
        for doc in documents:
            # Combine reference and content for better context
            text = f"{doc.reference}: {doc.content}"
            texts.append(text)
            
            doc_metadata.append({
                'chunk_id': doc.chunk_id,
                'reference': doc.reference,
                'page_number': doc.page_number,
                'themes': doc.themes,
                'content_length': len(doc.content)
            })
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Combine embeddings with metadata
            embedded_docs = []
            for i, (embedding, metadata) in enumerate(zip(embeddings, doc_metadata)):
                embedded_docs.append({
                    'id': metadata['chunk_id'],
                    'embedding': embedding.tolist(),  # Convert to list for JSON serialization
                    'metadata': {
                        **metadata,
                        'content': documents[i].content,  # Include original content
                        'embedding_model': self.model_name
                    }
                })
            
            logger.info(f"Generated {len(embedded_docs)} embeddings")
            return embedded_docs
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        doc_embeddings: List[np.ndarray]
    ) -> List[float]:
        """Compute cosine similarity between query and document embeddings"""
        similarities = []
        
        for doc_embedding in doc_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(float(similarity))
        
        return similarities
    
    def find_similar_documents(
        self, 
        query: str, 
        embedded_docs: List[Dict], 
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[Dict]:
        """
        Find most similar documents to query
        Returns sorted list of documents with similarity scores
        """
        query_embedding = self.generate_query_embedding(query)
        
        scored_docs = []
        for doc in embedded_docs:
            doc_embedding = np.array(doc['embedding'])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            if similarity >= min_similarity:
                scored_docs.append({
                    **doc,
                    'similarity_score': float(similarity)
                })
        
        # Sort by similarity score
        scored_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return scored_docs[:top_k]
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        if not self.model:
            return {'status': 'not_loaded'}
        
        return {
            'model_name': self.model_name,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'status': 'loaded'
        }