from typing import List, Dict, Optional, Tuple
import json
import logging
from pathlib import Path

from app.core.pdf_processor import PDFProcessor, VachanamrutDocument
from app.core.embeddings import EmbeddingService
from app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Main retrieval system combining PDF processing and semantic search
    Claude Projects approach: Simple, effective document retrieval
    """
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_service = EmbeddingService()
        self.documents: List[VachanamrutDocument] = []
        self.embedded_docs: List[Dict] = []
        self.is_initialized = False
    
    async def initialize(self, pdf_path: str, force_reprocess: bool = False) -> Dict:
        """Initialize the retrieval system with PDF processing"""
        logger.info("Initializing Document Retriever")
        
        # Check if we have cached embeddings
        embeddings_cache = Path("data/processed/embeddings.json")
        
        if not force_reprocess and embeddings_cache.exists():
            logger.info("Loading cached embeddings")
            try:
                with open(embeddings_cache, 'r') as f:
                    self.embedded_docs = json.load(f)
                self.is_initialized = True
                
                return {
                    'status': 'loaded_from_cache',
                    'total_documents': len(self.embedded_docs),
                    'cache_file': str(embeddings_cache)
                }
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Process PDF and generate embeddings
        logger.info("Processing PDF and generating embeddings")
        
        try:
            # Step 1: Process PDF
            self.documents = self.pdf_processor.process_pdf(pdf_path)
            logger.info(f"Extracted {len(self.documents)} passages")
            
            # Step 2: Generate embeddings
            self.embedded_docs = self.embedding_service.generate_embeddings(self.documents)
            
            # Step 3: Cache embeddings
            embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(embeddings_cache, 'w') as f:
                json.dump(self.embedded_docs, f, indent=2)
            
            self.is_initialized = True
            
            # Get processing stats
            stats = self.pdf_processor.get_document_stats(self.documents)
            model_info = self.embedding_service.get_model_info()
            
            return {
                'status': 'processed_and_cached',
                'pdf_stats': stats,
                'model_info': model_info,
                'total_documents': len(self.embedded_docs),
                'cache_file': str(embeddings_cache)
            }
            
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise
    
    async def search(
        self, 
        query: str, 
        theme_filter: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Search for relevant passages
        Claude Projects approach: Simple semantic search with optional theme filtering
        """
        if not self.is_initialized:
            raise RuntimeError("Retriever not initialized. Call initialize() first.")
        
        top_k = top_k or settings.TOP_K_RESULTS
        
        logger.info(f"Searching for: '{query}' (theme: {theme_filter}, top_k: {top_k})")
        
        try:
            # Filter by theme if specified
            search_docs = self.embedded_docs
            if theme_filter:
                search_docs = [
                    doc for doc in self.embedded_docs 
                    if theme_filter.lower() in [t.lower() for t in doc['metadata'].get('themes', [])]
                ]
                logger.info(f"Filtered to {len(search_docs)} documents for theme '{theme_filter}'")
            
            if not search_docs:
                logger.warning("No documents found matching theme filter")
                return []
            
            # Perform semantic search
            similar_docs = self.embedding_service.find_similar_documents(
                query=query,
                embedded_docs=search_docs,
                top_k=top_k,
                min_similarity=0.3  # Reasonable threshold
            )
            
            # Format results for API response
            results = []
            for doc in similar_docs:
                results.append({
                    'content': doc['metadata']['content'],
                    'reference': doc['metadata']['reference'],
                    'page_number': doc['metadata']['page_number'],
                    'themes': doc['metadata']['themes'],
                    'similarity_score': doc['similarity_score'],
                    'relevance_score': doc['similarity_score']  # Alias for compatibility
                })
            
            logger.info(f"Found {len(results)} relevant passages")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def get_themes_summary(self) -> Dict[str, int]:
        """Get count of documents per theme"""
        if not self.is_initialized:
            return {}
        
        theme_counts = {}
        for doc in self.embedded_docs:
            for theme in doc['metadata'].get('themes', []):
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        return dict(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_references_summary(self) -> List[str]:
        """Get list of all unique references"""
        if not self.is_initialized:
            return []
        
        references = set()
        for doc in self.embedded_docs:
            references.add(doc['metadata']['reference'])
        
        return sorted(references)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'initialized': self.is_initialized,
            'total_documents': len(self.embedded_docs),
            'total_themes': len(self.get_themes_summary()),
            'total_references': len(self.get_references_summary()),
            'embedding_model': self.embedding_service.get_model_info(),
            'processor_config': {
                'chunk_size': self.pdf_processor.chunk_size,
                'chunk_overlap': self.pdf_processor.chunk_overlap
            }
        }

# Global instance for dependency injection
_retriever_instance: Optional[DocumentRetriever] = None

def get_document_retriever() -> DocumentRetriever:
    """Dependency injection for DocumentRetriever"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = DocumentRetriever()
    return _retriever_instance