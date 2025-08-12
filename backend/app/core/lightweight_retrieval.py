from typing import List, Dict, Optional
import logging

from app.core.simple_vector_store import BasicVectorStore, create_sample_store
from app.core.simple_config import settings

logger = logging.getLogger(__name__)

class LightweightRetriever:
    """
    Lightweight document retriever with no external dependencies
    Claude Projects approach: Simple but effective spiritual guidance
    """
    
    def __init__(self):
        self.store = BasicVectorStore()
        self.is_initialized = False
    
    async def initialize(self, pdf_path: Optional[str] = None, force_reprocess: bool = False) -> Dict:
        """Initialize with sample spiritual content"""
        logger.info("Initializing Lightweight Retriever")
        
        # Try to load from cache first
        if not force_reprocess and self.store.load():
            self.is_initialized = True
            return {
                'status': 'loaded_from_cache',
                'total_documents': len(self.store.documents),
                'source': 'cached_data'
            }
        
        # Create sample store
        logger.info("Creating sample store with Vachanamrut content")
        self.store = create_sample_store()
        
        # Save for next time
        self.store.save()
        
        self.is_initialized = True
        
        stats = self.store.get_stats()
        
        return {
            'status': 'initialized_with_sample_data',
            'total_documents': stats['total_documents'],
            'unique_themes': stats['unique_themes'],
            'unique_references': stats['unique_references'],
            'theme_distribution': stats['theme_distribution']
        }
    
    async def search(
        self, 
        query: str, 
        theme_filter: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """Search for relevant passages"""
        
        if not self.is_initialized:
            await self.initialize()
        
        top_k = top_k or settings.TOP_K_RESULTS
        
        logger.info(f"Searching for: '{query}' (theme: {theme_filter}, top_k: {top_k})")
        
        try:
            # Search using simple text similarity
            results = self.store.search(
                query=query,
                top_k=top_k,
                theme_filter=theme_filter
            )
            
            # Format results for API response
            formatted_results = []
            for doc, similarity in results:
                formatted_results.append({
                    'content': doc.content,
                    'reference': doc.reference,
                    'page_number': doc.page_number,
                    'themes': doc.themes,
                    'similarity_score': similarity,
                    'relevance_score': similarity  # Alias for compatibility
                })
            
            logger.info(f"Found {len(formatted_results)} relevant passages")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def get_themes_summary(self) -> Dict[str, int]:
        """Get count of documents per theme"""
        if not self.is_initialized:
            return {}
        
        stats = self.store.get_stats()
        return stats.get('theme_distribution', {})
    
    def get_references_summary(self) -> List[str]:
        """Get list of all unique references"""
        if not self.is_initialized:
            return []
        
        references = set()
        for doc in self.store.documents:
            references.add(doc.reference)
        
        return sorted(references)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        stats = self.store.get_stats() if self.is_initialized else {}
        
        return {
            'initialized': self.is_initialized,
            'total_documents': stats.get('total_documents', 0),
            'unique_themes': stats.get('unique_themes', 0),
            'unique_references': stats.get('unique_references', 0),
            'theme_distribution': stats.get('theme_distribution', {}),
            'search_method': 'text_similarity',
            'storage_type': 'BasicVectorStore'
        }
    
    def get_theme_documents(self, theme: str) -> List[Dict]:
        """Get all documents for a specific theme"""
        if not self.is_initialized:
            return []
        
        docs = self.store.get_by_theme(theme)
        return [
            {
                'content': doc.content,
                'reference': doc.reference,
                'page_number': doc.page_number,
                'themes': doc.themes
            }
            for doc in docs
        ]

# Global instance for dependency injection
_lightweight_retriever_instance: Optional[LightweightRetriever] = None

def get_lightweight_retriever() -> LightweightRetriever:
    """Dependency injection for LightweightRetriever"""
    global _lightweight_retriever_instance
    if _lightweight_retriever_instance is None:
        _lightweight_retriever_instance = LightweightRetriever()
    return _lightweight_retriever_instance