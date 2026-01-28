"""
Unified Semantic Retrieval Service
Integrates sentence-transformers + FAISS for production-quality semantic search
"""

from typing import List, Dict, Optional
import logging
from pathlib import Path
import json

from app.core.config import settings
from app.core.semantic_search_engine import (
    VachanamrutSemanticSearch, 
    SearchQuery,
    SearchResult,
    DEPENDENCIES_AVAILABLE
)

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Production-ready semantic retrieval service
    Uses sentence-transformers + FAISS for high-quality search
    Falls back to text-based search if dependencies unavailable
    """
    
    def __init__(self):
        self.search_engine: Optional[VachanamrutSemanticSearch] = None
        self.is_initialized = False
        self.data_path = "data/processed/pdf_store.json"
        self.index_path = "data/processed/semantic_index"
    
    async def initialize(self, force_rebuild: bool = False) -> Dict:
        """Initialize the semantic search engine"""
        logger.info("Initializing Semantic Retriever...")
        
        try:
            # Check if we have a saved index
            index_dir = Path(self.index_path)
            
            if not force_rebuild and (index_dir / "search_data.json").exists():
                logger.info("Loading saved semantic index...")
                self.search_engine = VachanamrutSemanticSearch(
                    model_name=settings.EMBEDDING_MODEL
                )
                if self.search_engine.load_index(self.index_path):
                    self.is_initialized = True
                    return {
                        'status': 'loaded_from_cache',
                        'stats': self.search_engine.get_search_stats()
                    }
            
            # Load discourse data and build index
            data_file = Path(self.data_path)
            if not data_file.exists():
                # Try alternative path
                alt_path = Path("data/processed/working_vachanamrut.json")
                if alt_path.exists():
                    data_file = alt_path
                else:
                    logger.error(f"No discourse data found at {self.data_path}")
                    return {'status': 'error', 'message': 'No discourse data found'}
            
            # Load discourse data
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            discourses = data.get('passages', [])
            if not discourses:
                logger.error("No passages found in data file")
                return {'status': 'error', 'message': 'No passages in data'}
            
            # Initialize search engine
            self.search_engine = VachanamrutSemanticSearch(
                model_name=settings.EMBEDDING_MODEL
            )
            
            if self.search_engine.initialize(discourses):
                self.is_initialized = True
                
                # Save index for future use
                index_dir.mkdir(parents=True, exist_ok=True)
                self.search_engine.save_index(self.index_path)
                
                return {
                    'status': 'initialized',
                    'stats': self.search_engine.get_search_stats()
                }
            else:
                return {'status': 'error', 'message': 'Failed to initialize search engine'}
                
        except Exception as e:
            logger.error(f"Error initializing semantic retriever: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def search(
        self, 
        query: str, 
        top_k: int = None,
        theme_filter: Optional[str] = None,
        min_similarity: float = 0.25
    ) -> List[Dict]:
        """
        Search for relevant passages using semantic similarity
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.search_engine:
            logger.warning("Search engine not available")
            return []
        
        top_k = top_k or settings.TOP_K_RESULTS
        
        logger.info(f"Semantic search: '{query[:50]}...' (top_k={top_k}, theme={theme_filter})")
        
        try:
            # Create search query
            search_query = SearchQuery(
                query_text=query,
                theme_filter=theme_filter,
                max_results=top_k,
                min_similarity=min_similarity,
                search_mode="semantic" if DEPENDENCIES_AVAILABLE else "keyword"
            )
            
            # Perform search
            results = self.search_engine.search(search_query)
            
            # Format results for API
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.content,
                    'reference': result.reference,
                    'page_number': result.page_number,
                    'chapter': result.chapter,
                    'similarity_score': result.combined_score,
                    'semantic_score': result.semantic_score,
                    'relevance_score': result.combined_score,
                    'themes': result.spiritual_themes,
                    'explanation': result.explanation
                })
            
            logger.info(f"Found {len(formatted_results)} semantic results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def get_contextual_passages(
        self, 
        query: str, 
        theme: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Get contextual passages formatted for LLM integration
        """
        if not self.is_initialized or not self.search_engine:
            return []
        
        return self.search_engine.get_contextual_passages(query, theme, max_results)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'initialized': self.is_initialized,
            'method': 'semantic' if DEPENDENCIES_AVAILABLE else 'text_similarity',
            'dependencies_available': DEPENDENCIES_AVAILABLE,
            'model': settings.EMBEDDING_MODEL
        }
        
        if self.search_engine:
            status['search_stats'] = self.search_engine.get_search_stats()
        
        return status


# Global instance for dependency injection
_semantic_retriever_instance: Optional[SemanticRetriever] = None


def get_semantic_retriever() -> SemanticRetriever:
    """Dependency injection for SemanticRetriever"""
    global _semantic_retriever_instance
    if _semantic_retriever_instance is None:
        _semantic_retriever_instance = SemanticRetriever()
    return _semantic_retriever_instance

