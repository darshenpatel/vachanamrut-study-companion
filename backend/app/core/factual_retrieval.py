from typing import List, Dict, Optional
import logging
from pathlib import Path

from app.core.pdf_retrieval import get_pdf_store, VachanamrutPassage
from app.core.config import settings

logger = logging.getLogger(__name__)

class FactualRetriever:
    """
    Simplified factual retrieval system focused on PDF content
    No themes, just direct factual retrieval from Vachanamrut text
    """
    
    def __init__(self):
        self.pdf_store = get_pdf_store()
        self.is_initialized = False
        self.pdf_path = "data/raw/TheVachanamrut-searchable.pdf"
    
    async def initialize(self, force_reprocess: bool = False) -> Dict:
        """Initialize by processing the PDF"""
        logger.info("Initializing Factual Retriever with PDF processing")
        
        try:
            # Prefer loading cached extracted passages if present, even if the raw PDF
            # isn't available in the current environment (common in CI/smoke tests).
            if not Path(self.pdf_path).exists():
                logger.warning(f"PDF not found at {self.pdf_path}. Trying to load cached PDF store instead.")
                cached = self.pdf_store.load()
                if cached.get("status") == "loaded_from_cache":
                    self.is_initialized = True
                    stats = self.pdf_store.get_stats()
                    cached.update(stats)
                    cached["message"] = "Loaded cached PDF store; raw PDF not present."
                    logger.info(f"Factual retriever initialized from cache: {cached.get('total_passages', 0)} passages")
                    return cached

                logger.error(f"PDF not found at {self.pdf_path} and no cached PDF store available")
                return {
                    'status': 'error',
                    'message': f'PDF file not found at {self.pdf_path} and no cached store could be loaded'
                }
            
            # Process the PDF
            result = self.pdf_store.process_pdf(self.pdf_path, force_reprocess)
            self.is_initialized = True
            
            stats = self.pdf_store.get_stats()
            result.update(stats)
            
            logger.info(f"Factual retriever initialized: {result['total_passages']} passages")
            return result
            
        except Exception as e:
            logger.error(f"Error initializing factual retriever: {e}")
            return {
                'status': 'error', 
                'message': str(e)
            }
    
    async def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for factual information from the Vachanamrut"""
        
        if not self.is_initialized:
            await self.initialize()
        
        top_k = top_k or settings.TOP_K_RESULTS
        
        logger.info(f"Searching for factual information: '{query}' (top_k: {top_k})")
        
        try:
            # Search using PDF store
            results = self.pdf_store.search(query, top_k)
            
            # Format results for API response
            formatted_results = []
            for passage, similarity in results:
                formatted_results.append({
                    'content': passage.content,
                    'reference': passage.reference,
                    'page_number': passage.page_number,
                    'chapter': passage.chapter,
                    'similarity_score': similarity,
                    'relevance_score': similarity  # Alias for compatibility
                })
            
            logger.info(f"Found {len(formatted_results)} relevant passages")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during factual search: {e}")
            return []
    
    def get_passage_by_reference(self, reference: str) -> List[Dict]:
        """Get specific passage by Vachanamrut reference (e.g., 'Gadhada I-1')"""
        if not self.is_initialized:
            return []
        
        passages = self.pdf_store.get_by_reference(reference)
        return [
            {
                'content': p.content,
                'reference': p.reference,
                'page_number': p.page_number,
                'chapter': p.chapter
            }
            for p in passages
        ]
    
    def get_chapter_content(self, chapter: str) -> List[Dict]:
        """Get all passages from a specific chapter"""
        if not self.is_initialized:
            return []
        
        passages = self.pdf_store.get_by_chapter(chapter)
        return [
            {
                'content': p.content,
                'reference': p.reference,
                'page_number': p.page_number,
                'chapter': p.chapter
            }
            for p in passages
        ]
    
    def get_system_status(self) -> Dict:
        """Get current system status and statistics"""
        if not self.is_initialized:
            return {
                'initialized': False,
                'total_passages': 0,
                'method': 'pdf_text_extraction'
            }
        
        stats = self.pdf_store.get_stats()
        
        return {
            'initialized': self.is_initialized,
            'total_passages': stats.get('total_passages', 0),
            'unique_references': stats.get('unique_references', 0),
            'chapters': stats.get('chapters', {}),
            'average_passage_length': int(stats.get('average_length', 0)),
            'method': 'pdf_text_extraction',
            'storage_type': 'PDFVectorStore'
        }
    
    def search_specific_topics(self, topic_keywords: List[str], top_k: int = 10) -> List[Dict]:
        """Search for specific topics using multiple keywords"""
        if not self.is_initialized:
            return []
        
        # Combine keywords into a search query
        query = " ".join(topic_keywords)
        return self.search(query, top_k)

# Global instance for dependency injection
_factual_retriever_instance: Optional[FactualRetriever] = None

def get_factual_retriever() -> FactualRetriever:
    """Dependency injection for FactualRetriever"""
    global _factual_retriever_instance
    if _factual_retriever_instance is None:
        _factual_retriever_instance = FactualRetriever()
    return _factual_retriever_instance