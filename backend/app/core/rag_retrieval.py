"""
RAG (Retrieval Augmented Generation) System

This module implements the full RAG pipeline for the Vachanamrut companion,
similar to Claude Projects but with PDF-based knowledge retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio

from app.core.pdf_processor import PDFProcessor, VachanamrutDocument
from app.core.embeddings_store import VachanamrutEmbeddingsStore, SearchResult
from app.core.lightweight_retrieval import LightweightRetriever
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    query: str
    retrieved_passages: List[SearchResult]
    context_text: str
    themes_found: List[str]
    total_documents_searched: int

class VachanamrutRAGRetriever:
    """
    Main RAG retrieval system for Vachanamrut documents.
    Handles PDF processing, embedding generation, and semantic search.
    """
    
    def __init__(self):
        """Initialize the RAG retriever"""
        self.pdf_processor = PDFProcessor()
        self.embeddings_store = VachanamrutEmbeddingsStore()
        self._initialized = False
        
        logger.info("RAG Retriever initialized")
    
    async def initialize(self) -> None:
        """Initialize the RAG system"""
        if self._initialized:
            return
        
        logger.info("Initializing RAG system...")
        
        # Try to load existing embeddings store
        if self.embeddings_store.load_store():
            logger.info("Loaded existing embeddings store")
            self._initialized = True
            return
        
        # If no existing store, check for PDF files to process
        await self._process_available_pdfs()
        
        self._initialized = True
        logger.info("RAG system initialization complete")
    
    async def _process_available_pdfs(self) -> None:
        """Process any PDF files found in the data directory"""
        data_dir = Path("data/raw")
        pdf_files = list(data_dir.glob("*.pdf")) if data_dir.exists() else []
        
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            for pdf_path in pdf_files:
                try:
                    await self.ingest_pdf(pdf_path)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
        else:
            logger.info("No PDF files found, using sample documents")
            await self._create_sample_documents()
    
    async def ingest_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Ingest a PDF file into the RAG system.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Processing statistics
        """
        logger.info(f"Ingesting PDF: {pdf_path.name}")
        
        # Process PDF into documents
        documents = self.pdf_processor.process_pdf(str(pdf_path))
        
        if not documents:
            raise ValueError(f"No content extracted from {pdf_path}")
        
        # Add documents to embeddings store
        self.embeddings_store.add_documents(documents)
        
        # Get processing stats
        stats = self.pdf_processor.get_document_stats(documents)
        stats['pdf_file'] = pdf_path.name
        
        logger.info(f"Successfully ingested {len(documents)} documents from {pdf_path.name}")
        return stats
    
    async def _create_sample_documents(self) -> None:
        """Create sample documents for testing without PDFs"""
        logger.info("Creating sample documents for testing")
        
        sample_content = [
            {
                "content": "One who has firm faith in God and His Sant never experiences any difficulty, regardless of whatever severe calamities he may encounter. Why? Because he is confident that God is the all-doer and the cause of all causes; and also that God is extremely loving towards His devotees. Therefore, such a person never experiences any difficulties.",
                "reference": "Gadhada I-1",
                "page_number": 15,
                "themes": ["faith", "devotion", "surrender", "trust"]
            },
            {
                "content": "A person who has love for God should not allow his mind to become attached to any object other than God. Even if his mind does get attached to some other object, he should understand that attachment to be a flaw and should attempt to remove it. In this manner, he should maintain love for God alone.",
                "reference": "Gadhada I-2",
                "page_number": 16,
                "themes": ["devotion", "detachment", "love", "mind"]
            },
            {
                "content": "The key to spiritual progress is constant remembrance of God. One should always keep God in mind during all activities - whether eating, drinking, walking, or resting. This continuous awareness leads to realization of God's presence in all situations.",
                "reference": "Sarangpur-5",
                "page_number": 87,
                "themes": ["remembrance", "spiritual practice", "awareness", "devotion"]
            },
            {
                "content": "Just as a person becomes brahmarup by contemplating on Parabrahma, similarly, by contemplating on the divine actions of God, one develops the same qualities as God. This is because whatever a person contemplates upon, he develops the qualities of that object.",
                "reference": "Vadtal-18",
                "page_number": 203,
                "themes": ["contemplation", "divine qualities", "spiritual development", "brahmarup"]
            },
            {
                "content": "Selfless service to God and His devotees is the highest form of worship. When one serves without expectation of reward, the mind becomes purified and develops humility. Such service is more valuable than elaborate rituals or ceremonies.",
                "reference": "Ahmedabad-3",
                "page_number": 95,
                "themes": ["service", "seva", "humility", "worship", "selflessness"]
            },
            {
                "content": "True knowledge means to understand the difference between the body and the soul, and to realize that one's true identity is the soul, not the body. This understanding leads to detachment from worldly pleasures and attachment to God.",
                "reference": "Loya-7",
                "page_number": 142,
                "themes": ["knowledge", "soul", "body", "identity", "detachment"]
            },
            {
                "content": "The company of satsang is extremely important for spiritual progress. In the presence of true devotees and saints, one's mind naturally becomes purified, and negative tendencies are gradually eliminated. Therefore, one should always seek the company of those who are devoted to God.",
                "reference": "Gadhada II-13",
                "page_number": 178,
                "themes": ["satsang", "company", "spiritual progress", "purification"]
            },
            {
                "content": "Dharma means to act according to one's prescribed duties while maintaining devotion to God. One should perform all actions as worship to God, without attachment to the fruits of action. This transforms ordinary activities into spiritual practice.",
                "reference": "Kariyani-5",
                "page_number": 234,
                "themes": ["dharma", "duty", "worship", "detachment", "spiritual practice"]
            }
        ]
        
        # Convert to VachanamrutDocument objects
        documents = []
        for i, item in enumerate(sample_content):
            doc = VachanamrutDocument(
                content=item["content"],
                reference=item["reference"],
                page_number=item["page_number"],
                chunk_index=0,
                themes=item["themes"]
            )
            documents.append(doc)
        
        # Add to embeddings store
        self.embeddings_store.add_documents(documents)
        logger.info(f"Created {len(documents)} sample documents")
    
    async def retrieve_context(
        self, 
        query: str, 
        theme_filter: Optional[str] = None,
        top_k: int = None
    ) -> RetrievalResult:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            theme_filter: Optional theme to filter by
            top_k: Number of results to retrieve
            
        Returns:
            Retrieval result with context and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        top_k = top_k or settings.TOP_K_RESULTS
        
        logger.info(f"Retrieving context for query: '{query[:50]}...'")
        
        # Filter by theme if specified
        search_documents = self.embeddings_store.documents
        if theme_filter:
            search_documents = [
                doc for doc in search_documents 
                if theme_filter.lower() in [t.lower() for t in doc.themes]
            ]
            logger.info(f"Filtered to {len(search_documents)} documents by theme: {theme_filter}")
        
        # Perform semantic search
        if theme_filter:
            # For theme filtering, we need to search within the filtered set
            # This is a limitation we'll work with for now
            search_results = self.embeddings_store.search_similar(query, top_k * 2)
            # Filter results by theme
            filtered_results = [
                result for result in search_results
                if theme_filter.lower() in [t.lower() for t in result.document.themes]
            ][:top_k]
        else:
            search_results = self.embeddings_store.search_similar(query, top_k)
            filtered_results = search_results
        
        # Build context text
        context_passages = []
        themes_found = set()
        
        for result in filtered_results:
            doc = result.document
            context_passages.append(f"[{doc.reference}] {doc.content}")
            themes_found.update(doc.themes)
        
        context_text = "\n\n".join(context_passages)
        
        result = RetrievalResult(
            query=query,
            retrieved_passages=filtered_results,
            context_text=context_text,
            themes_found=sorted(list(themes_found)),
            total_documents_searched=len(search_documents)
        )
        
        logger.info(f"Retrieved {len(filtered_results)} passages, "
                   f"found themes: {result.themes_found}")
        
        return result
    
    def get_available_themes(self) -> List[str]:
        """Get all available themes in the knowledge base"""
        if not self._initialized:
            return []
        return self.embeddings_store.get_themes()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        stats = self.embeddings_store.get_stats()
        stats.update({
            "status": "initialized",
            "is_initialized": self._initialized,
            "available_themes": self.get_available_themes()
        })
        
        return stats
    
    async def add_pdf_documents(self, pdf_path: str) -> Dict[str, Any]:
        """
        Add documents from a new PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Processing statistics
        """
        return await self.ingest_pdf(Path(pdf_path))

# Global RAG retriever instance
rag_retriever = VachanamrutRAGRetriever()

async def get_rag_retriever() -> VachanamrutRAGRetriever:
    """Get the global RAG retriever instance"""
    if not rag_retriever._initialized:
        await rag_retriever.initialize()
    return rag_retriever