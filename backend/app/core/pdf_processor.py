from typing import List, Dict, Optional
import re
import logging
from pathlib import Path
import hashlib
from dataclasses import dataclass

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class VachanamrutDocument:
    """Represents a processed document chunk with metadata"""
    content: str
    reference: str
    page_number: int
    chunk_index: int = 0
    themes: Optional[List[str]] = None

    def __post_init__(self):
        if self.themes is None:
            self.themes = []
        self.chunk_id = f"{self.reference}_chunk_{self.chunk_index}"

class PDFProcessor:
    """Process Vachanamrut PDF following Claude Projects approach - simple and direct"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Initialize text splitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            # Separators optimized for spiritual texts
            separators=[
                "\n\n\n",  # Major section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                "।",       # Sanskrit/Gujarati sentence endings
                ".",       # English sentence endings
                "!",       # Exclamations
                "?",       # Questions
                " ",       # Word boundaries
                ""         # Character-level fallback
            ]
        )
        
    def process_pdf(self, pdf_path: str) -> List[VachanamrutDocument]:
        """
        Process PDF and extract structured documents with references
        Claude Projects approach: Focus on extracting meaningful passages with proper citations
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        documents = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Opening PDF with {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # Clean and process the extracted text
                        cleaned_text = self._clean_text(text)
                        
                        # Extract Vachanamrut passages from page
                        passages = self._extract_passages(cleaned_text, page_num)
                        documents.extend(passages)
                    else:
                        logger.warning(f"No text found on page {page_num}")
                        
            logger.info(f"Processed {len(documents)} passages from {len(pdf.pages)} pages")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
            
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted PDF text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Fix common PDF extraction issues
        text = text.replace("- ", "")  # Remove hyphenation
        text = text.replace("  ", " ")  # Multiple spaces
        
        # Preserve Sanskrit/Gujarati characters
        return text.strip()
    
    def _extract_passages(self, text: str, page_number: int) -> List[VachanamrutDocument]:
        """
        Extract individual Vachanamrut passages with their references
        Pattern: Looks for "Gadhada I-1", "Sarangpur-5", etc.
        """
        documents = []
        
        # Common Vachanamrut location patterns
        reference_pattern = r'([A-Za-z]+(?:\s+[IVX]+)?)-(\d+)'
        
        # Split text into potential passages
        passages = re.split(reference_pattern, text)
        
        current_reference = None
        current_content = ""
        
        for i, segment in enumerate(passages):
            if re.match(reference_pattern, segment):
                # This is a reference
                if current_reference and current_content.strip():
                    # Save previous passage
                    doc = self._create_document(
                        current_content.strip(), 
                        current_reference, 
                        page_number
                    )
                    if doc:
                        documents.append(doc)
                
                current_reference = segment
                current_content = ""
            else:
                # This is content
                current_content += segment
        
        # Don't forget the last passage
        if current_reference and current_content.strip():
            doc = self._create_document(
                current_content.strip(), 
                current_reference, 
                page_number
            )
            if doc:
                documents.append(doc)
        
        # If no clear references found, chunk the content
        if not documents and text.strip():
            chunks = self._chunk_text(text.strip())
            for i, chunk in enumerate(chunks):
                doc = VachanamrutDocument(
                    content=chunk,
                    reference=f"Page-{page_number}",
                    page_number=page_number,
                    chunk_index=i
                )
                documents.append(doc)
        
        return documents
    
    def _create_document(
        self, 
        content: str, 
        reference: str, 
        page_number: int
    ) -> Optional[VachanamrutDocument]:
        """Create a document with theme detection"""
        
        if len(content) < 50:  # Skip very short content
            return None
            
        # Basic theme detection based on keywords
        themes = self._detect_themes(content)
        
        # If content is too long, chunk it
        if len(content) > self.chunk_size:
            chunks = self._chunk_text(content)
            # Return first chunk with reference, others will be handled separately
            return VachanamrutDocument(
                content=chunks[0],
                reference=reference,
                page_number=page_number,
                chunk_index=0,
                themes=themes
            )
        
        return VachanamrutDocument(
            content=content,
            reference=reference,
            page_number=page_number,
            chunk_index=0,
            themes=themes
        )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Use LangChain text splitter for better chunking"""
        return self.text_splitter.split_text(text)
    
    def _detect_themes(self, content: str) -> List[str]:
        """Simple keyword-based theme detection"""
        content_lower = content.lower()
        themes = []
        
        theme_keywords = {
            'devotion': ['devotion', 'bhakti', 'love', 'worship', 'devotee'],
            'faith': ['faith', 'trust', 'belief', 'conviction', 'confidence'],
            'surrender': ['surrender', 'submission', 'ego', 'humility', 'abandon'],
            'service': ['service', 'seva', 'selfless', 'sacrifice', 'serve'],
            'knowledge': ['knowledge', 'wisdom', 'understanding', 'truth', 'realization'],
            'detachment': ['detachment', 'renunciation', 'worldly', 'attachment', 'desire'],
            'dharma': ['dharma', 'righteousness', 'duty', 'moral', 'virtue'],
            'meditation': ['meditation', 'contemplation', 'focus', 'concentrate', 'samadhi'],
            'guru': ['guru', 'master', 'teacher', 'guide', 'sant'],
            'satsang': ['satsang', 'company', 'association', 'fellowship', 'devotees']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                themes.append(theme)
        
        return themes[:3]  # Limit to top 3 themes

    def get_document_stats(self, documents: List[VachanamrutDocument]) -> Dict:
        """Get processing statistics"""
        total_content = sum(len(doc.content) for doc in documents)
        themes_count = {}
        
        for doc in documents:
            for theme in doc.themes:
                themes_count[theme] = themes_count.get(theme, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_characters': total_content,
            'average_length': total_content // len(documents) if documents else 0,
            'themes_distribution': themes_count,
            'unique_references': len(set(doc.reference for doc in documents))
        }