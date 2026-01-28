from typing import List, Dict, Optional, Tuple
import re
import logging
from pathlib import Path
import json
import hashlib
import os
import time
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from collections import Counter
import math
from functools import lru_cache

import pdfplumber
from app.core.cache_manager import LRUCache, CacheManager
from app.core.lazy_loader import LazyLoader
from app.core.compression_utils import DataCompressor
from app.core.text_cleaner import get_text_cleaner

logger = logging.getLogger(__name__)

@dataclass
class VachanamrutPassage:
    """Simple passage from Vachanamrut PDF"""
    id: str
    content: str
    reference: str
    page_number: int
    chapter: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VachanamrutPassage':
        return cls(**data)

class PDFVectorStore:
    """
    Optimized PDF-based retrieval system with advanced caching
    Focus: Extract factual content from Vachanamrut PDF and enable text-based search
    """
    
    def __init__(self, storage_path: str = "data/processed/pdf_store.json"):
        self.passages: List[VachanamrutPassage] = []
        self.storage_path = storage_path
        self.pdf_metadata_cache: Dict = {}
        
        # Advanced caching system
        self.search_cache = LRUCache(name="pdf_search", max_size=1000, max_memory_mb=50)
        self.passage_cache = LRUCache(name="passages", max_size=5000, max_memory_mb=100)
        self.cache_manager = CacheManager()
        self.data_compressor = DataCompressor()
        
        # Lazy loader for large data
        self.lazy_loader = LazyLoader(
            storage_path,
            loader_func=self._load_pdf_store,
            name="pdf_retrieval"
        )
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0,
            'memory_usage_mb': 0,
            'compression_ratio': 0,
            'last_updated': time.time(),
            'prefilter_efficiency': 0.0
        }
        
        # Index for faster lookups
        self._reference_index: Dict[str, List[VachanamrutPassage]] = {}
        self._chapter_index: Dict[str, List[VachanamrutPassage]] = {}
        self._word_index: Dict[str, List[VachanamrutPassage]] = {}
        
    def process_pdf(self, pdf_path: str, force_reprocess: bool = False) -> Dict:
        """Process the Vachanamrut PDF and extract passages with incremental processing"""
        
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Get PDF file metadata
        pdf_stat = pdf_path_obj.stat()
        current_pdf_info = {
            'file_size': pdf_stat.st_size,
            'modification_time': pdf_stat.st_mtime,
            'file_path': str(pdf_path_obj.absolute())
        }
        
        # Check if reprocessing is needed
        if not force_reprocess and self._should_skip_processing(current_pdf_info):
            logger.info("PDF unchanged, loading existing processing results")
            result = self.load()
            if result['status'] == 'loaded_from_cache':
                result['pdf_metadata'] = current_pdf_info
                return result
        
        # Update metadata cache
        self.pdf_metadata_cache = current_pdf_info
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_passages = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    # Extract passages from this page
                    page_passages = self._extract_passages_from_page(text, page_num)
                    all_passages.extend(page_passages)
                    
                    logger.info(f"Processed page {page_num}, found {len(page_passages)} passages")
                
                self.passages = all_passages
                # Clear advanced search cache when data changes
                self.search_cache.clear()
                self.passage_cache.clear()
                # Rebuild indexes for faster lookups
                self._build_indexes()
                self.save()
                
                return {
                    'status': 'processed',
                    'total_passages': len(self.passages),
                    'total_pages': len(pdf.pages)
                }
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def _extract_passages_from_page(self, text: str, page_num: int) -> List[VachanamrutPassage]:
        """Extract individual passages from a page of text with enhanced reference parsing"""
        passages = []
        
        # Use enhanced text cleaner for OCR artifact removal
        text_cleaner = get_text_cleaner()
        text = text_cleaner.clean_text(text)
        
        # Additional advanced text cleaning
        text = self._advanced_text_cleaning(text)
        
        # Enhanced reference patterns with OCR variant handling
        reference_patterns = [
            # Gadhada variations with Roman numerals (clean text)
            r'\b(Gadhada(?:ji)?\s+[IVX]+-\d+)\b',
            r'\b(Gadhada\s+[IVX]+\.\d+)\b',  # Gadhada I.1
            r'\b(Gadhada\s+[IVX]+\s+\d+)\b', # Gadhada I 1
            
            # OCR variants of Gadhada (common OCR errors: 5->S, l->1, ii->a, etc.)
            r'\b(Gadhad[a5iíl]\s*[IVX]?\s*[-.]?\s*\d+[-.]?\d*)\b',  # Gadhad5 1-9, Gadhadi 1-10
            r'\b(Gadhadii?\s*\d+-\d+)\b',  # Gadhadii 1-8
            r'\b(Gadhadl\s*\d+-\d+)\b',  # Gadhadl 1-9
            r'\b(Gadhadfi?\s*\d+-\d+)\b',  # Gadhadfi 1-9
            r'\b(Gadhad[a-z5]+\s+[IVX]?[-.\s]*\d+)\b',  # Generic Gadhada OCR variants
            
            # Other locations with various separators
            r'\b((?:Sarangpur|S[aā]rangpur|Siirangpur)[-\.\s]\d+)\b',  # Sarangpur with OCR variants
            r'\b((?:Vadtal|V[aā]dtal|Viidtal)[-\.\s]\d+)\b',  # Vadtal variants
            r'\b((?:Ahmedabad|Amdāvād|Amdavad|Amdiiviid|Amdivad)[-\.\s]\d+)\b',  # Ahmedabad variants
            r'\b((?:Kariyani|Kariyiini|Kariy[aā]ni)[-\.\s]\d+)\b',  # Kariyani variants
            r'\b((?:Loyej|Loy[aā]|Loyii)[-\.\s]\d+)\b',  # Loyej variants
            r'\b((?:Panchala|Panchālā|Panchal[aā])[-\.\s]\d+)\b',  # Panchala variants
            r'\b((?:Jetalpur|Jet[aā]lpur)[-\.\s]\d+)\b',  # Jetalpur variants
            
            # Additional patterns for edge cases
            r'\b(Gadhada\s+(?:First|Second|Third)\s+\d+)\b',  # Gadhada First 1
            r'\b([A-Z][a-z]+(?:pur|bad|ada|ni|ej|ala|tar)\s*[-\.\s]+\d+)\b',  # Generic location pattern
        ]
        
        # Find all possible references with confidence scoring
        found_references = []
        for i, pattern in enumerate(reference_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref = match.group(1)
                start_pos = match.start()
                confidence = 1.0 - (i * 0.1)  # Higher confidence for more specific patterns
                found_references.append((start_pos, ref, confidence))
        
        # Remove duplicates and sort by position, then by confidence
        unique_refs = {}
        for pos, ref, conf in found_references:
            normalized_ref = self._normalize_reference(ref)
            if normalized_ref not in unique_refs or unique_refs[normalized_ref][2] < conf:
                unique_refs[normalized_ref] = (pos, ref, conf)
        
        found_references = [(pos, ref, conf) for pos, ref, conf in unique_refs.values()]
        found_references.sort(key=lambda x: (x[0], -x[2]))  # Sort by position, then by confidence desc
        
        # Extract passages between references with improved boundary detection
        if found_references:
            for i, (pos, ref, confidence) in enumerate(found_references):
                # Get content after this reference until next reference or end
                start_pos = pos + len(ref)
                end_pos = found_references[i + 1][0] if i + 1 < len(found_references) else len(text)
                
                content = text[start_pos:end_pos].strip()
                
                # Enhanced content cleaning and validation
                content = self._enhanced_passage_content_cleaning(content)
                
                # Quality checks for passage content
                if self._is_valid_passage_content(content, ref, confidence):
                    # Extract chapter name with normalization
                    chapter = self._extract_chapter_name(ref)
                    
                    passage_id = self._generate_passage_id(ref, page_num)
                    
                    passage = VachanamrutPassage(
                        id=passage_id,
                        content=content,
                        reference=self._normalize_reference(ref),
                        page_number=page_num,
                        chapter=chapter
                    )
                    passages.append(passage)
        
        # If no valid references found, check for substantial content
        elif len(text) > 200:
            # Try to extract meaningful content even without clear references
            content = self._clean_passage_content(text)
            if len(content) > 100:
                passage_id = self._generate_passage_id(f"Page-{page_num}", page_num)
                passage = VachanamrutPassage(
                    id=passage_id,
                    content=content,
                    reference=f"Page {page_num}",
                    page_number=page_num,
                    chapter="Vachanamrut"
                )
                passages.append(passage)
        
        return passages
    
    def _advanced_text_cleaning(self, text: str) -> str:
        """Advanced text cleaning for better parsing"""
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR errors in references
        text = re.sub(r'Gadhada\s+I\s*-\s*(\d+)', r'Gadhada I-\1', text)
        text = re.sub(r'Gadhada\s+II\s*-\s*(\d+)', r'Gadhada II-\1', text)
        text = re.sub(r'([A-Z][a-z]+)\s*-\s*(\d+)', r'\1-\2', text)  # Generic location-number
        
        # Clean up common PDF artifacts
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)  # Remove invisible characters
        text = re.sub(r'\f', ' ', text)  # Form feed to space
        
        return text
    
    def _normalize_reference(self, reference: str) -> str:
        """Normalize reference format for consistency, handling OCR artifacts"""
        ref = reference.strip()
        
        # Normalize common variations
        ref = re.sub(r'\s+', ' ', ref)  # Multiple spaces to single
        ref = re.sub(r'(\w+)\s*-\s*(\d+)', r'\1-\2', ref)  # "Location - 1" to "Location-1"
        ref = re.sub(r'(\w+)\s+([IVX]+)\s*-\s*(\d+)', r'\1 \2-\3', ref)  # Gadhada I - 1" to "Gadhada I-1"
        ref = re.sub(r'(\w+)\s+([IVX]+)\.\s*(\d+)', r'\1 \2-\3', ref)  # "Gadhada I.1" to "Gadhada I-1"
        
        # Fix OCR artifacts in Gadhada variants
        # Common OCR errors: Gadhad5, Gadhadii, Gadhadi, Gadhadl, Gadhadfi -> Gadhada
        ref = re.sub(r'\b(Gadhad)[5iíl]\b', r'\1a', ref, flags=re.IGNORECASE)
        ref = re.sub(r'\bGadhadii\b', 'Gadhada', ref, flags=re.IGNORECASE)
        ref = re.sub(r'\bGadhadi\b', 'Gadhada', ref, flags=re.IGNORECASE)
        ref = re.sub(r'\bGadhadl\b', 'Gadhada', ref, flags=re.IGNORECASE)
        ref = re.sub(r'\bGadhadfi\b', 'Gadhada', ref, flags=re.IGNORECASE)
        ref = re.sub(r'\bGadhad[a-z5]+\b', 'Gadhada', ref, flags=re.IGNORECASE)
        
        # Normalize section numbers - convert formats like "1-9" to "I-9"
        # Handle pattern like "Gadhada 1-9" -> "Gadhada I-9"
        section_map = {'1': 'I', '2': 'II', '3': 'III'}
        for num, roman in section_map.items():
            # Match "Gadhada 1-" pattern and convert to "Gadhada I-"
            ref = re.sub(rf'\bGadhada\s*{num}-(\d+)', rf'Gadhada {roman}-\1', ref, flags=re.IGNORECASE)
        
        # Fix OCR artifacts in other location names
        ocr_location_fixes = {
            r'S[aāii]+rangpur': 'Sarangpur',
            r'V[aāii]+dtal': 'Vadtal',
            r'Amd[aāii]+v[aāii]+d': 'Ahmedabad',
            r'Kariy[aāii]+ni': 'Kariyani',
            r'Loy[aāii]+': 'Loyej',
            r'Panchal[aāii]+': 'Panchala',
            r'Jet[aāii]+lpur': 'Jetalpur',
        }
        
        for pattern, replacement in ocr_location_fixes.items():
            ref = re.sub(pattern, replacement, ref, flags=re.IGNORECASE)
        
        # Standardize common location names
        location_map = {
            'gadhadji': 'Gadhada',
            'gadhada first': 'Gadhada I',
            'gadhada second': 'Gadhada II', 
            'gadhada third': 'Gadhada III'
        }
        
        ref_lower = ref.lower()
        for old, new in location_map.items():
            if old in ref_lower:
                ref = re.sub(re.escape(old), new, ref, flags=re.IGNORECASE)
                break
        
        return ref
    
    def _enhanced_passage_content_cleaning(self, content: str) -> str:
        """Enhanced content cleaning with better noise removal"""
        if not content:
            return ""
        
        # Remove page headers/footers with more variations
        header_footer_patterns = [
            r'The VachanImrut \d+',
            r'\d+\s+The VachanImrut', 
            r'The Vachaniimrut \d+',
            r'The VachanHmrut \d+',
            r'Vachanamrut \d+',
            r'\d+\s+Vachanamrut',
            r'Page \d+ of \d+',
            r'\d+\s*/\s*\d+'  # Page numbers like "1/200"
        ]
        
        for pattern in header_footer_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove scattered punctuation and numbering at start
        content = re.sub(r'^[\d\.\s\-\:;,]+', '', content)
        content = re.sub(r'^[^\w]*', '', content)  # Remove non-word chars at start
        
        # Clean up excessive spacing and line breaks
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Remove very short fragments
        sentences = re.split(r'[.!?]+', content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        cleaned_content = '. '.join(meaningful_sentences)
        if meaningful_sentences and not cleaned_content.endswith('.'):
            cleaned_content += '.'
        
        return cleaned_content
    
    def _is_valid_passage_content(self, content: str, reference: str, confidence: float) -> bool:
        """Determine if extracted content is a valid passage"""
        if not content or len(content) < 30:
            return False
        
        # Skip content that looks like metadata
        skip_indicators = [
            'glossary', 'appendix', 'index', 'table', 'chart', 'diagram',
            'foldout', 'copyright', 'isbn', 'published', 'edition',
            'contents', 'introduction', 'preface'
        ]
        
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in skip_indicators):
            return False
        
        # Quality score based on various factors
        quality_score = 0.0
        
        # Length factor (moderate length preferred)
        if 50 <= len(content) <= 2000:
            quality_score += 0.3
        elif len(content) > 2000:
            quality_score += 0.1  # Very long might be multiple passages
        
        # Sentence structure (should have proper sentences)
        sentence_count = len(re.findall(r'[.!?]+', content))
        if sentence_count >= 2:
            quality_score += 0.2
        
        # Spiritual/philosophical content indicators
        spiritual_terms = [
            'god', 'devotion', 'faith', 'prayer', 'dharma', 'soul', 'spirit',
            'bhakti', 'surrender', 'worship', 'divine', 'sacred', 'holy',
            'swaminarayan', 'maharaj', 'sant', 'guru', 'devotee'
        ]
        
        spiritual_matches = sum(1 for term in spiritual_terms if term in content_lower)
        if spiritual_matches >= 2:
            quality_score += 0.3
        elif spiritual_matches >= 1:
            quality_score += 0.1
        
        # Reference confidence boost
        quality_score += confidence * 0.2
        
        return quality_score >= 0.4  # Threshold for acceptance
    
    def _clean_passage_content(self, content: str) -> str:
        """Clean up passage content to remove noise"""
        # Use enhanced text cleaner first
        text_cleaner = get_text_cleaner()
        content = text_cleaner.clean_passage(content)
        
        # Additional cleaning for header/footer removal
        content = re.sub(r'The VachanImrut \d+', '', content)
        content = re.sub(r'\d+\s+The VachanImrut', '', content)
        content = re.sub(r'The Vachaniimrut \d+', '', content)
        content = re.sub(r'The VachanHmrut \d+', '', content)
        
        # Remove glossary/appendix markers
        content = re.sub(r'Glossary.*$', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Appendix.*$', '', content, flags=re.IGNORECASE)
        
        # Remove excessive numbering/indexing
        content = re.sub(r'^\d+\s*\.\s*', '', content)
        content = re.sub(r'^\d+\s+', '', content)
        
        # Remove scattered punctuation
        content = re.sub(r'^[\.;,:]+\s*', '', content)
        
        # Clean up spacing
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Skip content that looks like glossary/appendix/index
        skip_patterns = [
            r'^\s*-\s*[a-z]+\s*-.*',  # Glossary entries like "- dharma - righteousness"
            r'^\s*\d+\.\s*[A-Z][a-z]+\s*-',  # Numbered definitions
            r'time measurements.*conversion',  # Time conversion tables
            r'cosmological chronology',  # Charts and diagrams
            r'foldout chart',  # Chart references
            r'^\s*[A-Z][a-z]+\s+[A-Z][a-z]+.*\d+\s*$',  # Index entries
        ]
        
        content_lower = content.lower()
        for pattern in skip_patterns:
            if re.search(pattern, content_lower):
                return ""  # Skip this content
        
        # Remove very short fragments at start/end
        lines = content.split('.')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 15:  # Only keep substantial sentences
                cleaned_lines.append(line)
        
        result = '. '.join(cleaned_lines) if cleaned_lines else content
        
        # Final check - if content is too short or looks like metadata, skip it
        if len(result) < 50 or any(word in result.lower() for word in ['glossary', 'appendix', 'index', 'chart', 'table']):
            return ""
        
        return result
    
    def _extract_chapter_name(self, reference: str) -> str:
        """Extract clean chapter name from reference, handling OCR variants"""
        ref_lower = reference.lower()
        
        # Map OCR variant patterns to clean chapter names
        # Using regex patterns to match common OCR errors
        chapter_patterns = [
            # Gadhada variants (most common OCR errors)
            (r'gadhad[a5iíl]|gadhadii|gadhadi|gadhadl|gadhadfi|gadhads|gadhadc|gadhadb|gadhadg|gadhadz|gadhadp|gadhadf|gadhadd|gadhadh|gadhadn|gadhadti', 'Gadhada'),
            
            # Sarangpur variants (OCR often confuses 'a' with 'ii', 'S' with 'Š', etc.)
            (r's[aāiíl]+rangpur|shrangpur|sgrangpur|sbrangpur|slrangpur|sirangpur|serangpur|ssrangpur|szlrangpur|skangpur', 'Sarangpur'),
            
            # Kariyani variants (very common OCR errors - many starting with g, e, k)
            (r'k[aāiíl]+riy[aāiíl]+ni|eriy[aāiíl]+ni|griy[aāiíl]+ni|kbriy|kiriy|khriy|klriy|keriy|griyhni|eriybni|eriysni|khriyhni|eyhni|kpriyhni|kiriyhni|griygni|eiriyini|kiiriysni|myiini|griysni|eriygni', 'Kariyani'),
            
            # Loyej/Loya variants
            (r'loy[aāiíl]+|loyii|loyej', 'Loya'),
            
            # Vadtal variants
            (r'v[aāiíl]+dtal|viidtal', 'Vadtal'),
            
            # Ahmedabad/Amdavad variants
            (r'ahmedabad|amd[aāiíl]+v[aāiíl]+d|amdiiviid|amdivad', 'Ahmedabad'),
            
            # Panchala variants
            (r'panchal[aāiíl]+|panchālā', 'Panchala'),
            
            # Jetalpur variants  
            (r'jet[aāiíl]+lpur|jetalpur', 'Jetalpur'),
            
            # Special cases
            (r'brahmapur', 'Brahmapur'),
            (r'upisani', 'Upisani'),
        ]
        
        for pattern, clean_name in chapter_patterns:
            if re.search(pattern, ref_lower):
                return clean_name
        
        # Fallback: extract first word and try to clean it
        first_word = reference.split()[0] if reference.split() else reference
        
        # Remove numbers and special characters from the first word
        clean_word = re.sub(r'[^a-zA-Z]', '', first_word)
        
        return clean_word.capitalize() if clean_word else 'Vachanamrut'
    
    def _generate_passage_id(self, reference: str, page_num: int) -> str:
        """Generate unique ID for passage"""
        content_hash = hashlib.md5(f"{reference}_{page_num}".encode()).hexdigest()[:8]
        return f"{reference}_{content_hash}".replace(' ', '_').replace('-', '_')
    
    def _generate_cache_key(self, query: str, top_k: int, min_score: float) -> str:
        """Generate normalized cache key for search results"""
        # Normalize query for better cache hits
        normalized_query = re.sub(r'\s+', ' ', query.lower().strip())
        return hashlib.md5(f"{normalized_query}_{top_k}_{min_score}".encode()).hexdigest()
    
    def _load_pdf_store(self) -> Dict:
        """Lazy loader function for PDF store data"""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.passages = [VachanamrutPassage.from_dict(p) for p in data.get('passages', [])]
                    self.pdf_metadata_cache = data.get('metadata', {})
                    logger.info(f"Loaded {len(self.passages)} passages from cache")
                    return {'status': 'loaded_from_cache', 'passages_count': len(self.passages)}
            else:
                logger.info("No cached PDF store found")
                return {'status': 'no_cache_found'}
        except Exception as e:
            logger.error(f"Error loading PDF store: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _build_indexes(self):
        """Build lookup indexes for faster searching"""
        logger.info("Building search indexes...")
        
        # Clear existing indexes
        self._reference_index.clear()
        self._chapter_index.clear()
        self._word_index.clear()
        
        for passage in self.passages:
            # Reference index
            ref = passage.reference
            if ref not in self._reference_index:
                self._reference_index[ref] = []
            self._reference_index[ref].append(passage)
            
            # Chapter index
            chapter = passage.chapter
            if chapter not in self._chapter_index:
                self._chapter_index[chapter] = []
            self._chapter_index[chapter].append(passage)
            
            # Word index (for common words)
            words = set(re.findall(r'\w+', passage.content.lower()))
            for word in words:
                if len(word) >= 3:  # Only index meaningful words
                    if word not in self._word_index:
                        self._word_index[word] = []
                    if passage not in self._word_index[word]:  # Avoid duplicates
                        self._word_index[word].append(passage)
        
        logger.info(f"Built indexes: {len(self._reference_index)} refs, "
                   f"{len(self._chapter_index)} chapters, {len(self._word_index)} words")
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Tuple[VachanamrutPassage, float]]:
        """Optimized search with advanced caching and performance tracking"""
        start_time = time.time()
        
        if not self.passages:
            logger.warning("No passages loaded for search")
            return []
        
        # Advanced cache key with normalization
        cache_key = self._generate_cache_key(query, top_k, min_score)
        
        # Check advanced cache first
        cached_result = self.search_cache.get(cache_key)
        if cached_result is not None:
            self.search_stats['cache_hits'] += 1
            self.search_stats['total_searches'] += 1
            logger.debug(f"Cache hit for query: {query[:50]}... (ratio: {self.search_stats['cache_hits']/max(1,self.search_stats['total_searches']):.2f})")
            return cached_result
        
        query_lower = query.lower()
        results = []
        
        # Pre-filter passages for better performance
        candidate_passages = self._prefilter_passages(query_lower)
        
        for passage in candidate_passages:
            content_lower = passage.content.lower()
            
            # Calculate enhanced similarity score
            similarity = self._calculate_text_similarity(query_lower, content_lower)
            
            if similarity >= min_score:
                results.append((passage, similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        final_results = results[:top_k]
        
        # Store in advanced cache (automatically handles LRU eviction)
        self.search_cache.put(cache_key, final_results)
        
        # Update performance stats
        search_time = time.time() - start_time
        self.search_stats['total_searches'] += 1
        self.search_stats['avg_search_time'] = (
            (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + search_time) /
            self.search_stats['total_searches']
        )
        
        # Update memory usage stats
        self.search_stats['memory_usage_mb'] = self._estimate_memory_usage()
        
        logger.debug(f"Search completed in {search_time:.3f}s for query: {query[:50]}... "
                    f"({len(final_results)} results from {len(candidate_passages)} candidates)")
        return final_results
    
    async def search_async(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Tuple[VachanamrutPassage, float]]:
        """Async version of search for better performance with large datasets"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Use thread pool for CPU-intensive similarity calculations
        with ThreadPoolExecutor(max_workers=4) as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, self.search, query, top_k, min_score
            )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB"""
        try:
            import sys
            cache_size = (sys.getsizeof(self.search_cache) + sys.getsizeof(self.passage_cache)) / 1024 / 1024
            passages_size = sum(sys.getsizeof(p.content) for p in self.passages) / 1024 / 1024
            return cache_size + passages_size
        except:
            return 0.0
    
    def _calculate_text_similarity(self, query: str, content: str) -> float:
        """Enhanced text similarity using TF-IDF and multiple scoring methods"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Extract words
        query_words = re.findall(r'\w+', query_lower)
        content_words = re.findall(r'\w+', content_lower)
        
        if not query_words:
            return 0.0
        
        # 1. TF-IDF based similarity
        tfidf_score = self._calculate_tfidf_similarity(query_words, content_words)
        
        # 2. Jaccard similarity (existing)
        query_set = set(query_words)
        content_set = set(content_words)
        intersection = query_set.intersection(content_set)
        union = query_set.union(content_set)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # 3. Sequence similarity for phrase matching
        sequence_score = self._calculate_sequence_similarity(query_lower, content_lower)
        
        # 4. Fuzzy matching for typos
        fuzzy_score = self._calculate_fuzzy_similarity(query_words, content_words)
        
        # 5. Semantic keyword boosts
        keyword_boost = self._calculate_keyword_boost(query_lower, content_lower)
        
        # 6. Position-based scoring (early mentions get higher scores)
        position_score = self._calculate_position_score(query_lower, content_lower)
        
        # Combine scores with weights
        weights = {
            'tfidf': 0.3,
            'jaccard': 0.2, 
            'sequence': 0.2,
            'fuzzy': 0.1,
            'keyword': 0.1,
            'position': 0.1
        }
        
        final_score = (
            tfidf_score * weights['tfidf'] +
            jaccard * weights['jaccard'] +
            sequence_score * weights['sequence'] +
            fuzzy_score * weights['fuzzy'] +
            keyword_boost * weights['keyword'] +
            position_score * weights['position']
        )
        
        return min(1.0, final_score)
    
    def _should_skip_processing(self, current_pdf_info: Dict) -> bool:
        """Check if PDF processing should be skipped based on file metadata"""
        if not Path(self.storage_path).exists():
            return False
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stored_metadata = data.get('pdf_metadata', {})
            
            # Compare file metadata
            return (
                stored_metadata.get('file_size') == current_pdf_info['file_size'] and
                stored_metadata.get('modification_time') == current_pdf_info['modification_time'] and
                stored_metadata.get('file_path') == current_pdf_info['file_path']
            )
        except Exception as e:
            logger.warning(f"Could not check PDF metadata: {e}")
            return False
    
    def _calculate_tfidf_similarity(self, query_words: List[str], content_words: List[str]) -> float:
        """Calculate TF-IDF based similarity score"""
        if not query_words or not content_words:
            return 0.0
        
        # Calculate term frequencies
        query_tf = Counter(query_words)
        content_tf = Counter(content_words)
        
        # Simple IDF calculation based on total passages (approximation)
        total_docs = len(self.passages) or 1
        
        score = 0.0
        for term in query_tf:
            if term in content_tf:
                # TF score
                tf_score = content_tf[term] / len(content_words)
                
                # Simple IDF (documents containing this term)
                docs_with_term = sum(1 for p in self.passages if term in p.content.lower())
                idf_score = math.log(total_docs / (docs_with_term + 1))
                
                score += tf_score * idf_score * query_tf[term]
        
        return min(1.0, score)
    
    def _calculate_sequence_similarity(self, query: str, content: str) -> float:
        """Calculate similarity based on sequence matching"""
        matcher = SequenceMatcher(None, query, content)
        return matcher.ratio() * 0.8  # Scale down as it's very sensitive
    
    def _calculate_fuzzy_similarity(self, query_words: List[str], content_words: List[str]) -> float:
        """Calculate fuzzy similarity to handle typos"""
        if not query_words or not content_words:
            return 0.0
        
        total_score = 0.0
        for q_word in query_words:
            best_match = 0.0
            for c_word in content_words:
                if len(q_word) > 3 and len(c_word) > 3:  # Only for longer words
                    similarity = SequenceMatcher(None, q_word, c_word).ratio()
                    if similarity > 0.8:  # High threshold for fuzzy match
                        best_match = max(best_match, similarity * 0.7)  # Scaled down
            total_score += best_match
        
        return total_score / len(query_words)
    
    def _calculate_keyword_boost(self, query: str, content: str) -> float:
        """Calculate boost for important spiritual keywords"""
        important_keywords = {
            'god': 1.0, 'devotion': 0.9, 'faith': 0.9, 'surrender': 0.8,
            'dharma': 0.8, 'bhakti': 0.9, 'guru': 0.8, 'sant': 0.8,
            'swaminarayan': 1.0, 'maharaj': 0.9, 'moksha': 0.8,
            'prayer': 0.7, 'worship': 0.7, 'service': 0.7, 'satsang': 0.8
        }
        
        boost_score = 0.0
        for keyword, weight in important_keywords.items():
            if keyword in query and keyword in content:
                boost_score += weight * 0.1
        
        return min(0.3, boost_score)  # Cap the boost
    
    def _calculate_position_score(self, query: str, content: str) -> float:
        """Give higher scores to matches appearing earlier in the passage"""
        query_words = query.split()
        if not query_words:
            return 0.0
        
        content_words = content.split()
        if not content_words:
            return 0.0
        
        position_scores = []
        for word in query_words:
            for i, content_word in enumerate(content_words):
                if word.lower() == content_word.lower():
                    # Earlier positions get higher scores
                    position_score = max(0, 1.0 - (i / len(content_words)))
                    position_scores.append(position_score)
                    break
        
        return sum(position_scores) / len(query_words) if position_scores else 0.0
    
    def _prefilter_passages(self, query: str) -> List[VachanamrutPassage]:
        """High-performance pre-filter using word index for ultra-fast lookup"""
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return self.passages
        
        # Cache key for prefiltering  
        prefilter_key = f"prefilter_{hash(frozenset(query_words))}"
        cached_result = self.passage_cache.get(prefilter_key)
        if cached_result is not None:
            return cached_result
        
        # Use word index for ultra-fast filtering
        if self._word_index:
            # Use dict with passage ID as key to avoid hashability issues
            candidate_dict = {}
            for word in query_words:
                if word in self._word_index:
                    for passage in self._word_index[word]:
                        candidate_dict[passage.id] = passage
            candidate_passages = list(candidate_dict.values())
            
            if candidate_passages:
                # Score candidates using intersection ratio
                passage_scores = []
                for passage in candidate_passages:
                    content_words = set(re.findall(r'\w+', passage.content.lower()))
                    intersection = query_words.intersection(content_words)
                    
                    if intersection:
                        match_ratio = len(intersection) / len(query_words)
                        content_quality = min(1.0, len(passage.content) / 200)
                        prefilter_score = match_ratio * content_quality
                        passage_scores.append((passage, prefilter_score))
                
                if passage_scores:
                    # Sort and limit candidates for optimal performance
                    passage_scores.sort(key=lambda x: x[1], reverse=True) 
                    max_candidates = min(len(passage_scores), max(50, len(self.passages) // 10))
                    filtered_passages = [p for p, _ in passage_scores[:max_candidates]]
                    
                    # Cache and return
                    self.passage_cache.put(prefilter_key, filtered_passages)
                    return filtered_passages
        
        # Fallback to original method if no word index
        return self._fallback_prefilter(query_words)
    
    def _fallback_prefilter(self, query_words: set) -> List[VachanamrutPassage]:
        """Fallback prefilter method when word index is not available"""
        passage_scores = []
        for passage in self.passages:
            content_words = set(re.findall(r'\w+', passage.content.lower()))
            intersection = query_words.intersection(content_words)
            
            if intersection:
                match_ratio = len(intersection) / len(query_words)
                content_quality = min(1.0, len(passage.content) / 200)
                prefilter_score = match_ratio * content_quality
                passage_scores.append((passage, prefilter_score))
        
        if not passage_scores:
            return self.passages
            
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        max_candidates = min(len(passage_scores), max(50, len(self.passages) // 10))
        return [p for p, _ in passage_scores[:max_candidates]]
    
    @lru_cache(maxsize=100)
    def _get_word_frequencies(self) -> Dict[str, int]:
        """Cache word frequencies across all passages for IDF calculation"""
        word_freq = {}
        for passage in self.passages:
            words = re.findall(r'\w+', passage.content.lower())
            for word in set(words):
                word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq
    
    def get_search_stats(self) -> Dict:
        """Get search performance statistics"""
        cache_hit_rate = (
            self.search_stats['cache_hits'] / self.search_stats['total_searches']
            if self.search_stats['total_searches'] > 0 else 0
        )
        
        return {
            **self.search_stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._search_cache),
            'passages_count': len(self.passages)
        }
    
    def clear_cache(self) -> None:
        """Clear all caches and reset stats"""
        self.search_cache.clear()
        self.passage_cache.clear()
        self.cache_manager.clear_all_caches()
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0,
            'memory_usage_mb': 0,
            'compression_ratio': 0,
            'last_updated': time.time()
        }
    
    def get_by_reference(self, reference: str) -> List[VachanamrutPassage]:
        """Get all passages for a specific reference"""
        return [p for p in self.passages if p.reference == reference]
    
    def get_by_chapter(self, chapter: str) -> List[VachanamrutPassage]:
        """Get all passages for a specific chapter"""
        return [p for p in self.passages if p.chapter.lower() == chapter.lower()]
    
    def save(self) -> None:
        """Save processed passages to disk"""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'passages': [passage.to_dict() for passage in self.passages],
            'total_passages': len(self.passages),
            'pdf_metadata': self.pdf_metadata_cache,
            'processing_info': {
                'type': 'pdf_extraction',
                'method': 'enhanced_text_similarity',
                'features': [
                    'tfidf_scoring',
                    'fuzzy_matching', 
                    'sequence_similarity',
                    'keyword_boosting',
                    'position_scoring',
                    'search_caching'
                ],
                'last_updated': time.time()
            }
        }
        
        # Try compression for large datasets
        try:
            if len(self.passages) > 100:
                compressed_result = self.data_compressor.compress_json(data)
                if compressed_result.compression_ratio > 0.3:  # If compression saves >30%
                    compressed_path = self.storage_path + '.gz'
                    with open(compressed_path, 'wb') as f:
                        f.write(compressed_result.compressed_data)
                    
                    self.search_stats['compression_ratio'] = compressed_result.compression_ratio
                    logger.info(f"Saved {len(self.passages)} passages compressed to {compressed_path} "
                              f"(saved {compressed_result.space_saved_mb:.1f}MB, ratio: {compressed_result.compression_ratio:.2f})")
                    return
        except Exception as e:
            logger.warning(f"Compression failed, falling back to regular save: {e}")
        
        # Regular save (fallback or small datasets)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.passages)} passages to {self.storage_path}")
    
    def load(self) -> Dict:
        """Load processed passages from disk"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.passages = [VachanamrutPassage.from_dict(p) for p in data['passages']]
            
            logger.info(f"Loaded {len(self.passages)} passages from {self.storage_path}")
            
            return {
                'status': 'loaded_from_cache',
                'total_passages': len(self.passages)
            }
            
        except Exception as e:
            logger.error(f"Error loading passages: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_stats(self) -> Dict:
        """Get statistics about the processed content"""
        if not self.passages:
            return {}
        
        chapters = {}
        references = set()
        
        for passage in self.passages:
            chapter = passage.chapter
            chapters[chapter] = chapters.get(chapter, 0) + 1
            references.add(passage.reference)
        
        return {
            'total_passages': len(self.passages),
            'unique_references': len(references),
            'chapters': chapters,
            'average_length': sum(len(p.content) for p in self.passages) / len(self.passages)
        }

# Global instance
_pdf_store_instance: Optional[PDFVectorStore] = None

def get_pdf_store() -> PDFVectorStore:
    """Get global PDF store instance"""
    global _pdf_store_instance
    if _pdf_store_instance is None:
        _pdf_store_instance = PDFVectorStore()
    return _pdf_store_instance