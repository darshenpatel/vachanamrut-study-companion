"""
Text Cleaner for Vachanamrut PDF Content
Fixes OCR artifacts and improves text quality
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class VachanamrutTextCleaner:
    """
    Cleans and normalizes text extracted from Vachanamrut PDF.
    Fixes common OCR errors and standardizes formatting.
    """
    
    def __init__(self):
        # Common OCR errors in Vachanamrut PDF
        self.ocr_corrections = {
            # Vachanamrut variations
            'VachanImrut': 'Vachanamrut',
            'Vachaniimrut': 'Vachanamrut',
            'VachanHmrut': 'Vachanamrut',
            'Vachanlmrut': 'Vachanamrut',
            'Vachan1mrut': 'Vachanamrut',
            
            # Maharaj variations
            'Mahiiriij': 'Maharaj',
            'Mahiiraj': 'Maharaj',
            'Mahariij': 'Maharaj',
            'Maharaij': 'Maharaj',
            'Mahiirij': 'Maharaj',
            'MahHrHj': 'Maharaj',
            'Mahlriij': 'Maharaj',
            'MahZrZj': 'Maharaj',
            'MahiirEj': 'Maharaj',
            
            # Swami variations
            'Swlmi': 'Swami',
            'Swtimi': 'Swami',
            'Sw5mi': 'Swami',
            'SwiImI': 'Swami',
            'Swdmi': 'Swami',
            
            # Shriji variations
            'ShrijI': 'Shriji',
            'ShrIjI': 'Shriji',
            'ShrIji': 'Shriji',
            
            # Common names with OCR errors
            'MuktZnand': 'Muktanand',
            'Muktiinand': 'Muktanand',
            'Muktlnand': 'Muktanand',
            'GopZlZnand': 'Gopalanand',
            'Gopgllnand': 'Gopalanand',
            'Gopiilinand': 'Gopalanand',
            'BrahrnZnand': 'Brahmanand',
            'Brahmtinand': 'Brahmanand',
            'NityzInand': 'Nityanand',
            'Nityiinand': 'Nityanand',
            'Sachchidiinand': 'Sachchidanand',
            'Sachchidtinand': 'Sachchidanand',
            'Bhajanlnand': 'Bhajananand',
            
            # Place name corrections
            'DBdg Khlchar': 'Dada Khachar',
            'DDdZ Khlchar': 'Dada Khachar',
            'Vastii Khlchar': 'Vasta Khachar',
            'Vast5 Khlchar': 'Vasta Khachar',
            'GadhadH': 'Gadhada',
            'Gadhadji': 'Gadhada',
            'KIriyiini': 'Kariyani',
            'Kiiriyiini': 'Kariyani',
            'KZriylni': 'Kariyani',
            'KIriylni': 'Kariyani',
            
            # Sanskrit/spiritual terms
            'Bhagwln': 'Bhagwan',
            'Bhagwgn': 'Bhagwan',
            'Bhigwat': 'Bhagwat',
            'PurlnZs': 'Puranas',
            'Purlns': 'Puranas',
            'ItihL': 'Itihas',
            'Itihiis': 'Itihas',
            'ShLstras': 'Shastras',
            'Shistras': 'Shastras',
            'shdstras': 'shastras',
            'dharrna': 'dharma',
            'gnzn': 'gnan',
            'gniin': 'gnan',
            'vairZgya': 'vairagya',
            'vairtgya': 'vairagya',
            'Ztmi': 'atma',
            'iitma': 'atma',
            'ZtmZ': 'atma',
            'ltml': 'atma',
            'Ztmti': 'atma',
            'jivitml': 'jivatma',
            'jivZtmii': 'jivatma',
            'Paramltmi': 'Paramatma',
            'ParamZtmi': 'Paramatma',
            'Purushottam': 'Purushottam',
            'bhiikta': 'bhakta',
            'bhakti': 'bhakti',
            'moksha': 'moksha',
            'darshan': 'darshan',
            'darsban': 'darshan',
            'piTgh': 'pagh',
            'pZgh': 'pagh',
            'PradhLn': 'Pradhan',
            'PradhHn': 'Pradhan',
            'iikZsh': 'akash',
            'ika\'sh': 'akash',
            'PkPsh': 'akash',
            
            # Number/date artifacts
            'Samvat ': 'Samvat ',
            'sudi ': 'sudi ',
            'vadi ': 'vadi ',
            
            # Common word OCR errors
            'cornmandments': 'commandments',
            'rnust': 'must',
            'frorn': 'from',
            'thern': 'them',
            'rnind': 'mind',
            'rnanner': 'manner',
            'rnay': 'may',
            'rnan': 'man',
            'wornen': 'women',
            'tirne': 'time',
            'sarne': 'same',
            'narne': 'name',
            'carne': 'came',
            
            # Broken words
            'spiri- tual': 'spiritual',
            'devo- tion': 'devotion',
            'lib- eration': 'liberation',
            'eter- nal': 'eternal',
        }
        
        # Regex patterns for systematic fixes
        self.regex_patterns = [
            # Fix "I I" or "I 1" section markers
            (r'\bI\s+I\b', ' '),
            (r'\bI\s+1\b', ' '),
            (r'\b1\s+I\b', ' '),
            
            # Remove page numbers at start/end of lines
            (r'^\d{1,3}\s+(?=[A-Z])', ''),
            (r'(?<=\s)\d{1,3}\s*$', ''),
            
            # Fix reference number patterns
            (r'(\d+)\s*-\s*(\d+)', r'\1-\2'),
            
            # Remove header/footer artifacts
            (r'The VachanImrut\s*\d*', ''),
            (r'\d+\s+The VachanImrut', ''),
            (r'The Vachan[a-zA-Z]*mrut\s*\d*', ''),
            
            # Fix excessive whitespace
            (r'\s{3,}', ' '),
            
            # Fix broken sentences
            (r'(?<=[a-z])-\s+(?=[a-z])', ''),
            
            # Remove invisible characters
            (r'[\u200b\u200c\u200d\ufeff]', ''),
            
            # Fix common punctuation issues
            (r',,', ','),
            (r'\.\.(?!\.)', '.'),
            (r'\s+([,\.;:!?])', r'\1'),
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning transformations to text
        """
        if not text:
            return ""
        
        # Apply direct word replacements
        for wrong, correct in self.ocr_corrections.items():
            text = text.replace(wrong, correct)
        
        # Apply regex patterns
        for pattern, replacement in self.regex_patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def clean_passage(self, passage: str) -> str:
        """
        Clean a single passage with additional context-aware fixes
        """
        # First apply general cleaning
        cleaned = self.clean_text(passage)
        
        # Remove any remaining header/footer patterns
        header_patterns = [
            r'^\s*\d+\s+',  # Page numbers at start
            r'\s+\d+\s*$',  # Page numbers at end
            r'^\s*[IVX]+[-\s]\d+\s+\d+\s*',  # Chapter reference with page
        ]
        
        for pattern in header_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()
    
    def fix_reference(self, reference: str) -> str:
        """
        Normalize a Vachanamrut reference to standard format
        """
        if not reference:
            return reference
        
        # Apply OCR corrections
        for wrong, correct in self.ocr_corrections.items():
            reference = reference.replace(wrong, correct)
        
        # Standardize format: "Location Section-Number"
        # e.g., "Gadhada I-1", "Sarangpur-5", "Vadtal-18"
        
        # Fix spacing around dashes
        reference = re.sub(r'\s*-\s*', '-', reference)
        
        # Fix Roman numeral spacing
        reference = re.sub(r'([A-Za-z])\s+([IVX]+)\s*-', r'\1 \2-', reference)
        
        # Capitalize first letter of location
        if reference:
            reference = reference[0].upper() + reference[1:]
        
        return reference.strip()
    
    def calculate_quality_score(self, text: str) -> float:
        """
        Calculate a quality score for extracted text
        Higher score = better quality extraction
        """
        if not text or len(text) < 20:
            return 0.0
        
        score = 0.5  # Base score
        
        # Positive indicators
        # Has proper sentences
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count >= 2:
            score += 0.1
        
        # Contains spiritual keywords
        spiritual_keywords = [
            'god', 'devotion', 'faith', 'dharma', 'soul', 'bhakti',
            'swaminarayan', 'maharaj', 'sant', 'devotee', 'spiritual'
        ]
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in spiritual_keywords if kw in text_lower)
        if keyword_matches >= 3:
            score += 0.2
        elif keyword_matches >= 1:
            score += 0.1
        
        # Reasonable length
        word_count = len(text.split())
        if 50 <= word_count <= 500:
            score += 0.15
        elif word_count > 500:
            score += 0.1
        
        # Negative indicators
        # Too many numbers (likely tables/indexes)
        number_ratio = len(re.findall(r'\d+', text)) / max(word_count, 1)
        if number_ratio > 0.3:
            score -= 0.2
        
        # Contains likely OCR garbage
        garbage_patterns = [
            r'[^\x00-\x7F]{5,}',  # Long non-ASCII sequences
            r'\b[bcdfghjklmnpqrstvwxz]{4,}\b',  # Consonant-only words
            r'(?i)glossary|appendix|index|table|chart',  # Metadata content
        ]
        for pattern in garbage_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def should_skip_passage(self, text: str) -> bool:
        """
        Determine if a passage should be skipped (not actual content)
        """
        if not text or len(text) < 50:
            return True
        
        text_lower = text.lower()
        
        # Skip indicators
        skip_patterns = [
            'glossary',
            'appendix',
            'index of',
            'table of contents',
            'bibliography',
            'copyright',
            'isbn',
            'published by',
            'all rights reserved',
            'printed in',
        ]
        
        for pattern in skip_patterns:
            if pattern in text_lower:
                return True
        
        # Skip if mostly numbers (tables, charts)
        words = text.split()
        number_words = sum(1 for w in words if re.match(r'^\d+\.?\d*$', w))
        if number_words / max(len(words), 1) > 0.5:
            return True
        
        return False
    
    def extract_date_info(self, text: str) -> str:
        """
        Extract date information from passage
        """
        # Pattern for Samvat dates
        date_pattern = r'(?:On\s+)?([A-Za-z]+)\s+(sudi|vadi)\s+(\d+),?\s*Samvat\s+(\d+)'
        match = re.search(date_pattern, text, re.IGNORECASE)
        
        if match:
            month = match.group(1)
            phase = match.group(2)
            day = match.group(3)
            year = match.group(4)
            return f"{month} {phase} {day}, Samvat {year}"
        
        return ""
    
    def extract_setting(self, text: str) -> str:
        """
        Extract setting/location information from passage
        """
        # Patterns for setting descriptions
        setting_patterns = [
            r'sitting\s+(?:in|on)\s+([^.]+)',
            r'was\s+(?:seated|sitting)\s+(?:in|on)\s+([^.]+)',
            r'(?:in|at)\s+the\s+(?:darbar|darbir|court)\s+(?:of|in)\s+([^.]+)',
        ]
        
        for pattern in setting_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""


# Global instance
_text_cleaner: VachanamrutTextCleaner = None


def get_text_cleaner() -> VachanamrutTextCleaner:
    """Get global text cleaner instance"""
    global _text_cleaner
    if _text_cleaner is None:
        _text_cleaner = VachanamrutTextCleaner()
    return _text_cleaner

