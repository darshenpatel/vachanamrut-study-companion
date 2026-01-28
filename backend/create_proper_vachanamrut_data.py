"""
Create properly structured Vachanamrut data for MVP
Addresses OCR errors and extracts actual discourse content
"""

import pdfplumber
import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class VachanamrutDiscourse:
    reference: str
    content: str
    page_number: int
    chapter: str
    discourse_number: int
    title: str = ""
    date_info: str = ""
    setting: str = ""
    quality_score: float = 0.0


class VachanamrutExtractor:
    """
    Production-ready extractor that handles OCR errors and creates structured data
    """
    
    def __init__(self):
        self.discourses: List[VachanamrutDiscourse] = []
        
        # OCR correction mappings for common errors
        self.ocr_fixes = {
            'Gadhad5': 'Gadhada',
            'Gadhadii': 'Gadhada', 
            'Gadhadl': 'Gadhada',
            'Gadhadti': 'Gadhada',
            'Gadhadi': 'Gadhada',
            'MahGriij': 'Maharaj',
            'MahlrFij': 'Maharaj',
            'Shriji': 'Shriji',
            'VachanHmrut': 'Vachanamrut',
            'VachanPmrut': 'Vachanamrut',
            'Siirangpur': 'Sarangpur',
            'Akshardhlm': 'Akshardham'
        }
        
        # Reference patterns accounting for OCR errors
        self.reference_patterns = [
            # Gadhada with various OCR errors
            r'(Gadhad[a5āădlti]+\s+[IVX]+[-\.\s]*\d+)',
            # Other locations
            r'(S[aāi]*rangpur[-\.\s]*\d+)',
            r'(Vadtal[-\.\s]*\d+)', 
            r'(Ahmedabad[-\.\s]*\d+)',
            r'(K[aā]riy[aā]ni[-\.\s]*\d+)',
            r'(Loy[aā][-\.\s]*\d+)',
            r'(Panch[aā]l[aā]?[-\.\s]*\d+)',
            r'(Vart[aā]l[-\.\s]*\d+)',
        ]
    
    def fix_ocr_errors(self, text: str) -> str:
        """Apply OCR fixes to improve text quality"""
        for error, correction in self.ocr_fixes.items():
            text = re.sub(r'\b' + re.escape(error) + r'\b', correction, text, flags=re.IGNORECASE)
        return text
    
    def extract_from_pdf(self, pdf_path: str, start_page: int = 45, max_pages: int = 100) -> List[VachanamrutDiscourse]:
        """
        Extract Vachanamrut discourses from PDF
        """
        print(f"Extracting Vachanamrut discourses starting from page {start_page}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = min(len(pdf.pages), start_page + max_pages)
                
                for page_idx in range(start_page - 1, total_pages):
                    page = pdf.pages[page_idx]
                    text = page.extract_text()
                    
                    if not text:
                        continue
                    
                    # Fix OCR errors
                    cleaned_text = self.fix_ocr_errors(text)
                    
                    # Look for discourse references
                    for pattern in self.reference_patterns:
                        matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
                        
                        for match in matches:
                            discourse = self._extract_discourse_from_match(
                                match, cleaned_text, page_idx + 1
                            )
                            
                            if discourse and self._validate_discourse(discourse):
                                self.discourses.append(discourse)
                                print(f"Extracted: {discourse.reference} from page {discourse.page_number}")
                
                print(f"Total discourses extracted: {len(self.discourses)}")
                return self.discourses
                
        except Exception as e:
            print(f"Extraction failed: {e}")
            return []
    
    def _extract_discourse_from_match(self, match, text: str, page_number: int) -> VachanamrutDiscourse:
        """Extract discourse content from a reference match"""
        reference = match.group(1).strip()
        
        # Clean up the reference
        reference = self._normalize_reference(reference)
        
        # Get content starting from the reference
        start_pos = match.start()
        
        # Look for content after the reference (next 2000 chars or until next reference)
        content_end = min(start_pos + 2000, len(text))
        raw_content = text[start_pos:content_end]
        
        # Clean and structure the content
        content = self._clean_discourse_content(raw_content)
        
        # Extract metadata
        title = self._extract_title(raw_content)
        date_info = self._extract_date_info(raw_content)
        setting = self._extract_setting(raw_content)
        
        # Parse reference components
        chapter, discourse_num = self._parse_reference(reference)
        
        # Calculate quality score
        quality = self._calculate_quality_score(content, reference)
        
        return VachanamrutDiscourse(
            reference=reference,
            content=content,
            page_number=page_number,
            chapter=chapter,
            discourse_number=discourse_num,
            title=title,
            date_info=date_info,
            setting=setting,
            quality_score=quality
        )
    
    def _normalize_reference(self, reference: str) -> str:
        """Normalize reference format"""
        # Fix common OCR errors in reference
        reference = self.fix_ocr_errors(reference)
        
        # Standardize format: "Gadhada I-8"
        match = re.match(r'([A-Za-z]+)\s*([IVX]*)\s*[-\.\s]*(\d+)', reference, re.IGNORECASE)
        if match:
            location = match.group(1).title()
            roman = match.group(2).upper() if match.group(2) else ""
            number = match.group(3)
            
            if roman:
                return f"{location} {roman}-{number}"
            else:
                return f"{location}-{number}"
        
        return reference
    
    def _clean_discourse_content(self, raw_content: str) -> str:
        """Clean and structure discourse content"""
        lines = raw_content.split('\n')
        cleaned_lines = []
        
        skip_header = True
        for line in lines:
            line = line.strip()
            
            # Skip header material (reference, title, date)
            if skip_header:
                if any(indicator in line.lower() for indicator in ['was sitting', 'assembly', 'thereupon', 'then']):
                    skip_header = False
                continue
            
            # Skip very short lines, page numbers, and formatting artifacts
            if len(line) < 10 or re.match(r'^\d+$', line) or line.startswith('1 1'):
                continue
            
            # Clean up common OCR issues
            line = re.sub(r'\s+', ' ', line)
            line = re.sub(r'["""]', '"', line)
            
            if line:
                cleaned_lines.append(line)
        
        # Join lines intelligently
        content = ' '.join(cleaned_lines)
        
        # Break into paragraphs at numbered sections
        content = re.sub(r'(\d+)\s+', r'\n\1. ', content)
        
        return content.strip()
    
    def _extract_title(self, text: str) -> str:
        """Extract discourse title if present"""
        lines = text.split('\n')[:5]  # Check first few lines
        
        for line in lines:
            line = line.strip()
            # Title characteristics: capitalized, not too short, not a date/reference
            if (len(line) > 10 and len(line) < 100 and 
                line[0].isupper() and 
                not re.search(r'\d{4}|\bSamvat\b|\bOn\b', line)):
                return line
        
        return ""
    
    def _extract_date_info(self, text: str) -> str:
        """Extract date information"""
        # Look for date patterns
        date_pattern = r'On\s+([^,]+,\s+Samvat\s+\d+)\s*\[([^\]]+)\]'
        match = re.search(date_pattern, text)
        
        if match:
            return f"{match.group(1)} ({match.group(2)})"
        
        return ""
    
    def _extract_setting(self, text: str) -> str:
        """Extract assembly setting information"""
        # Look for setting description
        setting_patterns = [
            r'was sitting in ([^\.]+\.)',
            r'assembly of ([^\.]+\.)'
        ]
        
        for pattern in setting_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                setting = match.group(1).strip()
                if len(setting) > 10 and len(setting) < 150:
                    return setting
        
        return ""
    
    def _parse_reference(self, reference: str) -> Tuple[str, int]:
        """Parse reference into chapter and discourse number"""
        match = re.match(r'([A-Za-z\s]+[IVX]*)\s*[-\s]*(\d+)', reference)
        if match:
            chapter = match.group(1).strip()
            discourse_num = int(match.group(2))
            return chapter, discourse_num
        return reference, 0
    
    def _calculate_quality_score(self, content: str, reference: str) -> float:
        """Calculate content quality score"""
        score = 0.0
        
        # Length check
        if len(content) > 100:
            score += 0.2
        if len(content) > 500:
            score += 0.3
        
        # Spiritual content keywords
        spiritual_words = ['god', 'bhagwan', 'swaminarayan', 'devotion', 'dharma', 
                          'liberation', 'soul', 'assembly', 'devotee', 'meditation',
                          'worship', 'spiritual', 'moksha', 'atma', 'paramatma']
        
        word_count = sum(1 for word in spiritual_words if word in content.lower())
        score += min(word_count * 0.05, 0.3)
        
        # Structure indicators
        if re.search(r'\d+\.', content):  # Numbered points
            score += 0.1
        
        # Reference consistency
        if any(part.lower() in content.lower() for part in reference.split()):
            score += 0.1
        
        return min(score, 1.0)
    
    def _validate_discourse(self, discourse: VachanamrutDiscourse) -> bool:
        """Validate if discourse meets quality standards"""
        # Must have substantial content
        if len(discourse.content) < 50:
            return False
        
        # Must have reasonable quality score
        if discourse.quality_score < 0.2:
            return False
        
        # Must have valid reference format
        if not re.match(r'[A-Za-z]+.*\d+', discourse.reference):
            return False
        
        return True
    
    def export_structured_data(self, output_path: str):
        """Export structured data for LLM use"""
        # Sort discourses by chapter and number
        sorted_discourses = sorted(
            self.discourses, 
            key=lambda d: (d.chapter, d.discourse_number)
        )
        
        export_data = {
            'metadata': {
                'total_discourses': len(sorted_discourses),
                'extraction_method': 'OCR-corrected structured extraction',
                'format_version': '1.0',
                'quality_stats': {
                    'avg_quality_score': sum(d.quality_score for d in sorted_discourses) / len(sorted_discourses),
                    'avg_content_length': sum(len(d.content) for d in sorted_discourses) / len(sorted_discourses),
                    'chapters_found': len(set(d.chapter for d in sorted_discourses))
                }
            },
            'passages': []  # Use 'passages' key for compatibility with existing system
        }
        
        for discourse in sorted_discourses:
            passage_data = {
                'id': f"{discourse.reference.replace(' ', '_').replace('-', '_')}",
                'reference': discourse.reference,
                'content': discourse.content,
                'page_number': discourse.page_number,
                'chapter': discourse.chapter,
                'title': discourse.title,
                'date_info': discourse.date_info,
                'setting': discourse.setting,
                'quality_score': discourse.quality_score,
                'word_count': len(discourse.content.split())
            }
            export_data['passages'].append(passage_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"Structured data exported to {output_path}")
        return export_data


def main():
    """Main extraction process"""
    extractor = VachanamrutExtractor()
    
    pdf_path = "/Users/darshen/Documents/vachanamrut-companion/backend/data/raw/TheVachanamrut-searchable.pdf"
    output_path = "/Users/darshen/Documents/vachanamrut-companion/backend/data/processed/structured_vachanamrut.json"
    
    # Extract discourses
    discourses = extractor.extract_from_pdf(pdf_path, start_page=45, max_pages=200)
    
    if discourses:
        # Export structured data
        export_data = extractor.export_structured_data(output_path)
        
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Total discourses extracted: {len(discourses)}")
        print(f"Average quality score: {export_data['metadata']['quality_stats']['avg_quality_score']:.2f}")
        print(f"Average content length: {export_data['metadata']['quality_stats']['avg_content_length']:.0f} chars")
        print(f"Chapters found: {export_data['metadata']['quality_stats']['chapters_found']}")
        
        # Show samples
        print(f"\nSample extractions:")
        for i, discourse in enumerate(discourses[:3]):
            print(f"{i+1}. {discourse.reference}")
            print(f"   Page {discourse.page_number} | Quality: {discourse.quality_score:.2f}")
            print(f"   Title: {discourse.title or 'None'}")
            print(f"   Content: {discourse.content[:100]}...")
            print()
        
        return True
    else:
        print("No discourses extracted - check PDF path and extraction logic")
        return False


if __name__ == "__main__":
    main()