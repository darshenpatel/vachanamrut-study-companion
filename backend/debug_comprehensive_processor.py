"""
Debug and fix the comprehensive processor
"""

import pdfplumber
import re
from pathlib import Path

def debug_pdf_structure():
    """
    Debug the PDF structure to understand extraction issues
    """
    pdf_path = "/Users/darshen/Documents/vachanamrut-companion/backend/data/raw/TheVachanamrut-searchable.pdf"
    
    print("=== DEBUGGING PDF STRUCTURE ===")
    
    reference_patterns = [
        r'(Gadhad[aā5dlti]+\s+[IVX]+\s*[-\.\s]*\d+)',  # Gadhada with Roman numerals
        r'(S[aāii]*rangpur\s*[-\.\s]*\d+)',            # Sarangpur
        r'(K[aāii]*riy[aāii]*ni\s*[-\.\s]*\d+)',       # Kariyani  
        r'(Loy[aāii]*\s*[-\.\s]*\d+)',                 # Loya
        r'(Panch[aāii]*l[aāii]*\s*[-\.\s]*\d+)',       # Panchala
        r'(Vart[aāii]*l\s*[-\.\s]*\d+)',               # Vartal
        r'(Ahmedabad\s*[-\.\s]*\d+)',                  # Ahmedabad
    ]
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages: {total_pages}")
        
        references_found = {}
        
        # Scan pages 40-100 for content
        for page_idx in range(39, min(100, total_pages)):  # 0-indexed
            page_num = page_idx + 1
            page = pdf.pages[page_idx]
            text = page.extract_text()
            
            if not text:
                continue
            
            # Look for any references
            page_refs = []
            for pattern in reference_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    page_refs.append(match.strip())
            
            if page_refs:
                references_found[page_num] = page_refs
                print(f"\\nPage {page_num}: {len(page_refs)} references found")
                for ref in page_refs[:3]:  # Show first 3
                    print(f"  - {ref}")
                
                # Show some content context
                if 'Gadhada' in text:
                    gadhada_pos = text.find('Gadhada')
                    context = text[max(0, gadhada_pos-50):gadhada_pos+200]
                    print(f"  Context: ...{context}...")
        
        print(f"\\n=== SUMMARY ===")
        print(f"Total pages with references: {len(references_found)}")
        
        # Count by type
        all_refs = []
        for page_refs in references_found.values():
            all_refs.extend(page_refs)
        
        print(f"Total references found: {len(all_refs)}")
        
        # Group by location
        locations = {}
        for ref in all_refs:
            location = ref.split()[0].lower()
            if location not in locations:
                locations[location] = 0
            locations[location] += 1
        
        print("References by location:")
        for location, count in sorted(locations.items()):
            print(f"  {location}: {count}")
        
        return references_found

def test_simple_extraction():
    """
    Test a simpler extraction approach
    """
    pdf_path = "/Users/darshen/Documents/vachanamrut-companion/backend/data/raw/TheVachanamrut-searchable.pdf"
    
    print("\\n=== TESTING SIMPLE EXTRACTION ===")
    
    extracted = []
    
    with pdfplumber.open(pdf_path) as pdf:
        # Focus on pages 45-150 where we know content exists
        for page_idx in range(44, min(150, len(pdf.pages))):
            page_num = page_idx + 1
            page = pdf.pages[page_idx]
            text = page.extract_text()
            
            if not text:
                continue
            
            # Simple pattern matching
            gadhada_matches = re.findall(r'Gadhad[aā5dlti]+\s+[IVX]+\s*[-\.\s]*\d+', text, re.IGNORECASE)
            sarangpur_matches = re.findall(r'S[aāii]*rangpur\s*[-\.\s]*\d+', text, re.IGNORECASE)
            
            all_matches = gadhada_matches + sarangpur_matches
            
            for match in all_matches:
                # Get some context around the match
                match_pos = text.find(match)
                if match_pos != -1:
                    content = text[match_pos:match_pos + 500]
                    
                    extracted.append({
                        'reference': match.strip(),
                        'page': page_num,
                        'content_preview': content[:200] + '...'
                    })
    
    print(f"Simple extraction found: {len(extracted)} discourses")
    
    # Show samples
    for i, item in enumerate(extracted[:10]):
        print(f"{i+1}. {item['reference']} (page {item['page']})")
        print(f"   Content: {item['content_preview'][:100]}...")
        print()
    
    return extracted

def create_working_processor():
    """
    Create a working version based on debugging results
    """
    print("\\n=== CREATING WORKING PROCESSOR ===")
    
    # This will be our simplified but working approach
    processor_code = '''
def extract_vachanamrut_discourses(pdf_path):
    """
    Simplified but working extraction approach
    """
    import pdfplumber
    import re
    import json
    from datetime import datetime
    
    discourses = []
    
    # Simple but effective patterns
    patterns = [
        r'Gadhad[aā5dlti]+\s+[IVX]+\s*[-\.\s]*\d+',
        r'S[aāii]*rangpur\s*[-\.\s]*\d+',
        r'K[aāii]*riy[aāii]*ni\s*[-\.\s]*\d+',
        r'Loy[aāii]*\s*[-\.\s]*\d+',
        r'Panch[aāii]*l[aāii]*\s*[-\.\s]*\d+',
        r'Vart[aāii]*l\s*[-\.\s]*\d+',
        r'Ahmedabad\s*[-\.\s]*\d+'
    ]
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in range(44, len(pdf.pages)):  # Start from page 45
            page_num = page_idx + 1
            page = pdf.pages[page_idx]
            text = page.extract_text()
            
            if not text:
                continue
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    ref = match.group().strip()
                    start_pos = match.start()
                    
                    # Extract content after reference
                    content = text[start_pos:start_pos + 1500]
                    
                    # Clean the reference
                    clean_ref = clean_reference(ref)
                    
                    if clean_ref and len(content) > 100:
                        discourses.append({
                            'id': clean_ref.lower().replace(' ', '_').replace('-', '_'),
                            'reference': clean_ref,
                            'content': content.strip(),
                            'page_number': page_num,
                            'chapter': extract_chapter(clean_ref),
                            'quality_score': calculate_quality(content)
                        })
    
    return discourses

def clean_reference(ref):
    """Clean and normalize reference"""
    # Fix OCR errors
    ref = ref.replace('5', 'ā').replace('ii', 'ā')
    
    # Normalize format
    if 'gadhada' in ref.lower():
        match = re.match(r'gadhad[aā]*\s*([ivx]+)\s*[-\.\s]*(\d+)', ref, re.IGNORECASE)
        if match:
            return f"Gadhada {match.group(1).upper()}-{match.group(2)}"
    elif 'sarangpur' in ref.lower():
        match = re.search(r'(\d+)', ref)
        if match:
            return f"Sarangpur-{match.group(1)}"
    
    return ref

def extract_chapter(ref):
    """Extract chapter name"""
    if 'gadhada' in ref.lower() and 'i' in ref.lower():
        if 'iii' in ref.lower():
            return 'Gadhada III'
        elif 'ii' in ref.lower():
            return 'Gadhada II'
        else:
            return 'Gadhada I'
    elif 'sarangpur' in ref.lower():
        return 'Sarangpur'
    return 'Unknown'

def calculate_quality(content):
    """Simple quality calculation"""
    score = 0.3  # Base score
    
    if len(content) > 500:
        score += 0.2
    if len(content) > 1000:
        score += 0.2
    
    spiritual_words = ['god', 'bhagwan', 'devotion', 'dharma', 'soul', 'meditation']
    for word in spiritual_words:
        if word in content.lower():
            score += 0.05
    
    return min(score, 1.0)
'''
    
    print("Working processor code created")
    return processor_code

if __name__ == "__main__":
    # Debug the PDF structure
    references = debug_pdf_structure()
    
    # Test simple extraction
    extracted = test_simple_extraction()
    
    # Create working processor
    processor_code = create_working_processor()
    
    print(f"\\n=== FINAL RESULTS ===")
    print(f"References found in PDF: {sum(len(refs) for refs in references.values())}")
    print(f"Simple extraction result: {len(extracted)} discourses")
    print("Working processor code created")