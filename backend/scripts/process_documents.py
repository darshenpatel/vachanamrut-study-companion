#!/usr/bin/env python3
"""
Script to process Vachanamrut PDF and generate embeddings
Usage: python scripts/process_documents.py [pdf_path]
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.retrieval import DocumentRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main processing function"""
    
    # Default PDF path - you'll need to add your Vachanamrut PDF here
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/vachanamrut.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Creating sample data structure...")
        
        # Create sample content to test the system
        sample_content = """
        Gadhada I-1
        
        Once, in the Durbar of Shriji Maharaj in Gadhada, many devotees had gathered. At that time, Shriji Maharaj said, "One who has firm faith in God and His Sant never experiences any difficulty, regardless of whatever severe calamities he may encounter. Why? Because he is confident that God is the all-doer and the cause of all causes; and also that God is extremely loving towards His devotees. Therefore, such a person never experiences any difficulties."
        
        Gadhada I-2
        
        On another occasion in Gadhada, Shriji Maharaj said, "A person who has love for God should not allow his mind to become attached to any object other than God. Even if his mind does get attached to some other object, he should understand that attachment to be a flaw and should attempt to remove it. In this manner, he should maintain love for God alone."
        
        Sarangpur-5
        
        In Sarangpur, Shriji Maharaj explained, "The key to spiritual progress is constant remembrance of God. One should always keep God in mind during all activities - whether eating, drinking, walking, or resting. This continuous awareness leads to realization of God's presence in all situations."
        """
        
        # Create sample PDF-like structure
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        with open("data/raw/sample_vachanamrut.txt", "w") as f:
            f.write(sample_content)
        
        print(f"PDF not found at {pdf_path}")
        print("Created sample content for testing.")
        print("To use with real PDF, place your Vachanamrut PDF at data/raw/vachanamrut.pdf")
        return
    
    print(f"Processing PDF: {pdf_path}")
    
    # Initialize retriever
    retriever = DocumentRetriever()
    
    try:
        result = await retriever.initialize(pdf_path, force_reprocess=True)
        
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(f"Status: {result['status']}")
        print(f"Total documents: {result['total_documents']}")
        
        if 'pdf_stats' in result:
            stats = result['pdf_stats']
            print(f"Total characters: {stats['total_characters']:,}")
            print(f"Average length: {stats['average_length']} chars")
            print(f"Unique references: {stats['unique_references']}")
            print(f"Themes found: {len(stats['themes_distribution'])}")
            
            if stats['themes_distribution']:
                print("\nTheme distribution:")
                for theme, count in sorted(stats['themes_distribution'].items(), 
                                         key=lambda x: x[1], reverse=True):
                    print(f"  {theme}: {count} passages")
        
        # Test search functionality
        print("\n" + "="*50)
        print("TESTING SEARCH")
        print("="*50)
        
        test_queries = [
            "How should I practice devotion?",
            "What is the role of faith in spiritual life?",
            "How to remember God constantly?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = await retriever.search(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['reference']} (score: {result['similarity_score']:.3f})")
                print(f"     {result['content'][:100]}...")
        
        # System status
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        status = retriever.get_system_status()
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print("\n✅ Processing complete! System ready for queries.")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        logging.exception("Detailed error:")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))