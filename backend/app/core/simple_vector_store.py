from typing import List, Dict, Optional, Tuple
import json
import math
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class SimpleDocument:
    """Simple document representation"""
    id: str
    content: str
    reference: str
    page_number: int
    themes: List[str]
    embedding_hash: str  # Simple hash instead of full embedding
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimpleDocument':
        return cls(**data)

class BasicVectorStore:
    """
    Ultra-simple vector store using text similarity
    Claude Projects approach: No external dependencies, just text matching
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.documents: List[SimpleDocument] = []
        self.storage_path = storage_path or "data/processed/simple_store.json"
        
    def add_documents(self, documents: List[SimpleDocument]) -> None:
        """Add documents to the store"""
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to store")
        
    def save(self) -> None:
        """Save store to disk"""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'documents': [doc.to_dict() for doc in self.documents],
            'total_documents': len(self.documents)
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self) -> bool:
        """Load store from disk"""
        if not Path(self.storage_path).exists():
            return False
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.documents = [SimpleDocument.from_dict(doc_data) for doc_data in data['documents']]
            logger.info(f"Loaded {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading store: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        theme_filter: Optional[str] = None
    ) -> List[Tuple[SimpleDocument, float]]:
        """
        Search using simple text similarity
        """
        if not self.documents:
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            # Apply theme filter if specified
            if theme_filter and theme_filter.lower() not in [t.lower() for t in doc.themes]:
                continue
            
            # Simple word overlap similarity
            doc_words = set(doc.content.lower().split())
            common_words = query_words.intersection(doc_words)
            
            if common_words:
                # Jaccard similarity
                similarity = len(common_words) / len(query_words.union(doc_words))
                
                # Boost for theme matches
                for theme in doc.themes:
                    if any(theme_word in query.lower() for theme_word in theme.split()):
                        similarity += 0.2
                
                # Boost for reference matches
                if any(word in doc.reference.lower() for word in query_words):
                    similarity += 0.1
                
                results.append((doc, min(similarity, 1.0)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_by_theme(self, theme: str) -> List[SimpleDocument]:
        """Get documents by theme"""
        return [
            doc for doc in self.documents 
            if theme.lower() in [t.lower() for t in doc.themes]
        ]
    
    def get_stats(self) -> Dict:
        """Get store statistics"""
        if not self.documents:
            return {'total_documents': 0}
        
        theme_count = {}
        reference_count = {}
        
        for doc in self.documents:
            for theme in doc.themes:
                theme_count[theme] = theme_count.get(theme, 0) + 1
            reference_count[doc.reference] = reference_count.get(doc.reference, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'unique_themes': len(theme_count),
            'unique_references': len(reference_count),
            'theme_distribution': dict(sorted(theme_count.items(), key=lambda x: x[1], reverse=True)),
            'most_common_references': dict(sorted(reference_count.items(), key=lambda x: x[1], reverse=True)[:10])
        }

def create_sample_store() -> BasicVectorStore:
    """Create sample store with spiritual content"""
    
    store = BasicVectorStore()
    
    # Sample Vachanamrut passages
    sample_passages = [
        {
            "id": "gadhada_1_1",
            "content": "One who has firm faith in God and His Sant never experiences any difficulty, regardless of whatever severe calamities he may encounter. Why? Because he is confident that God is the all-doer and the cause of all causes; and also that God is extremely loving towards His devotees. Therefore, such a person never experiences any difficulties.",
            "reference": "Gadhada I-1",
            "page_number": 15,
            "themes": ["faith", "devotion", "surrender", "trust"]
        },
        {
            "id": "gadhada_1_2", 
            "content": "A person who has love for God should not allow his mind to become attached to any object other than God. Even if his mind does get attached to some other object, he should understand that attachment to be a flaw and should attempt to remove it. In this manner, he should maintain love for God alone.",
            "reference": "Gadhada I-2",
            "page_number": 16,
            "themes": ["devotion", "detachment", "love", "mind"]
        },
        {
            "id": "sarangpur_5",
            "content": "The key to spiritual progress is constant remembrance of God. One should always keep God in mind during all activities - whether eating, drinking, walking, or resting. This continuous awareness leads to realization of God's presence in all situations.",
            "reference": "Sarangpur-5",
            "page_number": 87,
            "themes": ["meditation", "devotion", "remembrance", "awareness"]
        },
        {
            "id": "vadtal_18",
            "content": "True surrender means offering all of one's actions, thoughts, and desires to God. When a devotee completely surrenders to God's will, he experiences divine grace and protection in all circumstances. Surrender is the ultimate spiritual practice.",
            "reference": "Vadtal-18", 
            "page_number": 156,
            "themes": ["surrender", "grace", "devotion", "practice"]
        },
        {
            "id": "ahmedabad_3",
            "content": "Service to God and His devotees should be performed without any expectation of reward. Selfless service purifies the heart and brings one closer to divine realization. True service is done with humility and love.",
            "reference": "Ahmedabad-3",
            "page_number": 201,
            "themes": ["service", "selflessness", "purification", "humility"]
        },
        {
            "id": "loyej_7",
            "content": "Knowledge of the true nature of God, soul, and maya is essential for spiritual liberation. One should study the scriptures, contemplate their meaning, and seek guidance from realized souls to gain this knowledge.",
            "reference": "Loyej-7",
            "page_number": 243,
            "themes": ["knowledge", "wisdom", "study", "guru"]
        },
        {
            "id": "gadhada_2_13",
            "content": "Dharma means righteousness in thought, word, and deed. A devotee should always follow dharma, even in difficult circumstances. Adherence to dharma protects one from spiritual downfall and ensures divine grace.",
            "reference": "Gadhada II-13",
            "page_number": 298,
            "themes": ["dharma", "righteousness", "conduct", "protection"]
        },
        {
            "id": "kariyani_2",
            "content": "The company of true devotees is invaluable for spiritual growth. In satsang, one learns spiritual principles, gains inspiration, and receives blessings. One should always seek and cherish the association of genuine devotees.",
            "reference": "Kariyani-2",
            "page_number": 334,
            "themes": ["satsang", "devotees", "association", "growth"]
        }
    ]
    
    # Create documents
    documents = []
    for passage in sample_passages:
        # Simple hash for "embedding"
        content_hash = hashlib.md5(passage["content"].encode()).hexdigest()[:8]
        
        doc = SimpleDocument(
            id=passage["id"],
            content=passage["content"],
            reference=passage["reference"],
            page_number=passage["page_number"],
            themes=passage["themes"],
            embedding_hash=content_hash
        )
        documents.append(doc)
    
    store.add_documents(documents)
    return store