from typing import List, Optional, Dict
import logging

from app.schemas.theme import ThemeDetail

logger = logging.getLogger(__name__)

class ThemeService:
    def __init__(self):
        self.theme_data = {
            "devotion": {
                "description": "The practice of loving dedication and surrender to God, characterized by unwavering faith and constant remembrance.",
                "keywords": ["bhakti", "love", "surrender", "worship", "dedication"],
                "related_passages": ["Gadhada I-1", "Sarangpur-5", "Kariyani-2"],
                "related_themes": ["faith", "surrender", "service", "love"]
            },
            "faith": {
                "description": "Complete trust and confidence in God and His Sant, without doubt or hesitation.",
                "keywords": ["trust", "confidence", "belief", "conviction", "certainty"],  
                "related_passages": ["Gadhada I-15", "Gadhada II-13", "Vadtal-18"],
                "related_themes": ["devotion", "surrender", "guru", "satsang"]
            },
            "surrender": {
                "description": "Complete submission of one's will to God, abandoning ego and personal desires.",
                "keywords": ["submission", "ego", "will", "abandon", "humility"],
                "related_passages": ["Gadhada I-7", "Sarangpur-10", "Ahmedabad-3"],
                "related_themes": ["devotion", "faith", "detachment", "service"]
            },
            "service": {
                "description": "Selfless action performed for God and His devotees without expectation of reward.",
                "keywords": ["seva", "selfless", "action", "devotees", "sacrifice"],
                "related_passages": ["Gadhada II-31", "Vadtal-5", "Loyej-15"],
                "related_themes": ["devotion", "surrender", "detachment", "dharma"]
            },
            "knowledge": {
                "description": "Understanding of the true nature of God, soul, maya, and the path to liberation.",
                "keywords": ["understanding", "wisdom", "truth", "realization", "enlightenment"],
                "related_passages": ["Gadhada I-46", "Sarangpur-1", "Panchala-7"],
                "related_themes": ["meditation", "detachment", "guru", "dharma"]
            }
        }
    
    async def list_all_themes(self) -> List[str]:
        """Get list of all available themes"""
        return list(self.theme_data.keys())
    
    async def get_theme_detail(self, theme_name: str) -> Optional[ThemeDetail]:
        """Get detailed information about a specific theme"""
        theme_info = self.theme_data.get(theme_name.lower())
        
        if not theme_info:
            return None
            
        return ThemeDetail(
            name=theme_name,
            description=theme_info["description"],
            keywords=theme_info["keywords"],
            related_passages=theme_info["related_passages"], 
            related_themes=theme_info["related_themes"]
        )
    
    async def get_related_themes(self, message: str) -> List[str]:
        """Get themes related to a user message"""
        # Simple keyword matching for now
        # This would be enhanced with semantic similarity later
        message_lower = message.lower()
        related = []
        
        for theme, data in self.theme_data.items():
            if any(keyword in message_lower for keyword in data["keywords"]):
                related.append(theme)
        
        return related[:3]  # Return top 3 matches

# Dependency injection
def get_theme_service() -> ThemeService:
    return ThemeService()