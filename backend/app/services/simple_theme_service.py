from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class SimpleThemeService:
    def __init__(self):
        self.theme_data = {
            "devotion": {
                "description": "The practice of loving dedication and surrender to God, characterized by unwavering faith and constant remembrance.",
                "keywords": ["bhakti", "love", "surrender", "worship", "dedication", "devotee"],
                "related_passages": ["Gadhada I-1", "Sarangpur-5", "Kariyani-2"],
                "related_themes": ["faith", "surrender", "service", "love"]
            },
            "faith": {
                "description": "Complete trust and confidence in God and His Sant, without doubt or hesitation.",
                "keywords": ["trust", "confidence", "belief", "conviction", "certainty", "faith"],  
                "related_passages": ["Gadhada I-15", "Gadhada II-13", "Vadtal-18"],
                "related_themes": ["devotion", "surrender", "guru", "satsang"]
            },
            "surrender": {
                "description": "Complete submission of one's will to God, abandoning ego and personal desires.",
                "keywords": ["submission", "ego", "will", "abandon", "humility", "surrender"],
                "related_passages": ["Gadhada I-7", "Sarangpur-10", "Ahmedabad-3"],
                "related_themes": ["devotion", "faith", "detachment", "service"]
            },
            "service": {
                "description": "Selfless action performed for God and His devotees without expectation of reward.",
                "keywords": ["seva", "selfless", "action", "devotees", "sacrifice", "serve", "help"],
                "related_passages": ["Gadhada II-31", "Vadtal-5", "Loyej-15"],
                "related_themes": ["devotion", "surrender", "detachment", "dharma"]
            },
            "knowledge": {
                "description": "Understanding of the true nature of God, soul, maya, and the path to liberation.",
                "keywords": ["understanding", "wisdom", "truth", "realization", "enlightenment", "study", "learn"],
                "related_passages": ["Gadhada I-46", "Sarangpur-1", "Panchala-7"],
                "related_themes": ["meditation", "detachment", "guru", "dharma"]
            },
            "detachment": {
                "description": "Freedom from attachment to material objects and outcomes, while maintaining devotion to God.",
                "keywords": ["detachment", "attachment", "worldly", "material", "renunciation", "desires"],
                "related_passages": ["Gadhada I-2", "Vadtal-7", "Loyej-3"],
                "related_themes": ["knowledge", "surrender", "devotion", "dharma"]
            },
            "dharma": {
                "description": "Righteous conduct and moral principles that guide spiritual and worldly life.",
                "keywords": ["dharma", "righteousness", "duty", "moral", "virtue", "conduct", "right"],
                "related_passages": ["Gadhada II-13", "Ahmedabad-5", "Vadtal-12"],
                "related_themes": ["service", "knowledge", "detachment", "guru"]
            },
            "meditation": {
                "description": "Focused concentration on God's form, attributes, or divine presence.",
                "keywords": ["meditation", "concentration", "focus", "contemplation", "remembrance", "mind"],
                "related_passages": ["Sarangpur-5", "Gadhada I-32", "Loyej-11"],
                "related_themes": ["devotion", "knowledge", "remembrance", "practice"]
            },
            "guru": {
                "description": "The spiritual master who guides devotees on the path to God-realization.",
                "keywords": ["guru", "master", "teacher", "guide", "sant", "spiritual teacher"],
                "related_passages": ["Gadhada I-54", "Vadtal-3", "Kariyani-8"],
                "related_themes": ["faith", "knowledge", "satsang", "guidance"]
            },
            "satsang": {
                "description": "The company and association of true devotees and spiritual seekers.",
                "keywords": ["satsang", "company", "association", "fellowship", "devotees", "spiritual company"],
                "related_passages": ["Kariyani-2", "Gadhada II-25", "Loyej-9"],
                "related_themes": ["guru", "devotion", "faith", "growth"]
            }
        }
    
    async def list_all_themes(self) -> List[str]:
        """Get list of all available themes"""
        return list(self.theme_data.keys())
    
    async def get_theme_detail(self, theme_name: str) -> Optional[Dict]:
        """Get detailed information about a specific theme"""
        theme_info = self.theme_data.get(theme_name.lower())
        
        if not theme_info:
            return None
            
        return {
            'name': theme_name,
            'description': theme_info["description"],
            'keywords': theme_info["keywords"],
            'related_passages': theme_info["related_passages"], 
            'related_themes': theme_info["related_themes"]
        }
    
    async def get_related_themes(self, message: str) -> List[str]:
        """Get themes related to a user message"""
        message_lower = message.lower()
        related = []
        scored_themes = []
        
        for theme, data in self.theme_data.items():
            score = 0
            
            # Check for direct theme name mention
            if theme in message_lower:
                score += 10
            
            # Check for keyword matches
            for keyword in data["keywords"]:
                if keyword in message_lower:
                    score += 2
            
            # Check for related concept words
            concept_words = [
                'god', 'divine', 'spiritual', 'practice', 'path', 'teachings',
                'scripture', 'guidance', 'wisdom', 'growth', 'progress'
            ]
            for word in concept_words:
                if word in message_lower:
                    score += 1
                    
            if score > 0:
                scored_themes.append((theme, score))
        
        # Sort by score and return top 3
        scored_themes.sort(key=lambda x: x[1], reverse=True)
        related = [theme for theme, score in scored_themes[:3]]
        
        # If no themes found through scoring, return some default spiritual themes
        if not related:
            if any(word in message_lower for word in ['help', 'guidance', 'how', 'what', 'why']):
                related = ['devotion', 'faith', 'knowledge']
        
        return related

# Dependency injection
def get_simple_theme_service() -> SimpleThemeService:
    return SimpleThemeService()