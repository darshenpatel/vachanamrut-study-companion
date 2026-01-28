"""
Curated Theme-to-Passage Mappings for Vachanamrut
Based on the PRD requirement for 10 core spiritual themes
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ThemeMapping:
    """Represents a curated spiritual theme with associated content"""
    name: str
    description: str
    keywords: List[str]
    related_themes: List[str]
    key_passages: List[str]  # Vachanamrut references
    guidance: str  # Practical spiritual guidance


# Core 10 spiritual themes as defined in the PRD
SPIRITUAL_THEMES: Dict[str, ThemeMapping] = {
    "devotion": ThemeMapping(
        name="devotion",
        description="The path of loving devotion (bhakti) to God, encompassing worship, love, and dedication to the Divine.",
        keywords=["devotion", "bhakti", "love", "worship", "devotee", "bhakta", "adoration", "dedication"],
        related_themes=["faith", "surrender", "service", "satsang"],
        key_passages=["Gadhada I-1", "Gadhada I-2", "Sarangpur-5", "Gadhada I-72"],
        guidance="Cultivate love for God through daily prayer, remembrance, and seeing the divine in all beings. Devotion should be pure, without material expectations."
    ),
    
    "faith": ThemeMapping(
        name="faith",
        description="Unwavering trust and conviction in God, His divine form, and His eternal presence in one's life.",
        keywords=["faith", "trust", "belief", "conviction", "confidence", "shraddha", "certainty"],
        related_themes=["devotion", "surrender", "guru", "satsang"],
        key_passages=["Gadhada I-1", "Gadhada I-72", "Vadtal-18", "Kariyani-1"],
        guidance="Strengthen your faith through study of scriptures, association with devotees, and trusting in God's plan even during difficulties."
    ),
    
    "surrender": ThemeMapping(
        name="surrender",
        description="Complete offering of oneself to God's will, letting go of ego and personal desires for divine grace.",
        keywords=["surrender", "submission", "humility", "ego", "prapatti", "sharanagati", "offering"],
        related_themes=["faith", "devotion", "detachment", "dharma"],
        key_passages=["Vadtal-18", "Gadhada I-1", "Gadhada II-13", "Ahmedabad-3"],
        guidance="Practice surrender by offering all actions to God and accepting outcomes with equanimity. True surrender brings divine protection."
    ),
    
    "service": ThemeMapping(
        name="service",
        description="Selfless service (seva) to God and His devotees as an expression of devotion and spiritual practice.",
        keywords=["service", "seva", "selfless", "sacrifice", "serving", "helping", "dedication"],
        related_themes=["devotion", "dharma", "satsang", "surrender"],
        key_passages=["Ahmedabad-3", "Gadhada I-2", "Sarangpur-5", "Kariyani-2"],
        guidance="Engage in selfless service without expecting rewards. Serve God through serving others, especially devotees and the needy."
    ),
    
    "knowledge": ThemeMapping(
        name="knowledge",
        description="Spiritual wisdom (gnan) about God, the soul, and maya; understanding that leads to liberation.",
        keywords=["knowledge", "wisdom", "gnan", "understanding", "realization", "truth", "enlightenment"],
        related_themes=["guru", "meditation", "detachment", "dharma"],
        key_passages=["Loyej-7", "Gadhada I-72", "Kariyani-8", "Gadhada II-13"],
        guidance="Seek spiritual knowledge through study of scriptures, contemplation, and guidance from realized souls. Knowledge should lead to transformation."
    ),
    
    "detachment": ThemeMapping(
        name="detachment",
        description="Freedom from worldly attachments and desires; focusing on eternal spiritual values rather than temporary pleasures.",
        keywords=["detachment", "vairagya", "renunciation", "attachment", "desire", "worldly", "material"],
        related_themes=["knowledge", "surrender", "meditation", "dharma"],
        key_passages=["Gadhada I-2", "Gadhada I-72", "Kariyani-3", "Vadtal-18"],
        guidance="Gradually reduce attachment to material things while increasing attachment to God. Practice contentment and inner freedom."
    ),
    
    "dharma": ThemeMapping(
        name="dharma",
        description="Righteous conduct and moral duty; living according to divine principles in all circumstances.",
        keywords=["dharma", "righteousness", "duty", "moral", "virtue", "conduct", "ethics", "righteous"],
        related_themes=["service", "knowledge", "faith", "satsang"],
        key_passages=["Gadhada II-13", "Gadhada I-1", "Ahmedabad-3", "Kariyani-1"],
        guidance="Follow righteous conduct in all situations, guided by scriptural principles. Dharma protects those who uphold it."
    ),
    
    "meditation": ThemeMapping(
        name="meditation",
        description="Contemplation and focus on God's divine form; practices for spiritual concentration and inner connection.",
        keywords=["meditation", "dhyan", "contemplation", "focus", "concentration", "samadhi", "remembrance"],
        related_themes=["devotion", "knowledge", "detachment", "guru"],
        key_passages=["Sarangpur-5", "Kariyani-8", "Gadhada I-2", "Loyej-7"],
        guidance="Establish a regular meditation practice, focusing the mind on God's form and attributes. Constant remembrance leads to realization."
    ),
    
    "guru": ThemeMapping(
        name="guru",
        description="The spiritual master who guides disciples on the path to God; the importance of authentic spiritual guidance.",
        keywords=["guru", "master", "teacher", "sant", "guide", "sadhu", "spiritual teacher", "mentor"],
        related_themes=["faith", "knowledge", "satsang", "devotion"],
        key_passages=["Gadhada I-1", "Gadhada I-72", "Loyej-7", "Kariyani-2"],
        guidance="Seek guidance from authentic spiritual teachers and study revealed scriptures. The Guru illuminates the path to God."
    ),
    
    "satsang": ThemeMapping(
        name="satsang",
        description="Holy fellowship; the company of true devotees and saints that nurtures spiritual growth.",
        keywords=["satsang", "fellowship", "association", "company", "devotees", "community", "assembly"],
        related_themes=["guru", "devotion", "service", "faith"],
        key_passages=["Kariyani-2", "Gadhada I-1", "Sarangpur-5", "Ahmedabad-3"],
        guidance="Seek and cherish the association of genuine devotees. Satsang strengthens faith, provides inspiration, and accelerates spiritual progress."
    )
}


def get_theme(theme_name: str) -> ThemeMapping:
    """Get a theme mapping by name (case-insensitive)"""
    return SPIRITUAL_THEMES.get(theme_name.lower())


def get_all_themes() -> List[str]:
    """Get all available theme names"""
    return list(SPIRITUAL_THEMES.keys())


def get_theme_keywords(theme_name: str) -> List[str]:
    """Get keywords for a specific theme"""
    theme = get_theme(theme_name)
    return theme.keywords if theme else []


def detect_themes_in_text(text: str) -> List[str]:
    """Detect which themes are present in a text based on keywords"""
    text_lower = text.lower()
    detected = []
    
    for theme_name, theme in SPIRITUAL_THEMES.items():
        matches = sum(1 for keyword in theme.keywords if keyword in text_lower)
        if matches >= 2:  # Require at least 2 keyword matches
            detected.append(theme_name)
    
    return detected


def get_related_themes(theme_name: str) -> List[str]:
    """Get themes related to a given theme"""
    theme = get_theme(theme_name)
    return theme.related_themes if theme else []


def get_theme_guidance(theme_name: str) -> str:
    """Get practical guidance for a theme"""
    theme = get_theme(theme_name)
    return theme.guidance if theme else ""


def get_key_passages(theme_name: str) -> List[str]:
    """Get key Vachanamrut references for a theme"""
    theme = get_theme(theme_name)
    return theme.key_passages if theme else []


# Theme categories for organized display
THEME_CATEGORIES = {
    "Foundational": ["faith", "devotion", "surrender"],
    "Practice": ["service", "meditation", "dharma"],
    "Knowledge": ["knowledge", "detachment"],
    "Community": ["guru", "satsang"]
}


def get_themes_by_category() -> Dict[str, List[str]]:
    """Get themes organized by category"""
    return THEME_CATEGORIES

