from typing import List, Optional, Dict
import logging

from app.schemas.theme import ThemeDetail
from app.core.theme_mappings import (
    SPIRITUAL_THEMES,
    get_theme,
    get_all_themes,
    detect_themes_in_text,
    get_related_themes,
    get_themes_by_category,
    ThemeMapping
)
from app.core.semantic_retrieval import get_semantic_retriever

logger = logging.getLogger(__name__)


class ThemeService:
    """
    Service for managing spiritual themes with curated mappings.
    Provides theme discovery, details, and passage associations.
    """
    
    def __init__(self):
        self.semantic_retriever = None
        self._initialized = False
    
    async def list_all_themes(self) -> List[str]:
        """Get list of all 10 curated spiritual themes"""
        return get_all_themes()
    
    async def get_theme_detail(self, theme_name: str) -> Optional[ThemeDetail]:
        """Get detailed information about a spiritual theme"""
        try:
            theme = get_theme(theme_name)
            if not theme:
                logger.warning(f"Theme not found: {theme_name}")
                return None
            
            # Get actual passages from search if retriever is available
            related_passages = theme.key_passages.copy()
            
            try:
                if not self._initialized:
                    await self._ensure_initialized()
                
                if self.semantic_retriever:
                    # Search for passages related to this theme
                    search_results = await self.semantic_retriever.search(
                        query=f"{theme_name} {' '.join(theme.keywords[:3])}",
                        top_k=5,
                        theme_filter=None
                    )
                    
                    # Add found references
                    for result in search_results:
                        ref = result.get('reference', '')
                        if ref and ref not in related_passages:
                            related_passages.append(ref)
            except Exception as e:
                logger.warning(f"Could not search for theme passages: {e}")
            
            return ThemeDetail(
                name=theme.name,
                description=theme.description,
                keywords=theme.keywords,
                related_passages=related_passages[:8],  # Limit to 8
                related_themes=theme.related_themes
            )
            
        except Exception as e:
            logger.error(f"Error getting theme detail for {theme_name}: {e}")
            return None
    
    async def get_themes_with_details(self) -> List[Dict]:
        """Get all themes with their details for display"""
        themes_data = []
        
        for theme_name, theme in SPIRITUAL_THEMES.items():
            themes_data.append({
                "name": theme.name,
                "description": theme.description,
                "keywords": theme.keywords[:5],
                "relatedThemes": theme.related_themes,
                "guidance": theme.guidance
            })
        
        return themes_data
    
    async def get_related_themes(self, message: str) -> List[str]:
        """Get themes related to a user message"""
        try:
            # First detect themes from keywords in message
            detected = detect_themes_in_text(message)
            
            if detected:
                # Get related themes for the detected themes
                related = set(detected)
                for theme_name in detected[:2]:  # Check first 2 detected
                    related.update(get_related_themes(theme_name))
                return list(related)[:5]
            
            # If no keywords match, try semantic search
            if not self._initialized:
                await self._ensure_initialized()
            
            if self.semantic_retriever:
                search_results = await self.semantic_retriever.search(
                    query=message,
                    top_k=5
                )
                
                # Extract themes from search results
                themes = set()
                for result in search_results:
                    result_themes = result.get('themes', [])
                    if result_themes:
                        themes.update(result_themes[:2])
                
                # Map to curated themes
                curated = []
                for theme in themes:
                    theme_lower = theme.lower()
                    if theme_lower in SPIRITUAL_THEMES:
                        curated.append(theme_lower)
                
                return curated[:5]
            
            # Default themes if nothing found
            return ["devotion", "faith", "knowledge"]
            
        except Exception as e:
            logger.error(f"Error getting related themes: {e}")
            return ["devotion", "faith"]
    
    async def get_theme_guidance(self, theme_name: str) -> str:
        """Get practical spiritual guidance for a theme"""
        theme = get_theme(theme_name)
        if theme:
            return theme.guidance
        return ""
    
    async def get_themes_by_category(self) -> Dict[str, List[str]]:
        """Get themes organized by category"""
        return get_themes_by_category()
    
    async def search_themes(self, query: str) -> List[str]:
        """Search for themes matching a query"""
        query_lower = query.lower()
        matches = []
        
        for theme_name, theme in SPIRITUAL_THEMES.items():
            # Check name match
            if query_lower in theme_name:
                matches.append(theme_name)
                continue
            
            # Check keyword match
            for keyword in theme.keywords:
                if query_lower in keyword or keyword in query_lower:
                    matches.append(theme_name)
                    break
        
        return matches[:5]
    
    async def _ensure_initialized(self):
        """Initialize the semantic retriever"""
        if not self._initialized:
            try:
                self.semantic_retriever = get_semantic_retriever()
                await self.semantic_retriever.initialize()
                self._initialized = True
            except Exception as e:
                logger.warning(f"Could not initialize semantic retriever: {e}")
                self._initialized = True  # Mark as initialized to avoid retrying


# Dependency injection
def get_theme_service() -> ThemeService:
    return ThemeService()
