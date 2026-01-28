from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
import logging

from app.schemas.theme import ThemeResponse, ThemeDetail
from app.services.theme_service import ThemeService, get_theme_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=List[str])
async def list_themes(
    theme_service: ThemeService = Depends(get_theme_service)
) -> List[str]:
    """Get all available spiritual themes"""
    try:
        themes = await theme_service.list_all_themes()
        return themes
    except Exception as e:
        logger.error(f"Error listing themes: {e}")
        raise HTTPException(status_code=500, detail="Failed to list themes.")


@router.get("/details", response_model=List[Dict])
async def get_all_theme_details(
    theme_service: ThemeService = Depends(get_theme_service)
) -> List[Dict]:
    """Get all themes with their details for display"""
    try:
        themes = await theme_service.get_themes_with_details()
        return themes
    except Exception as e:
        logger.error(f"Error getting theme details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve themes.")


@router.get("/categories", response_model=Dict[str, List[str]])
async def get_themes_by_category(
    theme_service: ThemeService = Depends(get_theme_service)
) -> Dict[str, List[str]]:
    """Get themes organized by category"""
    try:
        categories = await theme_service.get_themes_by_category()
        return categories
    except Exception as e:
        logger.error(f"Error getting theme categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve categories.")


@router.get("/search/{query}")
async def search_themes(
    query: str,
    theme_service: ThemeService = Depends(get_theme_service)
) -> List[str]:
    """Search for themes matching a query"""
    try:
        matches = await theme_service.search_themes(query)
        return matches
    except Exception as e:
        logger.error(f"Error searching themes: {e}")
        raise HTTPException(status_code=500, detail="Failed to search themes.")


@router.get("/{theme_name}", response_model=ThemeDetail)
async def get_theme_details(
    theme_name: str,
    theme_service: ThemeService = Depends(get_theme_service)
) -> ThemeDetail:
    """Get detailed information about a specific theme"""
    try:
        theme_detail = await theme_service.get_theme_detail(theme_name)
        if not theme_detail:
            raise HTTPException(status_code=404, detail=f"Theme '{theme_name}' not found")
        return theme_detail
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting theme details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve theme details.")


@router.get("/{theme_name}/guidance")
async def get_theme_guidance(
    theme_name: str,
    theme_service: ThemeService = Depends(get_theme_service)
) -> Dict[str, str]:
    """Get practical spiritual guidance for a theme"""
    try:
        guidance = await theme_service.get_theme_guidance(theme_name)
        if not guidance:
            raise HTTPException(status_code=404, detail=f"Theme '{theme_name}' not found")
        return {"theme": theme_name, "guidance": guidance}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting theme guidance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve guidance.")
