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
    """Get all available themes"""
    try:
        themes = await theme_service.list_all_themes()
        return themes
    except Exception as e:
        logger.error(f"Error listing themes: {e}")
        raise HTTPException(status_code=500, detail="Failed to list themes.")

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