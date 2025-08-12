from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService, get_chat_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Send a message and receive AI response with citations"""
    try:
        logger.info(f"Processing chat request: {request.message[:50]}...")
        response = await chat_service.process_message(request)
        return response
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your message.")

@router.get("/themes", response_model=List[str])
async def get_available_themes(
    chat_service: ChatService = Depends(get_chat_service)
) -> List[str]:
    """Get list of available scriptural themes"""
    try:
        themes = await chat_service.get_themes()
        return themes
    except Exception as e:
        logger.error(f"Error getting themes: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve themes.")