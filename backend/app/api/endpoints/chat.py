from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
import time

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService, get_chat_service
from app.core.analytics import get_analytics_service, AnalyticsService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> ChatResponse:
    """Send a message and receive AI response with citations"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing chat request: {request.message[:50]}...")
        response = await chat_service.process_message(request)
        
        # Track interaction for analytics
        response_time_ms = int((time.time() - start_time) * 1000)
        
        try:
            interaction_id = analytics_service.track_interaction(
                query=request.message,
                response=response.response,
                theme=request.theme,
                citations_count=len(response.citations),
                related_themes=response.related_themes,
                response_time_ms=response_time_ms,
                search_method="semantic" if chat_service._use_semantic else "text"
            )
            
            # Add interaction ID to response for feedback tracking
            # Store in a way frontend can access (e.g., in response metadata)
            response.interaction_id = interaction_id
        except Exception as e:
            logger.warning(f"Failed to track analytics: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing your message."
        )


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
