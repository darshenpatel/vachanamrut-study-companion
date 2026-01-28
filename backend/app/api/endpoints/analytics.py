"""
Analytics API Endpoints
Provides usage metrics and feedback collection
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

from app.core.analytics import get_analytics_service, AnalyticsService

router = APIRouter()
logger = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback"""
    interaction_id: str = Field(..., description="ID of the interaction being rated")
    score: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, max_length=500, description="Optional text feedback")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool
    message: str


@router.get("/metrics")
async def get_usage_metrics(
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> Dict[str, Any]:
    """Get current usage metrics"""
    try:
        metrics = analytics_service.get_metrics()
        return {
            "status": "success",
            "data": metrics
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/daily-summary")
async def get_daily_summary(
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> Dict[str, Any]:
    """Get summary for current day"""
    try:
        summary = analytics_service.get_daily_summary()
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        logger.error(f"Error getting daily summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve daily summary")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> FeedbackResponse:
    """Submit user feedback for an interaction"""
    try:
        success = analytics_service.record_feedback(
            interaction_id=request.interaction_id,
            score=request.score,
            feedback_text=request.feedback_text
        )
        
        if success:
            return FeedbackResponse(
                success=True,
                message="Thank you for your feedback!"
            )
        else:
            return FeedbackResponse(
                success=False,
                message="Interaction not found. Feedback not recorded."
            )
            
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")


@router.get("/export")
async def export_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
) -> Dict[str, Any]:
    """Export analytics data for a date range"""
    try:
        data = analytics_service.export_data(start_date, end_date)
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        logger.error(f"Error exporting analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export analytics")

