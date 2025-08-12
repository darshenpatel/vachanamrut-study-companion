from fastapi import APIRouter
from app.schemas.response import HealthResponse

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Vachanamrut Study Companion API is running",
        version="1.0.0"
    )