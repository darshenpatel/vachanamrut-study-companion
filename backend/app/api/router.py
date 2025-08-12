from fastapi import APIRouter

from app.api.endpoints import chat, themes, health

api_router = APIRouter()

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["chat"]
)

api_router.include_router(
    themes.router,
    prefix="/themes",
    tags=["themes"]
)