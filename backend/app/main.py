from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging
from dotenv import load_dotenv

from app.api.router import api_router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Vachanamrut Study Companion API",
    description="AI-powered chat interface for Vachanamrut scriptures providing spiritual guidance through intelligent conversation",
    version="1.0.0",
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
)

# CORS middleware - production ready
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",  # New Vite dev server port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "https://localhost:3000",
    "https://localhost:3001",
    "http://frontend:3000",  # Docker service name
]

# Add production origins from environment
if prod_origins := os.getenv("ALLOWED_ORIGINS"):
    allowed_origins.extend(prod_origins.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted host middleware for production security
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["localhost", "127.0.0.1", "*.vercel.app", "*.railway.app"]
    )

# Include API routes
app.include_router(api_router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Vachanamrut Study Companion API", 
        "version": "1.0.0",
        "status": "active",
        "description": "AI-powered spiritual guidance system",
        "docs": "/api/docs" if settings.DEBUG else "Contact admin for API documentation"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Simple health check without async dependencies
        status = {
            "status": "healthy",
            "message": "API is running",
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Global exception handler
@app.exception_handler(500)
async def internal_server_error(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Vachanamrut Study Companion API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Skip RAG initialization for now to get basic connectivity working
    logger.info("Startup complete - skipping RAG initialization for debugging")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Vachanamrut Study Companion API")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )