from pydantic import BaseModel
from typing import Optional

class HealthResponse(BaseModel):
    status: str
    message: str
    version: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int