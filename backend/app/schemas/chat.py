from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import List, Optional
from datetime import datetime


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User's question or message")
    theme: Optional[str] = Field(None, description="Optional theme to focus the search")
    context: Optional[List[str]] = Field(None, description="Previous conversation context")

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        json_schema_extra={
            "example": {
                "message": "How should I approach spiritual growth?",
                "theme": "devotion",
                "context": ["previous_conversation_id"],
            }
        },
    )


class Citation(BaseModel):
    reference: str = Field(..., description="Source reference for the citation")
    passage: str = Field(..., description="The actual text passage")
    page_number: Optional[int] = Field(None, description="Page number in source document")
    relevance_score: Optional[float] = Field(None, description="Relevance score from 0-1")

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI-generated response")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")
    related_themes: List[str] = Field(default_factory=list, description="Related spiritual themes")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    interaction_id: Optional[str] = Field(None, description="ID for feedback tracking")

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        json_schema_extra={
            "example": {
                "response": "Spiritual growth requires consistent practice and surrender to divine will...",
                "citations": [
                    {
                        "reference": "Gadhada I-1",
                        "passage": "One who has firm faith in God...",
                        "pageNumber": 15,
                        "relevanceScore": 0.95,
                    }
                ],
                "relatedThemes": ["faith", "surrender", "devotion"],
                "timestamp": "2023-01-01T12:00:00Z",
                "interactionId": "20231201120000_1"
            }
        },
    )