from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import List, Optional


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class ThemeResponse(BaseModel):
    name: str = Field(..., description="Theme name")
    description: Optional[str] = Field(None, description="Theme description")
    keywords: List[str] = Field(default_factory=list, description="Related keywords")

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


class ThemeDetail(BaseModel):
    name: str = Field(..., description="Theme name")
    description: str = Field(..., description="Detailed theme description")
    keywords: List[str] = Field(default_factory=list, description="Related keywords")
    related_passages: List[str] = Field(default_factory=list, description="Key passage references")
    related_themes: List[str] = Field(default_factory=list, description="Connected themes")

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        json_schema_extra={
            "example": {
                "name": "devotion",
                "description": "The practice of loving dedication to God",
                "keywords": ["bhakti", "love", "surrender", "worship"],
                "relatedPassages": ["Gadhada I-1", "Sarangpur-5"],
                "relatedThemes": ["faith", "surrender", "service"],
            }
        },
    )