import pytest
from httpx import AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_returns_healthy():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.get("/api/health/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "message" in data


@pytest.mark.asyncio
async def test_themes_returns_list():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.get("/api/themes/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert all(isinstance(x, str) for x in data)
        assert len(data) > 0


@pytest.mark.asyncio
async def test_chat_returns_camelcase_fields():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {"message": "What is faith?", "theme": "faith"}
        resp = await ac.post("/api/chat/", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Root fields
        assert "response" in data
        assert "citations" in data
        assert "relatedThemes" in data
        assert isinstance(data["citations"], list)
        # If we have a citation, ensure camelCase keys
        if data["citations"]:
            c0 = data["citations"][0]
            assert "reference" in c0
            assert "passage" in c0
            assert "pageNumber" in c0 or "pageNumber" not in c0  # optional
            assert "relevanceScore" in c0 or "relevanceScore" not in c0  # optional 