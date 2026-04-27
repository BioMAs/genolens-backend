"""Integration tests for /datasets/{id}/ai/* endpoints."""
import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from httpx import AsyncClient, ASGITransport
from app.main import app


FAKE_DATASET_ID = str(uuid4())


@pytest.mark.asyncio
async def test_interpret_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/datasets/{FAKE_DATASET_ID}/ai/interpret",
            json={"chart_type": "volcano", "context": {}, "context_key": "test"}
        )
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_ask_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/v1/datasets/{FAKE_DATASET_ID}/ai/ask",
            json={"chart_type": "volcano", "context_key": "test", "question": "What is this?", "context": {}}
        )
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_conversations_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            f"/api/v1/datasets/{FAKE_DATASET_ID}/ai/conversations",
            params={"chart_type": "volcano", "context_key": "test"}
        )
    assert resp.status_code in (401, 403)
