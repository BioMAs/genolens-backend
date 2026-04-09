"""
Integration tests for Comments API endpoints.

Uses dependency_overrides for auth/db and patches the comments_service singleton.

Covers:
- GET /projects/{id}/comments → 200
- POST /projects/{id}/comments → 201 / 422
- GET /comments/{id}/thread → 200 / 404
- GET /projects/{id}/comments/count → 200 with count
- PATCH /comments/{id} → 200
- DELETE /comments/{id} → 200
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
from datetime import datetime
from httpx import AsyncClient, ASGITransport

from tests.conftest import (
    TEST_USER_ID, TEST_PROJECT_ID, TEST_COMMENT_ID,
    make_comment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _fake_user_dict():
    return {"sub": str(TEST_USER_ID), "email": "test@example.com"}


@pytest_asyncio.fixture
async def comments_client():
    """Client with auth and DB overrides for comments endpoint tests."""
    from app.main import app
    from app.api.deps.auth import get_current_user
    from app.api.deps.db import get_db

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    async def _override_user():
        return _fake_user_dict()

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_db

    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects/{id}/comments
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCommentsEndpoint:

    @pytest.mark.asyncio
    async def test_returns_200_with_list(self, comments_client):
        """Should return 200 with a list of comments."""
        client, _ = comments_client
        c = make_comment()
        with patch(
            "app.api.endpoints.comments.comments_service.get_comments",
            new=AsyncMock(return_value=[c]),
        ):
            response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}/comments")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, comments_client):
        """Should return 200 with empty list."""
        client, _ = comments_client
        with patch(
            "app.api.endpoints.comments.comments_service.get_comments",
            new=AsyncMock(return_value=[]),
        ):
            response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}/comments")

        assert response.status_code == 200
        assert response.json() == []


# ─────────────────────────────────────────────────────────────────────────────
# POST /projects/{id}/comments
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateCommentEndpoint:

    @pytest.mark.asyncio
    async def test_create_returns_201(self, comments_client):
        """Valid comment creation should return 201."""
        client, mock_db = comments_client
        c = make_comment(content="New comment")

        # Configure db.execute to return a proper (non-coroutine) result for
        # the project fetch used in the email-notification section.
        mock_project = MagicMock()
        mock_project.name = "Test Project"
        mock_exec_result = MagicMock()
        mock_exec_result.scalar_one_or_none.return_value = mock_project
        mock_db.execute = AsyncMock(return_value=mock_exec_result)

        with patch(
            "app.api.endpoints.comments.comments_service.create_comment",
            new=AsyncMock(return_value=c),
        ), patch(
            "app.api.endpoints.comments.history_service.log_activity",
            new=AsyncMock(),
        ), patch(
            "app.api.endpoints.comments.email_service.extract_mentions",
            return_value=[],
        ):
            response = await client.post(f"/api/v1/projects/{TEST_PROJECT_ID}/comments",
                json={"content": "New comment"},
            )

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_missing_content_returns_422(self, comments_client):
        """Body without content should return 422."""
        client, _ = comments_client
        response = await client.post(f"/api/v1/projects/{TEST_PROJECT_ID}/comments",
            json={},
        )
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /comments/{id}/thread
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCommentThreadEndpoint:

    @pytest.mark.asyncio
    async def test_thread_returns_200(self, comments_client):
        """Should return 200 with thread when comment exists."""
        client, _ = comments_client
        c = make_comment()
        with patch(
            "app.api.endpoints.comments.comments_service.get_comment_thread",
            new=AsyncMock(return_value=[c]),
        ):
            response = await client.get(f"/api/v1/comments/{TEST_COMMENT_ID}/thread")

        assert response.status_code == 200
        data = response.json()
        assert "comment" in data
        assert "reply_count" in data

    @pytest.mark.asyncio
    async def test_missing_comment_returns_404(self, comments_client):
        """Should return 404 when comment is not found."""
        client, _ = comments_client
        with patch(
            "app.api.endpoints.comments.comments_service.get_comment_thread",
            new=AsyncMock(return_value=[]),
        ):
            response = await client.get(f"/api/v1/comments/{uuid4()}/thread")

        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects/{id}/comments/count
# ─────────────────────────────────────────────────────────────────────────────

class TestCommentCountEndpoint:

    @pytest.mark.asyncio
    async def test_returns_count(self, comments_client):
        """Should return dict with count field."""
        client, _ = comments_client
        with patch(
            "app.api.endpoints.comments.comments_service.get_comment_count",
            new=AsyncMock(return_value=7),
        ):
            response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}/comments/count"
            )

        assert response.status_code == 200
        assert response.json()["count"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# PATCH /comments/{id}
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateCommentEndpoint:

    @pytest.mark.asyncio
    async def test_update_returns_200(self, comments_client):
        """Valid update should return 200 with updated comment."""
        client, _ = comments_client
        c = make_comment(content="Updated content")
        with patch(
            "app.api.endpoints.comments.comments_service.update_comment",
            new=AsyncMock(return_value=c),
        ), patch(
            "app.api.endpoints.comments.history_service.log_activity",
            new=AsyncMock(),
        ):
            response = await client.patch(f"/api/v1/comments/{TEST_COMMENT_ID}",
                json={"content": "Updated content"},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_not_found_returns_404(self, comments_client):
        """If comment not found, should return 404."""
        client, _ = comments_client
        with patch(
            "app.api.endpoints.comments.comments_service.update_comment",
            new=AsyncMock(return_value=None),
        ):
            response = await client.patch(f"/api/v1/comments/{uuid4()}",
                json={"content": "ghost"},
            )

        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /comments/{id}
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteCommentEndpoint:

    @pytest.mark.asyncio
    async def test_delete_returns_200(self, comments_client):
        """Successful delete should return 200."""
        client, _ = comments_client
        with patch(
            "app.api.endpoints.comments.comments_service.delete_comment",
            new=AsyncMock(return_value=True),
        ), patch(
            "app.api.endpoints.comments.history_service.log_activity",
            new=AsyncMock(),
        ):
            response = await client.delete(f"/api/v1/comments/{TEST_COMMENT_ID}")

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_not_found_returns_404(self, comments_client):
        """Deleting a non-existent comment should return 404."""
        client, _ = comments_client
        with patch(
            "app.api.endpoints.comments.comments_service.delete_comment",
            new=AsyncMock(return_value=False),
        ):
            response = await client.delete(f"/api/v1/comments/{uuid4()}")

        assert response.status_code == 404
