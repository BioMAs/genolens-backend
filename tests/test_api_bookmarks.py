"""
Integration tests for Bookmarks API endpoints.

Uses dependency_overrides for auth/db and patches the bookmarks_service singleton.

Covers:
- GET /projects/{id}/bookmarks → 200 list
- POST /projects/{id}/bookmarks → 201 / 422 (missing gene_symbol)
- GET /projects/{id}/bookmarks/check/{symbol} → 200 with is_bookmarked bool
- GET /gene-lists/ → 200 list
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID
from datetime import datetime
from httpx import AsyncClient, ASGITransport

from tests.conftest import (
    TEST_USER_ID, TEST_PROJECT_ID, TEST_BOOKMARK_ID, TEST_GENE_LIST_ID,
    make_bookmark, make_gene_list, make_fake_supabase_user,
)

# The exact auth function used in bookmarks router
_AUTH_PATH = "app.api.deps.auth.get_current_user"
_DB_PATH = "app.api.deps.db.get_db"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _fake_user_dict():
    """Simulate current_user dict as returned by auth.get_current_user."""
    return {"sub": str(TEST_USER_ID), "email": "test@example.com"}


@pytest_asyncio.fixture
async def bookmarks_client():
    """
    Client with auth overridden via dependency_overrides.
    DB is mocked via a simple AsyncMock injected as dependency.
    """
    from app.main import app
    from app.api.deps import get_current_user, get_db

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    fake_user = make_fake_supabase_user()

    async def _override_user():
        return fake_user

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_db

    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects/{id}/bookmarks
# ─────────────────────────────────────────────────────────────────────────────

class TestGetBookmarksEndpoint:

    @pytest.mark.asyncio
    async def test_returns_200_with_list(self, bookmarks_client):
        """Should return 200 with a list of bookmarks."""
        client, _ = bookmarks_client
        bm = make_bookmark()
        with patch(
            "app.api.endpoints.bookmarks.bookmarks_service.get_bookmarks",
            new=AsyncMock(return_value=[bm]),
        ):
            response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}/bookmarks")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_bookmarks(self, bookmarks_client):
        """Should return 200 with empty list when no bookmarks exist."""
        client, _ = bookmarks_client
        with patch(
            "app.api.endpoints.bookmarks.bookmarks_service.get_bookmarks",
            new=AsyncMock(return_value=[]),
        ):
            response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}/bookmarks")

        assert response.status_code == 200
        assert response.json() == []


# ─────────────────────────────────────────────────────────────────────────────
# POST /projects/{id}/bookmarks
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateBookmarkEndpoint:

    @pytest.mark.asyncio
    async def test_create_returns_201(self, bookmarks_client):
        """Valid bookmark creation should return 201."""
        client, _ = bookmarks_client
        bm = make_bookmark(gene_symbol="BRCA1")

        with patch(
            "app.api.endpoints.bookmarks.bookmarks_service.create_bookmark",
            new=AsyncMock(return_value=bm),
        ), patch(
            "app.api.endpoints.bookmarks.history_service.log_activity",
            new=AsyncMock(),
        ):
            response = await client.post(f"/api/v1/projects/{TEST_PROJECT_ID}/bookmarks",
                json={"gene_symbol": "BRCA1"},
            )

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_missing_gene_symbol_returns_422(self, bookmarks_client):
        """Body without gene_symbol should fail schema validation → 422."""
        client, _ = bookmarks_client
        response = await client.post(f"/api/v1/projects/{TEST_PROJECT_ID}/bookmarks",
            json={"notes": "no gene"},
        )
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects/{id}/bookmarks/check/{symbol}
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckBookmarkEndpoint:

    @pytest.mark.asyncio
    async def test_returns_is_bookmarked_true(self, bookmarks_client):
        """Should return {"is_bookmarked": true} when gene is bookmarked."""
        client, _ = bookmarks_client
        with patch(
            "app.api.endpoints.bookmarks.bookmarks_service.is_bookmarked",
            new=AsyncMock(return_value=True),
        ):
            response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}/bookmarks/check/TP53"
            )

        assert response.status_code == 200
        assert response.json()["is_bookmarked"] is True

    @pytest.mark.asyncio
    async def test_returns_is_bookmarked_false(self, bookmarks_client):
        """Should return {"is_bookmarked": false} when gene is not bookmarked."""
        client, _ = bookmarks_client
        with patch(
            "app.api.endpoints.bookmarks.bookmarks_service.is_bookmarked",
            new=AsyncMock(return_value=False),
        ):
            response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}/bookmarks/check/EGFR"
            )

        assert response.status_code == 200
        assert response.json()["is_bookmarked"] is False


# ─────────────────────────────────────────────────────────────────────────────
# GET /gene-lists/
# ─────────────────────────────────────────────────────────────────────────────

class TestGetGeneListsEndpoint:

    @pytest.mark.asyncio
    async def test_returns_200_with_lists(self, bookmarks_client):
        """Should return 200 with list of gene lists."""
        client, _ = bookmarks_client
        gl = make_gene_list()
        with patch(
            "app.api.endpoints.bookmarks.bookmarks_service.get_gene_lists",
            new=AsyncMock(return_value=[gl]),
        ):
            response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}/gene-lists")

        assert response.status_code == 200
        assert isinstance(response.json(), list)
