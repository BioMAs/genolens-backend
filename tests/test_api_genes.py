"""
Integration tests for /genes/search API endpoint.

Covers:
- GET /genes/search?q=TP53 → 200 with results structure
- GET /genes/search?q=X → 200 with empty results when no match
- GET /genes/search (no q param) → 422 validation error
- GET /genes/search?q=A (too short per min_length? actually min_length=1) → 200
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID
from datetime import datetime
from httpx import AsyncClient, ASGITransport

from tests.conftest import TEST_USER_ID, make_fake_supabase_user


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def genes_client():
    """
    Client with auth (SupabaseUser) and DB overridden for genes searches.
    The genes endpoint queries DB directly, so we mock db.execute.
    """
    from app.main import app
    from app.api.deps import get_current_user, get_db

    fake_user = make_fake_supabase_user()

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    # Default: empty result set
    empty_result = MagicMock()
    empty_result.all.return_value = []
    mock_db.execute = AsyncMock(return_value=empty_result)

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
# GET /genes/search
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneSearch:

    @pytest.mark.asyncio
    async def test_search_returns_200(self, genes_client):
        """A valid query should return 200."""
        client, _ = genes_client
        response = await client.get("/api/v1/genes/search", params={"q": "TP53"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_expected_keys(self, genes_client):
        """Response should contain results, total, and query."""
        client, _ = genes_client
        response = await client.get("/api/v1/genes/search", params={"q": "TP53"})
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_query_echoed_in_response(self, genes_client):
        """The query field should match the q parameter."""
        client, _ = genes_client
        response = await client.get("/api/v1/genes/search", params={"q": "BRCA1"})
        assert response.json()["query"] == "BRCA1"

    @pytest.mark.asyncio
    async def test_empty_results_when_no_match(self, genes_client):
        """When DB returns nothing, results should be empty list."""
        client, _ = genes_client
        response = await client.get("/api/v1/genes/search", params={"q": "NONEXISTENT_GENE_XYZ"})
        data = response.json()
        assert data["results"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_missing_q_param_returns_422(self, genes_client):
        """Without the required q parameter, should return 422."""
        client, _ = genes_client
        response = await client.get("/api/v1/genes/search")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_results_is_a_list(self, genes_client):
        """results field should always be a list."""
        client, _ = genes_client
        response = await client.get("/api/v1/genes/search", params={"q": "MYC"})
        assert isinstance(response.json()["results"], list)

    @pytest.mark.asyncio
    async def test_limit_parameter_accepted(self, genes_client):
        """limit parameter should be accepted without error."""
        client, _ = genes_client
        response = await client.get("/api/v1/genes/search", params={"q": "TP53", "limit": 5})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_project_id_filter_accepted(self, genes_client):
        """project_id filter should be accepted without error."""
        from uuid import uuid4
        client, _ = genes_client
        response = await client.get("/api/v1/genes/search",
            params={"q": "TP53", "project_id": str(uuid4())},
        )
        assert response.status_code == 200
