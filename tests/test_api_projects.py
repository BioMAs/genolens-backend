"""
Integration tests for /projects API endpoints.

Uses dependency_overrides to skip real DB and auth.

Covers:
- POST /projects/ → 201
- GET /projects/ → 200 with paginated list
- GET /projects/{id} → 200 / 404
- PATCH /projects/{id} → 200 / 404
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID
from datetime import datetime
from httpx import AsyncClient, ASGITransport

from tests.conftest import (
    TEST_USER_ID, TEST_PROJECT_ID,
    make_project, make_fake_supabase_user,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_db_with_project(project=None):
    """Return a configured mock DB for project queries."""
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()

    project = project or make_project()

    # scalar() used for COUNT → total
    db.scalar = AsyncMock(return_value=1)

    # execute() used for SELECT
    result = MagicMock()
    result.scalar_one_or_none.return_value = project
    result.scalars.return_value.all.return_value = [project]
    db.execute = AsyncMock(return_value=result)

    return db


@pytest_asyncio.fixture
async def projects_client():
    """
    AsyncClient with auth and DB dependencies overridden for project tests.
    """
    from app.main import app
    from app.api.deps import get_current_user, get_db
    from app.api.deps.license import require_active_license
    from app.api.deps.subscription import get_or_create_user
    from app.models.models import User, SubscriptionPlan, UserStatus, UserRole

    fake_user = make_fake_supabase_user()
    mock_db = _make_db_with_project()

    async def _override_user():
        return fake_user

    async def _override_db():
        yield mock_db

    async def _override_license():
        return None

    async def _override_db_user():
        u = User()
        u.id = TEST_USER_ID
        u.email = "test@example.com"
        u.role = UserRole.USER
        u.subscription_plan = SubscriptionPlan.TEAM  # max_projects = None (unlimited)
        u.status = UserStatus.ACTIVE
        return u

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[require_active_license] = _override_license
    app.dependency_overrides[get_or_create_user] = _override_db_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_db

    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────────────
# POST /projects/
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateProject:

    @pytest.mark.asyncio
    async def test_create_project_returns_201(self, projects_client):
        client, mock_db = projects_client

        # Simulate refresh populating the object
        created = make_project(name="New Project")
        created.updated_at = datetime.utcnow()
        mock_db.refresh.side_effect = lambda obj: setattr(obj, "id", TEST_PROJECT_ID) or \
            setattr(obj, "owner_id", TEST_USER_ID) or \
            setattr(obj, "created_at", datetime.utcnow()) or \
            setattr(obj, "updated_at", datetime.utcnow()) or \
            setattr(obj, "description", None)

        response = await client.post("/api/v1/projects/", json={"name": "New Project"})

        assert response.status_code == 201
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_create_project_without_name_returns_422(self, projects_client):
        client, _ = projects_client
        response = await client.post("/api/v1/projects/", json={})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects/
# ─────────────────────────────────────────────────────────────────────────────

class TestListProjects:

    @pytest.mark.asyncio
    async def test_list_projects_returns_200(self, projects_client):
        client, _ = projects_client
        response = await client.get("/api/v1/projects/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_projects_response_structure(self, projects_client):
        client, _ = projects_client
        response = await client.get("/api/v1/projects/")
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    @pytest.mark.asyncio
    async def test_search_reaches_the_query(self, projects_client):
        """`search` used to be accepted by the client and dropped by the server."""
        client, mock_db = projects_client
        response = await client.get("/api/v1/projects/", params={"search": "tumor"})

        assert response.status_code == 200
        # ilike renders as lower(col) LIKE lower(:param) on the default dialect.
        rendered = " ".join(str(call.args[0]) for call in mock_db.execute.await_args_list)
        assert "lower(projects.name) LIKE" in rendered
        assert "lower(projects.description) LIKE" in rendered

    @pytest.mark.asyncio
    async def test_sort_order_reaches_the_query(self, projects_client):
        client, mock_db = projects_client
        response = await client.get(
            "/api/v1/projects/", params={"sort_by": "name", "sort_order": "asc"}
        )

        assert response.status_code == 200
        rendered = " ".join(str(call.args[0]) for call in mock_db.execute.await_args_list)
        assert "ORDER BY projects.name ASC" in rendered

    @pytest.mark.asyncio
    async def test_rejects_unknown_sort_field(self, projects_client):
        client, _ = projects_client
        response = await client.get("/api/v1/projects/", params={"sort_by": "owner_id"})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /projects/{project_id}
# ─────────────────────────────────────────────────────────────────────────────

class TestGetProject:

    @pytest.mark.asyncio
    async def test_get_existing_project_returns_200(self, projects_client):
        client, _ = projects_client
        response = await client.get(f"/api/v1/projects/{TEST_PROJECT_ID}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_nonexistent_project_returns_404(self, projects_client):
        from app.main import app
        from app.api.deps import get_current_user, get_db

        fake_user = make_fake_supabase_user()
        db_no_result = AsyncMock()
        db_no_result.add = MagicMock()
        db_no_result.commit = AsyncMock()
        db_no_result.refresh = AsyncMock()

        result_404 = MagicMock()
        result_404.scalar_one_or_none.return_value = None
        db_no_result.execute = AsyncMock(return_value=result_404)

        async def _override_user():
            return fake_user

        async def _override_db():
            yield db_no_result

        app.dependency_overrides[get_current_user] = _override_user
        app.dependency_overrides[get_db] = _override_db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as c:
            from uuid import uuid4
            response = await c.get(f"/projects/{uuid4()}")

        app.dependency_overrides.clear()

        assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# PATCH /projects/{project_id}
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateProject:

    @pytest.mark.asyncio
    async def test_patch_project_returns_200(self, projects_client):
        client, mock_db = projects_client

        project = make_project()
        project.updated_at = datetime.utcnow()
        result = MagicMock()
        result.scalar_one_or_none.return_value = project
        mock_db.execute = AsyncMock(return_value=result)

        response = await client.patch(f"/api/v1/projects/{TEST_PROJECT_ID}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200
