"""
Integration tests for GET /projects/{id}/dashboard-stats endpoint.

The endpoint makes ~12 sequential db.execute() calls to aggregate metrics.
We mock each call in order using side_effect.

Covers:
- Owner gets 200 with full stats structure
- Member gets 200 (same data)
- Stranger gets 404
- Project with no data → all zeros
- Response JSON keys/types
- 7-day activity aggregation
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport

from tests.conftest import TEST_USER_ID, TEST_PROJECT_ID, make_project, make_fake_supabase_user


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_result(value):
    """Mock result where .scalar_one() returns value."""
    r = MagicMock()
    r.scalar_one.return_value = value
    r.scalar_one_or_none.return_value = value
    return r


def _all_result(rows):
    """Mock result where .all() returns rows."""
    r = MagicMock()
    r.all.return_value = rows
    return r


def _project_result(project):
    """Mock result for a project fetch (scalar_one_or_none)."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = project
    return r


def _make_execute_sequence(project, *, is_owner=True, ds_rows=None, activity_7d_rows=None):
    """
    Build the ordered list of mock execute() results matching the endpoint logic.

    Owner path (12 calls):
      0  owner_query            → project
      1  ds_counts              → dataset type/status rows
      2  deg_count_result       → total DEG genes
      3  enrichment_count_result→ total enrichment pathways
      4  comp_result            → comparison metadata rows
      5  bookmarks_result       → total bookmarks
      6  gene_lists_result      → total gene lists
      7  comments_result        → total comments
      8  members_result         → total members
      9  activity_total_result  → total activity events
      10 last_activity_result   → last activity timestamp
      11 activity_7d_result     → activity last 7 days

    Member path (14 calls — 2 extra at the start):
      0  owner_query            → None (not owner)
      1  member_query           → member record
      2  proj_query             → project  (then continues from ds_counts)
    """
    if ds_rows is None:
        ds_rows = []  # no datasets
    if activity_7d_rows is None:
        activity_7d_rows = []

    main_sequence = [
        _all_result(ds_rows),                   # ds_counts
        _scalar_result(42),                     # total deg genes
        _scalar_result(10),                     # total enrichment pathways
        _all_result([]),                         # comp_result (comparison metadata)
        _scalar_result(7),                      # total bookmarks
        _scalar_result(3),                      # total gene lists
        _scalar_result(15),                     # total comments
        _scalar_result(2),                      # total members
        _scalar_result(100),                    # total activity events
        _scalar_result(datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)),  # last_activity
        _all_result(activity_7d_rows),           # activity 7d
    ]

    if is_owner:
        return [_project_result(project)] + main_sequence
    else:
        # Not found as owner, check member
        fake_member = MagicMock()
        return [
            _project_result(None),    # owner_query → None
            _project_result(fake_member),  # member_query → member found
            _project_result(project),      # proj_query → project
        ] + main_sequence


def _make_db(side_effects):
    """Create a mock DB whose execute returns results in sequence."""
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock(side_effect=side_effects)
    return db


@pytest_asyncio.fixture
async def stats_client_owner():
    """Client authenticated as project owner."""
    from app.main import app
    from app.api.deps import get_current_user, get_db

    project = make_project()
    fake_user = make_fake_supabase_user()
    mock_db = _make_db(_make_execute_sequence(project, is_owner=True))

    async def _user():
        return fake_user

    async def _db():
        yield mock_db

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_db] = _db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_db

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def stats_client_member():
    """Client authenticated as project member (not owner)."""
    from app.main import app
    from app.api.deps import get_current_user, get_db

    project = make_project()

    # Use a different user_id so owner check fails → falls through to member check
    other_user = make_fake_supabase_user()
    other_user.user_id = UUID("00000000-0000-0000-0000-000000000099")

    mock_db = _make_db(_make_execute_sequence(project, is_owner=False))

    async def _user():
        return other_user

    async def _db():
        yield mock_db

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_db] = _db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_db

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def stats_client_stranger():
    """Client authenticated as a user with no access to this project."""
    from app.main import app
    from app.api.deps import get_current_user, get_db

    # Both owner check and member check return None
    no_access_sequence = [
        _project_result(None),    # owner_query → None
        _project_result(None),    # member_query → None
    ]
    mock_db = _make_db(no_access_sequence)

    stranger = make_fake_supabase_user()
    stranger.user_id = UUID("00000000-0000-0000-0000-000000000088")

    async def _user():
        return stranger

    async def _db():
        yield mock_db

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_db] = _db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()


URL = f"/api/v1/projects/{TEST_PROJECT_ID}/dashboard-stats"


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardStatsOwner:

    @pytest.mark.asyncio
    async def test_returns_200(self, stats_client_owner):
        client, _ = stats_client_owner
        response = await client.get(URL)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_required_keys(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        required = {
            "project_id", "project_name",
            "total_datasets", "datasets_ready", "datasets_processing", "datasets_failed",
            "dataset_breakdown",
            "total_comparisons", "total_deg_genes", "total_enrichment_pathways",
            "total_bookmarks", "total_gene_lists", "total_comments", "total_members",
            "total_activity_events", "activity_last_7_days", "last_activity_at",
        }
        assert required.issubset(data.keys())

    @pytest.mark.asyncio
    async def test_dataset_breakdown_keys(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        breakdown = data["dataset_breakdown"]
        assert set(breakdown.keys()) == {"matrix", "deg", "enrichment", "metadata", "other"}

    @pytest.mark.asyncio
    async def test_activity_7d_keys(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        a = data["activity_last_7_days"]
        assert set(a.keys()) == {
            "datasets_uploaded", "bookmarks_created", "comments_added", "analyses_run"
        }

    @pytest.mark.asyncio
    async def test_deg_count_returned(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        assert data["total_deg_genes"] == 42

    @pytest.mark.asyncio
    async def test_enrichment_count_returned(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        assert data["total_enrichment_pathways"] == 10

    @pytest.mark.asyncio
    async def test_bookmarks_count_returned(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        assert data["total_bookmarks"] == 7

    @pytest.mark.asyncio
    async def test_comments_count_returned(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        assert data["total_comments"] == 15

    @pytest.mark.asyncio
    async def test_members_count_returned(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        assert data["total_members"] == 2

    @pytest.mark.asyncio
    async def test_activity_total_returned(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        assert data["total_activity_events"] == 100

    @pytest.mark.asyncio
    async def test_last_activity_not_null(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        assert data["last_activity_at"] is not None

    @pytest.mark.asyncio
    async def test_numeric_values_are_non_negative(self, stats_client_owner):
        client, _ = stats_client_owner
        data = (await client.get(URL)).json()
        numeric_keys = [
            "total_datasets", "datasets_ready", "datasets_processing", "datasets_failed",
            "total_comparisons", "total_deg_genes", "total_enrichment_pathways",
            "total_bookmarks", "total_gene_lists", "total_comments", "total_members",
            "total_activity_events",
        ]
        for key in numeric_keys:
            assert data[key] >= 0, f"{key} should be non-negative"


class TestDashboardStatsMember:

    @pytest.mark.asyncio
    async def test_member_gets_200(self, stats_client_member):
        client, _ = stats_client_member
        response = await client.get(URL)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_member_response_has_stats(self, stats_client_member):
        client, _ = stats_client_member
        data = (await client.get(URL)).json()
        assert "total_datasets" in data
        assert "total_deg_genes" in data


class TestDashboardStatsStranger:

    @pytest.mark.asyncio
    async def test_stranger_gets_404(self, stats_client_stranger):
        client = stats_client_stranger
        response = await client.get(URL)
        assert response.status_code == 404


class TestDashboardStatsEmptyProject:
    """Project with no datasets, no activity."""

    @pytest_asyncio.fixture
    async def empty_client(self):
        from app.main import app
        from app.api.deps import get_current_user, get_db

        project = make_project()
        fake_user = make_fake_supabase_user()

        # All counts return 0, no rows
        seq = [
            _project_result(project),  # owner check
            _all_result([]),            # ds_counts (no datasets)
            _scalar_result(0),          # deg count
            _scalar_result(0),          # enrichment count
            _all_result([]),            # comp metadata
            _scalar_result(0),          # bookmarks
            _scalar_result(0),          # gene lists
            _scalar_result(0),          # comments
            _scalar_result(0),          # members
            _scalar_result(0),          # activity total
            _scalar_result(None),       # last_activity (None for empty project)
            _all_result([]),            # activity 7d
        ]
        mock_db = _make_db(seq)

        async def _user():
            return fake_user

        async def _db():
            yield mock_db

        app.dependency_overrides[get_current_user] = _user
        app.dependency_overrides[get_db] = _db

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as c:
            yield c

        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_200_for_empty_project(self, empty_client):
        response = await empty_client.get(URL)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_all_counts_are_zero(self, empty_client):
        data = (await empty_client.get(URL)).json()
        for key in ["total_datasets", "total_deg_genes", "total_enrichment_pathways",
                    "total_bookmarks", "total_gene_lists", "total_comments", "total_members"]:
            assert data[key] == 0, f"{key} should be 0 for empty project"

    @pytest.mark.asyncio
    async def test_last_activity_is_none(self, empty_client):
        data = (await empty_client.get(URL)).json()
        assert data["last_activity_at"] is None

    @pytest.mark.asyncio
    async def test_activity_7d_all_zeros(self, empty_client):
        data = (await empty_client.get(URL)).json()
        a = data["activity_last_7_days"]
        assert all(v == 0 for v in a.values())
