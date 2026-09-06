"""
Tests for the cross-project comparison catalog.

Covers both the pure builder (app.services.comparison_catalog) and the
GET /api/v1/comparisons endpoint that aggregates it across projects.
"""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.services.comparison_catalog import build_comparisons_from_datasets
from tests.conftest import TEST_PROJECT_ID, TEST_USER_ID, make_fake_supabase_user

PROJECT_A = TEST_PROJECT_ID
PROJECT_B = UUID("00000000-0000-0000-0000-0000000000b0")


def make_dataset(
    project_id: UUID = PROJECT_A,
    dataset_id: UUID | None = None,
    name: str = "deg_file.csv",
    type_: str = "DEG",
    status: str = "READY",
    metadata: dict | None = None,
    deg_up: int | None = None,
    deg_down: int | None = None,
    deg_total: int | None = None,
    updated_at: datetime | None = None,
):
    """Minimal duck-typed Dataset — the catalog only reads attributes."""
    d = MagicMock()
    d.id = dataset_id or uuid4()
    d.project_id = project_id
    d.name = name
    d.type = type_
    d.status = status
    d.dataset_metadata = metadata
    d.deg_up_count = deg_up
    d.deg_down_count = deg_down
    d.deg_significant_count = deg_total
    d.updated_at = updated_at or datetime(2026, 1, 1)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# build_comparisons_from_datasets
# ─────────────────────────────────────────────────────────────────────────────

class TestComparisonCatalog:

    def test_deg_dataset_uses_db_counts(self):
        d = make_dataset(
            metadata={"comparison_name": "treated_vs_control"},
            deg_up=10, deg_down=4, deg_total=14,
        )
        [summary] = build_comparisons_from_datasets([d])
        assert summary.name == "treated_vs_control"
        assert (summary.deg_up, summary.deg_down, summary.deg_total) == (10, 4, 14)
        assert summary.dataset_type == "SINGLE"

    def test_deg_dataset_falls_back_to_metadata_counts(self):
        d = make_dataset(
            metadata={"comparison_name": "a_vs_b", "deg_up": 3, "deg_down": 2, "deg_total": 5},
        )
        [summary] = build_comparisons_from_datasets([d])
        assert (summary.deg_up, summary.deg_down, summary.deg_total) == (3, 2, 5)

    def test_non_ready_datasets_are_skipped(self):
        d = make_dataset(status="PROCESSING", metadata={"comparison_name": "x_vs_y"})
        assert build_comparisons_from_datasets([d]) == []

    def test_global_dataset_yields_one_entry_per_comparison(self):
        d = make_dataset(
            type_="DEG",
            name="global.csv",
            metadata={"comparisons": {
                "a_vs_b": {"deg_up": 1, "deg_down": 1, "deg_total": 2},
                "c_vs_d": {"deg_up": 5, "deg_down": 0, "deg_total": 5},
            }},
        )
        names = {s.name: s for s in build_comparisons_from_datasets([d])}

        assert {"a_vs_b", "c_vs_d"} <= set(names)
        assert names["a_vs_b"].dataset_type == "GLOBAL"
        assert names["c_vs_d"].deg_total == 5
        # Long-standing quirk, kept as-is: a type=DEG dataset also goes through the
        # single-comparison branch, so a global file with no `comparison_name`
        # additionally surfaces an entry named after the file itself.
        assert names["global.csv"].dataset_type == "SINGLE"

    def test_global_comparison_does_not_duplicate_its_named_single_entry(self):
        """When both branches produce the same name they must collapse to one row."""
        d = make_dataset(
            type_="DEG",
            metadata={
                "comparison_name": "a_vs_b",
                "comparisons": {"a_vs_b": {"deg_up": 5, "deg_down": 5, "deg_total": 10}},
            },
            deg_up=1, deg_down=1, deg_total=2,
        )
        [summary] = build_comparisons_from_datasets([d])
        assert summary.dataset_type == "GLOBAL"
        assert summary.deg_total == 10

    def test_enrichment_flag_is_attributed_within_the_project(self):
        deg = make_dataset(metadata={"comparison_name": "a_vs_b"}, deg_up=1, deg_down=1, deg_total=2)
        enr = make_dataset(
            type_="ENRICHMENT",
            name="enrichment.csv",
            metadata={"enrichment_comparisons": ["a_vs_b"]},
        )
        [summary] = build_comparisons_from_datasets([deg, enr])
        assert summary.has_enrichment is True

    def test_enrichment_does_not_leak_across_projects(self):
        deg_a = make_dataset(project_id=PROJECT_A, metadata={"comparison_name": "shared"},
                             deg_up=1, deg_down=1, deg_total=2)
        deg_b = make_dataset(project_id=PROJECT_B, metadata={"comparison_name": "shared"},
                             deg_up=1, deg_down=1, deg_total=2)
        enr_a = make_dataset(project_id=PROJECT_A, type_="ENRICHMENT",
                             metadata={"enrichment_comparisons": ["shared"]})

        by_project = {
            s.dataset_id: s for s in build_comparisons_from_datasets([deg_a, deg_b, enr_a])
        }
        assert by_project[deg_a.id].has_enrichment is True
        assert by_project[deg_b.id].has_enrichment is False

    def test_same_comparison_name_in_two_datasets_is_not_collapsed(self):
        """Keying by name alone used to drop one of the two."""
        first = make_dataset(name="run1.csv", metadata={"comparison_name": "a_vs_b"},
                             deg_up=1, deg_down=1, deg_total=2)
        second = make_dataset(name="run2.csv", metadata={"comparison_name": "a_vs_b"},
                              deg_up=9, deg_down=9, deg_total=18)
        results = build_comparisons_from_datasets([first, second])
        assert len(results) == 2
        assert {r.dataset_id for r in results} == {first.id, second.id}


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/v1/comparisons
# ─────────────────────────────────────────────────────────────────────────────

def _client_for(rows):
    """Build an AsyncClient whose DB returns `rows` as (Dataset, project_name)."""
    from app.api.deps import get_current_user, get_db
    from app.main import app

    db = AsyncMock()
    result = MagicMock()
    result.all.return_value = rows
    db.execute = AsyncMock(return_value=result)

    async def _override_user():
        return make_fake_supabase_user()

    async def _override_db():
        yield db

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    return app, db


@pytest_asyncio.fixture
async def comparisons_client():
    rows = [
        (make_dataset(project_id=PROJECT_A, name="alpha.csv",
                      metadata={"comparison_name": "treated_vs_control"},
                      deg_up=10, deg_down=5, deg_total=15,
                      updated_at=datetime(2026, 3, 1)), "Project A"),
        (make_dataset(project_id=PROJECT_B, name="beta.csv",
                      metadata={"comparison_name": "ko_vs_wt"},
                      deg_up=2, deg_down=1, deg_total=3,
                      updated_at=datetime(2026, 1, 15)), "Project B"),
    ]
    app, db = _client_for(rows)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, db
    app.dependency_overrides.clear()


class TestListUserComparisons:

    @pytest.mark.asyncio
    async def test_returns_comparisons_from_every_project(self, comparisons_client):
        client, _ = comparisons_client
        response = await client.get("/api/v1/comparisons")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert {c["project_name"] for c in body["comparisons"]} == {"Project A", "Project B"}
        assert {c["name"] for c in body["comparisons"]} == {"treated_vs_control", "ko_vs_wt"}

    @pytest.mark.asyncio
    async def test_item_carries_its_origin(self, comparisons_client):
        client, _ = comparisons_client
        body = (await client.get("/api/v1/comparisons")).json()
        item = next(c for c in body["comparisons"] if c["name"] == "treated_vs_control")

        assert item["project_id"] == str(PROJECT_A)
        assert item["project_name"] == "Project A"
        assert item["dataset_name"] == "alpha.csv"
        assert item["updated_at"].startswith("2026-03-01")

    @pytest.mark.asyncio
    async def test_search_matches_comparison_name(self, comparisons_client):
        client, _ = comparisons_client
        body = (await client.get("/api/v1/comparisons", params={"search": "KO_VS"})).json()
        assert [c["name"] for c in body["comparisons"]] == ["ko_vs_wt"]
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_search_matches_project_name(self, comparisons_client):
        client, _ = comparisons_client
        body = (await client.get("/api/v1/comparisons", params={"search": "project b"})).json()
        assert [c["name"] for c in body["comparisons"]] == ["ko_vs_wt"]

    @pytest.mark.asyncio
    async def test_sort_by_deg_total_ascending(self, comparisons_client):
        client, _ = comparisons_client
        body = (await client.get(
            "/api/v1/comparisons",
            params={"sort_by": "deg_total", "sort_order": "asc"},
        )).json()
        assert [c["deg_total"] for c in body["comparisons"]] == [3, 15]

    @pytest.mark.asyncio
    async def test_sort_by_updated_at_desc_is_the_default(self, comparisons_client):
        client, _ = comparisons_client
        body = (await client.get("/api/v1/comparisons")).json()
        assert [c["name"] for c in body["comparisons"]] == ["treated_vs_control", "ko_vs_wt"]

    @pytest.mark.asyncio
    async def test_sort_by_name(self, comparisons_client):
        client, _ = comparisons_client
        body = (await client.get(
            "/api/v1/comparisons", params={"sort_by": "name", "sort_order": "asc"},
        )).json()
        assert [c["name"] for c in body["comparisons"]] == ["ko_vs_wt", "treated_vs_control"]

    @pytest.mark.asyncio
    async def test_pagination_reports_pages(self, comparisons_client):
        client, _ = comparisons_client
        body = (await client.get("/api/v1/comparisons", params={"page_size": 1})).json()
        assert body["total"] == 2
        assert body["total_pages"] == 2
        assert len(body["comparisons"]) == 1

    @pytest.mark.asyncio
    async def test_second_page_returns_the_remainder(self, comparisons_client):
        client, _ = comparisons_client
        body = (await client.get(
            "/api/v1/comparisons", params={"page_size": 1, "page": 2},
        )).json()
        assert [c["name"] for c in body["comparisons"]] == ["ko_vs_wt"]

    @pytest.mark.asyncio
    async def test_rejects_unknown_sort_field(self, comparisons_client):
        client, _ = comparisons_client
        response = await client.get("/api/v1/comparisons", params={"sort_by": "deg_up; DROP"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_homonymous_comparisons_in_two_projects_both_appear(self):
        rows = [
            (make_dataset(project_id=PROJECT_A, metadata={"comparison_name": "shared"},
                          deg_up=1, deg_down=1, deg_total=2), "Project A"),
            (make_dataset(project_id=PROJECT_B, metadata={"comparison_name": "shared"},
                          deg_up=7, deg_down=7, deg_total=14), "Project B"),
        ]
        app, _ = _client_for(rows)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            body = (await client.get("/api/v1/comparisons")).json()

        app.dependency_overrides.clear()
        assert body["total"] == 2
        assert {c["project_name"] for c in body["comparisons"]} == {"Project A", "Project B"}

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_an_empty_page(self):
        app, _ = _client_for([])
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            body = (await client.get("/api/v1/comparisons")).json()

        app.dependency_overrides.clear()
        assert body == {
            "comparisons": [], "total": 0, "page": 1, "page_size": 20, "total_pages": 1,
        }

    @pytest.mark.asyncio
    async def test_requires_authentication(self):
        """Without the auth override the real dependency runs and rejects the call."""
        from app.main import app

        app.dependency_overrides.clear()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/v1/comparisons")

        assert response.status_code in (401, 403)
