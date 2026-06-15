"""
Integration tests for POST /datasets/{dataset_id}/venn-analysis.

Focus on the cross-dataset (``comparison_refs``) path added so that the
Multi-Comparison feature works when each comparison lives in its own DEG
dataset, not only inside a single "global" multi-comparison dataset.

Covers:
- Cross-dataset Venn returns intersections across two datasets in the same project
- Referenced dataset in a different project is rejected (400)
- Fewer than 2 comparisons is rejected (400)
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from httpx import AsyncClient, ASGITransport

from tests.conftest import TEST_USER_ID, TEST_PROJECT_ID, make_fake_supabase_user, make_project


PATH_DATASET_ID = UUID("00000000-0000-0000-0000-0000000000a1")
OTHER_DATASET_ID = UUID("00000000-0000-0000-0000-0000000000a2")
FOREIGN_DATASET_ID = UUID("00000000-0000-0000-0000-0000000000a3")
OTHER_PROJECT_ID = UUID("00000000-0000-0000-0000-0000000000b9")


def _scalar_one(value):
    """Result whose scalar_one_or_none() returns `value`."""
    res = MagicMock()
    res.scalar_one_or_none.return_value = value
    return res


def _scalars_all(values):
    """Result whose scalars().all() returns `values`."""
    res = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = values
    res.scalars.return_value = scalars
    return res


def _rows_all(rows):
    """Result whose all() returns `rows`."""
    res = MagicMock()
    res.all.return_value = rows
    return res


def _make_dataset(dataset_id, project):
    ds = MagicMock()
    ds.id = dataset_id
    ds.project = project
    ds.project_id = project.id
    return ds


@pytest_asyncio.fixture
async def venn_client():
    """Client with auth + license overridden; DB execute is set per-test."""
    from app.main import app
    from app.api.deps import get_current_user, get_db
    from app.api.deps.license import require_active_license

    fake_user = make_fake_supabase_user()

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    async def _override_user():
        return fake_user

    async def _override_db():
        yield mock_db

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[require_active_license] = lambda: None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, mock_db

    app.dependency_overrides.clear()


class TestVennCrossDataset:

    @pytest.mark.asyncio
    async def test_cross_dataset_returns_intersections(self, venn_client):
        """Two comparisons in two datasets of the same project → intersections."""
        client, mock_db = venn_client
        project = make_project(project_id=TEST_PROJECT_ID, owner_id=TEST_USER_ID)
        path_dataset = _make_dataset(PATH_DATASET_ID, project)

        # Sequence of db.execute results:
        # 1) path dataset lookup
        # 2) referenced-datasets project validation (OTHER_DATASET_ID)
        # 3) DegGene genes for ref A
        # 4) DegGene genes for ref B
        other_row = MagicMock()
        other_row.id = OTHER_DATASET_ID
        other_row.project_id = TEST_PROJECT_ID

        mock_db.execute = AsyncMock(side_effect=[
            _scalar_one(path_dataset),
            _rows_all([other_row]),
            _scalars_all(["GENE1", "GENE2", "SHARED"]),
            _scalars_all(["GENE3", "SHARED"]),
        ])

        response = await client.post(
            f"/api/v1/datasets/{PATH_DATASET_ID}/venn-analysis",
            json={
                "comparison_refs": [
                    {"dataset_id": str(PATH_DATASET_ID), "comparison_name": "A", "label": "A"},
                    {"dataset_id": str(OTHER_DATASET_ID), "comparison_name": "B", "label": "B"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert set(data["sets"]) == {"A", "B"}
        assert data["total_genes"] == {"A": 3, "B": 2}
        # The A∩B intersection should contain exactly SHARED
        shared = next(i for i in data["intersections"] if sorted(i["sets"]) == ["A", "B"])
        assert shared["genes"] == ["SHARED"]

    @pytest.mark.asyncio
    async def test_foreign_project_dataset_rejected(self, venn_client):
        """A referenced dataset from a different project → 400."""
        client, mock_db = venn_client
        project = make_project(project_id=TEST_PROJECT_ID, owner_id=TEST_USER_ID)
        path_dataset = _make_dataset(PATH_DATASET_ID, project)

        foreign_row = MagicMock()
        foreign_row.id = FOREIGN_DATASET_ID
        foreign_row.project_id = OTHER_PROJECT_ID  # different project

        mock_db.execute = AsyncMock(side_effect=[
            _scalar_one(path_dataset),
            _rows_all([foreign_row]),
        ])

        response = await client.post(
            f"/api/v1/datasets/{PATH_DATASET_ID}/venn-analysis",
            json={
                "comparison_refs": [
                    {"dataset_id": str(PATH_DATASET_ID), "comparison_name": "A", "label": "A"},
                    {"dataset_id": str(FOREIGN_DATASET_ID), "comparison_name": "B", "label": "B"},
                ],
            },
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_fewer_than_two_comparisons_rejected(self, venn_client):
        """A single comparison ref → 400."""
        client, mock_db = venn_client
        project = make_project(project_id=TEST_PROJECT_ID, owner_id=TEST_USER_ID)
        path_dataset = _make_dataset(PATH_DATASET_ID, project)

        mock_db.execute = AsyncMock(side_effect=[_scalar_one(path_dataset)])

        response = await client.post(
            f"/api/v1/datasets/{PATH_DATASET_ID}/venn-analysis",
            json={
                "comparison_refs": [
                    {"dataset_id": str(PATH_DATASET_ID), "comparison_name": "A", "label": "A"},
                ],
            },
        )

        assert response.status_code == 400
