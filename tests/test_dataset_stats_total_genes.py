"""
`GET /datasets/{id}/stats` must never answer `total_genes: null`.

The ingestion worker wrote `deg_up_count` / `deg_down_count` /
`deg_significant_count` but never `total_genes`. The endpoint's fast path keys on
the DEG counts alone, so it returned `total_genes=None` and the response model
(`total_genes: int`) turned that into a 500 — on all 24 DEG datasets in
production on 2026-10-12 ("Failed to fetch genes-tested count" in the UI).

The worker stores the row count of the uploaded table in
`dataset_metadata["rows"]`, which is what `calculate_deg_stats` itself uses for
`total_genes`. The fast path falls back on it and persists it.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.models import Dataset, DatasetStatus, Project, ProjectMember, User, UserStatus


def _make_db(*, dataset, project):
    def _result_for(stmt):
        try:
            entity = stmt.column_descriptions[0]["entity"]
        except Exception:
            entity = None
        value = {Dataset: dataset, Project: project, ProjectMember: None}.get(entity)
        res = MagicMock()
        res.scalar_one_or_none = MagicMock(return_value=value)
        res.scalar = MagicMock(return_value=value)
        return res

    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock(side_effect=lambda stmt, *a, **k: _result_for(stmt))
    return db


def _make_dataset(project_id, *, total_genes, metadata):
    dataset = MagicMock(spec=Dataset)
    dataset.id = uuid4()
    dataset.project_id = project_id
    dataset.status = DatasetStatus.READY
    dataset.deg_up_count = 120
    dataset.deg_down_count = 80
    dataset.deg_significant_count = 200
    dataset.total_genes = total_genes
    dataset.dataset_metadata = metadata
    dataset.parquet_file_path = "does/not/exist.parquet"
    return dataset


@pytest_asyncio.fixture
async def call_stats():
    from app.api.deps.license import require_active_license
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    caller = User()
    caller.id = uuid4()
    caller.email = "owner@example.com"
    caller.status = UserStatus.ACTIVE

    project = MagicMock(spec=Project)
    project.id = uuid4()
    project.owner_id = caller.id

    async def _call(dataset):
        db = _make_db(dataset=dataset, project=project)

        async def _fake_db():
            yield db

        app.dependency_overrides[get_or_create_user] = lambda: caller
        app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
            user_id=caller.id, email=caller.email
        )
        app.dependency_overrides[get_db] = _fake_db
        app.dependency_overrides[require_active_license] = lambda: None
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.get(f"/api/v1/datasets/{dataset.id}/stats"), db

    _call.project = project
    yield _call
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_missing_total_genes_falls_back_on_metadata_rows(call_stats):
    dataset = _make_dataset(call_stats.project.id, total_genes=None, metadata={"rows": 24932})

    response, db = await call_stats(dataset)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_genes"] == 24932
    # The DEG counts stay those the worker wrote — never recomputed here.
    assert (body["up_genes"], body["down_genes"], body["significant_genes"]) == (120, 80, 200)
    # Persisted, so the next call is a plain column read.
    assert dataset.total_genes == 24932
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_stored_total_genes_is_returned_as_is(call_stats):
    dataset = _make_dataset(call_stats.project.id, total_genes=18000, metadata={"rows": 24932})

    response, db = await call_stats(dataset)

    assert response.status_code == 200, response.text
    assert response.json()["total_genes"] == 18000
