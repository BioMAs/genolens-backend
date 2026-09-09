"""
Un 429 de quota à l'upload d'un DEG ne doit laisser aucun dataset derrière lui.

`increment_analysis_usage` était appelé APRÈS le commit du dataset. Son
`rollback()` de secours — celui qui devait « annuler le dataset » quand une
requête concurrente venait d'épuiser le quota entre le contrôle
(`check_analysis_quota`) et l'écriture atomique — ne portait alors sur plus
rien : la ligne était déjà durable. L'appelant recevait un 429 pour un dataset
bel et bien créé, qui comptait ensuite contre la limite par projet sans avoir
jamais été facturé.

Le décompte partage donc désormais la transaction de l'insert. Le test épingle
l'invariant observable de bout en bout : sur le chemin refusé, aucun `commit()`
n'est émis, et un `rollback()` l'est.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.models import Project, SubscriptionPlan, User, UserRole, UserStatus

ENDPOINT = "/api/v1/datasets/upload"


def make_user(used: int) -> User:
    """TEAM (quota 150, datasets par projet illimités — la route saute alors la
    requête de comptage, ce qui garde `db.execute` à deux appels prévisibles)."""
    u = User()
    u.id = uuid4()
    u.email = "quota@example.com"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.TEAM
    u.analyses_used_this_month = used
    u.status = UserStatus.ACTIVE
    # Doit être dans le mois courant, sinon `_reset_quota_if_new_month` remet
    # le compteur à zéro et le scénario « quota saturé » disparaît.
    u.quota_reset_at = datetime.now(timezone.utc)
    return u


def make_client(*, as_user: User, update_returns: int | None):
    """`update_returns` est ce que rend l'UPDATE ... RETURNING atomique du
    décompte : un entier quand la ligne a bougé, `None` quand le quota l'a
    bloquée — c'est-à-dire quand une requête concurrente l'a épuisé."""
    from app.api.deps.license import require_active_license
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    project = MagicMock(spec=Project)
    project.id = uuid4()
    project.owner_id = as_user.id

    project_result = MagicMock()
    project_result.scalar_one_or_none = MagicMock(return_value=project)

    update_result = MagicMock()
    update_result.scalar = MagicMock(return_value=update_returns)

    async def _refresh(obj):
        # Reproduit AsyncSession.refresh() après INSERT : la base attribue la
        # clé primaire. Sans ça `DatasetUploadResponse.dataset_id` reste None et
        # le chemin accepté échoue sur une ResponseValidationError, sans rapport
        # avec le quota testé ici.
        if getattr(obj, "id", None) is None:
            obj.id = uuid4()

    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock(side_effect=_refresh)
    db.execute = AsyncMock(side_effect=[project_result, update_result])

    async def _fake_db():
        yield db

    app.dependency_overrides[get_or_create_user] = lambda: as_user
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=as_user.id, email=as_user.email
    )
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[require_active_license] = lambda: None

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver"), db, project


def upload_form(project_id):
    return (
        {
            "project_id": str(project_id),
            "name": "DEG quota",
            "dataset_type": "DEG",
        },
        {"file": ("deg.csv", b"gene,log2FC,padj\nTP53,1.2,0.01\n", "text/csv")},
    )


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


async def test_concurrent_quota_exhaustion_leaves_no_dataset_committed():
    user = make_user(used=149)  # TEAM : 1 unité restante au contrôle d'entrée
    client, db, project = make_client(as_user=user, update_returns=None)
    data, files = upload_form(project.id)

    with (
        patch(
            "app.api.endpoints.datasets.storage_service.upload_file",
            new=AsyncMock(return_value="projects/x/raw/deg.csv"),
        ),
        patch("app.api.endpoints.datasets.process_dataset_upload.delay", MagicMock()),
    ):
        async with client as c:
            resp = await c.post(ENDPOINT, data=data, files=files)

    assert resp.status_code == 429
    # Le coeur du correctif : l'insert n'a jamais été rendu durable, donc le
    # rollback l'emporte réellement avec lui.
    db.commit.assert_not_awaited()
    db.rollback.assert_awaited_once()


async def test_accepted_upload_commits_once_with_the_counter():
    user = make_user(used=10)
    client, db, project = make_client(as_user=user, update_returns=11)
    data, files = upload_form(project.id)

    with (
        patch(
            "app.api.endpoints.datasets.storage_service.upload_file",
            new=AsyncMock(return_value="projects/x/raw/deg.csv"),
        ),
        patch("app.api.endpoints.datasets.process_dataset_upload.delay", MagicMock()),
        patch("app.api.endpoints.datasets.history_service.log_activity", new=AsyncMock()),
    ):
        async with client as c:
            resp = await c.post(ENDPOINT, data=data, files=files)

    assert resp.status_code == 201
    # Un seul commit : dataset et compteur sont rendus durables ensemble.
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()
