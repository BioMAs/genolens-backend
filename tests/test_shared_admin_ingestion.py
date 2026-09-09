"""
Un membre ADMIN dépose dans le projet partagé, sur SON quota.

Les trois routes d'ingestion — `POST /datasets/upload`,
`POST /datasets/import-from-geo` et `POST /analyses` — filtraient sur
`Project.owner_id == current_user.user_id`. Un membre ADMIN, à qui le partage
donne pourtant le droit d'éditer, reprocesser et supprimer les datasets du
projet, recevait 404 « Project not found » sur un projet qu'il avait sous les
yeux. Elles passent sur `is_project_admin`, la barrière que le partage ADMIN
est justement censé ouvrir.

Le quota reste celui de l'APPELANT, pas du propriétaire : le décompte lit
`db_user`, résolu depuis le jeton, et c'est ce que ce fichier épingle. Un membre
consomme donc ses propres unités en déposant chez autrui — l'alternative
(débiter le propriétaire) laisserait un invité vider le compteur de quelqu'un
d'autre.

Un membre au niveau USER, lui, reste refusé : c'est ce qui distingue les deux
niveaux de partage.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.sql.dml import Update

from app.models.models import (
    Dataset,
    Project,
    ProjectMember,
    SubscriptionPlan,
    User,
    UserRole,
    UserStatus,
)


def make_user(used: int = 0) -> User:
    u = User()
    u.id = uuid4()
    u.email = f"{uuid4().hex[:8]}@example.com"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.TEAM
    u.analyses_used_this_month = used
    u.status = UserStatus.ACTIVE
    u.quota_reset_at = datetime.now(timezone.utc)
    return u


def make_client(*, caller: User, owner_id, member_level, captured: list):
    """`member_level` : le niveau de partage de l'appelant, ou None s'il n'est
    pas membre du tout. `captured` recueille les requêtes émises, pour vérifier
    ensuite QUI est débité."""
    from app.api.deps.license import require_active_license
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    project = MagicMock(spec=Project)
    project.id = uuid4()
    project.owner_id = owner_id

    membership = None
    if member_level is not None:
        membership = MagicMock(spec=ProjectMember)
        membership.project_id = project.id
        membership.user_id = caller.id
        membership.access_level = member_level

    def _result_for(stmt):
        captured.append(stmt)
        value = None
        if isinstance(stmt, Update):
            value = caller.analyses_used_this_month + 1
        else:
            try:
                entity = stmt.column_descriptions[0]["entity"]
            except Exception:
                entity = None
            if entity is Project:
                # Le mock doit honorer un eventuel filtre `owner_id` porte par
                # la requete : sans ca il rendrait le projet a n'importe qui, et
                # le test passerait au vert contre la version owner-only comme
                # contre la version corrigee — il ne prouverait rien.
                params = stmt.compile().params
                filtered_owner = next((v for k, v in params.items() if "owner_id" in k), None)
                if filtered_owner is not None and filtered_owner != project.owner_id:
                    value = None
                else:
                    value = project
            elif entity is ProjectMember:
                # `is_project_admin` ne cherche QUE les membres ADMIN : un
                # membre USER doit donc rester introuvable pour cette requête.
                value = membership if member_level == UserRole.ADMIN else None
            elif entity is Dataset:
                value = None
        res = MagicMock()
        res.scalar_one_or_none = MagicMock(return_value=value)
        res.scalar = MagicMock(return_value=value)
        res.scalar_one = MagicMock(return_value=0)
        res.unique = MagicMock(return_value=res)
        return res

    async def _refresh(obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid4()

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock(side_effect=_refresh)
    db.execute = AsyncMock(side_effect=lambda stmt, *a, **k: _result_for(stmt))
    db.scalar = AsyncMock(side_effect=lambda stmt, *a, **k: _result_for(stmt).scalar_one_or_none())

    async def _fake_db():
        yield db

    app.dependency_overrides[get_or_create_user] = lambda: caller
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=caller.id, email=caller.email
    )
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[require_active_license] = lambda: None

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver"), project


def upload_form(project_id):
    return (
        {"project_id": str(project_id), "name": "DEG partage", "dataset_type": "DEG"},
        {"file": ("deg.csv", b"gene,log2FC,padj\nTP53,1.2,0.01\n", "text/csv")},
    )


def quota_update_targets(captured) -> set:
    """Ids d'utilisateur visés par un UPDATE — c'est-à-dire qui est débité."""
    targets = set()
    for stmt in captured:
        if isinstance(stmt, Update):
            targets.update(
                str(v) for v in stmt.compile().params.values() if isinstance(v, type(uuid4()))
            )
    return targets


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


async def test_admin_member_can_upload_into_a_shared_project():
    member = make_user(used=3)
    owner_id = uuid4()
    captured: list = []
    client, project = make_client(
        caller=member, owner_id=owner_id, member_level=UserRole.ADMIN, captured=captured
    )
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
            resp = await c.post("/api/v1/datasets/upload", data=data, files=files)

    assert resp.status_code == 201, resp.text

    # Le decompte vise le membre, pas le proprietaire du projet.
    targets = quota_update_targets(captured)
    assert str(member.id) in targets, "le quota du membre n'a pas ete debite"
    assert str(owner_id) not in targets, "le quota du proprietaire a ete debite a sa place"


async def test_plain_member_still_cannot_upload():
    """Le partage USER donne la lecture, pas le depot — sinon les deux niveaux
    de partage se vaudraient."""
    member = make_user()
    captured: list = []
    client, project = make_client(
        caller=member, owner_id=uuid4(), member_level=UserRole.USER, captured=captured
    )
    data, files = upload_form(project.id)

    with (
        patch(
            "app.api.endpoints.datasets.storage_service.upload_file",
            new=AsyncMock(return_value="projects/x/raw/deg.csv"),
        ),
        patch("app.api.endpoints.datasets.process_dataset_upload.delay", MagicMock()),
    ):
        async with client as c:
            resp = await c.post("/api/v1/datasets/upload", data=data, files=files)

    assert resp.status_code in (403, 404)
    assert quota_update_targets(captured) == set(), "un refus ne doit rien decompter"


async def test_outsider_still_cannot_upload():
    outsider = make_user()
    captured: list = []
    client, project = make_client(
        caller=outsider, owner_id=uuid4(), member_level=None, captured=captured
    )
    data, files = upload_form(project.id)

    with (
        patch(
            "app.api.endpoints.datasets.storage_service.upload_file",
            new=AsyncMock(return_value="projects/x/raw/deg.csv"),
        ),
        patch("app.api.endpoints.datasets.process_dataset_upload.delay", MagicMock()),
    ):
        async with client as c:
            resp = await c.post("/api/v1/datasets/upload", data=data, files=files)

    assert resp.status_code in (403, 404)


@pytest.mark.parametrize("level,expected_ok", [(UserRole.ADMIN, True), (UserRole.USER, False)])
async def test_analysis_launch_follows_the_same_rule(level, expected_ok):
    """`POST /analyses` est l'autre porte d'entree des donnees : la meme regle
    doit y valoir, sans quoi le bouton « New Analysis » resterait casse pour un
    membre ADMIN alors que « Upload », qui pointe vers le meme ecran, marche."""
    member = make_user(used=1)
    captured: list = []
    client, project = make_client(
        caller=member, owner_id=uuid4(), member_level=level, captured=captured
    )

    comparisons_ds = MagicMock(spec=Dataset)
    comparisons_ds.id = uuid4()
    comparisons_ds.dataset_metadata = {"rows": 4}

    payload = {
        "project_id": str(project.id),
        "name": "Analyse partagee",
        "data_type": "transcriptomics",
        "matrix_dataset_id": str(uuid4()),
        "samples_dataset_id": str(uuid4()),
        "comparisons_dataset_id": str(comparisons_ds.id),
        "params": {},
    }

    # La tache est importee dans le corps de la route, donc patchee a la source.
    with patch("app.worker.tasks.run_self_service_analysis", MagicMock()):
        async with client as c:
            resp = await c.post("/api/v1/analyses", json=payload)

    if expected_ok:
        # Le controle projet est franchi : la route echoue plus loin (le dataset
        # de comparaisons est introuvable dans ce mock), jamais sur l'acces.
        assert resp.status_code != 403
        assert resp.json().get("detail") != "Project not found"
    else:
        assert resp.status_code in (403, 404)
