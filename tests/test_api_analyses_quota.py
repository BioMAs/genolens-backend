"""
`POST /analyses` ne vérifiait aucun quota. Le pipeline R produit un dataset DEG
par contraste, donc lancer une analyse à 20 contrastes avec 5 comparaisons
restantes doit être refusé au lancement — après, le calcul a déjà tourné et on
ne l'annule pas.

Le nombre de contrastes vit dans `dataset_metadata["total_rows"]` du dataset de
comparaisons ; il n'est pas dans la charge utile de la requête.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.models import (
    Dataset,
    Project,
    SubscriptionPlan,
    User,
    UserRole,
    UserStatus,
)

ENDPOINT = "/api/v1/analyses"


def make_user(
    plan: SubscriptionPlan = SubscriptionPlan.STARTER,
    role: UserRole = UserRole.USER,
    used: int = 0,
) -> User:
    u = User()
    u.id = uuid4()
    u.email = "quota@example.com"
    u.role = role
    u.subscription_plan = plan
    u.comparisons_used_this_month = used
    u.status = UserStatus.ACTIVE
    # DOIT etre dans le mois courant. `_reset_quota_if_new_month` remet
    # `comparisons_used_this_month` A ZERO quand `quota_reset_at` est None
    # (app/api/deps/subscription.py:257), ce qui annulerait chaque assertion
    # de quota de ce fichier : `used` serait toujours relu a 0.
    u.quota_reset_at = datetime.now(timezone.utc)
    return u


def make_comparisons_dataset(total_rows: int | None) -> Dataset:
    ds = MagicMock(spec=Dataset)
    ds.id = uuid4()
    ds.dataset_metadata = {} if total_rows is None else {"total_rows": total_rows}
    return ds


def make_client(*, as_user: User, comparisons_dataset: Dataset, project_owned: bool = True):
    """Client dont l'utilisateur résolu est `as_user`.

    `db.scalar` est appelé deux fois par la route : d'abord pour le projet,
    ensuite (après notre ajout) pour le dataset de comparaisons. On répond dans
    cet ordre. Sans l'override de get_db la route toucherait une vraie base.
    """
    from app.api.deps.license import require_active_license
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    project = MagicMock(spec=Project)
    project.id = uuid4()
    project.owner_id = as_user.id

    async def _refresh(obj):
        # Mirrors the real AsyncSession.refresh(): after INSERT, the DB
        # populates the server-defaulted timestamp columns. `SelfServiceAnalysis`
        # (app/models/models.py:1857) only sets these via `server_default`, so
        # without this the object keeps `created_at=None` and
        # `SelfServiceAnalysisResponse.from_orm` (analyses.py) blows up on
        # `.isoformat()` for every allowed (201) case — unrelated to the quota
        # gate under test. Same pattern as
        # tests/test_api_intersection_enrichment.py's `_refresh` side_effect.
        now = datetime.now(timezone.utc)
        if getattr(obj, "created_at", None) is None:
            obj.created_at = now
        obj.updated_at = now

    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock(side_effect=_refresh)
    db.execute = AsyncMock()
    db.scalar = AsyncMock(side_effect=[project if project_owned else None, comparisons_dataset])

    async def _fake_db():
        yield db

    app.dependency_overrides[get_or_create_user] = lambda: as_user
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=as_user.id, email=as_user.email
    )
    app.dependency_overrides[get_db] = _fake_db
    # `POST /analyses` porte `dependencies=[Depends(require_active_license)]`, et
    # `get_license_status()` rend `valid=False` des que GENOLENS_LICENSE_KEY ou
    # LICENSE_SECRET_KEY manque (app/core/license.py:89). Sans cet override, les
    # six tests recevraient 403 sans jamais atteindre le controle de quota, et les
    # tests de refus echoueraient (403 != 429). L'override rend le test hermetique
    # au .env du poste.
    app.dependency_overrides[require_active_license] = lambda: None

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver"), db, project


def payload_for(project_id, comparisons_dataset_id):
    return {
        "project_id": str(project_id),
        "name": "Analyse quota",
        # OmicsDataType (models.py:51) n'accepte que transcriptomics /
        # proteomics / lipidomics. Une autre valeur donne un 422, ce qui
        # ferait passer les tests de refus pour la mauvaise raison.
        "data_type": "transcriptomics",
        "matrix_dataset_id": str(uuid4()),
        "samples_dataset_id": str(uuid4()),
        "comparisons_dataset_id": str(comparisons_dataset_id),
        "params": {},
    }


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


async def test_refuses_when_requested_contrasts_exceed_remaining():
    user = make_user(used=25)  # STARTER: 30 - 25 = 5 restantes
    ds = make_comparisons_dataset(total_rows=12)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 429
    detail = res.json()["detail"]
    assert "12" in detail and "5" in detail
    # Pas `db.add.assert_not_called()` : `check_comparison_quota` peut
    # legitimement ajouter l'utilisateur pour persister une remise a zero
    # mensuelle. On verifie ce qui compte -- aucune analyse creee.
    added = [type(call.args[0]).__name__ for call in db.add.call_args_list]
    assert "SelfServiceAnalysis" not in added


async def test_allows_when_requested_contrasts_fit():
    user = make_user(used=25)  # 5 restantes
    ds = make_comparisons_dataset(total_rows=5)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201


async def test_refuses_when_quota_already_exhausted():
    user = make_user(used=30)
    ds = make_comparisons_dataset(total_rows=1)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 429


async def test_allows_when_total_rows_metadata_is_absent():
    """Dataset encore en traitement : on se rabat sur « au moins une
    comparaison restante ». Le worker plafonnera le compteur."""
    user = make_user(used=25)
    ds = make_comparisons_dataset(total_rows=None)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201


async def test_no_quota_check_for_admin_role():
    user = make_user(role=UserRole.ADMIN, used=999)
    ds = make_comparisons_dataset(total_rows=500)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201


async def test_unlimited_plan_is_never_refused():
    user = make_user(plan=SubscriptionPlan.ON_PREMISE, used=999)
    ds = make_comparisons_dataset(total_rows=500)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201
