"""
Quota au lancement d'une analyse, **une unité par analyse**.

L'unité facturable est celle que déclare la grille tarifaire
(`app/config/pricing.json`, `billable_unit.id = deg_dataset_upload`) : « +1 par
fichier accepté, **indépendamment du nombre de contrastes qu'il contient** ».
Une analyse à 20 contrastes coûte donc autant qu'une analyse à 1.

La première version de cette barrière comptait les contrastes : elle refusait
une analyse à 12 contrastes à un compte qui avait 5 unités restantes, et un
STARTER à 30/mois atteignait son plafond en 5 analyses au lieu de 30. Les tests
d'alors épinglaient ce comptage ; ils épinglent maintenant son inverse, et
`test_allows_an_analysis_declaring_more_contrasts_than_remain` est là pour que
la régression ne puisse pas revenir sans être vue.

Reste la réservation en vol : le compteur n'avance qu'à la fin du worker, donc
la route déduit les analyses déjà admises et pas encore comptées — une unité
chacune.
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
    u.analyses_used_this_month = used
    u.status = UserStatus.ACTIVE
    # DOIT etre dans le mois courant. `_reset_quota_if_new_month` remet
    # `analyses_used_this_month` A ZERO quand `quota_reset_at` est None
    # (app/api/deps/subscription.py:257), ce qui annulerait chaque assertion
    # de quota de ce fichier : `used` serait toujours relu a 0.
    u.quota_reset_at = datetime.now(timezone.utc)
    return u


def make_comparisons_dataset(declared: int | None, *, key: str = "rows") -> Dataset:
    """Dataset de comparaisons dont la métadonnée déclare `declared` lignes.

    `key` par défaut à `rows` : c'est celle que l'ingestion écrit réellement.
    """
    ds = MagicMock(spec=Dataset)
    ds.id = uuid4()
    ds.dataset_metadata = {} if declared is None else {key: declared}
    return ds


def make_metadata_dataset(metadata: dict) -> Dataset:
    """Dataset dont la métadonnée est fournie telle quelle (non fabriquée)."""
    ds = MagicMock(spec=Dataset)
    ds.id = uuid4()
    ds.dataset_metadata = metadata
    return ds


def make_client(
    *,
    as_user: User,
    comparisons_dataset: Dataset | None,
    project_owned: bool = True,
    in_flight: int = 0,
):
    """Client dont l'utilisateur résolu est `as_user`.

    `db.scalar` est appelé trois fois par la route : le projet, le dataset de
    comparaisons, puis le COUNT des analyses en vol. On répond dans cet ordre.
    `comparisons_dataset=None` simule un id absent du projet. Sans l'override
    de get_db la route toucherait une vraie base.

    `in_flight` est le nombre d'analyses PENDING / RUNNING de l'appelant, que
    renvoie le COUNT de `_in_flight_analyses` — troisième et dernier appel à
    `db.scalar` de la route.
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
    db.execute = AsyncMock(return_value=MagicMock())
    db.scalar = AsyncMock(
        side_effect=[project if project_owned else None, comparisons_dataset, in_flight]
    )

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


async def test_allows_an_analysis_declaring_more_contrasts_than_remain():
    """Le nombre de contrastes n'entre pas dans le quota : une analyse à 12
    contrastes passe avec une seule unité restante."""
    user = make_user(used=29)  # STARTER: 30 - 29 = 1 restante
    ds = make_comparisons_dataset(declared=12)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201
    added = [type(call.args[0]).__name__ for call in db.add.call_args_list]
    assert "SelfServiceAnalysis" in added


async def test_allows_when_a_unit_remains():
    user = make_user(used=25)  # 5 restantes
    ds = make_comparisons_dataset(declared=5)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201


async def test_refuses_when_quota_already_exhausted():
    user = make_user(used=30)
    ds = make_comparisons_dataset(declared=1)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 429
    # Pas `db.add.assert_not_called()` : `check_analysis_quota` peut
    # legitimement ajouter l'utilisateur pour persister une remise a zero
    # mensuelle. On verifie ce qui compte -- aucune analyse creee.
    added = [type(call.args[0]).__name__ for call in db.add.call_args_list]
    assert "SelfServiceAnalysis" not in added


async def test_allows_when_rows_metadata_is_absent():
    """La métadonnée de lignes n'est plus lue du tout par la barrière : un
    dataset encore en traitement ne change donc rien au verdict."""
    user = make_user(used=25)
    ds = make_comparisons_dataset(declared=None)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201


async def test_no_quota_check_for_admin_role():
    user = make_user(role=UserRole.ADMIN, used=999)
    ds = make_comparisons_dataset(declared=500)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201


async def test_unlimited_plan_is_never_refused():
    user = make_user(plan=SubscriptionPlan.ON_PREMISE, used=999)
    ds = make_comparisons_dataset(declared=500)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201


async def test_in_flight_analyses_count_one_unit_each():
    """La réservation compte les analyses en vol, pas leurs contrastes.

    Le compteur n'avance qu'à la fin du worker : sans déduction, dix analyses
    postées coup sur coup voient toutes le même reste et passent toutes. Mais
    la déduction se fait à raison d'une unité par analyse — sommer les
    contrastes déclarés, comme le faisait la première version, refusait des
    lancements légitimes.
    """
    user = make_user(used=28)  # STARTER: 2 restantes
    ds = make_comparisons_dataset(declared=1)
    # Deux analyses en vol, chacune déclarant 10 contrastes : 2 unités
    # réservées, donc 0 restante.
    client, db, project = make_client(as_user=user, comparisons_dataset=ds, in_flight=2)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 429
    added = [type(call.args[0]).__name__ for call in db.add.call_args_list]
    assert "SelfServiceAnalysis" not in added


async def test_in_flight_analyses_leave_room_when_under_the_quota():
    user = make_user(used=27)  # 3 restantes
    ds = make_comparisons_dataset(declared=40)
    # 2 unités réservées, 1 libre
    client, db, project = make_client(as_user=user, comparisons_dataset=ds, in_flight=2)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201
