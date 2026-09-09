"""
Contrôle d'accès projet : ce qu'un tiers ne doit pas atteindre, ce qu'un membre
partagé doit atteindre.

Deux familles de routes échouaient, dans les deux sens opposés.

1. Des routes SANS aucun contrôle : `POST /{id}/gsea`,
   `GET /{id}/gsea/{set}/enrichment-plot`, `POST /{id}/precompute-sample-clustering`,
   `POST /{id}/rerun-enrichment` et `GET /enrichment/{id}/comparisons` ne
   regardaient ni le propriétaire ni l'appartenance. N'importe quel compte
   authentifié pouvait les appeler avec l'id de dataset d'un autre client — lire
   ses gènes, ses corrélations d'échantillons, et pour `rerun-enrichment`
   réécrire son enrichissement. Deux d'entre elles portaient même le commentaire
   du contrôle manquant (« Fetch dataset for auth check », « 1. Check
   permissions »).

2. Des routes TROP fermées : les visualisations PCA / UMAP / boxplot et les deux
   routes de conversation IA comparaient `owner_id` à la main et rendaient 403 à
   un membre partagé, alors que la trentaine de routes voisines l'acceptent via
   `assert_project_read_access`. Un membre voyait la table des DEG mais pas
   l'ACP du même dataset.

Le mock de session répond d'après l'entité interrogée plutôt que dans un ordre
figé : les routes n'émettent pas les mêmes requêtes ni dans le même ordre, et un
`side_effect` positionnel casserait au premier refactor.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio

from app.models.models import Dataset, Project, ProjectMember, User, UserRole, UserStatus

# Motifs de refus émis par les barrières d'accès elles-mêmes, avant toute
# lecture. `assert_project_read_access` rend « Project not found » (un 404
# délibéré : révéler l'existence du projet à un tiers serait déjà une fuite) ;
# les contrôles écrits à la main rendent l'un des trois autres.
AUTHORIZATION_REFUSALS = {
    "Project not found",
    "Access denied",
    "Not authorized to access this dataset",
    "Admin access required for this project",
}

# ── Routes qui ne doivent jamais répondre à un tiers ─────────────────────────
# (méthode, gabarit d'URL, corps JSON, paramètres de requête)
OUTSIDER_MUST_BE_REFUSED = [
    ("POST", "/api/v1/datasets/{ds}/rerun-enrichment", None, None),
    ("POST", "/api/v1/datasets/{ds}/gsea", {"comparison_name": "A_vs_B"}, None),
    (
        "GET",
        "/api/v1/datasets/{ds}/gsea/HALLMARK_APOPTOSIS/enrichment-plot",
        None,
        {"comparison_name": "A_vs_B"},
    ),
    ("POST", "/api/v1/datasets/{ds}/precompute-sample-clustering", None, None),
    ("GET", "/api/v1/enrichment/{ds}/comparisons", None, None),
    ("POST", "/api/v1/datasets/{ds}/comparisons/A_vs_B/ask", {"question": "why?"}, None),
    ("GET", "/api/v1/datasets/{ds}/comparisons/A_vs_B/conversations", None, None),
    ("POST", "/api/v1/datasets/{ds}/comparisons/A_vs_B/visualizations/pca", {}, None),
    ("POST", "/api/v1/datasets/{ds}/comparisons/A_vs_B/visualizations/umap", {}, None),
    ("POST", "/api/v1/datasets/{ds}/comparisons/A_vs_B/visualizations/boxplot", {}, None),
]

# ── Routes qu'un membre partagé doit franchir ────────────────────────────────
# Elles rendaient 403 à un membre. Ce qu'elles font ENSUITE (lire un parquet
# absent, appeler le LLM) sort du périmètre : le test n'affirme que le
# franchissement de la barrière d'accès, pas le succès du calcul.
MEMBER_MUST_PASS_THE_GATE = [
    ("GET", "/api/v1/datasets/{ds}/comparisons/A_vs_B/conversations", None),
    ("POST", "/api/v1/datasets/{ds}/comparisons/A_vs_B/visualizations/pca", {}),
    ("POST", "/api/v1/datasets/{ds}/comparisons/A_vs_B/visualizations/umap", {}),
    ("POST", "/api/v1/datasets/{ds}/comparisons/A_vs_B/visualizations/boxplot", {}),
]


def make_user(role: UserRole = UserRole.USER) -> User:
    u = User()
    u.id = uuid4()
    u.email = f"{uuid4().hex[:8]}@example.com"
    u.role = role
    u.status = UserStatus.ACTIVE
    from app.models.models import SubscriptionPlan

    u.subscription_plan = SubscriptionPlan.TEAM
    # Indispensable : sans le module, `require_scientific_access` rend 403 AVANT
    # que la route n'atteigne le moindre contrôle projet, et les cas GSEA
    # passeraient au vert sans rien prouver.
    u.scientific_module_enabled = True
    return u


def make_db(*, dataset, project, member):
    """Session dont chaque réponse est choisie d'après l'entité interrogée.

    `member` à None simule l'absence de ligne `project_members` : c'est ce qui
    distingue le tiers du membre partagé, les deux étant par ailleurs
    indiscernables pour la route.
    """

    def _entity_of(stmt):
        try:
            return stmt.column_descriptions[0]["entity"]
        except Exception:  # requête textuelle (SELECT ... FROM deg_genes)
            return None

    def _result_for(stmt):
        entity = _entity_of(stmt)
        value = None
        if entity is Dataset:
            value = dataset
        elif entity is Project:
            value = project
        elif entity is ProjectMember:
            value = member
        res = MagicMock()
        res.scalar_one_or_none = MagicMock(return_value=value)
        res.scalar = MagicMock(return_value=value)
        res.scalars = MagicMock(
            return_value=MagicMock(
                all=MagicMock(return_value=[]), first=MagicMock(return_value=None)
            )
        )
        res.fetchall = MagicMock(return_value=[])
        res.all = MagicMock(return_value=[])
        # `.unique()` précède `.scalar_one_or_none()` sur les requêtes joinedload
        res.unique = MagicMock(return_value=res)
        return res

    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock(side_effect=lambda stmt, *a, **k: _result_for(stmt))
    db.scalar = AsyncMock(side_effect=lambda stmt, *a, **k: _entity_lookup(stmt, dataset, project))
    return db


def _entity_lookup(stmt, dataset, project):
    try:
        entity = stmt.column_descriptions[0]["entity"]
    except Exception:
        return None
    return dataset if entity is Dataset else (project if entity is Project else None)


def make_client(*, caller: User, dataset, project, member):
    from app.api.deps.license import require_active_license
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    db = make_db(dataset=dataset, project=project, member=member)

    async def _fake_db():
        yield db

    app.dependency_overrides[get_or_create_user] = lambda: caller
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=caller.id, email=caller.email
    )
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[require_active_license] = lambda: None

    from httpx import ASGITransport, AsyncClient

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


def make_fixtures(*, owner_id):
    project = MagicMock(spec=Project)
    project.id = uuid4()
    project.owner_id = owner_id

    dataset = MagicMock(spec=Dataset)
    dataset.id = uuid4()
    dataset.project_id = project.id
    dataset.project = project
    dataset.dataset_metadata = {}
    dataset.parquet_file_path = "does/not/exist.parquet"
    return dataset, project


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "method,template,body,params",
    OUTSIDER_MUST_BE_REFUSED,
    ids=[f"{m}:{t.split('{ds}')[-1] or '/'}" for m, t, _, _ in OUTSIDER_MUST_BE_REFUSED],
)
async def test_outsider_cannot_touch_another_tenants_dataset(method, template, body, params):
    """Un compte authentifié sans lien avec le projet doit être refusé.

    404 comme 403 conviennent — la seule chose inacceptable est que la route
    parte calculer, écrire ou renvoyer les données d'un autre client.
    """
    outsider = make_user()
    dataset, project = make_fixtures(owner_id=uuid4())  # propriétaire : quelqu'un d'autre
    client = make_client(caller=outsider, dataset=dataset, project=project, member=None)

    async with client as c:
        resp = await c.request(method, template.format(ds=dataset.id), json=body, params=params)

    detail = (
        resp.json().get("detail")
        if resp.headers.get("content-type", "").startswith("application/json")
        else None
    )

    # Le code seul ne suffit pas : deux de ces routes rendaient déjà 404, mais
    # « No DEG data found » — autrement dit elles étaient allées interroger les
    # données de l'autre client avant de constater qu'il n'y avait rien à
    # renvoyer. Un 404 fortuit n'est pas un refus. On exige donc un motif
    # d'AUTORISATION, émis avant que la route ne touche à quoi que ce soit.
    assert resp.status_code in (403, 404) and detail in AUTHORIZATION_REFUSALS, (
        f"{method} {template} a répondu {resp.status_code} « {detail} » à un tiers. "
        f"Attendu un refus d'accès parmi {sorted(AUTHORIZATION_REFUSALS)} — la route "
        f"ne doit pas atteindre les données d'un autre projet."
    )


@pytest.mark.parametrize(
    "method,template,body",
    MEMBER_MUST_PASS_THE_GATE,
    ids=[f"{m}:{t.split('/')[-1]}" for m, t, _ in MEMBER_MUST_PASS_THE_GATE],
)
async def test_shared_member_is_not_denied_access(method, template, body):
    """Un membre partagé franchit la barrière — il recevait 403.

    On n'affirme rien du résultat : le parquet du fixture n'existe pas, donc la
    route échoue plus loin. C'est précisément la preuve recherchée — elle a
    dépassé le contrôle d'accès.
    """
    member_user = make_user()
    dataset, project = make_fixtures(owner_id=uuid4())
    membership = MagicMock(spec=ProjectMember)
    membership.project_id = project.id
    membership.user_id = member_user.id
    membership.access_level = UserRole.USER

    client = make_client(caller=member_user, dataset=dataset, project=project, member=membership)

    async with client as c:
        resp = await c.request(method, template.format(ds=dataset.id), json=body)

    assert resp.status_code != 403, (
        f"{method} {template} a rendu 403 à un membre du projet, alors que les "
        f"routes voisines l'acceptent via assert_project_read_access."
    )
