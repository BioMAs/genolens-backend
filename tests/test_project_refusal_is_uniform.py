"""
Un tiers reçoit toujours le même refus : 404 « Project not found ».

Cinq modules portaient leur propre copie de la règle d'accès projet, et quatre
d'entre elles refusaient en **403 « Access denied »** là où la trentaine de
routes de `datasets.py` refusait en **404 « Project not found »**. Le code de
réponse dépendait donc du module touché, pas de la situation — et un 403 sur un
projet dont on n'est pas membre confirme au passage que ce projet existe.

Ces routes passent sur `assert_project_read_access`, la garde partagée : 404
dans les deux cas, projet inexistant comme appelant sans droit.

Ce fichier épingle le code exact, pas seulement « un refus ». C'est le seul
moyen qu'une régression vers 403 se voie : un test tolérant `(403, 404)`
laisserait chaque module redériver sa propre convention, ce qui est précisément
l'état d'où l'on vient.

`GET /analyses/{id}` et les routes de rapport gardent volontairement leur 403 —
voir `assert_project_access`, et le commentaire de `AnalysisResultsHub.tsx` qui
s'appuie sur la distinction pour dire « vous n'avez pas accès » plutôt que
« introuvable ».
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.models import (
    Dataset,
    Project,
    ProjectMember,
    SubscriptionPlan,
    User,
    UserRole,
    UserStatus,
)

# (méthode, chemin, corps JSON) — un tiers doit recevoir 404 sur chacune.
UNIFIED_ON_404 = [
    ("GET", "/api/v1/projects/{proj}/history", None),
    ("POST", "/api/v1/datasets/{ds}/intersection-enrichment", {"genes": ["TP53"]}),
    ("POST", "/api/v1/datasets/{ds}/gsea-async", {"comparison_name": "A_vs_B"}),
    ("GET", "/api/v1/projects/{proj}/gene-sets", None),
    (
        "POST",
        "/api/v1/projects/{proj}/gene-sets",
        {"name": "set", "genes": ["TP53"]},
    ),
    ("DELETE", "/api/v1/projects/{proj}/gene-sets/{gs}", None),
]


def make_user() -> User:
    u = User()
    u.id = uuid4()
    u.email = f"{uuid4().hex[:8]}@example.com"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.TEAM
    u.status = UserStatus.ACTIVE
    u.scientific_module_enabled = True
    return u


def make_client(*, caller: User, dataset, project):
    """L'appelant n'est ni propriétaire ni membre : toute requête `ProjectMember`
    revient vide."""
    from app.api.deps.license import require_active_license
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    def _value_for(stmt):
        try:
            entity = stmt.column_descriptions[0]["entity"]
        except Exception:
            return None
        if entity is Dataset:
            return dataset
        if entity is Project:
            return project
        return None  # jamais membre

    def _result_for(stmt):
        value = _value_for(stmt)
        res = MagicMock()
        res.scalar_one_or_none = MagicMock(return_value=value)
        res.scalar = MagicMock(return_value=value)
        res.unique = MagicMock(return_value=res)
        res.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        return res

    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock(side_effect=lambda stmt, *a, **k: _result_for(stmt))
    db.scalar = AsyncMock(side_effect=lambda stmt, *a, **k: _value_for(stmt))
    # `gene_sets` charge le projet par clé primaire plutôt que par SELECT.
    db.get = AsyncMock(side_effect=lambda model, pk, *a, **k: project if model is Project else None)

    async def _fake_db():
        yield db

    # DEUX identites d'utilisateur courant coexistent dans le projet :
    # `deps.supabase_deps.get_current_user` rend un `SupabaseUser` (attribut
    # `user_id`), `deps.auth.get_current_user` rend un `CurrentUser` (attribut
    # `id`). `history.py` depend du second. Surcharger un seul des deux laisse
    # l'autre authentifier pour de vrai, et la route repond 403 « Not
    # authenticated » — un faux rouge qui ressemble a s'y meprendre au 403
    # d'acces que ce fichier traque.
    from app.api.deps.auth import get_current_user as get_current_user_auth
    from app.core.security import CurrentUser

    app.dependency_overrides[get_or_create_user] = lambda: caller
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=caller.id, email=caller.email
    )
    app.dependency_overrides[get_current_user_auth] = lambda: CurrentUser(
        id=caller.id,
        email=caller.email,
        role=UserRole.USER,
        subscription_tier=SubscriptionPlan.TEAM,
        max_projects=10,
        current_project_count=0,
        features_access={},
        is_active=True,
    )
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[require_active_license] = lambda: None

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "method,template,body",
    UNIFIED_ON_404,
    ids=[f"{m}:{t.rsplit('/', 1)[-1] or t}" for m, t, _ in UNIFIED_ON_404],
)
async def test_outsider_gets_a_404_not_a_403(method, template, body):
    outsider = make_user()

    project = MagicMock(spec=Project)
    project.id = uuid4()
    project.owner_id = uuid4()  # quelqu'un d'autre

    dataset = MagicMock(spec=Dataset)
    dataset.id = uuid4()
    dataset.project_id = project.id
    dataset.project = project
    dataset.dataset_metadata = {}

    url = template.format(proj=project.id, ds=dataset.id, gs=uuid4())
    client = make_client(caller=outsider, dataset=dataset, project=project)

    async with client as c:
        resp = await c.request(method, url, json=body)

    assert resp.status_code == 404, (
        f"{method} {template} a répondu {resp.status_code} — attendu 404. "
        f"Un 403 confirme au tiers que le projet existe, et fait dépendre le "
        f"code de réponse du module touché plutôt que de la situation."
    )
    assert resp.json()["detail"] == "Project not found"


def test_dead_project_guard_is_gone():
    """`deps/auth.py::check_project_access` n'avait aucun appelant.

    La laisser en place, c'était offrir une sixième variante de la règle à qui
    chercherait un helper — avec sa propre exception pour le rôle ADMIN global,
    que ni `assert_project_access` ni `assert_project_read_access` n'appliquent.
    """
    import app.api.deps.auth as auth

    assert not hasattr(auth, "check_project_access")
