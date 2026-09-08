"""
Verrou sur `project_count` dans GET /users/me.

Le backend applique la limite de projets sur les projets POSSEDES uniquement
(app/api/endpoints/projects.py, create_project: `Project.owner_id == user`),
mais GET /projects renvoie l'union possedes-ou-partages (list_projects:
`or_(Project.owner_id == user, Project.id.in_(member_subquery))`) et son
`total` compte cette union. Un client qui lirait ce `total` pour afficher
"used/max" verrait un plafond atteint alors que le backend accepterait encore
des creations - un echec FERME qui bloque une action legitime.

`project_count` doit donc etre calcule avec exactement le meme predicat que
create_project. Pour verrouiller cette semantique (et pas seulement la
presence du champ), le test ci-dessous fait executer la requete SQL produite
par l'endpoint contre une vraie base SQLite en memoire, peuplee avec des
projets possedes ET partages : si l'implementation substituait le filtre de
list_projects, la requete renverrait 5 (2 possedes + 3 partages) et le test
echouerait sur `project_count == 2`.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine

from app.models.models import (
    Project,
    ProjectMember,
    SubscriptionPlan,
    User,
    UserRole,
    UserStatus,
)

ENDPOINT = "/api/v1/users/me"


def make_user() -> User:
    u = User()
    u.id = uuid4()
    u.email = "counts@example.com"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.STARTER
    u.status = UserStatus.ACTIVE
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.analyses_used_this_month = 0
    u.cosmetics_module_enabled = False
    u.report_customization_module_enabled = False
    u.scientific_module_enabled = False
    u.drug_discovery_module_enabled = False
    return u


def make_sqlite_engine():
    """A real (in-memory) engine so the endpoint's actual SQL runs for real.

    A bare AsyncMock for `db` would just return whatever we told it to and
    could never notice a swapped WHERE clause. Running the real statement
    against real rows is what makes the predicate swap detectable.
    """
    engine = create_engine("sqlite:///:memory:")
    Project.__table__.create(engine)
    ProjectMember.__table__.create(engine)
    return engine


def seed_owned_and_shared_projects(engine, *, owner_id, owned: int, shared: int):
    """`owner_id` owns `owned` projects and is a member (not owner) of `shared` more."""
    with engine.begin() as conn:
        for _ in range(owned):
            conn.execute(
                Project.__table__.insert().values(id=uuid4(), name="owned", owner_id=owner_id)
            )
        for _ in range(shared):
            other_project_id = uuid4()
            conn.execute(
                Project.__table__.insert().values(
                    id=other_project_id, name="shared", owner_id=uuid4()
                )
            )
            conn.execute(
                ProjectMember.__table__.insert().values(
                    id=uuid4(),
                    project_id=other_project_id,
                    user_id=owner_id,
                    access_level=UserRole.VIEWER,
                )
            )


def make_client(as_user: User, engine):
    from app.api.deps.subscription import get_or_create_user
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.core.supabase_auth import SupabaseUser
    from app.main import app

    async def _fake_db():
        db = AsyncMock()

        def _run(stmt):
            with engine.connect() as conn:
                return conn.execute(stmt).scalar()

        db.scalar = AsyncMock(side_effect=_run)
        yield db

    app.dependency_overrides[get_or_create_user] = lambda: as_user
    app.dependency_overrides[get_current_user] = lambda: SupabaseUser(
        user_id=as_user.id, email=as_user.email
    )
    app.dependency_overrides[get_db] = _fake_db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


async def test_project_count_reflects_owned_projects_only_not_shared():
    user = make_user()
    engine = make_sqlite_engine()
    seed_owned_and_shared_projects(engine, owner_id=user.id, owned=2, shared=3)

    async with make_client(user, engine) as client:
        res = await client.get(ENDPOINT)

    assert res.status_code == 200, res.text
    assert res.json()["project_count"] == 2


async def test_project_count_is_present_and_zero_for_a_user_with_no_projects():
    user = make_user()
    engine = make_sqlite_engine()  # no rows seeded at all

    async with make_client(user, engine) as client:
        res = await client.get(ENDPOINT)

    assert res.status_code == 200, res.text
    body = res.json()
    assert "project_count" in body
    assert body["project_count"] == 0
