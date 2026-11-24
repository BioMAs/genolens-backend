"""
`lookup_user_by_email` must return the account whose e-mail IS the one asked for, or None.

It used to call `GET /auth/v1/admin/users?email=<address>` and return `users[0]`. GoTrue
has no `email` parameter on that route (internal/api/admin.go reads only `filter`, `sort`,
`page` and `per_page`), so it ignored it and listed every user, newest first. The lookup
therefore answered with the most recently created account of the Supabase project, whatever
address was typed. `POST /projects/{id}/members` then made THAT account a member of the
project, with the requested access level, while the invitation e-mail went to the address
actually typed. And an address with no account never reached the "invite them to sign up"
branch, since the list is never empty.

GoTrue's `filter` narrows the list but cannot decide the answer on its own: it is a LIKE
substring match on the e-mail (`bob@lab.org` also matches `jimbob@lab.org`, and `_` matches
any character) or an ILIKE on the full name. Only the exact, case-insensitive comparison of
the returned e-mail does.

`FakeGoTrue` reproduces that route as GoTrue implements it: unknown parameters ignored,
`filter` applied as SQL LIKE / ILIKE, created_at descending, 50 per page by default.
"""

import re
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio

from app.core import supabase_auth
from app.core.config import settings
from app.models.models import Project, ProjectMember

_RealAsyncClient = httpx.AsyncClient
_EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _like(value: str, pattern: str) -> bool:
    """SQL `value LIKE '%' || pattern || '%'`: `_` is any one character, `%` any run."""
    body = "".join(
        "." if ch == "_" else ".*" if ch == "%" else re.escape(ch) for ch in pattern
    )
    return re.search(body, value, flags=re.DOTALL) is not None


def make_user(email: str, age_days: int, full_name: str = "") -> dict:
    """A GoTrue user row; GoTrue stores e-mails lower-cased. Lower age = newer."""
    return {
        "id": str(uuid4()),
        "email": email.lower(),
        "created_at": (_EPOCH - timedelta(days=age_days)).isoformat(),
        "user_metadata": {"full_name": full_name},
    }


class FakeGoTrue:
    """`GET /auth/v1/admin/users` as GoTrue serves it (offset pagination)."""

    def __init__(self, users: list):
        self.users = users
        self.requests: list = []
        self.status_code = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/auth/v1/admin/users"
        params = request.url.params
        self.requests.append(dict(params))
        if self.status_code != 200:
            return httpx.Response(self.status_code, json={"msg": "boom"})

        rows = self.users
        flt = params.get("filter", "")
        if flt:
            rows = [
                u
                for u in rows
                if _like(u["email"], flt)
                or _like(u["user_metadata"]["full_name"].lower(), flt.lower())
            ]
        rows = sorted(rows, key=lambda u: u["created_at"], reverse=True)
        page = int(params.get("page") or 1)
        per_page = int(params.get("per_page") or 50)
        rows = rows[(page - 1) * per_page : page * per_page]
        return httpx.Response(200, json={"users": rows, "aud": "authenticated"})


ALICE = make_user("alice@lab.org", age_days=400)
BOB = make_user("bob@lab.org", age_days=300)
A_UNDERSCORE_B = make_user("a_b@lab.org", age_days=250)
AXB = make_user("axb@lab.org", age_days=200)
JIMBOB = make_user("jimbob@lab.org", age_days=100)
NAMED_LIKE_AN_EMAIL = make_user("c.other@else.org", age_days=50, full_name="Carol carol@lab.org")
NEWEST = make_user("newest@lab.org", age_days=1)
DIRECTORY = [ALICE, BOB, A_UNDERSCORE_B, AXB, JIMBOB, NAMED_LIKE_AN_EMAIL, NEWEST]


@pytest.fixture
def gotrue(monkeypatch):
    fake = FakeGoTrue(list(DIRECTORY))

    def client_factory(*args, **kwargs):
        # Clients given their own transport (the ASGI test client) are left alone.
        kwargs.setdefault("transport", httpx.MockTransport(fake.handler))
        return _RealAsyncClient(*args, **kwargs)

    monkeypatch.setattr(supabase_auth.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://supabase.test")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "service-role-test-key")
    return fake


# ── The lookup itself ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_the_account_asked_for_not_the_newest(gotrue):
    user = await supabase_auth.lookup_user_by_email("bob@lab.org")

    assert user is not None
    assert user["id"] == BOB["id"]
    # The address is sent as GoTrue's `filter`, with explicit pagination so that
    # GoTrue's experimental cursor mode (taken when both are absent) never applies.
    assert gotrue.requests[0]["filter"] == "bob@lab.org"
    assert "page" in gotrue.requests[0] and "per_page" in gotrue.requests[0]


@pytest.mark.asyncio
async def test_an_address_with_no_account_returns_none(gotrue):
    assert await supabase_auth.lookup_user_by_email("nobody@lab.org") is None


@pytest.mark.asyncio
async def test_the_comparison_ignores_case_and_surrounding_spaces(gotrue):
    user = await supabase_auth.lookup_user_by_email("  Alice@LAB.org ")

    assert user is not None
    assert user["id"] == ALICE["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "asked,expected",
    [
        ("bob@lab.org", BOB),  # filter also matches the newer jimbob@lab.org
        ("a_b@lab.org", A_UNDERSCORE_B),  # `_` also matches the newer axb@lab.org
        ("carol@lab.org", None),  # only a full name contains it: no account
    ],
    ids=["substring", "underscore-wildcard", "full-name-only"],
)
async def test_what_the_filter_lets_through_never_decides(gotrue, asked, expected):
    user = await supabase_auth.lookup_user_by_email(asked)

    if expected is None:
        assert user is None
    else:
        assert user is not None and user["id"] == expected["id"]


@pytest.mark.asyncio
async def test_a_match_past_the_first_page_is_found(gotrue):
    # 2 500 newer accounts all contain "x@lab.org", so the oldest one, the exact
    # match, sits on the last page whatever page size the lookup asks for.
    target = make_user("x@lab.org", age_days=5000)
    gotrue.users = [make_user(f"u{i}x@lab.org", age_days=i) for i in range(2500)] + [target]

    user = await supabase_auth.lookup_user_by_email("x@lab.org")

    assert user is not None and user["id"] == target["id"]


@pytest.mark.asyncio
async def test_a_supabase_error_returns_none(gotrue):
    gotrue.status_code = 500

    assert await supabase_auth.lookup_user_by_email("bob@lab.org") is None


# ── Through POST /projects/{id}/members ──────────────────────────────────────

OWNER_ID = UUID("00000000-0000-0000-0000-0000000000f1")


@pytest_asyncio.fixture
async def members_api(gotrue, monkeypatch):
    """The invite endpoint over a mocked session; yields (client, db, invitations, project)."""
    from app.api.deps.license import require_active_license
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.api.endpoints import projects as projects_module
    from app.main import app

    project = MagicMock(spec=Project)
    project.id = uuid4()
    project.owner_id = OWNER_ID
    project.name = "Shared project"

    def execute(stmt, *args, **kwargs):
        entity = stmt.column_descriptions[0]["entity"]
        result = MagicMock()
        # The project exists; the invitee is not yet a member.
        result.scalar_one_or_none.return_value = project if entity is Project else None
        return result

    db = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock(side_effect=execute)

    async def refresh(obj):
        obj.id = uuid4()
        obj.created_at = obj.updated_at = datetime.now(timezone.utc)

    db.refresh = AsyncMock(side_effect=refresh)

    async def _fake_db():
        yield db

    sent = AsyncMock()
    monkeypatch.setattr(projects_module, "is_project_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(projects_module.email_service, "send_project_invitation", sent)
    monkeypatch.setattr(projects_module.history_service, "log_activity", AsyncMock())

    app.dependency_overrides[get_current_user] = lambda: supabase_auth.SupabaseUser(
        user_id=OWNER_ID, email="owner@lab.org"
    )
    app.dependency_overrides[get_db] = _fake_db
    app.dependency_overrides[require_active_license] = lambda: None

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client, db, sent, project

    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(require_active_license, None)


@pytest.mark.asyncio
async def test_inviting_an_address_with_no_account_adds_nobody(members_api):
    client, db, sent, project = members_api

    response = await client.post(
        f"/api/v1/projects/{project.id}/members", json={"user_email": "nobody@lab.org"}
    )

    assert response.status_code == 404, response.text
    db.add.assert_not_called()
    assert sent.await_args.kwargs["invitee_email"] == "nobody@lab.org"


@pytest.mark.asyncio
async def test_inviting_an_address_adds_that_account(members_api):
    client, db, sent, project = members_api

    response = await client.post(
        f"/api/v1/projects/{project.id}/members", json={"user_email": "bob@lab.org"}
    )

    assert response.status_code == 201, response.text
    (member,) = db.add.call_args.args
    assert isinstance(member, ProjectMember)
    assert member.user_id == UUID(BOB["id"])
