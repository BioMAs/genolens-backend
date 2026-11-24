"""
Shared test support for the local authentication module.

Registered as a pytest plugin from `tests/conftest.py` (`pytest_plugins`), so its fixtures are
available to every test without an import. Later tasks extend this module (app factory, client
helpers, fake mailer, ...); keep it free of anything specific to a single test file.

Everything runs against an in-memory SQLite database. Nothing hosted is called.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import jwt
import pytest
import pytest_asyncio

from app.models.models import Project, ProjectMember, SubscriptionPlan, User, UserRole, UserStatus
from app.models.service_token import ServiceToken

# A fixed instant, so expiry arithmetic in the tests never depends on the wall clock.
NOW = datetime(2026, 11, 24, 12, 0, 0, tzinfo=timezone.utc)

# The header the web client sends on every request; required on cookie-authenticated writes.
CSRF = {"X-GenoLens-Client": "test"}

# A password the policy accepts (length, not the e-mail, not common). Never a real one.
PW = "Lantern-Otter-Harbour-27"


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def access_cookie(token: str) -> dict:
    """The `gl_at` cookie as a request header (httpx deprecates per-request `cookies=`)."""
    return {"Cookie": f"gl_at={token}"}


@pytest_asyncio.fixture
async def auth_db():
    """A session factory over an in-memory SQLite database holding the auth tables."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.schema import CreateTable

    from app.models.auth import AuthEvent, AuthOneTimeToken, AuthRecoveryCode, AuthSession

    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    tables = (
        User,
        ServiceToken,
        Project,
        ProjectMember,
        AuthSession,
        AuthOneTimeToken,
        AuthRecoveryCode,
        AuthEvent,
    )
    try:
        async with engine.begin() as conn:
            for model in tables:
                # Table only: some Postgres index names collide in SQLite's single namespace.
                await conn.execute(CreateTable(model.__table__))
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


async def seed_user(sessions, **fields) -> User:
    """An ACTIVE TEAM account of a fictitious organisation (`@example.test`).

    Any column can be overridden through `fields`; none of the auth columns is set by default,
    so the account has no password, no MFA and no lock.
    """
    values = dict(
        id=uuid4(),
        email=f"user-{uuid4().hex[:8]}@example.test",
        role=UserRole.USER,
        subscription_plan=SubscriptionPlan.TEAM,
        status=UserStatus.ACTIVE,
    )
    values.update(fields)
    async with sessions() as s:
        user = User(**values)
        s.add(user)
        await s.commit()
        return user


@pytest.fixture
def local_keys(monkeypatch):
    """Two freshly generated signing keys (the first signs, the second is "the previous one").

    Sets `AUTH_JWT_PRIVATE_KEYS` and drops the key cache before and after the test, so that no
    key outlives it. Keys are generated here, never read from the environment or from a file.
    """
    from app.auth.keys import generate_private_key_b64, reset_key_cache
    from app.core.config import settings

    reset_key_cache()
    monkeypatch.setattr(
        settings,
        "AUTH_JWT_PRIVATE_KEYS",
        ",".join(generate_private_key_b64() for _ in range(2)),
    )
    try:
        yield
    finally:
        reset_key_cache()


@pytest.fixture
def enc_keys(monkeypatch):
    """Two freshly generated Fernet keys, `(k1, k2)`; the first encrypts, both decrypt.

    Sets `AUTH_ENCRYPTION_KEYS` and drops the cipher cache before and after the test, so that no
    key outlives it. A test that needs another combination (a rotation, say) sets the setting
    itself and calls `reset_encryption_cache()`. Keys are generated here, never read from the
    environment or from a file.
    """
    from app.auth.mfa import generate_encryption_key, reset_encryption_cache
    from app.core.config import settings

    k1, k2 = generate_encryption_key(), generate_encryption_key()
    reset_encryption_cache()
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", f"{k1},{k2}")
    try:
        yield k1, k2
    finally:
        reset_encryption_cache()


@dataclass(frozen=True)
class CapturedEmail:
    """One message as `send_email` handed it to the SMTP library: decoded, and never sent."""

    to: str
    subject: str
    text: str
    html: str


class SmtpCapture:
    """What `smtp_capture` yields: the messages that would have gone out, oldest first.

    Set `fail_with` to an exception and every later send raises it, like an unreachable server.
    """

    def __init__(self) -> None:
        self.messages: list[CapturedEmail] = []
        self.fail_with: Optional[Exception] = None

    @property
    def last(self) -> CapturedEmail:
        assert self.messages, "no e-mail was sent"
        return self.messages[-1]

    @property
    def last_text(self) -> str:
        return self.last.text

    async def send(self, message, **_smtp_options) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        # Serialise as the SMTP library would: a message that cannot be sent fails here too.
        message.as_bytes()
        bodies = {
            part.get_content_subtype(): part.get_payload(decode=True).decode("utf-8")
            for part in message.get_payload()
        }
        self.messages.append(
            CapturedEmail(
                to=message["To"],
                subject=message["Subject"],
                text=bodies["plain"],
                html=bodies["html"],
            )
        )


@pytest.fixture
def smtp_capture(monkeypatch):
    """Capture every e-mail `send_email` sends, instead of reaching an SMTP server.

    Replaces `aiosmtplib.send` and sets placeholder SMTP settings (`@example.test`, no real host)
    so that the module counts as configured: a `required=True` send is captured, not refused. A
    test of the unconfigured case sets `settings.SMTP_HOST` back to None itself. Nothing leaves
    the process.
    """
    from app.core.config import settings

    for name, value in (
        ("SMTP_HOST", "smtp.example.test"),
        ("SMTP_USER", "mailer"),
        ("SMTP_PASSWORD", "placeholder"),
        ("EMAIL_FROM_ADDRESS", "no-reply@example.test"),
    ):
        monkeypatch.setattr(settings, name, value)
    capture = SmtpCapture()
    monkeypatch.setattr("app.services.email_service.aiosmtplib.send", capture.send)
    return capture


@pytest.fixture
def settings_override(monkeypatch):
    """`settings_override(AUTH_MODE="local", ...)`: set settings for the rest of the test.

    Restored afterwards by `monkeypatch`. A name that is not a setting is an error, so a typo
    cannot silently test the default configuration.
    """
    from app.core.config import settings

    def apply(**values) -> None:
        for name, value in values.items():
            if not hasattr(settings, name):
                raise AttributeError(f"no setting named {name}")
            monkeypatch.setattr(settings, name, value)

    return apply


@dataclass
class DbSessionCounter:
    """How many database sessions the app opened through `get_db` (see `auth_client`)."""

    opened: int = 0


@pytest.fixture
def db_sessions_opened() -> DbSessionCounter:
    return DbSessionCounter()


@pytest_asyncio.fixture
async def auth_client(auth_db, db_sessions_opened, monkeypatch):
    """The app on the `auth_db` database, with the real authentication chain.

    Modelled on the `client` fixture of `tests/test_service_token.py`: every dependency override
    is cleared first (conftest's default account must not stand in for the caller), then only
    the database and the licence check are replaced. Nothing hosted is reached: the Supabase role
    lookup fails the test if called, and login events (which open a session of their own) are not
    recorded. The rate limiter starts and ends empty.

    Response schemas validate e-mails, and `.test` is a special-use domain the validator refuses
    unless told it runs in a test environment: its documented switch is turned on, so that the
    fictitious `@example.test` accounts can be returned.
    """
    import email_validator
    from httpx import ASGITransport, AsyncClient

    from app.api.deps.license import require_active_license
    from app.db.session import get_db
    from app.main import app
    from app.middleware.rate_limit import limiter

    async def _get_db():
        db_sessions_opened.opened += 1
        async with auth_db() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def _no_supabase(*_args, **_kwargs):
        raise AssertionError("Supabase was called")

    monkeypatch.setattr(email_validator, "TEST_ENVIRONMENT", True)
    limiter.reset()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[require_active_license] = lambda: None
    try:
        with (
            patch("app.core.supabase_auth.fetch_user_role_from_supabase", new=_no_supabase),
            patch("app.middleware.login_tracking._record_login_event", new=AsyncMock()),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as client:
                yield client
    finally:
        app.dependency_overrides.clear()
        limiter.reset()


@pytest.fixture
def local_session_for(auth_db, local_keys):
    """`await local_session_for(user)`: open a local session for `user` and return its tokens.

    Opened at the current instant (the API checks tokens and sessions against the wall clock).
    """
    from app.auth.sessions import IssuedSession, create_session

    async def open_session(user: User, amr: tuple[str, ...] = ("pwd",)) -> IssuedSession:
        async with auth_db() as session:
            issued = await create_session(
                session, user, list(amr), ip="203.0.113.7", user_agent="pytest"
            )
            await session.commit()
        return issued

    return open_session


@pytest.fixture
def supabase_token_for(monkeypatch):
    """`supabase_token_for(user)`: a hosted-provider token of `user` (HS256, one hour).

    The role lookup it triggers is answered here with the role of the row the token was minted
    for (what the hosted profile of a synced account holds); nothing hosted is reached. Extra
    `claims` override the default ones.
    """
    from app.core.config import settings

    roles: dict = {}

    async def _role(user_id):
        return roles.get(user_id)

    monkeypatch.setattr("app.core.supabase_auth.get_cached_user_role", _role)

    def mint(user: User, **claims) -> str:
        roles[user.id] = user.role.value.lower()
        payload = {
            "sub": str(user.id),
            "email": user.email,
            "aud": "authenticated",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        payload.update(claims)
        return jwt.encode(payload, settings.SUPABASE_JWT_SECRET, algorithm="HS256")

    return mint
