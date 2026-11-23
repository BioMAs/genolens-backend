"""
Shared test support for the local authentication module.

Registered as a pytest plugin from `tests/conftest.py` (`pytest_plugins`), so its fixtures are
available to every test without an import. Later tasks extend this module (app factory, client
helpers, fake mailer, ...); keep it free of anything specific to a single test file.

Everything runs against an in-memory SQLite database. Nothing hosted is called.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import pytest
import pytest_asyncio

from app.models.models import Project, ProjectMember, SubscriptionPlan, User, UserRole, UserStatus
from app.models.service_token import ServiceToken

# A fixed instant, so expiry arithmetic in the tests never depends on the wall clock.
NOW = datetime(2026, 11, 24, 12, 0, 0, tzinfo=timezone.utc)


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
