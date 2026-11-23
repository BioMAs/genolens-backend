"""
Shared test support for the local authentication module.

Registered as a pytest plugin from `tests/conftest.py` (`pytest_plugins`), so its fixtures are
available to every test without an import. Later tasks extend this module (app factory, client
helpers, fake mailer, ...); keep it free of anything specific to a single test file.

Everything runs against an in-memory SQLite database. Nothing hosted is called.
"""

from datetime import datetime, timezone
from uuid import uuid4

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
