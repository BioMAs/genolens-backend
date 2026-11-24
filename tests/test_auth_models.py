"""
Data model of the local authentication module: the new `users` columns and the four auth tables
(sessions, one-time tokens, MFA recovery codes, audit events).

Runs against an in-memory SQLite database (see `tests/auth_support.py`); nothing hosted is called.
"""

from datetime import timedelta

from sqlalchemy import select, text

from app.models.auth import (
    AuthEvent,
    AuthEventType,
    AuthOneTimeToken,
    AuthRecoveryCode,
    AuthSession,
    OneTimeTokenPurpose,
)
from app.models.models import User
from tests.auth_support import NOW, seed_user


async def test_auth_tables_round_trip(auth_db):
    # auth_db: SQLite engine with User + the four auth tables (fixture added in this task,
    # in tests/auth_support.py, registered from tests/conftest.py via pytest_plugins)
    user = await seed_user(auth_db)
    async with auth_db() as s:
        s.add(
            AuthSession(
                user_id=user.id,
                amr=["pwd", "otp"],
                refresh_token_hash="a" * 64,
                idle_expires_at=NOW + timedelta(days=7),
                expires_at=NOW + timedelta(days=30),
            )
        )
        s.add(
            AuthOneTimeToken(
                user_id=user.id,
                purpose=OneTimeTokenPurpose.INVITE,
                token_hash="b" * 64,
                expires_at=NOW + timedelta(days=7),
            )
        )
        s.add(AuthRecoveryCode(user_id=user.id, code_hash="c" * 64))
        s.add(
            AuthEvent(
                user_id=user.id,
                event_type=AuthEventType.LOGIN_SUCCEEDED.value,
                details={"k": 1},
            )
        )
        await s.commit()
    async with auth_db() as s:
        session = (await s.execute(select(AuthSession))).scalar_one()
        token = (await s.execute(select(AuthOneTimeToken))).scalar_one()
        db_user = await s.get(User, user.id)
    assert session.amr == ["pwd", "otp"] and session.revoked_at is None
    assert token.attempts == 0 and token.purpose is OneTimeTokenPurpose.INVITE
    assert db_user.failed_login_count == 0 and db_user.password_hash is None
    assert db_user.mfa_failed_count == 0


async def test_event_ids_increment_and_purpose_is_stored_as_its_value(auth_db):
    user = await seed_user(auth_db)
    async with auth_db() as s:
        s.add(
            AuthOneTimeToken(
                user_id=user.id,
                purpose=OneTimeTokenPurpose.PASSWORD_RESET,
                token_hash="d" * 64,
                expires_at=NOW + timedelta(hours=1),
            )
        )
        s.add(AuthEvent(user_id=user.id, event_type=AuthEventType.SIGNUP.value))
        s.add(AuthEvent(user_id=None, event_type=AuthEventType.BREAK_GLASS.value))
        await s.commit()
    async with auth_db() as s:
        stored = (await s.execute(text("SELECT purpose FROM auth_one_time_tokens"))).scalar_one()
        events = (await s.execute(select(AuthEvent).order_by(AuthEvent.id))).scalars().all()
    assert stored == "password_reset"
    assert [e.id for e in events] == [1, 2]
    assert events[0].details == {} and events[1].user_id is None


def test_enum_values_are_the_lowercase_member_names():
    assert all(m.value == m.name.lower() for m in AuthEventType)
    assert all(m.value == m.name.lower() for m in OneTimeTokenPurpose)
    assert len(AuthEventType) == 23 and len(OneTimeTokenPurpose) == 5
