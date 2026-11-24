"""
Tests of the server-side sessions and their rotating refresh tokens (`app/auth/sessions.py`).

Everything runs against the in-memory SQLite database of `tests/auth_support.py`, with a fixed
clock (`NOW`) passed as `now=`; nothing hosted is called, no wall clock is read. SQLite hands
datetimes back naive, so the assertions go through `_aware` rather than comparing raw columns.
"""

import logging
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import jwt
import pytest
from sqlalchemy import event, select
from sqlalchemy.dialects import postgresql

from app.auth.sessions import (
    ABSOLUTE_TTL,
    IDLE_TTL,
    REFRESH_GRACE,
    IssuedSession,
    RefreshOutcome,
    RefreshRejected,
    create_session,
    get_session_user,
    revoke_all_sessions,
    revoke_session,
    rotate_refresh,
)
from app.auth.tokens import hash_token
from app.models.auth import AuthSession
from app.models.models import User
from tests.auth_support import NOW, seed_user


def _aware(moment: datetime) -> datetime:
    """SQLite returns the naive wall clock of what was stored (UTC); Postgres an aware value."""
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


async def _row(auth_db, session_id) -> AuthSession:
    async with auth_db() as s:
        return (
            await s.execute(select(AuthSession).where(AuthSession.id == session_id))
        ).scalar_one()


async def _keep_alive(auth_db, issued, days) -> str:
    """Refresh every 6 days (inside the 7-day inactivity limit) up to day `days`; returns the
    newest refresh token."""
    token = issued.refresh_token
    for day in range(6, days + 1, 6):
        async with auth_db() as db:
            outcome = await rotate_refresh(db, token, now=NOW + timedelta(days=day))
            await db.commit()
        assert outcome.refresh_token
        token = outcome.refresh_token
    return token


def _claims(token: str) -> dict:
    """The claims of an access token, signature not checked (Task 3 owns that, and the fixed
    clock of these tests is not the wall clock the verifier reads)."""
    return jwt.decode(token, options={"verify_signature": False})


# --------------------------------------------------------------------------- constants


def test_ttl_constants():
    assert IDLE_TTL == timedelta(days=7)
    assert ABSOLUTE_TTL == timedelta(days=30)
    assert REFRESH_GRACE == timedelta(seconds=30)


# --------------------------------------------------------------------------- create_session


async def test_create_session_stores_hashes_only_and_issues_tokens(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd", "otp"], "203.0.113.7", "agent/1.0", now=NOW)
        await db.commit()

    assert isinstance(issued, IssuedSession)
    assert issued.refresh_token and issued.access_token
    assert issued.expires_at == NOW + ABSOLUTE_TTL

    row = await _row(auth_db, issued.session_id)
    assert row.user_id == user.id
    assert row.refresh_token_hash == hash_token(issued.refresh_token)
    assert row.previous_refresh_token_hash is None and row.rotated_at is None
    assert _aware(row.created_at) == NOW and _aware(row.last_used_at) == NOW
    assert _aware(row.idle_expires_at) == NOW + IDLE_TTL
    assert _aware(row.expires_at) == NOW + ABSOLUTE_TTL
    assert row.revoked_at is None and row.revoked_reason is None
    assert row.amr == ["pwd", "otp"]
    assert (row.ip, row.user_agent) == ("203.0.113.7", "agent/1.0")
    # The raw refresh token is stored nowhere, not even as a substring of another column.
    stored = " ".join(str(getattr(row, c.name)) for c in AuthSession.__table__.columns)
    assert issued.refresh_token not in stored

    claims = _claims(issued.access_token)
    assert claims["sub"] == str(user.id) and claims["sid"] == str(issued.session_id)
    assert claims["amr"] == ["pwd", "otp"]
    assert claims["iat"] == int(NOW.timestamp())


async def test_create_session_records_last_login_on_a_detached_user(auth_db, local_keys):
    # `seed_user` hands back an object whose session is closed: assigning its attribute alone
    # would be lost, so the login time has to reach the row some other way.
    user = await seed_user(auth_db)
    assert user.last_login_at is None
    async with auth_db() as db:
        await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()

    assert user.last_login_at == NOW  # the caller's object is up to date too
    async with auth_db() as s:
        stored = (await s.execute(select(User).where(User.id == user.id))).scalar_one()
        assert _aware(stored.last_login_at) == NOW


async def test_create_session_records_last_login_on_an_attached_user(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        attached = (await db.execute(select(User).where(User.id == user.id))).scalar_one()
        await create_session(db, attached, ["pwd"], None, None, now=NOW)
        await db.commit()
        assert _aware(attached.last_login_at) == NOW
    async with auth_db() as s:
        stored = (await s.execute(select(User).where(User.id == user.id))).scalar_one()
        assert _aware(stored.last_login_at) == NOW


async def test_create_session_does_not_commit(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.rollback()  # the caller's transaction fails: nothing of the session survives
    async with auth_db() as s:
        assert (await s.execute(select(AuthSession))).first() is None
        stored = (await s.execute(select(User).where(User.id == user.id))).scalar_one()
        assert stored.last_login_at is None
    assert issued.session_id  # the dataclass was built before the rollback, as the caller sees it


async def test_create_session_clips_what_the_columns_cannot_hold(auth_db, local_keys):
    # On Postgres an over-long header would be a DataError, that is a 500 on /login.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], "9" * 200, "u" * 2000, now=NOW)
        await db.commit()
    row = await _row(auth_db, issued.session_id)
    assert len(row.ip) == 64 and len(row.user_agent) == 512


async def test_create_session_reads_the_instant_not_the_wall_clock(auth_db, local_keys):
    # NOW expressed at UTC+2 is the same instant: what is stored and compared is UTC.
    user = await seed_user(auth_db)
    plus_two = NOW.astimezone(timezone(timedelta(hours=2)))
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=plus_two)
        await db.commit()
        assert issued.expires_at == NOW + ABSOLUTE_TTL
        assert await get_session_user(db, issued.session_id, user.id, now=NOW) is not None
        just_before = NOW + IDLE_TTL - timedelta(seconds=1)
        assert await get_session_user(db, issued.session_id, user.id, now=just_before)
        assert await get_session_user(db, issued.session_id, user.id, now=NOW + IDLE_TTL) is None


async def test_each_session_gets_its_own_tokens(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        one = await create_session(db, user, ["pwd"], None, None, now=NOW)
        two = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    assert one.session_id != two.session_id
    assert one.refresh_token != two.refresh_token


# --------------------------------------------------------------------------- rotate_refresh


async def test_rotation_then_grace_then_reuse(auth_db, local_keys):  # Review Focus 5
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd", "otp"], "1.2.3.4", "ua", now=NOW)
        await db.commit()
        first = await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=20))
        concurrent = await rotate_refresh(
            db, issued.refresh_token, now=NOW + timedelta(minutes=20, seconds=10)
        )
        assert first.refresh_token and concurrent.refresh_token is None and concurrent.access_token
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=21))
        assert e.value.reason == "reuse"
        assert (
            await get_session_user(db, issued.session_id, user.id, now=NOW + timedelta(minutes=21))
            is None
        )


async def test_rotation_replaces_the_current_hash_and_moves_the_clocks(auth_db, local_keys):
    user = await seed_user(auth_db)
    later = NOW + timedelta(days=2)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd", "otp"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as db:
        outcome = await rotate_refresh(db, issued.refresh_token, now=later)
        await db.commit()

    assert isinstance(outcome, RefreshOutcome)
    assert outcome.session_id == issued.session_id
    assert outcome.refresh_token and outcome.refresh_token != issued.refresh_token

    row = await _row(auth_db, issued.session_id)
    assert row.refresh_token_hash == hash_token(outcome.refresh_token)
    assert row.previous_refresh_token_hash == hash_token(issued.refresh_token)
    assert _aware(row.rotated_at) == later
    assert _aware(row.last_used_at) == later
    assert _aware(row.idle_expires_at) == later + IDLE_TTL
    assert _aware(row.expires_at) == NOW + ABSOLUTE_TTL  # the absolute expiry never moves
    assert row.revoked_at is None
    # Raw tokens are stored nowhere.
    stored = " ".join(str(getattr(row, c.name)) for c in AuthSession.__table__.columns)
    assert outcome.refresh_token not in stored and issued.refresh_token not in stored

    # The access token carries the amr of the session row, and its ids.
    claims = _claims(outcome.access_token)
    assert claims["amr"] == ["pwd", "otp"]
    assert claims["sub"] == str(user.id) and claims["sid"] == str(issued.session_id)
    assert claims["iat"] == int(later.timestamp())


async def test_the_new_refresh_token_keeps_the_chain_going(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    token = issued.refresh_token
    for step in range(1, 5):
        async with auth_db() as db:
            outcome = await rotate_refresh(db, token, now=NOW + timedelta(days=step))
            await db.commit()
        assert outcome.refresh_token
        token = outcome.refresh_token
    async with auth_db() as db:
        assert await get_session_user(
            db, issued.session_id, user.id, now=NOW + timedelta(days=4, minutes=1)
        )


async def test_unknown_refresh_token_is_invalid(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
        for garbage in ("", "not-a-token", hash_token("x"), "é" * 40):
            with pytest.raises(RefreshRejected) as e:
                await rotate_refresh(db, garbage, now=NOW)
            assert e.value.reason == "invalid"


async def test_refresh_rejected_exposes_its_reason_and_prints_no_secret():
    exc = RefreshRejected("expired")
    assert exc.reason == "expired" and "expired" in str(exc)


async def test_hash_of_a_stored_hash_is_not_a_refresh_token(auth_db, local_keys):
    # Presenting the stored sha256 itself must not authenticate: only its preimage does.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(db, hash_token(issued.refresh_token), now=NOW)
        assert e.value.reason == "invalid"


async def test_grace_returns_an_access_token_only_and_writes_nothing(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd", "otp"], None, None, now=NOW)
        await db.commit()
    rotated_at = NOW + timedelta(minutes=5)
    async with auth_db() as db:
        first = await rotate_refresh(db, issued.refresh_token, now=rotated_at)
        await db.commit()
    before = await _row(auth_db, issued.session_id)

    async with auth_db() as db:
        grace = await rotate_refresh(
            db, issued.refresh_token, now=rotated_at + timedelta(seconds=12)
        )
        await db.commit()

    assert grace.refresh_token is None and grace.session_id == issued.session_id
    claims = _claims(grace.access_token)
    assert claims["amr"] == ["pwd", "otp"] and claims["sid"] == str(issued.session_id)
    after = await _row(auth_db, issued.session_id)
    for column in AuthSession.__table__.columns:  # not a single column moved
        assert getattr(after, column.name) == getattr(before, column.name), column.name
    assert after.refresh_token_hash == hash_token(first.refresh_token)


async def test_grace_window_boundary(auth_db, local_keys):
    # Inclusive of REFRESH_GRACE itself, closed one tick later.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
        await rotate_refresh(db, issued.refresh_token, now=NOW)
        await db.commit()
        at_edge = await rotate_refresh(db, issued.refresh_token, now=NOW + REFRESH_GRACE)
        assert at_edge.refresh_token is None
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(
                db, issued.refresh_token, now=NOW + REFRESH_GRACE + timedelta(milliseconds=1)
            )
        assert e.value.reason == "reuse"


async def test_reuse_revokes_the_session_and_kills_the_legitimate_chain_too(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as db:
        legit = await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(hours=1))
        await db.commit()

    # The spent token is replayed, long after the grace window: a thief, or a client that
    # kept an old token. The revocation must survive the exception, once the caller commits.
    replay_at = NOW + timedelta(hours=2)
    async with auth_db() as db:
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(db, issued.refresh_token, now=replay_at)
        assert e.value.reason == "reuse"
        await db.commit()

    row = await _row(auth_db, issued.session_id)
    assert _aware(row.revoked_at) == replay_at
    assert row.revoked_reason == "refresh_reuse"

    # The honest holder of the newest token is cut off as well: the session is gone.
    async with auth_db() as db:
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(db, legit.refresh_token, now=replay_at + timedelta(seconds=1))
        assert e.value.reason == "expired"
        assert await get_session_user(db, issued.session_id, user.id, now=replay_at) is None


async def test_reuse_revocation_is_not_committed_by_the_function(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
        await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=1))
        await db.commit()
    async with auth_db() as db:
        with pytest.raises(RefreshRejected):
            await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(hours=1))
        await db.rollback()  # the caller forgot to commit: nothing was persisted behind its back
    assert (await _row(auth_db, issued.session_id)).revoked_at is None


async def test_only_one_generation_of_refresh_token_is_remembered(auth_db, local_keys):
    # A token two rotations old matches no column: it is unknown, not "reuse". Documented limit.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        second = await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(hours=1))
        await rotate_refresh(db, second.refresh_token, now=NOW + timedelta(hours=2))
        await db.commit()
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(hours=3))
        assert e.value.reason == "invalid"
        # ... and it does not take the live session down with it.
        assert await get_session_user(db, issued.session_id, user.id, now=NOW + timedelta(hours=3))


async def test_revoked_takes_precedence_over_reuse(auth_db, local_keys):
    # A spent token of a session that is already revoked is "expired", not "reuse": the dead
    # session is not revoked a second time (its reason and instant stay those of the first
    # revocation) and the caller does not log a second reuse alarm for it.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        legit = await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=1))
        await revoke_session(db, issued.session_id, "logout")
        await db.commit()
    before = await _row(auth_db, issued.session_id)
    assert before.revoked_reason == "logout"

    async with auth_db() as db:
        for spent_or_current in (issued.refresh_token, legit.refresh_token):
            with pytest.raises(RefreshRejected) as e:
                await rotate_refresh(db, spent_or_current, now=NOW + timedelta(hours=1))
            assert e.value.reason == "expired"
        await db.commit()
    after = await _row(auth_db, issued.session_id)
    assert (after.revoked_at, after.revoked_reason) == (before.revoked_at, "logout")


async def test_expired_takes_precedence_over_grace_and_reuse(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=1))
        await db.commit()
        # Same spent token, session idle for more than IDLE_TTL: dead, whatever the grace says.
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(
                db, issued.refresh_token, now=NOW + timedelta(minutes=1) + IDLE_TTL
            )
        assert e.value.reason == "expired"
    assert (await _row(auth_db, issued.session_id)).revoked_at is None  # nothing to revoke


async def test_idle_and_absolute_expiry(auth_db, local_keys):
    user = await seed_user(auth_db)

    # Idle: no refresh for more than 7 days.
    async with auth_db() as db:
        idle = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as db:
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(db, idle.refresh_token, now=NOW + timedelta(days=8))
        assert e.value.reason == "expired"
    # ... but one second short of the idle limit it still works.
    async with auth_db() as db:
        ok = await rotate_refresh(db, idle.refresh_token, now=NOW + IDLE_TTL - timedelta(seconds=1))
        assert ok.refresh_token

    # Absolute: a client that refreshes every 6 days is cut off at 30 days all the same.
    async with auth_db() as db:
        busy = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    token = await _keep_alive(auth_db, busy, 24)
    async with auth_db() as db:
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(db, token, now=NOW + timedelta(days=30))
        assert e.value.reason == "expired"
    async with auth_db() as db:  # the last instant before the absolute limit still renews
        last = await rotate_refresh(db, token, now=NOW + ABSOLUTE_TTL - timedelta(seconds=1))
        assert last.refresh_token


async def test_rotation_never_pushes_idle_expiry_past_the_absolute_one(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    await _keep_alive(auth_db, issued, 24)  # idle would be day 31, absolute is day 30
    row = await _row(auth_db, issued.session_id)
    assert _aware(row.last_used_at) == NOW + timedelta(days=24)
    assert _aware(row.idle_expires_at) == _aware(row.expires_at) == NOW + ABSOLUTE_TTL


async def test_rotate_looks_the_session_up_in_one_locked_query(auth_db, local_keys, monkeypatch):
    # Controller ruling: two concurrent refreshes must not both rotate. The row is read FOR
    # UPDATE (SQLite ignores that, Postgres honours it), by current OR previous hash in a single
    # statement: after waiting on the lock, Postgres re-checks the WHERE against the row the
    # winner committed, and a second statement keyed on the current hash alone would lose it.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as db:
        seen = []
        real_execute = db.execute

        async def spy(statement, *args, **kwargs):
            seen.append(statement)
            return await real_execute(statement, *args, **kwargs)

        monkeypatch.setattr(db, "execute", spy)
        await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=1))

    lookups = [
        s for s in seen if "auth_sessions" in str(s) and str(s).lstrip().startswith("SELECT")
    ]
    assert len(lookups) == 1
    sql = str(lookups[0].compile(dialect=postgresql.dialect()))
    assert sql.rstrip().endswith("FOR UPDATE")
    assert "auth_sessions.refresh_token_hash =" in sql
    assert "auth_sessions.previous_refresh_token_hash =" in sql
    assert " OR " in sql


async def test_rotate_sees_the_row_as_the_lock_returns_it_not_as_it_was_loaded(auth_db, local_keys):
    # A transaction that already holds the session in its identity map (loaded before waiting on
    # the lock) must act on what the winner committed, or it would rotate a second time.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as loser:
        stale = (
            await loser.execute(select(AuthSession).where(AuthSession.id == issued.session_id))
        ).scalar_one()
        assert stale.refresh_token_hash == hash_token(issued.refresh_token)
        async with auth_db() as winner:  # commits its rotation while `loser` holds the old row
            won = await rotate_refresh(winner, issued.refresh_token, now=NOW + timedelta(minutes=1))
            await winner.commit()
        lost = await rotate_refresh(
            loser, issued.refresh_token, now=NOW + timedelta(minutes=1, seconds=1)
        )
        await loser.commit()
    assert won.refresh_token and lost.refresh_token is None  # grace, not a second rotation
    row = await _row(auth_db, issued.session_id)
    assert row.refresh_token_hash == hash_token(won.refresh_token)  # the winner's token survives


async def test_two_sessions_are_independent(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        one = await create_session(db, user, ["pwd"], None, None, now=NOW)
        two = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
        await rotate_refresh(db, one.refresh_token, now=NOW + timedelta(hours=1))
        with pytest.raises(RefreshRejected):
            await rotate_refresh(db, one.refresh_token, now=NOW + timedelta(hours=2))
        await db.commit()
        assert await get_session_user(db, two.session_id, user.id, now=NOW + timedelta(hours=2))
        assert await rotate_refresh(db, two.refresh_token, now=NOW + timedelta(hours=2))


# --------------------------------------------------------------------------- revoke_session


async def test_revoke_session(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as db:
        assert await get_session_user(db, issued.session_id, user.id, now=NOW)
        await revoke_session(db, issued.session_id, "logout")
        await db.commit()
    row = await _row(auth_db, issued.session_id)
    assert row.revoked_at is not None and row.revoked_reason == "logout"
    async with auth_db() as db:
        assert await get_session_user(db, issued.session_id, user.id, now=NOW) is None
        with pytest.raises(RefreshRejected) as e:
            await rotate_refresh(db, issued.refresh_token, now=NOW)
        assert e.value.reason == "expired"


async def test_revoke_session_keeps_the_first_revocation_and_ignores_unknown_ids(
    auth_db, local_keys
):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await revoke_session(db, issued.session_id, "refresh_reuse")
        await db.commit()
        first = await _row(auth_db, issued.session_id)
        await revoke_session(db, issued.session_id, "logout")
        await revoke_session(db, uuid4(), "logout")  # no such session: not an error
        await db.commit()
    again = await _row(auth_db, issued.session_id)
    assert (again.revoked_at, again.revoked_reason) == (first.revoked_at, "refresh_reuse")


async def test_revoke_session_does_not_commit(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as db:
        await revoke_session(db, issued.session_id, "logout")
        await db.rollback()
    assert (await _row(auth_db, issued.session_id)).revoked_at is None


# --------------------------------------------------------------------------- revoke_all_sessions


async def test_revoke_all_keeps_current(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        s1 = await create_session(db, user, ["pwd"], None, None, now=NOW)
        s2 = await create_session(db, user, ["pwd"], None, None, now=NOW)
        s3 = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as db:
        count = await revoke_all_sessions(
            db, user.id, "password_changed", keep_session_id=s1.session_id
        )
        await db.commit()
    assert count == 2
    async with auth_db() as db:
        assert await get_session_user(db, s1.session_id, user.id, now=NOW)
        assert await get_session_user(db, s2.session_id, user.id, now=NOW) is None
        assert await get_session_user(db, s3.session_id, user.id, now=NOW) is None
    assert (await _row(auth_db, s2.session_id)).revoked_reason == "password_changed"
    assert (await _row(auth_db, s1.session_id)).revoked_reason is None


async def test_revoke_all_without_keep_revokes_everything_and_counts_only_live_rows(
    auth_db, local_keys
):
    user, other = await seed_user(auth_db), await seed_user(auth_db)
    async with auth_db() as db:
        mine = [await create_session(db, user, ["pwd"], None, None, now=NOW) for _ in range(3)]
        theirs = await create_session(db, other, ["pwd"], None, None, now=NOW)
        await revoke_session(db, mine[0].session_id, "logout")
        await db.commit()
    first = await _row(auth_db, mine[0].session_id)

    async with auth_db() as db:
        assert await revoke_all_sessions(db, user.id, "admin_reset") == 2  # the third was done
        await db.commit()
        assert await revoke_all_sessions(db, user.id, "admin_reset") == 0  # idempotent

    kept = await _row(auth_db, mine[0].session_id)
    assert (kept.revoked_at, kept.revoked_reason) == (first.revoked_at, "logout")
    assert (await _row(auth_db, mine[1].session_id)).revoked_reason == "admin_reset"
    async with auth_db() as db:
        # Another account's session is not touched.
        assert await get_session_user(db, theirs.session_id, other.id, now=NOW)


async def test_revoke_all_does_not_commit(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    async with auth_db() as db:
        assert await revoke_all_sessions(db, user.id, "x") == 1
        await db.rollback()
    assert (await _row(auth_db, issued.session_id)).revoked_at is None


# --------------------------------------------------------------------------- get_session_user


async def test_get_session_user_returns_the_pair_in_one_query(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()

    async with auth_db() as db:
        statements = []

        def count(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        event.listen(db.bind.sync_engine, "before_cursor_execute", count)
        try:
            found = await get_session_user(db, issued.session_id, user.id, now=NOW)
        finally:
            event.remove(db.bind.sync_engine, "before_cursor_execute", count)

    assert len(statements) == 1, statements
    assert found is not None
    got_user, got_session = found
    assert isinstance(got_user, User) and got_user.id == user.id
    assert isinstance(got_session, AuthSession) and got_session.id == issued.session_id


async def test_get_session_user_checks_both_ids(auth_db, local_keys):
    user, other = await seed_user(auth_db), await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
        assert await get_session_user(db, issued.session_id, user.id, now=NOW)
        # A valid session id under another account's id, or a made-up one: nothing.
        assert await get_session_user(db, issued.session_id, other.id, now=NOW) is None
        assert await get_session_user(db, uuid4(), user.id, now=NOW) is None


async def test_get_session_user_expiry_boundaries(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    tick = timedelta(seconds=1)
    async with auth_db() as db:
        assert await get_session_user(db, issued.session_id, user.id, now=NOW + IDLE_TTL - tick)
        assert await get_session_user(db, issued.session_id, user.id, now=NOW + IDLE_TTL) is None

    # A session kept alive by rotation is bounded by its absolute expiry.
    await _keep_alive(auth_db, issued, 24)
    async with auth_db() as db:
        assert await get_session_user(db, issued.session_id, user.id, now=NOW + ABSOLUTE_TTL - tick)
        assert (
            await get_session_user(db, issued.session_id, user.id, now=NOW + ABSOLUTE_TTL) is None
        )


async def test_get_session_user_does_not_touch_the_session(auth_db, local_keys):
    # Resolving an access token is a read: it must not slide the inactivity window (only a
    # refresh does) nor write anything.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        await db.commit()
    before = await _row(auth_db, issued.session_id)
    async with auth_db() as db:
        await get_session_user(db, issued.session_id, user.id, now=NOW + timedelta(days=1))
        await db.commit()
    after = await _row(auth_db, issued.session_id)
    assert _aware(after.last_used_at) == _aware(before.last_used_at) == NOW
    assert _aware(after.idle_expires_at) == _aware(before.idle_expires_at)


# --------------------------------------------------------------------------- hygiene


async def test_no_token_or_hash_reaches_the_logs(auth_db, local_keys, caplog):
    # The application's own loggers only: aiosqlite (the test driver) logs bound parameters at
    # DEBUG, which the production driver and the engine options (parameters hidden) do not.
    caplog.set_level(logging.DEBUG, logger="app")
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        outcome = await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=1))
        await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=1, seconds=5))
        with pytest.raises(RefreshRejected):
            await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(hours=1))
        await db.commit()
    secrets_ = [
        issued.refresh_token,
        issued.access_token,
        outcome.refresh_token,
        hash_token(issued.refresh_token),
        hash_token(outcome.refresh_token),
    ]
    own = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app"))
    for secret in secrets_:
        assert secret not in own


async def test_dataclasses_do_not_print_their_tokens(auth_db, local_keys):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        issued = await create_session(db, user, ["pwd"], None, None, now=NOW)
        outcome = await rotate_refresh(db, issued.refresh_token, now=NOW + timedelta(minutes=1))
    for shown in (repr(issued), repr(outcome)):
        assert issued.refresh_token not in shown and issued.access_token not in shown
        assert outcome.refresh_token not in shown and outcome.access_token not in shown
