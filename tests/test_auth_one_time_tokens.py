"""
Tests of the one-time tokens, the MFA step tokens and the audit log
(`app/auth/one_time_tokens.py`, `app/auth/audit.py`).

Everything runs against the in-memory SQLite database of `tests/auth_support.py`, with a fixed
clock (`NOW`) passed as `now=`; nothing hosted is called, no wall clock is read. SQLite hands
datetimes back naive, so the assertions go through `_aware` rather than comparing raw columns.
"""

import asyncio
import json
import logging
import math
import re
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import asyncpg
from starlette.requests import Request

from app.auth.audit import record_event
from app.auth.one_time_tokens import (
    MAX_CHALLENGE_ATTEMPTS,
    TOKEN_TTL,
    InvalidOneTimeToken,
    consume_token,
    count_recent_tokens,
    issue_token,
    peek_token,
    reserve_attempt,
)
from app.auth.tokens import hash_token
from app.models.auth import AuthEvent, AuthEventType, AuthOneTimeToken, OneTimeTokenPurpose
from tests.auth_support import NOW, seed_user

INVITE = OneTimeTokenPurpose.INVITE
RESET = OneTimeTokenPurpose.PASSWORD_RESET
CHALLENGE = OneTimeTokenPurpose.MFA_CHALLENGE


def _aware(moment: datetime) -> datetime:
    """SQLite returns the naive wall clock of what was stored (UTC); Postgres an aware value."""
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


async def _row(auth_db, raw: str) -> AuthOneTimeToken:
    async with auth_db() as s:
        return (
            await s.execute(
                select(AuthOneTimeToken).where(AuthOneTimeToken.token_hash == hash_token(raw))
            )
        ).scalar_one()


async def _issue(auth_db, user_id, purpose, now=NOW, created_by=None) -> str:
    async with auth_db() as db:
        raw = await issue_token(db, user_id, purpose, created_by=created_by, now=now)
        await db.commit()
    return raw


async def _consume(auth_db, raw, purpose, now=NOW) -> AuthOneTimeToken:
    async with auth_db() as db:
        token = await consume_token(db, raw, purpose, now=now)
        await db.commit()
    return token


# --------------------------------------------------------------------------- constants


def test_ttl_constants():
    assert TOKEN_TTL == {
        OneTimeTokenPurpose.INVITE: timedelta(days=7),
        OneTimeTokenPurpose.PASSWORD_RESET: timedelta(hours=1),
        OneTimeTokenPurpose.EMAIL_VERIFY: timedelta(hours=24),
        OneTimeTokenPurpose.MFA_CHALLENGE: timedelta(minutes=5),
        OneTimeTokenPurpose.MFA_ENROLL: timedelta(minutes=15),
    }
    assert MAX_CHALLENGE_ATTEMPTS == 5


def test_every_purpose_has_a_ttl():
    # A purpose added later without a lifetime would be a KeyError at issue time.
    assert set(TOKEN_TTL) == set(OneTimeTokenPurpose)


# --------------------------------------------------------------------------- issue_token


async def test_issue_stores_the_hash_only_and_returns_the_raw_token(auth_db):
    user, admin = await seed_user(auth_db), await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, INVITE, created_by=admin.id)

    assert isinstance(raw, str) and len(raw) >= 40
    row = await _row(auth_db, raw)
    assert row.token_hash == hash_token(raw)
    assert raw not in (row.token_hash, str(row.id))
    assert row.user_id == user.id
    assert row.purpose is INVITE
    assert row.created_by == admin.id
    assert row.used_at is None
    assert row.attempts == 0
    assert _aware(row.created_at) == NOW
    assert _aware(row.expires_at) == NOW + timedelta(days=7)


async def test_issue_without_created_by_leaves_it_null(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, RESET)
    assert (await _row(auth_db, raw)).created_by is None


async def test_every_token_is_distinct(auth_db):
    user = await seed_user(auth_db)
    raws = {await _issue(auth_db, user.id, INVITE) for _ in range(5)}
    assert len(raws) == 5


@pytest.mark.parametrize("purpose", list(OneTimeTokenPurpose))
async def test_each_purpose_lives_exactly_its_ttl(auth_db, purpose):
    user = await seed_user(auth_db)
    ttl = TOKEN_TTL[purpose]
    raw = await _issue(auth_db, user.id, purpose)
    assert _aware((await _row(auth_db, raw)).expires_at) == NOW + ttl

    async with auth_db() as db:
        await peek_token(db, raw, purpose, now=NOW + ttl - timedelta(seconds=1))
        for late in (ttl, ttl + timedelta(seconds=1)):
            with pytest.raises(InvalidOneTimeToken) as excinfo:
                await peek_token(db, raw, purpose, now=NOW + late)
            assert excinfo.value.reason == "expired"


async def test_issue_does_not_commit(auth_db):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        await issue_token(db, user.id, INVITE, now=NOW)
        await db.rollback()  # the caller's transaction fails: no token survives
    async with auth_db() as s:
        assert (await s.execute(select(AuthOneTimeToken))).first() is None


async def test_issue_flushes_so_the_row_is_visible_in_the_same_transaction(auth_db):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        raw = await issue_token(db, user.id, INVITE, now=NOW)
        # No commit: the same transaction can already consume it.
        token = await consume_token(db, raw, INVITE, now=NOW)
        assert token.user_id == user.id


async def test_a_new_token_invalidates_the_previous_one_of_the_same_purpose(auth_db):
    user = await seed_user(auth_db)
    first = await _issue(auth_db, user.id, INVITE)
    second = await _issue(auth_db, user.id, INVITE, now=NOW + timedelta(minutes=1))

    async with auth_db() as db:
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await consume_token(db, first, INVITE, now=NOW + timedelta(minutes=2))
        assert excinfo.value.reason == "used"
    # The earlier one was marked used at the moment the new one was issued; the new one works.
    assert _aware((await _row(auth_db, first)).used_at) == NOW + timedelta(minutes=1)
    token = await _consume(auth_db, second, INVITE, now=NOW + timedelta(minutes=2))
    assert token.user_id == user.id


async def test_a_new_token_only_touches_the_same_purpose_and_the_same_user(auth_db):
    user, other = await seed_user(auth_db), await seed_user(auth_db)
    other_purpose = await _issue(auth_db, user.id, RESET)
    other_user = await _issue(auth_db, other.id, INVITE)
    await _issue(auth_db, user.id, INVITE)  # invalidates nothing of the two above

    assert (await _row(auth_db, other_purpose)).used_at is None
    assert (await _row(auth_db, other_user)).used_at is None
    await _consume(auth_db, other_purpose, RESET)
    await _consume(auth_db, other_user, INVITE)


async def test_a_new_token_keeps_the_first_use_of_an_already_used_one(auth_db):
    user = await seed_user(auth_db)
    first = await _issue(auth_db, user.id, INVITE)
    await _consume(auth_db, first, INVITE, now=NOW + timedelta(minutes=5))
    await _issue(auth_db, user.id, INVITE, now=NOW + timedelta(hours=1))
    assert _aware((await _row(auth_db, first)).used_at) == NOW + timedelta(minutes=5)


async def test_issue_refreshes_a_token_object_the_session_already_holds(auth_db):
    # The earlier token is in this session's identity map as "unused"; the bulk invalidation must
    # not leave it looking usable to the code that peeked at it.
    user = await seed_user(auth_db)
    first = await _issue(auth_db, user.id, INVITE)
    async with auth_db() as db:
        held = await peek_token(db, first, INVITE, now=NOW)
        assert held.used_at is None
        await issue_token(db, user.id, INVITE, now=NOW + timedelta(minutes=1))
        assert _aware(held.used_at) == NOW + timedelta(minutes=1)


# --------------------------------------------------------------------------- consume_token


async def test_single_use_and_expiry(auth_db):
    user = await seed_user(auth_db)

    # single use
    raw = await _issue(auth_db, user.id, INVITE)
    token = await _consume(auth_db, raw, INVITE, now=NOW + timedelta(minutes=1))
    assert token.user_id == user.id and token.purpose is INVITE
    assert _aware(token.used_at) == NOW + timedelta(minutes=1)
    async with auth_db() as db:
        with pytest.raises(InvalidOneTimeToken):
            await consume_token(db, raw, INVITE, now=NOW + timedelta(minutes=2))

    # expiry: a fresh invitation is refused after 7 days and 1 second ...
    raw = await _issue(auth_db, user.id, INVITE)
    async with auth_db() as db:
        with pytest.raises(InvalidOneTimeToken):
            await consume_token(db, raw, INVITE, now=NOW + timedelta(days=7, seconds=1))
        # ... and still accepted the second before.
        token = await consume_token(db, raw, INVITE, now=NOW + timedelta(days=7, seconds=-1))
        assert token.used_at is not None

    # another purpose never matches, and does not burn the token
    raw = await _issue(auth_db, user.id, INVITE)
    async with auth_db() as db:
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await consume_token(db, raw, RESET, now=NOW)
        assert excinfo.value.reason == "wrong_purpose"
        assert (await consume_token(db, raw, INVITE, now=NOW)).user_id == user.id


async def test_a_token_is_valid_strictly_before_its_expiry(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, RESET)
    async with auth_db() as db:
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await consume_token(db, raw, RESET, now=NOW + TOKEN_TTL[RESET])
        assert excinfo.value.reason == "expired"
    # The refused attempt left the token untouched.
    assert (await _row(auth_db, raw)).used_at is None


async def test_unknown_and_malformed_tokens_are_refused(auth_db):
    await seed_user(auth_db)
    async with auth_db() as db:
        for raw in ("", "not-a-token", "A" * 5000, "\ud800 lone surrogate"):
            with pytest.raises(InvalidOneTimeToken) as excinfo:
                await consume_token(db, raw, INVITE, now=NOW)
            assert excinfo.value.reason == "unknown"
            with pytest.raises(InvalidOneTimeToken):
                await peek_token(db, raw, INVITE, now=NOW)


async def test_the_refusal_never_says_why(auth_db):
    # The message (what ends up in a log line or a response) is one and the same whether the token
    # never existed, belongs to another purpose, expired or was used. The cause is a machine
    # attribute for logs and tests only.
    user = await seed_user(auth_db)
    live = await _issue(auth_db, user.id, INVITE)
    used = await _issue(auth_db, user.id, RESET)
    await _consume(auth_db, used, RESET)

    cases = {
        "unknown": (hash_token("nobody"), INVITE, NOW),
        "wrong_purpose": (live, RESET, NOW),
        "expired": (live, INVITE, NOW + timedelta(days=8)),
        "used": (used, RESET, NOW),
    }
    messages, reasons = set(), {}
    async with auth_db() as db:
        for expected, (raw, purpose, now) in cases.items():
            with pytest.raises(InvalidOneTimeToken) as excinfo:
                await consume_token(db, raw, purpose, now=now)
            messages.add(str(excinfo.value))
            reasons[expected] = excinfo.value.reason
            for secret in (raw, hash_token(raw)):
                assert secret not in str(excinfo.value) and secret not in repr(excinfo.value)
    assert len(messages) == 1
    assert reasons == {name: name for name in cases}
    # The reason is a plain attribute, not part of the exception's arguments.
    assert excinfo.value.args == (next(iter(messages)),)


async def test_consume_is_one_conditional_update(auth_db, monkeypatch):
    # The race-safety is in the statement, not in a read followed by a write: one UPDATE that
    # only matches a token which is still unused, unexpired and of the right purpose.
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, INVITE)
    async with auth_db() as db:
        seen = []
        real_execute = db.execute

        async def spy(statement, *args, **kwargs):
            seen.append(statement)
            return await real_execute(statement, *args, **kwargs)

        monkeypatch.setattr(db, "execute", spy)
        await consume_token(db, raw, INVITE, now=NOW)

    updates = [s for s in seen if str(s).lstrip().startswith("UPDATE")]
    assert len(updates) == 1
    sql = str(updates[0].compile(dialect=postgresql.dialect()))
    assert "UPDATE auth_one_time_tokens SET used_at=" in sql
    for clause in (
        "auth_one_time_tokens.token_hash =",
        "auth_one_time_tokens.purpose =",
        "auth_one_time_tokens.used_at IS NULL",
        "auth_one_time_tokens.expires_at >",
    ):
        assert clause in sql
    strict = str(updates[0].compile(dialect=asyncpg.dialect()))
    assert re.search(r"expires_at > \$\d+::TIMESTAMP WITH TIME ZONE", strict)


async def test_a_stale_view_cannot_consume_a_token_another_transaction_consumed(auth_db):
    # Two requests hold the same unused token. The first consumes and commits; the second, which
    # looked at the token before that (its object still says "unused"), must be refused.
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, INVITE)
    async with auth_db() as slow, auth_db() as fast:
        stale = await peek_token(slow, raw, INVITE, now=NOW)
        assert stale.used_at is None

        await consume_token(fast, raw, INVITE, now=NOW)
        await fast.commit()

        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await consume_token(slow, raw, INVITE, now=NOW)
        assert excinfo.value.reason == "used"


async def test_of_many_concurrent_consumers_exactly_one_wins(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, INVITE)

    async def attempt():
        async with auth_db() as db:
            try:
                token = await consume_token(db, raw, INVITE, now=NOW)
            except InvalidOneTimeToken:
                return None
            await db.commit()
            return token

    results = await asyncio.gather(*(attempt() for _ in range(8)))
    assert len([r for r in results if r is not None]) == 1


async def test_consume_returns_the_row_as_stored_and_refreshes_a_held_object(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, INVITE)
    async with auth_db() as db:
        held = await peek_token(db, raw, INVITE, now=NOW)
        assert held.used_at is None
        returned = await consume_token(db, raw, INVITE, now=NOW + timedelta(seconds=3))
        assert returned is held  # one identity per row in a session
        assert _aware(held.used_at) == NOW + timedelta(seconds=3)


async def test_consume_does_not_commit(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, INVITE)
    async with auth_db() as db:
        await consume_token(db, raw, INVITE, now=NOW)
        await db.rollback()  # the caller's transaction fails: the token is still unspent
    assert (await _row(auth_db, raw)).used_at is None
    await _consume(auth_db, raw, INVITE)


# --------------------------------------------------------------------------- peek_token


async def test_peek_checks_like_consume_but_does_not_consume(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        for _ in range(3):
            token = await peek_token(db, raw, CHALLENGE, now=NOW + timedelta(minutes=1))
            assert token.user_id == user.id and token.used_at is None
        await db.commit()
    assert (await _row(auth_db, raw)).used_at is None

    async with auth_db() as db:
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await peek_token(db, raw, INVITE, now=NOW)
        assert excinfo.value.reason == "wrong_purpose"
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await peek_token(db, raw, CHALLENGE, now=NOW + timedelta(minutes=5))
        assert excinfo.value.reason == "expired"
    await _consume(auth_db, raw, CHALLENGE, now=NOW + timedelta(minutes=1))
    async with auth_db() as db:
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await peek_token(db, raw, CHALLENGE, now=NOW + timedelta(minutes=2))
        assert excinfo.value.reason == "used"


async def test_peek_sees_what_another_transaction_committed_not_what_it_loaded(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as slow, auth_db() as fast:
        await peek_token(slow, raw, CHALLENGE, now=NOW)
        await consume_token(fast, raw, CHALLENGE, now=NOW)
        await fast.commit()
        with pytest.raises(InvalidOneTimeToken):
            await peek_token(slow, raw, CHALLENGE, now=NOW)


# --------------------------------------------------------------------------- reserve_attempt


async def _attempts(auth_db, raw) -> tuple[int, bool]:
    """What is stored for a token: how many attempts are taken, and whether it is spent."""
    row = await _row(auth_db, raw)
    return row.attempts, row.used_at is not None


async def test_challenge_attempts_exhaust_after_five(auth_db):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        raw = await issue_token(db, user.id, CHALLENGE, now=NOW)
        await db.commit()

        left = []
        for _ in range(5):
            token = await reserve_attempt(db, raw, CHALLENGE, now=NOW)
            left.append(MAX_CHALLENGE_ATTEMPTS - token.attempts)
            await db.commit()  # what the login flow does before it looks at the code
        assert left == [4, 3, 2, 1, 0]

        for _ in range(2):
            with pytest.raises(InvalidOneTimeToken) as excinfo:
                await reserve_attempt(db, raw, CHALLENGE, now=NOW)
            assert excinfo.value.reason == "exhausted"
    # The refused ones counted nothing; the token is not spent, it just cannot be tried again.
    assert await _attempts(auth_db, raw) == (5, False)


async def test_a_correct_code_on_the_fifth_reservation_still_consumes(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        for _ in range(MAX_CHALLENGE_ATTEMPTS):
            token = await reserve_attempt(db, raw, CHALLENGE, now=NOW)
            await db.commit()
        assert token.attempts == MAX_CHALLENGE_ATTEMPTS  # the last attempt, and the right code
        done = await consume_token(db, raw, CHALLENGE, now=NOW)
        await db.commit()
        assert done.user_id == user.id and done.used_at is not None

        # Once consumed, nothing can reserve on it any more, and it says why.
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await reserve_attempt(db, raw, CHALLENGE, now=NOW)
        assert excinfo.value.reason == "used"
    assert await _attempts(auth_db, raw) == (5, True)


async def test_reserve_counts_the_attempt_before_the_code_is_checked(auth_db):
    # The returned token already includes this attempt, and nothing is spent: it is the caller's
    # `consume_token` that closes the challenge.
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        token = await reserve_attempt(db, raw, CHALLENGE, now=NOW)
        assert token.attempts == 1
        assert token.used_at is None
        assert token.user_id == user.id and token.purpose is CHALLENGE
        await db.commit()
    assert await _attempts(auth_db, raw) == (1, False)


async def test_a_reserved_attempt_is_not_handed_back_by_the_callers_rollback(auth_db):
    # The login flow commits the reservation, then answers a wrong code with a 401 whose request
    # rolls back: the attempt must stay spent.
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        await reserve_attempt(db, raw, CHALLENGE, now=NOW)
        await db.commit()
        await db.rollback()
    assert await _attempts(auth_db, raw) == (1, False)


async def test_reserve_does_not_commit(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        await reserve_attempt(db, raw, CHALLENGE, now=NOW)
        await db.rollback()  # nobody committed: the attempt was never taken
    assert await _attempts(auth_db, raw) == (0, False)


async def test_reserve_refuses_what_peek_and_consume_refuse_and_writes_nothing(auth_db):
    user = await seed_user(auth_db)
    live = await _issue(auth_db, user.id, CHALLENGE)
    used = await _issue(auth_db, user.id, OneTimeTokenPurpose.MFA_ENROLL)
    await _consume(auth_db, used, OneTimeTokenPurpose.MFA_ENROLL)
    spent = await _issue(auth_db, user.id, RESET)
    async with auth_db() as db:
        for _ in range(MAX_CHALLENGE_ATTEMPTS):
            await reserve_attempt(db, live, CHALLENGE, now=NOW)
        await db.commit()

    cases = {
        "unknown": (hash_token("nobody"), CHALLENGE, NOW),
        "wrong_purpose": (spent, CHALLENGE, NOW),
        "expired": (spent, RESET, NOW + TOKEN_TTL[RESET]),
        "used": (used, OneTimeTokenPurpose.MFA_ENROLL, NOW),
        "exhausted": (live, CHALLENGE, NOW),
    }
    messages = set()
    async with auth_db() as db:
        for expected, (raw, purpose, now) in cases.items():
            with pytest.raises(InvalidOneTimeToken) as excinfo:
                await reserve_attempt(db, raw, purpose, now=now)
            assert excinfo.value.reason == expected
            messages.add(str(excinfo.value))
            for secret in (raw, hash_token(raw)):
                assert secret not in str(excinfo.value) and secret not in repr(excinfo.value)
        await db.commit()
    assert len(messages) == 1  # the same words whatever the cause
    # Not one of the refusals touched a counter or spent a token.
    assert await _attempts(auth_db, spent) == (0, False)
    assert await _attempts(auth_db, used) == (0, True)
    assert await _attempts(auth_db, live) == (5, False)


async def test_a_wrong_purpose_does_not_burn_an_attempt(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        for _ in range(3):
            with pytest.raises(InvalidOneTimeToken) as excinfo:
                await reserve_attempt(db, raw, OneTimeTokenPurpose.MFA_ENROLL, now=NOW)
            assert excinfo.value.reason == "wrong_purpose"
        assert (await reserve_attempt(db, raw, CHALLENGE, now=NOW)).attempts == 1


async def test_an_expired_challenge_cannot_be_reserved_on_at_its_last_instant(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    ttl = TOKEN_TTL[CHALLENGE]
    async with auth_db() as db:
        assert (
            await reserve_attempt(db, raw, CHALLENGE, now=NOW + ttl - timedelta(seconds=1))
        ).attempts == 1
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await reserve_attempt(db, raw, CHALLENGE, now=NOW + ttl)
        assert excinfo.value.reason == "expired"


async def test_unknown_and_malformed_challenges_cannot_be_reserved_on(auth_db):
    await seed_user(auth_db)
    async with auth_db() as db:
        for raw in ("", "not-a-token", "A" * 5000, "\ud800 lone surrogate"):
            with pytest.raises(InvalidOneTimeToken) as excinfo:
                await reserve_attempt(db, raw, CHALLENGE, now=NOW)
            assert excinfo.value.reason == "unknown"


async def test_attempts_are_handed_out_one_by_one_when_many_arrive_before_any_check(auth_db):
    # The case the counter exists for. Seven guesses are made against one challenge, and none of
    # them has been verified (so no failure has been recorded anywhere) when the next one starts.
    # Counting failures after the check would let all seven through; reserving first lets five.
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)

    outcomes = []
    async with auth_db() as db:
        for _ in range(7):
            try:
                token = await reserve_attempt(db, raw, CHALLENGE, now=NOW)
                outcomes.append(token.attempts)
            except InvalidOneTimeToken as exc:
                outcomes.append(exc.reason)
        await db.commit()
    assert outcomes == [1, 2, 3, 4, 5, "exhausted", "exhausted"]
    assert await _attempts(auth_db, raw) == (5, False)


async def test_of_many_concurrent_reservations_exactly_five_succeed(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)

    async def attempt():
        async with auth_db() as db:
            try:
                await reserve_attempt(db, raw, CHALLENGE, now=NOW)
            except InvalidOneTimeToken as exc:
                return exc.reason
            await db.commit()
            return "reserved"

    results = await asyncio.gather(*(attempt() for _ in range(7)))
    assert sorted(results) == ["exhausted"] * 2 + ["reserved"] * 5
    assert await _attempts(auth_db, raw) == (5, False)


async def test_reservations_count_against_the_stored_value_whichever_session_makes_them(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as a, auth_db() as b:
        held = await peek_token(b, raw, CHALLENGE, now=NOW)  # b's copy says attempts == 0
        assert held.attempts == 0
        for expected in (1, 2, 3):
            assert (await reserve_attempt(a, raw, CHALLENGE, now=NOW)).attempts == expected
        await a.commit()
        # b reserves against what is stored, and its held object follows.
        assert (await reserve_attempt(b, raw, CHALLENGE, now=NOW)).attempts == 4
        assert held.attempts == 4
        await b.commit()
    assert await _attempts(auth_db, raw) == (4, False)


async def test_peek_counts_no_attempt(auth_db):
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        for _ in range(10):
            await peek_token(db, raw, CHALLENGE, now=NOW)
        assert (await reserve_attempt(db, raw, CHALLENGE, now=NOW)).attempts == 1


async def test_reserve_is_one_conditional_increment_statement(auth_db, monkeypatch):
    # The bound is in the statement: one UPDATE that increments in SQL and only matches a token
    # that is unused, unexpired, of the right purpose and below the maximum.
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        seen = []
        real_execute = db.execute

        async def spy(statement, *args, **kwargs):
            seen.append(statement)
            return await real_execute(statement, *args, **kwargs)

        monkeypatch.setattr(db, "execute", spy)
        await reserve_attempt(db, raw, CHALLENGE, now=NOW)

    updates = [s for s in seen if str(s).lstrip().startswith("UPDATE")]
    assert len(updates) == 1
    sql = str(updates[0].compile(dialect=postgresql.dialect()))
    assert "UPDATE auth_one_time_tokens SET attempts=(auth_one_time_tokens.attempts + " in sql
    for clause in (
        "auth_one_time_tokens.token_hash =",
        "auth_one_time_tokens.purpose =",
        "auth_one_time_tokens.used_at IS NULL",
        "auth_one_time_tokens.expires_at >",
        "auth_one_time_tokens.attempts <",
    ):
        assert clause in sql
    # The instant is declared as timestamptz: asyncpg, unlike SQLite, refuses an aware datetime for
    # a plain timestamp, which only a real Postgres would reveal.
    strict = str(updates[0].compile(dialect=asyncpg.dialect()))
    assert re.search(r"expires_at > \$\d+::TIMESTAMP WITH TIME ZONE", strict)


async def test_reserved_attempts_are_not_shared_between_tokens(auth_db):
    user = await seed_user(auth_db)
    first = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        for _ in range(MAX_CHALLENGE_ATTEMPTS):
            await reserve_attempt(db, first, CHALLENGE, now=NOW)
        # A new challenge starts from zero (the account-level counter is the caller's business).
        second = await issue_token(db, user.id, CHALLENGE, now=NOW)
        with pytest.raises(InvalidOneTimeToken) as excinfo:
            await reserve_attempt(db, first, CHALLENGE, now=NOW)
        assert excinfo.value.reason == "used"  # superseded by the new one
        assert (await reserve_attempt(db, second, CHALLENGE, now=NOW)).attempts == 1


async def test_an_mfa_challenge_end_to_end(auth_db):
    # What the login flow does: issue at the password step; at the code step reserve an attempt,
    # commit, verify; consume once the code is right.
    user = await seed_user(auth_db)
    async with auth_db() as db:
        raw = await issue_token(db, user.id, CHALLENGE, now=NOW)
        await db.commit()

        token = await reserve_attempt(db, raw, CHALLENGE, now=NOW + timedelta(seconds=10))
        await db.commit()
        assert MAX_CHALLENGE_ATTEMPTS - token.attempts == 4  # a wrong code: nothing more to record

        token = await reserve_attempt(db, raw, CHALLENGE, now=NOW + timedelta(seconds=20))
        await db.commit()
        assert MAX_CHALLENGE_ATTEMPTS - token.attempts == 3
        done = await consume_token(db, raw, CHALLENGE, now=NOW + timedelta(seconds=20))
        await db.commit()
        assert done.user_id == user.id
        for check in (reserve_attempt, peek_token):
            with pytest.raises(InvalidOneTimeToken):
                await check(db, raw, CHALLENGE, now=NOW + timedelta(seconds=30))


# --------------------------------------------------------------------------- count_recent_tokens


async def test_count_recent_tokens(auth_db):
    user, other = await seed_user(auth_db), await seed_user(auth_db)
    for minutes in (0, 10, 20, 30):  # four resets: three of them superseded, which still counts
        await _issue(auth_db, user.id, RESET, now=NOW + timedelta(minutes=minutes))
    await _issue(auth_db, user.id, INVITE, now=NOW + timedelta(minutes=20))
    await _issue(auth_db, other.id, RESET, now=NOW + timedelta(minutes=20))

    async with auth_db() as db:

        async def count(since, uid=user.id, p=RESET):
            return await count_recent_tokens(db, uid, p, since)

        assert await count(NOW - timedelta(hours=1)) == 4
        assert await count(NOW + timedelta(minutes=10)) == 3  # `since` itself is inside
        assert await count(NOW + timedelta(minutes=10, seconds=1)) == 2
        assert await count(NOW + timedelta(minutes=31)) == 0
        assert await count(NOW - timedelta(hours=1), p=INVITE) == 1
        assert await count(NOW - timedelta(hours=1), uid=other.id) == 1
        assert await count(NOW - timedelta(hours=1), uid=uuid4()) == 0
        # A naive `since` is read as UTC, like every other instant here.
        assert await count((NOW - timedelta(hours=1)).replace(tzinfo=None)) == 4


async def test_count_recent_tokens_includes_spent_ones(auth_db):
    # The rate limit counts what was asked for, not what is still usable.
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, RESET)
    await _consume(auth_db, raw, RESET)
    async with auth_db() as db:
        assert await count_recent_tokens(db, user.id, RESET, NOW - timedelta(minutes=1)) == 1


# --------------------------------------------------------------------------- hygiene


async def test_no_token_or_hash_reaches_the_logs(auth_db, caplog):
    # The application's own loggers only: aiosqlite (the test driver) logs bound parameters at
    # DEBUG, which the production driver and the engine options (parameters hidden) do not.
    caplog.set_level(logging.DEBUG, logger="app")
    user = await seed_user(auth_db)
    raw = await _issue(auth_db, user.id, CHALLENGE)
    async with auth_db() as db:
        await peek_token(db, raw, CHALLENGE, now=NOW)
        await reserve_attempt(db, raw, CHALLENGE, now=NOW)
        with pytest.raises(InvalidOneTimeToken):
            await reserve_attempt(db, raw, INVITE, now=NOW)
        with pytest.raises(InvalidOneTimeToken):
            await consume_token(db, raw, INVITE, now=NOW)
        await consume_token(db, raw, CHALLENGE, now=NOW)
        with pytest.raises(InvalidOneTimeToken):
            await consume_token(db, raw, CHALLENGE, now=NOW)
        await db.commit()
    own = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app"))
    for secret in (raw, hash_token(raw)):
        assert secret not in own


# =========================================================================== audit log


def _request(user_agent=None, client=("203.0.113.7", 54321)) -> Request:
    headers = [(b"user-agent", user_agent.encode())] if user_agent is not None else []
    return Request({"type": "http", "headers": headers, "client": client})


async def _events(auth_db) -> list[AuthEvent]:
    async with auth_db() as s:
        return list((await s.execute(select(AuthEvent).order_by(AuthEvent.id))).scalars())


async def test_audit_strips_secrets(auth_db):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        await record_event(
            db, AuthEventType.LOGIN_FAILED, user.id, password="x", code="1", reason="bad"
        )
        assert (await db.execute(select(AuthEvent))).scalar_one().details == {"reason": "bad"}


async def test_audit_records_the_event(auth_db):
    user = await seed_user(auth_db)
    async with auth_db() as db:
        assert (
            await record_event(
                db,
                AuthEventType.SESSION_REVOKED,
                user.id,
                _request(user_agent="Mozilla/5.0 (test)"),
                session_id="s-1",
                reason="logout",
            )
            is None
        )
        await db.commit()

    (event_row,) = await _events(auth_db)
    assert event_row.event_type == "session_revoked" == AuthEventType.SESSION_REVOKED.value
    assert event_row.user_id == user.id
    assert event_row.ip == "203.0.113.7"
    assert event_row.user_agent == "Mozilla/5.0 (test)"
    assert event_row.details == {"session_id": "s-1", "reason": "logout"}
    # An instant of the application's clock, not SQLite's second-resolution CURRENT_TIMESTAMP.
    created = _aware(event_row.created_at)
    assert abs(datetime.now(UTC) - created) < timedelta(minutes=1)


async def test_audit_without_user_request_or_details(auth_db):
    async with auth_db() as db:
        await record_event(db, AuthEventType.LOGIN_FAILED)
        await db.commit()
    (event_row,) = await _events(auth_db)
    assert event_row.user_id is None
    assert event_row.ip is None
    assert event_row.user_agent is None
    assert event_row.details == {}


async def test_audit_tolerates_a_request_without_client_or_user_agent(auth_db):
    async with auth_db() as db:
        await record_event(db, AuthEventType.LOGIN_FAILED, request=_request(client=None))
        await record_event(db, AuthEventType.LOGIN_FAILED, request=_request())
        await db.commit()
    first, second = await _events(auth_db)
    assert (first.ip, first.user_agent) == (None, None)
    assert (second.ip, second.user_agent) == ("203.0.113.7", None)


async def test_audit_clips_what_the_columns_cannot_hold(auth_db):
    # Both come straight from the request, and Postgres refuses a longer value: a hostile client
    # must not turn a failed login into a 500.
    async with auth_db() as db:
        await record_event(
            db,
            AuthEventType.LOGIN_FAILED,
            request=_request(user_agent="U" * 2000, client=("2001:db8::" + "f" * 100, 1)),
        )
        await db.commit()
    (event_row,) = await _events(auth_db)
    assert event_row.user_agent == "U" * 512
    assert event_row.ip == ("2001:db8::" + "f" * 100)[:64]


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "code",
        "token",
        "secret",
        "recovery_code",
        "new_password",
        "old_password",
        "raw_token",
        "refresh_token",
        "mfa_code",
        "client_secret",
        "totp_secret",
        "Password",
        "NEW_PASSWORD",
        "RecoveryCode",
        "status_code",  # over-stripped on purpose: see the module docstring
    ],
)
async def test_audit_drops_every_key_that_names_a_secret(auth_db, key):
    async with auth_db() as db:
        await record_event(db, AuthEventType.LOGIN_FAILED, reason="bad", **{key: "hunter2"})
        await db.commit()
    (event_row,) = await _events(auth_db)
    assert event_row.details == {"reason": "bad"}


async def test_audit_keeps_the_keys_that_name_nothing_secret(auth_db):
    kept = {"reason": "bad", "attempts_left": 3, "email_domain": "example.test", "method": "totp"}
    async with auth_db() as db:
        await record_event(db, AuthEventType.MFA_FAILED, **kept)
        await db.commit()
    (event_row,) = await _events(auth_db)
    assert event_row.details == kept


async def test_audit_strips_secrets_nested_in_the_details_too(auth_db):
    async with auth_db() as db:
        await record_event(
            db,
            AuthEventType.PASSWORD_CHANGED,
            changes={"password": "x", "field": "y", "inner": [{"secret": "z", "ok": 1}]},
            reason="self-service",
        )
        await db.commit()
    (event_row,) = await _events(auth_db)
    assert event_row.details == {
        "changes": {"field": "y", "inner": [{"ok": 1}]},
        "reason": "self-service",
    }


class _Colour(Enum):
    RED = "red"


class _Level(IntEnum):
    HIGH = 3


class _NoStr:
    def __str__(self):
        raise RuntimeError("no string for you")


async def test_audit_coerces_values_the_json_column_cannot_store(auth_db):
    # Whatever the caller passes, writing the audit line must not raise: it would otherwise take
    # the login (or the password reset) down with it.
    some_id = UUID("12345678-1234-5678-1234-567812345678")
    loop_ = {"name": "loop"}
    loop_["self"] = loop_
    deep: dict = {}
    cursor = deep
    for _ in range(40):
        cursor["next"] = {}
        cursor = cursor["next"]
    cursor["password"] = "never stored"

    async with auth_db() as db:
        await record_event(
            db,
            AuthEventType.INVITE_SENT,
            invited=some_id,
            at=NOW,
            on=date(2026, 11, 24),
            purpose=OneTimeTokenPurpose.INVITE,
            colour=_Colour.RED,
            level=_Level.HIGH,
            event=AuthEventType.LOGIN_FAILED,
            amount=Decimal("1.50"),
            members={"b"},
            pair=(some_id, 2),
            nested={"ids": [some_id], "when": NOW},
            keys={some_id: 1, 7: "seven", None: "none"},
            nan=float("nan"),
            inf=math.inf,
            ratio=0.5,
            flag=True,
            nothing=None,
            blob=b"\x00\x01",
            unprintable=_NoStr(),
            circular=loop_,
            deep=deep,
        )
        await db.commit()

    (event_row,) = await _events(auth_db)
    d = event_row.details
    assert d["invited"] == str(some_id)
    assert d["at"] == NOW.isoformat()
    assert d["on"] == "2026-11-24"
    assert d["purpose"] == "invite"
    assert d["colour"] == "red"
    assert d["level"] == 3
    assert d["event"] == "login_failed"
    assert d["amount"] == "1.50"
    assert d["members"] == ["b"]
    assert d["pair"] == [str(some_id), 2]
    assert d["nested"] == {"ids": [str(some_id)], "when": NOW.isoformat()}
    assert d["keys"] == {str(some_id): 1, "7": "seven", "None": "none"}
    assert d["nan"] == "nan" and d["inf"] == "inf"
    assert d["ratio"] == 0.5 and d["flag"] is True and d["nothing"] is None
    assert isinstance(d["blob"], str)
    assert d["unprintable"] == "<_NoStr>"
    assert "never stored" not in str(d)  # the secret under 40 levels of nesting did not leak
    assert d["circular"]["name"] == "loop"
    # It is all plain JSON now: the row round-trips.
    assert json.loads(json.dumps(d)) == d


async def test_audit_does_not_commit(auth_db):
    async with auth_db() as db:
        await record_event(db, AuthEventType.LOGIN_FAILED, reason="bad")
        await db.rollback()  # the caller's transaction fails: no event survives
    assert await _events(auth_db) == []


async def test_audit_flushes_so_the_event_is_visible_in_the_same_transaction(auth_db):
    async with auth_db() as db:
        before = []
        event.listen(db.bind.sync_engine, "before_cursor_execute", lambda *a: before.append(a[2]))
        await record_event(db, AuthEventType.LOGIN_FAILED, reason="bad")
        assert any("INSERT INTO auth_events" in sql for sql in before)
        await db.rollback()


async def test_audit_events_keep_their_order(auth_db):
    async with auth_db() as db:
        for event_type in (
            AuthEventType.LOGIN_FAILED,
            AuthEventType.LOGIN_SUCCEEDED,
            AuthEventType.SESSION_REVOKED,
        ):
            await record_event(db, event_type)
        await db.commit()
    assert [e.event_type for e in await _events(auth_db)] == [
        "login_failed",
        "login_succeeded",
        "session_revoked",
    ]
