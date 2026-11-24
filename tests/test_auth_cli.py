"""
The operator's command (`python -m app.auth.cli`) and the daily retention purge.

What is pinned down:
- `generate-keys` prints the two settings, and nothing else, on standard output; the keys it
  prints load back through the same code that reads the settings; it writes no file and logs
  nothing;
- `reset-mfa` and `unlock` act on the account through the shared admin actions (no logic of
  their own), find it ignoring case and surrounding spaces, refuse an unknown or shared address
  with exit code 2 and no change, work whatever `AUTH_MODE` says (break-glass has to work during
  the transition too), and never print a secret;
- `reset-mfa` records `BREAK_GLASS`, and its notification is best effort: the reset stands when
  the e-mail cannot leave;
- the purge keeps what the retention rules keep, deletes what they delete, at the boundary
  (365 days for events, 7 for one-time tokens, 30 for sessions), and is scheduled daily at 03:15.

Everything runs against an in-memory SQLite database; no address is real (`@example.test`).
"""

import logging
from datetime import UTC, datetime, timedelta
from io import StringIO
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.auth import cli
from app.auth.keys import load_signing_keys, reset_key_cache
from app.auth.mfa import encrypt_secret, load_cipher, reset_encryption_cache
from app.core.config import settings
from app.models.auth import (
    AuthEvent,
    AuthEventType,
    AuthOneTimeToken,
    AuthRecoveryCode,
    AuthSession,
    OneTimeTokenPurpose,
)
from app.models.models import User
from app.worker.tasks import auth_tasks
from app.worker.tasks.auth_tasks import (
    EVENT_RETENTION,
    SESSION_RETENTION,
    TOKEN_RETENTION,
    purge_expired_auth_records,
)
from tests.auth_support import NOW, seed_user

SECRET_BLOB = "gAAAA-this-is-the-encrypted-totp-secret"
MODES = pytest.mark.parametrize("mode", ["supabase", "local"])


# ── generate-keys ────────────────────────────────────────────────────────────


def _generated(capsys) -> dict[str, str]:
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("=", 1)[0] for line in lines] == [
        "AUTH_JWT_PRIVATE_KEYS",
        "AUTH_ENCRYPTION_KEYS",
    ]
    return dict(line.split("=", 1) for line in lines)


def test_generate_keys_output_is_loadable(capsys, monkeypatch):
    assert cli.main(["generate-keys"]) == 0
    out = _generated(capsys)

    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", out["AUTH_JWT_PRIVATE_KEYS"])
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", out["AUTH_ENCRYPTION_KEYS"])
    reset_key_cache()
    reset_encryption_cache()
    try:
        assert len(load_signing_keys()) == 1
        assert load_cipher().decrypt(encrypt_secret("JBSWY3DPEHPK3PXP").encode()) == (
            b"JBSWY3DPEHPK3PXP"
        )
    finally:
        reset_key_cache()
        reset_encryption_cache()


def test_generate_keys_is_two_lines_on_standard_output_and_nothing_on_standard_error(capsys):
    assert cli.main(["generate-keys"]) == 0

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 2 and captured.out.endswith("\n")
    assert captured.err == ""


def test_generate_keys_gives_new_keys_every_time(capsys):
    cli.main(["generate-keys"])
    first = _generated(capsys)
    cli.main(["generate-keys"])
    second = _generated(capsys)

    assert first["AUTH_JWT_PRIVATE_KEYS"] != second["AUTH_JWT_PRIVATE_KEYS"]
    assert first["AUTH_ENCRYPTION_KEYS"] != second["AUTH_ENCRYPTION_KEYS"]


def test_generate_keys_writes_nothing_to_disk_and_logs_nothing(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)

    assert cli.main(["generate-keys"]) == 0

    assert list(tmp_path.iterdir()) == []
    # (asyncio's own debug line about its selector is the event loop's, not the command's.)
    assert [r for r in caplog.records if r.name != "asyncio"] == []


def test_generate_keys_needs_neither_a_database_nor_a_mode(capsys, monkeypatch, settings_override):
    async def no_database(*_args, **_kwargs):
        raise AssertionError("generate-keys must not open a database session")

    monkeypatch.setattr("app.db.session.AsyncSessionLocal", no_database)
    settings_override(AUTH_MODE="supabase")

    assert cli.main(["generate-keys"]) == 0
    _generated(capsys)


# ── Usage ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [[], ["bogus"], ["reset-mfa"], ["unlock"], ["unlock", "--email"], ["generate-keys", "--x"]],
)
def test_a_wrong_command_line_exits_2_without_a_traceback(argv, capsys):
    assert cli.main(argv) == 2

    captured = capsys.readouterr()
    assert captured.out == "" and "usage:" in captured.err


# ── reset-mfa ────────────────────────────────────────────────────────────────


async def _enrolled(auth_db, **fields) -> User:
    """An account with a second factor: secret, pending secret, enabled date, counter."""
    return await seed_user(
        auth_db,
        password_hash="$argon2id$placeholder",
        mfa_totp_secret_enc=SECRET_BLOB,
        mfa_pending_secret_enc=SECRET_BLOB,
        mfa_enabled_at=NOW,
        mfa_last_used_step=12345,
        **fields,
    )


async def _give_recovery_codes(auth_db, user: User, how_many: int = 3) -> None:
    async with auth_db() as s:
        for _ in range(how_many):
            s.add(AuthRecoveryCode(user_id=user.id, code_hash=uuid4().hex + uuid4().hex))
        await s.commit()


async def _give_session(auth_db, user: User, **fields) -> None:
    values = dict(
        user_id=user.id,
        idle_expires_at=NOW + timedelta(days=7),
        expires_at=NOW + timedelta(days=30),
        refresh_token_hash=uuid4().hex + uuid4().hex,
        amr=["pwd", "otp"],
    )
    values.update(fields)
    async with auth_db() as s:
        s.add(AuthSession(**values))
        await s.commit()


async def _reload(auth_db, user_id) -> User:
    async with auth_db() as s:
        return await s.get(User, user_id)


async def _count(auth_db, model, *where) -> int:
    async with auth_db() as s:
        return await s.scalar(select(func.count()).select_from(model).where(*where))


async def _events(auth_db, event_type: AuthEventType) -> list[AuthEvent]:
    async with auth_db() as s:
        rows = await s.execute(
            select(AuthEvent).where(AuthEvent.event_type == event_type.value).order_by(AuthEvent.id)
        )
        return list(rows.scalars())


async def _run(auth_db, *argv):
    """The command on the test database: (exit code, standard output, standard error)."""
    out, err = StringIO(), StringIO()
    code = await cli.run(list(argv), session_factory=auth_db, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@MODES
async def test_break_glass_reset_records_event(auth_db, smtp_capture, settings_override, mode):
    settings_override(AUTH_MODE=mode)
    target = await _enrolled(auth_db)
    other = await _enrolled(auth_db)
    await _give_recovery_codes(auth_db, target, 3)
    await _give_recovery_codes(auth_db, other, 2)
    await _give_session(auth_db, target)
    await _give_session(auth_db, target)
    await _give_session(auth_db, other)

    code, out, err = await _run(auth_db, "reset-mfa", "--email", target.email)

    assert code == 0, err
    after = await _reload(auth_db, target.id)
    assert after.mfa_totp_secret_enc is None
    assert after.mfa_pending_secret_enc is None
    assert after.mfa_enabled_at is None
    assert after.mfa_last_used_step is None
    assert after.password_hash == "$argon2id$placeholder"  # the password is not the second factor
    assert await _count(auth_db, AuthRecoveryCode, AuthRecoveryCode.user_id == target.id) == 0
    assert await _count(auth_db, AuthRecoveryCode, AuthRecoveryCode.user_id == other.id) == 2
    async with auth_db() as s:
        ended = (
            (await s.execute(select(AuthSession).where(AuthSession.user_id == target.id)))
            .scalars()
            .all()
        )
        theirs = (
            (await s.execute(select(AuthSession).where(AuthSession.user_id == other.id)))
            .scalars()
            .one()
        )
    assert len(ended) == 2 and {s.revoked_reason for s in ended} == {"mfa_reset"}
    assert theirs.revoked_at is None
    assert (await _reload(auth_db, other.id)).mfa_enabled_at is not None

    [event] = await _events(auth_db, AuthEventType.BREAK_GLASS)
    assert event.user_id == target.id
    assert event.details == {"actor_id": None, "sessions_revoked": 2}
    assert await _events(auth_db, AuthEventType.MFA_RESET_BY_ADMIN) == []

    [mail] = smtp_capture.messages
    assert mail.to == target.email
    assert "Two-factor authentication was reset" in mail.subject
    # One line, no secret.
    assert len(out.strip().splitlines()) == 1 and target.email in out
    assert SECRET_BLOB not in out + err
    assert err == ""


async def test_the_account_is_found_ignoring_case_and_surrounding_spaces(auth_db, smtp_capture):
    target = await _enrolled(auth_db, email="Mixed.Case@Example.test")

    code, _out, err = await _run(auth_db, "reset-mfa", "--email", "  MIXED.case@EXAMPLE.TEST \t")

    assert code == 0, err
    assert (await _reload(auth_db, target.id)).mfa_enabled_at is None
    assert smtp_capture.last.to == "Mixed.Case@Example.test"


@pytest.mark.parametrize("command", ["reset-mfa", "unlock"])
async def test_an_unknown_address_is_refused_and_nothing_changes(auth_db, smtp_capture, command):
    bystander = await _enrolled(auth_db, failed_login_count=4)

    code, out, err = await _run(auth_db, command, "--email", "nobody@example.test")

    assert code == 2
    assert out == "" and "nobody@example.test" in err and "error" in err.lower()
    assert smtp_capture.messages == []
    assert await _count(auth_db, AuthEvent) == 0
    after = await _reload(auth_db, bystander.id)
    assert after.mfa_enabled_at is not None and after.failed_login_count == 4


@pytest.mark.parametrize("command", ["reset-mfa", "unlock"])
async def test_an_address_shared_by_two_accounts_is_refused_and_nothing_changes(
    auth_db, smtp_capture, command
):
    # The e-mail column is unique, but case-sensitively: the migration can leave two accounts
    # whose addresses differ by case only. Neither can be told from the other.
    first = await _enrolled(auth_db, email="Twin@example.test", failed_login_count=3)
    second = await _enrolled(auth_db, email="twin@example.test", failed_login_count=3)

    code, out, err = await _run(auth_db, command, "--email", "TWIN@example.test")

    assert code == 2
    assert out == "" and "several accounts" in err
    assert smtp_capture.messages == []
    assert await _count(auth_db, AuthEvent) == 0
    for account in (first, second):
        after = await _reload(auth_db, account.id)
        assert after.mfa_enabled_at is not None and after.failed_login_count == 3


@pytest.mark.parametrize("failure", ["send_fails", "not_configured"])
async def test_the_reset_stands_when_the_notification_cannot_be_sent(
    auth_db, smtp_capture, settings_override, failure
):
    if failure == "send_fails":
        smtp_capture.fail_with = OSError("smtp down")
    else:
        settings_override(SMTP_HOST=None)
    target = await _enrolled(auth_db)
    await _give_recovery_codes(auth_db, target)

    code, out, err = await _run(auth_db, "reset-mfa", "--email", target.email)

    assert code == 0
    assert "not sent" in err.lower() and "warning" in err.lower()
    assert "smtp down" not in err
    after = await _reload(auth_db, target.id)
    assert after.mfa_enabled_at is None and after.mfa_totp_secret_enc is None
    assert await _count(auth_db, AuthRecoveryCode, AuthRecoveryCode.user_id == target.id) == 0
    assert len(await _events(auth_db, AuthEventType.BREAK_GLASS)) == 1
    assert len(out.strip().splitlines()) == 1


async def test_a_reset_of_an_account_without_mfa_still_succeeds(auth_db, smtp_capture):
    target = await seed_user(auth_db)

    code, _out, err = await _run(auth_db, "reset-mfa", "--email", target.email)

    assert code == 0, err
    assert len(await _events(auth_db, AuthEventType.BREAK_GLASS)) == 1


async def test_the_command_uses_the_application_session_factory_by_default(
    auth_db, smtp_capture, monkeypatch
):
    target = await _enrolled(auth_db)
    monkeypatch.setattr("app.db.session.AsyncSessionLocal", auth_db)

    out, err = StringIO(), StringIO()
    code = await cli.run(["reset-mfa", "--email", target.email], out=out, err=err)

    assert code == 0, err.getvalue()
    assert (await _reload(auth_db, target.id)).mfa_enabled_at is None


# ── unlock ───────────────────────────────────────────────────────────────────


@MODES
async def test_unlock_clears_the_lock_and_records_it(
    auth_db, smtp_capture, settings_override, mode
):
    settings_override(AUTH_MODE=mode)
    locked = await seed_user(
        auth_db, failed_login_count=9, locked_until=NOW + timedelta(hours=1), mfa_enabled_at=NOW
    )
    other = await seed_user(auth_db, failed_login_count=2)

    code, out, err = await _run(auth_db, "unlock", "--email", locked.email.upper())

    assert code == 0, err
    after = await _reload(auth_db, locked.id)
    assert after.failed_login_count == 0 and after.locked_until is None
    assert after.mfa_enabled_at is not None  # unlocking is not resetting the second factor
    assert (await _reload(auth_db, other.id)).failed_login_count == 2
    [event] = await _events(auth_db, AuthEventType.ACCOUNT_UNLOCKED)
    assert event.user_id == locked.id and event.details == {"actor_id": None}
    assert smtp_capture.messages == []  # an unlock sends nothing
    assert len(out.strip().splitlines()) == 1 and locked.email in out
    assert err == ""


# ── The retention purge ──────────────────────────────────────────────────────

DAY = timedelta(days=1)
SECOND = timedelta(seconds=1)


def test_retention_is_one_year_for_events_a_week_for_links_a_month_for_sessions():
    assert EVENT_RETENTION == timedelta(days=365)
    assert TOKEN_RETENTION == timedelta(days=7)
    assert SESSION_RETENTION == timedelta(days=30)


async def _add(auth_db, *rows) -> None:
    async with auth_db() as s:
        s.add_all(rows)
        await s.commit()


async def _purge(auth_db, now=NOW) -> dict:
    async with auth_db() as s:
        counts = await purge_expired_auth_records(s, now)
        await s.commit()
    return counts


async def _surviving(auth_db, column, model) -> set:
    async with auth_db() as s:
        return set((await s.execute(select(column).select_from(model))).scalars())


def _event(created_at, kind="login_succeeded", **fields) -> AuthEvent:
    return AuthEvent(event_type=kind, created_at=created_at, details={}, **fields)


async def test_purge_respects_retention(auth_db):
    user = await seed_user(auth_db)
    await _add(
        auth_db,
        _event(NOW - 366 * DAY, user_id=user.id, kind="old"),
        _event(NOW - 364 * DAY, user_id=user.id, kind="recent"),
    )

    counts = await _purge(auth_db)

    assert counts["events"] == 1
    assert await _surviving(auth_db, AuthEvent.event_type, AuthEvent) == {"recent"}


async def test_an_event_is_kept_up_to_exactly_365_days_and_deleted_after(auth_db):
    await _add(
        auth_db,
        _event(NOW - 365 * DAY, kind="exactly_a_year"),
        _event(NOW - 365 * DAY - SECOND, kind="a_second_over"),
        _event(NOW - 365 * DAY + SECOND, kind="a_second_under"),
        _event(NOW, kind="today"),
        _event(NOW + DAY, kind="from_the_future"),
    )

    counts = await _purge(auth_db)

    assert counts["events"] == 1
    assert await _surviving(auth_db, AuthEvent.event_type, AuthEvent) == {
        "exactly_a_year",
        "a_second_under",
        "today",
        "from_the_future",
    }


async def test_events_of_a_deleted_account_are_purged_like_the_others(auth_db):
    await _add(auth_db, _event(NOW - 400 * DAY, user_id=None), _event(NOW - 10 * DAY, user_id=None))

    counts = await _purge(auth_db)

    assert counts["events"] == 1
    assert await _count(auth_db, AuthEvent) == 1


def _link(user, *, expires_at, used_at=None) -> AuthOneTimeToken:
    return AuthOneTimeToken(
        id=uuid4(),
        user_id=user.id,
        purpose=OneTimeTokenPurpose.PASSWORD_RESET,
        token_hash=uuid4().hex + uuid4().hex,
        expires_at=expires_at,
        used_at=used_at,
    )


async def test_one_time_tokens_go_a_week_after_they_expired_or_were_used(auth_db):
    user = await seed_user(auth_db)
    cases = {
        # (expires_at, used_at): deleted?
        "expired_8_days_ago": (_link(user, expires_at=NOW - 8 * DAY), True),
        "expired_6_days_ago": (_link(user, expires_at=NOW - 6 * DAY), False),
        "expired_exactly_7_days_ago": (_link(user, expires_at=NOW - 7 * DAY), False),
        "expired_7_days_and_a_second": (_link(user, expires_at=NOW - 7 * DAY - SECOND), True),
        "used_8_days_ago_still_valid": (
            _link(user, expires_at=NOW + DAY, used_at=NOW - 8 * DAY),
            True,
        ),
        "used_6_days_ago_still_valid": (
            _link(user, expires_at=NOW + DAY, used_at=NOW - 6 * DAY),
            False,
        ),
        "used_exactly_7_days_ago": (
            _link(user, expires_at=NOW + DAY, used_at=NOW - 7 * DAY),
            False,
        ),
        "used_8_days_ago_and_expired_6": (
            _link(user, expires_at=NOW - 6 * DAY, used_at=NOW - 8 * DAY),
            True,
        ),
        "valid_and_unused": (_link(user, expires_at=NOW + DAY), False),
        "valid_and_used_just_now": (_link(user, expires_at=NOW + DAY, used_at=NOW), False),
    }
    ids = {name: row.id for name, (row, _gone) in cases.items()}
    await _add(auth_db, *(row for row, _gone in cases.values()))

    counts = await _purge(auth_db)

    gone = {name for name, (_row, deleted) in cases.items() if deleted}
    assert counts["tokens"] == len(gone)
    kept = await _surviving(auth_db, AuthOneTimeToken.id, AuthOneTimeToken)
    assert {name for name, token_id in ids.items() if token_id in kept} == set(cases) - gone


def _session(user, *, expires_at, revoked_at=None, idle_expires_at=None) -> AuthSession:
    return AuthSession(
        id=uuid4(),
        user_id=user.id,
        idle_expires_at=idle_expires_at or expires_at,
        expires_at=expires_at,
        revoked_at=revoked_at,
        refresh_token_hash=uuid4().hex + uuid4().hex,
        amr=["pwd"],
    )


async def test_sessions_go_a_month_after_they_expired_or_were_revoked(auth_db):
    user = await seed_user(auth_db)
    live = NOW + 20 * DAY
    cases = {
        "expired_31_days_ago": (_session(user, expires_at=NOW - 31 * DAY), True),
        "expired_29_days_ago": (_session(user, expires_at=NOW - 29 * DAY), False),
        "expired_exactly_30_days_ago": (_session(user, expires_at=NOW - 30 * DAY), False),
        "expired_30_days_and_a_second": (
            _session(user, expires_at=NOW - 30 * DAY - SECOND),
            True,
        ),
        "revoked_31_days_ago_would_still_be_valid": (
            _session(user, expires_at=live, revoked_at=NOW - 31 * DAY),
            True,
        ),
        "revoked_29_days_ago": (_session(user, expires_at=live, revoked_at=NOW - 29 * DAY), False),
        "revoked_exactly_30_days_ago": (
            _session(user, expires_at=live, revoked_at=NOW - 30 * DAY),
            False,
        ),
        "idle_for_31_days": (
            _session(user, expires_at=live, idle_expires_at=NOW - 31 * DAY),
            True,
        ),
        "idle_for_29_days": (
            _session(user, expires_at=live, idle_expires_at=NOW - 29 * DAY),
            False,
        ),
        "live": (_session(user, expires_at=live, idle_expires_at=NOW + 3 * DAY), False),
        "live_and_revoked_just_now": (_session(user, expires_at=live, revoked_at=NOW), False),
        "expired_just_now": (_session(user, expires_at=NOW), False),
    }
    ids = {name: row.id for name, (row, _gone) in cases.items()}
    await _add(auth_db, *(row for row, _gone in cases.values()))

    counts = await _purge(auth_db)

    gone = {name for name, (_row, deleted) in cases.items() if deleted}
    assert counts["sessions"] == len(gone)
    kept = await _surviving(auth_db, AuthSession.id, AuthSession)
    assert {name for name, session_id in ids.items() if session_id in kept} == set(cases) - gone


async def test_the_purge_touches_nothing_else_and_a_second_run_has_nothing_to_do(auth_db):
    user = await seed_user(auth_db, mfa_enabled_at=NOW - 900 * DAY)
    await _give_recovery_codes(auth_db, user, 2)
    await _add(
        auth_db,
        _event(NOW - 400 * DAY, user_id=user.id),
        _link(user, expires_at=NOW - 20 * DAY),
        _session(user, expires_at=NOW - 60 * DAY),
    )

    first = await _purge(auth_db)
    second = await _purge(auth_db)

    assert first == {"events": 1, "tokens": 1, "sessions": 1}
    assert second == {"events": 0, "tokens": 0, "sessions": 0}
    assert await _count(auth_db, User) == 1
    assert await _count(auth_db, AuthRecoveryCode) == 2


async def test_the_purge_leaves_the_commit_to_the_caller(auth_db):
    await _add(auth_db, _event(NOW - 400 * DAY))

    async with auth_db() as s:
        assert (await purge_expired_auth_records(s, NOW))["events"] == 1
        await s.rollback()

    assert await _count(auth_db, AuthEvent) == 1


async def test_the_purge_works_at_the_instant_it_is_given_and_a_naive_one_is_utc(auth_db):
    await _add(auth_db, _event(NOW - 366 * DAY, kind="old"), _event(NOW - 10 * DAY, kind="new"))

    counts = await _purge(auth_db, now=NOW.replace(tzinfo=None))

    assert counts["events"] == 1
    assert await _surviving(auth_db, AuthEvent.event_type, AuthEvent) == {"new"}


# ── The Celery task ──────────────────────────────────────────────────────────


async def test_the_task_purges_through_the_application_session_and_commits(auth_db, monkeypatch):
    monkeypatch.setattr("app.db.session.AsyncSessionLocal", auth_db)
    wall_clock = datetime.now(UTC)
    user = await seed_user(auth_db)
    await _add(
        auth_db,
        _event(wall_clock - 366 * DAY, kind="old"),
        _event(wall_clock - 364 * DAY, kind="recent"),
        _link(user, expires_at=wall_clock - 8 * DAY),
        _session(user, expires_at=wall_clock - 31 * DAY),
    )

    counts = await auth_tasks._async_purge()

    assert counts == {"events": 1, "tokens": 1, "sessions": 1}
    assert await _surviving(auth_db, AuthEvent.event_type, AuthEvent) == {"recent"}
    assert await _count(auth_db, AuthOneTimeToken) == 0
    assert await _count(auth_db, AuthSession) == 0


async def test_the_task_logs_the_three_counts_and_nothing_sensitive(auth_db, monkeypatch, caplog):
    monkeypatch.setattr("app.db.session.AsyncSessionLocal", auth_db)
    user = await seed_user(auth_db)
    wall_clock = datetime.now(UTC)
    await _add(auth_db, _event(wall_clock - 400 * DAY, user_id=user.id))
    caplog.set_level(logging.INFO, logger=auth_tasks.logger.name)

    await auth_tasks._async_purge()

    [record] = [r for r in caplog.records if r.name == auth_tasks.logger.name]
    message = record.getMessage()
    assert "events=1" in message and "tokens=0" in message and "sessions=0" in message
    assert user.email not in message


def test_the_celery_task_runs_the_purge_and_returns_its_counts(monkeypatch):
    async def fake_purge():
        return {"events": 4, "tokens": 5, "sessions": 6}

    monkeypatch.setattr(auth_tasks, "_async_purge", fake_purge)

    assert auth_tasks.purge_auth_records() == {"events": 4, "tokens": 5, "sessions": 6}


def test_the_purge_is_a_registered_task_scheduled_daily_at_0315():
    from app.worker.celery_app import celery_app

    name = "app.worker.tasks.auth_tasks.purge_auth_records"
    assert auth_tasks.purge_auth_records.name == name
    assert name in celery_app.tasks
    assert "app.worker.tasks.auth_tasks" in celery_app.conf.include

    [entry] = [e for e in celery_app.conf.beat_schedule.values() if e["task"] == name]
    schedule = entry["schedule"]
    assert schedule.minute == {15} and schedule.hour == {3}
    assert schedule.day_of_month == {*range(1, 32)}, "must run every day of the month"
    assert schedule.month_of_year == {*range(1, 13)}, "must run every month"
    assert schedule.day_of_week == {*range(0, 7)}, "must run every day of the week"
    assert "purge-auth-records" in celery_app.conf.beat_schedule
