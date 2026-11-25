"""
The one-off copy of the Supabase credentials into the local `users` table
(`app/auth/migrate_from_supabase.py`, spec section 8).

What is pinned down:
- the planning rules of section 8, one test per rule, on a pure function (no I/O, the clock is a
  parameter): scope, password, verification and last-login dates, e-mail, role, banned and
  deleted accounts, duplicates, orphans, accounts out of scope;
- a password changed after the migration is never overwritten, not by the plan, not by `apply`
  even if the row changed after the plan was made, and not by a second run either;
- `apply_plan` writes in one transaction, returns how many accounts it changed, can be replayed,
  never blanks a value it has nothing to put in, and an inherited `$2a$10$` hash really signs the
  account in afterwards;
- the dry run writes nothing; neither the output nor the report ever holds a hash, and the report
  is private to its owner;
- the read of the hosted database is read-only and accepts both forms of connection string;
- the reset links go to the accounts that have no password, and only in local mode.

Everything runs against an in-memory SQLite database with synthetic `@example.test` accounts. The
hosted database is never reached; one optional test uses the throwaway Postgres of the test setup
(`localhost:55432`) and is skipped when it is not there.
"""

import json
import os
import stat
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Optional
from uuid import UUID, uuid4

import bcrypt
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import StatementError

from app.auth import migrate_from_supabase as migration
from app.auth.clock import aware_utc
from app.auth.migrate_from_supabase import (
    LocalUser,
    RemoteUser,
    apply_plan,
    fetch_remote_users,
    load_local_users,
    plan_migration,
    send_reset_links,
)
from app.auth.passwords import hash_password, verify_password
from app.models.auth import AuthEvent, AuthEventType, AuthOneTimeToken, OneTimeTokenPurpose
from app.models.models import User, UserRole, UserStatus
from tests.auth_support import NOW, PW, seed_user

BCRYPT_X = "$2a$10$" + "x" * 53
BCRYPT_Y = "$2a$10$" + "y" * 53


def L(  # noqa: N802 - a terse factory, like `R` below
    id: Optional[UUID] = None,
    email: Optional[str] = None,
    role: UserRole = UserRole.USER,
    password_changed_at: Optional[datetime] = None,
    auth_migrated_at: Optional[datetime] = None,
) -> LocalUser:
    id = id or uuid4()
    return LocalUser(
        id=id,
        email=email or f"{id.hex[:8]}@example.test",
        role=role,
        password_changed_at=password_changed_at,
        auth_migrated_at=auth_migrated_at,
    )


def R(  # noqa: N802
    id: UUID,
    email: Optional[str] = None,
    pw: Optional[str] = BCRYPT_X,
    confirmed: Optional[datetime] = None,
    signed_in: Optional[datetime] = None,
    banned_until: Optional[datetime] = None,
    deleted_at: Optional[datetime] = None,
    profile_role: Optional[str] = None,
) -> RemoteUser:
    return RemoteUser(
        id=id,
        email=f"{id.hex[:8]}@example.test" if email is None else email,
        encrypted_password=pw,
        email_confirmed_at=confirmed,
        last_sign_in_at=signed_in,
        banned_until=banned_until,
        deleted_at=deleted_at,
        profile_role=profile_role,
    )


def by_id(updates, user_id):
    matches = [u for u in updates if u.user_id == user_id]
    assert len(matches) <= 1
    return matches[0] if matches else None


def kinds(anomalies) -> set:
    return {a.kind for a in anomalies}


def anomalies_of(plan, user_id) -> set:
    return {a.kind for a in plan.anomalies if a.user_id == user_id}


# ---- the planning rules (spec section 8) --------------------------------------------------------


def test_rules():
    a, b, c, orphan, stranger = (uuid4() for _ in range(5))
    plan = plan_migration(
        local=[L(a), L(b, role=UserRole.ADMIN), L(c), L(orphan)],
        remote=[
            R(a, pw=BCRYPT_X, profile_role="analyst"),
            R(b, pw=BCRYPT_Y, profile_role="user"),
            R(c, pw=None),
            R(stranger),
        ],
        now=NOW,
    )
    ua, ub = by_id(plan.updates, a), by_id(plan.updates, b)
    assert ua.password_hash == BCRYPT_X and ua.role == UserRole.ANALYST
    assert ub.role is None  # protected local ADMIN kept
    assert kinds(plan.anomalies) == {"no_password", "missing_remote"}
    assert plan.out_of_scope == 1
    assert anomalies_of(plan, c) == {"no_password"}
    assert anomalies_of(plan, orphan) == {"missing_remote"}
    assert by_id(plan.updates, orphan) is None  # nothing to apply to an account Supabase lacks


def test_only_the_ids_present_locally_are_in_scope():
    mine, other_1, other_2 = uuid4(), uuid4(), uuid4()
    plan = plan_migration([L(mine)], [R(mine, pw=BCRYPT_X), R(other_1), R(other_2)], now=NOW)
    assert [u.user_id for u in plan.updates] == [mine]
    assert plan.out_of_scope == 2
    assert plan.anomalies == []


def test_an_empty_world_plans_nothing():
    plan = plan_migration([], [], now=NOW)
    assert (plan.updates, plan.anomalies, plan.out_of_scope) == ([], [], 0)


@pytest.mark.parametrize("prefix", ["$2a$10$", "$2b$10$", "$2y$12$"])
def test_a_bcrypt_hash_is_copied_as_it_is(prefix):
    uid = uuid4()
    hashed = prefix + "z" * 53
    update = by_id(plan_migration([L(uid)], [R(uid, pw=hashed)], now=NOW).updates, uid)
    assert update.password_hash == hashed


@pytest.mark.parametrize("stored", [None, "", "$argon2id$v=19$m=65536,t=3,p=4$abc$def", "plain"])
def test_no_bcrypt_hash_means_no_password_and_a_reset_link(stored):
    uid = uuid4()
    plan = plan_migration([L(uid)], [R(uid, pw=stored)], now=NOW)
    assert by_id(plan.updates, uid).password_hash is None
    assert anomalies_of(plan, uid) == {"no_password"}


def test_verification_and_last_login_dates_follow_supabase():
    uid = uuid4()
    confirmed, signed_in = NOW - timedelta(days=40), NOW - timedelta(hours=3)
    plan = plan_migration(
        [L(uid)], [R(uid, pw=BCRYPT_X, confirmed=confirmed, signed_in=signed_in)], now=NOW
    )
    update = by_id(plan.updates, uid)
    assert update.email_verified_at == confirmed
    assert update.last_login_at == signed_in


def test_remote_email_wins_unless_conflict():
    free, taken_by_other, same_up_to_case = uuid4(), uuid4(), uuid4()
    other_account = uuid4()
    plan = plan_migration(
        local=[
            L(free, email="old@example.test"),
            L(taken_by_other, email="mine@example.test"),
            L(same_up_to_case, email="Case@example.test"),
            L(other_account, email="held@example.test"),
        ],
        remote=[
            R(free, email="new@example.test"),
            R(taken_by_other, email="HELD@example.test"),  # the other account's, ignoring case
            R(same_up_to_case, email="case@example.test"),  # its own, in another case: no conflict
            R(other_account, email="held@example.test"),
        ],
        now=NOW,
    )
    assert by_id(plan.updates, free).email == "new@example.test"
    assert by_id(plan.updates, taken_by_other).email is None  # local e-mail kept
    assert by_id(plan.updates, same_up_to_case).email == "case@example.test"
    assert anomalies_of(plan, taken_by_other) == {"email_conflict"}
    assert anomalies_of(plan, free) == set()
    assert anomalies_of(plan, same_up_to_case) == set()
    conflict = next(a for a in plan.anomalies if a.kind == "email_conflict")
    assert "held@example.test" in conflict.detail.lower()  # the report says which address


def test_two_accounts_cannot_both_move_to_the_same_new_address():
    first, second = uuid4(), uuid4()
    plan = plan_migration(
        [L(first, email="a@example.test"), L(second, email="b@example.test")],
        [R(first, email="shared@example.test"), R(second, email="Shared@example.test")],
        now=NOW,
    )
    assert by_id(plan.updates, first).email == "shared@example.test"
    assert by_id(plan.updates, second).email is None
    assert anomalies_of(plan, second) == {"email_conflict"}


def test_swapped_addresses_are_conflicts_not_a_unique_violation():
    one, two = uuid4(), uuid4()
    plan = plan_migration(
        [L(one, email="a@example.test"), L(two, email="b@example.test")],
        [R(one, email="b@example.test"), R(two, email="a@example.test")],
        now=NOW,
    )
    assert by_id(plan.updates, one).email is None and by_id(plan.updates, two).email is None
    assert [a.kind for a in plan.anomalies] == ["email_conflict", "email_conflict"]


def test_an_account_without_a_remote_address_keeps_its_own():
    uid = uuid4()
    update = by_id(plan_migration([L(uid)], [R(uid, email="")], now=NOW).updates, uid)
    assert update.email is None


def test_the_supabase_address_is_written_lowercased_and_stripped():
    """Addresses are stored lowercased (spec section 7.1): the sign-in looks them up that way."""
    moved, own_in_capitals = uuid4(), uuid4()
    plan = plan_migration(
        [L(moved, email="old@example.test"), L(own_in_capitals, email="Mine@Example.test")],
        [
            R(moved, email="  New.Person@Example.TEST "),
            R(own_in_capitals, email="MINE@example.test"),
        ],
        now=NOW,
    )
    assert by_id(plan.updates, moved).email == "new.person@example.test"
    assert by_id(plan.updates, own_in_capitals).email == "mine@example.test"
    assert plan.anomalies == []


def test_out_of_scope_comes_from_the_hosted_count_when_it_is_given():
    """The read fetches the local accounts only and counts the others (`remote_total`)."""
    mine, missing = uuid4(), uuid4()
    plan = plan_migration([L(mine), L(missing)], [R(mine)], now=NOW, remote_total=7)
    assert plan.out_of_scope == 6
    assert plan_migration([L(mine)], [R(mine)], now=NOW, remote_total=1).out_of_scope == 0


def test_a_hash_never_shows_in_a_repr():
    uid = uuid4()
    remote = R(uid, pw=BCRYPT_X)
    plan = plan_migration([L(uid)], [remote], now=NOW)
    assert by_id(plan.updates, uid).password_hash == BCRYPT_X
    explicit = migration.UserUpdate(user_id=uid, password_hash=BCRYPT_Y)
    for text in (repr(remote), repr(plan), str(plan), repr(explicit)):
        assert BCRYPT_X not in text and BCRYPT_Y not in text and "$2a$" not in text


@pytest.mark.parametrize("local_role", [UserRole.ADMIN, UserRole.SCILICIUM_ADMIN])
@pytest.mark.parametrize("profile_role", ["user", "viewer", "nonsense", None])
def test_local_admins_are_never_touched_by_the_profile_role(local_role, profile_role):
    uid = uuid4()
    plan = plan_migration([L(uid, role=local_role)], [R(uid, profile_role=profile_role)], now=NOW)
    assert by_id(plan.updates, uid).role is None
    assert "role_invalid" not in anomalies_of(plan, uid)


@pytest.mark.parametrize(
    ("profile_role", "expected"),
    [
        ("analyst", UserRole.ANALYST),
        ("USER", UserRole.USER),
        ("Viewer", UserRole.VIEWER),
        ("subscriber", UserRole.SUBSCRIBER),
        ("scilicium_admin", UserRole.SCILICIUM_ADMIN),
        ("admin", UserRole.ADMIN),
    ],
)
def test_the_profile_role_is_applied_in_capitals_when_valid(profile_role, expected):
    uid = uuid4()
    plan = plan_migration(
        [L(uid, role=UserRole.VIEWER)], [R(uid, profile_role=profile_role)], now=NOW
    )
    assert by_id(plan.updates, uid).role == expected
    assert plan.anomalies == []


@pytest.mark.parametrize("profile_role", ["superuser", "ADMINS", "  not a role  "])
def test_an_invalid_profile_role_keeps_the_local_one_and_is_reported(profile_role):
    uid = uuid4()
    plan = plan_migration(
        [L(uid, role=UserRole.ANALYST)], [R(uid, profile_role=profile_role)], now=NOW
    )
    assert by_id(plan.updates, uid).role is None
    assert anomalies_of(plan, uid) == {"role_invalid"}
    assert profile_role.strip() in next(a for a in plan.anomalies).detail


@pytest.mark.parametrize("profile_role", [None, "", "   "])
def test_a_missing_profile_role_is_not_an_anomaly(profile_role):
    uid = uuid4()
    plan = plan_migration([L(uid)], [R(uid, profile_role=profile_role)], now=NOW)
    assert by_id(plan.updates, uid).role is None
    assert plan.anomalies == []


def test_banned_and_deleted_are_reported_not_applied():
    banned, deleted, both, ban_over, ok = (uuid4() for _ in range(5))
    plan = plan_migration(
        [L(banned), L(deleted), L(both), L(ban_over), L(ok)],
        [
            R(banned, pw=BCRYPT_X, banned_until=NOW + timedelta(days=1)),
            R(deleted, pw=BCRYPT_X, deleted_at=NOW - timedelta(days=1)),
            R(both, pw=BCRYPT_X, banned_until=NOW + timedelta(days=9), deleted_at=NOW),
            R(ban_over, pw=BCRYPT_X, banned_until=NOW - timedelta(seconds=1)),  # lifted
            R(ok, pw=BCRYPT_X),
        ],
        now=NOW,
    )
    assert anomalies_of(plan, banned) == {"banned"}
    assert anomalies_of(plan, deleted) == {"deleted"}
    assert anomalies_of(plan, both) == {"banned", "deleted"}
    assert anomalies_of(plan, ban_over) == set()
    for uid in (banned, deleted, both):
        assert by_id(plan.updates, uid) is None
    assert {u.user_id for u in plan.updates} == {ban_over, ok}


def test_a_ban_that_ends_exactly_now_is_over():
    uid = uuid4()
    plan = plan_migration([L(uid)], [R(uid, pw=BCRYPT_X, banned_until=NOW)], now=NOW)
    assert anomalies_of(plan, uid) == set() and by_id(plan.updates, uid) is not None


def test_a_naive_ban_date_is_read_as_utc():
    uid = uuid4()
    naive_future = (NOW + timedelta(hours=1)).replace(tzinfo=None)
    plan = plan_migration([L(uid)], [R(uid, banned_until=naive_future)], now=NOW)
    assert anomalies_of(plan, uid) == {"banned"}


def test_the_clock_is_a_parameter_not_the_wall_clock():
    uid = uuid4()
    ban = datetime(2030, 1, 1, tzinfo=timezone.utc)
    before = plan_migration([L(uid)], [R(uid, banned_until=ban)], now=ban - timedelta(days=1))
    after = plan_migration([L(uid)], [R(uid, banned_until=ban)], now=ban + timedelta(days=1))
    assert anomalies_of(before, uid) == {"banned"}
    assert anomalies_of(after, uid) == set()


def test_without_a_clock_the_plan_uses_the_current_time():
    uid = uuid4()
    soon = datetime.now(timezone.utc) + timedelta(hours=1)
    plan = plan_migration([L(uid)], [R(uid, banned_until=soon)])
    assert anomalies_of(plan, uid) == {"banned"}


def test_duplicate_local_emails_are_reported_for_each_account():
    first, second, third, alone = (uuid4() for _ in range(4))
    plan = plan_migration(
        [
            L(first, email="dup@example.test"),
            L(second, email="DUP@example.test"),
            L(third, email="dup@EXAMPLE.test"),
            L(alone, email="alone@example.test"),
        ],
        [],
        now=NOW,
    )
    assert {a.user_id for a in plan.anomalies if a.kind == "duplicate_email"} == {
        first,
        second,
        third,
    }
    assert anomalies_of(plan, alone) == {"missing_remote"}


# ---- never overwrite a password changed in the new system ---------------------------------------


def test_never_overwrites_a_password_changed_after_migration():
    # (c) migrated at T, changed at T+1: kept.
    # (d) migrated at T, changed before T (or never, or exactly at T): applied.
    changed, before, never_changed, tie = (uuid4() for _ in range(4))
    migrated = NOW - timedelta(days=10)
    plan = plan_migration(
        [
            L(changed, password_changed_at=migrated + timedelta(days=1), auth_migrated_at=migrated),
            L(before, password_changed_at=migrated - timedelta(days=1), auth_migrated_at=migrated),
            L(never_changed, password_changed_at=None, auth_migrated_at=migrated),
            L(tie, password_changed_at=migrated, auth_migrated_at=migrated),
        ],
        [R(uid, pw=BCRYPT_X) for uid in (changed, before, never_changed, tie)],
        now=NOW,
    )
    update = by_id(plan.updates, changed)
    assert update.keep_password is True and update.password_hash is None
    for uid in (before, never_changed, tie):
        update = by_id(plan.updates, uid)
        assert update.keep_password is False and update.password_hash == BCRYPT_X


def test_a_password_set_in_the_new_system_is_kept_even_before_the_first_migration():
    # (a) never migrated + password_changed_at set: only the new system writes that column, so the
    # Supabase hash must not replace it.
    uid = uuid4()
    plan = plan_migration(
        [L(uid, password_changed_at=NOW - timedelta(days=1), auth_migrated_at=None)],
        [R(uid, pw=BCRYPT_X)],
        now=NOW,
    )
    update = by_id(plan.updates, uid)
    assert update.keep_password is True and update.password_hash is None
    assert plan.anomalies == []  # it has a password: no reset link needed


def test_a_never_migrated_account_without_a_local_password_change_gets_the_password():
    # (b) never migrated + password_changed_at NULL: applied.
    uid = uuid4()
    plan = plan_migration(
        [L(uid, password_changed_at=None, auth_migrated_at=None)], [R(uid, pw=BCRYPT_X)], now=NOW
    )
    update = by_id(plan.updates, uid)
    assert update.keep_password is False and update.password_hash == BCRYPT_X


def test_naive_stored_dates_compare_with_aware_ones():
    # SQLite hands datetimes back naive: the comparison must not raise nor go wrong.
    uid = uuid4()
    migrated = NOW - timedelta(days=2)
    local = L(
        uid,
        password_changed_at=(migrated + timedelta(hours=1)).replace(tzinfo=None),
        auth_migrated_at=migrated,
    )
    update = by_id(plan_migration([local], [R(uid, pw=BCRYPT_X)], now=NOW).updates, uid)
    assert update.keep_password is True


def test_planning_is_pure_and_deterministic():
    a, b = uuid4(), uuid4()
    local = [L(a), L(b, email="x@example.test")]
    remote = [R(a, pw=BCRYPT_X, profile_role="viewer"), R(b, email="y@example.test"), R(uuid4())]
    before = (list(local), list(remote))
    assert plan_migration(local, remote, now=NOW) == plan_migration(local, remote, now=NOW)
    assert (local, remote) == before


# ---- apply_plan ---------------------------------------------------------------------------------


async def _row(sessions, user_id) -> User:
    async with sessions() as s:
        return await s.get(User, user_id)


async def test_apply_writes_what_the_plan_says_and_stamps_the_migration(auth_db):
    user = await seed_user(auth_db, email="old@example.test", role=UserRole.VIEWER)
    confirmed, signed_in = NOW - timedelta(days=30), NOW - timedelta(hours=5)
    plan = plan_migration(
        [LocalUser(user.id, user.email, user.role, None, None)],
        [
            R(
                user.id,
                email="new@example.test",
                pw=BCRYPT_X,
                confirmed=confirmed,
                signed_in=signed_in,
                profile_role="analyst",
            )
        ],
        now=NOW,
    )
    async with auth_db() as db:
        assert await apply_plan(db, plan, NOW) == 1
    row = await _row(auth_db, user.id)
    assert row.email == "new@example.test"
    assert row.password_hash == BCRYPT_X
    assert row.role == UserRole.ANALYST
    assert aware_utc(row.email_verified_at) == confirmed
    assert aware_utc(row.last_login_at) == signed_in
    assert aware_utc(row.auth_migrated_at) == NOW


async def test_apply_is_replayable(auth_db):
    user = await seed_user(auth_db)
    plan = plan_migration(
        [LocalUser(user.id, user.email, user.role, None, None)],
        [R(user.id, email=user.email, pw=BCRYPT_X, confirmed=NOW, profile_role="viewer")],
        now=NOW,
    )

    async def snapshot():
        row = await _row(auth_db, user.id)
        return (
            row.email,
            row.password_hash,
            row.role,
            row.email_verified_at,
            row.auth_migrated_at,
            row.password_changed_at,
        )

    async with auth_db() as db:
        assert await apply_plan(db, plan, NOW) == 1
    first = await snapshot()
    async with auth_db() as db:
        assert await apply_plan(db, plan, NOW) == 1
    assert await snapshot() == first
    async with auth_db() as db:  # even a later run of the same plan leaves the same rows
        await apply_plan(db, plan, NOW + timedelta(days=3))
    assert await snapshot() == first


async def test_apply_changes_only_the_accounts_of_the_plan(auth_db):
    planned = await seed_user(auth_db)
    bystander = await seed_user(auth_db, role=UserRole.VIEWER)
    plan = plan_migration(
        [LocalUser(planned.id, planned.email, planned.role, None, None)],
        [R(planned.id, email=planned.email, pw=BCRYPT_X), R(bystander.id, pw=BCRYPT_Y)],
        now=NOW,
    )
    async with auth_db() as db:
        assert await apply_plan(db, plan, NOW) == 1
    other = await _row(auth_db, bystander.id)
    assert other.password_hash is None and other.auth_migrated_at is None


async def test_apply_counts_only_the_rows_it_changed(auth_db):
    present = await seed_user(auth_db)
    gone = uuid4()
    plan = plan_migration(
        [L(present.id, email=present.email), L(gone)],
        [R(present.id, email=present.email, pw=BCRYPT_X), R(gone, pw=BCRYPT_Y)],
        now=NOW,
    )
    assert len(plan.updates) == 2
    async with auth_db() as db:
        assert await apply_plan(db, plan, NOW) == 1  # the second row no longer exists


async def test_apply_never_blanks_a_value_the_plan_has_nothing_for(auth_db):
    # Supabase has no bcrypt hash, no confirmation date and no sign-in for this account, but it
    # now has local ones (a reset link was answered, an e-mail confirmed, a local sign-in made).
    user = await seed_user(
        auth_db,
        password_hash=hash_password(PW),
        email_verified_at=NOW - timedelta(days=1),
        last_login_at=NOW - timedelta(hours=1),
        password_changed_at=NOW - timedelta(days=1),
        auth_migrated_at=NOW - timedelta(days=5),
    )
    before = await _row(auth_db, user.id)
    plan = plan_migration(
        [
            LocalUser(
                user.id, user.email, user.role, user.password_changed_at, user.auth_migrated_at
            )
        ],
        [R(user.id, email=user.email, pw=None)],
        now=NOW,
    )
    async with auth_db() as db:
        await apply_plan(db, plan, NOW)
    after = await _row(auth_db, user.id)
    assert after.password_hash == before.password_hash
    assert after.email_verified_at == before.email_verified_at
    assert after.last_login_at == before.last_login_at
    assert after.role == before.role and after.email == before.email


async def test_apply_keeps_a_password_changed_after_the_migration(auth_db):
    mine = hash_password(PW)
    migrated = NOW - timedelta(days=10)
    user = await seed_user(
        auth_db,
        password_hash=mine,
        password_changed_at=migrated + timedelta(days=2),
        auth_migrated_at=migrated,
    )
    plan = plan_migration(
        [LocalUser(user.id, user.email, user.role, user.password_changed_at, migrated)],
        [R(user.id, email=user.email, pw=BCRYPT_X, confirmed=NOW)],
        now=NOW,
    )
    assert by_id(plan.updates, user.id).keep_password is True
    async with auth_db() as db:
        assert await apply_plan(db, plan, NOW) == 1
    row = await _row(auth_db, user.id)
    assert row.password_hash == mine
    assert aware_utc(row.email_verified_at) == NOW  # the rest of the account is still synchronised


async def test_apply_cannot_overwrite_a_password_changed_after_the_plan_was_made(auth_db):
    # The plan saw a never-changed account; the owner chooses a new password before `apply`.
    migrated = NOW - timedelta(days=10)
    user = await seed_user(auth_db, auth_migrated_at=migrated)
    plan = plan_migration(
        [LocalUser(user.id, user.email, user.role, None, migrated)],
        [R(user.id, email=user.email, pw=BCRYPT_X)],
        now=NOW,
    )
    assert by_id(plan.updates, user.id).keep_password is False
    chosen = hash_password(PW)
    async with auth_db() as s:
        row = await s.get(User, user.id)
        row.password_hash, row.password_changed_at = chosen, NOW - timedelta(days=1)
        await s.commit()
    async with auth_db() as db:
        await apply_plan(db, plan, NOW)
    assert (await _row(auth_db, user.id)).password_hash == chosen


async def test_apply_keeps_a_password_set_before_the_first_migration_through_every_run(auth_db):
    # Never migrated, password chosen in the new system: kept by the first run AND by the next
    # ones (the account is not stamped, or the stamp would date the password before the migration).
    mine = hash_password(PW)
    user = await seed_user(auth_db, password_hash=mine, password_changed_at=NOW - timedelta(days=3))

    async def run(at: datetime) -> int:
        async with auth_db() as db:
            local = await load_local_users(db)
        plan = plan_migration(
            local, [R(user.id, email=user.email, pw=BCRYPT_X, confirmed=NOW)], now=at
        )
        assert by_id(plan.updates, user.id).keep_password is True
        async with auth_db() as db:
            return await apply_plan(db, plan, at)

    for days in (0, 2, 5):
        assert await run(NOW + timedelta(days=days)) == 1
        row = await _row(auth_db, user.id)
        assert row.password_hash == mine
        assert row.auth_migrated_at is None
        assert aware_utc(row.email_verified_at) == NOW  # the rest is still synchronised


async def test_apply_cannot_overwrite_a_password_set_after_the_plan_of_a_never_migrated_account(
    auth_db,
):
    # The plan saw an account with no local password change and never migrated; the owner chooses
    # a password before `apply` runs.
    user = await seed_user(auth_db)
    plan = plan_migration(
        [LocalUser(user.id, user.email, user.role, None, None)],
        [R(user.id, email=user.email, pw=BCRYPT_X)],
        now=NOW,
    )
    assert by_id(plan.updates, user.id).keep_password is False
    chosen = hash_password(PW)
    async with auth_db() as s:
        row = await s.get(User, user.id)
        row.password_hash, row.password_changed_at = chosen, NOW - timedelta(days=1)
        await s.commit()
    async with auth_db() as db:
        await apply_plan(db, plan, NOW)
    row = await _row(auth_db, user.id)
    assert row.password_hash == chosen and row.auth_migrated_at is None


async def test_a_protected_password_stays_protected_through_later_runs(auth_db):
    # Run 1 migrates; the owner changes the password; runs 2 and 3 must both leave it alone. A run
    # that moved `auth_migrated_at` forward would let the third overwrite what the second kept.
    user = await seed_user(auth_db)
    first_run = NOW - timedelta(days=10)

    async def run(at: datetime) -> None:
        async with auth_db() as db:
            local = await load_local_users(db)
        plan = plan_migration(local, [R(user.id, email=user.email, pw=BCRYPT_X)], now=at)
        async with auth_db() as db:
            await apply_plan(db, plan, at)

    await run(first_run)
    assert (await _row(auth_db, user.id)).password_hash == BCRYPT_X
    chosen = hash_password(PW)
    async with auth_db() as s:
        row = await s.get(User, user.id)
        row.password_hash, row.password_changed_at = chosen, first_run + timedelta(days=2)
        await s.commit()
    await run(first_run + timedelta(days=5))
    await run(first_run + timedelta(days=9))
    row = await _row(auth_db, user.id)
    assert row.password_hash == chosen
    assert aware_utc(row.auth_migrated_at) == first_run  # the date the account moved, kept


async def test_apply_plan_refuses_after_cutover(auth_db, settings_override):
    # In AUTH_MODE=local the local data is the authority: a plan made from Supabase's frozen
    # values would revert roles, e-mails and last sign-ins, so a caller cannot bypass the CLI guard.
    user = await seed_user(auth_db, role=UserRole.VIEWER)
    plan = plan_migration(
        [LocalUser(user.id, user.email, user.role, None, None)],
        [R(user.id, email="reverted@example.test", pw=BCRYPT_X, profile_role="admin")],
        now=NOW,
    )
    settings_override(AUTH_MODE="local")
    async with auth_db() as db:
        with pytest.raises(migration.MigrationError, match="AUTH_MODE=local"):
            await apply_plan(db, plan, NOW)
    row = await _row(auth_db, user.id)
    assert (row.email, row.role, row.password_hash, row.auth_migrated_at) == (
        user.email,
        UserRole.VIEWER,
        None,
        None,
    )


async def test_apply_is_all_or_nothing(auth_db):
    first = await seed_user(auth_db, email="first@example.test")
    second = await seed_user(auth_db, email="second@example.test")
    plan = plan_migration(
        [
            LocalUser(first.id, first.email, first.role, None, None),
            L(second.id, email=second.email),
        ],
        [
            R(first.id, email=first.email, pw=BCRYPT_X),
            R(second.id, email=second.email, pw=BCRYPT_Y),
        ],
        now=NOW,
    )
    # Break the second update (a role the column cannot hold: the driver refuses it when the
    # statement runs, after the first one has been sent).
    plan.updates[1].role = 12345  # type: ignore[assignment]
    async with auth_db() as db:
        with pytest.raises(StatementError):
            await apply_plan(db, plan, NOW)
    assert (await _row(auth_db, first.id)).password_hash is None  # the first one was rolled back
    assert (await _row(auth_db, first.id)).auth_migrated_at is None


async def test_apply_then_login_with_legacy_password(auth_db):
    # A hash exactly as Supabase stores it: Go bcrypt, `$2a$`, cost 10.
    stored = bcrypt.hashpw(PW.encode(), bcrypt.gensalt(10, prefix=b"2a")).decode()
    assert stored.startswith("$2a$10$")
    user = await seed_user(auth_db)
    plan = plan_migration(
        [LocalUser(user.id, user.email, user.role, None, None)],
        [R(user.id, email=user.email, pw=stored, confirmed=NOW)],
        now=NOW,
    )
    async with auth_db() as db:
        await apply_plan(db, plan, NOW)
    row = await _row(auth_db, user.id)
    assert row.password_hash == stored
    ok = verify_password(PW, row.password_hash)
    assert ok.ok and ok.upgraded_hash is not None and ok.upgraded_hash.startswith("$argon2")
    assert not verify_password(PW + "x", row.password_hash).ok


async def test_load_local_users_reads_what_the_plan_needs(auth_db):
    migrated = NOW - timedelta(days=3)
    user = await seed_user(
        auth_db, role=UserRole.ANALYST, password_changed_at=NOW, auth_migrated_at=migrated
    )
    async with auth_db() as db:
        loaded = await load_local_users(db)
    assert [u.id for u in loaded] == [user.id]
    (local,) = loaded
    assert (local.email, local.role) == (user.email, UserRole.ANALYST)
    assert aware_utc(local.password_changed_at) == NOW
    assert aware_utc(local.auth_migrated_at) == migrated


# ---- the command: dry run, report, output -------------------------------------------------------


class _Fetch:
    """Stands in for the read of the hosted database, which holds `users`: answers, as the real
    query does, with the accounts whose id it is given, plus the count of all of them. Records the
    connection string and the ids it got."""

    def __init__(self, users):
        self.users = users
        self.dsns: list[str] = []
        self.ids_asked: list[list[UUID]] = []

    async def __call__(self, dsn: str, local_ids):
        self.dsns.append(dsn)
        self.ids_asked.append(list(local_ids))
        wanted = set(local_ids)
        return migration.RemoteAccounts(
            users=[u for u in self.users if u.id in wanted], total=len(self.users)
        )


SECRET_DSN = "postgresql://postgres.abc:s3cret-Pa55@pooler.example.test:5432/postgres"


async def _command(argv, auth_db, remote, settings_override, tmp_path=None, dsn=SECRET_DSN):
    settings_override(SUPABASE_DB_URL=dsn)
    out, err = StringIO(), StringIO()
    code = await migration.run(
        argv, session_factory=auth_db, fetch=_Fetch(remote), out=out, err=err, now=NOW
    )
    return code, out.getvalue(), err.getvalue()


async def _scenario(auth_db):
    """A world with one of everything; returns the remote accounts."""
    ok = await seed_user(auth_db, email="ok@example.test", role=UserRole.VIEWER)
    no_pw = await seed_user(auth_db, email="nopw@example.test")
    orphan = await seed_user(auth_db, email="orphan@example.test")
    banned = await seed_user(auth_db, email="banned@example.test")
    stranger = uuid4()
    return [
        R(ok.id, email="ok@example.test", pw=BCRYPT_X, profile_role="analyst", confirmed=NOW),
        R(no_pw.id, email="nopw@example.test", pw=""),
        R(
            banned.id,
            email="banned@example.test",
            pw=BCRYPT_Y,
            banned_until=NOW + timedelta(days=2),
        ),
        R(stranger, email="stranger@example.test", pw=BCRYPT_Y),
    ], (ok, no_pw, orphan, banned)


async def test_the_default_is_a_dry_run_that_writes_nothing(auth_db, settings_override):
    remote, (ok, *_rest) = await _scenario(auth_db)
    code, out, err = await _command([], auth_db, remote, settings_override)
    assert code == 0, err
    async with auth_db() as s:
        assert await s.scalar(select(func.count()).where(User.auth_migrated_at.is_not(None))) == 0
        assert await s.scalar(select(func.count()).where(User.password_hash.is_not(None))) == 0
    assert (await _row(auth_db, ok.id)).role == UserRole.VIEWER
    assert "dry run" in out.lower()


async def test_the_summary_is_made_of_counts(auth_db, settings_override):
    remote, _users = await _scenario(auth_db)
    code, out, _err = await _command([], auth_db, remote, settings_override)
    assert code == 0
    lines = {
        line.split(":")[0].strip(): line.split(":", 1)[1].strip()
        for line in out.splitlines()
        if ":" in line
    }
    assert lines["updates"] == "2"  # ok and no_pw; banned is reported, orphan has no Supabase row
    assert lines["out of scope"] == "1"
    for kind, count in (
        ("no_password", 1),
        ("banned", 1),
        ("missing_remote", 1),
        ("email_conflict", 0),
        ("deleted", 0),
        ("role_invalid", 0),
        ("duplicate_email", 0),
    ):
        assert lines[kind] == str(count), (kind, out)


async def test_apply_is_refused_after_cutover_and_writes_nothing(auth_db, settings_override):
    remote, (ok, *_rest) = await _scenario(auth_db)
    settings_override(SUPABASE_DB_URL=SECRET_DSN, AUTH_MODE="local")
    fetch, out, err = _Fetch(remote), StringIO(), StringIO()
    code = await migration.run(
        ["--apply"], session_factory=auth_db, fetch=fetch, out=out, err=err, now=NOW
    )
    assert code == 2
    assert "AUTH_MODE=local" in err.getvalue() and "--apply" in err.getvalue()
    assert "s3cret" not in out.getvalue() + err.getvalue()
    assert out.getvalue() == ""
    assert fetch.dsns == []  # refused before Supabase was even read
    row = await _row(auth_db, ok.id)
    assert (row.role, row.password_hash, row.auth_migrated_at) == (UserRole.VIEWER, None, None)
    async with auth_db() as s:
        assert await s.scalar(select(func.count()).where(User.auth_migrated_at.is_not(None))) == 0


async def test_a_refused_apply_leaves_no_report(auth_db, settings_override, tmp_path):
    remote, _users = await _scenario(auth_db)
    settings_override(SUPABASE_DB_URL=SECRET_DSN, AUTH_MODE="local")
    out, err = StringIO(), StringIO()
    report = tmp_path / "report.json"
    code = await migration.run(
        ["--apply", "--report", str(report)],
        session_factory=auth_db,
        fetch=_Fetch(remote),
        out=out,
        err=err,
        now=NOW,
    )
    assert code == 2 and not report.exists()


async def test_dry_run_and_report_still_work_in_local_mode(auth_db, settings_override, tmp_path):
    remote, (ok, *_rest) = await _scenario(auth_db)
    settings_override(AUTH_MODE="local")
    report = tmp_path / "report.json"
    code, out, err = await _command(["--report", str(report)], auth_db, remote, settings_override)
    assert code == 0, err
    assert "dry run" in out.lower() and "updates: 2" in out
    data = json.loads(report.read_text())
    assert data["mode"] == "dry-run" and data["counts"]["updates"] == 2
    row = await _row(auth_db, ok.id)  # read-only on our side
    assert (row.role, row.password_hash, row.auth_migrated_at) == (UserRole.VIEWER, None, None)


async def test_apply_writes_and_says_how_many(auth_db, settings_override):
    remote, (ok, no_pw, _orphan, banned) = await _scenario(auth_db)
    code, out, err = await _command(["--apply"], auth_db, remote, settings_override)
    assert code == 0, err
    assert "applied" in out.lower() and "2" in out
    row = await _row(auth_db, ok.id)
    assert row.password_hash == BCRYPT_X and row.role == UserRole.ANALYST
    assert (await _row(auth_db, no_pw.id)).auth_migrated_at is not None
    assert (await _row(auth_db, banned.id)).auth_migrated_at is None  # reported, not applied
    # replay: same rows, same answer
    code, out, _err = await _command(["--apply"], auth_db, remote, settings_override)
    assert code == 0 and (await _row(auth_db, ok.id)).password_hash == BCRYPT_X


async def test_report_and_stdout_never_contain_hashes(auth_db, settings_override, tmp_path):
    remote, _users = await _scenario(auth_db)
    report = tmp_path / "report.json"
    code, out, err = await _command(
        ["--apply", "--report", str(report)], auth_db, remote, settings_override
    )
    assert code == 0, err
    text = report.read_text()
    for output in (out, err, text):
        assert "$2" not in output
        assert "$argon2" not in output
        assert BCRYPT_X not in output and BCRYPT_Y not in output
    # the password of the connection string is nowhere either
    for output in (out, err, text):
        assert "s3cret" not in output
    data = json.loads(text)
    assert data["counts"]["updates"] == 2 and data["counts"]["out_of_scope"] == 1
    assert data["counts"]["anomalies"]["no_password"] == 1
    by_kind = {a["kind"]: a for a in data["anomalies"]}
    assert set(by_kind) == {"no_password", "banned", "missing_remote"}
    assert set(by_kind["banned"]) == {"user_id", "kind", "detail"}
    assert "banned@example.test" in by_kind["banned"]["detail"]  # the report carries e-mails
    assert "orphan@example.test" in by_kind["missing_remote"]["detail"]
    # nothing but the planned data: no field of the report is a stored credential
    assert "password_hash" not in text and "encrypted_password" not in text


async def test_the_report_is_private_to_its_owner(auth_db, settings_override, tmp_path):
    remote, _users = await _scenario(auth_db)
    report = tmp_path / "report.json"
    report.write_text("old")
    report.chmod(0o644)
    await _command(["--report", str(report)], auth_db, remote, settings_override)
    assert stat.S_IMODE(os.stat(report).st_mode) == 0o600
    assert json.loads(report.read_text())["mode"] == "dry-run"


async def test_an_unwritable_report_stops_before_any_write(auth_db, settings_override, tmp_path):
    remote, (ok, *_rest) = await _scenario(auth_db)
    missing_dir = tmp_path / "nowhere" / "report.json"
    code, out, err = await _command(
        ["--apply", "--report", str(missing_dir)], auth_db, remote, settings_override
    )
    assert code == 2 and "report" in err.lower()
    assert (await _row(auth_db, ok.id)).auth_migrated_at is None


async def test_the_connection_string_is_passed_on_but_never_printed(auth_db, settings_override):
    remote, _users = await _scenario(auth_db)
    settings_override(SUPABASE_DB_URL=SECRET_DSN)
    fetch, out, err = _Fetch(remote), StringIO(), StringIO()
    await migration.run([], session_factory=auth_db, fetch=fetch, out=out, err=err, now=NOW)
    assert fetch.dsns == [SECRET_DSN]
    assert "s3cret" not in out.getvalue() + err.getvalue()


async def test_only_the_local_accounts_are_read_from_supabase(auth_db, settings_override):
    """The hosted project is shared with another application: its other accounts (and their
    hashes) are never fetched, only counted."""
    remote, users = await _scenario(auth_db)
    settings_override(SUPABASE_DB_URL=SECRET_DSN)
    fetch, out, err = _Fetch(remote), StringIO(), StringIO()
    code = await migration.run([], session_factory=auth_db, fetch=fetch, out=out, err=err, now=NOW)
    assert code == 0, err.getvalue()
    [asked] = fetch.ids_asked
    assert sorted(asked, key=str) == sorted((u.id for u in users), key=str)
    assert "out of scope: 1" in out.getvalue()  # the stranger, counted, never read


async def test_a_database_failure_prints_one_line_without_the_row(auth_db, settings_override):
    """asyncpg appends `DETAIL:` (which can quote the failing row) to its messages, and a
    SQLAlchemy error carries the statement and its parameters: only the class, the SQLSTATE and
    the first line of the message are printed."""
    from sqlalchemy.exc import IntegrityError

    class UniqueViolation(Exception):
        sqlstate = "23505"

    async def failing_apply(*_args, **_kwargs):
        raise IntegrityError(
            "UPDATE users SET password_hash=$1",
            (BCRYPT_X,),
            UniqueViolation(
                'duplicate key value violates unique constraint "users_email_key"\n'
                "DETAIL:  Key (email)=(row-owner@example.test) already exists."
            ),
        )

    remote, _users = await _scenario(auth_db)
    settings_override(SUPABASE_DB_URL=SECRET_DSN)
    out, err = StringIO(), StringIO()
    original = migration.apply_plan
    migration.apply_plan = failing_apply
    try:
        code = await migration.run(
            ["--apply"], session_factory=auth_db, fetch=_Fetch(remote), out=out, err=err, now=NOW
        )
    finally:
        migration.apply_plan = original
    assert code == 1
    assert err.getvalue() == (
        "error: IntegrityError [SQLSTATE 23505]: duplicate key value violates unique "
        'constraint "users_email_key"\n'
    )


def test_describe_keeps_the_first_line_and_drops_the_connection_string():
    class InsufficientPrivilege(Exception):
        sqlstate = "42501"

    text = migration._describe(
        InsufficientPrivilege(
            f"permission denied with {SECRET_DSN}\nDETAIL: (row-owner@example.test, {BCRYPT_X})"
        ),
        migration._secrets_of(SECRET_DSN),
    )
    assert text == "InsufficientPrivilege [SQLSTATE 42501]: permission denied with ***"
    assert migration._describe(RuntimeError(""), ()) == "RuntimeError"
    assert migration._describe(ValueError("bad\nworse"), ()) == "ValueError: bad"


async def test_a_failure_of_the_hosted_read_is_explained_without_the_connection_string(
    auth_db, settings_override
):
    async def unreachable(dsn, _local_ids):
        raise ConnectionError(f"could not connect with {dsn}")

    settings_override(SUPABASE_DB_URL=SECRET_DSN)
    out, err = StringIO(), StringIO()
    code = await migration.run(
        ["--apply"], session_factory=auth_db, fetch=unreachable, out=out, err=err, now=NOW
    )
    assert code == 1
    assert "ConnectionError" in err.getvalue()
    assert "s3cret" not in out.getvalue() + err.getvalue()
    assert "Pa55" not in out.getvalue() + err.getvalue()


async def test_a_missing_connection_string_is_a_configuration_error(auth_db, settings_override):
    settings_override(SUPABASE_DB_URL=None)
    out, err = StringIO(), StringIO()
    code = await migration.run(
        [], session_factory=auth_db, fetch=_Fetch([]), out=out, err=err, now=NOW
    )
    assert code == 2 and "SUPABASE_DB_URL" in err.getvalue() and out.getvalue() == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["--bogus"],
        ["--report"],
        ["--send-resets", "--apply"],
        ["--send-resets", "--report", "x.json"],
    ],
)
def test_usage_errors_return_2(argv, capsys):
    assert migration.main(argv) == 2
    assert "$2" not in capsys.readouterr().out


def test_help_is_not_an_error(capsys):
    assert migration.main(["--help"]) == 0
    assert "--apply" in capsys.readouterr().out


def test_main_without_a_connection_string_returns_2(settings_override, capsys):
    settings_override(SUPABASE_DB_URL=None)
    assert migration.main([]) == 2
    assert "SUPABASE_DB_URL" in capsys.readouterr().err


def test_main_refuses_apply_in_local_mode(settings_override, capsys):
    # Before any connection: the address is unreachable and would fail with exit 1 otherwise.
    settings_override(AUTH_MODE="local", SUPABASE_DB_URL="postgresql://u:p@127.0.0.1:1/db")
    assert migration.main(["--apply"]) == 2
    captured = capsys.readouterr()
    assert "AUTH_MODE=local" in captured.err and captured.out == ""


def test_main_refuses_resets_outside_local_mode(settings_override, capsys):
    settings_override(AUTH_MODE="supabase")
    assert migration.main(["--send-resets"]) == 2
    assert "local" in capsys.readouterr().err


# ---- the read of the hosted database ------------------------------------------------------------


class _FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.statements.append("BEGIN")

    async def __aexit__(self, *exc):
        self.conn.statements.append("ROLLBACK" if exc[0] else "COMMIT")


class _FakeConnection:
    def __init__(self, rows, total=None):
        self.rows = rows
        self.total = len(rows) if total is None else total
        self.statements: list[str] = []
        self.arguments: list[tuple] = []
        self.closed = False

    def transaction(self, **_options):
        return _FakeTransaction(self)

    async def execute(self, sql, *_args):
        self.statements.append(" ".join(sql.split()))

    async def fetch(self, sql, *args):
        self.statements.append(" ".join(sql.split()))
        self.arguments.append(args)
        return self.rows

    async def fetchval(self, sql, *args):
        self.statements.append(" ".join(sql.split()))
        self.arguments.append(args)
        return self.total

    async def close(self):
        self.closed = True


def _row_of(user_id, **overrides):
    values = dict(
        id=user_id,
        email="remote@example.test",
        encrypted_password=BCRYPT_X,
        email_confirmed_at=NOW,
        last_sign_in_at=NOW,
        banned_until=None,
        deleted_at=None,
        role="Analyst",
    )
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("given", "passed"),
    [
        (SECRET_DSN, SECRET_DSN),
        (
            "postgresql+asyncpg://u:p@h.example.test:5432/db",
            "postgresql://u:p@h.example.test:5432/db",
        ),
        ("postgres+asyncpg://u:p@h.example.test/db", "postgres://u:p@h.example.test/db"),
    ],
)
async def test_the_hosted_read_is_read_only_and_accepts_both_dsn_forms(monkeypatch, given, passed):
    uid, other_local = uuid4(), uuid4()
    conn = _FakeConnection([_row_of(uid)], total=40)
    seen = []

    async def connect(dsn, **_options):
        seen.append(dsn)
        return conn

    monkeypatch.setattr(migration.asyncpg, "connect", connect)
    accounts = await fetch_remote_users(given, [uid, other_local])
    assert seen == [passed]
    assert conn.closed
    assert conn.statements[0] == "BEGIN"
    assert conn.statements[1] == "SET TRANSACTION READ ONLY"
    assert conn.statements[-1] == "COMMIT"
    select_sql = conn.statements[2].lower()
    assert select_sql.startswith("select")
    assert "from auth.users" in select_sql and "left join public.profiles" in select_sql
    # Only the local accounts are fetched; the others are counted, never read.
    assert select_sql.endswith("where u.id = any($1::uuid[])")
    assert conn.arguments[0] == ([uid, other_local],)
    assert conn.statements[3].lower() == "select count(*) from auth.users"
    assert accounts.total == 40
    assert not any(
        word in " ".join(conn.statements).lower() for word in ("insert ", "update ", "delete ")
    )
    (user,) = accounts.users
    assert user == RemoteUser(
        id=uid,
        email="remote@example.test",
        encrypted_password=BCRYPT_X,
        email_confirmed_at=NOW,
        last_sign_in_at=NOW,
        banned_until=None,
        deleted_at=None,
        profile_role="Analyst",
    )


async def test_the_hosted_read_closes_the_connection_on_failure(monkeypatch):
    conn = _FakeConnection([])

    async def boom(sql, *_args):
        raise RuntimeError("query failed")

    conn.fetch = boom

    async def connect(dsn, **_options):
        return conn

    monkeypatch.setattr(migration.asyncpg, "connect", connect)
    with pytest.raises(RuntimeError):
        await fetch_remote_users(SECRET_DSN, [uuid4()])
    assert conn.closed and conn.statements[-1] == "ROLLBACK"


async def test_the_hosted_read_copes_with_accounts_that_have_no_e_mail_and_sorts_them(monkeypatch):
    low, high = UUID(int=1), UUID(int=2)
    conn = _FakeConnection([_row_of(high, email=None), _row_of(low, role=None)])

    async def connect(dsn, **_options):
        return conn

    monkeypatch.setattr(migration.asyncpg, "connect", connect)
    users = (await fetch_remote_users(SECRET_DSN, [high, low])).users
    assert [u.id for u in users] == [low, high]
    assert users[1].email == "" and users[0].profile_role is None


TEST_PG = "postgresql://postgres:postgres@localhost:55432/genolens_test"


@asynccontextmanager
async def _throwaway_hosted_schema():
    """`auth.users` and `public.profiles` as Supabase has them (minimal), in the throwaway
    Postgres of the test setup. Skips when it is unreachable; refuses to touch anything that
    already exists, and removes exactly what it created."""
    import asyncpg

    try:
        conn = await asyncpg.connect(TEST_PG, timeout=3)
    except Exception:
        pytest.skip("the throwaway Postgres (localhost:55432) is not reachable")
    created = False
    try:
        busy = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'auth') "
            "OR to_regclass('public.profiles') IS NOT NULL"
        )
        if busy:
            pytest.skip("the throwaway Postgres already has an auth schema or public.profiles")
        created = True
        await conn.execute(
            """
            CREATE SCHEMA auth;
            CREATE TABLE auth.users (
                id uuid PRIMARY KEY, email text, encrypted_password text,
                email_confirmed_at timestamptz, last_sign_in_at timestamptz,
                banned_until timestamptz, deleted_at timestamptz
            );
            CREATE TABLE public.profiles (id uuid PRIMARY KEY, role text);
            """
        )
        yield conn
    finally:
        if created:
            await conn.execute(
                "DROP TABLE IF EXISTS public.profiles; DROP SCHEMA IF EXISTS auth CASCADE;"
            )
        await conn.close()


async def test_the_hosted_read_against_a_real_postgres():
    with_profile, without_profile, stranger = uuid4(), uuid4(), uuid4()
    async with _throwaway_hosted_schema() as conn:
        await conn.execute(
            "INSERT INTO auth.users VALUES ($1, 'a@example.test', $2, $3, $3, NULL, NULL), "
            "($4, 'b@example.test', NULL, NULL, NULL, $5, $3), "
            "($6, 'other-app@example.test', $2, NULL, NULL, NULL, NULL)",
            with_profile,
            BCRYPT_X,
            NOW,
            without_profile,
            NOW + timedelta(days=1),
            stranger,
        )
        await conn.execute("INSERT INTO public.profiles VALUES ($1, 'analyst')", with_profile)
        accounts = await fetch_remote_users(
            TEST_PG.replace("postgresql://", "postgresql+asyncpg://"),
            [with_profile, without_profile, uuid4()],  # the last one has no Supabase account
        )
        assert accounts.total == 3
        users = {u.id: u for u in accounts.users}
        assert set(users) == {with_profile, without_profile}  # the stranger is never read
        assert users[with_profile] == RemoteUser(
            id=UUID(str(with_profile)),
            email="a@example.test",
            encrypted_password=BCRYPT_X,
            email_confirmed_at=NOW,
            last_sign_in_at=NOW,
            banned_until=None,
            deleted_at=None,
            profile_role="analyst",
        )
        assert users[without_profile].profile_role is None  # left join
        assert users[without_profile].deleted_at == NOW
        assert users[without_profile].banned_until == NOW + timedelta(days=1)
        # the read really is read-only: the same statement shape cannot write
        assert await conn.fetchval("SELECT count(*) FROM auth.users") == 3


async def test_apply_against_a_real_postgres():
    """The SQL of `apply_plan` (the guarded password write, the once-set migration date) on
    Postgres, which types its parameters where SQLite does not. The `users` table lives in a
    schema of its own, dropped afterwards."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.models.models import SubscriptionPlan

    schema = f"mig_test_{uuid4().hex[:8]}"
    base = create_async_engine(TEST_PG.replace("postgresql://", "postgresql+asyncpg://"))
    try:
        async with base.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    except Exception:
        await base.dispose()
        pytest.skip("the throwaway Postgres (localhost:55432) is not reachable")
    try:
        engine = base.execution_options(schema_translate_map={None: schema})
        async with engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: User.__table__.create(sync_conn))
        sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        fresh, changed, local_only = uuid4(), uuid4(), uuid4()
        mine = hash_password(PW)
        migrated = NOW - timedelta(days=5)
        for uid, email, extra in (
            (fresh, "fresh@example.test", {}),
            (
                changed,
                "changed@example.test",
                dict(
                    password_hash=mine,
                    password_changed_at=NOW - timedelta(days=1),
                    auth_migrated_at=migrated,
                ),
            ),
            (  # a password chosen in the new system before the account was ever migrated
                local_only,
                "local@example.test",
                dict(password_hash=mine, password_changed_at=NOW - timedelta(days=2)),
            ),
        ):
            async with sessions() as s:
                s.add(
                    User(
                        id=uid,
                        email=email,
                        role=UserRole.VIEWER,
                        subscription_plan=SubscriptionPlan.TEAM,
                        status=UserStatus.ACTIVE,
                        **extra,
                    )
                )
                await s.commit()
        remote = [
            R(
                fresh,
                email="renamed@example.test",
                pw=BCRYPT_X,
                confirmed=NOW,
                profile_role="analyst",
            ),
            R(changed, email="changed@example.test", pw=BCRYPT_Y, confirmed=NOW),
            R(local_only, email="local@example.test", pw=BCRYPT_Y, confirmed=NOW),
        ]
        for run_at in (NOW, NOW + timedelta(days=3)):  # and a replay, later
            async with sessions() as db:
                plan = plan_migration(await load_local_users(db), remote, run_at)
                assert await apply_plan(db, plan, run_at) == 3
        async with sessions() as s:
            first, second = await s.get(User, fresh), await s.get(User, changed)
            third = await s.get(User, local_only)
        assert (first.email, first.role, first.password_hash) == (
            "renamed@example.test",
            UserRole.ANALYST,
            BCRYPT_X,
        )
        assert aware_utc(first.auth_migrated_at) == NOW  # set once, not moved by the replay
        assert second.password_hash == mine  # chosen after the migration: kept
        assert aware_utc(second.auth_migrated_at) == migrated
        assert third.password_hash == mine and third.auth_migrated_at is None
    finally:
        async with base.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await base.dispose()


# ---- reset links --------------------------------------------------------------------------------


async def test_reset_links_refuse_outside_local_mode(auth_db, settings_override, smtp_capture):
    settings_override(AUTH_MODE="supabase")
    await seed_user(auth_db)
    async with auth_db() as db:
        with pytest.raises(migration.MigrationError, match="local"):
            await send_reset_links(db)
    assert smtp_capture.messages == []


async def _migrated_user(auth_db, **fields) -> User:
    """An account the migration moved (`auth_migrated_at` set), without a password unless given."""
    return await seed_user(auth_db, auth_migrated_at=NOW - timedelta(days=1), **fields)


async def _reset_tokens(auth_db) -> list[AuthOneTimeToken]:
    async with auth_db() as s:
        return list(
            (
                await s.scalars(
                    select(AuthOneTimeToken).where(
                        AuthOneTimeToken.purpose == OneTimeTokenPurpose.PASSWORD_RESET
                    )
                )
            ).all()
        )


async def test_reset_links_go_to_the_accounts_without_a_password(
    auth_db, settings_override, smtp_capture
):
    settings_override(AUTH_MODE="local", APP_URL="https://app.example.test")
    needs_1 = await _migrated_user(auth_db, email="n1@example.test")
    needs_2 = await _migrated_user(auth_db, email="n2@example.test", status=UserStatus.PENDING)
    has_password = await _migrated_user(
        auth_db, email="has@example.test", password_hash=hash_password(PW)
    )
    suspended = await _migrated_user(
        auth_db, email="susp@example.test", status=UserStatus.SUSPENDED
    )
    cancelled = await _migrated_user(
        auth_db, email="canc@example.test", status=UserStatus.CANCELLED
    )
    # Never migrated (an invitation of the new system, say): not this command's business.
    never_migrated = await seed_user(auth_db, email="invited@example.test")
    async with auth_db() as db:
        assert await send_reset_links(db) == 2
    assert sorted(m.to for m in smtp_capture.messages) == ["n1@example.test", "n2@example.test"]
    async with auth_db() as s:
        owners = {t.user_id for t in await _reset_tokens(auth_db)}
        assert owners == {needs_1.id, needs_2.id}
        assert {has_password.id, suspended.id, cancelled.id, never_migrated.id}.isdisjoint(owners)
        events = (await s.scalars(select(AuthEvent))).all()
        assert {e.event_type for e in events} == {AuthEventType.PASSWORD_RESET_REQUESTED.value}
        assert {e.user_id for e in events} == {needs_1.id, needs_2.id}
        assert all((e.details or {}).get("source") == "supabase_migration" for e in events)
    for message in smtp_capture.messages:
        assert "https://app.example.test/auth/reset#token=" in message.text


async def test_reset_links_skip_an_account_that_holds_a_live_link(
    auth_db, settings_override, smtp_capture
):
    """An account with an unused, unexpired reset link (from an earlier run, or one it asked for
    itself) is not mailed again; an expired or spent link does not count."""
    from app.auth.clock import now_utc
    from app.auth.one_time_tokens import consume_token, issue_token

    settings_override(AUTH_MODE="local")
    live = await _migrated_user(auth_db, email="live@example.test")
    expired = await _migrated_user(auth_db, email="expired@example.test")
    spent = await _migrated_user(auth_db, email="spent@example.test")
    async with auth_db() as db:
        await issue_token(db, live.id, OneTimeTokenPurpose.PASSWORD_RESET)
        await issue_token(
            db, expired.id, OneTimeTokenPurpose.PASSWORD_RESET, now=now_utc() - timedelta(hours=2)
        )
        raw = await issue_token(db, spent.id, OneTimeTokenPurpose.PASSWORD_RESET)
        await consume_token(db, raw, OneTimeTokenPurpose.PASSWORD_RESET)
        await db.commit()

    async with auth_db() as db:
        assert await send_reset_links(db) == 2
    assert sorted(m.to for m in smtp_capture.messages) == [
        "expired@example.test",
        "spent@example.test",
    ]


async def test_a_failed_e_mail_does_not_stop_the_others_and_a_rerun_mails_only_it(
    auth_db, settings_override, smtp_capture, caplog
):
    settings_override(AUTH_MODE="local", APP_URL="https://app.example.test")
    users = [await _migrated_user(auth_db, email=f"u{i}@example.test") for i in range(3)]
    unlucky = next(u.id for u in users if u.email == "u1@example.test")
    sent_to = []

    async def flaky(message, **_options):
        if message["To"] == "u1@example.test":
            raise OSError("connection reset")
        sent_to.append(message["To"])

    import app.services.email_service as email_service

    original = email_service.aiosmtplib.send
    email_service.aiosmtplib.send = flaky
    try:
        failed: list[UUID] = []
        async with auth_db() as db:
            assert await send_reset_links(db, failed=failed) == 2
    finally:
        email_service.aiosmtplib.send = original
    assert sorted(sent_to) == ["u0@example.test", "u2@example.test"]
    assert failed == [unlucky]
    # The link of the e-mail that did not leave is withdrawn (spent), so that it is not mistaken
    # for a live link; the others stay live. No token ever reaches the logs.
    tokens = {t.user_id: t for t in await _reset_tokens(auth_db)}
    assert len(tokens) == 3
    assert tokens[unlucky].used_at is not None
    assert all(t.used_at is None for uid, t in tokens.items() if uid != unlucky)
    assert "token=" not in caplog.text

    # SMTP answers again: the rerun mails the failure only.
    async with auth_db() as db:
        assert await send_reset_links(db) == 1
    assert [m.to for m in smtp_capture.messages] == ["u1@example.test"]


async def test_smtp_not_configured_stops_the_run_and_withdraws_the_link(
    auth_db, settings_override, smtp_capture
):
    settings_override(AUTH_MODE="local", SMTP_HOST=None)
    first = await _migrated_user(auth_db, email="a1@example.test")
    await _migrated_user(auth_db, email="a2@example.test")
    async with auth_db() as db:
        with pytest.raises(migration.MigrationError, match="SMTP"):
            await send_reset_links(db)
    [token] = await _reset_tokens(auth_db)  # only the first was tried
    assert token.user_id == first.id and token.used_at is not None

    settings_override(SMTP_HOST="smtp.example.test")
    async with auth_db() as db:
        assert await send_reset_links(db) == 2  # nobody was left out
    assert sorted(m.to for m in smtp_capture.messages) == ["a1@example.test", "a2@example.test"]


async def test_the_command_sends_resets_and_reports_counts(
    auth_db, settings_override, smtp_capture
):
    settings_override(AUTH_MODE="local", SUPABASE_DB_URL=None)
    await _migrated_user(auth_db, email="n1@example.test")
    await _migrated_user(auth_db, email="n2@example.test")
    out, err = StringIO(), StringIO()
    code = await migration.run(
        ["--send-resets"], session_factory=auth_db, fetch=_Fetch([]), out=out, err=err, now=NOW
    )
    assert code == 0, err.getvalue()  # no connection string needed: nothing is read from Supabase
    assert "2" in out.getvalue() and len(smtp_capture.messages) == 2
    assert "example.test" not in out.getvalue()  # counts, not addresses


async def test_the_command_exits_1_when_some_resets_could_not_be_sent(
    auth_db, settings_override, smtp_capture
):
    settings_override(AUTH_MODE="local")
    await _migrated_user(auth_db, email="n1@example.test")
    smtp_capture.fail_with = OSError("unreachable")
    out, err = StringIO(), StringIO()
    code = await migration.run(
        ["--send-resets"], session_factory=auth_db, fetch=_Fetch([]), out=out, err=err, now=NOW
    )
    assert code == 1
    assert "1" in out.getvalue() and "failed" in out.getvalue().lower()
    # The advice says what a rerun does: mail the failures only.
    assert "run --send-resets again" in err.getvalue()
    assert "only the accounts whose e-mail did not leave" in err.getvalue()


async def test_the_command_refuses_resets_in_supabase_mode(
    auth_db, settings_override, smtp_capture
):
    settings_override(AUTH_MODE="supabase")
    await seed_user(auth_db)
    out, err = StringIO(), StringIO()
    code = await migration.run(
        ["--send-resets"], session_factory=auth_db, fetch=_Fetch([]), out=out, err=err, now=NOW
    )
    assert code == 2 and "local" in err.getvalue() and smtp_capture.messages == []
