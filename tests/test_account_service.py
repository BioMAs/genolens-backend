"""
Tests for AccountService — account invitation and status lifecycle.

These cover the two defects that were deliberately not carried over when the
service was ported from feature/phase1-account-management:

1. The original created the local User row with a fresh uuid4(), unlinked to
   Supabase Auth, so an invited person could never log in. The port must key the
   row on the Supabase auth id.
2. The original sent a hand-rolled email through a helper absent from main. The
   port delegates to Supabase's invite endpoint.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.models.models import SubscriptionPlan, User, UserRole, UserStatus
from app.services.account_service import AccountService

AUTH_ID = "11111111-2222-3333-4444-555555555555"
ADMIN_ID = uuid4()


def _no_existing_user(db):
    """Make db.execute(...).scalar_one_or_none() return None."""
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=result)


def _existing_user(db, user):
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=user)
    db.execute = AsyncMock(return_value=result)


def make_user(status=UserStatus.PENDING, email="invitee@example.com", user_id=None):
    u = User()
    u.id = user_id or UUID(AUTH_ID)
    u.email = email
    u.full_name = "Invitee"
    u.role = UserRole.USER
    u.subscription_plan = SubscriptionPlan.STARTER
    u.status = status
    return u


# ── invite_user ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invite_keys_the_local_row_on_the_supabase_auth_id(mock_db):
    """The regression that mattered: never a fresh uuid4."""
    _no_existing_user(mock_db)
    with patch.object(
        AccountService, "_send_supabase_invite", AsyncMock(return_value={"id": AUTH_ID})
    ):
        service = AccountService(mock_db)
        user = await service.invite_user(
            email="invitee@example.com",
            full_name="Invitee",
            plan=SubscriptionPlan.TEAM,
            invited_by_admin_id=ADMIN_ID,
        )

    assert user.id == UUID(AUTH_ID)
    mock_db.add.assert_called_once()
    assert mock_db.add.call_args[0][0] is user


@pytest.mark.asyncio
async def test_invite_creates_a_pending_account_with_the_requested_plan(mock_db):
    _no_existing_user(mock_db)
    with patch.object(
        AccountService, "_send_supabase_invite", AsyncMock(return_value={"id": AUTH_ID})
    ):
        user = await AccountService(mock_db).invite_user(
            email="invitee@example.com",
            full_name="Invitee",
            plan=SubscriptionPlan.TEAM,
            invited_by_admin_id=ADMIN_ID,
        )

    assert user.status is UserStatus.PENDING
    assert user.subscription_plan is SubscriptionPlan.TEAM
    assert user.role is UserRole.USER
    mock_db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_invite_stores_subscription_ends_at_as_iso_string(mock_db):
    """The column is String(50), not a timestamp."""
    _no_existing_user(mock_db)
    ends = datetime(2027, 1, 31, 12, 0, tzinfo=timezone.utc)
    with patch.object(
        AccountService, "_send_supabase_invite", AsyncMock(return_value={"id": AUTH_ID})
    ):
        user = await AccountService(mock_db).invite_user(
            email="invitee@example.com",
            full_name=None,
            plan=SubscriptionPlan.STARTER,
            invited_by_admin_id=ADMIN_ID,
            subscription_ends_at=ends,
        )

    assert isinstance(user.subscription_ends_at, str)
    assert user.subscription_ends_at == ends.isoformat()


@pytest.mark.asyncio
async def test_invite_leaves_subscription_ends_at_none_when_not_given(mock_db):
    _no_existing_user(mock_db)
    with patch.object(
        AccountService, "_send_supabase_invite", AsyncMock(return_value={"id": AUTH_ID})
    ):
        user = await AccountService(mock_db).invite_user(
            email="invitee@example.com",
            full_name=None,
            plan=SubscriptionPlan.STARTER,
            invited_by_admin_id=ADMIN_ID,
        )
    assert user.subscription_ends_at is None


@pytest.mark.asyncio
async def test_invite_rejects_a_duplicate_email_before_calling_supabase(mock_db):
    _existing_user(mock_db, make_user())
    invite = AsyncMock()
    with patch.object(AccountService, "_send_supabase_invite", invite):
        with pytest.raises(ValueError, match="already exists"):
            await AccountService(mock_db).invite_user(
                email="invitee@example.com",
                full_name=None,
                plan=SubscriptionPlan.STARTER,
                invited_by_admin_id=ADMIN_ID,
            )
    invite.assert_not_awaited()
    mock_db.add.assert_not_called()


@pytest.mark.asyncio
async def test_invite_does_not_persist_when_supabase_returns_no_id(mock_db):
    """A local row with no auth account is exactly the state to avoid."""
    _no_existing_user(mock_db)
    with patch.object(
        AccountService, "_send_supabase_invite", AsyncMock(return_value={})
    ):
        with pytest.raises(RuntimeError, match="no user id"):
            await AccountService(mock_db).invite_user(
                email="invitee@example.com",
                full_name=None,
                plan=SubscriptionPlan.STARTER,
                invited_by_admin_id=ADMIN_ID,
            )
    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_invite_does_not_persist_when_supabase_rejects(mock_db):
    _no_existing_user(mock_db)
    with patch.object(
        AccountService,
        "_send_supabase_invite",
        AsyncMock(side_effect=RuntimeError("Supabase invite failed (422): already registered")),
    ):
        with pytest.raises(RuntimeError, match="422"):
            await AccountService(mock_db).invite_user(
                email="invitee@example.com",
                full_name=None,
                plan=SubscriptionPlan.STARTER,
                invited_by_admin_id=ADMIN_ID,
            )
    mock_db.add.assert_not_called()


# ── resend_invitation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resend_invitation_reinvites_a_pending_user(mock_db):
    _existing_user(mock_db, make_user(status=UserStatus.PENDING))
    invite = AsyncMock(return_value={"id": AUTH_ID})
    with patch.object(AccountService, "_send_supabase_invite", invite):
        user = await AccountService(mock_db).resend_invitation(UUID(AUTH_ID))
    invite.assert_awaited_once()
    assert user.email == "invitee@example.com"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [UserStatus.ACTIVE, UserStatus.SUSPENDED, UserStatus.CANCELLED]
)
async def test_resend_invitation_refuses_non_pending_users(mock_db, status):
    _existing_user(mock_db, make_user(status=status))
    invite = AsyncMock()
    with patch.object(AccountService, "_send_supabase_invite", invite):
        with pytest.raises(ValueError, match="pending"):
            await AccountService(mock_db).resend_invitation(UUID(AUTH_ID))
    invite.assert_not_awaited()


@pytest.mark.asyncio
async def test_resend_invitation_raises_not_found_for_unknown_user(mock_db):
    _no_existing_user(mock_db)
    with pytest.raises(ValueError, match="not found"):
        await AccountService(mock_db).resend_invitation(uuid4())


# ── status transitions ────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,expected",
    [
        ("activate_user", UserStatus.ACTIVE),
        ("suspend_user", UserStatus.SUSPENDED),
        ("cancel_user", UserStatus.CANCELLED),
    ],
)
async def test_status_transitions_persist(mock_db, method, expected):
    user = make_user(status=UserStatus.PENDING)
    _existing_user(mock_db, user)
    service = AccountService(mock_db)
    result = await getattr(service, method)(UUID(AUTH_ID))
    assert result.status is expected
    mock_db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_set_status_raises_not_found_for_unknown_user(mock_db):
    _no_existing_user(mock_db)
    with pytest.raises(ValueError, match="not found"):
        await AccountService(mock_db).set_status(uuid4(), UserStatus.ACTIVE)
    mock_db.commit.assert_not_awaited()
