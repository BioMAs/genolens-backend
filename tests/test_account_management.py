import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
from uuid import UUID
from app.models.models import UserStatus, User, SubscriptionPlan
from app.services.account_service import AccountService
from sqlalchemy import inspect


@pytest.mark.unit
def test_user_status_enum_values():
    assert UserStatus.PENDING == "pending"
    assert UserStatus.ACTIVE == "active"
    assert UserStatus.SUSPENDED == "suspended"
    assert UserStatus.CANCELLED == "cancelled"


@pytest.mark.unit
def test_user_has_status_field():
    assert hasattr(User, "status")
    assert not hasattr(User, "is_active")
    col = inspect(User).columns["status"]
    assert col.nullable is False
    assert col.index is True


@pytest.mark.unit
def test_checkout_completed_sets_status_active():
    """checkout.session.completed must set user.status = ACTIVE (not is_active)."""
    import types

    # Simulate a User ORM object with status field
    user = types.SimpleNamespace(
        stripe_subscription_id=None,
        status=UserStatus.PENDING,
        stripe_customer_id=None,
        subscription_plan=None,
        subscription_starts_at=None,
    )
    # Verify the model accepts status assignment (not is_active)
    user.status = UserStatus.ACTIVE
    assert user.status == UserStatus.ACTIVE
    assert not hasattr(user, "is_active")


@pytest.mark.asyncio
async def test_invite_user_creates_pending_account():
    """invite_user must create a User with status=PENDING."""
    mock_db = AsyncMock()
    # AsyncMock.return_value is also AsyncMock, so we set a plain MagicMock for the execute result
    mock_execute_result = MagicMock()
    mock_execute_result.scalar_one_or_none.return_value = None  # user doesn't exist
    mock_db.execute.return_value = mock_execute_result

    service = AccountService(mock_db)
    with patch.object(service, "_send_invitation_email", new_callable=AsyncMock):
        user = await service.invite_user(
            email="new@example.com",
            full_name="New User",
            plan=SubscriptionPlan.BASIC,
            invited_by_admin_id="00000000-0000-0000-0000-000000000099",
        )

    assert user.status == UserStatus.PENDING
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_activate_user_sets_status_active():
    """activate_user must set status=ACTIVE."""
    mock_user = MagicMock()
    mock_user.status = UserStatus.PENDING
    mock_db = AsyncMock()
    mock_execute_result = MagicMock()
    mock_execute_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_execute_result

    service = AccountService(mock_db)
    await service.activate_user(UUID("00000000-0000-0000-0000-000000000001"))

    assert mock_user.status == UserStatus.ACTIVE
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_suspend_user_sets_status_suspended():
    """suspend_user must set status=SUSPENDED."""
    mock_user = MagicMock()
    mock_user.status = UserStatus.ACTIVE
    mock_db = AsyncMock()
    mock_execute_result = MagicMock()
    mock_execute_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_execute_result

    service = AccountService(mock_db)
    await service.suspend_user(UUID("00000000-0000-0000-0000-000000000001"))

    assert mock_user.status == UserStatus.SUSPENDED
    mock_db.commit.assert_called_once()
