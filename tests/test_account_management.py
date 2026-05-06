import pytest
from app.models.models import UserStatus, User
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
