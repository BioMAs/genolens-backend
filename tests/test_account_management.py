import pytest
from app.models.models import UserStatus, User


def test_user_status_enum_values():
    assert UserStatus.PENDING == "pending"
    assert UserStatus.ACTIVE == "active"
    assert UserStatus.SUSPENDED == "suspended"
    assert UserStatus.CANCELLED == "cancelled"


def test_user_has_status_field():
    # Verify the column exists on the mapped class
    assert hasattr(User, "status")
    assert not hasattr(User, "is_active")  # old field must be gone
