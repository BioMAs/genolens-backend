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
