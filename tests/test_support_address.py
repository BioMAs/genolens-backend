"""
Every place that tells a reader whom to contact names the `SUPPORT_EMAIL` setting, read at the
moment it speaks: the expiry warning e-mail, and the refusals of an inactive or expired account,
of an expired licence, of an exhausted AI quota and of a demo account's analysis quota. (The
security e-mails of the local authentication are covered in `tests/test_auth_emails.py`.)

An instance that sets SUPPORT_EMAIL must not show the default address anywhere; one that does not
set it keeps showing the default, exactly as before.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.core.config import Settings
from app.models.models import SubscriptionPlan, User, UserRole, UserStatus

CONFIGURED = "help-desk@example.test"
DEFAULT = Settings.model_fields["SUPPORT_EMAIL"].default


@pytest.fixture
def support_address(settings_override):
    settings_override(SUPPORT_EMAIL=CONFIGURED)


def a_user(**fields) -> User:
    user = User()
    user.id = uuid4()
    user.email = "someone@example.test"
    user.role = UserRole.USER
    user.subscription_plan = SubscriptionPlan.TEAM
    user.status = UserStatus.ACTIVE
    user.subscription_ends_at = None
    user.is_demo = False
    user.ai_requests_limit = None
    user.ai_requests_used = 0
    for name, value in fields.items():
        setattr(user, name, value)
    return user


def a_request(method: str = "POST", path: str = "/api/v1/projects/") -> Request:
    return Request(
        {"type": "http", "method": method, "path": path, "headers": [], "query_string": b""}
    )


def says_the_configured_address(text: str) -> bool:
    return CONFIGURED in text and DEFAULT not in text


def test_the_default_is_unchanged():
    assert DEFAULT == "support@scilicium.com"


def test_the_expiry_warning_names_the_configured_address(support_address):
    from app.services.email_service import _expiry_warning_html, _expiry_warning_text

    text = _expiry_warning_text(3, "31 December 2026", "https://example.test/pricing")
    html = _expiry_warning_html(3, "31 December 2026", "https://example.test/pricing")

    assert says_the_configured_address(text) and says_the_configured_address(html)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"status": UserStatus.SUSPENDED}, id="account_inactive"),
        pytest.param(
            {"subscription_ends_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
            id="subscription_expired",
        ),
    ],
)
async def test_the_account_refusals_name_the_configured_address(support_address, fields):
    from app.api.deps.account_state import require_active_account

    with pytest.raises(HTTPException) as refused:
        await require_active_account(a_request(), a_user(**fields))

    assert says_the_configured_address(refused.value.detail["message"])


async def test_the_licence_refusal_names_the_configured_address(support_address, monkeypatch):
    import app.api.deps.license as license_deps
    from app.core.license import LicenseStatus

    monkeypatch.setattr(license_deps, "get_license_status", lambda: LicenseStatus(valid=False))

    with pytest.raises(HTTPException) as refused:
        await license_deps.require_active_license()

    assert says_the_configured_address(refused.value.detail["message"])


async def test_the_ai_quota_refusal_names_the_configured_address(support_address):
    from app.api.deps import subscription

    db = AsyncMock()
    result = MagicMock()
    result.scalar = MagicMock(return_value=None)  # the capped UPDATE matched nothing
    db.execute = AsyncMock(return_value=result)

    with pytest.raises(HTTPException) as refused:
        await subscription.reserve_ai_request(a_user(ai_requests_limit=5, ai_requests_used=5), db)

    assert refused.value.detail["error"] == "ai_quota_exhausted"
    assert says_the_configured_address(refused.value.detail["message"])


def test_the_demo_quota_hint_names_the_configured_address(support_address):
    from app.api.deps.subscription import analysis_quota_renewal_hint

    assert says_the_configured_address(analysis_quota_renewal_hint(a_user(is_demo=True)))
