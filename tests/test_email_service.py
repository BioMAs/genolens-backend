"""
Tests for the email notification service.

Tests:
- extract_mentions (regex-based, no I/O)
- send_email when SMTP is not configured → returns False, no exception
- send_email when SMTP IS configured → calls aiosmtplib.send
- send_project_invitation helper
- send_mention_notification helper
- send_reply_notification helper
"""
import re as _re

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services import email_service as _es
from app.services.email_service import (
    extract_mentions,
    send_email,
    send_project_invitation,
    send_mention_notification,
    send_reply_notification,
)


# ──────────────────────────────────────────────────────────────────────────────
# extract_mentions
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractMentions:
    def test_single_mention(self):
        text = "Hey @alice@example.com, check this out!"
        result = extract_mentions(text)
        assert "alice@example.com" in result

    def test_multiple_mentions(self):
        text = "@alice@example.com and @bob@lab.org are both mentioned"
        result = extract_mentions(text)
        assert "alice@example.com" in result
        assert "bob@lab.org" in result

    def test_no_mentions(self):
        text = "This is a plain comment without any mentions."
        assert extract_mentions(text) == []

    def test_duplicates_deduplicated(self):
        text = "@alice@example.com and @alice@example.com again"
        result = extract_mentions(text)
        assert result.count("alice@example.com") == 1

    def test_mention_at_start(self):
        text = "@charlie@genomics.io please review"
        result = extract_mentions(text)
        assert "charlie@genomics.io" in result

    def test_invalid_not_matched(self):
        text = "@notanemail and @also-not"
        result = extract_mentions(text)
        assert result == []

    def test_mixed_valid_invalid(self):
        text = "Hey @alice@example.com and @notvalid"
        result = extract_mentions(text)
        assert "alice@example.com" in result
        assert len(result) == 1


# ──────────────────────────────────────────────────────────────────────────────
# send_email
# ──────────────────────────────────────────────────────────────────────────────

class TestSendEmail:
    @pytest.mark.asyncio
    async def test_returns_false_when_not_configured(self):
        """send_email should return False silently when SMTP is not set up."""
        with patch("app.services.email_service.settings") as mock_settings:
            mock_settings.SMTP_HOST = ""
            mock_settings.SMTP_USER = ""
            mock_settings.SMTP_PASSWORD = ""
            mock_settings.EMAIL_FROM_ADDRESS = ""
            mock_settings.APP_URL = "http://localhost:3000"
            mock_settings.EMAIL_FROM_NAME = "GenoLens"

            result = await send_email(
                to="user@example.com",
                subject="Test",
                html_body="<p>Test</p>",
                text_body="Test",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_calls_aiosmtplib_when_configured(self):
        """send_email should call aiosmtplib.send when SMTP is configured."""
        with patch("app.services.email_service.settings") as mock_settings, \
             patch("app.services.email_service.aiosmtplib.send", new_callable=AsyncMock) as mock_send:

            mock_settings.SMTP_HOST = "smtp.example.com"
            mock_settings.SMTP_PORT = 587
            mock_settings.SMTP_USER = "user@example.com"
            mock_settings.SMTP_PASSWORD = "secret"
            mock_settings.SMTP_TLS = False
            mock_settings.SMTP_STARTTLS = True
            mock_settings.EMAIL_FROM_ADDRESS = "no-reply@genolens.io"
            mock_settings.EMAIL_FROM_NAME = "GenoLens"
            mock_settings.APP_URL = "http://localhost:3000"

            result = await send_email(
                to="recipient@example.com",
                subject="Hello",
                html_body="<p>Hello</p>",
                text_body="Hello",
            )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_smtp_error(self):
        """send_email should return False (not raise) if SMTP fails."""
        with patch("app.services.email_service.settings") as mock_settings, \
             patch("app.services.email_service.aiosmtplib.send", side_effect=Exception("SMTP error")):

            mock_settings.SMTP_HOST = "smtp.example.com"
            mock_settings.SMTP_PORT = 587
            mock_settings.SMTP_USER = "user@example.com"
            mock_settings.SMTP_PASSWORD = "secret"
            mock_settings.SMTP_TLS = False
            mock_settings.SMTP_STARTTLS = True
            mock_settings.EMAIL_FROM_ADDRESS = "no-reply@genolens.io"
            mock_settings.EMAIL_FROM_NAME = "GenoLens"
            mock_settings.APP_URL = "http://localhost:3000"

            result = await send_email(
                to="recipient@example.com",
                subject="Hello",
                html_body="<p>Hello</p>",
                text_body="Hello",
            )

        assert result is False


# ──────────────────────────────────────────────────────────────────────────────
# High-level helpers (project_invitation, mention, reply)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_project_invitation_calls_send_email():
    """send_project_invitation should delegate to send_email with the right subject."""
    with patch("app.services.email_service.send_email", new_callable=AsyncMock) as mock_send, \
         patch("app.services.email_service.settings") as mock_settings:

        mock_settings.APP_URL = "http://localhost:3000"
        mock_send.return_value = True

        result = await send_project_invitation(
            invitee_email="new@example.com",
            inviter_email="owner@example.com",
            project_id="proj-123",
            project_name="My RNA-seq Project",
            access_level="USER",
        )

    assert result is True
    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args
    assert call_kwargs[1]["to"] == "new@example.com"
    assert "My RNA-seq Project" in call_kwargs[1]["subject"]


@pytest.mark.asyncio
async def test_send_mention_notification_calls_send_email():
    """send_mention_notification should delegate to send_email with the right subject."""
    with patch("app.services.email_service.send_email", new_callable=AsyncMock) as mock_send, \
         patch("app.services.email_service.settings") as mock_settings:

        mock_settings.APP_URL = "http://localhost:3000"
        mock_send.return_value = True

        result = await send_mention_notification(
            mentioned_email="alice@example.com",
            author_email="bob@example.com",
            project_id="proj-123",
            project_name="My Project",
            comment_id="cmt-456",
            comment_excerpt="Hey @alice@example.com, check this gene",
        )

    assert result is True
    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args
    assert call_kwargs[1]["to"] == "alice@example.com"
    assert call_kwargs[1]["subject"] == "You were mentioned in “My Project” — GenoLens"


@pytest.mark.asyncio
async def test_send_reply_notification_calls_send_email():
    """send_reply_notification should delegate to send_email with the right subject."""
    with patch("app.services.email_service.send_email", new_callable=AsyncMock) as mock_send, \
         patch("app.services.email_service.settings") as mock_settings:

        mock_settings.APP_URL = "http://localhost:3000"
        mock_send.return_value = True

        result = await send_reply_notification(
            parent_author_email="alice@example.com",
            replier_email="bob@example.com",
            project_id="proj-123",
            project_name="My Project",
            comment_id="cmt-789",
            original_excerpt="Interesting finding here",
            reply_excerpt="I agree, especially at chromosome 5",
        )

    assert result is True
    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args
    assert call_kwargs[1]["to"] == "alice@example.com"
    assert call_kwargs[1]["subject"] == "New reply to your comment — “My Project” — GenoLens"


# ──────────────────────────────────────────────────────────────────────────────
# English copy (the app UI is English; these emails used to be French)
# ──────────────────────────────────────────────────────────────────────────────


# Accented letters, guillemets and the words the French templates used.
_FRENCH = _re.compile(
    r"[àâçéèêëîïôûùœ«»]|\b(Bonjour|Vous|votre|projet|forfait|comparaisons?|Lien direct)\b",
    _re.I,
)


def _rendered_emails():
    inv = dict(invitee_email="new@example.com", inviter_email="owner@example.com",
               project_name="P", access_level="ADMIN", project_url="http://x/p")
    men = dict(mentioned_email="a@example.com", author_email="b@example.com",
               project_name="P", comment_excerpt="hi", comment_url="http://x/c")
    rep = dict(parent_author_email="a@example.com", replier_email="b@example.com",
               project_name="P", original_excerpt="o", reply_excerpt="r", comment_url="http://x/c")
    quo = dict(to="a@example.com", used=24, quota=30, plan="TEAM", pricing_url="http://x/pricing")
    return {
        "invitation_html": _es._project_invitation_html(**inv),
        "invitation_text": _es._project_invitation_text(**inv),
        "mention_html": _es._mention_notification_html(**men),
        "mention_text": _es._mention_notification_text(**men),
        "reply_html": _es._reply_notification_html(**rep),
        "reply_text": _es._reply_notification_text(**rep),
        "quota_html": _es._quota_warning_html(**quo),
        "quota_text": _es._quota_warning_text(**quo),
    }


@pytest.mark.parametrize("name", list(_rendered_emails()))
def test_email_bodies_are_english(name):
    body = _rendered_emails()[name]
    match = _FRENCH.search(body)
    assert match is None, f"{name} still contains French: {match.group(0)!r}"


def test_invitation_uses_the_app_role_label():
    assert "Admin role" in _es._project_invitation_text(
        invitee_email="n@e.com", inviter_email="o@e.com",
        project_name="P", access_level="ADMIN", project_url="http://x",
    )


def test_quota_warning_names_the_plan_and_counts_analyses():
    text = _es._quota_warning_text(
        to="a@e.com", used=24, quota=30, plan="TEAM", pricing_url="http://x"
    )
    assert "24 of your 30 analyses" in text
    assert "(Pro plan)" in text
    assert "6 analyses left" in text
    assert _es._quota_warning_subject(1) == "GenoLens — 1 analysis left this month"


def test_plan_label_falls_back_for_an_unknown_plan():
    assert _es._plan_label("ON_PREMISE") == "Enterprise"
    assert _es._plan_label("SOMETHING_ELSE") == "Something Else"
