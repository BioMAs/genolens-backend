"""Tests of the authentication e-mails (`app/auth/emails.py`) and of `send_email(required=True)`.

Nothing is sent: `smtp_capture` (tests/auth_support.py) replaces the SMTP library and fills in
placeholder SMTP settings. The recipient is always an `@example.test` address.
"""

import logging
from html import escape
from urllib.parse import quote

import pytest

from app.auth.emails import (
    _duration,
    _expires_in,
    send_account_exists_email,
    send_invitation_email,
    send_mfa_enabled_email,
    send_mfa_reset_email,
    send_password_changed_email,
    send_password_reset_email,
    send_recovery_code_used_email,
    send_verify_email,
)
from app.core.config import settings
from app.services.email_service import EmailNotConfiguredError, EmailSendError, send_email

TO = "u@example.test"

# Every e-mail of the module, as a zero-argument call.
ALL_EMAILS = [
    pytest.param(lambda: send_invitation_email(TO, "tok"), id="invitation"),
    pytest.param(lambda: send_verify_email(TO, "tok"), id="verify"),
    pytest.param(lambda: send_password_reset_email(TO, "tok"), id="password_reset"),
    pytest.param(lambda: send_account_exists_email(TO, "tok"), id="account_exists"),
    pytest.param(lambda: send_password_changed_email(TO), id="password_changed"),
    pytest.param(lambda: send_mfa_enabled_email(TO), id="mfa_enabled"),
    pytest.param(lambda: send_mfa_reset_email(TO), id="mfa_reset"),
    pytest.param(lambda: send_recovery_code_used_email(TO, 4), id="recovery_code_used"),
]

# The e-mails that carry a one-time link: (sender, path of the link, how long it is valid).
LINK_EMAILS = [
    pytest.param(send_invitation_email, "/auth/invite", "7 days", id="invitation"),
    pytest.param(send_verify_email, "/auth/verify-email", "24 hours", id="verify"),
    pytest.param(send_password_reset_email, "/auth/reset", "1 hour", id="password_reset"),
    pytest.param(send_account_exists_email, "/auth/reset", "1 hour", id="account_exists"),
]


def app_url() -> str:
    return settings.APP_URL.rstrip("/")


# ---- send_email(required=...) ----------------------------------------------------------------


async def test_required_email_raises_when_smtp_missing(monkeypatch):
    monkeypatch.setattr(settings, "SMTP_HOST", None)
    with pytest.raises(EmailNotConfiguredError):
        await send_password_reset_email("u@example.test", "tok")


@pytest.mark.parametrize(
    "missing", ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_FROM_ADDRESS"]
)
async def test_required_email_raises_for_each_missing_smtp_setting(
    monkeypatch, smtp_capture, missing
):
    monkeypatch.setattr(settings, missing, None)
    with pytest.raises(EmailNotConfiguredError):
        await send_email(TO, "s", "<p>h</p>", "t", required=True)
    assert smtp_capture.messages == []


@pytest.mark.parametrize("send", ALL_EMAILS)
async def test_every_auth_email_fails_loudly_when_smtp_is_missing(monkeypatch, smtp_capture, send):
    monkeypatch.setattr(settings, "SMTP_HOST", None)
    with pytest.raises(EmailNotConfiguredError):
        await send()
    assert smtp_capture.messages == []


@pytest.mark.parametrize("send", ALL_EMAILS)
async def test_every_auth_email_fails_loudly_when_smtp_errors(smtp_capture, send):
    smtp_capture.fail_with = ConnectionRefusedError("down")
    with pytest.raises(EmailSendError):
        await send()


async def test_required_email_send_error_keeps_the_cause(smtp_capture):
    smtp_capture.fail_with = ConnectionRefusedError("down")
    with pytest.raises(EmailSendError) as excinfo:
        await send_email(TO, "s", "<p>h</p>", "t", required=True)
    assert excinfo.value.__cause__ is smtp_capture.fail_with
    assert TO not in str(excinfo.value)  # an address must not travel with the error


def test_the_two_errors_are_runtime_errors_and_distinct():
    assert issubclass(EmailNotConfiguredError, RuntimeError)
    assert issubclass(EmailSendError, RuntimeError)
    assert not issubclass(EmailSendError, EmailNotConfiguredError)
    assert not issubclass(EmailNotConfiguredError, EmailSendError)


async def test_required_email_returns_true_once_sent(smtp_capture):
    assert await send_email(TO, "Subject", "<p>h</p>", "t", required=True) is True
    assert [(m.to, m.subject, m.text, m.html) for m in smtp_capture.messages] == [
        (TO, "Subject", "t", "<p>h</p>")
    ]


async def test_optional_email_keeps_current_behaviour(monkeypatch):
    monkeypatch.setattr(settings, "SMTP_HOST", None)
    assert await send_email("u@example.test", "s", "<p>h</p>", "t") is False


async def test_optional_email_still_swallows_an_smtp_error(smtp_capture):
    smtp_capture.fail_with = ConnectionRefusedError("down")
    assert await send_email(TO, "s", "<p>h</p>", "t") is False
    assert await send_email(TO, "s", "<p>h</p>", "t", required=False) is False


async def test_optional_email_is_still_sent_when_configured(smtp_capture):
    assert await send_email(TO, "s", "<p>h</p>", "t") is True
    assert smtp_capture.last_text == "t"


# ---- links ------------------------------------------------------------------------------------


async def test_links_point_to_app(monkeypatch, smtp_capture):
    await send_invitation_email("u@example.test", "RAW")
    assert f"{settings.APP_URL}/auth/invite?token=RAW" in smtp_capture.last_text


@pytest.mark.parametrize("send, path, validity", LINK_EMAILS)
async def test_link_is_in_both_bodies_and_the_mail_goes_to_the_recipient(
    smtp_capture, send, path, validity
):
    await send(TO, "RAW")
    mail = smtp_capture.last
    link = f"{app_url()}{path}?token=RAW"
    assert mail.to == TO
    assert link in mail.text
    assert f'href="{link}"' in mail.html


@pytest.mark.parametrize("send, path, validity", LINK_EMAILS)
async def test_token_is_url_encoded_in_the_link(smtp_capture, send, path, validity):
    raw = "a+b/c d&e=f?g#h%"
    await send(TO, raw)
    mail = smtp_capture.last
    link = f"{app_url()}{path}?token={quote(raw, safe='')}"
    assert link in mail.text
    assert link in mail.html
    assert raw not in mail.text
    assert raw not in mail.html


@pytest.mark.parametrize("send, path, validity", LINK_EMAILS)
async def test_trailing_slash_of_app_url_is_not_doubled(
    monkeypatch, smtp_capture, send, path, validity
):
    monkeypatch.setattr(settings, "APP_URL", "https://app.example.test/")
    await send(TO, "RAW")
    assert f"https://app.example.test{path}?token=RAW" in smtp_capture.last_text
    assert "test//auth" not in smtp_capture.last_text
    assert "test//auth" not in smtp_capture.last.html


@pytest.mark.parametrize("send, path, validity", LINK_EMAILS)
async def test_link_is_escaped_in_html(monkeypatch, smtp_capture, send, path, validity):
    monkeypatch.setattr(settings, "APP_URL", 'https://app.example.test/?a=1&b="<2>"')
    await send(TO, "RAW")
    link = f'https://app.example.test/?a=1&b="<2>"{path}?token=RAW'
    html = smtp_capture.last.html
    assert escape(link) in html
    assert link not in html
    assert link in smtp_capture.last_text  # the text part is not HTML: nothing to escape


@pytest.mark.parametrize("send, path, validity", LINK_EMAILS)
async def test_validity_is_stated_in_both_bodies(smtp_capture, send, path, validity):
    await send(TO, "RAW")
    sentence = f"This link expires in {validity}."
    assert sentence in smtp_capture.last_text
    assert sentence in smtp_capture.last.html


async def test_reset_email_says_one_hour(smtp_capture):
    await send_password_reset_email(TO, "RAW")
    assert "This link expires in 1 hour." in smtp_capture.last_text


@pytest.mark.parametrize("send, path, validity", LINK_EMAILS)
async def test_link_emails_say_what_to_do_if_not_asked_for(smtp_capture, send, path, validity):
    await send(TO, "RAW")
    assert "you can ignore this email" in smtp_capture.last_text
    assert "you can ignore this email" in smtp_capture.last.html


async def test_account_exists_email_offers_the_reset_link_and_reveals_nothing_more(smtp_capture):
    await send_account_exists_email(TO, "RAW")
    text = smtp_capture.last_text
    assert f"{app_url()}/auth/reset?token=RAW" in text
    assert "already" in text.lower()


async def test_the_raw_token_is_never_logged(smtp_capture, caplog):
    raw = "SECRET-RAW-TOKEN-VALUE"
    with caplog.at_level(logging.DEBUG):
        await send_invitation_email(TO, raw)
        await send_verify_email(TO, raw)
        await send_password_reset_email(TO, raw)
        await send_account_exists_email(TO, raw)
    assert caplog.records, "the send should leave a trace"
    assert raw not in caplog.text
    assert all(raw not in m.subject for m in smtp_capture.messages)


async def test_the_raw_token_is_not_logged_when_the_send_fails(smtp_capture, caplog):
    raw = "SECRET-RAW-TOKEN-VALUE"
    smtp_capture.fail_with = ConnectionRefusedError("down")
    with caplog.at_level(logging.DEBUG), pytest.raises(EmailSendError) as excinfo:
        await send_password_reset_email(TO, raw)
    assert raw not in caplog.text
    assert raw not in str(excinfo.value)


# ---- security notifications -------------------------------------------------------------------


@pytest.mark.parametrize(
    "send, subject_part, unexpected",
    [
        pytest.param(
            send_password_changed_email,
            "password was changed",
            "If this was not you",
            id="password_changed",
        ),
        pytest.param(send_mfa_enabled_email, "two-factor", "If this was not you", id="mfa_enabled"),
        pytest.param(
            send_mfa_reset_email, "two-factor", "If you did not expect this", id="mfa_reset"
        ),
    ],
)
async def test_notifications_go_to_the_recipient_and_tell_what_to_do_if_unexpected(
    smtp_capture, send, subject_part, unexpected
):
    await send(TO)
    mail = smtp_capture.last
    assert mail.to == TO
    assert subject_part in mail.subject.lower()
    assert "support@scilicium.com" in mail.text
    assert "support@scilicium.com" in mail.html
    assert unexpected in mail.text
    assert unexpected in mail.html


# The e-mails that name the support address.
SUPPORT_EMAILS = [
    pytest.param(lambda: send_password_changed_email(TO), id="password_changed"),
    pytest.param(lambda: send_mfa_enabled_email(TO), id="mfa_enabled"),
    pytest.param(lambda: send_mfa_reset_email(TO), id="mfa_reset"),
    pytest.param(lambda: send_recovery_code_used_email(TO, 4), id="recovery_code_used"),
]


@pytest.mark.parametrize("send", SUPPORT_EMAILS)
async def test_the_support_address_is_the_configured_one(smtp_capture, settings_override, send):
    """Read from the settings when the e-mail is sent, not when the module was imported."""
    settings_override(SUPPORT_EMAIL="help-desk@example.test")

    await send()

    mail = smtp_capture.last
    assert "help-desk@example.test" in mail.text and "help-desk@example.test" in mail.html
    assert "support@scilicium.com" not in mail.text + mail.html


def test_the_support_address_defaults_to_the_one_production_uses():
    from app.core.config import Settings

    assert Settings.model_fields["SUPPORT_EMAIL"].default == "support@scilicium.com"


async def test_password_changed_email_points_to_the_password_reset_page(smtp_capture):
    await send_password_changed_email(TO)
    assert f"{app_url()}/auth/forgot" in smtp_capture.last_text
    assert "token=" not in smtp_capture.last_text


async def test_mfa_enabled_email_says_a_code_is_now_required(smtp_capture):
    await send_mfa_enabled_email(TO)
    assert "turned on" in smtp_capture.last_text
    assert "code" in smtp_capture.last_text


async def test_mfa_reset_email_says_an_administrator_did_it_and_what_comes_next(smtp_capture):
    await send_mfa_reset_email(TO)
    text = smtp_capture.last_text
    assert "administrator" in text
    assert "set it up again" in text


@pytest.mark.parametrize(
    "remaining, expected, suggests_regenerating",
    [
        (9, "9 recovery codes left", False),
        (3, "3 recovery codes left", False),
        (2, "2 recovery codes left", True),
        (1, "1 recovery code left", True),
        (0, "no recovery codes left", True),
    ],
)
async def test_recovery_code_used_email_states_what_remains(
    smtp_capture, remaining, expected, suggests_regenerating
):
    await send_recovery_code_used_email(TO, remaining)
    mail = smtp_capture.last
    assert mail.to == TO
    assert expected in mail.text
    assert expected in mail.html
    assert ("generate a new set" in mail.text) is suggests_regenerating
    assert ("generate a new set" in mail.html) is suggests_regenerating
    assert "support@scilicium.com" in mail.text
    assert "If this was not you" in mail.text


async def test_recovery_code_used_email_never_shows_a_negative_count(smtp_capture):
    await send_recovery_code_used_email(TO, -3)
    assert "no recovery codes left" in smtp_capture.last_text
    assert "-3" not in smtp_capture.last_text


# ---- shape of every message --------------------------------------------------------------------


@pytest.mark.parametrize("send", ALL_EMAILS)
async def test_every_email_has_a_subject_a_text_and_an_html_part(smtp_capture, send):
    await send()
    mail = smtp_capture.last
    assert mail.subject.strip()
    assert mail.text.strip()
    assert mail.html.lstrip().lower().startswith("<!doctype html>")
    assert "GenoLens" in mail.subject


@pytest.mark.parametrize("send", ALL_EMAILS)
async def test_every_email_has_no_unrendered_placeholder(smtp_capture, send):
    await send()
    mail = smtp_capture.last
    for body in (mail.subject, mail.text):
        assert "{" not in body
        assert "}" not in body
    for body in (mail.subject, mail.text, mail.html):
        assert "None" not in body


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (7 * 86400, "7 days"),
        (2 * 86400, "2 days"),
        (86400, "24 hours"),  # not "1 day"
        (47 * 3600, "47 hours"),
        (3600, "1 hour"),
        (2 * 3600, "2 hours"),
        (60, "1 minute"),
        (15 * 60, "15 minutes"),
        (90 * 60, "90 minutes"),
        (45, "45 seconds"),
        (1, "1 second"),
    ],
)
def test_duration_reads_naturally(seconds, expected):
    assert _duration(seconds) == expected


def test_validity_sentences_follow_the_real_token_lifetimes():
    from datetime import timedelta

    from app.auth.one_time_tokens import TOKEN_TTL
    from app.models.auth import OneTimeTokenPurpose

    assert TOKEN_TTL[OneTimeTokenPurpose.INVITE] == timedelta(days=7)
    assert TOKEN_TTL[OneTimeTokenPurpose.EMAIL_VERIFY] == timedelta(hours=24)
    assert TOKEN_TTL[OneTimeTokenPurpose.PASSWORD_RESET] == timedelta(hours=1)
    assert _expires_in(OneTimeTokenPurpose.INVITE) == "This link expires in 7 days."
    assert _expires_in(OneTimeTokenPurpose.EMAIL_VERIFY) == "This link expires in 24 hours."
    assert _expires_in(OneTimeTokenPurpose.PASSWORD_RESET) == "This link expires in 1 hour."
