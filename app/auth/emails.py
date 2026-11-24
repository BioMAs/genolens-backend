"""
The e-mails of the local authentication: invitation, e-mail verification, password reset,
"you already have an account", and the four security notifications.

Every one is sent with `required=True` (see `send_email`): if SMTP is not configured, or the server
refuses the message, the call raises `EmailNotConfiguredError` / `EmailSendError` instead of logging
and carrying on. A login link or a security alert that never left must not look as if it had.
Callers decide what to do with the failure; nothing is swallowed here, and nothing returns a flag.

Copy is English, like the rest of the app, short and plain: one action per e-mail, how long a link
is valid where there is one, and what to do if the reader did not expect the message. No customer
is named. The address to contact is the `SUPPORT_EMAIL` setting, read when the e-mail is sent.

Secrets. A raw token appears only in the link (URL-encoded, so that no character of it can break
the link or the HTML) and is never logged, neither here nor by `send_email`, which logs the
recipient and the subject only. Values interpolated in HTML are escaped.
"""

from html import escape
from typing import Optional, Sequence
from urllib.parse import quote

from app.auth.one_time_tokens import TOKEN_TTL
from app.core.config import settings
from app.models.auth import OneTimeTokenPurpose
from app.services.email_service import _base_layout, send_email

# A button: (its label, the URL it opens).
_Action = tuple[str, str]


def _link(path: str, raw_token: str) -> str:
    """`path` on the app, carrying the token: `<APP_URL><path>?token=<url-encoded token>`."""
    return f"{settings.APP_URL.rstrip('/')}{path}?token={quote(raw_token, safe='')}"


def _duration(seconds: int) -> str:
    """ "7 days", "24 hours", "1 hour", "5 minutes": the largest unit that divides `seconds`.

    Days only from two (24 hours reads better than "1 day"), and the unit is always plural but
    for one.
    """
    if seconds % 86400 == 0 and seconds >= 2 * 86400:
        count, unit = seconds // 86400, "day"
    elif seconds % 3600 == 0:
        count, unit = seconds // 3600, "hour"
    elif seconds % 60 == 0:
        count, unit = seconds // 60, "minute"
    else:
        count, unit = seconds, "second"
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def _expires_in(purpose: OneTimeTokenPurpose) -> str:
    """ "This link expires in 1 hour.", from the lifetime the token is really issued with."""
    return f"This link expires in {_duration(int(TOKEN_TTL[purpose].total_seconds()))}."


def _render(
    subject: str,
    intro: Sequence[str],
    action: Optional[_Action] = None,
    after: Sequence[str] = (),
) -> tuple[str, str]:
    """The `(html, text)` bodies: a greeting, `intro`, the button, then `after`.

    Paragraphs are plain strings; they are escaped for the HTML and used as they are for the text.
    The button is repeated as a bare link under the message, for clients that do not render it.
    """
    html = ["<p>Hello,</p>"] + [f"<p>{escape(paragraph)}</p>" for paragraph in intro]
    text = ["Hello,", *intro]
    if action is not None:
        label, url = action
        html.append(f'<a href="{escape(url)}" class="btn">{escape(label)}</a>')
        text.append(f"{label}: {url}")
    html += [f"<p>{escape(paragraph)}</p>" for paragraph in after]
    text += after
    if action is not None:
        url = action[1]
        html.append(
            '<div class="meta"><p>If the button does not work, copy this link into your '
            f'browser: <a href="{escape(url)}">{escape(url)}</a></p></div>'
        )
    text.append("— The GenoLens team")
    return _base_layout(escape(subject), "\n".join(html)), "\n\n".join(text)


async def _send(
    to: str,
    subject: str,
    intro: Sequence[str],
    action: Optional[_Action] = None,
    after: Sequence[str] = (),
) -> None:
    html, text = _render(subject, intro, action, after)
    await send_email(to, subject, html, text, required=True)


# ---- e-mails that carry a one-time link ---------------------------------------------------------


async def send_invitation_email(to: str, raw_token: str) -> None:
    """The invitation of an administrator: the link lets the invitee choose a password."""
    await _send(
        to,
        "You are invited to GenoLens",
        ["You have been invited to create a GenoLens account. Choose a password to get started."],
        ("Accept the invitation", _link("/auth/invite", raw_token)),
        [
            _expires_in(OneTimeTokenPurpose.INVITE),
            "If you did not expect this invitation, you can ignore this email.",
        ],
    )


async def send_verify_email(to: str, raw_token: str) -> None:
    """Confirmation of the address of a self-service sign-up."""
    await _send(
        to,
        "Confirm your GenoLens email address",
        ["Confirm your email address to finish creating your GenoLens account."],
        ("Confirm email address", _link("/auth/verify-email", raw_token)),
        [
            _expires_in(OneTimeTokenPurpose.EMAIL_VERIFY),
            "If you did not create a GenoLens account, you can ignore this email.",
        ],
    )


async def send_password_reset_email(to: str, raw_token: str) -> None:
    """The link to choose a new password."""
    await _send(
        to,
        "Reset your GenoLens password",
        ["We received a request to reset the password of your GenoLens account."],
        ("Choose a new password", _link("/auth/reset", raw_token)),
        [
            _expires_in(OneTimeTokenPurpose.PASSWORD_RESET),
            "If you did not request this, you can ignore this email: your password stays "
            "unchanged.",
        ],
    )


async def send_account_exists_email(to: str, raw_token: str) -> None:
    """Sent instead of a verification link when someone signs up with an address that already has
    an account, so that the sign-up page answers the same whether the address is known or not.
    It offers the way back in: a password reset."""
    await _send(
        to,
        "You already have a GenoLens account",
        [
            "Someone tried to create a GenoLens account with this email address, but you "
            "already have one.",
            "If that was you and you forgot your password, you can choose a new one.",
        ],
        ("Choose a new password", _link("/auth/reset", raw_token)),
        [
            _expires_in(OneTimeTokenPurpose.PASSWORD_RESET),
            "If you did not try to sign up, you can ignore this email: your account is "
            "unchanged.",
        ],
    )


# ---- security notifications ---------------------------------------------------------------------


async def send_password_changed_email(to: str) -> None:
    """After a password change or reset. There is no token: the button leads to the page that
    asks for a reset link, for a reader whose password was changed by someone else."""
    await _send(
        to,
        "Your GenoLens password was changed",
        [
            "The password of your GenoLens account was just changed.",
            "If this was you, no action is needed. If this was not you, reset your password now "
            f"and contact {settings.SUPPORT_EMAIL}.",
        ],
        ("Reset my password", f"{settings.APP_URL.rstrip('/')}/auth/forgot"),
    )


async def send_mfa_enabled_email(to: str) -> None:
    """After two-factor authentication was turned on."""
    await _send(
        to,
        "Two-factor authentication is on for your GenoLens account",
        [
            "Two-factor authentication was just turned on for your GenoLens account. From now "
            "on, signing in requires a code from your authenticator app.",
            "If this was not you, reset your password and contact "
            f"{settings.SUPPORT_EMAIL} right away.",
        ],
    )


async def send_mfa_reset_email(to: str) -> None:
    """After an administrator reset two-factor authentication on the account."""
    await _send(
        to,
        "Two-factor authentication was reset on your GenoLens account",
        [
            "An administrator reset two-factor authentication on your GenoLens account. You will "
            "be asked to set it up again the next time you sign in.",
            f"If you did not expect this, contact {settings.SUPPORT_EMAIL} right away.",
        ],
    )


def _recovery_codes(count: int) -> str:
    """ "no recovery codes", "1 recovery code", "5 recovery codes" (a negative count is none)."""
    if count <= 0:
        return "no recovery codes"
    return "1 recovery code" if count == 1 else f"{count} recovery codes"


async def send_recovery_code_used_email(to: str, remaining: int) -> None:
    """After a recovery code was used to sign in: how many are left, and, from two or fewer, to
    generate a new set before being locked out."""
    intro = [
        "A recovery code was just used to sign in to your GenoLens account. "
        f"You have {_recovery_codes(remaining)} left."
    ]
    if remaining <= 2:
        intro.append(
            "To avoid being locked out, generate a new set of recovery codes from your account "
            "security settings."
        )
    intro.append(
        "If this was not you, reset your password and contact "
        f"{settings.SUPPORT_EMAIL} right away."
    )
    await _send(to, "A recovery code was used on your GenoLens account", intro)
