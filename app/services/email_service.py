"""
Email notification service for GenoLens.

Supports:
- Project invitation emails
- Comment mention notifications (@username/@email)
- Comment reply notifications
- Monthly analysis quota warning (80 %)

All user-facing copy is in English, like the app UI.

Uses async SMTP (aiosmtplib) with configurable SMTP settings.
Gracefully degrades if email is not configured (logs warning, does not raise).
"""
import logging
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import aiosmtplib

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML Email Templates
# ---------------------------------------------------------------------------

def _base_layout(title: str, content: str) -> str:
    """Wrap content in a consistent HTML layout."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0; padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #f4f6f8; color: #1a1a2e;
    }}
    .container {{
      max-width: 560px; margin: 40px auto;
      background: #ffffff; border-radius: 12px;
      overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }}
    .header {{
      background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
      padding: 32px 40px;
    }}
    .header h1 {{
      margin: 0; color: #ffffff;
      font-size: 22px; font-weight: 700;
      letter-spacing: -0.5px;
    }}
    .header p {{
      margin: 6px 0 0; color: rgba(255,255,255,0.8);
      font-size: 13px;
    }}
    .body {{
      padding: 36px 40px;
    }}
    .body p {{
      margin: 0 0 16px; font-size: 15px;
      line-height: 1.6; color: #374151;
    }}
    .highlight-box {{
      background: #f5f3ff; border-left: 4px solid #6366f1;
      border-radius: 4px; padding: 16px 20px; margin: 20px 0;
    }}
    .highlight-box p {{
      margin: 0; color: #4c1d95; font-style: italic; font-size: 14px;
    }}
    .btn {{
      display: inline-block; padding: 14px 28px;
      background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
      color: #ffffff !important; text-decoration: none;
      border-radius: 8px; font-size: 15px; font-weight: 600;
      margin: 8px 0 24px;
    }}
    .meta {{
      font-size: 13px; color: #6b7280;
      border-top: 1px solid #e5e7eb; padding-top: 20px; margin-top: 8px;
    }}
    .footer {{
      background: #f9fafb; padding: 20px 40px;
      text-align: center; font-size: 12px; color: #9ca3af;
    }}
    .footer a {{ color: #6366f1; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🧬 GenoLens</h1>
      <p>Transcriptomics analysis platform</p>
    </div>
    <div class="body">
      {content}
    </div>
    <div class="footer">
      You are receiving this email because you use
      <a href="{settings.APP_URL}">GenoLens</a>.
      Please do not reply to this email.
    </div>
  </div>
</body>
</html>"""


def _role_label(access_level: str) -> str:
    """Project access level as the app shows it (ProjectMembersModal)."""
    return {
        "ADMIN": "Admin",
        "USER": "User",
        "VIEWER": "Viewer",
    }.get(access_level.upper(), access_level)


def _project_invitation_html(
    invitee_email: str,
    inviter_email: str,
    project_name: str,
    access_level: str,
    project_url: str,
) -> str:
    role_label = _role_label(access_level)

    content = f"""
      <p>Hello,</p>
      <p>
        <strong>{inviter_email}</strong> has invited you to collaborate on the project
        <strong>“{project_name}”</strong> with the <strong>{role_label}</strong> role.
      </p>
      <p>
        GenoLens lets you explore and analyze gene expression data:
        volcano plots, heatmaps, GSEA, GO enrichment and more.
      </p>
      <a href="{project_url}" class="btn">Open the project →</a>
      <div class="meta">
        <p>
          If you don't have a GenoLens account yet, sign up with
          <strong>{invitee_email}</strong> to get access to the shared project automatically.
        </p>
        <p>Direct link: <a href="{project_url}">{project_url}</a></p>
      </div>
    """
    return _base_layout(f"Invitation to the project “{project_name}”", content)


def _project_invitation_text(
    invitee_email: str,
    inviter_email: str,
    project_name: str,
    access_level: str,
    project_url: str,
) -> str:
    return (
        f"Hello,\n\n"
        f"{inviter_email} has invited you to collaborate on the project “{project_name}” "
        f"with the {_role_label(access_level)} role.\n\n"
        f"Open the project: {project_url}\n\n"
        f"If you don't have an account yet, sign up with {invitee_email}.\n\n"
        f"— The GenoLens team"
    )


def _mention_notification_html(
    mentioned_email: str,
    author_email: str,
    project_name: str,
    comment_excerpt: str,
    comment_url: str,
) -> str:
    excerpt_escaped = comment_excerpt.replace("<", "&lt;").replace(">", "&gt;")
    content = f"""
      <p>Hello,</p>
      <p>
        <strong>{author_email}</strong> mentioned you in a comment
        on the project <strong>“{project_name}”</strong>:
      </p>
      <div class="highlight-box">
        <p>{excerpt_escaped}</p>
      </div>
      <a href="{comment_url}" class="btn">View the comment →</a>
      <div class="meta">
        <p>Direct link: <a href="{comment_url}">{comment_url}</a></p>
      </div>
    """
    return _base_layout(f"You were mentioned in “{project_name}”", content)


def _mention_notification_text(
    mentioned_email: str,
    author_email: str,
    project_name: str,
    comment_excerpt: str,
    comment_url: str,
) -> str:
    return (
        f"Hello,\n\n"
        f"{author_email} mentioned you in a comment "
        f"on the project “{project_name}”:\n\n"
        f"  {comment_excerpt}\n\n"
        f"View the comment: {comment_url}\n\n"
        f"— The GenoLens team"
    )


def _reply_notification_html(
    parent_author_email: str,
    replier_email: str,
    project_name: str,
    original_excerpt: str,
    reply_excerpt: str,
    comment_url: str,
) -> str:
    orig_escaped = original_excerpt.replace("<", "&lt;").replace(">", "&gt;")
    reply_escaped = reply_excerpt.replace("<", "&lt;").replace(">", "&gt;")
    content = f"""
      <p>Hello,</p>
      <p>
        <strong>{replier_email}</strong> replied to your comment
        on the project <strong>“{project_name}”</strong>.
      </p>
      <p style="color:#6b7280; font-size:13px; margin-bottom:4px;">Your comment:</p>
      <div class="highlight-box" style="border-color:#d1d5db; background:#f9fafb;">
        <p style="color:#6b7280;">{orig_escaped}</p>
      </div>
      <p style="color:#6b7280; font-size:13px; margin-bottom:4px;">Reply from {replier_email}:</p>
      <div class="highlight-box">
        <p>{reply_escaped}</p>
      </div>
      <a href="{comment_url}" class="btn">View the reply →</a>
      <div class="meta">
        <p>Direct link: <a href="{comment_url}">{comment_url}</a></p>
      </div>
    """
    return _base_layout(f"New reply to your comment — “{project_name}”", content)


def _reply_notification_text(
    parent_author_email: str,
    replier_email: str,
    project_name: str,
    original_excerpt: str,
    reply_excerpt: str,
    comment_url: str,
) -> str:
    return (
        f"Hello,\n\n"
        f"{replier_email} replied to your comment "
        f"on the project “{project_name}”.\n\n"
        f"Your comment: {original_excerpt}\n\n"
        f"Reply: {reply_excerpt}\n\n"
        f"View the reply: {comment_url}\n\n"
        f"— The GenoLens team"
    )


# ---------------------------------------------------------------------------
# Core send function
# ---------------------------------------------------------------------------

def _is_email_configured() -> bool:
    """Return True if SMTP is properly configured."""
    return bool(
        settings.SMTP_HOST
        and settings.SMTP_USER
        and settings.SMTP_PASSWORD
        and settings.EMAIL_FROM_ADDRESS
    )


async def send_email(
    to: str,
    subject: str,
    html_body: str,
    text_body: str,
) -> bool:
    """
    Send an email via SMTP.

    Returns True on success, False on any error (to avoid blocking callers).
    """
    if not _is_email_configured():
        logger.warning(
            "Email not configured — skipping notification to %s. "
            "Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM_ADDRESS in .env",
            to,
        )
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM_ADDRESS}>"
    msg["To"] = to

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            use_tls=settings.SMTP_TLS,
            start_tls=settings.SMTP_STARTTLS,
            timeout=15,
        )
        logger.info("Email sent to %s — %s", to, subject)
        return True
    except Exception as exc:
        logger.error("Failed to send email to %s: %s", to, exc)
        return False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

async def send_access_request(
    requester_email: str,
    kind: str,
    item: str,
    details: Optional[str] = None,
) -> bool:
    """
    Notify the sales inbox that a user requested a plan change or a module.

    *kind* is "plan" or "module"; *item* is the human-readable name.
    """
    label = "Plan" if kind == "plan" else "Module"
    subject = f"{label} request — {item} · GenoLens"
    detail_html = f"<p style='color:#5b6472'>{details}</p>" if details else ""
    detail_text = f"\nDetails: {details}" if details else ""
    html_body = (
        f"<p><b>{requester_email}</b> requested the <b>{item}</b> {kind}.</p>"
        f"{detail_html}"
        f"<p style='color:#9aa0ab;font-size:12px'>Sent from the GenoLens app.</p>"
    )
    text_body = f"{requester_email} requested the {item} {kind}.{detail_text}\n\nSent from the GenoLens app."
    return await send_email(
        to=settings.SALES_EMAIL,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
    )


async def send_project_invitation(
    invitee_email: str,
    inviter_email: str,
    project_id: str,
    project_name: str,
    access_level: str,
) -> bool:
    """Send a project invitation email to *invitee_email*."""
    project_url = f"{settings.APP_URL}/projects/{project_id}"
    return await send_email(
        to=invitee_email,
        subject=f"Invitation to the project “{project_name}” — GenoLens",
        html_body=_project_invitation_html(
            invitee_email=invitee_email,
            inviter_email=inviter_email,
            project_name=project_name,
            access_level=access_level,
            project_url=project_url,
        ),
        text_body=_project_invitation_text(
            invitee_email=invitee_email,
            inviter_email=inviter_email,
            project_name=project_name,
            access_level=access_level,
            project_url=project_url,
        ),
    )


async def send_mention_notification(
    mentioned_email: str,
    author_email: str,
    project_id: str,
    project_name: str,
    comment_id: str,
    comment_excerpt: str,
) -> bool:
    """Send a mention notification email to *mentioned_email*."""
    comment_url = f"{settings.APP_URL}/projects/{project_id}?comment={comment_id}"
    return await send_email(
        to=mentioned_email,
        subject=f"You were mentioned in “{project_name}” — GenoLens",
        html_body=_mention_notification_html(
            mentioned_email=mentioned_email,
            author_email=author_email,
            project_name=project_name,
            comment_excerpt=comment_excerpt,
            comment_url=comment_url,
        ),
        text_body=_mention_notification_text(
            mentioned_email=mentioned_email,
            author_email=author_email,
            project_name=project_name,
            comment_excerpt=comment_excerpt,
            comment_url=comment_url,
        ),
    )


async def send_reply_notification(
    parent_author_email: str,
    replier_email: str,
    project_id: str,
    project_name: str,
    comment_id: str,
    original_excerpt: str,
    reply_excerpt: str,
) -> bool:
    """Send a reply notification to the author of the parent comment."""
    comment_url = f"{settings.APP_URL}/projects/{project_id}?comment={comment_id}"
    return await send_email(
        to=parent_author_email,
        subject=f"New reply to your comment — “{project_name}” — GenoLens",
        html_body=_reply_notification_html(
            parent_author_email=parent_author_email,
            replier_email=replier_email,
            project_name=project_name,
            original_excerpt=original_excerpt,
            reply_excerpt=reply_excerpt,
            comment_url=comment_url,
        ),
        text_body=_reply_notification_text(
            parent_author_email=parent_author_email,
            replier_email=replier_email,
            project_name=project_name,
            original_excerpt=original_excerpt,
            reply_excerpt=reply_excerpt,
            comment_url=comment_url,
        ),
    )


def _plan_label(plan: str) -> str:
    """Display name of a plan id, e.g. TEAM -> "Pro", from the pricing grid."""
    try:
        from app.core.pricing import get_pricing

        return get_pricing().get_plan(plan).name_en
    except Exception:  # unknown id or grid unavailable: never block the email
        return str(getattr(plan, "value", plan)).replace("_", " ").title()


def _analyses(n: int) -> str:
    """ "1 analysis" / "3 analyses" — the billing unit is the analysis, not the contrast."""
    return f"{n} analysis" if n == 1 else f"{n} analyses"


def _quota_warning_subject(remaining: int) -> str:
    return f"GenoLens — {_analyses(remaining)} left this month"


def _quota_warning_html(
    to: str,
    used: int,
    quota: int,
    plan: str,
    pricing_url: str,
) -> str:
    remaining = quota - used
    content = f"""
      <p>Hello,</p>
      <p>
        You have used <strong>{used} of your {quota} analyses</strong>
        this month (<strong>{_plan_label(plan)}</strong> plan).
      </p>
      <div class="highlight-box">
        <p>You have <strong>{_analyses(remaining)}</strong> left this month.</p>
      </div>
      <p>
        To keep running analyses without interruption, you can upgrade your plan
        from the pricing page.
      </p>
      <a href="{pricing_url}" class="btn">View plans →</a>
      <div class="meta">
        <p>Direct link: <a href="{pricing_url}">{pricing_url}</a></p>
      </div>
    """
    return _base_layout(_quota_warning_subject(remaining), content)


def _quota_warning_text(
    to: str,
    used: int,
    quota: int,
    plan: str,
    pricing_url: str,
) -> str:
    remaining = quota - used
    return (
        f"Hello,\n\n"
        f"You have used {used} of your {quota} analyses this month "
        f"({_plan_label(plan)} plan).\n\n"
        f"You have {_analyses(remaining)} left this month.\n\n"
        f"To keep running analyses without interruption, see our plans:\n"
        f"{pricing_url}\n\n"
        f"— The GenoLens team"
    )


async def send_quota_warning_email(
    to: str,
    used: int,
    quota: int,
    plan: str,
) -> bool:
    """Send a quota warning email when usage approaches the monthly limit."""
    remaining = quota - used
    pricing_url = f"{settings.APP_URL}/pricing"
    return await send_email(
        to=to,
        subject=_quota_warning_subject(remaining),
        html_body=_quota_warning_html(
            to=to,
            used=used,
            quota=quota,
            plan=plan,
            pricing_url=pricing_url,
        ),
        text_body=_quota_warning_text(
            to=to,
            used=used,
            quota=quota,
            plan=plan,
            pricing_url=pricing_url,
        ),
    )


# ---------------------------------------------------------------------------
# Mention parsing helper
# ---------------------------------------------------------------------------

# Regex matching @email patterns in comment text, e.g. "@alice@example.com"
_MENTION_RE = re.compile(r"@([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})")


def extract_mentions(text: str) -> list[str]:
    """
    Extract all e-mail addresses mentioned with '@' prefix in *text*.

    Example:
        extract_mentions("Hey @alice@lab.com, check this!")
        # → ["alice@lab.com"]
    """
    return list(set(_MENTION_RE.findall(text)))
