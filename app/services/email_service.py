"""
Email notification service for GenoLens.

Supports:
- Project invitation emails
- Comment mention notifications (@username/@email)
- Comment reply notifications

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
<html lang="fr">
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
      <p>Plateforme d'analyse génomique</p>
    </div>
    <div class="body">
      {content}
    </div>
    <div class="footer">
      Vous recevez cet e-mail car vous utilisez
      <a href="{settings.APP_URL}">GenoLens</a>.
      Ne répondez pas directement à cet e-mail.
    </div>
  </div>
</body>
</html>"""


def _project_invitation_html(
    invitee_email: str,
    inviter_email: str,
    project_name: str,
    access_level: str,
    project_url: str,
) -> str:
    role_label = {
        "ADMIN": "Administrateur",
        "USER": "Collaborateur",
        "VIEWER": "Lecteur",
    }.get(access_level.upper(), access_level)

    content = f"""
      <p>Bonjour,</p>
      <p>
        <strong>{inviter_email}</strong> vous invite à collaborer sur le projet
        <strong>« {project_name} »</strong> avec le rôle
        <strong>{role_label}</strong>.
      </p>
      <p>
        GenoLens vous permet d'explorer et d'analyser des données d'expression génique :
        volcano plots, heatmaps, GSEA, GO enrichment et bien plus.
      </p>
      <a href="{project_url}" class="btn">Accéder au projet →</a>
      <div class="meta">
        <p>
          Si vous n'avez pas encore de compte GenoLens, créez-en un avec l'adresse
          <strong>{invitee_email}</strong> pour accéder automatiquement au projet partagé.
        </p>
        <p>Lien direct : <a href="{project_url}">{project_url}</a></p>
      </div>
    """
    return _base_layout(f"Invitation au projet « {project_name} »", content)


def _project_invitation_text(
    invitee_email: str,
    inviter_email: str,
    project_name: str,
    access_level: str,
    project_url: str,
) -> str:
    return (
        f"Bonjour,\n\n"
        f"{inviter_email} vous invite à collaborer sur le projet « {project_name} » "
        f"avec le rôle {access_level}.\n\n"
        f"Accédez au projet ici : {project_url}\n\n"
        f"Si vous n'avez pas de compte, créez-en un avec l'adresse {invitee_email}.\n\n"
        f"— L'équipe GenoLens"
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
      <p>Bonjour,</p>
      <p>
        <strong>{author_email}</strong> vous a mentionné dans un commentaire
        du projet <strong>« {project_name} »</strong> :
      </p>
      <div class="highlight-box">
        <p>{excerpt_escaped}</p>
      </div>
      <a href="{comment_url}" class="btn">Voir le commentaire →</a>
      <div class="meta">
        <p>Lien direct : <a href="{comment_url}">{comment_url}</a></p>
      </div>
    """
    return _base_layout(f"Vous avez été mentionné dans « {project_name} »", content)


def _mention_notification_text(
    mentioned_email: str,
    author_email: str,
    project_name: str,
    comment_excerpt: str,
    comment_url: str,
) -> str:
    return (
        f"Bonjour,\n\n"
        f"{author_email} vous a mentionné dans un commentaire "
        f"du projet « {project_name} » :\n\n"
        f"  {comment_excerpt}\n\n"
        f"Voir le commentaire : {comment_url}\n\n"
        f"— L'équipe GenoLens"
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
      <p>Bonjour,</p>
      <p>
        <strong>{replier_email}</strong> a répondu à votre commentaire
        dans le projet <strong>« {project_name} »</strong>.
      </p>
      <p style="color:#6b7280; font-size:13px; margin-bottom:4px;">Votre commentaire :</p>
      <div class="highlight-box" style="border-color:#d1d5db; background:#f9fafb;">
        <p style="color:#6b7280;">{orig_escaped}</p>
      </div>
      <p style="color:#6b7280; font-size:13px; margin-bottom:4px;">Réponse de {replier_email} :</p>
      <div class="highlight-box">
        <p>{reply_escaped}</p>
      </div>
      <a href="{comment_url}" class="btn">Voir la réponse →</a>
      <div class="meta">
        <p>Lien direct : <a href="{comment_url}">{comment_url}</a></p>
      </div>
    """
    return _base_layout(f"Réponse à votre commentaire — « {project_name} »", content)


def _reply_notification_text(
    parent_author_email: str,
    replier_email: str,
    project_name: str,
    original_excerpt: str,
    reply_excerpt: str,
    comment_url: str,
) -> str:
    return (
        f"Bonjour,\n\n"
        f"{replier_email} a répondu à votre commentaire "
        f"dans le projet « {project_name} ».\n\n"
        f"Votre commentaire : {original_excerpt}\n\n"
        f"Réponse : {reply_excerpt}\n\n"
        f"Voir la réponse : {comment_url}\n\n"
        f"— L'équipe GenoLens"
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
        subject=f"Invitation au projet « {project_name} » — GenoLens",
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
        subject=f"Vous avez été mentionné dans « {project_name} » — GenoLens",
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
        subject=f"Réponse à votre commentaire — « {project_name} » — GenoLens",
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


def _quota_warning_html(
    to: str,
    used: int,
    quota: int,
    plan: str,
    pricing_url: str,
) -> str:
    remaining = quota - used
    content = f"""
      <p>Bonjour,</p>
      <p>
        Vous avez utilisé <strong>{used} comparaisons</strong> sur <strong>{quota}</strong>
        ce mois-ci (forfait <strong>{plan}</strong>).
      </p>
      <div class="highlight-box">
        <p>Il vous reste <strong>{remaining} comparaison{'s' if remaining > 1 else ''}</strong> pour ce mois.</p>
      </div>
      <p>
        Pour continuer vos analyses sans interruption, vous pouvez passer à un forfait supérieur
        depuis votre espace tarifaire.
      </p>
      <a href="{pricing_url}" class="btn">Voir les forfaits →</a>
      <div class="meta">
        <p>Lien direct : <a href="{pricing_url}">{pricing_url}</a></p>
      </div>
    """
    return _base_layout(f"GenoLens — {remaining} comparaisons restantes ce mois", content)


def _quota_warning_text(
    to: str,
    used: int,
    quota: int,
    plan: str,
    pricing_url: str,
) -> str:
    remaining = quota - used
    return (
        f"Bonjour,\n\n"
        f"Vous avez utilisé {used} comparaisons sur {quota} ce mois-ci (forfait {plan}).\n\n"
        f"Il vous reste {remaining} comparaison{'s' if remaining > 1 else ''} pour ce mois.\n\n"
        f"Pour continuer vos analyses sans interruption, consultez nos forfaits :\n"
        f"{pricing_url}\n\n"
        f"— L'équipe GenoLens"
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
        subject=f"GenoLens — {remaining} comparaisons restantes ce mois",
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
