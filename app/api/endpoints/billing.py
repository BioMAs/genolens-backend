"""
Billing endpoints — Stripe checkout, portal, subscription info, and webhook.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.subscription import get_or_create_user
from app.core.config import settings
from app.core.supabase_auth import SupabaseUser
from app.models.models import SubscriptionPlan, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

#: Environments where an unsigned Stripe webhook may be processed.
#:
#: The webhook is unauthenticated by design — Stripe proves itself with a
#: signature, not a token. The fallback that parses the payload when
#: STRIPE_WEBHOOK_SECRET is unset used to apply EVERYWHERE, so with the secret
#: missing in production anyone could forge an event and grant themselves a
#: plan. That is worse than the self-service plan endpoint that was removed:
#: it needs no account at all.
#:
#: An allowlist rather than `not is_production`, so admitting a new environment
#: is a deliberate edit. Staging handles real Stripe traffic and must verify.
_UNVERIFIED_WEBHOOK_ENVIRONMENTS = frozenset({"development", "test"})


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    plan: str
    billing_cycle: str = "monthly"


class CheckoutResponse(BaseModel):
    checkout_url: str


class PortalResponse(BaseModel):
    portal_url: str


class SubscriptionResponse(BaseModel):
    plan: str
    is_active: bool
    stripe_customer_id: str | None
    # Colonnes String(50) portant de l'ISO 8601, pas des DateTime — le client
    # les formate. Elles existaient sur le modèle sans être servies, ce qui
    # rendait invisible le bloc « Renews » du dashboard.
    subscription_starts_at: str | None
    subscription_ends_at: str | None
    analyses_used_this_month: int
    analyses_quota: int | None
    analyses_remaining: int | None
    # Anciens noms, servis le temps que le frontend deploye bascule. Backend et
    # frontend se deploient separement : les retirer dans le meme lot que leur
    # remplacement casserait la version en ligne entre les deux deploiements.
    # A retirer une fois le frontend passe.
    comparisons_used_this_month: int
    comparisons_quota: int | None
    comparisons_remaining: int | None
    can_use_ai: bool
    can_use_multi_comparison: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subscription_period_end(subscription: dict) -> Optional[int]:
    """Unix timestamp at which the subscription's current billing period ends.

    Read from two places on purpose. Stripe moved `current_period_end` off the
    Subscription object and onto its items in API version 2025-03-31.basil, and
    nothing here pins an API version — the shape therefore depends on the
    account's default, which can change under us. Reading only the top level
    (as the abandoned feature/stripe-integration branch did) silently yields
    None on a modern account, leaving the renewal date empty: exactly the bug
    this is meant to fix.

    With several items (a plan plus add-ons) the periods can differ; access
    ends at the latest of them.
    """
    top_level = subscription.get("current_period_end")
    if top_level:
        return int(top_level)

    per_item = [
        int(item["current_period_end"])
        for item in subscription.get("items", {}).get("data", [])
        if item.get("current_period_end")
    ]
    return max(per_item) if per_item else None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    body: CheckoutRequest,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    db_user: Annotated[User, Depends(get_or_create_user)],
) -> Any:
    """
    Create a Stripe Checkout session for the given plan and billing cycle.
    Returns the checkout URL to redirect the user to.
    """
    from app.services.stripe_service import create_checkout_session

    success_url = f"{settings.APP_URL}/billing/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{settings.APP_URL}/pricing"

    try:
        url = await create_checkout_session(
            user_id=str(db_user.id),
            user_email=db_user.email,
            plan=body.plan.upper(),
            billing_cycle=body.billing_cycle,
            success_url=success_url,
            cancel_url=cancel_url,
            stripe_customer_id=db_user.stripe_customer_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("Stripe checkout error for user %s: %s", db_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create checkout session. Please try again.",
        )

    return {"checkout_url": url}


@router.get("/portal", response_model=PortalResponse)
async def billing_portal(
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    db_user: Annotated[User, Depends(get_or_create_user)],
) -> Any:
    """
    Create a Stripe Billing Portal session.
    The user must already have a stripe_customer_id.
    """
    from app.services.stripe_service import create_portal_session

    if not db_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription found. Please subscribe first.",
        )

    return_url = f"{settings.APP_URL}/pricing"

    try:
        url = await create_portal_session(
            stripe_customer_id=db_user.stripe_customer_id,
            return_url=return_url,
        )
    except Exception as exc:
        logger.error("Stripe portal error for user %s: %s", db_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to open billing portal. Please try again.",
        )

    return {"portal_url": url}


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    db_user: Annotated[User, Depends(get_or_create_user)],
) -> Any:
    """Get the current user's subscription info and quota metrics."""
    return {
        "plan": db_user.subscription_plan.value,
        "is_active": db_user.is_active,
        "stripe_customer_id": db_user.stripe_customer_id,
        "subscription_starts_at": db_user.subscription_starts_at,
        "subscription_ends_at": db_user.subscription_ends_at,
        "analyses_used_this_month": db_user.analyses_used_this_month,
        "analyses_quota": db_user.analyses_quota,
        "analyses_remaining": db_user.analyses_remaining,
        # Alias depreciés — voir SubscriptionResponse.
        "comparisons_used_this_month": db_user.analyses_used_this_month,
        "comparisons_quota": db_user.analyses_quota,
        "comparisons_remaining": db_user.analyses_remaining,
        "can_use_ai": db_user.can_use_ai,
        "can_use_multi_comparison": db_user.can_use_multi_comparison,
    }


@router.post("/webhook", status_code=status.HTTP_200_OK)
async def stripe_webhook(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Handle Stripe webhook events.
    Updates subscription_plan and stripe_customer_id on the User when a
    subscription is created or updated.
    """
    import stripe as stripe_lib

    from app.services.stripe_service import _get_price_to_plan, handle_webhook_event

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if not settings.STRIPE_WEBHOOK_SECRET:
        if settings.ENVIRONMENT.lower() not in _UNVERIFIED_WEBHOOK_ENVIRONMENTS:
            # Fail closed. An unverifiable event must never reach the handler:
            # it can change a subscription plan and a renewal date.
            logger.error(
                "STRIPE_WEBHOOK_SECRET is not configured in environment %r — "
                "refusing the webhook rather than trusting an unsigned payload",
                settings.ENVIRONMENT,
            )
            # 503, not 400: the payload may well be genuine and the fault is
            # ours, so Stripe should keep retrying until the secret is set
            # instead of giving up on a real event.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Webhook verification is not configured.",
            )
        logger.warning(
            "STRIPE_WEBHOOK_SECRET not set — signature verification skipped "
            "(allowed in %r only)", settings.ENVIRONMENT,
        )
        import json

        try:
            event = json.loads(payload)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")
    else:
        try:
            event = handle_webhook_event(payload, sig_header)
        except stripe_lib.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = event.get("type", "")
    logger.info("Stripe webhook received: %s", event_type)

    # Handle subscription events
    if event_type in ("customer.subscription.created", "customer.subscription.updated"):
        subscription = event["data"]["object"]
        customer_id: str = subscription["customer"]

        # Resolve the user FIRST. The plan lookup used to come first and bailed
        # out early on an empty item list or an unrecognised price, so nothing
        # downstream ran — including the renewal-date sync below, which must
        # happen whether or not we recognise what was bought.
        result = await db.execute(select(User).where(User.stripe_customer_id == customer_id))
        user = result.scalar_one_or_none()

        if not user:
            # Try to match by client_reference_id stored in checkout session metadata
            client_ref = subscription.get("metadata", {}).get("client_reference_id")
            if client_ref:
                from uuid import UUID

                try:
                    result = await db.execute(select(User).where(User.id == UUID(client_ref)))
                    user = result.scalar_one_or_none()
                except ValueError:
                    pass

        if not user:
            logger.warning("No user found for Stripe customer: %s", customer_id)
            return {"status": "ok", "note": "no matching user"}

        # Renewal date. Nothing populated this from Stripe before, so the
        # "Renewal Date" row the profile page renders was blank for every
        # Stripe customer and only ever filled in by an admin by hand.
        # subscription_ends_at is a String(50) column holding ISO 8601, per
        # app/services/account_service.py.
        period_end = _subscription_period_end(subscription)
        if period_end:
            user.subscription_ends_at = datetime.fromtimestamp(
                period_end, tz=timezone.utc
            ).isoformat()
        else:
            logger.warning(
                "No current_period_end on subscription for customer %s "
                "(neither top-level nor on items)",
                customer_id,
            )

        # Plan, when we recognise the price.
        items = subscription.get("items", {}).get("data", [])
        plan_key = None
        if items:
            price_id: str = items[0]["price"]["id"]
            plan_key = _get_price_to_plan().get(price_id)
            if not plan_key:
                logger.warning("Unknown price_id in webhook: %s", price_id)

        if plan_key:
            try:
                user.subscription_plan = SubscriptionPlan(plan_key)
                logger.info("Updated user %s plan to %s", user.id, plan_key)
            except ValueError:
                logger.error("Invalid plan key from webhook: %s", plan_key)

        user.stripe_customer_id = customer_id
        db.add(user)
        await db.commit()

    elif event_type == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        customer_id = subscription["customer"]
        result = await db.execute(select(User).where(User.stripe_customer_id == customer_id))
        user = result.scalar_one_or_none()
        if user:
            user.subscription_plan = SubscriptionPlan.STARTER
            db.add(user)
            await db.commit()
            logger.info("Subscription cancelled for user %s — reverted to STARTER", user.id)

    return {"status": "ok"}
