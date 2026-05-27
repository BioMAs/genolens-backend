"""
Billing endpoints — Stripe checkout, portal, subscription info, and webhook.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.subscription import get_or_create_user
from app.core.config import settings
from app.core.supabase_auth import SupabaseUser
from app.models.models import User, SubscriptionPlan

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])


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
    comparisons_used_this_month: int
    comparisons_quota: int | None
    comparisons_remaining: int | None
    can_use_ai: bool
    can_use_multi_comparison: bool


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
        "comparisons_used_this_month": db_user.comparisons_used_this_month,
        "comparisons_quota": db_user.comparisons_quota,
        "comparisons_remaining": db_user.comparisons_remaining,
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
    from app.services.stripe_service import handle_webhook_event, _get_price_to_plan
    import stripe as stripe_lib

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if not settings.STRIPE_WEBHOOK_SECRET:
        logger.warning("STRIPE_WEBHOOK_SECRET not set — webhook verification skipped")
        # In dev without webhook secret, parse raw JSON
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
        items = subscription.get("items", {}).get("data", [])

        if not items:
            return {"status": "ok"}

        price_id: str = items[0]["price"]["id"]
        price_to_plan = _get_price_to_plan()
        plan_key = price_to_plan.get(price_id)

        if not plan_key:
            logger.warning("Unknown price_id in webhook: %s", price_id)
            return {"status": "ok", "note": f"Unknown price_id: {price_id}"}

        # Find user by stripe_customer_id
        result = await db.execute(
            select(User).where(User.stripe_customer_id == customer_id)
        )
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

        if user:
            try:
                user.subscription_plan = SubscriptionPlan(plan_key)
                user.stripe_customer_id = customer_id
                db.add(user)
                await db.commit()
                logger.info("Updated user %s plan to %s", user.id, plan_key)
            except ValueError:
                logger.error("Invalid plan key from webhook: %s", plan_key)
        else:
            logger.warning("No user found for Stripe customer: %s", customer_id)

    elif event_type == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        customer_id = subscription["customer"]
        result = await db.execute(
            select(User).where(User.stripe_customer_id == customer_id)
        )
        user = result.scalar_one_or_none()
        if user:
            user.subscription_plan = SubscriptionPlan.STARTER
            db.add(user)
            await db.commit()
            logger.info("Subscription cancelled for user %s — reverted to STARTER", user.id)

    return {"status": "ok"}
