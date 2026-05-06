"""
Stripe webhook receiver for GenoLens.

Handles:
- checkout.session.completed  → activate subscription
- customer.subscription.updated → sync subscription status
- customer.subscription.deleted → downgrade to BASIC
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

import stripe
from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.config import settings
from app.models.models import SubscriptionPlan, User, UserStatus
from app.services import email_service
from app.services.stripe_service import construct_webhook_event, get_stripe_client

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_user_by_id(db: AsyncSession, user_id: str) -> User | None:
    """Look up a user by their primary key (Supabase Auth UUID)."""
    try:
        uid = UUID(user_id)
    except ValueError:
        return None
    result = await db.execute(select(User).where(User.id == uid))
    return result.scalar_one_or_none()


async def _get_user_by_stripe_customer(db: AsyncSession, customer_id: str) -> User | None:
    """Look up a user by their Stripe customer ID."""
    result = await db.execute(
        select(User).where(User.stripe_customer_id == customer_id)
    )
    return result.scalar_one_or_none()


def _price_id_to_plan(price_id: str) -> SubscriptionPlan:
    """Map a Stripe price ID to a SubscriptionPlan enum value."""
    if price_id == settings.stripe_price_premium_monthly:
        return SubscriptionPlan.PREMIUM
    if price_id == settings.stripe_price_advanced_monthly:
        return SubscriptionPlan.ADVANCED
    raise ValueError(
        f"Unknown Stripe price_id '{price_id}'. "
        "Configure STRIPE_PRICE_PREMIUM_MONTHLY and STRIPE_PRICE_ADVANCED_MONTHLY."
    )


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

async def _handle_checkout_completed(
    session: stripe.checkout.Session,
    db: AsyncSession,
) -> None:
    """
    Fired when a Stripe Checkout Session completes successfully.
    Activates the user's subscription in the database.
    """
    user_id: str | None = session.get("client_reference_id")
    customer_id: str | None = session.get("customer")
    subscription_id: str | None = session.get("subscription")

    if not user_id:
        logger.error(
            "checkout.session.completed: missing client_reference_id — session %s",
            session.get("id"),
        )
        return

    user = await _get_user_by_id(db, user_id)
    if not user:
        logger.error(
            "checkout.session.completed: user not found for id=%s", user_id
        )
        return

    # Idempotency guard — skip if already processed
    if user.stripe_subscription_id == subscription_id:
        logger.info(
            "checkout.session.completed: already processed for user=%s, skipping", user_id
        )
        return

    # Determine the plan by fetching subscription items from Stripe
    plan = SubscriptionPlan.PREMIUM  # safe default
    if subscription_id:
        try:
            client = get_stripe_client()
            sub = await asyncio.to_thread(
                client.v1.subscriptions.retrieve,
                subscription_id,
                params={"expand": ["items.data.price"]},
            )
            items = sub.get("items", {}).get("data", [])
            if items:
                price_id = items[0].get("price", {}).get("id", "")
                plan = _price_id_to_plan(price_id)
        except stripe.StripeError as exc:
            logger.warning(
                "checkout.session.completed: could not retrieve subscription %s: %s",
                subscription_id,
                exc,
            )

    user.stripe_customer_id = customer_id
    user.stripe_subscription_id = subscription_id
    user.subscription_plan = plan
    user.subscription_starts_at = datetime.now(timezone.utc).isoformat()
    user.status = UserStatus.ACTIVE

    db.add(user)
    await db.commit()

    logger.info(
        "checkout.session.completed: activated user=%s plan=%s customer=%s sub=%s",
        user_id, plan, customer_id, subscription_id,
    )

    # Send welcome email — isolated so email failures don't cause Stripe retries
    try:
        await email_service.send_subscription_welcome_email(
            to_email=user.email, plan=plan.value
        )
    except Exception as email_exc:
        logger.warning(
            "checkout.session.completed: welcome email failed for user=%s: %s",
            user_id, email_exc,
        )


async def _handle_subscription_updated(
    subscription: stripe.Subscription,
    db: AsyncSession,
) -> None:
    """
    Fired when a Stripe subscription is modified (plan change, renewal, past_due, etc.).
    Syncs status, plan, and period-end date to the database.
    """
    customer_id: str | None = subscription.get("customer")
    if not customer_id:
        logger.error("customer.subscription.updated: missing customer field")
        return

    user = await _get_user_by_stripe_customer(db, customer_id)
    if not user:
        logger.error(
            "customer.subscription.updated: no user found for customer=%s", customer_id
        )
        return

    sub_status: str = subscription.get("status", "")
    current_period_end: int | None = subscription.get("current_period_end")

    if current_period_end:
        user.subscription_ends_at = datetime.fromtimestamp(
            current_period_end, tz=timezone.utc
        ).isoformat()

    if sub_status == "active":
        user.status = UserStatus.ACTIVE
    elif sub_status in ("past_due", "canceled", "unpaid"):
        user.status = UserStatus.CANCELLED
        user.subscription_plan = SubscriptionPlan.BASIC

    # Detect plan change from price_id
    try:
        items = subscription.get("items", {}).get("data", [])
        if items:
            price_id = items[0].get("price", {}).get("id")
            if price_id:
                new_plan = _price_id_to_plan(price_id)
                user.subscription_plan = new_plan
    except ValueError:
        logger.warning("subscription.updated: unknown price_id, keeping current plan")

    db.add(user)
    await db.commit()

    logger.info(
        "customer.subscription.updated: user=%s status=%s period_end=%s",
        user.id, sub_status, current_period_end,
    )


async def _handle_subscription_deleted(
    subscription: stripe.Subscription,
    db: AsyncSession,
) -> None:
    """
    Fired when a Stripe subscription is fully cancelled/deleted.
    Downgrades the user to BASIC and clears billing identifiers.
    """
    customer_id: str | None = subscription.get("customer")
    if not customer_id:
        logger.error("customer.subscription.deleted: missing customer field")
        return

    user = await _get_user_by_stripe_customer(db, customer_id)
    if not user:
        logger.error(
            "customer.subscription.deleted: no user found for customer=%s", customer_id
        )
        return

    user.subscription_plan = SubscriptionPlan.BASIC
    user.status = UserStatus.CANCELLED
    user.stripe_subscription_id = None

    db.add(user)
    await db.commit()

    logger.info(
        "customer.subscription.deleted: downgraded user=%s to BASIC", user.id
    )

    # Send cancellation email — isolated so email failures don't cause Stripe retries
    try:
        await email_service.send_subscription_cancelled_email(to_email=user.email)
    except Exception as email_exc:
        logger.warning(
            "customer.subscription.deleted: cancellation email failed for user=%s: %s",
            user.id, email_exc,
        )


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@router.post("/webhook", status_code=status.HTTP_200_OK)
async def stripe_webhook(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    stripe_signature: Annotated[str | None, Header(alias="stripe-signature")] = None,
) -> JSONResponse:
    """
    Receive and process Stripe webhook events.

    - No authentication — Stripe calls this endpoint directly.
    - Signature verified via the ``Stripe-Signature`` header.
    - Returns 200 immediately; processing is synchronous but fast.
    """
    payload: bytes = await request.body()

    if not stripe_signature:
        logger.warning("stripe_webhook: missing Stripe-Signature header")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Missing Stripe-Signature header"},
        )

    # --- Verify signature ---
    try:
        event = construct_webhook_event(payload, stripe_signature)
    except stripe.SignatureVerificationError as exc:
        logger.warning("stripe_webhook: invalid signature — %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid Stripe signature"},
        )
    except RuntimeError as exc:
        # Webhook secret not configured
        logger.error("stripe_webhook: configuration error — %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Webhook not configured"},
        )

    event_type: str = event.get("type", "")
    event_data = event.get("data", {}).get("object", {})

    logger.info("stripe_webhook: received event type=%s id=%s", event_type, event.get("id"))

    # --- Dispatch ---
    try:
        if event_type == "checkout.session.completed":
            await _handle_checkout_completed(event_data, db)

        elif event_type == "customer.subscription.updated":
            await _handle_subscription_updated(event_data, db)

        elif event_type == "customer.subscription.deleted":
            await _handle_subscription_deleted(event_data, db)

        else:
            logger.debug("stripe_webhook: ignoring unhandled event type=%s", event_type)
            return JSONResponse(content={"status": "ignored"})

    except Exception as exc:
        # Log and return 500 so Stripe will retry
        logger.exception(
            "stripe_webhook: unhandled error processing event type=%s id=%s: %s",
            event_type, event.get("id"), exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal error; will retry"},
        )

    return JSONResponse(content={"status": "ok"})
