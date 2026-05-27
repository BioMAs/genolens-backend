"""
Stripe billing service.

Price IDs are stored in environment variables and mapped here.
The actual Stripe products/prices must be created manually in the Stripe Dashboard.
"""
from __future__ import annotations

import logging
from typing import Optional

import stripe

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plan → price_id mapping (set via env vars after Stripe Dashboard setup)
# ---------------------------------------------------------------------------

def _get_plan_price_map() -> dict[str, dict[str, str]]:
    """Build the plan→price mapping from settings. Called at request time so
    env vars are always current (useful for testing)."""
    return {
        "STARTER": {
            "monthly": settings.STRIPE_PRICE_STARTER_MONTHLY,
            "annual": settings.STRIPE_PRICE_STARTER_ANNUAL,
        },
        "TEAM": {
            "monthly": settings.STRIPE_PRICE_TEAM_MONTHLY,
            "annual": settings.STRIPE_PRICE_TEAM_ANNUAL,
        },
    }


def _get_price_to_plan() -> dict[str, str]:
    """Reverse mapping: price_id → plan key. Used in webhook handler."""
    result: dict[str, str] = {}
    for plan, cycles in _get_plan_price_map().items():
        for price_id in cycles.values():
            if price_id:
                result[price_id] = plan
    return result


# ---------------------------------------------------------------------------
# Stripe API helpers
# ---------------------------------------------------------------------------

def _get_stripe_client() -> stripe.Stripe | None:
    """Return a configured stripe client, or None if key is not set."""
    if not settings.STRIPE_SECRET_KEY:
        logger.warning("STRIPE_SECRET_KEY not configured — billing features disabled")
        return None
    return stripe


def _init_stripe() -> None:
    """Configure stripe API key."""
    stripe.api_key = settings.STRIPE_SECRET_KEY


async def create_checkout_session(
    user_id: str,
    user_email: str,
    plan: str,
    billing_cycle: str,
    success_url: str,
    cancel_url: str,
    stripe_customer_id: Optional[str] = None,
) -> str:
    """
    Create a Stripe Checkout session and return the URL.
    Raises ValueError if plan/cycle is invalid or Stripe key is missing.
    """
    _init_stripe()
    if not settings.STRIPE_SECRET_KEY:
        raise ValueError("Stripe not configured. Set STRIPE_SECRET_KEY in environment.")

    plan_map = _get_plan_price_map()
    if plan not in plan_map:
        raise ValueError(f"Unknown plan: {plan}. Must be one of {list(plan_map.keys())}")
    if billing_cycle not in ("monthly", "annual"):
        raise ValueError(f"Unknown billing_cycle: {billing_cycle}")

    price_id = plan_map[plan][billing_cycle]
    if not price_id:
        raise ValueError(
            f"Price ID for {plan}/{billing_cycle} not configured. "
            f"Set STRIPE_PRICE_{plan}_{billing_cycle.upper()} in environment."
        )

    params: dict = {
        "mode": "subscription",
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "client_reference_id": user_id,
        "metadata": {"plan": plan, "billing_cycle": billing_cycle},
    }
    if stripe_customer_id:
        params["customer"] = stripe_customer_id
    else:
        params["customer_email"] = user_email

    session = stripe.checkout.Session.create(**params)
    return session.url


async def create_portal_session(
    stripe_customer_id: str,
    return_url: str,
) -> str:
    """Create a Stripe Billing Portal session and return the URL."""
    _init_stripe()
    if not settings.STRIPE_SECRET_KEY:
        raise ValueError("Stripe not configured. Set STRIPE_SECRET_KEY in environment.")
    session = stripe.billing_portal.Session.create(
        customer=stripe_customer_id,
        return_url=return_url,
    )
    return session.url


def handle_webhook_event(payload: bytes, sig_header: str) -> dict:
    """
    Verify and parse a Stripe webhook event.
    Returns the event dict.
    Raises stripe.error.SignatureVerificationError if signature is invalid.
    """
    _init_stripe()
    event = stripe.Webhook.construct_event(
        payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
    )
    return event
