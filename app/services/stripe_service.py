"""
Stripe billing service for GenoLens.

Exposes:
- get_stripe_client()           — initialise and return the Stripe v8+ client
- create_checkout_session()     — create a Stripe Checkout Session for a subscription
- create_billing_portal_session() — create a Stripe Billing Portal session
- construct_webhook_event()     — verify and parse an incoming Stripe webhook payload
"""
import asyncio
import logging

import stripe

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_stripe_client() -> stripe.StripeClient:
    """
    Initialise and return the Stripe SDK client (v8+ style).

    Raises:
        RuntimeError: if STRIPE_SECRET_KEY is not configured.
    """
    if not settings.stripe_secret_key:
        raise RuntimeError(
            "Stripe is not configured. Set STRIPE_SECRET_KEY in your environment."
        )
    return stripe.StripeClient(api_key=settings.stripe_secret_key)


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------

async def create_checkout_session(
    user_id: str,
    user_email: str,
    plan: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe Checkout Session for a subscription plan.

    Args:
        user_id:     Internal user identifier stored as client_reference_id so
                     the webhook can link the session back to our user.
        user_email:  Pre-fills the customer e-mail on the Stripe-hosted page.
        plan:        "PREMIUM" or "ADVANCED" (case-insensitive).
        success_url: URL Stripe redirects to after a successful payment.
        cancel_url:  URL Stripe redirects to if the user cancels.

    Returns:
        The hosted Checkout Session URL.

    Raises:
        ValueError:  if *plan* is not recognised or the matching price ID is
                     not configured.
        stripe.StripeError: on any Stripe API error.
    """
    plan_upper = plan.upper()
    price_map = {
        "PREMIUM": settings.stripe_price_premium_monthly,
        "ADVANCED": settings.stripe_price_advanced_monthly,
    }

    if plan_upper not in price_map:
        raise ValueError(
            f"Unknown plan '{plan}'. Valid options are: {', '.join(price_map)}"
        )

    price_id = price_map[plan_upper]
    if not price_id:
        raise ValueError(
            f"Stripe price ID for plan '{plan}' is not configured. "
            f"Set STRIPE_PRICE_{plan_upper}_MONTHLY in your environment."
        )

    client = get_stripe_client()

    session = await asyncio.to_thread(
        client.v1.checkout.sessions.create,
        {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "customer_email": user_email,
            "client_reference_id": user_id,
            "success_url": success_url,
            "cancel_url": cancel_url,
        },
    )

    logger.info(
        "Checkout session created: session_id=%s user_id=%s plan=%s",
        session.id,
        user_id,
        plan_upper,
    )

    if not session.url:
        raise RuntimeError("Stripe returned a checkout session with no URL")
    return session.url


# ---------------------------------------------------------------------------
# Billing portal
# ---------------------------------------------------------------------------

async def create_billing_portal_session(
    stripe_customer_id: str,
    return_url: str,
) -> str:
    """
    Create a Stripe Billing Portal session for an existing customer.

    Args:
        stripe_customer_id: The Stripe customer ID (cus_...).
        return_url:         URL the customer is redirected to after leaving the portal.

    Returns:
        The Billing Portal session URL.

    Raises:
        stripe.StripeError: on any Stripe API error.
    """
    client = get_stripe_client()

    portal_session = await asyncio.to_thread(
        client.v1.billing_portal.sessions.create,
        {
            "customer": stripe_customer_id,
            "return_url": return_url,
        },
    )

    logger.info(
        "Billing portal session created: customer_id=%s", stripe_customer_id
    )

    if not portal_session.url:
        raise RuntimeError("Stripe returned a billing portal session with no URL")
    return portal_session.url


# ---------------------------------------------------------------------------
# Webhook verification
# ---------------------------------------------------------------------------

def construct_webhook_event(payload: bytes, sig_header: str) -> stripe.Event:
    """
    Verify the Stripe webhook signature and return the parsed Event object.

    Args:
        payload:    Raw request body bytes.
        sig_header: Value of the ``Stripe-Signature`` HTTP header.

    Returns:
        A verified :class:`stripe.Event` instance.

    Raises:
        stripe.SignatureVerificationError: if the signature is invalid.
        RuntimeError: if STRIPE_WEBHOOK_SECRET is not configured.
    """
    if not settings.stripe_webhook_secret:
        raise RuntimeError(
            "Stripe webhook secret is not configured. "
            "Set STRIPE_WEBHOOK_SECRET in your environment."
        )
    client = get_stripe_client()
    return client.construct_event(payload, sig_header, settings.stripe_webhook_secret)
