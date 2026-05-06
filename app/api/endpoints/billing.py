"""
Billing endpoints for GenoLens.

Provides authenticated routes for Stripe Checkout, Billing Portal, and
subscription status.
"""
from typing import Any, Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.deps.subscription import get_or_create_user
from app.core.config import settings
from app.models.models import User
from app.services import stripe_service

router = APIRouter()

VALID_PLANS = {"PREMIUM", "ADVANCED"}


class CheckoutRequest(BaseModel):
    plan: str  # "PREMIUM" or "ADVANCED"


# ---------------------------------------------------------------------------
# POST /billing/checkout
# ---------------------------------------------------------------------------

@router.post("/checkout")
async def create_checkout(
    body: CheckoutRequest,
    current_user: Annotated[User, Depends(get_or_create_user)],
) -> Any:
    """
    Initiate a Stripe Checkout session for a new subscription.

    Returns a redirect URL to the Stripe-hosted checkout page.
    Returns 409 if the user already has an active Stripe customer (use portal).
    """
    plan = body.plan.upper()

    if plan not in VALID_PLANS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid plan '{body.plan}'. Valid options are: {', '.join(sorted(VALID_PLANS))}",
        )

    if current_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "You already have an active subscription. "
                "Please use the billing portal to manage your plan."
            ),
        )

    success_url = f"{settings.APP_URL}/profile?payment=success"
    cancel_url = f"{settings.APP_URL}/pricing"

    try:
        checkout_url = await stripe_service.create_checkout_session(
            user_id=str(current_user.user_id),
            user_email=current_user.email,
            plan=plan,
            success_url=success_url,
            cancel_url=cancel_url,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment provider error. Please try again later.",
        )

    return {"checkout_url": checkout_url}


# ---------------------------------------------------------------------------
# GET /billing/portal
# ---------------------------------------------------------------------------

@router.get("/portal")
async def create_portal(
    current_user: Annotated[User, Depends(get_or_create_user)],
) -> Any:
    """
    Create a Stripe Billing Portal session for subscription management.

    Returns a redirect URL to the Stripe-hosted billing portal.
    Returns 404 if the user has no Stripe customer record.
    """
    if not current_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found.",
        )

    return_url = f"{settings.APP_URL}/profile"

    try:
        portal_url = await stripe_service.create_billing_portal_session(
            stripe_customer_id=current_user.stripe_customer_id,
            return_url=return_url,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment provider error. Please try again later.",
        )

    return {"portal_url": portal_url}


# ---------------------------------------------------------------------------
# GET /billing/subscription
# ---------------------------------------------------------------------------

@router.get("/subscription")
async def get_subscription(
    current_user: Annotated[User, Depends(get_or_create_user)],
) -> Any:
    """
    Return the current user's subscription state from the database.

    No Stripe API call is made — data is read directly from our DB.
    """
    return {
        "plan": current_user.subscription_plan.value,
        "status": current_user.status.value,
        "subscription_starts_at": current_user.subscription_starts_at,
        "subscription_ends_at": current_user.subscription_ends_at,
        "stripe_customer_id": current_user.stripe_customer_id,
        "ai_interpretations_used": current_user.ai_interpretations_used,
        "ai_tokens_purchased": current_user.ai_tokens_purchased,
        "ai_tokens_used": current_user.ai_tokens_used,
    }
