"""
Supabase JWT validation utilities.
"""
from typing import Optional
from uuid import UUID
import jwt
from jwt import PyJWTError
from pydantic import BaseModel
import httpx

from app.core.config import settings
from app.models.models import UserRole, SubscriptionPlan


class SupabaseUser(BaseModel):
    """Parsed Supabase user from JWT token."""
    user_id: UUID
    email: str
    role: UserRole = UserRole.USER
    subscription_tier: SubscriptionPlan = SubscriptionPlan.BASIC
    max_projects: int = 1
    features_access: dict = {}
    user_metadata: dict = {}


async def fetch_user_role_from_supabase(user_id: UUID) -> Optional[str]:
    """
    Fetch user role from Supabase profiles table.

    Args:
        user_id: User UUID

    Returns:
        Role string if found, None otherwise
    """
    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"id": f"eq.{user_id}", "select": "role"}
            )

            if response.status_code == 200:
                data = response.json()
                # print(f"DEBUG: Fetched role for user {user_id}: {data}")
                if data and len(data) > 0:
                    return data[0].get("role")
            else:
                print(f"DEBUG: Failed to fetch role. Status: {response.status_code}, Body: {response.text}")

        return None
    except Exception as e:
        print(f"Error fetching user role from Supabase: {repr(e)}")
        return None


async def verify_supabase_token(token: str) -> Optional[SupabaseUser]:
    """
    Verify Supabase JWT token and extract user information.
    Fetches the user role from the profiles table in Supabase.

    Args:
        token: JWT token from Authorization header

    Returns:
        SupabaseUser if token is valid, None otherwise
    """
    try:
        # Decode JWT using Supabase JWT secret
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            options={"verify_aud": True}
        )

        # Extract user information from JWT claims
        user_id = UUID(payload.get("sub"))
        email = payload.get("email")

        if not user_id or not email:
            return None

        # Fetch role from Supabase profiles table
        role_str = await fetch_user_role_from_supabase(user_id)

        # Parse role or default to USER
        # Convert to uppercase to match Python Enum values
        try:
            role = UserRole(role_str.upper()) if role_str else UserRole.USER
        except (ValueError, AttributeError):
            # If role from DB is invalid or None, default to USER
            role = UserRole.USER

        # Extract user metadata for subscription info (fallback)
        user_metadata = payload.get("user_metadata", {})
        
        # Handle subscription plan mapping
        sub_plan_str = user_metadata.get("subscription_tier", "BASIC").upper()
        if sub_plan_str == "FREE":
            sub_plan_str = "BASIC"
            
        try:
            subscription_plan = SubscriptionPlan(sub_plan_str)
        except ValueError:
            subscription_plan = SubscriptionPlan.BASIC

        return SupabaseUser(
            user_id=user_id,
            email=email,
            role=role,
            subscription_tier=subscription_plan,
            max_projects=user_metadata.get("max_projects", 1),
            features_access=user_metadata.get("features_access", {}),
            user_metadata=user_metadata
        )

    except PyJWTError:
        return None
    except (ValueError, KeyError):
        return None
