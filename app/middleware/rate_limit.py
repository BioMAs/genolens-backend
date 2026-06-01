"""
Rate limiting middleware and dependencies for GenOlens.
"""
import logging
from typing import Callable
from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.models.models import SubscriptionPlan

logger = logging.getLogger(__name__)


def get_user_identifier(request: Request) -> str:
    """
    Get user identifier for rate limiting.
    Uses user ID from JWT token if available, otherwise IP address.
    """
    try:
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            # Extract user ID from token using JWT decode (simple extraction without full validation)
            try:
                import jwt
                # Decode without verification (just to extract user ID for rate limiting)
                payload = jwt.decode(token, options={"verify_signature": False})
                user_id = payload.get("sub")
                if user_id:
                    return f"user:{user_id}"
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Could not extract user ID from token: {e}")
    
    # Fallback to IP address
    return f"ip:{get_remote_address(request)}"


def get_rate_limit_by_plan(request: Request) -> str:
    """
    Determine rate limit based on user's subscription plan.
    
    Returns:
        str: Rate limit string (e.g., "100/hour", "500/hour")
    """
    try:
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            user_id = get_user_id_from_token(token)
            
            if user_id:
                # Note: In a real implementation, you would query the database
                # to get the user's subscription plan. For now, we use a default.
                # This is a simplified version for the rate limiter setup.
                # The actual plan lookup would be done through a database query.
                pass
    except Exception:
        pass
    
    # Default rate limit for unauthenticated users
    return "100/hour"


# Create rate limiter instance
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=["1000/hour"],  # Default global limit
    storage_uri="memory://",  # Use in-memory storage (consider Redis for production)
    headers_enabled=True,  # Enable rate limit headers in response
)


# Rate limit configurations by subscription plan
RATE_LIMITS = {
    SubscriptionPlan.STARTER: "100/hour",
    SubscriptionPlan.TEAM: "500/hour",
    SubscriptionPlan.ON_PREMISE: "2000/hour",
}

# Special rate limits for specific endpoints
AI_RATE_LIMIT = "5/minute"  # Strict limit for AI endpoints
UPLOAD_RATE_LIMIT = "10/hour"  # Limit for file uploads
