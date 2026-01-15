"""
Security utilities for Supabase JWT token validation.
"""
import jwt
from typing import Optional
from uuid import UUID
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import BaseModel

from app.core.config import settings
from app.models.models import UserRole, SubscriptionPlan


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenPayload(BaseModel):
    """JWT token payload structure from Supabase."""
    sub: str  # User ID
    email: Optional[str] = None
    role: Optional[str] = None
    aud: str = "authenticated"


class CurrentUser(BaseModel):
    """Current authenticated user with SaaS information."""
    id: UUID
    email: str
    full_name: Optional[str] = None
    role: UserRole
    subscription_tier: SubscriptionPlan
    max_projects: int
    current_project_count: int
    features_access: dict
    is_active: bool


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)
    
    to_encode.update({"exp": expire, "aud": "authenticated"})
    encoded_jwt = jwt.encode(to_encode, settings.SUPABASE_JWT_SECRET, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> TokenPayload:
    """
    Verify and decode a Supabase JWT token.

    Args:
        token: The JWT token string

    Returns:
        TokenPayload: Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Use raw secret string as verified by debug script
        # print(f"DEBUG: Verifying token: {token[:10]}...")
        # print(f"DEBUG: Using secret: {settings.SUPABASE_JWT_SECRET[:5]}...{settings.SUPABASE_JWT_SECRET[-5:]}")
        
        payload = jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated"
        )
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid Token Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        print(f"❌ Token Validation Exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_user_from_token(token_payload: TokenPayload) -> UUID:
    """
    Extract user ID from token payload.
    The full user will be loaded from database by the dependency.

    Args:
        token_payload: Decoded token payload

    Returns:
        UUID: User ID
    """
    return UUID(token_payload.sub)
