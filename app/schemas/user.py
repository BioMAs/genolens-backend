from uuid import UUID
from typing import Optional
from pydantic import BaseModel, EmailStr
from app.models.models import UserRole, SubscriptionPlan

class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = True

class UserCreate(UserBase):
    email: EmailStr
    id: UUID

class UserInDBBase(UserBase):
    id: UUID
    role: UserRole
    subscription_plan: SubscriptionPlan
    ai_interpretations_used: int
    ai_tokens_purchased: int
    ai_tokens_used: int

    class Config:
        from_attributes = True

class User(UserInDBBase):
    pass

class UserSelf(User):
    """Schema for returning the user's own profile with subscription details"""
    pass
