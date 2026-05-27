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
    """Schema for returning the user's own profile with quota details."""
    # Monthly quota tracking (column on User model)
    comparisons_used_this_month: int = 0
    # Computed from @property methods on User model
    comparisons_quota: Optional[int] = None       # None = unlimited
    comparisons_remaining: Optional[int] = None   # None = unlimited
    max_projects: Optional[int] = None
    max_datasets_per_project: Optional[int] = None
    can_use_ai: bool = False
    can_use_multi_comparison: bool = False
    can_export_advanced: bool = False
