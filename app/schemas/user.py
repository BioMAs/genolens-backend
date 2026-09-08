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
    analyses_used_this_month: int = 0
    # Computed from @property methods on User model
    analyses_quota: Optional[int] = None       # None = unlimited
    analyses_remaining: Optional[int] = None   # None = unlimited
    # Anciens noms, servis le temps que le frontend deploye bascule. Ils sont
    # alimentes par des proprietes d'alias sur le modele User. A retirer avec
    # elles. Ne pas ecrire de nouveau code contre ces trois champs.
    comparisons_used_this_month: int = 0
    comparisons_quota: Optional[int] = None
    comparisons_remaining: Optional[int] = None
    max_projects: Optional[int] = None
    # Projets POSSÉDÉS uniquement (pas partagés) — même prédicat que la limite
    # appliquée par create_project, pour que le client puisse afficher "used/max"
    # sans se tromper sur ce qui est compté. Un GET /projects mêle owned+shared.
    project_count: int = 0
    max_datasets_per_project: Optional[int] = None
    can_use_ai: bool = False
    can_use_multi_comparison: bool = False
    can_export_advanced: bool = False
    # Add-on modules (unlocked per-user by an admin)
    has_cosmetics_module: bool = False
    has_report_customization: bool = False
    has_scientific_module: bool = False
    has_drug_discovery_module: bool = False
