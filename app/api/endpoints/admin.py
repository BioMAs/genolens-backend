"""
Admin endpoints for user and system management.
Requires ADMIN role.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import httpx

from app.api.deps import get_db, require_admin
from app.core.supabase_auth import SupabaseUser
from app.core.config import settings
from app.models.models import Project, Dataset, ProjectMember, User, UserRole as UserRoleEnum, SubscriptionPlan, AIUsageLog
from pydantic import BaseModel


router = APIRouter(prefix="/admin", tags=["Admin"])


# Pydantic Schemas
class UserProfile(BaseModel):
    """User profile from Supabase and local DB."""
    id: UUID
    email: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str
    subscription_plan: str
    ai_interpretations_used: int = 0
    ai_tokens_purchased: int = 0
    ai_tokens_used: int = 0
    ai_interpretations_remaining: int = 0
    created_at: str
    updated_at: str
    last_sign_in_at: Optional[str] = None
    confirmed_at: Optional[str] = None

class SubscriptionUpdate(BaseModel):
    """Schema for updating subscription."""
    plan: SubscriptionPlan

class TokenAdd(BaseModel):
    """Schema for adding AI tokens."""
    tokens: int

class AIUsageLogResponse(BaseModel):
    """Schema for AI usage log."""
    id: UUID
    user_id: UUID
    dataset_id: Optional[UUID] = None
    action_type: str
    comparison_name: Optional[str] = None
    model_used: str
    tokens_used: int
    was_free: bool
    created_at: str
    user_email: Optional[str] = None


class UserRoleUpdate(BaseModel):
    """Schema for updating user role."""
    role: str


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "user"


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    full_name: Optional[str] = None
    role: Optional[str] = None


class SystemStats(BaseModel):
    """System statistics."""
    total_users: int
    total_projects: int
    total_datasets: int
    active_users: int
    users_by_plan: dict[str, int]
    estimated_revenue: float


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: Optional[str] = None
    description: Optional[str] = None
    owner_id: Optional[UUID] = None


class ProjectMemberCreate(BaseModel):
    """Schema for adding a project member."""
    user_id: UUID
    access_level: str = "VIEWER"


class ProjectMemberResponse(BaseModel):
    """Project member with user information."""
    id: UUID
    project_id: UUID
    user_id: UUID
    access_level: str
    created_at: str
    updated_at: str
    user_email: Optional[str] = None
    user_full_name: Optional[str] = None


# Admin Endpoints

@router.get("/users", response_model=List[UserProfile])
async def list_users(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    List all users with their profiles, auth info, and subscription status.
    Admin only.
    """
    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            # Get local user stats first (including subscription info)
            query = select(User)
            result = await db.execute(query)
            local_users = {str(u.id): u for u in result.scalars().all()}
            
            # Get profiles
            profiles_response = await client.get(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"select": "id,full_name,avatar_url,role,created_at,updated_at"}
            )

            if profiles_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch profiles from Supabase"
                )

            profiles = profiles_response.json()

            # Get auth users for email and last_sign_in
            auth_response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users",
                headers=headers
            )

            auth_users = {}
            if auth_response.status_code == 200:
                auth_data = auth_response.json()
                users_list = auth_data.get("users", []) if isinstance(auth_data, dict) else auth_data
                for user in users_list:
                    auth_users[user["id"]] = {
                        "email": user.get("email"),
                        "last_sign_in_at": user.get("last_sign_in_at"),
                        "confirmed_at": user.get("confirmed_at")
                    }

            # Merge profile, auth data, and local subscription data
            result = []
            for profile in profiles:
                user_id = profile["id"]
                auth_info = auth_users.get(user_id, {})
                local_user = local_users.get(user_id)
                
                # Default values if user not in local DB yet
                sub_plan = "BASIC"
                ai_used = 0
                ai_purchased = 0
                ai_tokens_used = 0
                ai_remaining = 0
                
                if local_user:
                    sub_plan = local_user.subscription_plan.value
                    ai_used = local_user.ai_interpretations_used
                    ai_purchased = local_user.ai_tokens_purchased
                    ai_tokens_used = local_user.ai_tokens_used
                    ai_remaining = local_user.ai_interpretations_remaining

                result.append(UserProfile(
                    id=UUID(user_id),
                    email=auth_info.get("email"),
                    full_name=profile.get("full_name"),
                    avatar_url=profile.get("avatar_url"),
                    role=profile.get("role", "USER"),
                    subscription_plan=sub_plan,
                    ai_interpretations_used=ai_used,
                    ai_interpretations_remaining=ai_remaining,
                    ai_tokens_purchased=ai_purchased,
                    ai_tokens_used=ai_tokens_used,
                    created_at=profile.get("created_at"),
                    updated_at=profile.get("updated_at"),
                    last_sign_in_at=auth_info.get("last_sign_in_at"),
                    confirmed_at=auth_info.get("confirmed_at")
                ))
            
            return result
    except Exception as e:
        print(f"Error fetching users: {e}")
        # In case of error (e.g. Supabase connection), try to return local users at least
        return []


@router.get("/users/{user_id}", response_model=UserProfile)
async def get_user_details(
    user_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific user profile.
    Admin only.
    """
    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"id": f"eq.{user_id}", "select": "id,full_name,avatar_url,role,created_at,updated_at"}
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    profile = data[0]
                    
                    # Get local DB data
                    query = select(User).where(User.id == user_id)
                    result = await db.execute(query)
                    local_user = result.scalar_one_or_none()
                    
                    # Get auth email
                    auth_response = await client.get(
                        f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                        headers=headers
                    )
                    
                    email = None
                    last_sign_in = None
                    confirmed_at = None
                    
                    if auth_response.status_code == 200:
                        user_data = auth_response.json()
                        email = user_data.get("email")
                        last_sign_in = user_data.get("last_sign_in_at")
                        confirmed_at = user_data.get("confirmed_at")

                    # Default subscription data
                    sub_plan = "BASIC"
                    ai_used = 0
                    ai_remaining = 0
                    ai_purchased = 0
                    ai_tokens_used = 0

                    if local_user:
                        sub_plan = local_user.subscription_plan.value
                        ai_used = local_user.ai_interpretations_used
                        ai_remaining = local_user.ai_interpretations_remaining
                        ai_purchased = local_user.ai_tokens_purchased
                        ai_tokens_used = local_user.ai_tokens_used

                    return UserProfile(
                        id=UUID(profile["id"]),
                        email=email,
                        full_name=profile.get("full_name"),
                        avatar_url=profile.get("avatar_url"),
                        role=profile.get("role", "USER"),
                        subscription_plan=sub_plan,
                        ai_interpretations_used=ai_used,
                        ai_interpretations_remaining=ai_remaining,
                        ai_tokens_purchased=ai_purchased,
                        ai_tokens_used=ai_tokens_used,
                        created_at=profile.get("created_at"),
                        updated_at=profile.get("updated_at"),
                        last_sign_in_at=last_sign_in,
                        confirmed_at=confirmed_at
                    )
            
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching user details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/users/{user_id}/subscription", response_model=UserProfile)
async def update_user_subscription(
    user_id: UUID,
    sub_update: SubscriptionUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a user's subscription plan.
    Admin only.
    """
    from app.api.deps.subscription import get_or_create_user as get_or_create_user_dep
    
    # We can't reuse the dependency easily here as it depends on current user context
    # So we manually fetch/create the target user
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        # If user doesn't exist in local DB (e.g. only in Supabase), create them
        # We need to fetch email/name from Supabase first
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers=headers
            )
            if user_response.status_code != 200:
                raise HTTPException(status_code=404, detail="User not found in Supabase")
            
            user_data = user_response.json()
            
            user = User(
                id=user_id,
                email=user_data.get("email"),
                full_name=user_data.get("user_metadata", {}).get("full_name"),
                role=UserRoleEnum.USER,
                subscription_plan=sub_update.plan
            )
            db.add(user)
    else:
        user.subscription_plan = sub_update.plan
    
    await db.commit()
    await db.refresh(user)
    
    # Return full profile (reuse get_user_details logic)
    return await get_user_details(user_id, current_user, db)


@router.post("/users/{user_id}/tokens")
async def add_ai_tokens(
    user_id: UUID,
    token_data: TokenAdd,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Add purchased AI tokens to a user.
    Admin only.
    """
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        # Create user if missing locally (basic plan)
        # This simplifies things, assumes admin knows what they are doing
        raise HTTPException(
            status_code=404, 
            detail="User not initialized in local database. Please update subscription first."
        )
    
    user.ai_tokens_purchased += token_data.tokens
    await db.commit()
    await db.refresh(user)
    
    return {"message": f"Added {token_data.tokens} tokens", "total_purchased": user.ai_tokens_purchased}


@router.get("/ai-usage", response_model=List[AIUsageLogResponse])
async def get_ai_usage_logs(
    limit: int = 100,
    user_id: Optional[UUID] = None,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent AI usage logs.
    Admin only.
    """
    query = select(AIUsageLog).order_by(AIUsageLog.created_at.desc()).limit(limit)
    
    if user_id:
        query = query.where(AIUsageLog.user_id == user_id)
        
    result = await db.execute(query)
    logs = result.scalars().all()
    
    # We need to join with user emails, which are in Supabase or local User table
    # Let's try local User table first as it's faster
    user_ids = list(set(log.user_id for log in logs))
    
    user_query = select(User.id, User.email).where(User.id.in_(user_ids))
    user_result = await db.execute(user_query)
    user_map = {u.id: u.email for u in user_result.all()}
    
    return [
        AIUsageLogResponse(
            id=log.id,
            user_id=log.user_id,
            dataset_id=log.dataset_id,
            action_type=log.action_type,
            comparison_name=log.comparison_name,
            model_used=log.model_used,
            tokens_used=log.tokens_used,
            was_free=log.was_free,
            created_at=log.created_at.isoformat(),
            user_email=user_map.get(log.user_id)
        )
        for log in logs
    ]


@router.patch("/users/{user_id}/role", response_model=UserProfile)
async def update_user_role(
    user_id: UUID,
    role_update: UserRoleUpdate,
    current_user: SupabaseUser = Depends(require_admin)
):
    """
    Update a user's role.
    Admin only.
    """
    # Validate role
    from app.models.models import UserRole
    try:
        UserRole(role_update.role.upper())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join([r.value for r in UserRole])}"
        )

    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

        async with httpx.AsyncClient() as client:
            # Update role in profiles table
            response = await client.patch(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"id": f"eq.{user_id}"},
                json={"role": role_update.role.lower()}
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to update user role: {response.text}"
                )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user role: {str(e)}"
        )


@router.get("/stats", response_model=SystemStats)
async def get_system_stats(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get system statistics.
    Admin only.
    """
    try:
        # Count projects
        result = await db.execute(select(func.count()).select_from(Project))
        total_projects = result.scalar() or 0

        # Count datasets
        result = await db.execute(select(func.count()).select_from(Dataset))
        total_datasets = result.scalar() or 0

        # Count unique project owners (active users)
        result = await db.execute(select(Project.owner_id).distinct())
        active_users = len(result.scalars().all())

        # Get local users for detailed stats
        result = await db.execute(select(User))
        local_users = result.scalars().all()
        
        # Calculate revenue and plan distribution
        users_by_plan = {"BASIC": 0, "PREMIUM": 0, "ADVANCED": 0}
        estimated_revenue = 0.0
        
        # Pricing model (for estimation)
        PRICES = {
            "BASIC": 0,
            "PREMIUM": 29.0, # Monthly
            "ADVANCED": 99.0, # Monthly
            "TOKEN": 0.10  # Per token
        }

        for u in local_users:
            plan = u.subscription_plan.value if hasattr(u.subscription_plan, 'value') else u.subscription_plan
            users_by_plan[plan] = users_by_plan.get(plan, 0) + 1
            
            # Monthly recurring revenue
            estimated_revenue += PRICES.get(plan, 0)
            
            # One-time token revenue
            estimated_revenue += (u.ai_tokens_purchased * PRICES["TOKEN"])

        # Fetch total users from Supabase (including those not in local DB)
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"select": "id"}
            )

            total_users = 0
            if response.status_code == 200:
                total_users = len(response.json())
        
        # Adjust BASIC count for users not in local DB
        local_users_count = sum(users_by_plan.values())
        if total_users > local_users_count:
            users_by_plan["BASIC"] += (total_users - local_users_count)

        return SystemStats(
            total_users=total_users,
            total_projects=total_projects,
            total_datasets=total_datasets,
            active_users=active_users,
            users_by_plan=users_by_plan,
            estimated_revenue=round(estimated_revenue, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching system stats: {str(e)}"
        )


@router.post("/users", response_model=UserProfile)
async def create_user(
    user_data: UserCreate,
    current_user: SupabaseUser = Depends(require_admin)
):
    """
    Create a new user in Supabase.
    Admin only.
    """
    # Validate role
    from app.models.models import UserRole
    try:
        UserRole(user_data.role.upper())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {', '.join([r.value for r in UserRole])}"
        )

    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            # Create user via Supabase Admin API
            response = await client.post(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users",
                headers=headers,
                json={
                    "email": user_data.email,
                    "password": user_data.password,
                    "email_confirm": True,
                    "user_metadata": {
                        "full_name": user_data.full_name
                    }
                }
            )

            if response.status_code not in [200, 201]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to create user: {response.text}"
                )

            user_response = response.json()
            user_id = user_response.get("id")

            # Create profile entry
            profile_response = await client.post(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers={**headers, "Prefer": "return=representation"},
                json={
                    "id": user_id,
                    "full_name": user_data.full_name,
                    "role": user_data.role.lower()
                }
            )

            if profile_response.status_code not in [200, 201]:
                # Try to rollback user creation
                await client.delete(
                    f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                    headers=headers
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user profile"
                )

            return profile_response.json()[0] if isinstance(profile_response.json(), list) else profile_response.json()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )


@router.patch("/users/{user_id}", response_model=UserProfile)
async def update_user(
    user_id: UUID,
    user_update: UserUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a user's profile.
    Admin only.
    """
    # Validate role if provided
    if user_update.role:
        from app.models.models import UserRole
        try:
            UserRole(user_update.role.upper())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {', '.join([r.value for r in UserRole])}"
            )

    try:
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

        # Build update payload
        update_data = {}
        if user_update.full_name is not None:
            update_data["full_name"] = user_update.full_name
        if user_update.role is not None:
            update_data["role"] = user_update.role.lower()

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{settings.SUPABASE_URL}/rest/v1/profiles",
                headers=headers,
                params={"id": f"eq.{user_id}"},
                json=update_data
            )

            if response.status_code == 200:
                # Return full updated profile
                return await get_user_details(user_id, current_user, db)
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to update user: {response.text}"
                )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user: {str(e)}"
        )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a user from Supabase and local database.
    Admin only.
    WARNING: This permanently deletes the user and all their data (projects, legacy data, etc).
    """
    # Prevent self-deletion
    if str(user_id) == str(current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    try:
        from sqlalchemy import text, delete
        
        # 1. Delete legacy 'sequencing_projects' (if exists)
        # This table causes FK violation if not cleaned up
        try:
            # Use nested transaction to prevent breaking the main transaction if table doesn't exist
            async with db.begin_nested():
                await db.execute(text("DELETE FROM sequencing_projects WHERE user_id = :uid"), {"uid": user_id})
        except Exception as e:
            # Table might not exist or other error, log and continue
            print(f"Warning cleaning legacy projects: {e}")

        # 2. Delete modern 'projects' owned by user
        # This will cascade to datasets, samples, members
        await db.execute(delete(Project).where(Project.owner_id == user_id))
        
        # 3. Delete local 'users' record
        await db.execute(delete(User).where(User.id == user_id))
        
        await db.commit()

        # 4. Delete user from Supabase Auth
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            # 4.1 Clean up remote Supabase tables (legacy)
            # 'sequencing_projects' exists in Supabase DB and blocks deletion
            try:
                await client.delete(
                    f"{settings.SUPABASE_URL}/rest/v1/sequencing_projects",
                    headers=headers,
                    params={"user_id": f"eq.{user_id}"}
                )
            except Exception as e:
                print(f"Warning: Failed to clean up remote sequencing_projects: {e}")

            # 4.2 Delete user via Supabase Admin API
            response = await client.delete(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers=headers
            )

            if response.status_code == 200:
                return {"message": "User deleted successfully"}
            elif response.status_code == 404:
                # If not found in Auth but we cleaned up local DB, consider it success or just gone
                return {"message": "User deleted (was not found in Supabase Auth)"}
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to delete user from Supabase: {response.text}"
                )

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting user: {str(e)}"
        )


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a project.
    Admin only.
    WARNING: This permanently deletes the project and all associated datasets.
    """
    try:
        # Find project
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Delete project (cascade will handle datasets)
        await db.delete(project)
        await db.commit()

        return {"message": "Project deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting project: {str(e)}"
        )


@router.get("/projects")
async def list_all_projects(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    List all projects in the system with owner information.
    Admin only.
    """
    try:
        result = await db.execute(select(Project).offset(skip).limit(limit))
        projects = result.scalars().all()

        # Get owner emails from Supabase
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            # Get auth users for owner emails
            auth_response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users",
                headers=headers
            )

            owner_info = {}
            if auth_response.status_code == 200:
                auth_data = auth_response.json()
                users_list = auth_data.get("users", []) if isinstance(auth_data, dict) else auth_data
                for user in users_list:
                    owner_info[user["id"]] = {
                        "email": user.get("email"),
                        "full_name": user.get("user_metadata", {}).get("full_name")
                    }

        return [
            {
                "id": str(p.id),
                "name": p.name,
                "description": p.description,
                "owner_id": str(p.owner_id),
                "owner_email": owner_info.get(str(p.owner_id), {}).get("email"),
                "owner_full_name": owner_info.get(str(p.owner_id), {}).get("full_name"),
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None
            }
            for p in projects
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching projects: {str(e)}"
        )


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: UUID,
    project_update: ProjectUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a project's name or description.
    Admin only.
    """
    try:
        # Find project
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Update fields
        if project_update.name is not None:
            project.name = project_update.name
        if project_update.description is not None:
            project.description = project_update.description
        if project_update.owner_id is not None:
            # Verify user exists locally
            user_result = await db.execute(select(User).where(User.id == project_update.owner_id))
            user = user_result.scalar_one_or_none()
            
            if not user:
                # Try to fetch from Supabase to auto-create local user
                try:
                    headers = {
                        "apikey": settings.SUPABASE_KEY,
                        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
                        "Content-Type": "application/json",
                    }
                    async with httpx.AsyncClient() as client:
                        sb_res = await client.get(
                            f"{settings.SUPABASE_URL}/auth/v1/admin/users/{project_update.owner_id}",
                            headers=headers
                        )
                        if sb_res.status_code == 200:
                            sb_user = sb_res.json()
                            # Create local user
                            user = User(
                                id=project_update.owner_id,
                                email=sb_user.get("email"),
                                full_name=sb_user.get("user_metadata", {}).get("full_name"),
                                role=UserRoleEnum.USER, # Default to USER
                                subscription_plan=SubscriptionPlan.BASIC
                            )
                            db.add(user)
                            # We don't commit here, we'll commit with the project update
                except Exception as e:
                    print(f"Error fetching user from Supabase: {e}")

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User {project_update.owner_id} not found"
                )
            project.owner_id = project_update.owner_id

        db.add(project)
        await db.commit()
        await db.refresh(project)

        return {
            "id": str(project.id),
            "name": project.name,
            "description": project.description,
            "owner_id": str(project.owner_id),
            "created_at": project.created_at.isoformat() if project.created_at else None,
            "updated_at": project.updated_at.isoformat() if project.updated_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating project: {str(e)}"
        )


@router.get("/projects/{project_id}/members", response_model=List[ProjectMemberResponse])
async def list_project_members(
    project_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    List all members of a project with their user information.
    Admin only.
    """
    try:
        # Verify project exists
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Get project members
        result = await db.execute(select(ProjectMember).where(ProjectMember.project_id == project_id))
        members = result.scalars().all()

        # Get user information from Supabase
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            auth_response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users",
                headers=headers
            )

            user_info = {}
            if auth_response.status_code == 200:
                auth_data = auth_response.json()
                users_list = auth_data.get("users", []) if isinstance(auth_data, dict) else auth_data
                for user in users_list:
                    user_info[user["id"]] = {
                        "email": user.get("email"),
                        "full_name": user.get("user_metadata", {}).get("full_name")
                    }

        return [
            ProjectMemberResponse(
                id=m.id,
                project_id=m.project_id,
                user_id=m.user_id,
                access_level=m.access_level.value if hasattr(m.access_level, 'value') else m.access_level,
                created_at=m.created_at.isoformat() if m.created_at else None,
                updated_at=m.updated_at.isoformat() if m.updated_at else None,
                user_email=user_info.get(str(m.user_id), {}).get("email"),
                user_full_name=user_info.get(str(m.user_id), {}).get("full_name")
            )
            for m in members
        ]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching project members: {str(e)}"
        )


@router.post("/projects/{project_id}/members", response_model=ProjectMemberResponse)
async def add_project_member(
    project_id: UUID,
    member_data: ProjectMemberCreate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Add a user to a project.
    Admin only.
    """
    # Validate access level
    try:
        access_level = UserRoleEnum(member_data.access_level.upper())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid access level. Must be one of: {', '.join([r.value for r in UserRoleEnum])}"
        )

    try:
        # Verify project exists
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Check if member already exists
        result = await db.execute(select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == member_data.user_id
        ))
        existing_member = result.scalar_one_or_none()

        if existing_member:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already a member of this project"
            )

        # Create new member
        new_member = ProjectMember(
            project_id=project_id,
            user_id=member_data.user_id,
            access_level=access_level
        )
        db.add(new_member)
        await db.commit()
        await db.refresh(new_member)

        # Get user information
        headers = {
            "apikey": settings.SUPABASE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY}",
            "Content-Type": "application/json",
        }

        user_email = None
        user_full_name = None
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                f"{settings.SUPABASE_URL}/auth/v1/admin/users/{member_data.user_id}",
                headers=headers
            )
            if user_response.status_code == 200:
                user_data = user_response.json()
                user_email = user_data.get("email")
                user_full_name = user_data.get("user_metadata", {}).get("full_name")

        return ProjectMemberResponse(
            id=new_member.id,
            project_id=new_member.project_id,
            user_id=new_member.user_id,
            access_level=new_member.access_level.value if hasattr(new_member.access_level, 'value') else new_member.access_level,
            created_at=new_member.created_at.isoformat() if new_member.created_at else None,
            updated_at=new_member.updated_at.isoformat() if new_member.updated_at else None,
            user_email=user_email,
            user_full_name=user_full_name
        )

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding project member: {str(e)}"
        )


@router.delete("/projects/{project_id}/members/{user_id}")
async def remove_project_member(
    project_id: UUID,
    user_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Remove a user from a project.
    Admin only.
    """
    try:
        # Find member
        result = await db.execute(select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user_id
        ))
        member = result.scalar_one_or_none()

        if not member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project member not found"
            )

        await db.delete(member)
        await db.commit()

        return {"message": "Project member removed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing project member: {str(e)}"
        )
