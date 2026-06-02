"""
Project API endpoints using SQLAlchemy.
"""
from uuid import UUID
from typing import Optional, Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func, delete, or_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta, timezone

from app.api.deps import get_db, get_current_user
from app.api.deps.license import require_active_license
from app.api.deps.subscription import get_or_create_user
from app.core.supabase_auth import SupabaseUser, lookup_user_by_email
from app.models.models import Project, Dataset, ProjectMember, GeneBookmark, GeneList, ProjectComment, DegGene, EnrichmentPathway, DatasetType, DatasetStatus, ActivityEventType, ProjectActivityLog, UserRole, User
from app.services import email_service, history_service
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
    ProjectSummaryResponse,
    ProjectStats,
    ComparisonSummary,
    PaginatedComparisonsResponse,
    ProjectMemberCreate,
    ProjectMemberUpdate,
    ProjectMemberResponse,
    ProjectMemberListResponse,
    ProjectDashboardStats,
    DatasetBreakdown,
    ActivityBreakdown,
)


router = APIRouter(prefix="/projects", tags=["projects"])


async def _check_project_admin(project: Project, current_user_id: UUID, db: AsyncSession) -> bool:
    """
    Returns True if current_user_id is the project owner OR a member with access_level == ADMIN.
    """
    if project.owner_id == current_user_id:
        return True
    member_query = select(ProjectMember).where(
        ProjectMember.project_id == project.id,
        ProjectMember.user_id == current_user_id,
        ProjectMember.access_level == UserRole.ADMIN,
    )
    result = await db.execute(member_query)
    return result.scalar_one_or_none() is not None


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_active_license)])
async def create_project(
    project_in: ProjectCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    db_user: Annotated[User, Depends(get_or_create_user)],
) -> Project:
    """
    Create a new project.

    - **name**: Project name (required)
    - **description**: Optional project description
    """
    # ── Enforce project limit by plan ─────────────────────────────────────────
    if db_user.max_projects is not None:
        count_result = await db.execute(
            select(func.count()).select_from(Project).where(
                Project.owner_id == current_user.user_id
            )
        )
        project_count = count_result.scalar_one()
        if project_count >= db_user.max_projects:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Project limit reached ({db_user.max_projects} projects max "
                    f"on {db_user.subscription_plan.value} plan). "
                    f"Upgrade to TEAM or ON_PREMISE for unlimited projects."
                )
            )
    # ─────────────────────────────────────────────────────────────────────────

    project = Project(
        name=project_in.name,
        description=project_in.description,
        owner_id=current_user.user_id
    )

    db.add(project)
    await db.commit()
    await db.refresh(project)

    return project


@router.get("", response_model=ProjectListResponse)
@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page")
) -> dict:
    """
    List all projects owned by or shared with the current user (paginated).
    """
    member_subquery = select(ProjectMember.project_id).where(ProjectMember.user_id == current_user.user_id)
    access_filter = or_(Project.owner_id == current_user.user_id, Project.id.in_(member_subquery))

    # Get total count
    count_query = select(func.count()).select_from(Project).where(access_filter)
    total = await db.scalar(count_query)

    # Get paginated projects
    offset = (page - 1) * page_size
    query = select(Project).where(access_filter)\
        .order_by(Project.created_at.desc())\
        .offset(offset).limit(page_size)
    
    result = await db.execute(query)
    projects = result.scalars().all()

    return {
        "items": projects,
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> Project:
    """
    Get a specific project by ID.
    """
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Vérifier que l'utilisateur est propriétaire ou membre du projet
    if project.owner_id != current_user.user_id:
        member_query = select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.user_id
        )
        member_result = await db.execute(member_query)
        member = member_result.scalar_one_or_none()
        if not member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    project_in: ProjectUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> Project:
    """
    Update a project.
    """
    # Check if project exists
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    if not await _check_project_admin(project, current_user.user_id, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only project admins can update this project"
        )

    # Update project
    update_data = project_in.model_dump(exclude_unset=True)
    if not update_data:
        return project

    for field, value in update_data.items():
        setattr(project, field, value)
    
    await db.commit()
    await db.refresh(project)

    return project


def _build_comparisons_from_datasets(datasets: list) -> list[ComparisonSummary]:
    """
    Build the list of ComparisonSummary from a list of Dataset objects.
    """
    comparisons_dict: dict[str, ComparisonSummary] = {}

    for d in datasets:
        metadata = d.dataset_metadata or {}

        # Single file per comparison (old way)
        if d.type == "DEG":
            # Prefer explicit comparison_name; fall back to first item in list; then dataset name
            comp_name = metadata.get('comparison_name') if metadata else None
            if not comp_name and isinstance(metadata.get('comparisons'), list) and metadata['comparisons']:
                comp_name = metadata['comparisons'][0]
            if not comp_name:
                comp_name = d.name
            # Use pre-calculated DB counts as primary source, metadata as fallback
            deg_up = d.deg_up_count or (metadata.get('deg_up', 0) if metadata else 0)
            deg_down = d.deg_down_count or (metadata.get('deg_down', 0) if metadata else 0)
            deg_total = d.deg_significant_count or (metadata.get('deg_total', deg_up + deg_down) if metadata else deg_up + deg_down)
            comparisons_dict[comp_name] = ComparisonSummary(
                name=comp_name,
                deg_up=deg_up,
                deg_down=deg_down,
                deg_total=deg_total,
                has_enrichment=False,
                dataset_id=d.id,
                dataset_type='SINGLE'
            )

        # Global DEG file (new way)
        if metadata and 'comparisons' in metadata and isinstance(metadata['comparisons'], dict):
            for comp_name, comp_info in metadata['comparisons'].items():
                deg_up = comp_info.get('deg_up', 0)
                deg_down = comp_info.get('deg_down', 0)
                deg_total = comp_info.get('deg_total', deg_up + deg_down)
                comparisons_dict[comp_name] = ComparisonSummary(
                    name=comp_name,
                    deg_up=deg_up,
                    deg_down=deg_down,
                    deg_total=deg_total,
                    has_enrichment=False,
                    dataset_id=d.id,
                    dataset_type='GLOBAL'
                )

    # Mark comparisons with enrichment (check both ENRICHMENT datasets and DEG datasets)
    for d in datasets:
        metadata = d.dataset_metadata or {}
        if metadata:
            enrichment_comparisons = metadata.get('enrichment_comparisons', [])
            for comp_name in enrichment_comparisons:
                if comp_name in comparisons_dict:
                    comparisons_dict[comp_name].has_enrichment = True

    return list(comparisons_dict.values())


@router.get("/{project_id}/comparisons", response_model=PaginatedComparisonsResponse)
async def list_project_comparisons(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    """
    Get paginated comparisons for a project.
    """
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    if project.owner_id != current_user.user_id:
        member_query = select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.user_id
        )
        member_result = await db.execute(member_query)
        if not member_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

    datasets_query = select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    result = await db.execute(datasets_query)
    datasets = result.scalars().all()

    all_comparisons = _build_comparisons_from_datasets(datasets)
    total = len(all_comparisons)
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_comparisons = all_comparisons[offset:offset + page_size]

    return {
        "comparisons": page_comparisons,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@router.get("/{project_id}/summary", response_model=ProjectSummaryResponse)
async def get_project_summary(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Get optimized project summary with pre-computed statistics.
    """
    # Get project
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    if project.owner_id != current_user.user_id:
        member_query = select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.user_id
        )
        member_result = await db.execute(member_query)
        if not member_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

    # Get all datasets for this project
    datasets_query = select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.created_at.desc())
    result = await db.execute(datasets_query)
    datasets = result.scalars().all()

    # Compute statistics
    total_datasets = len(datasets)
    processing_count = sum(1 for d in datasets if d.status == "PROCESSING")
    ready_count = sum(1 for d in datasets if d.status == "READY")
    failed_count = sum(1 for d in datasets if d.status == "FAILED")
    original_files_count = sum(1 for d in datasets if d.raw_file_path)

    # Count comparisons using the shared helper (no need to return them here)
    all_comparisons = _build_comparisons_from_datasets(datasets)
    total_comparisons = len(all_comparisons)

    # Get original file names
    original_files = [d.name for d in datasets if d.raw_file_path]

    return {
        "project": project,
        "stats": ProjectStats(
            total_datasets=total_datasets,
            total_comparisons=total_comparisons,
            processing_count=processing_count,
            ready_count=ready_count,
            failed_count=failed_count,
            original_files_count=original_files_count
        ),
        "comparisons": all_comparisons,
        "original_files": original_files
    }


@router.get("/{project_id}/dashboard-stats", response_model=ProjectDashboardStats)
async def get_project_dashboard_stats(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Get rich dashboard statistics aggregated across all project entities.
    Covers datasets, analyses output, bookmarks, comments, members, and activity.
    Accessible to the project owner and all project members.
    """
    # Access check: owner or member
    owner_query = select(Project).where(
        Project.id == project_id,
        Project.owner_id == current_user.user_id
    )
    owner_result = await db.execute(owner_query)
    project = owner_result.scalar_one_or_none()

    if not project:
        # Check membership
        member_query = select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.user_id
        )
        member_result = await db.execute(member_query)
        member = member_result.scalar_one_or_none()
        if not member:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        # Fetch project for member
        proj_query = select(Project).where(Project.id == project_id)
        proj_result = await db.execute(proj_query)
        project = proj_result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    # ── Datasets ────────────────────────────────────────────────────────────────
    ds_counts = await db.execute(
        select(Dataset.type, Dataset.status, func.count().label("n"))
        .where(Dataset.project_id == project_id)
        .group_by(Dataset.type, Dataset.status)
    )
    ds_rows = ds_counts.all()

    total_datasets = 0
    ds_ready = ds_processing = ds_failed = 0
    db_matrix = db_deg = db_enrich = db_meta = db_other = 0

    for row in ds_rows:
        ds_type, ds_status, n = row
        total_datasets += n
        if str(ds_status).upper() == "READY":
            ds_ready += n
        elif str(ds_status).upper() == "PROCESSING":
            ds_processing += n
        elif str(ds_status).upper() == "FAILED":
            ds_failed += n

        t = str(ds_type).upper()
        if t == "MATRIX":
            db_matrix += n
        elif t == "DEG":
            db_deg += n
        elif t in ("ENRICHMENT", "ENRICHMENT_PATHWAY"):
            db_enrich += n
        elif "METADATA" in t:
            db_meta += n
        else:
            db_other += n

    # ── Analysis output ─────────────────────────────────────────────────────────
    deg_count_result = await db.execute(
        select(func.count()).select_from(DegGene)
        .join(Dataset, DegGene.dataset_id == Dataset.id)
        .where(Dataset.project_id == project_id)
    )
    total_deg = deg_count_result.scalar_one() or 0

    enrichment_count_result = await db.execute(
        select(func.count()).select_from(EnrichmentPathway)
        .join(Dataset, EnrichmentPathway.dataset_id == Dataset.id)
        .where(Dataset.project_id == project_id)
    )
    total_enrichment = enrichment_count_result.scalar_one() or 0

    # Count unique comparison names in datasets metadata
    comp_result = await db.execute(
        select(Dataset.dataset_metadata)
        .where(Dataset.project_id == project_id, Dataset.type == DatasetType.DEG)
    )
    comp_names = {
        (meta or {}).get("comparison_name", "")
        for meta, in comp_result.all()
    }
    total_comparisons = len([c for c in comp_names if c])

    # ── Collaboration ───────────────────────────────────────────────────────────
    bookmarks_result = await db.execute(
        select(func.count()).select_from(GeneBookmark)
        .where(GeneBookmark.project_id == project_id)
    )
    total_bookmarks = bookmarks_result.scalar_one() or 0

    gene_lists_result = await db.execute(
        select(func.count()).select_from(GeneList)
        .where(GeneList.project_id == project_id)
    )
    total_gene_lists = gene_lists_result.scalar_one() or 0

    comments_result = await db.execute(
        select(func.count()).select_from(ProjectComment)
        .where(ProjectComment.project_id == project_id)
    )
    total_comments = comments_result.scalar_one() or 0

    members_result = await db.execute(
        select(func.count()).select_from(ProjectMember)
        .where(ProjectMember.project_id == project_id)
    )
    total_members = members_result.scalar_one() or 0

    # ── Activity – total + last 7 days ──────────────────────────────────────────
    activity_total_result = await db.execute(
        select(func.count()).select_from(ProjectActivityLog)
        .where(ProjectActivityLog.project_id == project_id)
    )
    total_activity = activity_total_result.scalar_one() or 0

    last_activity_result = await db.execute(
        select(ProjectActivityLog.created_at)
        .where(ProjectActivityLog.project_id == project_id)
        .order_by(ProjectActivityLog.created_at.desc())
        .limit(1)
    )
    last_activity_row = last_activity_result.scalar_one_or_none()

    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    activity_7d_result = await db.execute(
        select(ProjectActivityLog.event_type, func.count().label("n"))
        .where(
            ProjectActivityLog.project_id == project_id,
            ProjectActivityLog.created_at >= week_ago,
        )
        .group_by(ProjectActivityLog.event_type)
    )
    activity_7d_rows = activity_7d_result.all()

    uploads_7d = bookmarks_7d = comments_7d = analyses_7d = 0
    ANALYSIS_EVENTS = {
        ActivityEventType.ENRICHMENT_RUN,
        ActivityEventType.CLUSTERING_RUN,
        ActivityEventType.GSEA_RUN,
        ActivityEventType.GO_ENRICHMENT_RUN,
    }
    for evt_type, n in activity_7d_rows:
        et = ActivityEventType(evt_type) if isinstance(evt_type, str) else evt_type
        if et == ActivityEventType.DATASET_UPLOADED:
            uploads_7d += n
        elif et in (ActivityEventType.BOOKMARK_CREATED, ActivityEventType.BOOKMARK_BATCH_CREATED):
            bookmarks_7d += n
        elif et == ActivityEventType.COMMENT_ADDED:
            comments_7d += n
        elif et in ANALYSIS_EVENTS:
            analyses_7d += n

    return ProjectDashboardStats(
        project_id=project.id,
        project_name=project.name,
        total_datasets=total_datasets,
        datasets_ready=ds_ready,
        datasets_processing=ds_processing,
        datasets_failed=ds_failed,
        dataset_breakdown=DatasetBreakdown(
            matrix=db_matrix, deg=db_deg, enrichment=db_enrich,
            metadata=db_meta, other=db_other,
        ),
        total_comparisons=total_comparisons,
        total_deg_genes=total_deg,
        total_enrichment_pathways=total_enrichment,
        total_bookmarks=total_bookmarks,
        total_gene_lists=total_gene_lists,
        total_comments=total_comments,
        total_members=total_members,
        total_activity_events=total_activity,
        activity_last_7_days=ActivityBreakdown(
            datasets_uploaded=uploads_7d,
            bookmarks_created=bookmarks_7d,
            comments_added=comments_7d,
            analyses_run=analyses_7d,
        ),
        last_activity_at=last_activity_row,
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> None:
    """
    Delete a project (and all associated data via cascade).
    """
    # Check if project exists
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    if not await _check_project_admin(project, current_user.user_id, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only project admins can delete this project"
        )

    # Delete project (cascade will handle datasets)
    await db.delete(project)
    await db.commit()


# ============================================
# Project Members Endpoints (Collaboration)
# ============================================

@router.post("/{project_id}/members", response_model=ProjectMemberResponse, status_code=status.HTTP_201_CREATED)
async def invite_project_member(
    project_id: UUID,
    member_in: ProjectMemberCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> ProjectMember:
    """
    Invite a user to a project by email.
    
    Only the project owner can invite members.
    
    - **user_email**: Email of the user to invite
    - **access_level**: Role to assign (USER or ADMIN)
    """
    # Check if project exists and user is owner
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    if not await _check_project_admin(project, current_user.user_id, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only project admins can invite members"
        )

    # ── Step 1: resolve email → Supabase user_id ──────────────────────────
    supabase_user = await lookup_user_by_email(member_in.user_email)
    if supabase_user is None:
        # User doesn't exist yet — we still create a pending invitation email
        # and return a 202 Accepted (no DB record created yet).
        await email_service.send_project_invitation(
            invitee_email=member_in.user_email,
            inviter_email=current_user.email,
            project_id=str(project_id),
            project_name=project.name,
            access_level=member_in.access_level.value,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No GenoLens account found for '{member_in.user_email}'. "
                "An invitation email has been sent so they can sign up and access the project."
            ),
        )

    invitee_user_id = UUID(supabase_user["id"])

    # Prevent adding yourself
    if invitee_user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot add yourself as a project member"
        )

    # ── Step 2: check for duplicate membership ─────────────────────────────
    existing_query = select(ProjectMember).where(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == invitee_user_id,
    )
    existing_result = await db.execute(existing_query)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User is already a member of this project"
        )

    # ── Step 3: create membership record ──────────────────────────────────
    member = ProjectMember(
        project_id=project_id,
        user_id=invitee_user_id,
        access_level=member_in.access_level,
    )
    db.add(member)
    await db.commit()
    await db.refresh(member)

    # Log sharing activity
    await history_service.log_activity(
        db, project_id, current_user.user_id, ActivityEventType.PROJECT_SHARED,
        entity_type="member",
        entity_id=str(invitee_user_id),
        entity_name=member_in.user_email,
        extra_metadata={"access_level": member_in.access_level.value},
    )

    # ── Step 4: send invitation email (non-blocking) ───────────────────────
    await email_service.send_project_invitation(
        invitee_email=member_in.user_email,
        inviter_email=current_user.email,
        project_id=str(project_id),
        project_name=project.name,
        access_level=member_in.access_level.value,
    )

    return member


@router.get("/{project_id}/members", response_model=ProjectMemberListResponse)
async def list_project_members(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    List all members of a project.
    
    Only project members and owner can view the member list.
    """
    # Check if project exists
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Check if user has access (owner or member)
    if project.owner_id != current_user.user_id:
        member_query = select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == current_user.user_id
        )
        member_result = await db.execute(member_query)
        member = member_result.scalar_one_or_none()
        
        if not member:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
    
    # Get all members
    members_query = select(ProjectMember).where(ProjectMember.project_id == project_id)
    members_result = await db.execute(members_query)
    members = members_result.scalars().all()
    
    # TODO: Enrich with user info from Supabase Auth
    # For now, return basic member info
    
    return {
        "members": members,
        "total": len(members)
    }


@router.patch("/{project_id}/members/{member_user_id}", response_model=ProjectMemberResponse)
async def update_project_member(
    project_id: UUID,
    member_user_id: UUID,
    member_update: ProjectMemberUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> ProjectMember:
    """
    Update a project member's access level.
    
    Only the project owner can update member roles.
    
    - **access_level**: New role (USER or ADMIN)
    """
    # Check if project exists and user is owner
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    if not await _check_project_admin(project, current_user.user_id, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only project admins can update member roles"
        )
    
    # Get the member
    member_query = select(ProjectMember).where(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == member_user_id
    )
    member_result = await db.execute(member_query)
    member = member_result.scalar_one_or_none()
    
    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    # Update access level
    member.access_level = member_update.access_level
    
    await db.commit()
    await db.refresh(member)
    
    return member


@router.delete("/{project_id}/members/{member_user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_project_member(
    project_id: UUID,
    member_user_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> None:
    """
    Remove a member from a project.
    
    Only the project owner can remove members.
    Members cannot remove themselves (they should leave instead).
    """
    # Check if project exists and user is owner
    query = select(Project).where(Project.id == project_id)
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    if not await _check_project_admin(project, current_user.user_id, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only project admins can remove members"
        )

    # Prevent owner from removing themselves
    if member_user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Owner cannot be removed from project"
        )
    
    # Get the member
    member_query = select(ProjectMember).where(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == member_user_id
    )
    member_result = await db.execute(member_query)
    member = member_result.scalar_one_or_none()
    
    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    # Delete member
    await db.delete(member)
    await db.commit()

