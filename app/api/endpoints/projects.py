"""
Project API endpoints using SQLAlchemy.
"""
from uuid import UUID
from typing import Optional, Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.core.supabase_auth import SupabaseUser
from app.models.models import Project, Dataset
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
    ProjectSummaryResponse,
    ProjectStats,
    ComparisonSummary
)


router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_in: ProjectCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> Project:
    """
    Create a new project.

    - **name**: Project name (required)
    - **description**: Optional project description
    """
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
    List all projects owned by the current user (paginated).
    """
    # Get total count
    count_query = select(func.count()).select_from(Project).where(Project.owner_id == current_user.user_id)
    total = await db.scalar(count_query)

    # Get paginated projects
    offset = (page - 1) * page_size
    query = select(Project).where(Project.owner_id == current_user.user_id)\
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
    query = select(Project).where(
        Project.id == project_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
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
    # Check if project exists and belongs to user
    query = select(Project).where(
        Project.id == project_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
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
    query = select(Project).where(
        Project.id == project_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
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

    # Extract comparisons from datasets
    comparisons_dict: dict[str, ComparisonSummary] = {}

    for d in datasets:
        metadata = d.dataset_metadata or {}

        # Single file per comparison (old way)
        if d.type == "DEG":
            comp_name = metadata.get('comparison_name', d.name) if metadata else d.name

            # Extract DEG counts from metadata if available
            deg_up = metadata.get('deg_up', 0) if metadata else 0
            deg_down = metadata.get('deg_down', 0) if metadata else 0
            deg_total = metadata.get('deg_total', deg_up + deg_down) if metadata else 0

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
        if metadata and 'comparisons' in metadata:
            for comp_name, comp_info in metadata['comparisons'].items():
                # Extract pre-computed DEG counts if available
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

    # Mark comparisons with enrichment
    for d in datasets:
        metadata = d.dataset_metadata or {}
        if d.type == "ENRICHMENT" and metadata:
            enrichment_comparisons = metadata.get('enrichment_comparisons', [])
            for comp_name in enrichment_comparisons:
                if comp_name in comparisons_dict:
                    comparisons_dict[comp_name].has_enrichment = True

    # Get original file names
    original_files = [d.name for d in datasets if d.raw_file_path]

    return {
        "project": project,
        "stats": ProjectStats(
            total_datasets=total_datasets,
            total_comparisons=len(comparisons_dict),
            processing_count=processing_count,
            ready_count=ready_count,
            failed_count=failed_count,
            original_files_count=original_files_count
        ),
        "comparisons": list(comparisons_dict.values()),
        "original_files": original_files
    }


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> None:
    """
    Delete a project (and all associated data via cascade).
    """
    # Check if project exists and belongs to user
    query = select(Project).where(
        Project.id == project_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Delete project (cascade will handle datasets)
    await db.delete(project)
    await db.commit()

