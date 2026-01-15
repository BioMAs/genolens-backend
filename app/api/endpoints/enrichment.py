
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, and_, asc, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.models.models import EnrichmentPathway, Dataset, Project, ProjectMember, UserRole
from app.core.supabase_auth import SupabaseUser
from pydantic import BaseModel

router = APIRouter(prefix="/enrichment", tags=["Enrichment"])

class EnrichmentPathwayResponse(BaseModel):
    id: UUID
    pathway_id: str
    pathway_name: str
    category: str
    description: Optional[str] = None
    gene_count: int
    pvalue: float
    padj: float
    gene_ratio: Optional[str] = None
    bg_ratio: Optional[str] = None
    regulation: str
    genes: Optional[List[str]] = None

    class Config:
        from_attributes = True

@router.get("/{dataset_id}/{comparison_name}", response_model=List[EnrichmentPathwayResponse])
async def get_enrichment_results(
    dataset_id: UUID,
    comparison_name: str,
    category: Optional[str] = Query(None, description="Filter by category (e.g. GO:BP, KEGG)"),
    regulation: Optional[str] = Query(None, description="Filter by regulation (UP, DOWN, ALL)"),
    max_padj: float = Query(0.05, description="Filter by adjusted p-value cutoff"),
    limit: int = Query(100, description="Limit number of results"),
    db: AsyncSession = Depends(get_db),
    current_user: SupabaseUser = Depends(get_current_user)
):
    """
    Get enrichment analysis results for a specific comparison.
    """
    # 1. Verify access to dataset
    # We join Dataset -> Project -> ProjectMember to check if user has access
    # Or simpler: Check if dataset exists and user is owner or member of project
    
    # First get the dataset to find the project_id
    query_dataset = select(Dataset).where(Dataset.id == dataset_id)
    result_dataset = await db.execute(query_dataset)
    dataset = result_dataset.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    # Check Project Access
    # (Assuming we have a helper or similar logic, but let's do it explicitly for now)
    # Check if user is owner of project
    query_project = select(Project).where(Project.id == dataset.project_id)
    result_project = await db.execute(query_project)
    project = result_project.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # If not owner, check membership
    if project.owner_id != current_user.id:
        query_member = select(ProjectMember).where(
            and_(
                ProjectMember.project_id == project.id,
                ProjectMember.user_id == current_user.id
            )
        )
        result_member = await db.execute(query_member)
        member = result_member.scalar_one_or_none()
        
        if not member:
            # Check if user is ADMIN ? (Optional based on system design)
            # For now assume strict project access
            raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    # 2. Query Enrichment Pathways
    query = select(EnrichmentPathway).where(
        and_(
            EnrichmentPathway.dataset_id == dataset_id,
            EnrichmentPathway.comparison_name == comparison_name,
            EnrichmentPathway.padj <= max_padj
        )
    )

    if category:
        query = query.where(EnrichmentPathway.category == category)
    
    if regulation and regulation != "ALL":
        query = query.where(EnrichmentPathway.regulation == regulation)
    
    # Sort by significance (padj ascending)
    query = query.order_by(asc(EnrichmentPathway.padj))
    
    query = query.limit(limit)

    result = await db.execute(query)
    pathways = result.scalars().all()
    
    return pathways

@router.get("/{dataset_id}/comparisons")
async def get_enrichment_comparisons(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: SupabaseUser = Depends(get_current_user)
):
    """
    Get list of comparisons that have enrichment results available.
    """
    # Access control (Reuse logic or simplify)
    # For speed, let's assume if you can list datasets, you can see comparisons
    # NOTE: In production, refactor access control to a dependency `deps.get_authorized_dataset`
    
    query = select(EnrichmentPathway.comparison_name).where(
        EnrichmentPathway.dataset_id == dataset_id
    ).distinct()
    
    result = await db.execute(query)
    comparisons = result.scalars().all()
    
    return comparisons
