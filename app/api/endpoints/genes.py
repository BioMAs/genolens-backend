"""
Gene Search API endpoints.
Search for genes across projects by querying DegGene records directly.
"""
from uuid import UUID
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.api.deps import get_db, get_current_user
from app.core.supabase_auth import SupabaseUser
from app.models.models import Project, Dataset, DatasetType, DatasetStatus, DegGene, ProjectMember

router = APIRouter(prefix="/genes", tags=["genes"])


class GeneSearchResult(BaseModel):
    """Single gene search result with DEG context and stats."""
    gene_symbol: str
    gene_id: Optional[str] = None
    project_id: str
    project_name: str
    dataset_id: str
    dataset_name: str
    dataset_type: str
    comparison_name: Optional[str] = None
    log_fc: Optional[float] = None
    padj: Optional[float] = None
    regulation: Optional[str] = None
    base_mean: Optional[float] = None


class GeneSearchResponse(BaseModel):
    """Response for gene search."""
    results: list[GeneSearchResult]
    total: int
    query: str


@router.get("/search", response_model=GeneSearchResponse)
async def search_genes(
    q: Annotated[str, Query(min_length=1, max_length=100, description="Gene symbol or ID to search for")],
    project_id: Annotated[Optional[UUID], Query(description="Limit search to specific project")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum results to return")] = 20,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)] = None
) -> GeneSearchResponse:
    """
    Search for genes across all user's DEG datasets.

    Searches the DegGene table by gene_name (symbol) or gene_id (Ensembl/Entrez).
    Returns matching genes with their DEG statistics and project/comparison context.

    **Parameters:**
    - **q**: Gene symbol or ID (e.g., "TP53", "ENSG00000141510")
    - **project_id**: Optional project ID to limit search scope
    - **limit**: Maximum number of results (default 20, max 100)

    **Example:**
    ```
    GET /genes/search?q=TP53&limit=10
    GET /genes/search?q=BCL2&project_id=uuid-here
    ```
    """
    search_term = q.lower()

    # Get IDs of projects shared with the user (as member)
    member_stmt = select(ProjectMember.project_id).where(
        ProjectMember.user_id == current_user.user_id
    )
    member_result = await db.execute(member_stmt)
    member_project_ids = [row[0] for row in member_result.all()]

    # Build main query: DegGene → Dataset → Project
    stmt = (
        select(DegGene, Dataset, Project)
        .join(Dataset, DegGene.dataset_id == Dataset.id)
        .join(Project, Dataset.project_id == Project.id)
        .where(
            or_(
                func.lower(DegGene.gene_name).contains(search_term),
                func.lower(DegGene.gene_id).contains(search_term),
            ),
            Dataset.status == DatasetStatus.READY,
            Dataset.type == DatasetType.DEG,
            or_(
                Project.owner_id == current_user.user_id,
                Project.id.in_(member_project_ids),
            ),
        )
        .order_by(DegGene.padj.asc().nulls_last())
        .limit(limit)
    )

    if project_id:
        stmt = stmt.where(Project.id == project_id)

    result = await db.execute(stmt)
    rows = result.all()

    results = [
        GeneSearchResult(
            gene_symbol=deg_gene.gene_name or deg_gene.gene_id,
            gene_id=deg_gene.gene_id,
            project_id=str(project.id),
            project_name=project.name,
            dataset_id=str(dataset.id),
            dataset_name=dataset.name,
            dataset_type=dataset.type,
            comparison_name=deg_gene.comparison_name,
            log_fc=deg_gene.log_fc,
            padj=deg_gene.padj,
            regulation=deg_gene.regulation,
            base_mean=deg_gene.base_mean,
        )
        for deg_gene, dataset, project in rows
    ]

    return GeneSearchResponse(
        results=results,
        total=len(results),
        query=q,
    )
