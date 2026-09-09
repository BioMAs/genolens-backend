from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, asc, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.project_access import assert_project_read_access
from app.core.supabase_auth import SupabaseUser
from app.models.models import Dataset, EnrichmentPathway

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


@router.get("/{dataset_id}/comparisons")
async def get_enrichment_comparisons(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: SupabaseUser = Depends(get_current_user),
):
    """
    Get list of comparisons that have enrichment results available.
    """
    # La route rendait les noms de comparaison de n'importe quel dataset a
    # n'importe quel compte authentifie : elle partait directement sur
    # `enrichment_pathways` sans jamais regarder a qui appartenait le projet.
    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    ).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await assert_project_read_access(db, dataset.project_id, current_user.user_id)

    query = (
        select(EnrichmentPathway.comparison_name)
        .where(EnrichmentPathway.dataset_id == dataset_id)
        .distinct()
    )

    result = await db.execute(query)
    comparisons = result.scalars().all()

    return comparisons


@router.get("/{dataset_id}/{comparison_name}", response_model=List[EnrichmentPathwayResponse])
async def get_enrichment_results(
    dataset_id: UUID,
    comparison_name: str,
    category: Optional[str] = Query(None, description="Filter by category (e.g. GO:BP, KEGG)"),
    regulation: Optional[str] = Query(None, description="Filter by regulation (UP, DOWN, ALL)"),
    max_padj: float = Query(0.05, description="Filter by adjusted p-value cutoff"),
    limit: int = Query(100, description="Limit number of results"),
    db: AsyncSession = Depends(get_db),
    current_user: SupabaseUser = Depends(get_current_user),
):
    """
    Get enrichment analysis results for a specific comparison.
    """
    # Acces au projet, via le helper partage.
    #
    # Le controle etait ecrit a la main ici, et il lisait `current_user.id` —
    # un attribut que `SupabaseUser` n'a pas (il expose `user_id`). La ligne
    # levait donc AttributeError pour CHAQUE appelant, proprietaire compris, et
    # la route rendait 500 au lieu de servir l'enrichissement. Le helper est le
    # meme que celui des trente routes voisines : owner ou membre du projet.
    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    ).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await assert_project_read_access(db, dataset.project_id, current_user.user_id)

    # 2. Query Enrichment Pathways
    query = select(EnrichmentPathway).where(
        and_(
            EnrichmentPathway.dataset_id == dataset_id,
            EnrichmentPathway.comparison_name == comparison_name,
            EnrichmentPathway.padj <= max_padj,
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
