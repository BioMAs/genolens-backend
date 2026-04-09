"""
Gene Search API endpoints.
Search for genes across projects and return their locations.
"""
from uuid import UUID
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, or_, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.api.deps import get_db, get_current_user
from app.core.supabase_auth import SupabaseUser
from app.models.models import Project, Dataset, DatasetType, DatasetStatus

router = APIRouter(prefix="/genes", tags=["genes"])


class GeneSearchResult(BaseModel):
    """Single gene search result."""
    gene_symbol: str
    gene_id: Optional[str] = None
    project_id: str
    project_name: str
    dataset_id: str
    dataset_name: str
    dataset_type: str
    comparison_name: Optional[str] = None


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
    Search for genes across all user's projects.
    
    Searches in:
    - DEG comparison names (from metadata)
    - Dataset names
    - Dataset descriptions
    
    Returns list of matching locations where the gene might be found.
    
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
    results = []
    
    # Build base query for user's projects and datasets
    query_builder = (
        select(Project, Dataset)
        .join(Dataset, Dataset.project_id == Project.id)
        .where(
            Project.owner_id == current_user.user_id,
            Dataset.status == DatasetStatus.READY,
        )
    )
    
    # Optionally filter by project
    if project_id:
        query_builder = query_builder.where(Project.id == project_id)
    
    # Execute query
    result = await db.execute(query_builder)
    rows = result.all()
    
    # Search pattern (case-insensitive)
    search_term = q.lower()
    
    # Process each dataset
    for project, dataset in rows:
        # Check dataset name and description
        dataset_matches = (
            search_term in dataset.name.lower() or
            (dataset.description and search_term in dataset.description.lower())
        )
        
        # For DEG datasets, check comparison names in metadata
        if dataset.type == DatasetType.DEG and dataset.dataset_metadata:
            metadata = dataset.dataset_metadata
            
            # Check for single comparison name
            if isinstance(metadata.get("comparison_name"), str):
                comp_name = metadata["comparison_name"]
                if search_term in comp_name.lower():
                    results.append(
                        GeneSearchResult(
                            gene_symbol=q.upper(),
                            project_id=str(project.id),
                            project_name=project.name,
                            dataset_id=str(dataset.id),
                            dataset_name=dataset.name,
                            dataset_type=dataset.type,
                            comparison_name=comp_name
                        )
                    )
            
            # Check for multiple comparisons
            elif isinstance(metadata.get("comparisons"), (list, dict)):
                comparisons = metadata["comparisons"]
                comp_names = comparisons if isinstance(comparisons, list) else list(comparisons.keys())
                
                for comp_name in comp_names:
                    if search_term in str(comp_name).lower():
                        results.append(
                            GeneSearchResult(
                                gene_symbol=q.upper(),
                                project_id=str(project.id),
                                project_name=project.name,
                                dataset_id=str(dataset.id),
                                dataset_name=dataset.name,
                                dataset_type=dataset.type,
                                comparison_name=comp_name
                            )
                        )
        
        # Add dataset-level match if found
        if dataset_matches:
            results.append(
                GeneSearchResult(
                    gene_symbol=q.upper(),
                    project_id=str(project.id),
                    project_name=project.name,
                    dataset_id=str(dataset.id),
                    dataset_name=dataset.name,
                    dataset_type=dataset.type,
                    comparison_name=None
                )
            )
    
    # Remove duplicates and limit results
    unique_results = []
    seen = set()
    
    for result in results:
        key = (result.project_id, result.dataset_id, result.comparison_name)
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
            
            if len(unique_results) >= limit:
                break
    
    return GeneSearchResponse(
        results=unique_results,
        total=len(unique_results),
        query=q
    )
