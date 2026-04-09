"""
API endpoints for gene bookmarks and lists.
"""
import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.db import get_db
from app.api.deps.auth import get_current_user
from app.schemas.bookmark import (
    GeneBookmarkCreate,
    GeneBookmarkUpdate,
    GeneBookmarkResponse,
    GeneListCreate,
    GeneListUpdate,
    GeneListResponse,
    GeneListAddGenes,
    GeneListRemoveGenes,
    BookmarkBatchCreate,
    BookmarkBatchResponse
)
from app.services.bookmarks_service import bookmarks_service
from app.services import history_service
from app.models.models import ActivityEventType

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Gene Bookmarks
# ============================================================================

@router.get("/projects/{project_id}/bookmarks", response_model=List[GeneBookmarkResponse])
async def get_bookmarks(
    project_id: UUID,
    gene_symbol: Optional[str] = Query(None, description="Filter by gene symbol"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get user's gene bookmarks for a project."""
    try:
        user_id = UUID(current_user["sub"])
        bookmarks = await bookmarks_service.get_bookmarks(
            db,
            user_id=user_id,
            project_id=project_id,
            gene_symbol=gene_symbol
        )
        return bookmarks
    except Exception as e:
        logger.error(f"Error fetching bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/bookmarks/check/{gene_symbol}")
async def check_bookmark(
    project_id: UUID,
    gene_symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Check if a gene is bookmarked."""
    try:
        user_id = UUID(current_user["sub"])
        is_bookmarked = await bookmarks_service.is_bookmarked(
            db,
            user_id=user_id,
            project_id=project_id,
            gene_symbol=gene_symbol
        )
        return {"is_bookmarked": is_bookmarked}
    except Exception as e:
        logger.error(f"Error checking bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/bookmarks", response_model=GeneBookmarkResponse, status_code=201)
async def create_bookmark(
    project_id: UUID,
    bookmark: GeneBookmarkCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a gene bookmark."""
    try:
        user_id = UUID(current_user["sub"])
        new_bookmark = await bookmarks_service.create_bookmark(
            db,
            user_id=user_id,
            project_id=project_id,
            gene_symbol=bookmark.gene_symbol,
            gene_id=bookmark.gene_id,
            notes=bookmark.notes,
            tags=bookmark.tags,
            color=bookmark.color,
            is_favorite=bookmark.is_favorite
        )
        await history_service.log_activity(
            db, project_id, user_id, ActivityEventType.BOOKMARK_CREATED,
            entity_type="bookmark",
            entity_id=str(new_bookmark.id),
            entity_name=new_bookmark.gene_symbol,
        )
        return new_bookmark
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/bookmarks/batch", response_model=BookmarkBatchResponse, status_code=201)
async def create_bookmarks_batch(
    project_id: UUID,
    batch: BookmarkBatchCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create multiple bookmarks at once."""
    try:
        user_id = UUID(current_user["sub"])
        created = []
        skipped = 0
        
        for gene_symbol in batch.gene_symbols:
            try:
                bookmark = await bookmarks_service.create_bookmark(
                    db,
                    user_id=user_id,
                    project_id=project_id,
                    gene_symbol=gene_symbol,
                    notes=batch.notes,
                    tags=batch.tags,
                    color=batch.color,
                    is_favorite=batch.is_favorite
                )
                created.append(bookmark)
            except ValueError:
                # Already exists
                skipped += 1
        
        if created:
            await history_service.log_activity(
                db, project_id, user_id, ActivityEventType.BOOKMARK_BATCH_CREATED,
                entity_type="bookmark_batch",
                extra_metadata={"created": len(created), "skipped": skipped},
            )
        return {
            "created": len(created),
            "skipped": skipped,
            "bookmarks": created
        }
    except Exception as e:
        logger.error(f"Error batch creating bookmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/bookmarks/{bookmark_id}", response_model=GeneBookmarkResponse)
async def update_bookmark(
    bookmark_id: UUID,
    bookmark: GeneBookmarkUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a bookmark."""
    try:
        user_id = UUID(current_user["sub"])
        updated = await bookmarks_service.update_bookmark(
            db,
            bookmark_id=bookmark_id,
            user_id=user_id,
            notes=bookmark.notes,
            tags=bookmark.tags,
            color=bookmark.color,
            is_favorite=bookmark.is_favorite
        )
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/bookmarks/{bookmark_id}", status_code=204)
async def delete_bookmark(
    bookmark_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a bookmark."""
    try:
        user_id = UUID(current_user["sub"])
        await bookmarks_service.delete_bookmark(
            db,
            bookmark_id=bookmark_id,
            user_id=user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting bookmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Gene Lists
# ============================================================================

@router.get("/projects/{project_id}/gene-lists", response_model=List[GeneListResponse])
async def get_gene_lists(
    project_id: UUID,
    include_public: bool = Query(True, description="Include public lists from other users"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get user's gene lists for a project."""
    try:
        user_id = UUID(current_user["sub"])
        gene_lists = await bookmarks_service.get_gene_lists(
            db,
            user_id=user_id,
            project_id=project_id,
            include_public=include_public
        )
        return gene_lists
    except Exception as e:
        logger.error(f"Error fetching gene lists: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/gene-lists", response_model=GeneListResponse, status_code=201)
async def create_gene_list(
    project_id: UUID,
    gene_list: GeneListCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a gene list."""
    try:
        user_id = UUID(current_user["sub"])
        new_list = await bookmarks_service.create_gene_list(
            db,
            user_id=user_id,
            project_id=project_id,
            name=gene_list.name,
            description=gene_list.description,
            genes=gene_list.genes,
            color=gene_list.color,
            is_public=gene_list.is_public,
            tags=gene_list.tags
        )
        return new_list
    except Exception as e:
        logger.error(f"Error creating gene list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/gene-lists/{list_id}", response_model=GeneListResponse)
async def update_gene_list(
    list_id: UUID,
    gene_list: GeneListUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a gene list."""
    try:
        user_id = UUID(current_user["sub"])
        updated = await bookmarks_service.update_gene_list(
            db,
            list_id=list_id,
            user_id=user_id,
            name=gene_list.name,
            description=gene_list.description,
            genes=gene_list.genes,
            color=gene_list.color,
            is_public=gene_list.is_public,
            tags=gene_list.tags
        )
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating gene list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gene-lists/{list_id}/add-genes", response_model=GeneListResponse)
async def add_genes_to_list(
    list_id: UUID,
    request: GeneListAddGenes,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Add genes to a list."""
    try:
        user_id = UUID(current_user["sub"])
        updated = await bookmarks_service.add_genes_to_list(
            db,
            list_id=list_id,
            user_id=user_id,
            genes=request.genes
        )
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding genes to list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gene-lists/{list_id}/remove-genes", response_model=GeneListResponse)
async def remove_genes_from_list(
    list_id: UUID,
    request: GeneListRemoveGenes,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Remove genes from a list."""
    try:
        user_id = UUID(current_user["sub"])
        updated = await bookmarks_service.remove_genes_from_list(
            db,
            list_id=list_id,
            user_id=user_id,
            genes=request.genes
        )
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error removing genes from list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/gene-lists/{list_id}", status_code=204)
async def delete_gene_list(
    list_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a gene list."""
    try:
        user_id = UUID(current_user["sub"])
        await bookmarks_service.delete_gene_list(
            db,
            list_id=list_id,
            user_id=user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting gene list: {e}")
        raise HTTPException(status_code=500, detail=str(e))
