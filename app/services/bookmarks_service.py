"""
Service for managing gene bookmarks and custom gene lists.
"""
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, delete
from sqlalchemy.orm import joinedload

from app.models.models import GeneBookmark, GeneList, Project
from app.core.monitoring import timing_decorator

logger = logging.getLogger(__name__)


class BookmarksService:
    """Service for gene bookmarks and custom gene lists."""

    # ============================================================================
    # Gene Bookmarks
    # ============================================================================

    @timing_decorator(name="get_bookmarks")
    async def get_bookmarks(
        self,
        db: AsyncSession,
        user_id: UUID,
        project_id: UUID,
        gene_symbol: Optional[str] = None
    ) -> List[GeneBookmark]:
        """
        Get user's gene bookmarks for a project.
        
        Args:
            db: Database session
            user_id: User ID
            project_id: Project ID
            gene_symbol: Optional filter by gene symbol
            
        Returns:
            List of bookmarks
        """
        query = select(GeneBookmark).where(
            and_(
                GeneBookmark.user_id == user_id,
                GeneBookmark.project_id == project_id
            )
        ).options(joinedload(GeneBookmark.project))
        
        if gene_symbol:
            query = query.where(GeneBookmark.gene_symbol == gene_symbol)
        
        result = await db.execute(query)
        return result.scalars().all()

    async def is_bookmarked(
        self,
        db: AsyncSession,
        user_id: UUID,
        project_id: UUID,
        gene_symbol: str
    ) -> bool:
        """Check if a gene is bookmarked."""
        query = select(func.count(GeneBookmark.id)).where(
            and_(
                GeneBookmark.user_id == user_id,
                GeneBookmark.project_id == project_id,
                GeneBookmark.gene_symbol == gene_symbol
            )
        )
        result = await db.execute(query)
        count = result.scalar()
        return count > 0

    @timing_decorator(name="create_bookmark")
    async def create_bookmark(
        self,
        db: AsyncSession,
        user_id: UUID,
        project_id: UUID,
        gene_symbol: str,
        gene_id: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        color: Optional[str] = None,
        is_favorite: bool = True
    ) -> GeneBookmark:
        """
        Create a gene bookmark.
        
        Args:
            db: Database session
            user_id: User ID
            project_id: Project ID
            gene_symbol: Gene symbol
            gene_id: Optional gene ID
            notes: Optional notes
            tags: Optional tags
            color: Optional color (hex)
            is_favorite: Favorite flag
            
        Returns:
            Created bookmark
        """
        # Check if already exists
        existing = await self.is_bookmarked(db, user_id, project_id, gene_symbol)
        if existing:
            raise ValueError(f"Gene {gene_symbol} is already bookmarked in this project")
        
        bookmark = GeneBookmark(
            id=uuid4(),
            user_id=user_id,
            project_id=project_id,
            gene_symbol=gene_symbol,
            gene_id=gene_id,
            notes=notes,
            tags=tags or [],
            color=color,
            is_favorite=is_favorite,
            metadata={}
        )
        
        db.add(bookmark)
        await db.commit()
        await db.refresh(bookmark)
        
        logger.info(f"Created bookmark for gene {gene_symbol} in project {project_id}")
        return bookmark

    @timing_decorator(name="update_bookmark")
    async def update_bookmark(
        self,
        db: AsyncSession,
        bookmark_id: UUID,
        user_id: UUID,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        color: Optional[str] = None,
        is_favorite: Optional[bool] = None
    ) -> GeneBookmark:
        """
        Update a bookmark.
        
        Args:
            db: Database session
            bookmark_id: Bookmark ID
            user_id: User ID (for permission check)
            notes: Optional new notes
            tags: Optional new tags
            color: Optional new color
            is_favorite: Optional new favorite flag
            
        Returns:
            Updated bookmark
        """
        query = select(GeneBookmark).where(
            and_(
                GeneBookmark.id == bookmark_id,
                GeneBookmark.user_id == user_id
            )
        )
        result = await db.execute(query)
        bookmark = result.scalar_one_or_none()
        
        if not bookmark:
            raise ValueError(f"Bookmark {bookmark_id} not found or access denied")
        
        if notes is not None:
            bookmark.notes = notes
        if tags is not None:
            bookmark.tags = tags
        if color is not None:
            bookmark.color = color
        if is_favorite is not None:
            bookmark.is_favorite = is_favorite
        
        await db.commit()
        await db.refresh(bookmark)
        
        logger.info(f"Updated bookmark {bookmark_id}")
        return bookmark

    @timing_decorator(name="delete_bookmark")
    async def delete_bookmark(
        self,
        db: AsyncSession,
        bookmark_id: UUID,
        user_id: UUID
    ) -> bool:
        """
        Delete a bookmark.
        
        Args:
            db: Database session
            bookmark_id: Bookmark ID
            user_id: User ID (for permission check)
            
        Returns:
            True if deleted
        """
        result = await db.execute(
            delete(GeneBookmark).where(
                and_(
                    GeneBookmark.id == bookmark_id,
                    GeneBookmark.user_id == user_id
                )
            ).returning(GeneBookmark.id)
        )
        
        deleted = result.scalar_one_or_none()
        if not deleted:
            raise ValueError(f"Bookmark {bookmark_id} not found or access denied")
        
        await db.commit()
        logger.info(f"Deleted bookmark {bookmark_id}")
        return True

    # ============================================================================
    # Gene Lists
    # ============================================================================

    @timing_decorator(name="get_gene_lists")
    async def get_gene_lists(
        self,
        db: AsyncSession,
        user_id: UUID,
        project_id: UUID,
        include_public: bool = True
    ) -> List[GeneList]:
        """
        Get user's gene lists for a project.
        
        Args:
            db: Database session
            user_id: User ID
            project_id: Project ID
            include_public: Include public lists from other users
            
        Returns:
            List of gene lists
        """
        conditions = [GeneList.project_id == project_id]
        
        if include_public:
            conditions.append(
                or_(
                    GeneList.user_id == user_id,
                    GeneList.is_public == True
                )
            )
        else:
            conditions.append(GeneList.user_id == user_id)
        
        query = select(GeneList).where(and_(*conditions)).options(
            joinedload(GeneList.project)
        )
        
        result = await db.execute(query)
        return result.scalars().all()

    @timing_decorator(name="create_gene_list")
    async def create_gene_list(
        self,
        db: AsyncSession,
        user_id: UUID,
        project_id: UUID,
        name: str,
        description: Optional[str] = None,
        genes: Optional[List[str]] = None,
        color: Optional[str] = None,
        is_public: bool = False,
        tags: Optional[List[str]] = None
    ) -> GeneList:
        """
        Create a gene list.
        
        Args:
            db: Database session
            user_id: User ID
            project_id: Project ID
            name: List name
            description: Optional description
            genes: Optional list of gene symbols
            color: Optional color
            is_public: Public visibility
            tags: Optional tags
            
        Returns:
            Created gene list
        """
        genes = genes or []
        
        gene_list = GeneList(
            id=uuid4(),
            name=name,
            description=description,
            user_id=user_id,
            project_id=project_id,
            genes=genes,
            gene_count=len(genes),
            color=color,
            is_public=is_public,
            tags=tags or [],
            metadata={}
        )
        
        db.add(gene_list)
        await db.commit()
        await db.refresh(gene_list)
        
        logger.info(f"Created gene list '{name}' with {len(genes)} genes in project {project_id}")
        return gene_list

    @timing_decorator(name="update_gene_list")
    async def update_gene_list(
        self,
        db: AsyncSession,
        list_id: UUID,
        user_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        genes: Optional[List[str]] = None,
        color: Optional[str] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None
    ) -> GeneList:
        """
        Update a gene list.
        
        Args:
            db: Database session
            list_id: List ID
            user_id: User ID (for permission check)
            name: Optional new name
            description: Optional new description
            genes: Optional new gene list
            color: Optional new color
            is_public: Optional new public flag
            tags: Optional new tags
            
        Returns:
            Updated gene list
        """
        query = select(GeneList).where(
            and_(
                GeneList.id == list_id,
                GeneList.user_id == user_id
            )
        )
        result = await db.execute(query)
        gene_list = result.scalar_one_or_none()
        
        if not gene_list:
            raise ValueError(f"Gene list {list_id} not found or access denied")
        
        if name is not None:
            gene_list.name = name
        if description is not None:
            gene_list.description = description
        if genes is not None:
            gene_list.genes = genes
            gene_list.gene_count = len(genes)
        if color is not None:
            gene_list.color = color
        if is_public is not None:
            gene_list.is_public = is_public
        if tags is not None:
            gene_list.tags = tags
        
        await db.commit()
        await db.refresh(gene_list)
        
        logger.info(f"Updated gene list {list_id}")
        return gene_list

    @timing_decorator(name="add_genes_to_list")
    async def add_genes_to_list(
        self,
        db: AsyncSession,
        list_id: UUID,
        user_id: UUID,
        genes: List[str]
    ) -> GeneList:
        """Add genes to an existing list."""
        query = select(GeneList).where(
            and_(
                GeneList.id == list_id,
                GeneList.user_id == user_id
            )
        )
        result = await db.execute(query)
        gene_list = result.scalar_one_or_none()
        
        if not gene_list:
            raise ValueError(f"Gene list {list_id} not found or access denied")
        
        # Add new genes (avoid duplicates)
        current_genes = set(gene_list.genes)
        new_genes = [g for g in genes if g not in current_genes]
        
        if new_genes:
            gene_list.genes = gene_list.genes + new_genes
            gene_list.gene_count = len(gene_list.genes)
            
            await db.commit()
            await db.refresh(gene_list)
            
            logger.info(f"Added {len(new_genes)} genes to list {list_id}")
        
        return gene_list

    @timing_decorator(name="remove_genes_from_list")
    async def remove_genes_from_list(
        self,
        db: AsyncSession,
        list_id: UUID,
        user_id: UUID,
        genes: List[str]
    ) -> GeneList:
        """Remove genes from a list."""
        query = select(GeneList).where(
            and_(
                GeneList.id == list_id,
                GeneList.user_id == user_id
            )
        )
        result = await db.execute(query)
        gene_list = result.scalar_one_or_none()
        
        if not gene_list:
            raise ValueError(f"Gene list {list_id} not found or access denied")
        
        # Remove genes
        genes_to_remove = set(genes)
        updated_genes = [g for g in gene_list.genes if g not in genes_to_remove]
        
        gene_list.genes = updated_genes
        gene_list.gene_count = len(updated_genes)
        
        await db.commit()
        await db.refresh(gene_list)
        
        logger.info(f"Removed {len(genes)} genes from list {list_id}")
        return gene_list

    @timing_decorator(name="delete_gene_list")
    async def delete_gene_list(
        self,
        db: AsyncSession,
        list_id: UUID,
        user_id: UUID
    ) -> bool:
        """Delete a gene list."""
        result = await db.execute(
            delete(GeneList).where(
                and_(
                    GeneList.id == list_id,
                    GeneList.user_id == user_id
                )
            ).returning(GeneList.id)
        )
        
        deleted = result.scalar_one_or_none()
        if not deleted:
            raise ValueError(f"Gene list {list_id} not found or access denied")
        
        await db.commit()
        logger.info(f"Deleted gene list {list_id}")
        return True


# Singleton instance
bookmarks_service = BookmarksService()
