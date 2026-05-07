"""
Pydantic schemas for gene bookmarks and lists.
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Gene Bookmark Schemas
# ============================================================================

class GeneBookmarkBase(BaseModel):
    """Base schema for gene bookmarks."""
    gene_symbol: str = Field(..., description="Gene symbol")
    gene_id: Optional[str] = Field(None, description="Gene ID (e.g., ENSG ID)")
    notes: Optional[str] = Field(None, description="User notes")
    tags: List[str] = Field(default_factory=list, description="Tags")
    color: Optional[str] = Field(None, description="Color (hex)")
    is_favorite: bool = Field(True, description="Favorite flag")


class GeneBookmarkCreate(GeneBookmarkBase):
    """Schema for creating a gene bookmark."""
    pass


class GeneBookmarkUpdate(BaseModel):
    """Schema for updating a gene bookmark."""
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    color: Optional[str] = None
    is_favorite: Optional[bool] = None


class GeneBookmarkResponse(GeneBookmarkBase):
    """Schema for gene bookmark response."""
    id: UUID
    user_id: UUID
    project_id: UUID
    extra_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Gene List Schemas
# ============================================================================

class GeneListBase(BaseModel):
    """Base schema for gene lists."""
    name: str = Field(..., description="List name")
    description: Optional[str] = Field(None, description="Description")
    genes: List[str] = Field(default_factory=list, description="Gene symbols")
    color: Optional[str] = Field(None, description="Color (hex)")
    is_public: bool = Field(False, description="Public visibility")
    tags: List[str] = Field(default_factory=list, description="Tags")


class GeneListCreate(GeneListBase):
    """Schema for creating a gene list."""
    pass


class GeneListUpdate(BaseModel):
    """Schema for updating a gene list."""
    name: Optional[str] = None
    description: Optional[str] = None
    genes: Optional[List[str]] = None
    color: Optional[str] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None


class GeneListAddGenes(BaseModel):
    """Schema for adding genes to a list."""
    genes: List[str] = Field(..., description="Gene symbols to add")


class GeneListRemoveGenes(BaseModel):
    """Schema for removing genes from a list."""
    genes: List[str] = Field(..., description="Gene symbols to remove")


class GeneListResponse(GeneListBase):
    """Schema for gene list response."""
    id: UUID
    user_id: UUID
    project_id: UUID
    gene_count: int = Field(..., description="Number of genes")
    extra_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Bulk Operations
# ============================================================================

class BookmarkBatchCreate(BaseModel):
    """Schema for batch creating bookmarks."""
    gene_symbols: List[str] = Field(..., description="Gene symbols to bookmark")
    tags: List[str] = Field(default_factory=list, description="Tags to apply")
    notes: Optional[str] = Field(None, description="Notes to apply")
    color: Optional[str] = Field(None, description="Color to apply")
    is_favorite: bool = Field(True, description="Favorite flag")


class BookmarkBatchResponse(BaseModel):
    """Response for batch bookmark creation."""
    created: int = Field(..., description="Number created")
    skipped: int = Field(..., description="Number skipped (already exist)")
    bookmarks: List[GeneBookmarkResponse]
