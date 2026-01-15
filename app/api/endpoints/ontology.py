
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.api.deps import get_db, get_current_user
from app.models.models import GeneSet, GeneSetDatabase
from app.core.supabase_auth import SupabaseUser
from pydantic import BaseModel

router = APIRouter(prefix="/ontology", tags=["Gene Ontology"])

class GoTerm(BaseModel):
    id: str # GO:0001234
    name: str # Term name
    namespace: str # biological_process
    definition: str # Long text
    parents: List[str]
    children: List[str]
    level: int

@router.get("/search")
async def search_ontology(
    q: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    # current_user: SupabaseUser = Depends(get_current_user) # Public or Private?
):
    """Search for GO terms by ID or name."""
    if len(q) < 2:
        return []

    # Search in name or description (which holds the term name)
    query = select(GeneSet).where(
        or_(
            GeneSet.name.ilike(f"%{q}%"), # Matches GO:XXXX
            GeneSet.description.ilike(f"%{q}%") # Matches "mitochondrion"
        )
    ).where(
        GeneSet.database.in_([GeneSetDatabase.GO_BP, GeneSetDatabase.GO_MF, GeneSetDatabase.GO_CC])
    ).limit(limit)

    result = await db.execute(query)
    terms = result.scalars().all()
    
    return [
        {
            "id": t.name,
            "name": t.description,
            "database": t.database
        }
        for t in terms
    ]

@router.get("/term/{term_id}")
async def get_term_details(
    term_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get full details/hierarchy for a GO term."""
    query = select(GeneSet).where(GeneSet.name == term_id)
    result = await db.execute(query)
    term = result.scalar_one_or_none()
    
    if not term:
        raise HTTPException(status_code=404, detail="Term not found")
    
    meta = term.gene_set_metadata
    
    # We might want to fetch parent/child names instead of just IDs
    # But for MVP, IDs might suffice, or the frontend can fetch them.
    # To be nice, let's fetch names of parents/children if possible, but that requires extra queries.
    # Let's return IDs and let frontend fetch/cache or do a quick join query here if needed.
    # Optimization: perform a second query to get names of parents/children.
    
    related_ids = (meta.get("parents", []) + meta.get("children", []))
    related_map = {}
    if related_ids:
        rel_query = select(GeneSet.name, GeneSet.description).where(GeneSet.name.in_(related_ids))
        rel_res = await db.execute(rel_query)
        related_map = {r.name: r.description for r in rel_res.all()}

    return {
        "id": term.name,
        "name": term.description,
        "definition": meta.get("description", ""), # confusing naming in load_obo vs model
        "namespace": meta.get("namespace"),
        "level": meta.get("level"),
        "parents": [{"id": pid, "name": related_map.get(pid, pid)} for pid in meta.get("parents", [])],
        "children": [{"id": cid, "name": related_map.get(cid, cid)} for cid in meta.get("children", [])]
    }
