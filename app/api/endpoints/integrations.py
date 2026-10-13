"""
External Integrations API endpoints.

Routes:
  POST /integrations/string/network          – PPI network for a gene list
  POST /integrations/string/partners         – Top interaction partners for a gene
  POST /integrations/string/enrichment       – Functional enrichment via STRING
  GET  /integrations/geo/search              – Search public GEO datasets
  POST /integrations/cytoscape/cx2           – Export network to CX2 format
  POST /integrations/cytoscape/graphml       – Export network to GraphML
  POST /integrations/cytoscape/cytoscapejs   – Export network to cytoscape.js JSON
"""
import logging
import json
from typing import Optional, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.deps.project_access import assert_project_read_access
from app.core.supabase_auth import SupabaseUser
from app.models.models import Dataset
from app.services.external_integrations import (
    string_service,
    geo_service,
    cytoscape_exporter,
)
from app.services.species import declared_species, species_taxon

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/integrations", tags=["integrations"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StringNetworkRequest(BaseModel):
    gene_symbols: list[str] = Field(..., min_length=1, max_length=100)
    species: int = Field(
        default=9606,
        description="NCBI taxonomy ID (9606=human). With `dataset_id`, only the fallback "
        "for a dataset that records no species.",
    )
    dataset_id: Optional[UUID] = Field(
        default=None,
        description="Dataset the genes come from: the species its analysis was launched with "
        "wins over `species`.",
    )
    required_score: int = Field(default=400, ge=0, le=1000)
    limit: int = Field(default=10, ge=1, le=50)


class StringPartnersRequest(BaseModel):
    gene_symbol: str
    species: int = Field(default=9606)
    required_score: int = Field(default=700, ge=0, le=1000)
    limit: int = Field(default=20, ge=1, le=50)


class StringEnrichmentRequest(BaseModel):
    gene_symbols: list[str] = Field(..., min_length=1, max_length=500)
    species: int = Field(default=9606)


class CytoscapeExportRequest(BaseModel):
    network: dict = Field(..., description="Network dict with 'nodes' and 'edges'")
    network_name: str = Field(default="GenoLens PPI Network", max_length=200)


# ---------------------------------------------------------------------------
# STRING DB endpoints
# ---------------------------------------------------------------------------

async def _dataset_taxon(db: AsyncSession, dataset_id: UUID, user_id: UUID) -> Optional[int]:
    """STRING taxon of the species a dataset records, ``None`` when it records none we know.

    404 for a dataset the caller cannot read, so the route does not confirm it exists.
    """
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await assert_project_read_access(db, dataset.project_id, user_id)
    return species_taxon(await declared_species(db, dataset))


@router.post("/string/network", summary="Protein-protein interaction network")
async def string_network(
    body: StringNetworkRequest,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Fetch a PPI network for a list of gene symbols from STRING DB.

    Returns nodes and edges with confidence scores.
    Useful for visualising the interaction landscape of a DEG list.

    The organism is the one recorded for `dataset_id` when there is one — the species the
    analysis was launched with — else the caller's `species`. `species` in the response is the
    taxon STRING was asked about, and `species_source` says which of the two it came from
    (`dataset` or `request`).
    """
    species, species_source = body.species, "request"
    if body.dataset_id is not None:
        taxon = await _dataset_taxon(db, body.dataset_id, current_user.user_id)
        if taxon is not None:
            species, species_source = taxon, "dataset"

    result = await string_service.get_network(
        gene_symbols=body.gene_symbols,
        species=species,
        required_score=body.required_score,
        limit=body.limit,
    )
    result["species_source"] = species_source
    if "error" in result:
        logger.warning("STRING network error for %s genes: %s", len(body.gene_symbols), result["error"])
    return result


@router.post("/string/partners", summary="Top interaction partners for a gene")
async def string_partners(
    body: StringPartnersRequest,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> dict:
    """
    Return the top interaction partners for a single gene from STRING DB.

    Good for exploring a gene of interest directly from the DEG table.
    """
    result = await string_service.get_interaction_partners(
        gene_symbol=body.gene_symbol,
        species=body.species,
        required_score=body.required_score,
        limit=body.limit,
    )
    return result


@router.post("/string/enrichment", summary="Functional enrichment via STRING")
async def string_enrichment(
    body: StringEnrichmentRequest,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> dict:
    """
    Run functional enrichment of a gene list using STRING's annotation databases
    (GO Biological Process, KEGG, Reactome, WikiPathways, etc.).

    Complements the local GO enrichment analysis with STRING's broader annotation
    coverage and independent statistical testing.
    """
    result = await string_service.get_functional_enrichment(
        gene_symbols=body.gene_symbols,
        species=body.species,
    )
    return result


# ---------------------------------------------------------------------------
# GEO / NCBI endpoints
# ---------------------------------------------------------------------------

@router.get("/geo/search", summary="Search public GEO datasets")
async def geo_search(
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    q: str = Query(..., min_length=3, description="Free-text search query"),
    max_results: int = Query(default=10, ge=1, le=50),
    db: str = Query(default="gds", pattern="^gds$"),
    counts_only: bool = Query(
        default=False,
        description="Restrict to series with NCBI-generated RNA-seq counts (importable)",
    ),
) -> dict:
    """
    Search NCBI GEO for public gene expression datasets.

    Useful for finding reference datasets comparable to the user's own data.
    Results include accession, title, organism, number of samples and a direct
    GEO link.

    Example queries:
      - "breast cancer RNA-seq Homo sapiens"
      - "Alzheimer disease hippocampus"
      - "GSE223547"
    """
    result = await geo_service.search_datasets(
        query=q,
        max_results=max_results,
        db=db,
        counts_only=counts_only,
    )
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"NCBI API error: {result['error']}",
        )
    return result


# ---------------------------------------------------------------------------
# Cytoscape export endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/cytoscape/cx2",
    summary="Export network to CX2 (Cytoscape Desktop ≥ 3.9)",
    responses={200: {"content": {"application/json": {}}}},
)
async def export_cx2(
    body: CytoscapeExportRequest,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> Response:
    """
    Convert a STRING network to a **CX2** document.

    The returned JSON can be saved as `network.cx2` and opened directly in
    Cytoscape Desktop ≥ 3.9 or in NDEx.
    """
    cx2 = cytoscape_exporter.to_cx2(body.network, body.network_name)
    return Response(
        content=json.dumps(cx2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{body.network_name}.cx2"'},
    )


@router.post(
    "/cytoscape/graphml",
    summary="Export network to GraphML",
    responses={200: {"content": {"application/xml": {}}}},
)
async def export_graphml(
    body: CytoscapeExportRequest,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> Response:
    """
    Convert a STRING network to **GraphML** XML.

    GraphML is supported by Cytoscape, Gephi, igraph, NetworkX, yEd, and most
    other graph tools.  Save as `network.graphml`.
    """
    graphml = cytoscape_exporter.to_graphml(body.network, body.network_name)
    return Response(
        content=graphml,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{body.network_name}.graphml"'},
    )


@router.post(
    "/cytoscape/cytoscapejs",
    summary="Export network as cytoscape.js elements JSON",
)
async def export_cytoscapejs(
    body: CytoscapeExportRequest,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> dict:
    """
    Convert a STRING network to **cytoscape.js** `elements` format.

    The returned JSON can be passed directly to `cytoscape({ elements: ... })`
    in the browser without any additional transformation.
    """
    return cytoscape_exporter.to_cytoscape_js(body.network)
