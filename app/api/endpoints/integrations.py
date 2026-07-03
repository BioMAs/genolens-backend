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

from fastapi import APIRouter, Depends, HTTPException, Query, status, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.supabase_auth import SupabaseUser
from app.services.external_integrations import (
    string_service,
    geo_service,
    cytoscape_exporter,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/integrations", tags=["integrations"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StringNetworkRequest(BaseModel):
    gene_symbols: list[str] = Field(..., min_length=1, max_length=100)
    species: int = Field(default=9606, description="NCBI taxonomy ID (9606=human)")
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

@router.post("/string/network", summary="Protein-protein interaction network")
async def string_network(
    body: StringNetworkRequest,
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
) -> dict:
    """
    Fetch a PPI network for a list of gene symbols from STRING DB.

    Returns nodes and edges with confidence scores.  
    Useful for visualising the interaction landscape of a DEG list.
    """
    result = await string_service.get_network(
        gene_symbols=body.gene_symbols,
        species=body.species,
        required_score=body.required_score,
        limit=body.limit,
    )
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
    db: str = Query(default="gds", pattern="^(gds|geo)$"),
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
