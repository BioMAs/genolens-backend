"""
External Integrations Service for GenoLens.

Provides lightweight, async wrappers around public bioinformatics APIs:

  - **STRING DB**  : Protein-protein interaction networks (no API key required)
                     https://string-db.org/help/api/
  - **NCBI/GEO**   : Public gene expression dataset search via E-utilities
                     https://www.ncbi.nlm.nih.gov/books/NBK25499/
  - **Cytoscape**  : Export helper that generates CX2-compatible JSON
                     (importable by Cytoscape Desktop ≥ 3.9 and cytoscape.js)

All network calls use `httpx.AsyncClient` with sane timeouts so they never
block the event loop.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRING_API = "https://string-db.org/api"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DEFAULT_TIMEOUT = 15.0  # seconds


# ===========================================================================
# 1. STRING DB helpers
# ===========================================================================

class StringDBService:
    """
    Thin async wrapper around the STRING v12 public API.

    Species codes:
        9606  → Homo sapiens
        10090 → Mus musculus
        10116 → Rattus norvegicus
        7227  → Drosophila melanogaster
        6239  → Caenorhabditis elegans
    """

    # Default STRING caller identity (required by API)
    CALLER_IDENTITY = "genolens_app"

    async def get_network(
        self,
        gene_symbols: list[str],
        species: int = 9606,
        required_score: int = 400,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Fetch the interaction network for a list of proteins from STRING.

        Returns a dict with:
          - nodes  : list of {id, name, score, annotation}
          - edges  : list of {source, target, score, evidence}
          - species: query species
          - count  : number of edges

        Args:
            gene_symbols    : Up to 100 gene/protein symbols.
            species         : NCBI taxonomy ID (default 9606 = human).
            required_score  : Minimum combined interaction score 0-1000 (default 400).
            limit           : Max interaction partners per seed (default 10).
        """
        if not gene_symbols:
            return {"nodes": [], "edges": [], "species": species, "count": 0}

        # Clamp to 100 genes (STRING API limit)
        identifiers = "\r".join(gene_symbols[:100])

        params = {
            "identifiers": identifiers,
            "species": species,
            "required_score": required_score,
            "limit": limit,
            "caller_identity": self.CALLER_IDENTITY,
            "network_flavor": "confidence",
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.post(
                    f"{STRING_API}/json/network",
                    data=params,
                )
                resp.raise_for_status()
                raw = resp.json()
            except httpx.TimeoutException:
                logger.warning("STRING DB request timed out")
                return {"nodes": [], "edges": [], "species": species, "count": 0, "error": "timeout"}
            except httpx.HTTPStatusError as exc:
                logger.warning("STRING DB HTTP error: %s", exc)
                return {"nodes": [], "edges": [], "species": species, "count": 0, "error": str(exc)}

        return self._parse_network(raw, species)

    async def get_interaction_partners(
        self,
        gene_symbol: str,
        species: int = 9606,
        required_score: int = 700,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Fetch the top interaction partners for a single gene.

        Returns the same structure as `get_network`.
        """
        params = {
            "identifiers": gene_symbol,
            "species": species,
            "required_score": required_score,
            "limit": limit,
            "caller_identity": self.CALLER_IDENTITY,
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.post(
                    f"{STRING_API}/json/interaction_partners",
                    data=params,
                )
                resp.raise_for_status()
                raw = resp.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                logger.warning("STRING DB error: %s", exc)
                return {"nodes": [], "edges": [], "species": species, "count": 0}

        return self._parse_network(raw, species)

    async def get_functional_enrichment(
        self,
        gene_symbols: list[str],
        species: int = 9606,
    ) -> dict[str, Any]:
        """
        Functional enrichment of a gene list using STRING annotations
        (GO terms, KEGG pathways, Reactome, etc.).

        Returns dict with:
          - enrichments : list of {category, term, description, number_of_genes,
                                   number_of_genes_in_background, p_value, fdr,
                                   matching_genes}
        """
        if not gene_symbols:
            return {"enrichments": []}

        identifiers = "\r".join(gene_symbols[:500])
        params = {
            "identifiers": identifiers,
            "species": species,
            "caller_identity": self.CALLER_IDENTITY,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{STRING_API}/json/enrichment",
                    data=params,
                )
                resp.raise_for_status()
                raw = resp.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                logger.warning("STRING enrichment error: %s", exc)
                return {"enrichments": []}

        enrichments = []
        for item in raw:
            enrichments.append({
                "category": item.get("category", ""),
                "term": item.get("term", ""),
                "description": item.get("description", ""),
                "number_of_genes": item.get("number_of_genes", 0),
                "number_of_genes_in_background": item.get("number_of_genes_in_background", 0),
                "p_value": item.get("p_value", 1.0),
                "fdr": item.get("fdr", 1.0),
                "matching_genes": item.get("matching_proteins_in_your_network", ""),
            })

        # Sort by FDR
        enrichments.sort(key=lambda x: x["fdr"])
        return {"enrichments": enrichments}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_network(self, raw: list[dict], species: int) -> dict[str, Any]:
        """Convert STRING JSON response to our internal graph format."""
        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        for item in raw:
            # STRING returns interactions where both partners are listed
            a_id = item.get("stringId_A", "")
            b_id = item.get("stringId_B", "")
            a_name = item.get("preferredName_A", a_id)
            b_name = item.get("preferredName_B", b_id)
            score = item.get("score", 0)

            nodes[a_name] = {
                "id": a_name,
                "name": a_name,
                "string_id": a_id,
                "annotation": item.get("annotation", ""),
            }
            nodes[b_name] = {
                "id": b_name,
                "name": b_name,
                "string_id": b_id,
                "annotation": item.get("annotation", ""),
            }
            edges.append({
                "source": a_name,
                "target": b_name,
                "score": round(score, 4),
                "evidence": self._evidence_from_score(score),
            })

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "species": species,
            "count": len(edges),
        }

    @staticmethod
    def _evidence_from_score(score: float) -> str:
        if score >= 0.9:
            return "highest"
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"


# ===========================================================================
# 2. GEO / NCBI E-utilities helpers
# ===========================================================================

class GEOService:
    """
    Async wrapper around NCBI E-utilities to search and summarise GEO datasets.

    No API key is required for low-volume queries (< 10 req/s).
    """

    async def search_datasets(
        self,
        query: str,
        max_results: int = 10,
        db: str = "gds",
    ) -> dict[str, Any]:
        """
        Search GEO DataSets (db='gds') or GEO Series (db='gse') for datasets
        matching the given query string.

        Args:
            query       : Free-text query, e.g. "breast cancer RNA-seq Homo sapiens".
            max_results : Maximum number of records to return (default 10).
            db          : NCBI database identifier ('gds' or 'geo').

        Returns dict with:
          - total    : total hits in NCBI
          - ids      : list of GEO UIDs
          - datasets : list of summarised dataset records (populated if ≤ 50 ids)
        """
        params = {
            "db": db,
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.get(NCBI_ESEARCH, params=params)
                resp.raise_for_status()
                search_result = resp.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                logger.warning("NCBI esearch error: %s", exc)
                return {"total": 0, "ids": [], "datasets": [], "error": str(exc)}

        esearch = search_result.get("esearchresult", {})
        total = int(esearch.get("count", 0))
        ids = esearch.get("idlist", [])

        datasets: list[dict] = []
        if ids:
            datasets = await self._fetch_summaries(ids, db)

        return {
            "total": total,
            "ids": ids,
            "datasets": datasets,
            "query": query,
            "db": db,
        }

    async def _fetch_summaries(
        self,
        ids: list[str],
        db: str = "gds",
    ) -> list[dict]:
        """Fetch ESummary records for the given GEO UIDs."""
        params = {
            "db": db,
            "id": ",".join(ids[:50]),
            "retmode": "json",
        }

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.get(NCBI_ESUMMARY, params=params)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                logger.warning("NCBI esummary error: %s", exc)
                return []

        result_set = data.get("result", {})
        uids = result_set.get("uids", [])

        datasets = []
        for uid in uids:
            rec = result_set.get(uid, {})
            datasets.append(self._parse_gds_record(uid, rec))

        return datasets

    @staticmethod
    def _parse_gds_record(uid: str, rec: dict) -> dict:
        """Normalise a GDS ESummary record into a simplified dict."""
        # accession may be GDS, GSE, GPL, or GSM
        accession = rec.get("accession", f"UID:{uid}")
        title = rec.get("title", "")
        summary = rec.get("summary", "")
        organism = rec.get("taxon", "")
        samples_n = rec.get("n_samples", 0)
        platform = rec.get("gpl", "")
        dataset_type = rec.get("type", "")
        entry_type = rec.get("entrytype", "")
        pub_date = rec.get("pdat", "")

        geo_link = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
        return {
            "uid": uid,
            "accession": accession,
            "title": title,
            "summary": summary[:400] + "…" if len(summary) > 400 else summary,
            "organism": organism,
            "samples_n": samples_n,
            "platform": platform,
            "type": dataset_type,
            "entry_type": entry_type,
            "pub_date": pub_date,
            "geo_link": geo_link,
        }


# ===========================================================================
# 3. Cytoscape export (CX2-like JSON)
# ===========================================================================

class CytoscapeExporter:
    """
    Converts a STRING-style network (nodes + edges) to formats importable
    by Cytoscape Desktop and cytoscape.js.

    Supported formats:
      - cx2       : CX2 JSON (Cytoscape Desktop ≥ 3.9 native format)
      - graphml   : GraphML XML (universally supported)
      - json      : Simple cytoscape.js elements JSON
    """

    # ------------------------------------------------------------------
    # CX2 format
    # ------------------------------------------------------------------

    def to_cx2(
        self,
        network: dict[str, Any],
        network_name: str = "GenoLens PPI Network",
    ) -> dict[str, Any]:
        """
        Build a minimal CX2 document from a STRING network dict.

        CX2 spec: https://cytoscape.org/cx/cx2/
        The returned dict can be serialised with json.dumps and saved as *.cx2.
        """
        nodes = network.get("nodes", [])
        edges = network.get("edges", [])

        node_index: dict[str, int] = {n["id"]: i for i, n in enumerate(nodes)}

        cx_nodes = []
        for i, node in enumerate(nodes):
            cx_nodes.append({
                "id": i,
                "x": 0.0,
                "y": 0.0,
                "v": {
                    "name": node["name"],
                    "annotation": node.get("annotation", ""),
                    "string_id": node.get("string_id", ""),
                },
            })

        cx_edges = []
        for i, edge in enumerate(edges):
            s = node_index.get(edge["source"], -1)
            t = node_index.get(edge["target"], -1)
            if s == -1 or t == -1:
                continue
            cx_edges.append({
                "id": i,
                "s": s,
                "t": t,
                "v": {
                    "interaction": "interacts with",
                    "score": edge.get("score", 0),
                    "evidence": edge.get("evidence", ""),
                },
            })

        metadata = [
            {"name": "nodes", "elementCount": len(cx_nodes), "version": "1.0"},
            {"name": "edges", "elementCount": len(cx_edges), "version": "1.0"},
        ]

        return [
            {"CXVersion": "2.0", "hasFragments": False},
            {"metaData": metadata},
            {"networkAttributes": [{"v": {
                "name": network_name,
                "description": f"Exported from GenoLens. Species: {network.get('species', '')}",
                "__Annotations": [],
            }}]},
            {"nodes": cx_nodes},
            {"edges": cx_edges},
            {"status": [{"success": True}]},
        ]

    # ------------------------------------------------------------------
    # GraphML format
    # ------------------------------------------------------------------

    def to_graphml(
        self,
        network: dict[str, Any],
        network_name: str = "GenoLens PPI Network",
    ) -> str:
        """Return a GraphML XML string for the given network."""
        nodes = network.get("nodes", [])
        edges = network.get("edges", [])

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/graphml"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/graphml '
            'http://graphml.graphdrawing.org/graphml/1.0/graphml.xsd">',
            '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
            '  <key id="annotation" for="node" attr.name="annotation" attr.type="string"/>',
            '  <key id="score" for="edge" attr.name="score" attr.type="double"/>',
            '  <key id="evidence" for="edge" attr.name="evidence" attr.type="string"/>',
            f'  <graph id="{network_name}" edgedefault="undirected">',
        ]

        for node in nodes:
            nid = _xml_escape(node["id"])
            lines.append(f'    <node id="{nid}">')
            lines.append(f'      <data key="name">{_xml_escape(node.get("name", node["id"]))}</data>')
            lines.append(f'      <data key="annotation">{_xml_escape(node.get("annotation", ""))}</data>')
            lines.append('    </node>')

        for i, edge in enumerate(edges):
            src = _xml_escape(edge["source"])
            tgt = _xml_escape(edge["target"])
            lines.append(f'    <edge id="e{i}" source="{src}" target="{tgt}">')
            lines.append(f'      <data key="score">{edge.get("score", 0)}</data>')
            lines.append(f'      <data key="evidence">{edge.get("evidence", "")}</data>')
            lines.append('    </edge>')

        lines += ['  </graph>', '</graphml>']
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # cytoscape.js JSON format
    # ------------------------------------------------------------------

    def to_cytoscape_js(self, network: dict[str, Any]) -> dict[str, Any]:
        """
        Return cytoscape.js `elements` JSON so the frontend can render
        the network with `cytoscape` npm package without any server round-trip.
        """
        nodes = [
            {
                "data": {
                    "id": node["id"],
                    "label": node.get("name", node["id"]),
                    "annotation": node.get("annotation", ""),
                    "string_id": node.get("string_id", ""),
                }
            }
            for node in network.get("nodes", [])
        ]
        edges = [
            {
                "data": {
                    "id": f"{edge['source']}_{edge['target']}",
                    "source": edge["source"],
                    "target": edge["target"],
                    "score": edge.get("score", 0),
                    "evidence": edge.get("evidence", ""),
                }
            }
            for edge in network.get("edges", [])
        ]
        return {"elements": {"nodes": nodes, "edges": edges}}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _xml_escape(text: str) -> str:
    """Minimal XML character escaping."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

string_service = StringDBService()
geo_service = GEOService()
cytoscape_exporter = CytoscapeExporter()
