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

import gzip
import io
import json
import logging
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRING_API = "https://string-db.org/api"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# GEO web download endpoint (NCBI-generated RNA-seq counts + annotation tables)
NCBI_GEO_DOWNLOAD = "https://www.ncbi.nlm.nih.gov/geo/download/"
# FTP-served series matrix (sample-level metadata)
NCBI_GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"

DEFAULT_TIMEOUT = 15.0  # seconds
DOWNLOAD_TIMEOUT = 120.0  # seconds — larger for count-matrix downloads

# Per-organism NCBI reference assembly used to name the generated count files.
# Only human and mouse are processed by NCBI (see rnaseqcounts.html).
GEO_ORGANISMS: dict[str, dict[str, str]] = {
    "human": {"assembly": "GRCh38.p13", "annot": "Human.GRCh38.p13.annot.tsv.gz"},
    "mouse": {"assembly": "GRCm39", "annot": "Mouse.GRCm39.annot.tsv.gz"},
}


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
        counts_only: bool = False,
    ) -> dict[str, Any]:
        """
        Search GEO DataSets (db='gds') or GEO Series (db='gse') for datasets
        matching the given query string.

        Args:
            query       : Free-text query, e.g. "breast cancer RNA-seq Homo sapiens".
            max_results : Maximum number of records to return (default 10).
            db          : NCBI database identifier ('gds' or 'geo').
            counts_only : When True, restrict results to series that have
                          NCBI-generated RNA-seq counts (i.e. importable).

        Returns dict with:
          - total    : total hits in NCBI
          - ids      : list of GEO UIDs
          - datasets : list of summarised dataset records (populated if ≤ 50 ids)
        """
        term = f'({query}) AND "rnaseq counts"[Filter]' if counts_only else query
        params = {
            "db": db,
            "term": term,
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

    # ------------------------------------------------------------------
    # NCBI-generated RNA-seq counts + sample metadata (import support)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_organism(organism: str) -> str:
        """Map a free-text organism to a supported GEO key ('human'/'mouse')."""
        value = (organism or "").strip().lower()
        if value in ("human", "homo sapiens", "hsapiens", "hs"):
            return "human"
        if value in ("mouse", "mus musculus", "mmusculus", "mm"):
            return "mouse"
        raise ValueError(
            f"Unsupported organism '{organism}'. NCBI-generated counts are only "
            "available for human and mouse."
        )

    @classmethod
    def _raw_counts_url(cls, accession: str, organism: str) -> str:
        key = cls._normalize_organism(organism)
        assembly = GEO_ORGANISMS[key]["assembly"]
        filename = f"{accession}_raw_counts_{assembly}_NCBI.tsv.gz"
        return (
            f"{NCBI_GEO_DOWNLOAD}?type=rnaseq_counts&acc={accession}"
            f"&format=file&file={urllib.parse.quote(filename)}"
        )

    @classmethod
    def _annotation_url(cls, organism: str) -> str:
        key = cls._normalize_organism(organism)
        filename = GEO_ORGANISMS[key]["annot"]
        return (
            f"{NCBI_GEO_DOWNLOAD}?type=rnaseq_counts"
            f"&format=file&file={urllib.parse.quote(filename)}"
        )

    async def check_counts_availability(self, accession: str, organism: str) -> bool:
        """
        Return True if NCBI has generated an RNA-seq raw-count matrix for the
        given series. Issues a lightweight ranged GET (the download endpoint
        does not support HEAD reliably) and inspects the response.
        """
        try:
            url = self._raw_counts_url(accession, organism)
        except ValueError:
            return False

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            try:
                resp = await client.get(url, headers={"Range": "bytes=0-1023"})
            except httpx.HTTPError as exc:
                logger.warning("GEO counts availability check failed for %s: %s", accession, exc)
                return False

        # Missing files return an HTML error page (text/html); real counts are gzip.
        content_type = resp.headers.get("content-type", "")
        if resp.status_code not in (200, 206):
            return False
        if "html" in content_type.lower():
            return False
        return bool(resp.content)

    async def download_rnaseq_counts(self, accession: str, organism: str) -> pd.DataFrame:
        """
        Download and parse the NCBI-generated raw-count matrix for a series.

        Returns a DataFrame indexed by the raw NCBI GeneID (first column),
        with one column per GSM sample accession.
        """
        url = self._raw_counts_url(accession, organism)
        raw = await self._download_bytes(url, f"raw counts for {accession}")
        data = gzip.decompress(raw)
        df = pd.read_csv(io.BytesIO(data), sep="\t")
        if df.empty or df.shape[1] < 2:
            raise ValueError(f"Empty or malformed count matrix for {accession}")
        # First column holds the GeneID; keep it as a plain column named 'gene_id'.
        df = df.rename(columns={df.columns[0]: "gene_id"})
        df["gene_id"] = df["gene_id"].astype(str)
        return df

    async def download_annotation(self, organism: str) -> dict[str, str]:
        """
        Return a {GeneID -> gene symbol} mapping for the organism, cached on
        local disk so it is fetched at most once per assembly.
        """
        key = self._normalize_organism(organism)
        cache_dir = Path(settings.LOCAL_STORAGE_PATH) / "geo_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{key}_geneid_symbol.json"

        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt GEO annotation cache %s — refetching", cache_file)

        url = self._annotation_url(organism)
        raw = await self._download_bytes(url, f"{key} gene annotation")
        data = gzip.decompress(raw)
        annot = pd.read_csv(io.BytesIO(data), sep="\t", usecols=lambda c: c in ("GeneID", "Symbol"))
        mapping = {
            str(gid): str(sym)
            for gid, sym in zip(annot["GeneID"], annot["Symbol"])
            if pd.notna(sym) and str(sym).strip()
        }
        try:
            cache_file.write_text(json.dumps(mapping))
        except OSError as exc:
            logger.warning("Could not write GEO annotation cache: %s", exc)
        return mapping

    async def download_series_metadata(self, accession: str) -> pd.DataFrame:
        """
        Download the series matrix header and build a per-sample metadata sheet.

        Returns a DataFrame with a 'sample' column (GSM accession, matching the
        count-matrix columns) plus 'title' and one column per characteristic.
        """
        raw = await self._download_series_matrix(accession)
        text = gzip.decompress(raw).decode("utf-8", errors="replace")

        accessions: list[str] = []
        titles: list[str] = []
        characteristics: list[list[str]] = []  # one list per !Sample_characteristics_ch1 row

        for line in text.splitlines():
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            key, _, rest = line.partition("\t")
            values = [v.strip().strip('"') for v in rest.split("\t")]
            if key == "!Sample_geo_accession":
                accessions = values
            elif key == "!Sample_title":
                titles = values
            elif key == "!Sample_characteristics_ch1":
                characteristics.append(values)

        if not accessions:
            raise ValueError(f"No sample metadata found in series matrix for {accession}")

        rows: list[dict[str, str]] = []
        for i, gsm in enumerate(accessions):
            row: dict[str, str] = {"sample": gsm}
            if i < len(titles):
                row["title"] = titles[i]
            for char_row in characteristics:
                if i >= len(char_row):
                    continue
                raw_val = char_row[i]
                if ":" in raw_val:
                    ck, _, cv = raw_val.partition(":")
                    col = ck.strip() or f"characteristic_{len(row)}"
                    row[col] = cv.strip()
                elif raw_val:
                    row[f"characteristic_{len(row)}"] = raw_val
            rows.append(row)

        return pd.DataFrame(rows)

    async def _download_series_matrix(self, accession: str) -> bytes:
        """Fetch the gzipped series matrix file from the NCBI GEO FTP mirror."""
        # Series live under a stub dir where the last 3 digits become 'nnn'
        # (e.g. GSE164073 → GSE164nnn, GSE9999 → GSE9nnn, GSE123 → GSEnnn).
        digits = accession[3:]
        prefix = digits[:-3] if len(digits) > 3 else ""
        stub = f"GSE{prefix}nnn"
        url = (
            f"{NCBI_GEO_FTP}/{stub}/{accession}/matrix/"
            f"{accession}_series_matrix.txt.gz"
        )
        return await self._download_bytes(url, f"series matrix for {accession}")

    @staticmethod
    async def _download_bytes(url: str, what: str) -> bytes:
        """GET a URL and return raw bytes, raising ValueError on HTTP errors."""
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("GEO download failed (%s): %s", what, exc)
                raise ValueError(f"Failed to download {what} from GEO") from exc
        if not resp.content:
            raise ValueError(f"Empty response downloading {what} from GEO")
        return resp.content


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
