"""
Tests for the external integrations service.

Covers:
  - StringDBService.get_network        (mocked httpx)
  - StringDBService.get_interaction_partners
  - StringDBService.get_functional_enrichment
  - GEOService.search_datasets         (mocked httpx)
  - GEOService._parse_gds_record
  - CytoscapeExporter.to_cx2
  - CytoscapeExporter.to_graphml
  - CytoscapeExporter.to_cytoscape_js
  - _xml_escape helper
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.services.external_integrations import (
    StringDBService,
    GEOService,
    CytoscapeExporter,
    _xml_escape,
    string_service,
    geo_service,
    cytoscape_exporter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_STRING_RESPONSE = [
    {
        "stringId_A": "9606.ENSP00000269305",
        "stringId_B": "9606.ENSP00000338694",
        "preferredName_A": "TP53",
        "preferredName_B": "MDM2",
        "score": 0.999,
        "annotation": "p53 – MDM2 interaction",
    },
    {
        "stringId_A": "9606.ENSP00000269305",
        "stringId_B": "9606.ENSP00000349459",
        "preferredName_A": "TP53",
        "preferredName_B": "CDKN1A",
        "score": 0.99,
        "annotation": "p53 – p21 interaction",
    },
]

SAMPLE_GEO_SEARCH_RESPONSE = {
    "esearchresult": {
        "count": "42",
        "idlist": ["200178967", "200156738"],
    }
}

SAMPLE_GEO_SUMMARY_RESPONSE = {
    "result": {
        "uids": ["200178967", "200156738"],
        "200178967": {
            "accession": "GDS5847",
            "title": "Breast cancer RNA-seq dataset",
            "summary": "A sample dataset for breast cancer gene expression profiling.",
            "taxon": "Homo sapiens",
            "n_samples": 20,
            "gpl": "GPL16791",
            "type": "Expression profiling by high throughput sequencing",
            "entrytype": "GDS",
            "pdat": "2025/01/15",
        },
        "200156738": {
            "accession": "GSE156738",
            "title": "Lung adenocarcinoma transcriptome",
            "summary": "RNA-seq in lung adenocarcinoma vs normal.",
            "taxon": "Homo sapiens",
            "n_samples": 30,
            "gpl": "GPL24676",
            "type": "Expression profiling by high throughput sequencing",
            "entrytype": "GSE",
            "pdat": "2024/06/10",
        },
    }
}

SAMPLE_NETWORK = {
    "nodes": [
        {"id": "TP53", "name": "TP53", "string_id": "9606.ENSP00000269305", "annotation": "Tumor protein p53"},
        {"id": "MDM2", "name": "MDM2", "string_id": "9606.ENSP00000338694", "annotation": "Mouse double minute 2"},
        {"id": "CDKN1A", "name": "CDKN1A", "string_id": "9606.ENSP00000349459", "annotation": "Cyclin-dependent kinase inhibitor"},
    ],
    "edges": [
        {"source": "TP53", "target": "MDM2", "score": 0.999, "evidence": "highest"},
        {"source": "TP53", "target": "CDKN1A", "score": 0.99, "evidence": "highest"},
    ],
    "species": 9606,
    "count": 2,
}


# ─────────────────────────────────────────────────────────────────────────────
# StringDBService tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStringDBService:

    @pytest.mark.asyncio
    async def test_get_network_returns_nodes_and_edges(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_STRING_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch("app.services.external_integrations.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            service = StringDBService()
            result = await service.get_network(["TP53", "MDM2", "CDKN1A"])

        assert len(result["nodes"]) == 3  # TP53, MDM2, CDKN1A
        assert len(result["edges"]) == 2
        assert result["count"] == 2
        assert result["species"] == 9606

    @pytest.mark.asyncio
    async def test_get_network_empty_input(self):
        service = StringDBService()
        result = await service.get_network([])
        assert result == {"nodes": [], "edges": [], "species": 9606, "count": 0}

    @pytest.mark.asyncio
    async def test_get_network_timeout_returns_empty(self):
        with patch("app.services.external_integrations.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_client

            service = StringDBService()
            result = await service.get_network(["TP53"])

        assert result["count"] == 0
        assert "error" in result
        assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_get_interaction_partners(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_STRING_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch("app.services.external_integrations.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            service = StringDBService()
            result = await service.get_interaction_partners("TP53")

        assert len(result["nodes"]) >= 2
        assert result["count"] > 0

    @pytest.mark.asyncio
    async def test_get_functional_enrichment_empty(self):
        service = StringDBService()
        result = await service.get_functional_enrichment([])
        assert result == {"enrichments": []}

    @pytest.mark.asyncio
    async def test_get_functional_enrichment_returns_sorted(self):
        enrichment_resp = [
            {"category": "KEGG", "term": "hsa05200", "description": "Pathways in cancer",
             "number_of_genes": 15, "number_of_genes_in_background": 342,
             "p_value": 0.001, "fdr": 0.05, "matching_proteins_in_your_network": "TP53,MYC"},
            {"category": "Process", "term": "GO:0006915", "description": "Apoptotic process",
             "number_of_genes": 8, "number_of_genes_in_background": 200,
             "p_value": 0.0001, "fdr": 0.002, "matching_proteins_in_your_network": "TP53,BCL2"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = enrichment_resp
        mock_resp.raise_for_status = MagicMock()

        with patch("app.services.external_integrations.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            service = StringDBService()
            result = await service.get_functional_enrichment(["TP53", "MYC", "BCL2"])

        enrichments = result["enrichments"]
        assert len(enrichments) == 2
        # Should be sorted by FDR ascending
        assert enrichments[0]["fdr"] <= enrichments[1]["fdr"]
        assert enrichments[0]["term"] == "GO:0006915"  # fdr=0.002 comes first

    def test_evidence_from_score(self):
        s = StringDBService()
        assert s._evidence_from_score(0.95) == "highest"
        assert s._evidence_from_score(0.75) == "high"
        assert s._evidence_from_score(0.5) == "medium"
        assert s._evidence_from_score(0.2) == "low"

    def test_parse_network_deduplicates_nodes(self):
        s = StringDBService()
        # Both rows share TP53 as preferredName_A
        result = s._parse_network(SAMPLE_STRING_RESPONSE, 9606)
        node_names = [n["name"] for n in result["nodes"]]
        assert node_names.count("TP53") == 1  # deduplicated


# ─────────────────────────────────────────────────────────────────────────────
# GEOService tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGEOService:

    @pytest.mark.asyncio
    async def test_search_datasets_returns_datasets(self):
        esearch_resp = MagicMock()
        esearch_resp.json.return_value = SAMPLE_GEO_SEARCH_RESPONSE
        esearch_resp.raise_for_status = MagicMock()

        esummary_resp = MagicMock()
        esummary_resp.json.return_value = SAMPLE_GEO_SUMMARY_RESPONSE
        esummary_resp.raise_for_status = MagicMock()

        with patch("app.services.external_integrations.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            # First call = esearch, second = esummary
            mock_client.get = AsyncMock(side_effect=[esearch_resp, esummary_resp])
            mock_client_cls.return_value = mock_client

            service = GEOService()
            result = await service.search_datasets("breast cancer RNA-seq")

        assert result["total"] == 42
        assert len(result["datasets"]) == 2
        assert result["datasets"][0]["accession"] == "GDS5847"

    @pytest.mark.asyncio
    async def test_search_datasets_ncbi_error(self):
        with patch("app.services.external_integrations.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_client

            service = GEOService()
            result = await service.search_datasets("breast cancer")

        assert result["total"] == 0
        assert "error" in result

    def test_parse_gds_record_fields(self):
        uid = "200178967"
        rec = SAMPLE_GEO_SUMMARY_RESPONSE["result"][uid]
        parsed = GEOService._parse_gds_record(uid, rec)

        assert parsed["uid"] == uid
        assert parsed["accession"] == "GDS5847"
        assert parsed["organism"] == "Homo sapiens"
        assert parsed["samples_n"] == 20
        assert "geo_link" in parsed
        assert "GDS5847" in parsed["geo_link"]

    def test_parse_gds_record_truncates_long_summary(self):
        long_summary = "A" * 500
        rec = {"summary": long_summary, "accession": "GDS9999"}
        parsed = GEOService._parse_gds_record("999", rec)
        assert len(parsed["summary"]) <= 403  # 400 chars + "…"
        assert parsed["summary"].endswith("…")

    def test_parse_gds_record_short_summary_not_truncated(self):
        rec = {"summary": "Short summary.", "accession": "GDS0001"}
        parsed = GEOService._parse_gds_record("1", rec)
        assert parsed["summary"] == "Short summary."


# ─────────────────────────────────────────────────────────────────────────────
# CytoscapeExporter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCytoscapeExporter:

    def test_to_cx2_structure(self):
        exporter = CytoscapeExporter()
        cx2 = exporter.to_cx2(SAMPLE_NETWORK, "Test Network")

        # Must be a list (CX2 array format)
        assert isinstance(cx2, list)
        # First element declares CX version
        assert cx2[0].get("CXVersion") == "2.0"
        # Must contain nodes and edges fragments
        keys = []
        for item in cx2:
            keys.extend(item.keys())
        assert "nodes" in keys
        assert "edges" in keys

    def test_to_cx2_nodes_count(self):
        exporter = CytoscapeExporter()
        cx2 = exporter.to_cx2(SAMPLE_NETWORK)
        nodes_fragment = next(item for item in cx2 if "nodes" in item)
        assert len(nodes_fragment["nodes"]) == 3

    def test_to_cx2_edges_count(self):
        exporter = CytoscapeExporter()
        cx2 = exporter.to_cx2(SAMPLE_NETWORK)
        edges_fragment = next(item for item in cx2 if "edges" in item)
        assert len(edges_fragment["edges"]) == 2

    def test_to_cx2_empty_network(self):
        exporter = CytoscapeExporter()
        empty = {"nodes": [], "edges": [], "species": 9606, "count": 0}
        cx2 = exporter.to_cx2(empty)
        nodes_fragment = next(item for item in cx2 if "nodes" in item)
        assert len(nodes_fragment["nodes"]) == 0

    def test_to_graphml_valid_xml(self):
        exporter = CytoscapeExporter()
        graphml = exporter.to_graphml(SAMPLE_NETWORK, "Test")

        assert graphml.startswith('<?xml')
        assert '<graphml' in graphml
        assert '<node id="TP53"' in graphml
        assert '<node id="MDM2"' in graphml
        assert '<edge' in graphml
        assert 'source="TP53"' in graphml

    def test_to_graphml_xml_special_chars(self):
        network = {
            "nodes": [{"id": "GENE&A", "name": "Gene <A>", "annotation": "test"}],
            "edges": [],
            "species": 9606,
            "count": 0,
        }
        exporter = CytoscapeExporter()
        graphml = exporter.to_graphml(network)
        # Special chars must be escaped
        assert "&amp;" in graphml or "&lt;" in graphml

    def test_to_cytoscape_js_structure(self):
        exporter = CytoscapeExporter()
        result = exporter.to_cytoscape_js(SAMPLE_NETWORK)

        assert "elements" in result
        elements = result["elements"]
        assert "nodes" in elements
        assert "edges" in elements
        assert len(elements["nodes"]) == 3
        assert len(elements["edges"]) == 2

    def test_to_cytoscape_js_node_has_label(self):
        exporter = CytoscapeExporter()
        result = exporter.to_cytoscape_js(SAMPLE_NETWORK)
        node = result["elements"]["nodes"][0]
        assert "data" in node
        assert "id" in node["data"]
        assert "label" in node["data"]

    def test_to_cytoscape_js_edge_has_source_target(self):
        exporter = CytoscapeExporter()
        result = exporter.to_cytoscape_js(SAMPLE_NETWORK)
        edge = result["elements"]["edges"][0]
        assert "source" in edge["data"]
        assert "target" in edge["data"]
        assert "score" in edge["data"]


# ─────────────────────────────────────────────────────────────────────────────
# _xml_escape helper
# ─────────────────────────────────────────────────────────────────────────────

class TestXmlEscape:
    def test_ampersand(self):
        assert _xml_escape("A & B") == "A &amp; B"

    def test_less_than(self):
        assert _xml_escape("<tag>") == "&lt;tag&gt;"

    def test_quote(self):
        assert _xml_escape('say "hello"') == 'say &quot;hello&quot;'

    def test_no_special_chars(self):
        assert _xml_escape("plain text") == "plain text"

    def test_combined(self):
        result = _xml_escape('<gene name="A&B">')
        assert '&lt;' in result
        assert '&amp;' in result
        assert '&quot;' in result


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons are properly instantiated
# ─────────────────────────────────────────────────────────────────────────────

class TestSingletons:
    def test_string_service_singleton(self):
        assert isinstance(string_service, StringDBService)

    def test_geo_service_singleton(self):
        assert isinstance(geo_service, GEOService)

    def test_cytoscape_exporter_singleton(self):
        assert isinstance(cytoscape_exporter, CytoscapeExporter)
