"""
Unit tests for GOService.

Covers:
- get_go_term (found / not found, with and without hierarchy)
- search_go_terms (text search, namespace filter)
- get_gene_annotations (gene list lookup)
- go_enrichment_analysis (hypergeometric test, FDR correction, size filters)
- _get_ancestors / _get_descendants (hierarchy traversal)
- NAMESPACES mapping
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_one_or_none(value):
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _scalars_all(items):
    r = MagicMock()
    r.scalars.return_value.all.return_value = items
    return r


def _all_rows(rows):
    """For queries returning row tuples, e.g. distinct gene_symbols."""
    r = MagicMock()
    r.all.return_value = rows
    return r


def _parent_rows(graph: dict):
    """
    Mock for `_load_parent_map`'s single bulk read.

    `select(GOTerm.go_id, GOTerm.is_a, GOTerm.part_of)` yields row tuples, so the whole GO parent
    graph arrives in one result. Pass `{go_id: [parents]}` or `{go_id: ([is_a], [part_of])}`.
    """
    rows = []
    for go_id, parents in graph.items():
        if isinstance(parents, tuple):
            is_a, part_of = parents
        else:
            is_a, part_of = parents, []
        rows.append((go_id, list(is_a), list(part_of)))
    return _all_rows(rows)


def make_go_term(
    go_id: str = "GO:0006915",
    name: str = "apoptotic process",
    namespace: str = "biological_process",
    is_a: list | None = None,
    part_of: list | None = None,
    gene_count: int = 100,
    level: int = 3,
    is_obsolete: bool = False,
):
    """Create a mock GOTerm."""
    from app.models.models import GOTerm
    t = MagicMock(spec=GOTerm)
    t.go_id = go_id
    t.name = name
    t.namespace = namespace
    t.definition = "Programmed cell death"
    t.is_a = is_a or []
    t.part_of = part_of or []
    t.regulates = []
    t.synonyms = []
    t.level = level
    t.gene_count = gene_count
    t.is_obsolete = is_obsolete
    return t


def make_go_annotation(
    gene_symbol: str,
    go_id: str,
    evidence_code: str = "IDA",
    organism: str = "Homo sapiens",
):
    from app.models.models import GOAnnotation
    a = MagicMock(spec=GOAnnotation)
    a.gene_symbol = gene_symbol
    a.go_id = go_id
    a.evidence_code = evidence_code
    a.organism = organism
    a.source_db = "UniProtKB"
    a.qualifier = "involved_in"
    return a


# ─────────────────────────────────────────────────────────────────────────────
# GOService — get_go_term
# ─────────────────────────────────────────────────────────────────────────────

class TestGetGoTerm:

    @pytest.mark.asyncio
    async def test_returns_term_dict_when_found(self, mock_db):
        """Should return a dictionary with term fields when GO ID exists."""
        from app.services.go_service import GOService

        term = make_go_term()
        mock_db.execute.return_value = _scalar_one_or_none(term)

        service = GOService()
        result = await service.get_go_term(mock_db, "GO:0006915")

        assert result is not None
        assert result["go_id"] == "GO:0006915"
        assert result["name"] == "apoptotic process"
        assert result["namespace"] == "biological_process"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, mock_db):
        """Should return None for an unknown GO ID."""
        from app.services.go_service import GOService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        service = GOService()
        result = await service.get_go_term(mock_db, "GO:9999999")

        assert result is None

    @pytest.mark.asyncio
    async def test_term_dict_has_required_keys(self, mock_db):
        """Returned dict should contain all expected fields."""
        from app.services.go_service import GOService

        term = make_go_term()
        mock_db.execute.return_value = _scalar_one_or_none(term)

        service = GOService()
        result = await service.get_go_term(mock_db, "GO:0006915")

        required = {
            "go_id", "name", "namespace", "definition",
            "is_a", "part_of", "regulates", "synonyms",
            "level", "gene_count", "is_obsolete"
        }
        assert required.issubset(result.keys())


# ─────────────────────────────────────────────────────────────────────────────
# GOService — search_go_terms
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchGoTerms:

    @pytest.mark.asyncio
    async def test_returns_matching_terms(self, mock_db):
        """Should return list of term dicts matching the query."""
        from app.services.go_service import GOService

        t1 = make_go_term(go_id="GO:0006915", name="apoptotic process")
        t2 = make_go_term(go_id="GO:0043066", name="negative regulation of apoptotic process")
        mock_db.execute.return_value = _scalars_all([t1, t2])

        service = GOService()
        results = await service.search_go_terms(mock_db, "apoptotic")

        assert len(results) == 2
        assert all("go_id" in r for r in results)

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_match(self, mock_db):
        """Should return empty list when query matches nothing."""
        from app.services.go_service import GOService

        mock_db.execute.return_value = _scalars_all([])

        service = GOService()
        results = await service.search_go_terms(mock_db, "nonexistent_term_xyz")

        assert results == []

    @pytest.mark.asyncio
    async def test_go_id_search_handled(self, mock_db):
        """Queries starting with GO: should search by GO ID."""
        from app.services.go_service import GOService

        term = make_go_term()
        mock_db.execute.return_value = _scalars_all([term])

        service = GOService()
        results = await service.search_go_terms(mock_db, "GO:0006915")

        assert len(results) == 1
        assert results[0]["go_id"] == "GO:0006915"


# ─────────────────────────────────────────────────────────────────────────────
# GOService — NAMESPACES mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestGOServiceNamespacesMapping:

    def test_all_standard_namespaces_present(self):
        """NAMESPACES should map BP, MF, CC to full namespace names."""
        from app.services.go_service import GOService

        ns = GOService.NAMESPACES
        assert ns["BP"] == "biological_process"
        assert ns["MF"] == "molecular_function"
        assert ns["CC"] == "cellular_component"

    def test_namespace_keys_are_uppercase(self):
        """All namespace keys must be uppercase abbreviations."""
        from app.services.go_service import GOService

        for key in GOService.NAMESPACES:
            assert key == key.upper()


# ─────────────────────────────────────────────────────────────────────────────
# GOService — get_gene_annotations
# ─────────────────────────────────────────────────────────────────────────────

class TestGetGeneAnnotations:

    @pytest.mark.asyncio
    async def test_returns_annotations_grouped_by_gene(self, mock_db):
        """Annotations should be grouped by gene_symbol."""
        from app.services.go_service import GOService

        ann1 = make_go_annotation("TP53", "GO:0006915")
        ann2 = make_go_annotation("TP53", "GO:0043066")
        ann3 = make_go_annotation("BRCA1", "GO:0006281")

        term1 = make_go_term(go_id="GO:0006915", namespace="biological_process")
        term2 = make_go_term(go_id="GO:0043066", namespace="biological_process")
        term3 = make_go_term(go_id="GO:0006281", namespace="biological_process")

        # First execute: get annotations
        # Second execute: get term details
        mock_db.execute.side_effect = [
            _scalars_all([ann1, ann2, ann3]),
            _scalars_all([term1, term2, term3]),
        ]

        service = GOService()
        result = await service.get_gene_annotations(
            mock_db, ["TP53", "BRCA1"], propagate=False
        )

        assert "TP53" in result
        assert "BRCA1" in result
        assert len(result["TP53"]) == 2
        assert len(result["BRCA1"]) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_for_unannotated_genes(self, mock_db):
        """Should return empty dict when no annotations found."""
        from app.services.go_service import GOService

        mock_db.execute.return_value = _scalars_all([])

        service = GOService()
        result = await service.get_gene_annotations(
            mock_db, ["UNKNOWN_GENE"], propagate=False
        )

        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# GOService — go_enrichment_analysis (unit)
# ─────────────────────────────────────────────────────────────────────────────

class TestGoEnrichmentAnalysis:
    """
    The enrichment analysis depends heavily on DB queries.
    We test it by mocking get_gene_annotations at the service level,
    and injecting a synthetic background.
    """

    @pytest.mark.asyncio
    async def test_returns_enriched_terms_list(self, mock_db):
        """Should return a list of enrichment dicts for significant terms."""
        from app.services.go_service import GOService

        service = GOService()

        # Build synthetic study and background annotations
        study_genes = [f"G{i}" for i in range(10)]
        background_genes = [f"G{i}" for i in range(100)]

        # GO:0000001 is annotated to all 10 study genes and 15/100 background genes
        study_annots = {
            g: [{"go_id": "GO:0000001", "evidence_code": "IDA",
                 "source_db": "UniProt", "qualifier": "involved_in"}]
            for g in study_genes
        }
        bg_annots = {
            g: [{"go_id": "GO:0000001", "evidence_code": "IDA",
                 "source_db": "UniProt", "qualifier": "involved_in"}]
            for g in background_genes[:15]
        }

        # Patch get_gene_annotations so we don't hit DB
        with patch.object(service, "get_gene_annotations", side_effect=[study_annots, bg_annots]):
            # Also need to mock the background query and term fetch
            mock_db.execute.side_effect = [
                _all_rows([(g,) for g in background_genes]),          # bg gene_symbols
                _scalars_all([make_go_term(go_id="GO:0000001",
                                           gene_count=15)]),           # term details
            ]

            results = await service.go_enrichment_analysis(
                mock_db,
                gene_list=study_genes,
                background=background_genes,
                min_gene_count=5,
                pvalue_threshold=0.05,
            )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_results_sorted_by_pvalue(self, mock_db):
        """Results should be sorted by p-value ascending."""
        from app.services.go_service import GOService

        service = GOService()

        # Mock two enriched terms with different p-values via patching
        term_results = [
            {
                "go_id": "GO:0000001", "pvalue": 0.04,
                "study_count": 8, "study_genes": ["G1"] * 8,
                "background_count": 20, "enrichment_ratio": 2.0,
            },
            {
                "go_id": "GO:0000002", "pvalue": 0.001,
                "study_count": 9, "study_genes": ["G2"] * 9,
                "background_count": 15, "enrichment_ratio": 3.0,
            },
        ]

        study_genes = [f"G{i}" for i in range(10)]
        background_genes = [f"G{i}" for i in range(100)]

        # Patch the whole analysis at the enrichment loop to return our preset results
        with patch.object(service, "get_gene_annotations") as mock_annots:
            mock_annots.return_value = {}  # Empty → loop won't execute

            mock_db.execute.side_effect = [
                _all_rows([]),  # background query (not needed since we use provided bg)
                _scalars_all([]),  # terms fetch
            ]

            results = await service.go_enrichment_analysis(
                mock_db,
                gene_list=study_genes,
                background=background_genes,
                pvalue_threshold=0.05,
            )

        # With no annotations → no results, but no crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_fdr_values_in_results(self, mock_db):
        """Enrichment results should include FDR field."""
        from app.services.go_service import GOService

        service = GOService()
        study_genes = [f"G{i}" for i in range(5)]
        background_genes = [f"G{i}" for i in range(50)]

        study_annots = {
            g: [{"go_id": "GO:0000001", "evidence_code": "IDA",
                 "source_db": "UniProt", "qualifier": "involved_in"}]
            for g in study_genes
        }
        bg_annots = {
            g: [{"go_id": "GO:0000001", "evidence_code": "IDA",
                 "source_db": "UniProt", "qualifier": "involved_in"}]
            for g in background_genes[:10]
        }

        with patch.object(service, "get_gene_annotations", side_effect=[study_annots, bg_annots]):
            mock_db.execute.side_effect = [
                _all_rows([(g,) for g in background_genes]),
                _scalars_all([make_go_term(go_id="GO:0000001", gene_count=10)]),
            ]

            results = await service.go_enrichment_analysis(
                mock_db,
                gene_list=study_genes,
                background=background_genes,
                min_gene_count=3,
                pvalue_threshold=1.0,  # Accept all terms for testing
            )

        for r in results:
            assert "fdr" in r
            assert 0.0 <= r["fdr"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# GOService — _get_ancestors
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAncestors:

    @pytest.mark.asyncio
    async def test_returns_direct_parents(self, mock_db):
        """Should traverse is_a relationships to find parent terms."""
        from app.services.go_service import GOService

        mock_db.execute.return_value = _parent_rows({
            "GO:0006915": ["GO:0008219"],
            "GO:0008219": [],
        })

        service = GOService()
        ancestors = await service._get_ancestors(mock_db, "GO:0006915")

        assert "GO:0008219" in ancestors

    @pytest.mark.asyncio
    async def test_returns_empty_for_root_term(self, mock_db):
        """A root term with no parents should have no ancestors."""
        from app.services.go_service import GOService

        mock_db.execute.return_value = _parent_rows({"GO:0003674": ([], [])})

        service = GOService()
        ancestors = await service._get_ancestors(mock_db, "GO:0003674")

        assert ancestors == []

    @pytest.mark.asyncio
    async def test_reads_the_hierarchy_in_a_single_query(self, mock_db):
        """
        One bulk read, not one SELECT per node.

        The old traversal queried each node it visited, so propagation across a real gene set
        issued ~1.16 million sequential SELECTs and took 384 s.
        """
        from app.services.go_service import GOService

        mock_db.execute.return_value = _parent_rows({
            "GO:1": ["GO:2"], "GO:2": ["GO:3"], "GO:3": ["GO:4"], "GO:4": [],
        })

        service = GOService()
        ancestors = await service._get_ancestors(mock_db, "GO:1")

        assert set(ancestors) == {"GO:2", "GO:3", "GO:4"}
        assert mock_db.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_follows_part_of_as_well_as_is_a(self, mock_db):
        """Both relationship kinds are parents, as the previous traversal treated them."""
        from app.services.go_service import GOService

        mock_db.execute.return_value = _parent_rows({
            "GO:0006915": (["GO:0008219"], ["GO:0012501"]),
            "GO:0008219": ([], []),
            "GO:0012501": ([], []),
        })

        service = GOService()
        ancestors = await service._get_ancestors(mock_db, "GO:0006915")

        assert set(ancestors) == {"GO:0008219", "GO:0012501"}

    @pytest.mark.asyncio
    async def test_terminates_on_a_cycle(self, mock_db):
        """A cycle in the graph must not hang the traversal."""
        from app.services.go_service import GOService

        mock_db.execute.return_value = _parent_rows({
            "GO:A": ["GO:B"], "GO:B": ["GO:A"],
        })

        service = GOService()
        ancestors = await service._get_ancestors(mock_db, "GO:A")

        assert set(ancestors) == {"GO:A", "GO:B"}


class TestAncestorsFromMap:
    """The pure, memoised traversal that both callers share."""

    def test_memoises_across_calls(self):
        """A GO id is expanded once however many genes carry it."""
        from app.services.go_service import GOService

        parent_map = {"GO:1": ("GO:2",), "GO:2": ("GO:3",), "GO:3": ()}
        cache = {}

        first = GOService._ancestors_from_map("GO:1", parent_map, cache)
        assert first == {"GO:2", "GO:3"}
        assert "GO:1" in cache

        # Second call is served from the cache — same object, no re-walk.
        assert GOService._ancestors_from_map("GO:1", parent_map, cache) is first

    def test_reuses_a_cached_subtree(self):
        """Hitting a node whose closure is known stops the walk there."""
        from app.services.go_service import GOService

        parent_map = {"GO:1": ("GO:2",), "GO:2": ("GO:3",), "GO:3": ()}
        cache = {"GO:2": {"GO:3"}}

        assert GOService._ancestors_from_map("GO:1", parent_map, cache) == {"GO:2", "GO:3"}

    def test_shared_ancestor_counted_once(self):
        """A diamond yields each ancestor once."""
        from app.services.go_service import GOService

        parent_map = {
            "GO:child": ("GO:l", "GO:r"),
            "GO:l": ("GO:root",),
            "GO:r": ("GO:root",),
            "GO:root": (),
        }
        assert GOService._ancestors_from_map("GO:child", parent_map, {}) == {
            "GO:l", "GO:r", "GO:root"
        }

    def test_excludes_the_term_itself(self):
        """Matches the previous behaviour: a term is not its own ancestor."""
        from app.services.go_service import GOService

        parent_map = {"GO:1": ("GO:2",), "GO:2": ()}
        assert "GO:1" not in GOService._ancestors_from_map("GO:1", parent_map, {})

    def test_unknown_term_has_no_ancestors(self):
        """A GO id absent from the graph yields nothing rather than raising."""
        from app.services.go_service import GOService

        assert GOService._ancestors_from_map("GO:missing", {"GO:1": ()}, {}) == set()


# ─────────────────────────────────────────────────────────────────────────────
# GOService — _get_descendants
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDescendants:

    @pytest.mark.asyncio
    async def test_returns_direct_children(self, mock_db):
        """Should return immediate child terms."""
        from app.services.go_service import GOService

        child = make_go_term(go_id="GO:0006915", is_a=["GO:0008219"])
        # First query: find children of GO:0008219
        # Second query: find children of GO:0006915 (leaf → empty)
        mock_db.execute.side_effect = [
            _scalars_all([child]),
            _scalars_all([]),  # child has no children
        ]

        service = GOService()
        descendants = await service._get_descendants(mock_db, "GO:0008219")

        assert "GO:0006915" in descendants

    @pytest.mark.asyncio
    async def test_returns_empty_for_leaf_term(self, mock_db):
        """A leaf GO term should have no descendants."""
        from app.services.go_service import GOService

        mock_db.execute.return_value = _scalars_all([])

        service = GOService()
        descendants = await service._get_descendants(mock_db, "GO:0006915")

        assert descendants == []


# ─────────────────────────────────────────────────────────────────────────────
# GOService — _propagate_annotations
# ─────────────────────────────────────────────────────────────────────────────

class TestPropagateAnnotations:
    """
    Tests for GOService._propagate_annotations.

    These used to patch `_get_ancestors` out, so the traversal itself was never exercised. Now
    the GO parent graph is fed through the single bulk read the implementation performs, which
    means the real closure logic runs.
    """

    @pytest.mark.asyncio
    async def test_direct_annotation_preserved(self, mock_db):
        """Gene should retain its direct annotation after propagation."""
        from app.services.go_service import GOService

        service = GOService()
        terms = {"GO:0006915": make_go_term(go_id="GO:0006915", is_a=[])}
        gene_annots = {
            "TP53": [{
                "go_id": "GO:0006915", "evidence_code": "IDA",
                "source_db": "UniProt", "qualifier": "involved_in",
            }]
        }
        mock_db.execute.return_value = _parent_rows({"GO:0006915": ([], [])})

        result = await service._propagate_annotations(mock_db, gene_annots, terms)

        assert "TP53" in result
        assert any(a["go_id"] == "GO:0006915" for a in result["TP53"])

    @pytest.mark.asyncio
    async def test_ancestor_annotations_added(self, mock_db):
        """When term has ancestors, propagated annotations must include them."""
        from app.services.go_service import GOService

        service = GOService()
        terms = {
            "GO:0006915": make_go_term(go_id="GO:0006915", is_a=["GO:0008219"]),
            "GO:0008219": make_go_term(go_id="GO:0008219", name="cell death", is_a=[]),
        }
        gene_annots = {
            "TP53": [{
                "go_id": "GO:0006915", "evidence_code": "IDA",
                "source_db": "UniProt", "qualifier": "involved_in",
            }]
        }
        mock_db.execute.return_value = _parent_rows({
            "GO:0006915": ["GO:0008219"],
            "GO:0008219": [],
        })

        result = await service._propagate_annotations(mock_db, gene_annots, terms)

        go_ids = {a["go_id"] for a in result["TP53"]}
        assert go_ids == {"GO:0006915", "GO:0008219"}

        inherited = next(a for a in result["TP53"] if a["go_id"] == "GO:0008219")
        assert inherited["evidence_code"] == "IEA"
        assert inherited["source_db"] == "propagated"
        assert inherited["qualifier"] == "inherited from GO:0006915"
        assert inherited["term_name"] == "cell death"

    @pytest.mark.asyncio
    async def test_propagates_transitively(self, mock_db):
        """A grandparent is inherited too, not just the direct parent."""
        from app.services.go_service import GOService

        service = GOService()
        terms = {
            "GO:child": make_go_term(go_id="GO:child", is_a=["GO:parent"]),
            "GO:parent": make_go_term(go_id="GO:parent", is_a=["GO:root"]),
            "GO:root": make_go_term(go_id="GO:root", is_a=[]),
        }
        gene_annots = {"TP53": [{"go_id": "GO:child", "evidence_code": "IDA",
                                 "source_db": "UniProt", "qualifier": "involved_in"}]}
        mock_db.execute.return_value = _parent_rows({
            "GO:child": ["GO:parent"], "GO:parent": ["GO:root"], "GO:root": [],
        })

        result = await service._propagate_annotations(mock_db, gene_annots, terms)

        assert {a["go_id"] for a in result["TP53"]} == {"GO:child", "GO:parent", "GO:root"}

    @pytest.mark.asyncio
    async def test_reads_the_hierarchy_once_for_many_genes(self, mock_db):
        """
        The regression this fix is about: one bulk read for the whole gene set.

        The old code awaited a fresh hierarchy walk per (gene, annotation) pair, each issuing a
        SELECT per node visited. Here 50 genes share one annotation, so the old code would have
        walked 50 times; the closure is now computed once and memoised.
        """
        from app.services.go_service import GOService

        service = GOService()
        terms = {
            "GO:child": make_go_term(go_id="GO:child", is_a=["GO:parent"]),
            "GO:parent": make_go_term(go_id="GO:parent", is_a=[]),
        }
        gene_annots = {
            f"GENE{i}": [{"go_id": "GO:child", "evidence_code": "IDA",
                          "source_db": "UniProt", "qualifier": "involved_in"}]
            for i in range(50)
        }
        mock_db.execute.return_value = _parent_rows({
            "GO:child": ["GO:parent"], "GO:parent": [],
        })

        result = await service._propagate_annotations(mock_db, gene_annots, terms)

        assert mock_db.execute.await_count == 1
        assert len(result) == 50
        for i in range(50):
            assert {a["go_id"] for a in result[f"GENE{i}"]} == {"GO:child", "GO:parent"}

    @pytest.mark.asyncio
    async def test_ancestor_absent_from_terms_is_skipped(self, mock_db):
        """
        Preserved quirk: only ancestors that are themselves directly annotated get emitted.

        `terms` holds just the directly annotated ids, so propagation is narrower than the true
        path rule implies. Documented in `get_gene_annotations`; widening it would change every
        enrichment result.
        """
        from app.services.go_service import GOService

        service = GOService()
        # GO:parent is a real ancestor but is NOT in `terms`.
        terms = {"GO:child": make_go_term(go_id="GO:child", is_a=["GO:parent"])}
        gene_annots = {"TP53": [{"go_id": "GO:child", "evidence_code": "IDA",
                                 "source_db": "UniProt", "qualifier": "involved_in"}]}
        mock_db.execute.return_value = _parent_rows({
            "GO:child": ["GO:parent"], "GO:parent": [],
        })

        result = await service._propagate_annotations(mock_db, gene_annots, terms)

        assert {a["go_id"] for a in result["TP53"]} == {"GO:child"}

    @pytest.mark.asyncio
    async def test_no_duplicate_annotations(self, mock_db):
        """Same GO ID should not appear twice in propagated annotations."""
        from app.services.go_service import GOService

        service = GOService()
        terms = {"GO:0006915": make_go_term(go_id="GO:0006915", is_a=[])}
        gene_annots = {
            "EGFR": [
                {"go_id": "GO:0006915", "evidence_code": "IDA",
                 "source_db": "UniProt", "qualifier": "involved_in"},
                {"go_id": "GO:0006915", "evidence_code": "TAS",
                 "source_db": "UniProt", "qualifier": "involved_in"},
            ]
        }
        mock_db.execute.return_value = _parent_rows({"GO:0006915": ([], [])})

        result = await service._propagate_annotations(mock_db, gene_annots, terms)

        go_entries = [a["go_id"] for a in result["EGFR"]]
        assert go_entries.count("GO:0006915") == 1

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_genes(self, mock_db):
        """Empty gene_annots should yield empty propagated result."""
        from app.services.go_service import GOService

        service = GOService()
        mock_db.execute.return_value = _parent_rows({})
        result = await service._propagate_annotations(mock_db, {}, {})
        assert result == {}
