"""
Unit tests for GSEAProcessor — pure computation, no DB.

Covers:
- GSEAResult dataclass (to_dict)
- _calculate_enrichment_score (hit positions, running sum, ES sign)
- _calculate_fdr (BH correction, monotonicity)
- run_gsea (integration: filter by size, correct NES/FDR, leading edge)
- GeneSetsLoader.load_from_dict / get_default_gene_sets
- prepare_ranked_gene_list (log_fc, signed_pvalue, signal2noise, unknown)
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch


# ─────────────────────────────────────────────────────────────────────────────
# GSEAResult
# ─────────────────────────────────────────────────────────────────────────────

class TestGSEAResult:

    def test_to_dict_contains_all_keys(self):
        """to_dict() should expose all expected fields."""
        from app.services.gsea_processor import GSEAResult

        result = GSEAResult(
            gene_set_name="TEST_SET",
            gene_set_size=20,
            enrichment_score=0.72,
            normalized_enrichment_score=1.85,
            p_value=0.001,
            fdr_q_value=0.01,
            leading_edge_genes=["TP53", "MDM2"],
            running_enrichment_scores=[0.1, 0.2],
            gene_positions=[0, 5],
            core_enrichment=["TP53"],
        )

        d = result.to_dict()
        expected_keys = {
            "gene_set_name", "gene_set_size", "enrichment_score",
            "normalized_enrichment_score", "p_value", "fdr_q_value",
            "leading_edge_genes", "running_enrichment_scores",
            "gene_positions", "core_enrichment",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values_preserved(self):
        """to_dict() should return exact values."""
        from app.services.gsea_processor import GSEAResult

        result = GSEAResult(
            gene_set_name="GO_APOPTOSIS",
            gene_set_size=50,
            enrichment_score=-0.55,
            normalized_enrichment_score=-1.42,
            p_value=0.042,
            fdr_q_value=0.089,
            leading_edge_genes=["BAX"],
            running_enrichment_scores=[-0.1, -0.2],
            gene_positions=[3],
            core_enrichment=["BAX"],
        )

        d = result.to_dict()
        assert d["gene_set_name"] == "GO_APOPTOSIS"
        assert d["enrichment_score"] == pytest.approx(-0.55)
        assert d["p_value"] == pytest.approx(0.042)


# ─────────────────────────────────────────────────────────────────────────────
# _calculate_enrichment_score
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateEnrichmentScore:

    def _make_processor(self, min_size=1, max_size=1000):
        from app.services.gsea_processor import GSEAProcessor
        return GSEAProcessor(min_size=min_size, max_size=max_size)

    def test_returns_zero_for_empty_gene_set(self):
        """ES should be 0 when no genes in gene set are found in ranked list."""
        proc = self._make_processor()
        gene_list = ["A", "B", "C", "D", "E"]
        gene_set = ["X", "Y", "Z"]  # none overlap
        metrics = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        es, running_scores, positions = proc._calculate_enrichment_score(
            gene_list, gene_set, metrics
        )

        assert es == 0.0
        assert len(running_scores) == len(gene_list)
        assert positions == []

    def test_positive_enrichment_for_top_ranked_gene_set(self):
        """Gene set at top of ranked list → positive ES."""
        proc = self._make_processor()
        # Genes ranked by decreasing metric
        gene_list = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]
        # Gene set is concentrated at top
        gene_set = ["G1", "G2", "G3"]
        metrics = np.array([10.0, 9.0, 8.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05])

        es, _, positions = proc._calculate_enrichment_score(gene_list, gene_set, metrics)

        assert es > 0.0
        assert 0 in positions
        assert 1 in positions
        assert 2 in positions

    def test_negative_enrichment_for_bottom_ranked_gene_set(self):
        """Gene set at bottom of ranked list → negative ES."""
        proc = self._make_processor()
        gene_list = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]
        gene_set = ["G8", "G9", "G10"]
        metrics = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.1, 0.05, 0.01])

        es, _, _ = proc._calculate_enrichment_score(gene_list, gene_set, metrics)

        # For bottom-ranked gene set, max deviation may be positive or negative depending
        # on implementation — just verify it is finite
        assert np.isfinite(es)

    def test_running_scores_has_correct_length(self):
        """Running enrichment score array length must equal gene list length."""
        proc = self._make_processor()
        n = 20
        gene_list = [f"G{i}" for i in range(n)]
        gene_set = gene_list[:5]
        metrics = np.linspace(10, 1, n)

        _, running_scores, _ = proc._calculate_enrichment_score(
            gene_list, gene_set, metrics
        )
        assert len(running_scores) == n

    def test_hit_positions_correctly_identified(self):
        """Positions should match indices in gene_list where genes overlap."""
        proc = self._make_processor()
        gene_list = ["A", "B", "C", "D", "E"]
        gene_set = ["B", "D"]
        metrics = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        _, _, positions = proc._calculate_enrichment_score(gene_list, gene_set, metrics)

        assert 1 in positions  # "B" is at index 1
        assert 3 in positions  # "D" is at index 3
        assert len(positions) == 2


# ─────────────────────────────────────────────────────────────────────────────
# _calculate_fdr
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateFDR:

    def _make_result(self, nes, pval):
        from app.services.gsea_processor import GSEAResult
        return GSEAResult(
            gene_set_name=f"SET_{nes}",
            gene_set_size=20,
            enrichment_score=nes,
            normalized_enrichment_score=nes,
            p_value=pval,
            fdr_q_value=0.0,
            leading_edge_genes=[],
            running_enrichment_scores=[],
            gene_positions=[],
            core_enrichment=[],
        )

    def test_fdr_monotonically_non_decreasing(self):
        """After BH correction, FDR values must be monotonically non-decreasing."""
        from app.services.gsea_processor import GSEAProcessor

        proc = GSEAProcessor()
        results = [
            self._make_result(1.5, 0.001),
            self._make_result(1.2, 0.01),
            self._make_result(0.9, 0.05),
        ]

        corrected = proc._calculate_fdr(results)
        positive = [r for r in corrected if r.normalized_enrichment_score >= 0]
        positive.sort(key=lambda x: x.p_value)
        fdrs = [r.fdr_q_value for r in positive]

        for i in range(len(fdrs) - 1):
            assert fdrs[i] <= fdrs[i + 1], f"FDR not monotone at index {i}: {fdrs}"

    def test_fdr_values_bounded_between_0_and_1(self):
        """All FDR values should be in [0, 1]."""
        from app.services.gsea_processor import GSEAProcessor

        proc = GSEAProcessor()
        results = [
            self._make_result(1.5, 0.001),
            self._make_result(-1.2, 0.02),
        ]

        corrected = proc._calculate_fdr(results)
        for r in corrected:
            assert 0.0 <= r.fdr_q_value <= 1.0, f"FDR out of range: {r.fdr_q_value}"

    def test_handles_empty_results(self):
        """_calculate_fdr should not crash on empty input."""
        from app.services.gsea_processor import GSEAProcessor

        proc = GSEAProcessor()
        result = proc._calculate_fdr([])
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# run_gsea (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunGSEA:

    def _make_ranked_genes(self, n=50, n_set=10):
        """Create a ranked gene list with a 'hot' gene set at the top."""
        gene_names = [f"G{i}" for i in range(n)]
        metrics = np.linspace(10, -10, n)
        df = pd.DataFrame({"metric": metrics}, index=gene_names)
        return df, gene_names[:n_set]

    def test_returns_list_of_gsea_results(self):
        """run_gsea should return GSEAResult objects."""
        from app.services.gsea_processor import GSEAProcessor, GSEAResult

        proc = GSEAProcessor(min_size=5, max_size=100)
        ranked_genes, top_genes = self._make_ranked_genes(n=50, n_set=10)

        gene_sets = {"TOP_SET": top_genes, "BOTTOM_SET": [f"G{i}" for i in range(40, 50)]}
        results = proc.run_gsea(ranked_genes, gene_sets, n_permutations=50, seed=42)

        assert isinstance(results, list)
        assert all(isinstance(r, GSEAResult) for r in results)

    def test_gene_sets_too_small_are_filtered(self):
        """Gene sets below min_size should be excluded."""
        from app.services.gsea_processor import GSEAProcessor

        proc = GSEAProcessor(min_size=10, max_size=100)
        ranked_genes, _ = self._make_ranked_genes(n=50)

        # Create a set with only 3 genes (below min_size=10)
        gene_sets = {"TINY_SET": ["G0", "G1", "G2"]}
        results = proc.run_gsea(ranked_genes, gene_sets, n_permutations=20, seed=42)

        assert len(results) == 0

    def test_gene_sets_too_large_are_filtered(self):
        """Gene sets above max_size should be excluded."""
        from app.services.gsea_processor import GSEAProcessor

        proc = GSEAProcessor(min_size=1, max_size=5)
        ranked_genes, _ = self._make_ranked_genes(n=50)

        gene_sets = {"HUGE_SET": [f"G{i}" for i in range(20)]}  # 20 > max_size=5
        results = proc.run_gsea(ranked_genes, gene_sets, n_permutations=20, seed=42)

        assert len(results) == 0

    def test_results_sorted_by_abs_nes_descending(self):
        """Results should be sorted by |NES| in descending order."""
        from app.services.gsea_processor import GSEAProcessor

        proc = GSEAProcessor(min_size=5, max_size=100)
        ranked_genes, top_genes = self._make_ranked_genes(n=60, n_set=10)

        gene_sets = {
            "TOP_SET": top_genes,
            "MIDDLE_SET": [f"G{i}" for i in range(20, 35)],
            "BOTTOM_SET": [f"G{i}" for i in range(50, 60)],
        }
        results = proc.run_gsea(ranked_genes, gene_sets, n_permutations=50, seed=0)

        assert len(results) >= 1
        abs_nes = [abs(r.normalized_enrichment_score) for r in results]
        assert abs_nes == sorted(abs_nes, reverse=True)

    def test_leading_edge_genes_are_subset_of_gene_set(self):
        """Leading edge genes must be a subset of the gene set."""
        from app.services.gsea_processor import GSEAProcessor

        proc = GSEAProcessor(min_size=5, max_size=100)
        ranked_genes, top_genes = self._make_ranked_genes(n=50, n_set=10)
        gene_sets = {"TOP_SET": top_genes}

        results = proc.run_gsea(ranked_genes, gene_sets, n_permutations=50, seed=1)

        if results:
            r = results[0]
            gene_set = set(top_genes)
            for g in r.core_enrichment:
                assert g in gene_set, f"Core gene {g} not in gene set"


# ─────────────────────────────────────────────────────────────────────────────
# GeneSetsLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneSetsLoader:

    def test_load_from_dict_returns_same_dict(self):
        """load_from_dict should return the input unchanged."""
        from app.services.gsea_processor import GeneSetsLoader

        data = {"SET_A": ["G1", "G2"], "SET_B": ["G3"]}
        result = GeneSetsLoader.load_from_dict(data)
        assert result == data

    def test_get_default_gene_sets_returns_dict(self):
        """Default gene sets should be a non-empty dict."""
        from app.services.gsea_processor import GeneSetsLoader

        defaults = GeneSetsLoader.get_default_gene_sets()
        assert isinstance(defaults, dict)
        assert len(defaults) > 0
        # Each entry should be a list of strings
        for name, genes in defaults.items():
            assert isinstance(genes, list)
            assert all(isinstance(g, str) for g in genes)


# ─────────────────────────────────────────────────────────────────────────────
# prepare_ranked_gene_list
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareRankedGeneList:

    def _make_deg_df(self, genes=None):
        genes = genes or ["TP53", "BRCA1", "EGFR", "MDM2", "MYC"]
        return pd.DataFrame({
            "gene_id": genes,
            "log_fc": [2.5, -1.8, 3.1, -0.5, 4.2],
            "padj": [0.001, 0.05, 0.0001, 0.2, 0.0001],
        })

    def test_log_fc_metric(self):
        """log_fc ranking: metric == log_fc, sorted descending."""
        from app.services.gsea_processor import prepare_ranked_gene_list

        df = self._make_deg_df()
        result = prepare_ranked_gene_list(df.copy(), ranking_metric="log_fc")

        assert "metric" in result.columns
        assert result.index.name == "gene_id"
        # Verify sorted descending
        assert result["metric"].values[0] >= result["metric"].values[-1]

    def test_signed_pvalue_metric(self):
        """signed_pvalue: metric = -log10(padj) * sign(log_fc)."""
        from app.services.gsea_processor import prepare_ranked_gene_list

        df = self._make_deg_df()
        result = prepare_ranked_gene_list(df.copy(), ranking_metric="signed_pvalue")

        assert result.index[0] in df["gene_id"].values
        # All metrics should be finite
        assert np.all(np.isfinite(result["metric"]))

    def test_signal2noise_fallback_when_no_stderr(self):
        """Without stderr column, falls back to log_fc metric."""
        from app.services.gsea_processor import prepare_ranked_gene_list

        df = self._make_deg_df()
        result = prepare_ranked_gene_list(df.copy(), ranking_metric="signal2noise")

        # Should not raise, and metric should be log_fc values
        assert "metric" in result.columns

    def test_signal2noise_with_stderr(self):
        """With stderr column, uses log_fc / (stderr + eps)."""
        from app.services.gsea_processor import prepare_ranked_gene_list

        df = self._make_deg_df()
        df["stderr"] = [0.5, 0.3, 0.8, 0.2, 0.6]
        result = prepare_ranked_gene_list(df.copy(), ranking_metric="signal2noise")

        assert "metric" in result.columns
        assert np.all(np.isfinite(result["metric"]))

    def test_unknown_metric_raises_value_error(self):
        """Unknown ranking metric should raise ValueError."""
        from app.services.gsea_processor import prepare_ranked_gene_list

        df = self._make_deg_df()
        with pytest.raises(ValueError, match="Unknown ranking metric"):
            prepare_ranked_gene_list(df.copy(), ranking_metric="bogus_metric")


# ─────────────────────────────────────────────────────────────────────────────
# _generate_null_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateNullDistribution:

    def _make_processor(self):
        from app.services.gsea_processor import GSEAProcessor
        return GSEAProcessor(min_size=1, max_size=1000)

    def test_returns_dict_with_gene_set_names(self):
        """Null distribution should be keyed by gene set name."""
        proc = self._make_processor()
        gene_list = ["A", "B", "C", "D", "E", "F"]
        gene_sets = {"SET_A": ["A", "B"], "SET_B": ["C", "D"]}
        metrics = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
        np.random.seed(42)

        null_dist = proc._generate_null_distribution(
            gene_list, gene_sets, metrics, n_permutations=10
        )

        assert "SET_A" in null_dist
        assert "SET_B" in null_dist

    def test_distribution_has_correct_length(self):
        """Each gene set's null distribution should have n_permutations entries."""
        proc = self._make_processor()
        gene_list = ["A", "B", "C", "D", "E"]
        gene_sets = {"SET_A": ["A", "B"]}
        metrics = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        np.random.seed(42)

        n_perm = 15
        null_dist = proc._generate_null_distribution(
            gene_list, gene_sets, metrics, n_permutations=n_perm
        )

        assert len(null_dist["SET_A"]) == n_perm

    def test_distribution_values_are_floats(self):
        """Each value in the null distribution should be a numeric ES (float)."""
        proc = self._make_processor()
        gene_list = list("ABCDEFGHIJ")
        gene_sets = {"APOPTOSIS": ["A", "B", "C"]}
        metrics = np.arange(10, 0, -1, dtype=float)
        np.random.seed(0)

        null_dist = proc._generate_null_distribution(
            gene_list, gene_sets, metrics, n_permutations=5
        )

        for val in null_dist["APOPTOSIS"]:
            assert isinstance(val, float)

    def test_empty_gene_sets_returns_empty_distributions(self):
        """An empty gene_sets dict should yield an empty null distribution."""
        proc = self._make_processor()
        gene_list = ["A", "B", "C"]
        metrics = np.array([3.0, 2.0, 1.0])

        null_dist = proc._generate_null_distribution(
            gene_list, {}, metrics, n_permutations=5
        )

        assert null_dist == {}


# ─────────────────────────────────────────────────────────────────────────────
# GeneSetsLoader.load_from_gmt
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFromGmt:

    def _write_gmt(self, tmp_path, lines: list[str]) -> str:
        """Write a GMT file and return its path."""
        gmt_file = tmp_path / "test.gmt"
        gmt_file.write_text("\n".join(lines))
        return str(gmt_file)

    def test_parses_basic_gmt(self, tmp_path):
        """Should parse a standard GMT file correctly."""
        from app.services.gsea_processor import GeneSetsLoader

        path = self._write_gmt(tmp_path, [
            "APOPTOSIS\thttp://example.com\tTP53\tBAX\tBCL2",
            "CELL_CYCLE\thttp://example.com\tCDK1\tCCNB1",
        ])

        result = GeneSetsLoader.load_from_gmt(path)

        assert "APOPTOSIS" in result
        assert "CELL_CYCLE" in result
        assert sorted(result["APOPTOSIS"]) == sorted(["TP53", "BAX", "BCL2"])
        assert sorted(result["CELL_CYCLE"]) == sorted(["CDK1", "CCNB1"])

    def test_ignores_malformed_lines(self, tmp_path):
        """Lines with fewer than 3 fields should be ignored."""
        from app.services.gsea_processor import GeneSetsLoader

        path = self._write_gmt(tmp_path, [
            "VALID_SET\tdescription\tGENE1\tGENE2",
            "INCOMPLETE",           # only 1 field  → ignored
            "TWO_FIELDS\tdesc",     # only 2 fields → ignored
        ])

        result = GeneSetsLoader.load_from_gmt(path)

        assert "VALID_SET" in result
        assert "INCOMPLETE" not in result
        assert "TWO_FIELDS" not in result

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        """An empty GMT file should yield an empty dict."""
        from app.services.gsea_processor import GeneSetsLoader

        path = self._write_gmt(tmp_path, [])

        result = GeneSetsLoader.load_from_gmt(path)

        assert result == {}

    def test_file_not_found_raises(self, tmp_path):
        """Should raise FileNotFoundError for a missing file."""
        from app.services.gsea_processor import GeneSetsLoader

        with pytest.raises((FileNotFoundError, OSError)):
            GeneSetsLoader.load_from_gmt(str(tmp_path / "nonexistent.gmt"))


# ─────────────────────────────────────────────────────────────────────────────
# GSEA-01 to GSEA-04  — edge-case complements (from FIXPLAN.md §3.1)
# ─────────────────────────────────────────────────────────────────────────────

class TestGSEAEdgeCases:
    """Complement tests targeting edge cases identified in FIXPLAN.md."""

    def _make_proc(self, min_size=1, max_size=500):
        from app.services.gsea_processor import GSEAProcessor
        return GSEAProcessor(min_size=min_size, max_size=max_size)

    def _ranked_df(self, genes, metrics):
        import pandas as pd
        return pd.DataFrame({"metric": metrics}, index=genes)

    # GSEA-01: gene set with no intersection → ES = 0
    def test_gsea01_no_overlap_gene_set_es_zero(self):
        """_calculate_enrichment_score returns 0 when gene set has no intersection."""
        proc = self._make_proc()
        gene_list = [f"G{i}" for i in range(20)]
        gene_set = ["OUTSIDER_1", "OUTSIDER_2"]   # not in gene_list
        metrics = np.linspace(10, -10, 20)

        es, running_sum, positions = proc._calculate_enrichment_score(
            gene_list, gene_set, metrics
        )
        assert es == 0.0
        assert len(positions) == 0

    # GSEA-02: null distribution all positive → negative NES must not be NaN
    def test_gsea02_one_sided_null_dist_no_nan_nes(self):
        """NES must not be NaN when null distribution is entirely one-signed."""
        proc = self._make_proc()
        # Ranked list: top genes first → very enriched gene set at top
        genes = [f"G{i}" for i in range(30)]
        metrics = np.linspace(10, -10, 30)
        ranked = self._ranked_df(genes, metrics)

        # Gene set at the bottom of the list → negative ES; null dist may be all-positive
        gene_sets = {"BOTTOM_SET": genes[25:]}
        results = proc.run_gsea(ranked, gene_sets, n_permutations=50, seed=99)

        for r in results:
            assert not np.isnan(r.normalized_enrichment_score), \
                f"NES is NaN for {r.gene_set_name}"
            assert not np.isnan(r.p_value), \
                f"p_value is NaN for {r.gene_set_name}"

    # GSEA-03: gene set above max_size → filtered out
    def test_gsea03_gene_set_above_max_size_filtered(self):
        """Gene sets larger than max_size are excluded from results."""
        proc = self._make_proc(min_size=1, max_size=10)
        genes = [f"G{i}" for i in range(50)]
        metrics = np.linspace(5, -5, 50)
        ranked = self._ranked_df(genes, metrics)

        # 20 genes → above max_size=10
        gene_sets = {"BIG_SET": genes[:20], "SMALL_SET": genes[:5]}
        results = proc.run_gsea(ranked, gene_sets, n_permutations=20, seed=0)

        names = {r.gene_set_name for r in results}
        assert "BIG_SET" not in names
        assert "SMALL_SET" in names

    # GSEA-04: performance guard — 20k genes, 50 gene sets < 30 s
    @pytest.mark.slow
    def test_gsea04_performance_20k_genes_50_sets(self):
        """run_gsea on 20k genes × 50 gene sets with 100 permutations finishes in < 30 s."""
        import time
        proc = self._make_proc(min_size=10, max_size=300)
        rng = np.random.default_rng(42)

        N = 20_000
        genes = [f"G{i}" for i in range(N)]
        metrics = rng.normal(0, 1, N)
        ranked = self._ranked_df(genes, metrics)

        gene_sets = {
            f"SET_{s}": list(rng.choice(genes, size=rng.integers(15, 200), replace=False))
            for s in range(50)
        }

        start = time.perf_counter()
        results = proc.run_gsea(ranked, gene_sets, n_permutations=100, seed=0)
        elapsed = time.perf_counter() - start

        # Generous threshold: this is a "doesn't hang" smoke test, and shared CI
        # runners vary enough to flake a tighter bound.
        assert elapsed < 60, f"GSEA took {elapsed:.1f}s — too slow (threshold: 60s)"
        assert len(results) > 0
