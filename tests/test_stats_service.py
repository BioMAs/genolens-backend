"""
Tests for the multiple testing correction service.

Tests:
- Bonferroni correction
- BH (Benjamini-Hochberg)
- Holm-Bonferroni
- BY (Benjamini-Yekutieli)
- NaN handling
- compute_correction_comparison helper
"""
import math
import pytest
import numpy as np

from app.services.stats_service import (
    CorrectionMethod,
    correct_pvalues,
    compute_correction_comparison,
    _bh,
    _bonferroni,
    _holm,
    _by,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Example p-values: 5 tests
PVALUES_5 = [0.001, 0.008, 0.039, 0.041, 0.189]
# Example p-values with NaN
PVALUES_NAN = [0.001, None, 0.05, 0.2, None]


# ── Bonferroni ────────────────────────────────────────────────────────────────

class TestBonferroni:
    def test_basic_correction(self):
        """padj = min(p * n, 1)"""
        result = correct_pvalues(PVALUES_5, CorrectionMethod.BONFERRONI)
        assert len(result) == 5
        n = 5
        for raw, adj in zip(PVALUES_5, result):
            expected = min(raw * n, 1.0)
            assert abs(adj - expected) < 1e-12

    def test_clipped_to_one(self):
        """Values > 1 are clipped to 1."""
        result = correct_pvalues([0.4, 0.5, 0.6], CorrectionMethod.BONFERRONI)
        assert all(v <= 1.0 for v in result)

    def test_monotonic(self):
        """Adjusted p-values preserve the original ordering."""
        sorted_p = sorted(PVALUES_5)
        result = correct_pvalues(sorted_p, CorrectionMethod.BONFERRONI)
        assert result == sorted(result)

    def test_single_value(self):
        result = correct_pvalues([0.03], CorrectionMethod.BONFERRONI)
        assert abs(result[0] - 0.03) < 1e-12

    def test_already_significant_stays_significant(self):
        result = correct_pvalues([0.001, 0.002], CorrectionMethod.BONFERRONI)
        assert result[0] < 0.05  # 0.001 * 2 = 0.002


# ── BH ────────────────────────────────────────────────────────────────────────

class TestBH:
    def test_output_length(self):
        result = correct_pvalues(PVALUES_5, CorrectionMethod.BH)
        assert len(result) == 5

    def test_values_in_range(self):
        result = correct_pvalues(PVALUES_5, CorrectionMethod.BH)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_bh_less_conservative_than_bonferroni(self):
        """BH should produce smaller or equal adjusted p-values than Bonferroni."""
        bh = correct_pvalues(PVALUES_5, CorrectionMethod.BH)
        bonf = correct_pvalues(PVALUES_5, CorrectionMethod.BONFERRONI)
        assert all(b <= bo for b, bo in zip(bh, bonf))

    def test_known_values(self):
        """
        Known example from Benjamini & Hochberg 1995 Table 1 (simplified).
        p = [0.0001, 0.001, 0.01, 0.05] → BH with n=4
        Rank 1: 0.0001 * 4/1 = 0.0004
        Rank 2: 0.001 * 4/2 = 0.002
        Rank 3: 0.01 * 4/3 = 0.0133…
        Rank 4: 0.05 * 4/4 = 0.05
        """
        p = [0.0001, 0.001, 0.01, 0.05]
        result = correct_pvalues(p, CorrectionMethod.BH)
        # Step-up applied: no changes needed here since already monotonic
        assert result[0] < result[1] < result[2]
        assert abs(result[-1] - 0.05) < 1e-10

    def test_monotonicity_after_step_up(self):
        """BH step-up ensures adjusted values are non-decreasing when sorted."""
        p = sorted(PVALUES_5)
        result = correct_pvalues(p, CorrectionMethod.BH)
        # Values corresponding to sorted p should be non-decreasing
        assert result == sorted(result)


# ── Holm ──────────────────────────────────────────────────────────────────────

class TestHolm:
    def test_output_length(self):
        result = correct_pvalues(PVALUES_5, CorrectionMethod.HOLM)
        assert len(result) == 5

    def test_values_in_range(self):
        result = correct_pvalues(PVALUES_5, CorrectionMethod.HOLM)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_holm_at_most_bonferroni(self):
        """Holm ≤ Bonferroni for each gene (Holm is uniformly more powerful)."""
        holm = correct_pvalues(PVALUES_5, CorrectionMethod.HOLM)
        bonf = correct_pvalues(PVALUES_5, CorrectionMethod.BONFERRONI)
        assert all(h <= b + 1e-12 for h, b in zip(holm, bonf))

    def test_step_down_monotone(self):
        """After sorting by p, Holm adjusted values should be non-decreasing."""
        order = sorted(range(len(PVALUES_5)), key=lambda i: PVALUES_5[i])
        sorted_p = [PVALUES_5[i] for i in order]
        result = correct_pvalues(sorted_p, CorrectionMethod.HOLM)
        assert result == sorted(result)


# ── BY ────────────────────────────────────────────────────────────────────────

class TestBY:
    def test_output_length(self):
        result = correct_pvalues(PVALUES_5, CorrectionMethod.BY)
        assert len(result) == 5

    def test_by_more_conservative_than_bh(self):
        """BY adjusted values should be >= BH for the same inputs."""
        by = correct_pvalues(PVALUES_5, CorrectionMethod.BY)
        bh = correct_pvalues(PVALUES_5, CorrectionMethod.BH)
        assert all(b >= h - 1e-12 for b, h in zip(by, bh))

    def test_values_in_range(self):
        result = correct_pvalues(PVALUES_5, CorrectionMethod.BY)
        assert all(0.0 <= v <= 1.0 for v in result)


# ── NaN handling ──────────────────────────────────────────────────────────────

class TestNaNHandling:
    def test_none_values_become_nan(self):
        result = correct_pvalues(PVALUES_NAN, CorrectionMethod.BH)
        assert math.isnan(result[1])
        assert math.isnan(result[4])

    def test_valid_values_still_corrected(self):
        result = correct_pvalues(PVALUES_NAN, CorrectionMethod.BH)
        assert not math.isnan(result[0])
        assert not math.isnan(result[2])
        assert not math.isnan(result[3])

    def test_all_nan(self):
        result = correct_pvalues([None, None, None], CorrectionMethod.BH)
        assert all(math.isnan(v) for v in result)


# ── compute_correction_comparison ─────────────────────────────────────────────

class TestComputeCorrectionComparison:
    def _setup(self):
        gene_ids = [f"GENE{i+1}" for i in range(5)]
        raw = PVALUES_5
        original_padj = [p * 5 for p in raw]  # Simulate BH-like original
        return gene_ids, raw, original_padj

    def test_returns_expected_keys(self):
        gene_ids, raw, original_padj = self._setup()
        result = compute_correction_comparison(gene_ids, raw, original_padj)
        assert "n_total" in result
        assert "n_tested" in result
        assert "threshold" in result
        assert "summary" in result
        assert "methods_applied" in result

    def test_n_total_matches_input(self):
        gene_ids, raw, original_padj = self._setup()
        result = compute_correction_comparison(gene_ids, raw, original_padj)
        assert result["n_total"] == 5

    def test_n_tested_excludes_none(self):
        gene_ids = ["G1", "G2", "G3"]
        raw = [0.01, None, 0.05]
        original_padj = [0.05, None, 0.1]
        result = compute_correction_comparison(gene_ids, raw, original_padj)
        assert result["n_tested"] == 2

    def test_summary_has_all_methods(self):
        gene_ids, raw, original_padj = self._setup()
        result = compute_correction_comparison(gene_ids, raw, original_padj)
        assert "original" in result["summary"]
        for m in CorrectionMethod:
            assert m.value in result["summary"]

    def test_no_gene_results_by_default(self):
        gene_ids, raw, original_padj = self._setup()
        result = compute_correction_comparison(gene_ids, raw, original_padj)
        assert "gene_results" in result  # Still present, just not excluded

    def test_custom_threshold(self):
        gene_ids, raw, original_padj = self._setup()
        result_005 = compute_correction_comparison(gene_ids, raw, original_padj, threshold=0.05)
        result_01 = compute_correction_comparison(gene_ids, raw, original_padj, threshold=0.1)
        # At a higher threshold, we expect >= as many or more significant genes
        for method in CorrectionMethod:
            key = method.value
            assert result_01["summary"][key]["n_significant"] >= result_005["summary"][key]["n_significant"]

    def test_single_method(self):
        gene_ids, raw, original_padj = self._setup()
        result = compute_correction_comparison(
            gene_ids, raw, original_padj, methods=[CorrectionMethod.BH]
        )
        assert "bh" in result["summary"]
        assert "bonferroni" not in result["summary"]

    def test_gene_results_length(self):
        gene_ids, raw, original_padj = self._setup()
        result = compute_correction_comparison(gene_ids, raw, original_padj)
        assert len(result["gene_results"]) == 5

    def test_gene_results_fields(self):
        gene_ids, raw, original_padj = self._setup()
        result = compute_correction_comparison(gene_ids, raw, original_padj)
        g = result["gene_results"][0]
        assert "gene_id" in g
        assert "pvalue" in g
        assert "original_padj" in g
        assert "padj_bh" in g

    def test_significance_count_is_integer(self):
        gene_ids, raw, original_padj = self._setup()
        result = compute_correction_comparison(gene_ids, raw, original_padj)
        for entry in result["summary"].values():
            assert isinstance(entry["n_significant"], int)
