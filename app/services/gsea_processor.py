"""
Gene Set Enrichment Analysis (GSEA) Processor

This service implements GSEA analysis for ranked gene lists.
Supports multiple gene set databases: GO, KEGG, Reactome, WikiPathways, MSigDB.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class GSEAResult:
    """Result of GSEA analysis for a single gene set"""
    gene_set_name: str
    gene_set_size: int
    enrichment_score: float
    normalized_enrichment_score: float
    p_value: float
    fdr_q_value: float
    leading_edge_genes: List[str]
    running_enrichment_scores: List[float]
    gene_positions: List[int]
    core_enrichment: List[str]

    def to_dict(self) -> dict:
        return {
            "gene_set_name": self.gene_set_name,
            "gene_set_size": self.gene_set_size,
            "enrichment_score": self.enrichment_score,
            "normalized_enrichment_score": self.normalized_enrichment_score,
            "p_value": self.p_value,
            "fdr_q_value": self.fdr_q_value,
            "leading_edge_genes": self.leading_edge_genes,
            "running_enrichment_scores": self.running_enrichment_scores,
            "gene_positions": self.gene_positions,
            "core_enrichment": self.core_enrichment
        }


class GSEAProcessor:
    """
    Gene Set Enrichment Analysis processor

    Implements the GSEA algorithm as described in:
    Subramanian et al. (2005) PNAS 102(43):15545-15550
    """

    def __init__(self, min_size: int = 15, max_size: int = 500, power: float = 1.0):
        """
        Initialize GSEA processor

        Args:
            min_size: Minimum gene set size to consider
            max_size: Maximum gene set size to consider
            power: Weighting exponent for enrichment score (default=1.0)
        """
        self.min_size = min_size
        self.max_size = max_size
        self.power = power

    def run_gsea(
        self,
        ranked_genes: pd.DataFrame,
        gene_sets: Dict[str, List[str]],
        metric_column: str = "metric",
        n_permutations: int = 1000,
        seed: int = 42
    ) -> List[GSEAResult]:
        """
        Run GSEA analysis on ranked gene list

        Args:
            ranked_genes: DataFrame with gene IDs and ranking metric
            gene_sets: Dictionary mapping gene set names to gene lists
            metric_column: Column name containing ranking metric
            n_permutations: Number of permutations for null distribution
            seed: Random seed for reproducibility

        Returns:
            List of GSEAResult objects
        """
        np.random.seed(seed)

        # Ensure genes are ranked by metric (descending for positive, ascending for negative)
        ranked_genes = ranked_genes.sort_values(by=metric_column, ascending=False)
        gene_list = ranked_genes.index.tolist()
        metrics = ranked_genes[metric_column].values

        results = []

        # Filter gene sets by size
        filtered_gene_sets = {
            name: genes for name, genes in gene_sets.items()
            if self.min_size <= len(genes) <= self.max_size
        }

        logger.info(f"Running GSEA on {len(filtered_gene_sets)} gene sets")

        # Calculate enrichment scores for all gene sets
        es_observed = {}
        es_details = {}

        for gene_set_name, gene_set in filtered_gene_sets.items():
            es, running_es, positions = self._calculate_enrichment_score(
                gene_list, gene_set, metrics
            )
            es_observed[gene_set_name] = es
            es_details[gene_set_name] = {
                "running_es": running_es,
                "positions": positions,
                "gene_set": gene_set
            }

        # Generate null distribution via permutations
        logger.info(f"Running {n_permutations} permutations for null distribution")
        null_distributions = self._generate_null_distribution(
            gene_list, filtered_gene_sets, metrics, n_permutations
        )

        # Calculate normalized enrichment scores and p-values
        for gene_set_name in filtered_gene_sets.keys():
            es = es_observed[gene_set_name]
            null_dist = null_distributions[gene_set_name]

            # Normalize ES
            if es >= 0:
                mean_null = np.mean([x for x in null_dist if x >= 0])
            else:
                mean_null = np.abs(np.mean([x for x in null_dist if x < 0]))

            nes = es / mean_null if mean_null != 0 else 0

            # Calculate p-value
            if es >= 0:
                p_value = np.sum(null_dist >= es) / len(null_dist)
            else:
                p_value = np.sum(null_dist <= es) / len(null_dist)

            # Extract leading edge genes
            details = es_details[gene_set_name]
            running_es = details["running_es"]
            positions = details["positions"]
            gene_set = details["gene_set"]

            # Find peak position
            if es >= 0:
                peak_idx = np.argmax(running_es)
            else:
                peak_idx = np.argmin(running_es)

            # Leading edge: genes before peak
            leading_edge_positions = [p for p in positions if p <= peak_idx]
            leading_edge_genes = [gene_list[p] for p in leading_edge_positions]

            # Core enrichment: intersection of gene set and leading edge
            core_enrichment = [g for g in leading_edge_genes if g in gene_set]

            results.append(GSEAResult(
                gene_set_name=gene_set_name,
                gene_set_size=len(gene_set),
                enrichment_score=es,
                normalized_enrichment_score=nes,
                p_value=p_value,
                fdr_q_value=0.0,  # Will be calculated after all p-values are available
                leading_edge_genes=leading_edge_genes[:10],  # Top 10
                running_enrichment_scores=running_es.tolist(),
                gene_positions=positions,
                core_enrichment=core_enrichment[:20]  # Top 20
            ))

        # Calculate FDR q-values using Benjamini-Hochberg
        results = self._calculate_fdr(results)

        # Sort by NES (descending absolute value)
        results.sort(key=lambda x: abs(x.normalized_enrichment_score), reverse=True)

        return results

    def _calculate_enrichment_score(
        self,
        gene_list: List[str],
        gene_set: List[str],
        metrics: np.ndarray
    ) -> Tuple[float, np.ndarray, List[int]]:
        """
        Calculate enrichment score for a gene set

        Returns:
            Tuple of (enrichment_score, running_scores, hit_positions)
        """
        N = len(gene_list)
        gene_set_set = set(gene_set)

        # Identify hit positions
        hit_positions = [i for i, gene in enumerate(gene_list) if gene in gene_set_set]
        Nh = len(hit_positions)

        if Nh == 0:
            return 0.0, np.zeros(N), []

        # Calculate running sum
        running_sum = np.zeros(N)

        # Weighted increment for hits
        hit_metrics = np.abs(metrics[hit_positions]) ** self.power
        Nr = np.sum(hit_metrics)

        if Nr == 0:
            Nr = 1  # Avoid division by zero

        # Calculate running enrichment score
        for i in range(N):
            if i in hit_positions:
                hit_idx = hit_positions.index(i)
                increment = hit_metrics[hit_idx] / Nr
            else:
                increment = -1.0 / (N - Nh)

            if i == 0:
                running_sum[i] = increment
            else:
                running_sum[i] = running_sum[i-1] + increment

        # Enrichment score is maximum deviation from zero
        es_positive = np.max(running_sum)
        es_negative = np.min(running_sum)

        if abs(es_positive) > abs(es_negative):
            es = es_positive
        else:
            es = es_negative

        return es, running_sum, hit_positions

    def _generate_null_distribution(
        self,
        gene_list: List[str],
        gene_sets: Dict[str, List[str]],
        metrics: np.ndarray,
        n_permutations: int
    ) -> Dict[str, List[float]]:
        """
        Generate null distribution via phenotype permutation

        Returns:
            Dictionary mapping gene set names to null ES distributions
        """
        null_distributions = {name: [] for name in gene_sets.keys()}

        for _ in range(n_permutations):
            # Permute the metrics (phenotype permutation)
            permuted_metrics = np.random.permutation(metrics)

            # Calculate ES for each gene set with permuted metrics
            for gene_set_name, gene_set in gene_sets.items():
                es, _, _ = self._calculate_enrichment_score(
                    gene_list, gene_set, permuted_metrics
                )
                null_distributions[gene_set_name].append(es)

        return null_distributions

    def _calculate_fdr(self, results: List[GSEAResult]) -> List[GSEAResult]:
        """
        Calculate FDR q-values using Benjamini-Hochberg procedure
        """
        # Separate positive and negative NES
        positive_results = [r for r in results if r.normalized_enrichment_score >= 0]
        negative_results = [r for r in results if r.normalized_enrichment_score < 0]

        # Calculate FDR for each group separately
        for result_group in [positive_results, negative_results]:
            if not result_group:
                continue

            # Sort by p-value
            result_group.sort(key=lambda x: x.p_value)

            # Benjamini-Hochberg
            n = len(result_group)
            for i, result in enumerate(result_group):
                rank = i + 1
                result.fdr_q_value = result.p_value * n / rank

            # Ensure monotonicity
            for i in range(n - 2, -1, -1):
                if result_group[i].fdr_q_value > result_group[i + 1].fdr_q_value:
                    result_group[i].fdr_q_value = result_group[i + 1].fdr_q_value

        return results


class GeneSetsLoader:
    """
    Load gene sets from various databases
    """

    @staticmethod
    def load_from_gmt(file_path: str) -> Dict[str, List[str]]:
        """
        Load gene sets from GMT file format

        GMT format: gene_set_name\tdescription\tgene1\tgene2\t...
        """
        gene_sets = {}

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    gene_set_name = parts[0]
                    genes = parts[2:]  # Skip description
                    gene_sets[gene_set_name] = genes

        return gene_sets

    @staticmethod
    def load_from_dict(gene_sets_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Load gene sets from dictionary
        """
        return gene_sets_dict

    @staticmethod
    def get_default_gene_sets() -> Dict[str, List[str]]:
        """
        Get default gene sets (placeholder - in production would load from database)
        """
        # This is a placeholder - in production, load from actual databases
        return {
            "GO_CELL_CYCLE": ["CDK1", "CCNB1", "CCNA2", "CDC20"],
            "GO_APOPTOSIS": ["BAX", "BCL2", "CASP3", "TP53"],
            "KEGG_MAPK_PATHWAY": ["MAPK1", "MAPK3", "RAF1", "MEK1"],
        }


def prepare_ranked_gene_list(
    deg_data: pd.DataFrame,
    ranking_metric: str = "signal2noise"
) -> pd.DataFrame:
    """
    Prepare ranked gene list for GSEA

    Args:
        deg_data: DataFrame with columns [gene_id, log_fc, padj, ...]
        ranking_metric: Method to rank genes
            - "signal2noise": log_fc / stderr (requires stderr column)
            - "log_fc": Simple log fold change
            - "signed_pvalue": -log10(padj) * sign(log_fc)

    Returns:
        DataFrame with gene_id as index and 'metric' column
    """
    if ranking_metric == "log_fc":
        deg_data["metric"] = deg_data["log_fc"]

    elif ranking_metric == "signed_pvalue":
        # Avoid log(0) by adding small epsilon
        padj_safe = deg_data["padj"].replace(0, 1e-300)
        deg_data["metric"] = -np.log10(padj_safe) * np.sign(deg_data["log_fc"])

    elif ranking_metric == "signal2noise":
        if "stderr" not in deg_data.columns:
            # Fallback to log_fc if stderr not available
            logger.warning("stderr column not found, falling back to log_fc ranking")
            deg_data["metric"] = deg_data["log_fc"]
        else:
            deg_data["metric"] = deg_data["log_fc"] / (deg_data["stderr"] + 1e-10)

    else:
        raise ValueError(f"Unknown ranking metric: {ranking_metric}")

    # Set gene_id as index
    ranked_genes = deg_data.set_index("gene_id")[["metric"]]

    # Sort by metric (descending)
    ranked_genes = ranked_genes.sort_values(by="metric", ascending=False)

    return ranked_genes
