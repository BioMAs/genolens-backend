"""
GO (Gene Ontology) Service

Provides functionality for:
- GO term queries and hierarchy navigation
- GO enrichment analysis
- Annotation propagation through GO hierarchy
- Term relationship mapping
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from collections import defaultdict, deque
import numpy as np
from scipy.stats import hypergeom

from app.models.models import GOTerm, GOAnnotation, GeneSet, GeneSetDatabase
from app.core.monitoring import timing_decorator
from app.services.stats_service import correct_pvalues, CorrectionMethod

logger = logging.getLogger(__name__)


class GOService:
    """Service for Gene Ontology operations."""

    # GO namespaces mapping
    NAMESPACES = {
        'BP': 'biological_process',
        'MF': 'molecular_function',
        'CC': 'cellular_component'
    }

    @timing_decorator(name="get_go_term")
    async def get_go_term(
        self,
        db: AsyncSession,
        go_id: str,
        include_ancestors: bool = False,
        include_descendants: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a GO term with optional hierarchy information.
        
        Args:
            db: Database session
            go_id: GO identifier (e.g., GO:0006915)
            include_ancestors: Whether to include parent terms
            include_descendants: Whether to include child terms
            
        Returns:
            Dictionary with term information and optional hierarchy
        """
        query = select(GOTerm).where(GOTerm.go_id == go_id)
        result = await db.execute(query)
        term = result.scalar_one_or_none()
        
        if not term:
            return None
        
        term_dict = {
            "go_id": term.go_id,
            "name": term.name,
            "namespace": term.namespace,
            "definition": term.definition,
            "is_a": term.is_a,
            "part_of": term.part_of,
            "regulates": term.regulates,
            "synonyms": term.synonyms,
            "level": term.level,
            "gene_count": term.gene_count,
            "is_obsolete": term.is_obsolete
        }
        
        if include_ancestors:
            term_dict["ancestors"] = await self._get_ancestors(db, go_id)
        
        if include_descendants:
            term_dict["descendants"] = await self._get_descendants(db, go_id)
        
        return term_dict

    @timing_decorator(name="search_go_terms")
    async def search_go_terms(
        self,
        db: AsyncSession,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search GO terms by name or GO ID.
        
        Args:
            db: Database session
            query: Search query (term name or GO ID)
            namespace: Filter by namespace (BP, MF, CC)
            limit: Maximum results
            
        Returns:
            List of matching GO terms
        """
        conditions = [GOTerm.is_obsolete == False]
        
        # If query looks like a GO ID, search by ID
        if query.startswith("GO:"):
            conditions.append(GOTerm.go_id.ilike(f"%{query}%"))
        else:
            # Fuzzy text search on name
            conditions.append(GOTerm.name.ilike(f"%{query}%"))
        
        if namespace:
            full_namespace = self.NAMESPACES.get(namespace.upper(), namespace)
            conditions.append(GOTerm.namespace == full_namespace)
        
        stmt = select(GOTerm).where(and_(*conditions)).limit(limit)
        result = await db.execute(stmt)
        terms = result.scalars().all()
        
        return [{
            "go_id": t.go_id,
            "name": t.name,
            "namespace": t.namespace,
            "definition": t.definition,
            "gene_count": t.gene_count,
            "level": t.level
        } for t in terms]

    @timing_decorator(name="get_gene_annotations")
    async def get_gene_annotations(
        self,
        db: AsyncSession,
        gene_symbols: List[str],
        namespace: Optional[str] = None,
        organism: str = "Homo sapiens",
        evidence_codes: Optional[List[str]] = None,
        propagate: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get GO annotations for a list of genes.
        
        Args:
            db: Database session
            gene_symbols: List of gene symbols
            namespace: Filter by namespace (BP, MF, CC)
            organism: Organism name
            evidence_codes: Filter by evidence codes (e.g., ["IDA", "IMP"])
            propagate: Whether to include inherited annotations from ancestors
            
        Returns:
            Dictionary mapping gene symbols to their GO annotations
        """
        conditions = [
            GOAnnotation.gene_symbol.in_(gene_symbols),
            GOAnnotation.organism == organism
        ]
        
        if evidence_codes:
            conditions.append(GOAnnotation.evidence_code.in_(evidence_codes))
        
        # Get direct annotations
        stmt = select(GOAnnotation).where(and_(*conditions))
        result = await db.execute(stmt)
        annotations = result.scalars().all()
        
        # Group by gene
        gene_annots = defaultdict(list)
        go_ids = set()
        
        for annot in annotations:
            go_ids.add(annot.go_id)
            gene_annots[annot.gene_symbol].append({
                "go_id": annot.go_id,
                "evidence_code": annot.evidence_code,
                "source_db": annot.source_db,
                "qualifier": annot.qualifier
            })
        
        # Get GO term details.
        #
        # NOTE: `terms` holds only the *directly annotated* GO ids (namespace-filtered). Ancestor
        # propagation below can therefore only emit an ancestor that is itself directly annotated
        # somewhere in this gene set — narrower than the true path rule the docstring claims.
        # Behaviour preserved deliberately: widening it changes every enrichment result, which is
        # a scientific call, not a refactor.
        if go_ids:
            term_query = select(GOTerm).where(GOTerm.go_id.in_(list(go_ids)))
            if namespace:
                full_namespace = self.NAMESPACES.get(namespace.upper(), namespace)
                term_query = term_query.where(GOTerm.namespace == full_namespace)
            
            term_result = await db.execute(term_query)
            terms = {t.go_id: t for t in term_result.scalars().all()}
            
            # Enrich annotations with term details
            for gene, annots in gene_annots.items():
                for annot in annots:
                    term = terms.get(annot["go_id"])
                    if term:
                        annot["term_name"] = term.name
                        annot["namespace"] = term.namespace
                        annot["level"] = term.level
        
        # Propagate annotations if requested
        if propagate and go_ids:
            propagated = await self._propagate_annotations(db, gene_annots, terms)
            return propagated
        
        return dict(gene_annots)

    @timing_decorator(name="go_enrichment_analysis")
    async def go_enrichment_analysis(
        self,
        db: AsyncSession,
        gene_list: List[str],
        background: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        organism: str = "Homo sapiens",
        min_gene_count: int = 5,
        max_gene_count: int = 500,
        pvalue_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Perform GO enrichment analysis using hypergeometric test.
        
        Args:
            db: Database session
            gene_list: List of genes of interest (e.g., DEGs)
            background: Background gene list (if None, uses all annotated genes)
            namespace: Filter by namespace (BP, MF, CC)
            organism: Organism name
            min_gene_count: Minimum genes per term
            max_gene_count: Maximum genes per term
            pvalue_threshold: P-value threshold for significance
            
        Returns:
            List of enriched GO terms with statistics
        """
        # Get annotations for gene list
        gene_annots = await self.get_gene_annotations(
            db, gene_list, namespace, organism, propagate=True
        )
        
        # Get background annotations
        if background is None:
            # Use all genes with annotations in this namespace
            bg_query = select(GOAnnotation.gene_symbol).distinct()
            conditions = [GOAnnotation.organism == organism]
            
            if namespace:
                # Join with GOTerm to filter by namespace
                bg_query = bg_query.join(
                    GOTerm,
                    GOAnnotation.go_id == GOTerm.go_id
                )
                full_namespace = self.NAMESPACES.get(namespace.upper(), namespace)
                conditions.append(GOTerm.namespace == full_namespace)
            
            bg_query = bg_query.where(and_(*conditions))
            bg_result = await db.execute(bg_query)
            background = [row[0] for row in bg_result.all()]
        
        bg_annots = await self.get_gene_annotations(
            db, background, namespace, organism, propagate=True
        )
        
        N = len(background)  # Total genes in background
        n = len(gene_list)   # Genes in study set
        
        # Count genes per GO term
        study_go_counts = defaultdict(set)
        for gene, annots in gene_annots.items():
            for annot in annots:
                study_go_counts[annot["go_id"]].add(gene)
        
        bg_go_counts = defaultdict(set)
        for gene, annots in bg_annots.items():
            for annot in annots:
                bg_go_counts[annot["go_id"]].add(gene)
        
        # Perform hypergeometric test for ALL GO terms (no raw p-value pre-filter)
        enrichment_results = []
        all_go_ids = []

        for go_id, study_genes in study_go_counts.items():
            k = len(study_genes)
            K = len(bg_go_counts.get(go_id, set()))

            if K < min_gene_count or K > max_gene_count:
                continue

            pvalue = hypergeom.sf(k - 1, N, K, n)
            enrichment_results.append({
                "go_id": go_id,
                "pvalue": float(pvalue),
                "study_count": k,
                "study_genes": list(study_genes),
                "background_count": K,
                "enrichment_ratio": (k / n) / (K / N) if K > 0 else 0,
            })
            all_go_ids.append(go_id)

        # Apply BH FDR on the full set of tested terms (correct procedure)
        if enrichment_results:
            raw_pvals = [r["pvalue"] for r in enrichment_results]
            adj_pvals = correct_pvalues(raw_pvals, method=CorrectionMethod.BH)
            for result, fdr in zip(enrichment_results, adj_pvals):
                result["fdr"] = float(fdr)

        # Filter by FDR threshold AFTER correction
        enrichment_results = [r for r in enrichment_results if r["fdr"] <= pvalue_threshold]

        # Fetch GO term details for the significant set
        if enrichment_results:
            sig_go_ids = [r["go_id"] for r in enrichment_results]
            term_query = select(GOTerm).where(GOTerm.go_id.in_(sig_go_ids))
            term_result = await db.execute(term_query)
            terms = {t.go_id: t for t in term_result.scalars().all()}

            for result in enrichment_results:
                term = terms.get(result["go_id"])
                if term:
                    result["term_name"] = term.name
                    result["namespace"] = term.namespace
                    result["definition"] = term.definition
                    result["level"] = term.level

        enrichment_results.sort(key=lambda x: x["fdr"])
        logger.info(f"✅ GO enrichment: {len(enrichment_results)} significant terms (padj≤{pvalue_threshold})")

        return enrichment_results

    async def _load_parent_map(self, db: AsyncSession) -> Dict[str, Tuple[str, ...]]:
        """
        The whole GO parent graph in one query: ``go_id -> (is_a + part_of)``.

        ~42k rows and three narrow columns. Loading it once replaces the per-node ``SELECT`` that
        the hierarchy walk used to issue, which is what made annotation propagation quadratic in
        round trips. ``is_a``/``part_of`` are JSON columns rather than Postgres arrays, so a
        recursive CTE would have to unnest JSON without an index to help it — reading the graph
        into memory is both simpler and faster.
        """
        rows = (
            await db.execute(select(GOTerm.go_id, GOTerm.is_a, GOTerm.part_of))
        ).all()
        return {
            row[0]: tuple((row[1] or []) + (row[2] or []))
            for row in rows
        }

    @staticmethod
    def _ancestors_from_map(
        go_id: str,
        parent_map: Dict[str, Tuple[str, ...]],
        cache: Dict[str, Set[str]],
    ) -> Set[str]:
        """
        Transitive ancestors of ``go_id``, memoised in ``cache`` across calls.

        ``cache`` is owned by the caller so a GO id is expanded **once** however many genes carry
        it — the old code re-walked the hierarchy for every (gene, annotation) pair, which on a
        7357-gene run meant 112550 walks over just 12193 distinct ids.

        Iterative rather than recursive, and it stops at any node whose closure is already known.
        ``go_id`` itself is not included, matching the previous behaviour.
        """
        hit = cache.get(go_id)
        if hit is not None:
            return hit

        ancestors: Set[str] = set()
        to_visit = deque(parent_map.get(go_id, ()))
        while to_visit:
            current_id = to_visit.popleft()
            if current_id in ancestors:
                continue
            ancestors.add(current_id)

            known = cache.get(current_id)
            if known is not None:
                # Closure already computed for this node — no need to walk past it.
                ancestors |= known
                continue
            to_visit.extend(parent_map.get(current_id, ()))

        cache[go_id] = ancestors
        return ancestors

    async def _get_ancestors(self, db: AsyncSession, go_id: str) -> List[str]:
        """
        Get all ancestor GO IDs (transitive closure up the hierarchy).

        One bulk read of the parent graph instead of one ``SELECT`` per node visited. For a single
        term that trades ~10 point queries for one larger query; the batch caller
        (:meth:`_propagate_annotations`) loads the graph itself and shares it.
        """
        parent_map = await self._load_parent_map(db)
        return list(self._ancestors_from_map(go_id, parent_map, {}))

    async def _get_descendants(self, db: AsyncSession, go_id: str) -> List[str]:
        """Get all descendant GO IDs (recursive traversal down the hierarchy)."""
        descendants = set()
        
        # Find all terms that have go_id as parent
        query = select(GOTerm).where(
            or_(
                GOTerm.is_a.contains([go_id]),
                GOTerm.part_of.contains([go_id])
            )
        )
        result = await db.execute(query)
        children = result.scalars().all()
        
        for child in children:
            if child.go_id not in descendants:
                descendants.add(child.go_id)
                # Recursively get descendants of this child
                child_descendants = await self._get_descendants(db, child.go_id)
                descendants.update(child_descendants)
        
        return list(descendants)

    async def _propagate_annotations(
        self,
        db: AsyncSession,
        gene_annots: Dict[str, List[Dict[str, Any]]],
        terms: Dict[str, GOTerm]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Propagate annotations up the GO hierarchy.
        True path rule: if a gene is annotated to a term, it's also annotated to all ancestors.

        The hierarchy is read **once** into a parent map and every ancestor closure is memoised
        across genes. Previously this awaited a fresh hierarchy walk — itself one ``SELECT`` per
        node — for every (gene, annotation) pair: ~1.16 million sequential queries for a
        7357-gene run, which took 384 s. Same output, three queries.
        """
        parent_map = await self._load_parent_map(db)
        ancestor_cache: Dict[str, Set[str]] = {}

        propagated = defaultdict(list)

        for gene, annots in gene_annots.items():
            seen_go_ids = set()

            for annot in annots:
                go_id = annot["go_id"]

                # Add direct annotation
                if go_id not in seen_go_ids:
                    propagated[gene].append(annot.copy())
                    seen_go_ids.add(go_id)

                # Get ancestors and add inherited annotations.
                # Only for terms present in `terms`, and only ancestors that are themselves in
                # `terms` — preserved from the original. See the note in `get_gene_annotations`
                # on why that makes propagation narrower than the true path rule implies.
                if go_id not in terms:
                    continue

                for ancestor_id in self._ancestors_from_map(go_id, parent_map, ancestor_cache):
                    if ancestor_id in seen_go_ids:
                        continue
                    ancestor_term = terms.get(ancestor_id)
                    if ancestor_term:
                        propagated[gene].append({
                            "go_id": ancestor_id,
                            "term_name": ancestor_term.name,
                            "namespace": ancestor_term.namespace,
                            "level": ancestor_term.level,
                            "evidence_code": "IEA",  # Inferred from Electronic Annotation
                            "source_db": "propagated",
                            "qualifier": f"inherited from {go_id}"
                        })
                        seen_go_ids.add(ancestor_id)

        return dict(propagated)


    @timing_decorator(name="run_ora_enrichment")
    async def run_ora_enrichment(
        self,
        db: AsyncSession,
        gene_list: List[str],
        database: GeneSetDatabase,
        background: Optional[List[str]] = None,
        organism: str = "Homo sapiens",
        min_gene_count: int = 5,
        max_gene_count: int = 500,
        padj_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Over-Representation Analysis (ORA) using gene sets from GeneSetDatabase
        (KEGG, REACTOME, HALLMARK, etc.).

        Uses the same hypergeometric test as go_enrichment_analysis, BH FDR
        applied on ALL tested gene sets (no raw p-value pre-filter).

        Args:
            db: Database session
            gene_list: Genes of interest (e.g. significant DEGs)
            database: Which GeneSetDatabase to query (KEGG, REACTOME, HALLMARK, …)
            background: Background gene list. If None, uses all genes in gene sets for this database.
            organism: Not used directly but kept for API consistency
            min_gene_count: Min overlap genes to test a gene set
            max_gene_count: Max size of a gene set
            padj_threshold: FDR threshold for reporting results

        Returns:
            List of enriched gene sets sorted by FDR (ascending)
        """
        study_set = set(g.upper() for g in gene_list if g)

        # Load all gene sets for the requested database
        gs_result = await db.execute(
            select(GeneSet).where(
                GeneSet.database == database,
                GeneSet.size >= min_gene_count,
                GeneSet.size <= max_gene_count,
            )
        )
        gene_sets = gs_result.scalars().all()

        if not gene_sets:
            logger.info(f"[ORA] No gene sets found for database {database.value}")
            return []

        # Build background
        if background is None:
            all_bg_genes: set = set()
            for gs in gene_sets:
                all_bg_genes.update(g.upper() for g in gs.genes if g)
            background_set = all_bg_genes
        else:
            background_set = set(g.upper() for g in background if g)

        N = len(background_set)
        n = len(study_set & background_set)  # study genes present in background

        if n == 0 or N == 0:
            logger.info(f"[ORA] Empty intersection for {database.value}")
            return []

        # Hypergeometric test for each gene set
        results: List[Dict[str, Any]] = []
        for gs in gene_sets:
            gs_genes = set(g.upper() for g in gs.genes if g)
            overlap = study_set & gs_genes & background_set
            k = len(overlap)
            K = len(gs_genes & background_set)

            if K < min_gene_count or K > max_gene_count or k == 0:
                continue

            pvalue = float(hypergeom.sf(k - 1, N, K, n))
            results.append({
                "pathway_id": gs.name,
                "pathway_name": gs.description or gs.name,
                "database": database.value,
                "pvalue": pvalue,
                "study_count": k,
                "background_count": K,
                "study_genes": sorted(overlap),
                "gene_ratio": f"{k}/{n}",
                "bg_ratio": f"{K}/{N}",
                "enrichment_ratio": (k / n) / (K / N) if K > 0 and n > 0 else 0,
            })

        if not results:
            return []

        # Apply BH FDR on the full set of tested gene sets
        raw_pvals = [r["pvalue"] for r in results]
        adj_pvals = correct_pvalues(raw_pvals, method=CorrectionMethod.BH)
        for r, fdr in zip(results, adj_pvals):
            r["fdr"] = float(fdr)

        # Filter by FDR threshold and sort
        results = [r for r in results if r["fdr"] <= padj_threshold]
        results.sort(key=lambda x: x["fdr"])

        logger.info(f"[ORA] {database.value}: {len(results)} enriched gene sets (padj≤{padj_threshold})")
        return results


# Global instance
go_service = GOService()
