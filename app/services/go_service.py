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

from app.models.models import GOTerm, GOAnnotation
from app.core.monitoring import timing_decorator

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
        
        # Get GO term details
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
        # Get annotations for gene list (no propagation — direct GOA annotations are sufficient
        # for hypergeometric enrichment and propagation via recursive BFS is too slow at scale)
        gene_annots = await self.get_gene_annotations(
            db, gene_list, namespace, organism, propagate=False
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
            db, background, namespace, organism, propagate=False
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
        
        # Perform hypergeometric test for each GO term
        enrichment_results = []
        go_ids_to_fetch = []
        
        for go_id, study_genes in study_go_counts.items():
            k = len(study_genes)  # Genes in study with this GO term
            K = len(bg_go_counts.get(go_id, set()))  # Genes in background with this GO term
            
            # Filter by gene count
            if K < min_gene_count or K > max_gene_count:
                continue
            
            # Hypergeometric test (survival function = P(X >= k))
            pvalue = hypergeom.sf(k - 1, N, K, n)
            
            if pvalue <= pvalue_threshold:
                enrichment_results.append({
                    "go_id": go_id,
                    "pvalue": float(pvalue),
                    "study_count": k,
                    "study_genes": list(study_genes),
                    "background_count": K,
                    "enrichment_ratio": (k / n) / (K / N) if K > 0 else 0
                })
                go_ids_to_fetch.append(go_id)
        
        # Fetch GO term details
        if go_ids_to_fetch:
            term_query = select(GOTerm).where(GOTerm.go_id.in_(go_ids_to_fetch))
            term_result = await db.execute(term_query)
            terms = {t.go_id: t for t in term_result.scalars().all()}
            
            for result in enrichment_results:
                term = terms.get(result["go_id"])
                if term:
                    result["go_name"] = term.name
                    result["namespace"] = term.namespace
                    result["definition"] = term.definition
                    result["level"] = term.level
        
        # Sort by p-value
        enrichment_results.sort(key=lambda x: x["pvalue"])
        
        # Apply multiple testing correction (Benjamini-Hochberg FDR)
        if enrichment_results:
            m = len(enrichment_results)
            for i, result in enumerate(enrichment_results):
                result["rank"] = i + 1
                result["fdr"] = min(1.0, result["pvalue"] * m / (i + 1))
        
        logger.info(f"✅ GO enrichment: {len(enrichment_results)} significant terms found")
        
        return enrichment_results

    async def _get_ancestors(self, db: AsyncSession, go_id: str) -> List[str]:
        """Get all ancestor GO IDs (recursive traversal up the hierarchy)."""
        ancestors = set()
        to_visit = deque([go_id])
        
        while to_visit:
            current_id = to_visit.popleft()
            
            query = select(GOTerm).where(GOTerm.go_id == current_id)
            result = await db.execute(query)
            term = result.scalar_one_or_none()
            
            if term:
                parents = term.is_a + term.part_of
                for parent_id in parents:
                    if parent_id not in ancestors:
                        ancestors.add(parent_id)
                        to_visit.append(parent_id)
        
        return list(ancestors)

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
        """
        propagated = defaultdict(list)
        
        for gene, annots in gene_annots.items():
            seen_go_ids = set()
            
            for annot in annots:
                go_id = annot["go_id"]
                
                # Add direct annotation
                if go_id not in seen_go_ids:
                    propagated[gene].append(annot.copy())
                    seen_go_ids.add(go_id)
                
                # Get ancestors and add inherited annotations
                term = terms.get(go_id)
                if term:
                    ancestors = await self._get_ancestors(db, go_id)
                    
                    for ancestor_id in ancestors:
                        if ancestor_id not in seen_go_ids:
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


# Global instance
go_service = GOService()
