#!/usr/bin/env python3
"""Quick test: validate GO enrichment and GSEA return results with real data."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.services.go_service import GOService
from app.services.gsea_processor import GSEAProcessor

DATASET_ID = "1399ebf5-e15b-4f67-9e65-e487a2c5922c"
COMPARISON_NAME = "chRCC_pS_vs_ccRCC_pS"


async def main():
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # 1. Fetch DEG gene names (same logic as endpoint)
        result = await session.execute(
            text("""
                SELECT COALESCE(NULLIF(gene_name, ''), gene_id) AS gene
                FROM deg_genes
                WHERE dataset_id = :dataset_id
                  AND comparison_name = :cmp
                  AND padj < 0.05
                LIMIT 500
            """),
            {"dataset_id": DATASET_ID, "cmp": COMPARISON_NAME},
        )
        study_genes = [row[0] for row in result.fetchall()]
        print(f"\n[DEG] {len(study_genes)} genes with padj < 0.05")
        print(f"  Sample: {study_genes[:10]}")

        if not study_genes:
            print("ERROR: no DEGs found")
            return

        # 2. GO enrichment
        print("\n[GO] Running enrichment ...")
        go_service = GOService()
        try:
            go_results = await go_service.go_enrichment_analysis(
                db=session,
                gene_list=study_genes,
                background=None,
                namespace=None,
                organism="Homo sapiens",
                min_gene_count=3,
                max_gene_count=500,
                pvalue_threshold=0.05,
            )
            print(f"[GO] Results: {len(go_results)} significant terms")
            if go_results:
                top = sorted(go_results, key=lambda x: x.get("pvalue", 1))[:5]
                for t in top:
                    print(f"  {t.get('go_id')} {t.get('name','?')[:50]} p={t.get('pvalue',1):.2e}")
            else:
                print("  WARNING: 0 results returned")
        except Exception as e:
            print(f"[GO] ERROR: {e}")

        # 3. GSEA
        print("\n[GSEA] Running analysis ...")

        # Fetch ranked genes
        result2 = await session.execute(
            text("""
                SELECT COALESCE(NULLIF(gene_name, ''), gene_id) AS gene, log_fc
                FROM deg_genes
                WHERE dataset_id = :dataset_id
                  AND comparison_name = :cmp
                  AND log_fc IS NOT NULL
                ORDER BY log_fc DESC
                LIMIT 1000
            """),
            {"dataset_id": DATASET_ID, "cmp": COMPARISON_NAME},
        )
        ranked_genes = {row[0]: row[1] for row in result2.fetchall()}
        print(f"  {len(ranked_genes)} ranked genes")

        from app.services.gene_set_loader import GeneSetLoader, GeneSetDatabase
        loader = GeneSetLoader(session)
        gene_sets = await loader.get_gene_sets(
            database=GeneSetDatabase.GO_BP,
            organism="Homo sapiens",
            min_size=5,
            max_size=500,
        )
        print(f"  {len(gene_sets)} gene sets available for GO_BP")

        if gene_sets:
            import pandas as pd
            # Build ranked DataFrame as expected by GSEAProcessor
            ranked_df = pd.DataFrame(
                list(ranked_genes.items()), columns=["gene", "metric"]
            ).set_index("gene")

            # Convert GeneSet objects to dict {name: [genes]}
            gene_sets_dict = {
                gs.name: gs.genes if isinstance(gs.genes, list) else []
                for gs in gene_sets
                if gs.genes
            }
            print(f"  {len(gene_sets_dict)} non-empty gene sets passed to GSEA")

            gsea = GSEAProcessor()
            try:
                gsea_results = gsea.run_gsea(
                    ranked_genes=ranked_df,
                    gene_sets=gene_sets_dict,
                    n_permutations=100,
                )
                print(f"[GSEA] Results: {len(gsea_results)} terms")
                if gsea_results:
                    for r in gsea_results[:3]:
                        print(f"  {r.gene_set_name[:50]} NES={r.nes:.3f} p={r.pvalue:.3e}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[GSEA] ERROR: {e}")
        else:
            print("[GSEA] No gene sets found - skipping")

    await engine.dispose()
    print("\nDone.")


asyncio.run(main())
