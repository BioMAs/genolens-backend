"""
Backfill deg_genes table for existing DEG datasets that have 0 entries.

Usage (from backend/ dir):
    docker compose exec api python scripts/backfill_deg_genes.py
"""
import asyncio
import io
import logging
import sys

import pandas as pd
from sqlalchemy import delete, insert, select, text, update

sys.path.insert(0, "/app")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    from app.db.session import AsyncSessionLocal
    from app.models.models import Dataset, DatasetStatus, DatasetType, DegGene
    from app.services.data_processor import data_processor
    from app.services.storage import storage_service

    async with AsyncSessionLocal() as db:
        # Find all READY DEG datasets with no entries in deg_genes
        result = await db.execute(
            text(
                """
                SELECT d.id, d.name, d.parquet_file_path,
                       d.deg_up_count, d.deg_down_count
                FROM datasets d
                LEFT JOIN deg_genes dg ON dg.dataset_id = d.id
                WHERE d.type = 'DEG'
                  AND d.status = 'READY'
                  AND d.parquet_file_path IS NOT NULL
                GROUP BY d.id
                HAVING COUNT(dg.id) = 0
                ORDER BY d.created_at
                """
            )
        )
        rows = result.fetchall()

    if not rows:
        logger.info("No datasets to backfill. All DEG datasets have deg_genes entries.")
        return

    logger.info(f"Found {len(rows)} DEG dataset(s) to backfill.")

    for row in rows:
        dataset_id = str(row.id)
        name = row.name
        parquet_path = row.parquet_file_path
        logger.info(f"\n--- Backfilling: {name} ({dataset_id}) ---")
        logger.info(f"    Parquet: {parquet_path}")

        try:
            parquet_bytes = await storage_service.download_file(parquet_path)
        except Exception as exc:
            logger.error(f"    SKIP — could not download Parquet: {exc}")
            continue

        df = pd.read_parquet(io.BytesIO(parquet_bytes))
        logger.info(f"    Parquet rows: {len(df)}, columns: {list(df.columns)[:10]}")

        comparisons_map = data_processor._detect_comparisons(df.columns.tolist())
        if not comparisons_map:
            logger.warning(f"    SKIP — no comparison columns detected.")
            continue

        logger.info(f"    Comparisons detected: {list(comparisons_map.keys())}")

        try:
            deg_genes_data = await data_processor.extract_deg_genes_for_db(
                parquet_bytes, comparisons_map
            )
        except Exception as exc:
            logger.error(f"    SKIP — extract_deg_genes_for_db failed: {exc}")
            continue

        async with AsyncSessionLocal() as db:
            # Delete stale entries (idempotent)
            await db.execute(delete(DegGene).where(DegGene.dataset_id == row.id))
            await db.commit()

            chunk_size = 1000
            total_inserted = 0
            for comp, genes_list in deg_genes_data.items():
                if not genes_list:
                    logger.warning(f"    No genes for comparison '{comp}'")
                    continue
                records = [
                    {
                        "dataset_id": dataset_id,
                        "comparison_name": comp,
                        "gene_id": g["gene_id"],
                        "log_fc": g.get("log_fc"),
                        "padj": g.get("padj"),
                        "pvalue": g.get("pvalue"),
                        "regulation": g.get("regulation"),
                        "gene_name": g.get("gene_name"),
                    }
                    for g in genes_list
                ]
                for i in range(0, len(records), chunk_size):
                    await db.execute(insert(DegGene), records[i : i + chunk_size])
                    await db.commit()
                total_inserted += len(records)
                logger.info(f"    ✓ {len(records)} genes inserted for '{comp}'")

            # Update deg_up_count / deg_down_count if NULL
            if row.deg_up_count is None:
                padj_col = next((c for c in df.columns if c.startswith("padj:")), None)
                logfc_col = next((c for c in df.columns if c.startswith("logFC:")), None)
                if padj_col and logfc_col:
                    sig_mask = df[padj_col] < 0.05
                    up_count = int((sig_mask & (df[logfc_col] > 0)).sum())
                    down_count = int((sig_mask & (df[logfc_col] < 0)).sum())
                    await db.execute(
                        update(Dataset)
                        .where(Dataset.id == row.id)
                        .values(deg_up_count=up_count, deg_down_count=down_count)
                    )
                    await db.commit()
                    logger.info(
                        f"    ✓ Updated deg_up_count={up_count}, deg_down_count={down_count}"
                    )

            logger.info(f"    ✓ Total inserted: {total_inserted} genes")

    logger.info("\nBackfill complete.")


if __name__ == "__main__":
    asyncio.run(main())
