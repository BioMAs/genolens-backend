"""
Demo seed service — generates a coherent synthetic demo project for test accounts.

Creates:
  - 1 project "GenoLens Demo — RNAseq Analysis"
  - 2 DEG datasets with real synthetic Parquet files
    • KO_vs_WT : p53 / apoptosis / cell-cycle context
    • Treatment_vs_Control : inflammatory / JAK-STAT / PI3K context
  - DegGene rows (bulk-inserted) for each comparison
  - EnrichmentPathway rows for each comparison
"""
from __future__ import annotations

import io
import logging
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Dataset,
    DatasetStatus,
    DatasetType,
    DegGene,
    EnrichmentPathway,
    Project,
)
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

DEMO_PROJECT_NAME = "GenoLens Demo — RNAseq Analysis"

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DemoAlreadyExistsError(Exception):
    """Raised when the user already has a demo project."""


# ---------------------------------------------------------------------------
# Synthetic gene data — biologically coherent (human genes)
# ---------------------------------------------------------------------------

# Each entry: (gene_id, gene_name, baseMean, log2FC, padj)
_KO_VS_WT_GENES: list[tuple[str, str, float, float, float]] = [
    # UP-regulated (p53/apoptosis/DNA damage context)
    ("ENSG00000141510", "TP53",    1850.2,  2.81,  2.1e-08),
    ("ENSG00000124762", "CDKN1A",  942.5,   3.42,  1.3e-07),
    ("ENSG00000105329", "TGFB1",   623.1,   2.15,  4.5e-06),
    ("ENSG00000116717", "GADD45A", 312.4,   4.10,  8.2e-09),
    ("ENSG00000087088", "BAX",     780.3,   2.64,  3.1e-06),
    ("ENSG00000171791", "BCL2L11", 415.6,   1.98,  2.8e-05),
    ("ENSG00000139684", "BBC3",    289.7,   3.75,  6.4e-08),
    ("ENSG00000128052", "KDR",     531.2,   1.72,  1.7e-04),
    ("ENSG00000112715", "VEGFA",   692.8,   2.04,  5.3e-05),
    ("ENSG00000146648", "EGFR",    1024.3,  1.88,  3.9e-05),
    ("ENSG00000135679", "MDM2",    884.1,   1.53,  2.2e-04),
    ("ENSG00000112679", "PUMA",    198.4,   4.51,  1.1e-09),
    # DOWN-regulated (proliferation suppression)
    ("ENSG00000136997", "MYC",     2103.4, -2.97,  5.6e-09),
    ("ENSG00000110092", "CCND1",   1432.7, -2.34,  4.1e-07),
    ("ENSG00000135446", "CDK4",    891.2,  -1.87,  6.8e-05),
    ("ENSG00000171428", "BCL2",    743.5,  -2.56,  2.4e-07),
    ("ENSG00000148773", "MCL1",    521.8,  -1.65,  8.9e-05),
    ("ENSG00000101412", "E2F1",    678.3,  -2.12,  1.4e-06),
    ("ENSG00000132646", "PCNA",    956.4,  -3.08,  7.3e-08),
    ("ENSG00000139687", "RB1",     1102.5, -1.74,  4.6e-05),
    ("ENSG00000134057", "CCNB1",   834.6,  -2.45,  3.2e-06),
    ("ENSG00000145386", "CCNA2",   623.1,  -2.89,  8.7e-08),
    ("ENSG00000100815", "TRIP12",  412.3,  -1.42,  3.1e-04),
    # Not significant
    ("ENSG00000111640", "GAPDH",   9821.4,  0.04,  0.91),
    ("ENSG00000075624", "ACTB",    8734.2, -0.07,  0.87),
    ("ENSG00000136872", "ALDOA",   2341.5,  0.12,  0.74),
    ("ENSG00000197971", "MBP",     312.1,   0.09,  0.82),
    ("ENSG00000179218", "CALR",    1456.8, -0.05,  0.96),
    ("ENSG00000162409", "PRKAA1",  645.3,   0.18,  0.61),
    ("ENSG00000117461", "PIK3R1",  738.2,  -0.22,  0.54),
]

_TREATMENT_VS_CONTROL_GENES: list[tuple[str, str, float, float, float]] = [
    # UP-regulated (inflammatory / cytokine / JAK-STAT)
    ("ENSG00000136244", "IL6",     312.5,   5.12,  3.4e-12),
    ("ENSG00000232810", "TNF",     548.3,   4.78,  8.1e-11),
    ("ENSG00000169429", "CXCL8",   421.7,   5.64,  1.2e-13),
    ("ENSG00000125538", "IL1B",    389.4,   4.21,  2.7e-10),
    ("ENSG00000060718", "COL11A1", 234.6,   2.89,  4.3e-06),
    ("ENSG00000107862", "ICAM1",   612.8,   3.47,  6.5e-08),
    ("ENSG00000162692", "VCAM1",   478.3,   3.12,  2.1e-07),
    ("ENSG00000108691", "CCL2",    289.7,   4.83,  5.6e-11),
    ("ENSG00000115415", "STAT1",   834.2,   2.34,  3.8e-06),
    ("ENSG00000168610", "STAT3",   912.5,   2.67,  8.4e-07),
    ("ENSG00000019582", "CD44",    756.3,   1.95,  4.7e-05),
    ("ENSG00000100311", "PDGFB",   321.8,   2.78,  1.9e-06),
    ("ENSG00000168209", "DDIT4",   287.4,   3.45,  7.2e-08),
    ("ENSG00000164362", "TERT",    198.6,   2.23,  6.1e-05),
    # DOWN-regulated (tumour suppressors / negative regulators)
    ("ENSG00000171862", "PTEN",    1234.5, -2.41,  5.7e-07),
    ("ENSG00000105647", "PIK3R2",  543.2,  -1.87,  8.3e-05),
    ("ENSG00000135679", "MDM2",    634.7,  -1.62,  2.4e-04),
    ("ENSG00000145604", "SKP2",    478.9,  -2.08,  1.8e-05),
    ("ENSG00000175387", "SMAD2",   812.3,  -1.74,  3.6e-05),
    ("ENSG00000168078", "PBK",     312.4,  -2.95,  4.2e-08),
    ("ENSG00000103197", "TSC2",    623.8,  -1.56,  5.9e-04),
    ("ENSG00000102908", "NFATC4",  289.1,  -2.18,  9.3e-06),
    # Not significant
    ("ENSG00000111640", "GAPDH",   9634.5,  0.06,  0.88),
    ("ENSG00000075624", "ACTB",    8912.3, -0.03,  0.93),
    ("ENSG00000126522", "ARG2",    412.6,   0.14,  0.71),
    ("ENSG00000198804", "MT-CO1",  5234.7, -0.08,  0.84),
    ("ENSG00000169174", "PCSK9",   378.4,   0.19,  0.63),
    ("ENSG00000116285", "ERRFI1",  523.1,  -0.11,  0.79),
    ("ENSG00000141510", "TP53",    1923.4,  0.31,  0.21),
]

# ---------------------------------------------------------------------------
# Synthetic enrichment pathway data
# ---------------------------------------------------------------------------

_KO_VS_WT_PATHWAYS: list[dict] = [
    {
        "pathway_id": "GO:0006915",
        "pathway_name": "apoptotic process",
        "category": "GO:BP",
        "description": "A programmed cell death process...",
        "gene_count": 12,
        "pvalue": 3.2e-06,
        "padj": 1.8e-04,
        "gene_ratio": "12/23",
        "bg_ratio": "280/18000",
        "genes": ["TP53", "BAX", "BCL2L11", "BBC3", "PUMA", "CDKN1A", "GADD45A", "MDM2", "BCL2", "MCL1", "E2F1", "RB1"],
        "regulation": "UP",
    },
    {
        "pathway_id": "hsa04115",
        "pathway_name": "p53 signaling pathway",
        "category": "KEGG",
        "description": "p53 activation leads to cell-cycle arrest and apoptosis",
        "gene_count": 10,
        "pvalue": 5.6e-07,
        "padj": 4.2e-05,
        "gene_ratio": "10/23",
        "bg_ratio": "68/7342",
        "genes": ["TP53", "CDKN1A", "GADD45A", "MDM2", "BAX", "CCND1", "CDK4", "RB1", "VEGFA", "PCNA"],
        "regulation": "ALL",
    },
    {
        "pathway_id": "GO:0006974",
        "pathway_name": "cellular response to DNA damage stimulus",
        "category": "GO:BP",
        "description": "Any process that results in a change in state...",
        "gene_count": 9,
        "pvalue": 2.1e-05,
        "padj": 8.7e-04,
        "gene_ratio": "9/23",
        "bg_ratio": "310/18000",
        "genes": ["TP53", "CDKN1A", "GADD45A", "BAX", "BBC3", "PUMA", "PCNA", "CCNB1", "CCNA2"],
        "regulation": "UP",
    },
    {
        "pathway_id": "GO:0051726",
        "pathway_name": "regulation of cell cycle",
        "category": "GO:BP",
        "description": "Any process that modulates the rate or extent of cell cycle progression",
        "gene_count": 8,
        "pvalue": 4.3e-05,
        "padj": 1.2e-03,
        "gene_ratio": "8/23",
        "bg_ratio": "420/18000",
        "genes": ["CCND1", "CDK4", "E2F1", "RB1", "PCNA", "CCNB1", "CCNA2", "CDKN1A"],
        "regulation": "DOWN",
    },
    {
        "pathway_id": "hsa05200",
        "pathway_name": "Pathways in cancer",
        "category": "KEGG",
        "description": "Signaling pathways dysregulated in cancer",
        "gene_count": 11,
        "pvalue": 8.9e-05,
        "padj": 2.1e-03,
        "gene_ratio": "11/23",
        "bg_ratio": "530/7342",
        "genes": ["TP53", "MYC", "EGFR", "VEGFA", "BCL2", "RB1", "CCND1", "CDK4", "E2F1", "MDM2", "PTEN"],
        "regulation": "ALL",
    },
]

_TREATMENT_VS_CONTROL_PATHWAYS: list[dict] = [
    {
        "pathway_id": "hsa04668",
        "pathway_name": "TNF signaling pathway",
        "category": "KEGG",
        "description": "TNF-alpha-mediated signaling in inflammation",
        "gene_count": 11,
        "pvalue": 2.8e-08,
        "padj": 1.4e-06,
        "gene_ratio": "11/22",
        "bg_ratio": "110/7342",
        "genes": ["TNF", "IL6", "CXCL8", "IL1B", "ICAM1", "VCAM1", "CCL2", "STAT1", "STAT3", "PTEN", "SMAD2"],
        "regulation": "UP",
    },
    {
        "pathway_id": "GO:0006954",
        "pathway_name": "inflammatory response",
        "category": "GO:BP",
        "description": "The immediate defensive reaction...",
        "gene_count": 13,
        "pvalue": 4.1e-09,
        "padj": 3.1e-07,
        "gene_ratio": "13/22",
        "bg_ratio": "390/18000",
        "genes": ["IL6", "TNF", "CXCL8", "IL1B", "ICAM1", "VCAM1", "CCL2", "STAT1", "STAT3", "CD44", "TNF", "IL1B", "PDGFB"],
        "regulation": "UP",
    },
    {
        "pathway_id": "hsa04151",
        "pathway_name": "PI3K-Akt signaling pathway",
        "category": "KEGG",
        "description": "PI3K-Akt pathway regulating cell survival and proliferation",
        "gene_count": 9,
        "pvalue": 7.3e-06,
        "padj": 2.9e-04,
        "gene_ratio": "9/22",
        "bg_ratio": "354/7342",
        "genes": ["PTEN", "PIK3R2", "PDGFB", "STAT3", "MDM2", "TSC2", "SKP2", "VEGFA", "EGFR"],
        "regulation": "ALL",
    },
    {
        "pathway_id": "hsa04630",
        "pathway_name": "JAK-STAT signaling pathway",
        "category": "KEGG",
        "description": "Cytokine-receptor-mediated JAK-STAT signaling",
        "gene_count": 7,
        "pvalue": 1.4e-05,
        "padj": 4.2e-04,
        "gene_ratio": "7/22",
        "bg_ratio": "162/7342",
        "genes": ["STAT1", "STAT3", "IL6", "IL1B", "SOCS3", "JAK2", "NFATC4"],
        "regulation": "UP",
    },
    {
        "pathway_id": "GO:0045087",
        "pathway_name": "innate immune response",
        "category": "GO:BP",
        "description": "Innate immune mechanisms activated early in response to infection or injury",
        "gene_count": 8,
        "pvalue": 3.5e-05,
        "padj": 8.9e-04,
        "gene_ratio": "8/22",
        "bg_ratio": "450/18000",
        "genes": ["TNF", "IL6", "IL1B", "CXCL8", "CCL2", "ICAM1", "STAT1", "PTEN"],
        "regulation": "UP",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_deg_dataframe(
    genes: list[tuple[str, str, float, float, float]],
) -> pd.DataFrame:
    """Build a DEG Parquet-compatible DataFrame from the raw gene list."""
    rows = []
    for gene_id, gene_name, base_mean, log2fc, padj in genes:
        pvalue = padj * 10.0  # synthetic pvalue slightly larger than padj
        rows.append(
            {
                "gene_id": gene_id,
                "gene_name": gene_name,
                "baseMean": round(base_mean, 2),
                "log2FoldChange": round(log2fc, 4),
                "pvalue": float(f"{pvalue:.2e}"),
                "padj": float(f"{padj:.2e}"),
            }
        )
    return pd.DataFrame(rows)


def _dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to Parquet bytes (snappy compressed)."""
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    buf.seek(0)
    return buf.read()


def _regulation(log2fc: float, padj: float) -> str:
    if padj >= 0.05:
        return "NS"
    return "UP" if log2fc > 0 else "DOWN"


def _build_column_mapping() -> dict:
    return {
        "gene_id": "gene_id",
        "gene_name": "gene_name",
        "base_mean": "baseMean",
        "log_fc": "log2FoldChange",
        "pvalue": "pvalue",
        "padj": "padj",
    }


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


async def create_demo_data(
    db: AsyncSession,
    user_id: UUID,
) -> Project:
    """
    Create a full demo project for *user_id*.

    Raises:
        DemoAlreadyExistsError: if the user already owns a project with the
            DEMO_PROJECT_NAME.
    """
    # Guard — one demo per user
    existing = await db.execute(
        select(Project).where(
            Project.owner_id == user_id,
            Project.name == DEMO_PROJECT_NAME,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise DemoAlreadyExistsError(
            f"User {user_id} already has a demo project named '{DEMO_PROJECT_NAME}'."
        )

    # ------------------------------------------------------------------ #
    # 1. Create project
    # ------------------------------------------------------------------ #
    project = Project(
        id=uuid4(),
        owner_id=user_id,
        name=DEMO_PROJECT_NAME,
        description=(
            "Demo project — two RNA-seq differential expression analyses "
            "illustrating the GenoLens platform capabilities. "
            "Context: mouse knockout (KO_vs_WT) and drug treatment "
            "(Treatment_vs_Control) experiments."
        ),
    )
    db.add(project)
    await db.flush()  # get project.id without full commit

    logger.info("[demo_seed] Project created: %s (owner=%s)", project.id, user_id)

    # ------------------------------------------------------------------ #
    # 2. Build the two datasets (Parquet + DB rows + DegGenes + Pathways)
    # ------------------------------------------------------------------ #
    comparisons = [
        (
            "KO_vs_WT",
            "DEG Analysis — KO vs WT",
            "Differential expression in CRISPR knockout vs wild-type cells (p53/apoptosis context)",
            _KO_VS_WT_GENES,
            _KO_VS_WT_PATHWAYS,
        ),
        (
            "Treatment_vs_Control",
            "DEG Analysis — Treatment vs Control",
            "Differential expression after drug treatment (inflammatory/JAK-STAT context)",
            _TREATMENT_VS_CONTROL_GENES,
            _TREATMENT_VS_CONTROL_PATHWAYS,
        ),
    ]

    for comp_name, ds_name, ds_desc, gene_data, pathway_data in comparisons:
        dataset_id = uuid4()

        # ---- 2a. Build & upload Parquet
        df = _build_deg_dataframe(gene_data)
        parquet_bytes = _dataframe_to_parquet_bytes(df)

        parquet_path = (
            f"projects/{project.id}/processed/{dataset_id}.parquet"
        )
        await storage_service.upload_file(parquet_path, parquet_bytes)
        logger.info("[demo_seed] Parquet uploaded: %s", parquet_path)

        # ---- 2b. Compute pre-calculated stats
        sig_mask = df["padj"] < 0.05
        deg_up = int(((df["log2FoldChange"] > 0) & sig_mask).sum())
        deg_down = int(((df["log2FoldChange"] < 0) & sig_mask).sum())
        deg_sig = int(sig_mask.sum())
        total = len(df)

        # ---- 2c. Create Dataset record
        dataset = Dataset(
            id=dataset_id,
            project_id=project.id,
            name=ds_name,
            description=ds_desc,
            type=DatasetType.DEG,
            status=DatasetStatus.READY,
            raw_file_path=None,
            parquet_file_path=parquet_path,
            column_mapping=_build_column_mapping(),
            dataset_metadata={
                "comparisons": [comp_name],
                "original_filename": f"{comp_name}_demo.parquet",
                "is_normalized": True,
                "contains_all_genes": True,
                "demo": True,
            },
            deg_up_count=deg_up,
            deg_down_count=deg_down,
            deg_significant_count=deg_sig,
            total_genes=total,
        )
        db.add(dataset)
        await db.flush()

        logger.info(
            "[demo_seed] Dataset created: %s (%s), up=%d down=%d",
            dataset_id, comp_name, deg_up, deg_down,
        )

        # ---- 2d. Bulk-insert DegGene rows
        deg_records = [
            {
                "id": uuid4(),
                "dataset_id": dataset_id,
                "comparison_name": comp_name,
                "gene_id": g[0],
                "gene_name": g[1],
                "base_mean": g[2],
                "log_fc": g[3],
                "padj": g[4],
                "pvalue": g[4] * 10.0,
                "regulation": _regulation(g[3], g[4]),
            }
            for g in gene_data
        ]
        await db.execute(insert(DegGene), deg_records)

        logger.info(
            "[demo_seed] Inserted %d DegGene rows for %s", len(deg_records), comp_name
        )

        # ---- 2e. Bulk-insert EnrichmentPathway rows
        pathway_records = [
            {
                "id": uuid4(),
                "dataset_id": dataset_id,
                "comparison_name": comp_name,
                "pathway_id": p["pathway_id"],
                "pathway_name": p["pathway_name"],
                "category": p["category"],
                "description": p.get("description"),
                "gene_count": p["gene_count"],
                "pvalue": p["pvalue"],
                "padj": p["padj"],
                "gene_ratio": p["gene_ratio"],
                "bg_ratio": p["bg_ratio"],
                "genes": p["genes"],
                "regulation": p["regulation"],
            }
            for p in pathway_data
        ]
        await db.execute(insert(EnrichmentPathway), pathway_records)

        logger.info(
            "[demo_seed] Inserted %d pathways for %s", len(pathway_records), comp_name
        )

    await db.commit()
    await db.refresh(project)

    logger.info("[demo_seed] Done. Project: %s", project.id)
    return project
