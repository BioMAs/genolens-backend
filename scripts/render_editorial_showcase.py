#!/usr/bin/env python3
"""
Render the OmicSkin editorial report on the demo data, offline (no database): a layout showcase.

  * Science pages — computed from the demo inputs of the analysis wizard (demo_count_matrix.csv,
    demo_samplesheet.csv: library sizes, detected genes, PCA and heatmap on log2 CPM) and from the
    synthetic DEG / enrichment tables of the demo seed (app/services/demo_seed_service.py).
  * Claim pages — the mock CosmOmics nova of the validated reference
    (tests/fixtures/cosmomics_nova_mock.json): the demo matrix has 55 genes, too few to carry
    claim evidence of its own. No AI text is invented.

Every page is marked "Illustrative data".

Usage (worker image — Chromium installed), from backend/ with the repository mounted:
    python scripts/render_editorial_showcase.py \\
        --counts ../demo_count_matrix.csv --samples ../demo_samplesheet.csv \\
        --out ../docs/omicskin-report-editorial/demo_editorial_showcase.pdf
"""
import argparse
import asyncio
import io
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.core.config import settings  # noqa: E402
from app.services.cosmomics_nova import default_claim_subgroups  # noqa: E402
from app.services.cosmomics_nova_svg import render_nova_svg  # noqa: E402
from app.services.data_processor import data_processor  # noqa: E402
from app.services.demo_seed_service import (  # noqa: E402
    _TREATMENT_VS_CONTROL_GENES,
    _TREATMENT_VS_CONTROL_PATHWAYS,
    _regulation,
)
from app.services.omicskin_score import score_claims  # noqa: E402
from app.services.report_editorial_content import (  # noqa: E402
    EditorialInputs,
    build_editorial_context,
)
from app.services.report_editorial_service import (  # noqa: E402
    LOGO_PATH,
    build_figures,
    data_uri,
    html_to_pdf,
    render_html,
)

COMPARISON = "Treatment_vs_Control"
TEST, CONTROL = "Treatment", "Control"
NOVA_FIXTURE = BACKEND / "tests" / "fixtures" / "cosmomics_nova_mock.json"


def log2_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    values = counts.set_index("gene_id").apply(pd.to_numeric, errors="coerce").fillna(0)
    cpm = values / values.sum(axis=0) * 1e6
    return np.log2(cpm + 1).reset_index()


def to_parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


async def build(counts_path: Path, samples_path: Path) -> EditorialInputs:
    counts = pd.read_csv(counts_path)
    sheet = pd.read_csv(samples_path)
    sample_conditions = {str(r.sample_id): str(r.condition) for r in sheet.itertuples()}
    samples = [c for c in counts.columns if c != "gene_id"]

    qc_rows = [
        {
            "sample": s,
            "library_size": int(pd.to_numeric(counts[s], errors="coerce").fillna(0).sum()),
            "detected_genes": int((pd.to_numeric(counts[s], errors="coerce").fillna(0) > 0).sum()),
            "condition": sample_conditions.get(s, "Unknown"),
        }
        for s in samples
    ]

    expression_bytes = to_parquet(log2_cpm(counts))
    pca = await data_processor.calculate_pca(expression_bytes, n_components=2)

    degs = [
        {
            "gene_id": g[0],
            "gene_name": g[1],
            "log_fc": g[3],
            "padj": g[4],
            "pvalue": g[4] * 10.0,
            "regulation": _regulation(g[3], g[4]),
        }
        for g in _TREATMENT_VS_CONTROL_GENES
    ]
    enr_up = [p for p in _TREATMENT_VS_CONTROL_PATHWAYS if p["regulation"] == "UP"]
    enr_down = [p for p in _TREATMENT_VS_CONTROL_PATHWAYS if p["regulation"] == "DOWN"]

    nova = json.loads(NOVA_FIXTURE.read_text(encoding="utf-8"))
    params = {"species": "human"}
    figures = build_figures(
        degs,
        enr_up,
        enr_down,
        {"pca": pca, "vst": expression_bytes},
        sample_conditions,
        TEST,
        CONTROL,
        params,
    )
    return EditorialInputs(
        comparison_name=COMPARISON,
        test=TEST,
        control=CONTROL,
        project_name="GenoLens demo dataset",
        degs=degs,
        n_up=sum(1 for g in degs if g["regulation"] == "UP"),
        n_down=sum(1 for g in degs if g["regulation"] == "DOWN"),
        enr_up=enr_up,
        enr_down=enr_down,
        qc_rows=qc_rows,
        sample_conditions=sample_conditions,
        params=params,
        has_analysis=False,
        analysis_label=None,
        nova=nova,
        scores=score_claims(nova),
        term_genes={},
        figures=figures,
        nova_light=render_nova_svg(nova, "light", id_prefix="nova-light", center_label="Active"),
        nova_dark=render_nova_svg(nova, "dark", id_prefix="nova-dark", center_label="Active"),
        cover_info={"client_ref": "DEMO", "sponsor_name": "Illustrative example"},
        logo_data_uri=data_uri(LOGO_PATH),
        referential_version=default_claim_subgroups().version,
        app_version=settings.APP_VERSION,
        generated_at=datetime.now(timezone.utc),
        illustrative=True,
    )


async def main(args: argparse.Namespace) -> None:
    inputs = await build(Path(args.counts), Path(args.samples))
    context = build_editorial_context(inputs)
    with tempfile.TemporaryDirectory(prefix="showcase_") as work_dir:
        pdf, overflow = await html_to_pdf(render_html(context), work_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pdf)
    print(
        f"{out}: {context['total']} pages, {len(pdf):,} bytes"
        + (f" — content clipped on pages {overflow}" if overflow else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--counts", required=True, help="demo_count_matrix.csv")
    parser.add_argument("--samples", required=True, help="demo_samplesheet.csv")
    parser.add_argument("--out", required=True, help="output PDF")
    asyncio.run(main(parser.parse_args()))
