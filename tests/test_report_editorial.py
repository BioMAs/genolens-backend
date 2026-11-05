"""
OmicSkin editorial report — HTML rendering on a fixture (no DB), confidentiality of the claim
referential, page structure, AI labels, and the Chromium PDF (skipped without Chromium).

Fixture: the mock nova of the validated report (tests/fixtures/cosmomics_nova_mock.json) and a
synthetic comparison "AL-7" versus "Vehicle".
"""

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from app.services.cosmetics_taxonomy import normalize_pathway_id
from app.services.cosmomics_nova import CLAIM_SUBGROUPS_PATH
from app.services.cosmomics_nova_svg import render_nova_svg
from app.services.omicskin_score import ALL_CLAIMS, score_claims
from app.services.report_editorial_content import (
    EditorialInputs,
    build_editorial_context,
    cap_text,
    looks_english,
    md_to_html,
)
from app.services.report_editorial_service import (
    build_figures,
    chromium_available,
    html_to_pdf,
    render_html,
    split_comparison,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cosmomics_nova_mock.json"
ID_RE = re.compile(r"GO:\d{7}|R-HSA-\d+|CL:\d+|hsa\d{5}")

ENGLISH_AI = (
    "At this dose the active induces the cornified-envelope genes and the ceramide synthesis "
    "enzymes, which is consistent with a reinforcement of the barrier. The repression of the "
    "pro-inflammatory cytokines suggests a lowered inflammatory tone in the tissue."
)
FRENCH_AI = (
    "Le traitement induit les gènes de l'enveloppe cornée et des céramides, ce qui est "
    "cohérent avec un renforcement de la barrière. La répression des cytokines est nette dans "
    "les kératinocytes et pour les voies de l'inflammation."
)
NOVA_AI = (
    "### Barrier\nThe **barrier** programme is induced, with *coherent* evidence.\n"
    "- Soothing is supported."
)


def _synthetic_comparison():
    rng = np.random.default_rng(1)
    degs = []
    for i in range(40):
        degs.append(
            dict(
                gene_id=f"ENSG{i:011d}",
                gene_name=f"UPG{i}",
                log_fc=float(1 + 2 * rng.random()),
                padj=float(10 ** -rng.uniform(2, 8)),
                regulation="UP",
            )
        )
    for i in range(30):
        degs.append(
            dict(
                gene_id=f"ENSG{100 + i:011d}",
                gene_name=f"DNG{i}",
                log_fc=float(-1 - 2 * rng.random()),
                padj=float(10 ** -rng.uniform(2, 7)),
                regulation="DOWN",
            )
        )
    for i in range(300):
        degs.append(
            dict(
                gene_id=f"ENSG{1000 + i:011d}",
                gene_name=f"NSG{i}",
                log_fc=float(rng.normal(0, 0.3)),
                padj=float(rng.uniform(0.06, 1)),
                regulation="NS",
            )
        )
    samples = [f"S{i}" for i in range(1, 9)]
    conditions = {s: ("Vehicle" if i < 4 else "AL-7") for i, s in enumerate(samples)}
    vst = pd.DataFrame({"gene_id": [g["gene_id"] for g in degs]})
    for s in samples:
        shift = np.array([g["log_fc"] for g in degs]) * (1 if conditions[s] == "AL-7" else 0)
        vst[s] = 8 + shift + rng.normal(0, 0.2, len(degs))
    buf = io.BytesIO()
    vst.to_parquet(buf)
    pca = {
        "data": [
            {
                "sample": s,
                "x": float((1 if conditions[s] == "AL-7" else -1) * 10 + rng.normal()),
                "y": float(rng.normal()),
            }
            for s in samples
        ],
        "explained_variance": [0.54, 0.14],
    }
    qc_rows = [
        {
            "sample": s,
            "library_size": int(4.3e6 + i * 1e5),
            "detected_genes": 14100 + i * 90,
            "condition": conditions[s],
        }
        for i, s in enumerate(samples)
    ]
    enr_up = [
        {
            "pathway_id": f"GO:{i:07d}",
            "pathway_name": f"Up-regulated programme {i}",
            "category": "GO:BP",
            "padj": 10 ** -(6 - i * 0.3),
            "gene_count": 10 + i,
            "regulation": "UP",
        }
        for i in range(12)
    ]
    enr_down = [
        {
            "pathway_id": f"R-HSA-{i}",
            "pathway_name": f"Down-regulated programme {i}",
            "category": "Reactome",
            "padj": 10 ** -(5 - i * 0.3),
            "gene_count": 8 + i,
            "regulation": "DOWN",
        }
        for i in range(9)
    ]
    return degs, buf.getvalue(), pca, qc_rows, conditions, enr_up, enr_down


def _inputs(**overrides) -> EditorialInputs:
    nova = json.loads(FIXTURE.read_text())
    degs, vst, pca, qc_rows, conditions, enr_up, enr_down = _synthetic_comparison()
    term_genes = {}
    for claim in nova["claims"]:
        for sg in claim["subgroups"][:2]:
            for t in sg["terms"]:
                if t["effect"] > 0:
                    term_genes[(normalize_pathway_id(t["id"]), t["obs"])] = {"UPG1", "UPG2", "DNG3", "UPG7"}
    params = {"species": "human", "fdr": 0.05, "min_log2fc": 0.585}
    figures = build_figures(
        degs, enr_up, enr_down, {"pca": pca, "vst": vst}, conditions, "AL-7", "Vehicle", params
    )
    base = dict(
        comparison_name="AL-7_vs_Vehicle",
        test="AL-7",
        control="Vehicle",
        project_name="OmicSkin test study",
        degs=degs,
        n_up=40,
        n_down=30,
        enr_up=enr_up,
        enr_down=enr_down,
        qc_rows=qc_rows,
        sample_conditions=conditions,
        params=params,
        has_analysis=True,
        analysis_label="Test analysis (12345678)",
        nova=nova,
        scores=score_claims(nova),
        term_genes=term_genes,
        figures=figures,
        nova_light=render_nova_svg(nova, "light", id_prefix="nova-light", center_label="AL-7"),
        nova_dark=render_nova_svg(nova, "dark", id_prefix="nova-dark", center_label="AL-7"),
        comparison_ai=ENGLISH_AI,
        comparison_ai_model="google/gemma-4-E4B-it",
        nova_ai=NOVA_AI,
        nova_ai_model="google/gemma-4-E4B-it",
        cover_info={"client_ref": "SLC-TEST", "checked_by": "J. Doe", "prepared_by": "A. B."},
        referential_version="1",
        app_version="1.0.0",
    )
    base.update(overrides)
    return EditorialInputs(**base)


@pytest.fixture(scope="module")
def inputs():
    return _inputs()


@pytest.fixture(scope="module")
def context(inputs):
    return build_editorial_context(inputs)


@pytest.fixture(scope="module")
def html(context):
    return render_html(context)


# ── Confidentiality of the claim referential ───────────────────────────────


def test_referential_table_never_reaches_the_html(html, inputs):
    doc = yaml.safe_load(Path(CLAIM_SUBGROUPS_PATH).read_text(encoding="utf-8"))
    # The YAML keys (a dump of the table would carry them) and its header never appear.
    row_keys = {k for row in doc["terms"] for k in row}
    for key in row_keys | {"terms"}:
        assert f"{key}:" not in html, key
    assert "expected_direction" not in html
    assert "CONFIDENTIAL" not in html.upper()
    # Every ontology id printed belongs to a term observed in this comparison's nova:
    # no row of the referential leaks beyond what the data show.
    observed = {t["id"] for c in inputs.nova["claims"] for s in c["subgroups"] for t in s["terms"]}
    printed = set(ID_RE.findall(html))
    assert printed, "the term tables should print the observed ids"
    assert printed <= observed
    # Referential ids in their printed form (Reactome "HSA-…" is shown as "R-HSA-…").
    ref_ids = (str(r["term_id"]) for r in doc["terms"])
    shown = {("R-" + t if t.startswith("HSA-") else t) for t in ref_ids}
    leaked = [
        t
        for t in shown - observed
        if ID_RE.fullmatch(t) and re.search(re.escape(t) + r"(?!\d)", html)
    ]
    assert not leaked


def test_only_the_referential_version_is_printed(html):
    assert "claim_subgroups v1" in html


# ── Page structure ─────────────────────────────────────────────────────────


def test_each_strong_claim_has_its_page(html, inputs):
    strong = [s.name for s in inputs.scores if s.scored and s.score >= 75]
    assert strong == ["Barrier", "Soothing", "Firming", "Regeneration", "Anti-ageing"]
    for name in strong:
        assert html.count(f'data-claim="{name}" data-kind="claim"') == 1, name


def test_moderate_claims_sit_on_the_reservations_page(html):
    assert html.count('data-claim="Antioxidant" data-kind="reservation"') == 1


def test_every_claim_appears_exactly_once(html):
    for name in ALL_CLAIMS:
        assert html.count(f'data-claim="{name}"') == 1, name
    for name in ("Anti-wrinkles", "Energising", "Healing"):
        assert f'data-claim="{name}" data-kind="no-evidence"' in html


def test_page_sequence_and_numbering(context, html):
    kinds = [p["kind"] for p in context["pages"]]
    assert kinds == ["cover", "verdict", "nova"] + ["claim"] * 5 + [
        "reservations",
        "science_design",
        "science_deg",
        "science_enrichment",
        "science_heatmap",
        "method",
    ]
    total = context["total"]
    assert total == 14
    assert f"02 / {total}" in html and f"{total:02d} / {total}" in html


def test_verdict_headline_and_ladder(context):
    verdict = context["pages"][1]
    assert verdict["headline"] == (
        "Five claims strongly supported, one moderately, three not evidenced."
    )
    assert [r["name"] for r in verdict["ladder"]][:6] == [
        "Barrier",
        "Soothing",
        "Firming",
        "Regeneration",
        "Anti-ageing",
        "Antioxidant",
    ]
    assert {s["value"] for s in verdict["stats"]} >= {"6 / 9", "5"}


def test_no_skin_stack_and_no_dose_block_by_default(html):
    assert "Skin Stack" not in html
    assert "Dose progression" not in html


def test_dose_block_only_with_a_dose_series():
    nova = json.loads(FIXTURE.read_text())
    weaker = json.loads(FIXTURE.read_text())
    for c in weaker["claims"]:
        c["best_nl"] = c["best_for_nl"] = c["best_nl"] / 3  # the score reads `best_for_nl`
    ctx = build_editorial_context(_inputs(dose_series=[("0.1 %", weaker), ("1 %", nova)]))
    page = render_html(ctx)
    assert "Dose progression" in page
    assert "checked for monotony across doses" in page


def test_claim_figures_and_genes(html):
    assert "UPG1" in html  # key gene chips from the claim's supporting terms
    assert html.count('class="ring"') >= 6
    assert "Fig. 1" in html and "Fig. 4" in html


# ── AI texts ────────────────────────────────────────────────────────────────


def test_reviewer_label_when_checked(html):
    assert "AI-generated interpretation — reviewed by J. Doe" in html


def test_reviewer_label_without_reviewer():
    page = render_html(build_editorial_context(_inputs(cover_info={})))
    assert "AI-generated, not yet reviewed" in page
    assert "reviewed by" not in page


def test_french_interpretation_is_not_printed():
    page = render_html(build_editorial_context(_inputs(comparison_ai=FRENCH_AI)))
    assert "No English AI interpretation is available" in page
    assert "enveloppe cornée" not in page


def test_english_interpretation_is_printed(html):
    assert "cornified-envelope genes" in html


def test_llm_markup_is_escaped():
    page = render_html(
        build_editorial_context(_inputs(comparison_ai=ENGLISH_AI + " <script>alert(1)</script>"))
    )
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_markdown_subset():
    out = str(md_to_html(NOVA_AI))
    assert "<h4>Barrier</h4>" in out and "<b>barrier</b>" in out and "<i>coherent</i>" in out
    assert "<ul><li>Soothing is supported.</li></ul>" in out


def test_language_detection():
    assert looks_english(ENGLISH_AI)
    assert not looks_english(FRENCH_AI)
    assert not looks_english("")


def test_cap_text_cuts_on_a_boundary():
    text = "First paragraph.\n\n" + "Second paragraph is long. " * 50
    capped, truncated = cap_text(text, 120)
    assert truncated and capped.startswith("First paragraph.") and capped.endswith("…")


def test_split_comparison():
    assert split_comparison("KO_vs_WT") == ("KO", "WT")
    assert split_comparison("treated vs control") == ("treated", "control")
    assert split_comparison("single") == ("single", "")


def test_sparse_comparison_still_renders():
    """No claim, no enrichment, no PCA, no AI: the report keeps a coherent page set."""
    empty = {"contrast": "A_vs_B", "claims": [], "unlinked": []}
    ctx = build_editorial_context(
        _inputs(
            nova=empty,
            scores=score_claims(empty),
            enr_up=[],
            enr_down=[],
            comparison_ai=None,
            nova_ai=None,
            figures={
                "pca": None,
                "volcano": None,
                "dot_up": None,
                "dot_down": None,
                "heatmap": None,
            },
            nova_light=render_nova_svg(empty, "light", id_prefix="l"),
            nova_dark=render_nova_svg(empty, "dark", id_prefix="d"),
        )
    )
    kinds = [p["kind"] for p in ctx["pages"]]
    assert kinds == [
        "cover",
        "verdict",
        "nova",
        "reservations",
        "science_design",
        "science_deg",
        "method",
    ]
    page = render_html(ctx)
    assert "No claim shows significant evidence" in page
    assert page.count('data-kind="no-evidence"') == 9


# ── PDF (Chromium) ──────────────────────────────────────────────────────────


@pytest.mark.skipif(not chromium_available(), reason="Playwright Chromium is not installed")
async def test_pdf_is_produced(tmp_path, context, html):
    pdf, overflow = await html_to_pdf(html, str(tmp_path))
    assert pdf[:5] == b"%PDF-" and len(pdf) > 50_000
    assert len(re.findall(rb"/Type\s*/Page(?!s)", pdf)) == context["total"]
    assert overflow == [], f"content clipped on pages {overflow}"


# ── Worker lifecycle ────────────────────────────────────────────────────────


def _session_with(job):
    from contextlib import asynccontextmanager

    db = MagicMock()
    db.get = AsyncMock(return_value=job)
    db.commit = AsyncMock()

    @asynccontextmanager
    async def _fresh():
        yield db

    return db, _fresh


async def test_job_lifecycle_done():
    from app.models.report_job import ReportJobStatus
    from app.worker.tasks import report_task

    job = SimpleNamespace(id="j1", dataset_id="d1", status=None, pdf_path=None, error_message=None)
    db, fresh = _session_with(job)
    upload = AsyncMock()
    with (
        patch.object(report_task, "_fresh_session", fresh),
        patch("app.services.storage.storage_service.upload_file", new=upload),
    ):
        out = await report_task._run_comparison_job(
            "00000000-0000-0000-0000-000000000001", AsyncMock(return_value=b"%PDF-1.7")
        )
    assert out["status"] == "done" and job.status == ReportJobStatus.DONE
    assert job.pdf_path == "reports/comparisons/d1/00000000-0000-0000-0000-000000000001.pdf"
    upload.assert_awaited_once()


async def test_job_lifecycle_failed():
    from app.models.report_job import ReportJobStatus
    from app.worker.tasks import report_task

    job = SimpleNamespace(id="j1", dataset_id="d1", status=None, pdf_path=None, error_message=None)
    db, fresh = _session_with(job)
    with patch.object(report_task, "_fresh_session", fresh):
        out = await report_task._run_comparison_job(
            "00000000-0000-0000-0000-000000000001", AsyncMock(side_effect=RuntimeError("boom"))
        )
    assert out["status"] == "failed" and job.status == ReportJobStatus.FAILED
    assert job.error_message == "boom"


def test_editorial_task_runs_on_the_chromium_worker_queue():
    from app.worker.tasks.report_task import (
        generate_comparison_report,
        generate_comparison_report_editorial,
    )

    assert generate_comparison_report_editorial.queue == "default"
    assert generate_comparison_report.queue == "r_analysis"
