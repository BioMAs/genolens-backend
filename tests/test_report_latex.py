"""
Unit tests for the SciLicium LaTeX report renderer (no pdflatex compilation).
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services import report_latex_service as rl


def test_tex_escape_specials():
    assert rl.tex_escape("a_b & c% 50 #x $y") == r"a\_b \& c\% 50 \#x \$y"
    assert rl.tex_escape(None) == ""


def test_sanitize_and_condition_display():
    assert rl.sanitize_condition_name("BTSC73_MCT1_CRISPR") == "BTSC73 MCT1 CRISPR"
    assert rl.format_condition_display("KO_x", "test") == "KO x (test)"


def test_deg_table_generator():
    out = rl.generate_deg_table([
        {"gene_name": "TP53", "log_fc": 2.5, "pvalue": 1e-9, "padj": 1e-6, "regulation": "UP"},
        {"gene_name": "BCL2", "log_fc": -1.8, "pvalue": 1e-4, "padj": 2e-3, "regulation": "DOWN"},
    ])
    assert r"\begin{longtable}" in out and "TP53" in out and r"\end{longtable}" in out


def test_enrichment_and_qc_generators():
    enr = rl.generate_enrichment_table(
        [{"pathway_name": "cell cycle (GO:0007049)", "category": "GO:BP", "pvalue": 1e-8, "padj": 1e-5}],
        "Up-regulated genes",
    )
    assert "cell cycle" in enr and "GO:BP" in enr
    qc = rl.generate_qc_table(
        [{"sample": "s1", "library_size": 1234567, "detected_genes": 12000, "condition": "CTL"}],
        "Proj",
    )
    assert "1.23M" in qc and "12,000" in qc


def test_empty_generators():
    assert "No differentially expressed" in rl.generate_deg_table([])
    assert "No enriched terms" in rl.generate_enrichment_table([], "x")
    assert "QC data not available" in rl.generate_qc_table([], "x")


def test_main_template_renders():
    """The vendored SciLicium .tex renders with the Jinja LaTeX delimiters."""
    svc = rl.report_latex_service
    keys = [
        "jj_file_title", "jj_report_title", "jj_report_subtitle", "jj_report_version",
        "jj_report_number", "jj_report_author", "jj_report_clientref", "jj_sponsor_name",
        "jj_sponsor_contact", "jj_sponsor_email", "jj_sponsor_address", "jj_test_facility_name",
        "jj_test_facility_contact", "jj_test_facility_email", "jj_test_facility_address",
        "jj_test_site_name", "jj_test_site_contact", "jj_test_site_email", "jj_test_site_address",
        "jj_report_project", "jj_report_prepared_by", "jj_report_checked_by", "jj_report_approved_by",
        "jj_analysis_date", "jj_pipeline_version", "jj_genome_version", "jj_materials_methods",
        "jj_executive_summary", "jj_conclusion", "jj_projects_results", "jj_appendix",
        "ref_appendix_soft_versions",
    ]
    out = svc.env.get_template(rl.MAIN_TEMPLATE).render(**{k: "" for k in keys})
    assert "documentclass" in out and "begin{document}" in out


# ── AI texts: label, staleness ──────────────────────────────────────────────

def _comparison(**extra) -> dict:
    return {
        "comp_id": "A_vs_B",
        "safe_id": "A-vs-B",
        "test": "A",
        "control": "B",
        "n_up": 1,
        "n_down": 0,
        "degs": [],
        "enr_up": [],
        "enr_down": [],
        **extra,
    }


def test_latex_ai_box_names_model_and_date():
    day = datetime(2026, 11, 5, 9, 30, tzinfo=timezone.utc)
    section = rl.build_comparison_section(
        _comparison(
            ai_interpretation="Reading A.", ai_model="org/model_v1", ai_generated_at=day
        )
    )
    assert r"AI-generated text (model org/model\_v1, generated 2026-11-05)" in section
    assert "AI-generated interpretation" not in section
    assert "Reading A." in section
    # No date on record: the label names the model alone.
    undated = rl.build_comparison_section(
        _comparison(ai_interpretation="Reading A.", ai_model="org/model_v1")
    )
    assert r"AI-generated text (model org/model\_v1)" in undated


def test_latex_prints_no_ai_box_without_a_text():
    section = rl.build_comparison_section(_comparison(ai_stale=True))
    assert "AI-generated" not in section and "Biological Summary" not in section


def _ai_row(**counts):
    return SimpleNamespace(
        dataset_id=uuid4(),
        comparison_name="A_vs_B",
        interpretation="Reading A.",
        model="org/model_v1",
        created_at=datetime(2026, 11, 5, 9, 30, tzinfo=timezone.utc),
        deg_up=counts.get("deg_up", 10),
        deg_down=counts.get("deg_down", 5),
        pathways_count=counts.get("pathways_count", 3),
    )


async def _built_with(ai, context):
    """`_build_comparison` on a DEG dataset whose cached comparison text is `ai`, with the
    counts the text would be written from today given by `context` (None: no DEG at all)."""
    deg_ds = SimpleNamespace(id=uuid4(), dataset_metadata={}, name="DEG")
    gene_rows = MagicMock()
    gene_rows.scalars.return_value.all.return_value = []
    db = MagicMock()
    db.execute = AsyncMock(return_value=gene_rows)
    db.scalar = AsyncMock(return_value=ai)
    with patch(
        "app.services.ai_interpreter.build_comparison_ai_context",
        new=AsyncMock(return_value=context),
    ):
        return await rl.report_latex_service._build_comparison(
            db, deg_ds, [], {}, "/tmp", False,
            comparison_name="A_vs_B", make_figures=False, include_ai=True,
        )


def _context(up=10, down=5, pathways=3):
    return {
        "deg_summary": {"up_count": up, "down_count": down, "total": up + down},
        "top_pathways": [{"pathway_name": f"Term {i}"} for i in range(pathways)],
        "top_genes": [],
    }


async def test_comparison_ai_text_is_current_while_the_counts_stand():
    comp = await _built_with(_ai_row(), _context())
    assert comp["ai_interpretation"] == "Reading A."
    assert comp["ai_model"] == "org/model_v1"
    assert comp["ai_generated_at"] == datetime(2026, 11, 5, 9, 30, tzinfo=timezone.utc)
    assert not comp.get("ai_stale")


@pytest.mark.parametrize(
    "counts",
    [
        {"deg_up": 11},
        {"deg_down": 4},
        {"pathways_count": 4},
    ],
)
async def test_comparison_ai_text_is_stale_when_the_deg_counts_changed(counts):
    """The text was written from deg_up / deg_down / pathways_count: another count, another data."""
    comp = await _built_with(_ai_row(**counts), _context())
    assert "ai_interpretation" not in comp
    assert comp["ai_stale"] is True
    assert "AI-generated" not in rl.build_comparison_section(comp)


async def test_comparison_ai_text_is_stale_when_the_comparison_has_no_deg_now():
    comp = await _built_with(_ai_row(), None)
    assert "ai_interpretation" not in comp and comp["ai_stale"] is True


async def test_no_stored_comparison_text_is_not_a_stale_one():
    comp = await _built_with(None, _context())
    assert "ai_interpretation" not in comp and not comp.get("ai_stale")
