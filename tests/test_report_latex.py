"""
Unit tests for the SciLicium LaTeX report renderer (no pdflatex compilation).
"""
import logging
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services import report_latex_service as rl

needs_pdflatex = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")

# The variables the main template reads.
MAIN_TEMPLATE_KEYS = [
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


def test_tex_escape_specials():
    assert rl.tex_escape("a_b & c% 50 #x $y") == r"a\_b \& c\% 50 \#x \$y"
    assert rl.tex_escape(None) == ""


def test_tex_escape_maps_unicode_symbols():
    out = rl.tex_escape("p ≤ 0.05, ≥ 2, ± 1, α → β")
    for command in (
        r"\ensuremath{\leq}", r"\ensuremath{\geq}", r"\ensuremath{\pm}",
        r"\ensuremath{\alpha}", r"\ensuremath{\rightarrow}", r"\ensuremath{\beta}",
    ):
        assert command in out
    assert all(rl._printable(ch) for ch in out)


def test_tex_escape_replaces_an_unprintable_character(caplog):
    with caplog.at_level(logging.WARNING, logger=rl.logger.name):
        assert rl.tex_escape("ok 🙂") == "ok ?"
    assert "U+1F642" in caplog.text


def test_the_warning_names_the_field_and_never_prints_the_text(caplog):
    with caplog.at_level(logging.WARNING, logger=rl.logger.name):
        rl.plaintext_to_latex("a confidential finding 🙂 and 🙃", field="conclusion")
    assert "conclusion" in caplog.text and "U+1F642" in caplog.text and "U+1F643" in caplog.text
    assert "confidential" not in caplog.text


def test_a_printable_text_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=rl.logger.name):
        rl.tex_escape("p ≤ 0.05 — µM, 20 °C, Δ, é, ñ, Ł")
    assert caplog.records == []


def test_the_printable_set():
    assert all(rl._printable(chr(c)) for c in range(0x20, 0x7F))
    assert all(rl._printable(ch) for ch in "\t\n\r")
    # Control characters are an error for pdflatex, whatever the encoding.
    assert not any(rl._printable(chr(c)) for c in [*range(0, 9), 0x0B, *range(0x0E, 0x20), 0x7F])
    assert not any(rl._printable(chr(c)) for c in range(0x80, 0xA0))
    assert not rl._printable("🙂") and not rl._printable("\ud83d") and not rl._printable("\u2264")
    assert rl._printable("é") and rl._printable("Ł")


def test_every_table_entry_is_a_pure_ascii_command():
    for char, command in rl.UNICODE_TO_LATEX.items():
        assert all(ord(c) < 0x80 for c in command), char
        assert rl.latex_safe_unicode(char) == command


def test_the_punctuation_the_conclusion_always_mapped_is_unchanged():
    assert rl.plaintext_to_latex("it’s “fine” – a—b… 5 × 3 − 1 20°\u00a0C\u202fx") == (
        "it's ``fine'' -- a---b\\ldots{} 5 \\texttimes{} 3 - 1 20\\textdegree{} C x"
    )


def test_plaintext_to_latex_keeps_its_paragraphs_and_maps_symbols():
    out = rl.plaintext_to_latex("p ≤ 0.05\nsoft wrap\n\n\n#2 & α")
    assert out == "p \\ensuremath{\\leq} 0.05 soft wrap\n\n\\#2 \\& \\ensuremath{\\alpha}"


def test_the_jinja_filter_maps_symbols_too():
    template = rl.report_latex_service.env.from_string(
        r"\JINJAVAR{x|tex_escape} \JINJAVAR{xs|map('tex_escape')|join(', ')}"
    )
    assert template.render(x="a ≤ b_c", xs=["α", "10%"]) == (
        r"a \ensuremath{\leq} b\_c \ensuremath{\alpha}, 10\%"
    )


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
    # No thousands comma: siunitx's S column reads "12,000" as 12.000 (twelve) and prints it so.
    assert "1.23M" in qc and "12000" in qc and "12,000" not in qc


def test_empty_generators():
    assert "No differentially expressed" in rl.generate_deg_table([])
    assert "No enriched terms" in rl.generate_enrichment_table([], "x")
    assert "QC data not available" in rl.generate_qc_table([], "x")


def test_main_template_renders():
    """The vendored SciLicium .tex renders with the Jinja LaTeX delimiters."""
    svc = rl.report_latex_service
    out = svc.env.get_template(rl.MAIN_TEMPLATE).render(**{k: "" for k in MAIN_TEMPLATE_KEYS})
    assert "documentclass" in out and "begin{document}" in out


def test_each_text_of_the_report_is_named_in_the_warning(caplog):
    data = {
        "project": SimpleNamespace(name="Proj"),
        "analysis": SimpleNamespace(id=uuid4(), name="An"),
        "params": {},
        "comparison_name": "A_vs_B",
        "dataset_id": uuid4(),
        "comparisons": [_comparison(ai_interpretation="note 🙂", ai_model="m")],
        "qc_rows": [],
        "pca_fig": None,
        "conclusion": "end 🙂",
        "materials_methods": "methods 🙂",
        "cover_info": {"prepared_by": "someone 🙂"},
        "branding": {"institute_name": "Lab 🙂"},
    }
    with caplog.at_level(logging.WARNING, logger=rl.logger.name):
        tex = rl.report_latex_service.assemble_tex(data)
    for field in (
        "conclusion", "materials_methods", "ai_text", "cover.prepared_by", "branding.institute_name",
    ):
        assert f"'{field}'" in caplog.text, field
    assert "🙂" not in tex


def _minimal(body: str) -> str:
    return "\\documentclass{article}\n\\begin{document}\n" + body + "\n\\end{document}\n"


@needs_pdflatex
def test_every_escaped_symbol_compiles(tmp_path):
    """Whatever the sanitizer lets through compiles in the report, and whatever it refuses
    (emoji, control characters, a lone surrogate) is replaced by "?" and compiles too."""
    kept = "".join(chr(c) for c in range(0xA0, 0x180) if rl._printable(chr(c)))
    refused = "\U0001F642 \x00 \x07 \x1b \x7f \x85 \ud83d \ufffe"
    ascii_printable = "".join(chr(c) for c in range(0x20, 0x7F))
    probe = "\n\n".join([" ".join(rl.UNICODE_TO_LATEX), kept, ascii_printable, refused])
    data = {k: "" for k in MAIN_TEMPLATE_KEYS}
    data["jj_conclusion"] = rl.plaintext_to_latex(probe, field="conclusion")
    tex = rl.report_latex_service.env.get_template(rl.MAIN_TEMPLATE).render(**data)
    assert rl.report_latex_service.compile_pdf(tex, str(tmp_path)).startswith(b"%PDF")


@needs_pdflatex
def test_a_symbol_in_a_cover_field_compiles(tmp_path):
    cover = rl.tex_escape("α ≤ β → γ · 5 µM", field="cover.project_name")
    data = {k: "" for k in MAIN_TEMPLATE_KEYS}
    for key in (
        "jj_file_title", "jj_report_subtitle", "jj_report_author", "jj_sponsor_name",
        "jj_report_project", "jj_report_prepared_by",
    ):
        data[key] = cover
    tex = rl.report_latex_service.env.get_template(rl.MAIN_TEMPLATE).render(**data)
    assert rl.report_latex_service.compile_pdf(tex, str(tmp_path)).startswith(b"%PDF")


@needs_pdflatex
def test_pdflatex_failure_message_stays_short(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger=rl.logger.name):
        with pytest.raises(rl.LatexCompileError) as caught:
            rl.report_latex_service.compile_pdf(_minimal(r"\undefinedcommandxyz"), str(tmp_path))
    message = str(caught.value)
    assert message == (
        "The LaTeX report could not be compiled (pdflatex pass 1); the details are in the worker log."
    )
    assert "report.tex" not in message and "/tmp" not in message and "l." not in message
    assert len(message) < 200
    # The end of the pdflatex log is in the worker log, not in the message.
    assert "undefinedcommandxyz" in caplog.text


def test_a_failed_pass_logs_the_end_of_the_pdflatex_log_and_raises_the_short_message(tmp_path, caplog):
    (tmp_path / "report.log").write_text("! Undefined control sequence.\nl.3 \\secretmacro")
    failed = MagicMock(returncode=1, stdout="", stderr="")
    with patch.object(rl.subprocess, "run", return_value=failed):
        with caplog.at_level(logging.ERROR, logger=rl.logger.name):
            with pytest.raises(rl.LatexCompileError) as caught:
                rl.report_latex_service.compile_pdf(_minimal("x"), str(tmp_path))
    assert "secretmacro" not in str(caught.value) and "pass 1" in str(caught.value)
    assert "secretmacro" in caplog.text and "pass 1" in caplog.text


def test_no_pdf_after_a_clean_exit_is_a_short_error_too(tmp_path, caplog):
    ok = MagicMock(returncode=0, stdout="all fine", stderr="")
    with patch.object(rl.subprocess, "run", return_value=ok):
        with caplog.at_level(logging.ERROR, logger=rl.logger.name):
            with pytest.raises(rl.LatexCompileError, match="pdflatex pass 2"):
                rl.report_latex_service.compile_pdf(_minimal("x"), str(tmp_path))
    assert "all fine" in caplog.text


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
