"""
Unit tests for the SciLicium LaTeX report renderer (no pdflatex compilation).
"""
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
