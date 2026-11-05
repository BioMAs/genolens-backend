"""
OmicSkin editorial report — HTML rendering on a fixture (no DB), confidentiality of the claim
referential, page structure, AI labels, and the Chromium PDF (skipped without Chromium).

Fixture: the mock nova of the validated report (tests/fixtures/cosmomics_nova_mock.json) and a
synthetic comparison "AL-7" versus "Vehicle".
"""

import html as html_lib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import yaml

from app.services.cosmetics_taxonomy import normalize_pathway_id
from app.services.cosmomics_nova import CLAIM_SUBGROUPS_PATH, TermHit
from app.services.cosmomics_nova_svg import render_nova_svg
from app.services.omicskin_score import ALL_CLAIMS, BAND_LABELS, score_claims
from app.services.report_editorial_content import (
    EditorialInputs,
    build_editorial_context,
    cap_text,
    enrichment_facts,
    enrichment_text,
    looks_english,
    md_to_html,
)
from app.services.report_editorial_figures import volcano_sampling
from app.services.report_editorial_service import (
    build_figures,
    chromium_available,
    hits_to_enrichment,
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
    # Descriptive, by state: no claim is "supported" or "evidenced" by the headline.
    assert verdict["headline"] == (
        "Six claims with evidence for and against, three not evaluated (n.e.)."
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


# ── What the report says is what was computed (D23b-7) ─────────────────────

OVERSTATED = (
    "proof of efficacy",
    "substantiation",
    "every pathway behind every claim",
    "converges",
    "proven",
    "validated",
    "confirmed",
)
P_SOURCE_GENOLENS = "padj (Benjamini–Hochberg, GenoLens enrichment)"
GENOLENS_PROVENANCE = {
    "dataset_id": "d-1",
    "comparison": "AL-7_vs_Vehicle",
    "enrichment_dataset_ids": ["e-1"],
    "analysis_id": "a-1",  # an analysis run in GenoLens, its enrichment on a dataset of its own
    "enrichment_source": "annodb",
    "enrichment_fdr": 0.05,
    "enrichment_databases": ["GO", "Reactome"],
    "enrichment_categories": None,
    "padj_columns": [],
    "source_versions": [],
    "referential_version": 1,
    "alpha": 0.05,
    "app_version": "1.0.0",
    "generated_at": "2026-11-05T09:30:00+00:00",
    "demo": False,
}


def _text(page: str) -> str:
    """The page as the reader sees it: tags dropped, entities decoded, one space between words."""
    plain = html_lib.unescape(re.sub(r"<[^>]+>", " ", str(page)))
    return " ".join(plain.replace("\u202f", " ").replace("\xa0", " ").split())


def _honest_nova(mutate=None, **provenance):
    """The mock nova with the facts a current payload carries (alpha, p_source, provenance)."""
    nova = json.loads(FIXTURE.read_text())
    nova.update(
        alpha=0.05,
        p_source=P_SOURCE_GENOLENS,
        not_evaluated=["Anti-wrinkles", "Energising", "Healing"],
        provenance={**GENOLENS_PROVENANCE, **provenance},
    )
    if mutate:
        mutate(nova)
    return nova


def _inputs_for(nova, **overrides) -> EditorialInputs:
    return _inputs(
        nova=nova,
        scores=score_claims(nova),
        nova_light=render_nova_svg(
            nova, "light", id_prefix="nova-light", center_label="AL-7", demo_mark=False
        ),
        nova_dark=render_nova_svg(
            nova, "dark", id_prefix="nova-dark", center_label="AL-7", demo_mark=False
        ),
        **overrides,
    )


def _context(nova=None, **overrides):
    return build_editorial_context(_inputs_for(nova or _honest_nova(), **overrides))


def _page(nova=None, **overrides) -> str:
    return render_html(_context(nova, **overrides))


def test_editorial_has_no_overstated_wording(html):
    for page in (html, _page(), _page(use_ai=False)):
        text = _text(page).lower()
        for phrase in OVERSTATED:
            assert phrase not in text, phrase


def test_cover_and_nova_page_describe_the_content():
    text = _text(_page())
    assert "Transcriptomic evidence review" in text and "Pathway evidence by claim" in text
    assert "The enriched terms of the referential, by claim" in text
    # The cover prints the engine version, not a document version nobody maintains.
    assert "Version 1.0" not in text and "GenoLens 1.0.0" in text


def test_scope_does_not_promise_a_dossier():
    scope = next(p["scope"] for p in _context()["pages"] if p["kind"] == "method")
    assert "does not by itself establish a clinical effect" in scope
    assert "substantiation" not in scope.lower()


def test_methods_follow_the_provenance_of_a_deposited_table():
    nova = _honest_nova(
        enrichment_source="uploaded",
        padj_columns=["qvalue"],
        enrichment_fdr=None,
        enrichment_databases=None,
    )
    nova["p_source"] = "adjusted p from uploaded tables (column qvalue)"
    text = _text(_page(nova))
    assert "column qvalue" in text
    assert "hypergeometric" not in text.lower() and "benjamini" not in text.lower()
    assert "kegg" not in text.lower() and "wikipathways" not in text.lower()
    # A deposited table whose column was not recorded says so, and still names no test.
    unrecorded = _honest_nova(
        enrichment_source="uploaded", padj_columns=[], enrichment_fdr=None, enrichment_databases=None
    )
    unrecorded["p_source"] = "adjusted p from uploaded tables (column not recorded)"
    text = _text(_page(unrecorded))
    assert "column not recorded" in text and "hypergeometric" not in text.lower()


def test_methods_print_the_recorded_enrichment_cutoff_never_a_default():
    params = {"species": "human", "fdr": 0.01, "min_log2fc": 0.585, "enrichment_fdr": 0.05}
    text = _text(_page(_honest_nova(enrichment_fdr=0.1), params=params))
    assert "adjusted p < 0.1" in text
    assert "adjusted p < 0.05" not in text and "≤ 0.05" not in text
    # Not recorded: the report says so, and the 0.05 of the analysis parameters is not printed.
    text = _text(_page(_honest_nova(enrichment_fdr=None), params=params))
    assert "cut-off not recorded" in text
    assert "adjusted p < 0.05" not in text and "≤ 0.05" not in text


def test_methods_list_only_the_databases_that_were_run():
    for databases, kegg in (
        (["GO", "KEGG"], True),
        (["GO", "Reactome"], False),
        ([], False),
        (None, False),
    ):
        text = _text(_page(_honest_nova(enrichment_databases=databases)))
        assert ("KEGG" in text) is kegg
        assert "WikiPathways" not in text and "MSigDB" not in text
    listed = _text(_page(_honest_nova(enrichment_databases=["GO", "WikiPathways"])))
    assert "GO, WikiPathways" in listed
    # An analysis that is known and picked nothing ran "all categories", and says no more than
    # that; an unknown pick (the analysis is gone) is not "all categories" (R-F1).
    assert "all categories" in _text(_page(_honest_nova(enrichment_databases=[])))
    unknown = _text(_page(_honest_nova(enrichment_databases=None)))
    assert "all categories" not in unknown and "Databases: not recorded" in unknown


def test_methods_name_the_differential_method_from_the_dataset_metadata():
    text = _text(_page(de_primary_method="edgeR", de_method_run="edger"))
    assert "called with edgeR" in text and "DESeq2" not in text and "limma" not in text
    text = _text(_page(de_primary_method="DESeq2", de_method_run="all"))
    assert "called with DESeq2" in text
    assert "edgeR and limma-voom were run on the same counts" in text
    # Imported results: no method is named, none is described.
    text = _text(_page(has_analysis=False, analysis_label=None, params={}))
    assert "imported results" in text
    assert "DESeq2" not in text and "edgeR" not in text and "limma" not in text
    # An analysis run in GenoLens before the method was recorded: not guessed either.
    text = _text(_page())
    assert "the method that decided the calls is not recorded" in text
    assert "DESeq2" not in text and "edgeR" not in text


def test_methods_of_a_legacy_enrichment_name_no_test():
    """Rows stored by an earlier GenoLens enrichment (no analysis, no enrichment dataset, no
    uploaded table): the report says the parameters are not recorded, and names none (R-T4-3)."""
    legacy = _honest_nova(
        enrichment_source="legacy",
        analysis_id=None,
        enrichment_dataset_ids=[],
        enrichment_fdr=None,
        enrichment_databases=None,
    )
    params = {"species": "human", "fdr": 0.01, "min_log2fc": 0.585}
    text = _text(_page(legacy, params=params))
    assert (
        "earlier GenoLens method whose parameters (test, correction, cut-off, databases) are not recorded"
        in text
    )
    assert "hypergeometric" not in text.lower() and "benjamini" not in text.lower()
    assert "all categories" not in text and "cut-off not recorded" not in text
    assert "adjusted p < 0.05" not in text and "≤ 0.05" not in text
    # An analysis run in GenoLens keeps its test and its cut-off.
    assert "hypergeometric" in _text(_page(params=params))


def test_stouffer_text_says_unsigned_and_not_independent():
    method = next(p for p in _context()["pages"] if p["kind"] == "method")
    text = _text(method["method_nova"])
    assert "unsigned" in text and "independent" in text and "not weighted" in text
    # R-P1: a term at or above alpha is marked, not filtered.
    assert "terms not significant at α still contribute with their weight" in text
    # The referential is described by what it does (R-T4-2): no identifier family is listed, so
    # no database appears that the run did not list.
    assert "links it to that claim" in text
    for family in ("Gene Ontology", "Reactome", "Cell Ontology", "STRING", "KEGG", "WikiPathways"):
        assert family not in text, family
    for databases in (["GO", "KEGG"], None):
        method = next(
            p
            for p in _context(_honest_nova(enrichment_databases=databases))["pages"]
            if p["kind"] == "method"
        )
        assert "Ontology" not in _text(method["method_nova"])
    assert "Cell Ontology" not in _text(_page())


def test_strings_recorded_with_the_data_are_escaped():
    """A column header, a source version or a method name comes from an upload or a metadata
    record: it is printed as text, never as markup."""
    nova = _honest_nova(
        enrichment_source="uploaded", padj_columns=["<b>q</b>"], source_versions=["<i>v1</i>"]
    )
    nova["p_source"] = "adjusted p from uploaded tables (column <b>q</b>)"
    page = _page(nova, de_primary_method="<u>x</u>", de_method_run="all")
    for markup in ("<b>q</b>", "<i>v1</i>", "<u>x</u>"):
        assert markup not in page
    assert "&lt;b&gt;q&lt;/b&gt;" in page and "&lt;i&gt;v1&lt;/i&gt;" in page
    assert "&lt;u&gt;x&lt;/u&gt;" in page


def test_traceability_names_the_source_versions():
    text = _text(_page(_honest_nova(source_versions=["pipeline 1.2.3", "tables 2026-10"])))
    assert "Data source version pipeline 1.2.3, tables 2026-10" in text
    assert "Data source version not recorded" in _text(_page(_honest_nova(source_versions=[])))
    assert "Significance threshold α = 0.05" in text


def test_not_evaluated_claim_prints_ne():
    ctx = _context()
    strip = [s for p in ctx["pages"] if p["kind"] == "reservations" for s in p["strip"]]
    not_evaluated = [s for s in strip if s["name"] in ("Anti-wrinkles", "Energising", "Healing")]
    assert len(not_evaluated) == 3 and all(s["title"].startswith("n.e.") for s in not_evaluated)
    ladder = ctx["pages"][1]["ladder"]
    assert [r["band_short"] for r in ladder if not r["scored"]] == ["n.e."] * 3
    page = render_html(ctx)
    text = _text(page)
    assert "n.e. — no significant term for this claim in the enrichment results" in text
    assert "NO EVIDENCE" not in page and "not evidenced" not in text
    assert "No significant evidence" not in text
    # What was stored is not said: a deposited table is kept as it was given (R-T6-1).
    deposited = _honest_nova(
        enrichment_source="uploaded", padj_columns=["qvalue"], enrichment_fdr=None
    )
    deposited["p_source"] = "adjusted p from uploaded tables (column qvalue)"
    for page in (text, _text(_page(deposited))):
        assert "n.e. — no significant term for this claim in the enrichment results" in page
        assert "not stored" not in page.replace("is stored for this comparison", "")
        assert "only significant terms are stored" not in page.lower()


def test_zero_against_is_not_presented_as_a_confirmation():
    note = (
        "Only terms present in the enrichment results can count against a claim; "
        "no contrary term is not a confirmation."
    )

    def notes(nova):
        ctx = _context(nova)
        blocks = {
            p["claim"]["name"]: p["claim"]["zero_against_note"]
            for p in ctx["pages"]
            if p["kind"] == "claim"
        }
        return {name for name, n in blocks.items() if n}, _text(render_html(ctx)).count(note)

    # Barrier, Soothing and Firming have no sub-group against them in the reference mock, but
    # each has a contrary term (S_anti > 0): "0 against" is then no absence of contrary evidence,
    # and the note is not printed (R-T4-4 a).
    assert notes(_honest_nova()) == (set(), 0)

    def no_contrary_term(nova):
        claim = next(c for c in nova["claims"] if c["name"] == "Barrier")
        claim.update(s_anti=0.0, coherence=1.0, state="for_only")

    # A claim with no contrary term at all carries it: on its page, and on the wheel (the cover's
    # and page 3's).
    names, count = notes(_honest_nova(no_contrary_term))
    assert names == {"Barrier"} and count == 1 + 2


def test_sub_groups_are_not_called_pathways():
    """A sub-group is a group of pathways, not a pathway (R-T4-4 b)."""
    text = _text(_page())
    assert "5 significant sub-groups linked to no claim" in text
    assert "pathways going" not in text and "Significant pathways" not in text
    assert "significant pathway sub-group" not in text
    none = _text(_page(_honest_nova(lambda n: n.update(unlinked=[], unlinked_total=0))))
    assert "Significant sub-groups linked to no claim would appear here" in none
    # The reservations page: "Supported, but with sub-groups going the other way."
    with_against = _context(_honest_nova())
    headlines = [p["headline"] for p in with_against["pages"] if p["kind"] == "reservations"]
    assert headlines == ["Supported, but with sub-groups going the other way."]


def _guide(ctx) -> str:
    return dict(ctx["pages"][1]["guide"])["How to read this review."]


def test_reading_guide_follows_the_bands():
    import copy

    default = _guide(_context())
    assert "dossier" not in default.lower() and "without evidence" not in default
    assert default.endswith(
        "— then the claims supported with reservations and the claims with weak evidence or "
        "not evaluated (n.e.)."
    )

    def weaken(nova):
        for claim in nova["claims"]:
            claim["best_nl"] = claim["best_for_nl"] = 1.0
            claim["coherence"] = 0.5

    # Every scored claim under 25: they are scored (they print "Score N · weak evidence"), the
    # others are not evaluated; the guide says neither "none is scored" nor "without evidence".
    ctx = _context(_honest_nova(weaken))
    text = _guide(ctx)
    assert text == (
        "Page 4 lists the claims with weak evidence (score below 25) and the claims not "
        "evaluated (n.e.)."
    )
    assert "Score 6 · weak evidence." in _text(render_html(ctx))

    def all_scored(nova):
        weaken(nova)
        for name in nova["not_evaluated"]:
            extra = copy.deepcopy(nova["claims"][0])
            extra["name"] = name
            nova["claims"].append(extra)
        nova["not_evaluated"] = []

    text = _guide(_context(_honest_nova(all_scored)))
    assert text == "Page 4 lists the claims, all with weak evidence (score below 25)."

    # Nothing significant, or no result at all: said as such.
    empty = {"contrast": "A_vs_B", "claims": [], "unlinked": [], "not_evaluated": []}
    assert "why none of them is evaluated (n.e.): no term of theirs is significant" in _guide(
        _context(empty)
    )
    none = {"contrast": "A_vs_B", "status": "no_enrichment", "claims": [], "unlinked": []}
    assert "why none is evaluated: this comparison has no enrichment result" in _guide(
        _context(none, enr_up=[], enr_down=[])
    )
    assert "none of them is scored" not in default + text


def test_lowest_band_is_weak_evidence():
    def weaken(nova):
        claim = next(c for c in nova["claims"] if c["name"] == "Antioxidant")
        claim["best_nl"] = claim["best_for_nl"] = 1.0
        claim["coherence"] = 0.5

    nova = _honest_nova(weaken)
    antioxidant = next(s for s in score_claims(nova) if s.name == "Antioxidant")
    assert (antioxidant.score, antioxidant.band) == (6, "not_supported")
    assert antioxidant.label == "Weak evidence" == BAND_LABELS["not_supported"]
    assert BAND_LABELS["no_evidence"] == "n.e."
    text = _text(_page(nova))
    assert "Score 6 · weak evidence." in text
    assert "< 25 weak evidence" in text
    assert "not supported" not in text.lower()


def test_both_direction_term_is_marked():
    def both(nova):
        for sg in nova["claims"][0]["subgroups"]:
            for term in sg["terms"]:
                term["both_directions"] = True

    # The mark of the terms table (the methods mention the symbol too, in a sentence).
    assert ">↑↓</td>" not in _page()
    page = _page(_honest_nova(both))
    assert ">↑↓</td>" in page
    assert "↑↓ = enriched in both" in _text(page)


def test_a_term_not_significant_at_alpha_is_marked_and_still_counted():
    def not_significant(nova):
        for sg in nova["claims"][0]["subgroups"]:
            for term in sg["terms"]:
                term["significant"] = False

    assert "· n.s.</span>" not in _page()
    page = _page(_honest_nova(not_significant))
    assert "· n.s.</span>" in page
    text = _text(page)
    assert "n.s. = not significant at α" in text and "counted with its weight" in text


def _claim(**overrides) -> dict:
    claim = {
        "name": "Barrier",
        "state": "both_ways",
        "importance": 20.0,
        "s_pro": 14.0,
        "s_anti": 6.0,
        "coherence": 0.7,
        "n_for": 1,
        "n_against": 1,
        "n_balanced": 1,
        "best_subgroup": "Group A",
        "best_nl": 9.0,
        "best_for_subgroup": "Group B",
        "best_for_nl": 6.0,
        "subgroups": [
            {
                "name": "Group A",
                "nl": 9.0,
                "s_pro": 0.0,
                "s_anti": 6.0,
                "delta": -6.0,
                "balance": -1.0,
                "obs": "down",
                "terms": [],
            },
            {
                "name": "Group B",
                "nl": 6.0,
                "s_pro": 14.0,
                "s_anti": 0.0,
                "delta": 14.0,
                "balance": 1.0,
                "obs": "up",
                "terms": [],
            },
            {
                "name": "Group C",
                "nl": 2.0,
                "s_pro": 0.0,
                "s_anti": 0.0,
                "delta": 0.0,
                "balance": 0.0,
                "obs": "up",
                "terms": [],
            },
        ],
    }
    claim.update(overrides)
    return claim


def test_a_claim_is_led_by_its_best_supporting_subgroup():
    from app.services.report_editorial_content import claim_statement

    claim = _claim()
    text = claim_statement(score_claims({"claims": [claim]})[0], "AL-7", "Vehicle")
    # The most significant sub-group (Group A) goes the other way: it does not lead the claim.
    assert "led by Group B" in text and "led by Group A" not in text
    assert "Group A" in text and "other way" in text
    assert "Group C" not in text and "balanced" in text  # a balanced sub-group is not contrary
    # No supporting sub-group: the statement says so instead of naming a leader.
    bare = _claim(best_for_subgroup=None, best_for_nl=None, n_for=0, coherence=0.0)
    text = claim_statement(score_claims({"claims": [bare]})[0], "AL-7", "Vehicle")
    assert text.startswith("No pathway sub-group supports the Barrier claim")
    assert "led by" not in text


def test_a_balanced_subgroup_is_not_greyed_like_a_contrary_one():
    from app.services.report_editorial_svg import subgroup_bars

    bars = str(
        subgroup_bars(
            [
                {"name": "Group A", "nl": 5.0, "delta": 2.0, "balance": 0.4},
                {"name": "Group B", "nl": 4.0, "delta": 0.0, "balance": 0.0},
                {"name": "Group C", "nl": 3.0, "delta": -2.0, "balance": -0.4},
            ]
        )
    )
    opacity = {name: float(o) for o, name in re.findall(r'opacity="([\d.]+)">(Group [ABC])<', bars)}
    assert opacity == {"Group A": 1.0, "Group B": 1.0, "Group C": 0.55}


def test_no_enrichment_is_not_reported_as_no_significant_claim():
    nova = {"contrast": "A_vs_B", "status": "no_enrichment", "claims": [], "unlinked": []}
    ctx = _context(nova, enr_up=[], enr_down=[])
    text = _text(render_html(ctx))
    assert "No enrichment result is stored for this comparison" in text
    assert "No claim has a significant term" not in text
    # Every claim is n.e. for want of a result, not for want of a significant term (R-T4-4 c).
    strip = [s for p in ctx["pages"] if p["kind"] == "reservations" for s in p["strip"]]
    assert len(strip) == 9
    assert all(s["title"] == "n.e. — no enrichment result for this comparison." for s in strip)
    assert "no significant term for this claim" not in text


def test_claim_pages_keep_the_users_own_signoff_card(html):
    # The "Scientific review" card is the sign-off of the cover form, not an AI claim.
    assert "Report checked by J. Doe." in html and "Not yet reviewed." not in html
    assert "Not yet reviewed." in _page(cover_info={})


def test_thinned_volcano_says_sample():
    def gene(i, regulation):
        return {
            "gene_id": f"g{i}",
            "gene_name": f"G{i}",
            "log_fc": 2.0 if regulation == "UP" else 0.1,
            "padj": 0.001 if regulation == "UP" else 0.9,
            "regulation": regulation,
        }

    degs = [gene(i, "NS") for i in range(9000)] + [gene(9000 + i, "UP") for i in range(20)]
    assert volcano_sampling(degs) == (7980, 9000)
    assert "a sample of 7 980 of 9 000 genes" in _text(_page(degs=degs))
    # Nothing thinned: every point is a gene, and the caption says no more than that.
    assert volcano_sampling(degs[8000:]) is None
    text = _text(_page())
    assert "Each point is a gene" in text and "a sample of" not in text


def test_enrichment_table_has_no_pvalue_copied_from_padj():
    up, down = hits_to_enrichment(
        [
            TermHit("GO:0000001", "Term A", 1e-4, "up", "GO:BP", 5),
            TermHit("GO:0000002", "Term B", None, "down"),
        ]
    )
    assert [r["pathway_id"] for r in up] == ["GO:0000001"] and down == []
    assert up[0]["padj"] == 1e-4
    assert up[0].get("pvalue") is None  # the raw p-value is unknown here, not the adjusted one


async def _collected(
    nova, *, requester_is_demo=False, test="AL-7", metadata=None, methods=None
):
    from app.services import report_editorial_service as module

    data = {
        "comparisons": [
            {
                "test": test,
                "control": "Vehicle",
                "degs": [],
                "n_up": 0,
                "n_down": 0,
                # `report_latex_methods.methods_context` of the comparison
                "methods": methods or {},
            }
        ],
        "params": {},
        "expression": {},
        "qc_rows": [],
        "project": None,
        "analysis": None,
    }
    db = MagicMock()
    db.get = AsyncMock(return_value=SimpleNamespace(dataset_metadata=metadata or {}))
    with (
        patch.object(
            module.report_latex_service, "collect_comparison_data", AsyncMock(return_value=data)
        ),
        patch.object(module, "compute_nova", AsyncMock(return_value=(nova, []))),
        patch.object(module, "build_figures", MagicMock(return_value={})),
    ):
        return await module.report_editorial_service.collect(
            db,
            uuid4(),
            "LongNameTreatment_vs_Vehicle",
            "/nonexistent",
            demo_watermark=requester_is_demo,
            use_ai=False,
            use_branding=False,
        )


async def test_demo_dataset_report_is_watermarked():
    # The mark follows the dataset, not only the account that asks for the report.
    inputs = await _collected(_honest_nova(demo=True))
    assert inputs.demo_watermark is True
    assert '<div class="watermark"><span>DEMO</span></div>' in render_html(
        build_editorial_context(inputs)
    )
    # The page carries the mark over the picture: the inlined wheels do not repeat it.
    assert ">DEMO<" not in inputs.nova_light and ">DEMO<" not in inputs.nova_dark
    plain = await _collected(_honest_nova(demo=False))
    assert plain.demo_watermark is False
    assert 'class="watermark"' not in render_html(build_editorial_context(plain))
    # A demo account's request stays marked whatever the dataset says.
    assert (await _collected(_honest_nova(demo=False), requester_is_demo=True)).demo_watermark


async def test_the_dataset_records_the_method_and_the_wheel_names_no_invented_condition():
    inputs = await _collected(
        _honest_nova(), metadata={"primary_method": "limma", "de_method": "limma"}
    )
    assert (inputs.de_primary_method, inputs.de_method_run) == ("limma", "limma")
    # A condition too long for the disc and none recorded on the nova: the disc stays empty.
    assert "Active" not in inputs.nova_light and "Active" not in inputs.nova_dark


# ── AI texts ────────────────────────────────────────────────────────────────


GENERATED_AT = datetime(2026, 11, 5, 9, 30, tzinfo=timezone.utc)
LABEL = "AI-generated text (model google/gemma-4-E4B-it, generated 2026-11-05)"


def _dated(**overrides) -> EditorialInputs:
    return _inputs(
        comparison_ai_generated_at=GENERATED_AT, nova_ai_generated_at=GENERATED_AT, **overrides
    )


def test_editorial_ai_label_names_model_and_date():
    page = render_html(build_editorial_context(_dated()))
    # The reading of the nova (page 3) and the comparison text each carry the label.
    assert page.count(f'<h3 class="ai-label">{LABEL}</h3>') == 2
    # Without a date the label still names the model.
    undated = render_html(build_editorial_context(_inputs()))
    assert '<h3 class="ai-label">AI-generated text (model google/gemma-4-E4B-it)</h3>' in undated
    assert "generated 2026" not in undated


def _css_text_transform(css: str, tag: str, classes: set, ancestors: list) -> str:
    """The `text-transform` the cascade gives an element: the declaration of the rule with the
    highest specificity (then the last one) among the rules whose selector matches it. `ancestors`
    are the class sets of its parents, nearest first. Selectors of the form `tag.class .class tag`
    only (what this stylesheet uses); `inherit`/initial values are out of scope."""

    def compound(sel):
        m = re.fullmatch(r"([a-z0-9]*)((?:\.[\w-]+)*)", sel)
        return (m.group(1), set(m.group(2).split(".")[1:])) if m else None

    def matches(sel_tag, sel_classes, el_tag, el_classes):
        return (not sel_tag or sel_tag == el_tag) and sel_classes <= el_classes

    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    best = (-1, -1, -1, None)  # (classes, tags, order, value)
    order = 0
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        decl = re.search(r"(?<![\w-])text-transform\s*:\s*([\w-]+)", body)
        if not decl:
            continue
        for selector in selectors.split(","):
            order += 1
            parts = [compound(p) for p in selector.split()]
            if not parts or None in parts:
                continue
            last_tag, last_classes = parts[-1]
            if not matches(last_tag, last_classes, tag, classes):
                continue
            remaining = list(ancestors)
            ok = True
            for part_tag, part_classes in reversed(parts[:-1]):
                while remaining and not matches(part_tag, part_classes, "div", remaining[0]):
                    remaining.pop(0)
                if not remaining:
                    ok = False
                    break
                remaining.pop(0)
            if not ok:
                continue
            specificity = (
                sum(len(c) for _t, c in parts),
                sum(1 for t, _c in parts if t),
                order,
                decl.group(1),
            )
            if specificity[:3] > best[:3]:
                best = specificity
    return best[3] or "none"


def test_the_ai_label_prints_as_written_not_uppercased():
    """The label names the model by its id and the day: the h3 headings of the page are uppercased
    by CSS, which would turn `gemma-4-E4B-it` into `GEMMA-4-E4B-IT` in the PDF (recette D4)."""
    page = render_html(build_editorial_context(_dated()))
    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", page, re.S))
    assert css
    labels = re.findall(
        r'<div class="ai">\s*<h3([^>]*)>' + re.escape(LABEL) + "</h3>", page
    )
    assert len(labels) == 2  # the reading of the nova and the comparison text
    for attrs in labels:
        classes = set(re.findall(r'class="([^"]*)"', attrs)[0].split()) if "class=" in attrs else set()
        assert "text-transform" not in attrs  # no inline style that transforms it either
        assert _css_text_transform(css, "h3", classes, [{"ai"}, {"page"}]) == "none"
    # The rule engine of this test sees the headings that stay uppercased: the other h3 of the
    # page, and the heading of an AI box that holds no text.
    assert _css_text_transform(css, "h3", set(), [{"card"}, {"page"}]) == "uppercase"
    assert _css_text_transform(css, "h3", set(), [{"ai"}, {"page"}]) == "uppercase"
    empty = render_html(
        build_editorial_context(_dated(comparison_ai=None, comparison_ai_model=None))
    )
    assert re.search(r'<div class="ai">\s*<h3>AI interpretation</h3>', empty)


def test_editorial_has_no_reviewed_by():
    """A form field does not prove a review: the sign-off block stays, the AI label never says it."""
    for cover in ({"checked_by": "J. Doe", "prepared_by": "A. B."}, {}):
        page = render_html(build_editorial_context(_dated(cover_info=cover)))
        assert "reviewed by" not in page.lower()
        assert "AI-generated, not yet reviewed" not in page
        assert "AI-generated interpretation" not in page
    signed = render_html(build_editorial_context(_dated()))
    assert "Checked by" in signed and "J. Doe" in signed  # the sign-off block is the user's own


def test_editorial_guide_and_scope_name_the_label_not_a_review_status():
    ctx = build_editorial_context(_dated())
    guide = " ".join(text for page in ctx["pages"] for _t, text in page.get("guide", []))
    assert "labelled with its model and date" in guide
    scope = next(p["scope"] for p in ctx["pages"] if p["kind"] == "method")
    assert "model and the date" in scope and "reviewed" not in scope


def test_editorial_leaves_out_a_stale_nova_text():
    page = render_html(
        build_editorial_context(_dated(nova_ai=None, nova_ai_model=None, nova_ai_stale=True))
    )
    assert "The <b>barrier</b> programme" not in page and "Soothing is supported" not in page
    assert "Nova AI text</td><td>not included (generated on other data)</td>" in page
    # The comparison text, still current, is printed with its label.
    assert page.count(f'<h3 class="ai-label">{LABEL}</h3>') == 1


def test_editorial_leaves_out_a_stale_comparison_text():
    page = render_html(
        build_editorial_context(
            _dated(comparison_ai=None, comparison_ai_model=None, comparison_ai_stale=True)
        )
    )
    assert "cornified-envelope genes" not in page
    assert "Comparison AI text" in page and "not included (generated on other data)" in page
    # The box says why it is empty, rather than that nothing was ever generated.
    assert "generated on other data and is not included" in page
    assert "No AI interpretation has been generated" not in page


def test_nothing_is_said_about_stale_texts_in_a_report_without_ai():
    page = render_html(
        build_editorial_context(
            _inputs(use_ai=False, nova_ai=None, comparison_ai=None, nova_ai_stale=True)
        )
    )
    assert "generated on other data" not in page


def test_context_lists_the_ai_texts_it_prints():
    ctx = build_editorial_context(_dated())
    assert ctx["ai_texts"] == [
        {"kind": "nova", "model": "google/gemma-4-E4B-it", "generated_at": GENERATED_AT.isoformat()},
        {
            "kind": "comparison",
            "model": "google/gemma-4-E4B-it",
            "generated_at": GENERATED_AT.isoformat(),
        },
    ]
    # A text that is not printed is not listed: a French comparison text, a report without AI,
    # a stale nova text.
    french = build_editorial_context(_dated(comparison_ai=FRENCH_AI))
    assert [t["kind"] for t in french["ai_texts"]] == ["nova"]
    assert build_editorial_context(_dated(use_ai=False))["ai_texts"] == []
    stale = build_editorial_context(_dated(nova_ai=None, nova_ai_stale=True))
    assert [t["kind"] for t in stale["ai_texts"]] == ["comparison"]


async def test_the_cached_nova_text_is_printed_only_while_its_data_stand():
    from app.models.models import CosmeticInterpretation
    from app.services.cosmomics_nova import NOVA_MODEL_TAG
    from app.services.report_common import nova_inputs_sha256
    from app.services.report_editorial_service import report_editorial_service

    nova = json.loads(FIXTURE.read_text())
    nova.setdefault("provenance", {})
    row = CosmeticInterpretation(
        interpretation=NOVA_AI,
        model=f"{NOVA_MODEL_TAG}m-1",
        claims_count=len(nova["claims"]),
        inputs_sha256=nova_inputs_sha256(nova, "m-1"),
        created_at=GENERATED_AT,
        updated_at=GENERATED_AT,
    )

    def db_with(cached):
        db = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = cached
        db.execute = AsyncMock(return_value=result)
        return db

    current = await report_editorial_service._nova_ai(
        db_with(row), uuid4(), "AL-7_vs_Vehicle", nova, False, False
    )
    assert (current.text, current.model, current.generated_at) == (NOVA_AI, "m-1", GENERATED_AT)
    assert current.stale is False

    # The data moved on since the text was written (here: another alpha) ...
    moved = {**nova, "alpha": 0.01}
    left_out = await report_editorial_service._nova_ai(
        db_with(row), uuid4(), "AL-7_vs_Vehicle", moved, False, False
    )
    assert left_out.text is None and left_out.stale is True
    # ... or the text predates the fingerprint: it is left out too, and never regenerated here.
    row.inputs_sha256 = None
    undated = await report_editorial_service._nova_ai(
        db_with(row), uuid4(), "AL-7_vs_Vehicle", nova, False, False
    )
    assert undated.text is None and undated.stale is True
    # No cached row: nothing printed, and nothing to be stale.
    none = await report_editorial_service._nova_ai(
        db_with(None), uuid4(), "AL-7_vs_Vehicle", nova, False, False
    )
    assert none.text is None and none.stale is False


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
    assert "No claim has a significant term" in page
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


# ── R-F1: the methods say what the provenance's enrichment_source says ────────────────


def _no_enrichment(**provenance):
    """A nova with no stored enrichment result (status no_enrichment) and the given provenance."""

    def mutate(nova):
        nova.update(status="no_enrichment", claims=[], unlinked=[], unlinked_total=0)

    return _honest_nova(mutate, **provenance)


NO_RESULT = "No enrichment result is stored for this comparison"
RUN_CLAIMS = (
    "hypergeometric",
    "benjamini",
    "over-representation",
    "cut-off",
    "all categories",
    "databases:",
    "earlier genolens method",
    "terms kept",
)


@pytest.mark.parametrize(
    "provenance",
    [
        # What the payload of an analysis whose enrichment never ran carried (R-F1): the analysis
        # parameters stood in for a run, and the methods asserted an ORA.
        dict(enrichment_source="none", enrichment_fdr=None, enrichment_databases=None,
             enrichment_dataset_ids=[]),
        # A reader that meets the status first must not trust a stale GenoLens provenance either.
        dict(enrichment_source="annodb", enrichment_fdr=0.05, enrichment_databases=[]),
        dict(enrichment_source="legacy", analysis_id=None, enrichment_dataset_ids=[],
             enrichment_fdr=None, enrichment_databases=None),
        dict(enrichment_source="uploaded", padj_columns=["qvalue"], enrichment_fdr=None),
    ],
    ids=["none", "annodb-status", "legacy-status", "uploaded-status"],
)
def test_methods_of_a_comparison_with_no_enrichment_say_only_that_none_is_stored(provenance):
    nova = _no_enrichment(**provenance)
    ctx = _context(nova, enr_up=[], enr_down=[])
    pages = {p["kind"]: p for p in ctx["pages"]}
    method = pages["method"]
    # The places that used to state a run: the design sentence, the Versions rows, the nova
    # paragraph, and the sentence above the enrichment figures.
    texts = {
        "design": _text(pages["science_design"]["design_text"]),
        "versions": " ".join(f"{k} {v}" for k, v in method["versions"]),
        "nova": _text(method["method_nova"]),
        "figures": enrichment_text(enrichment_facts(nova)),
    }
    assert NO_RESULT in texts["design"] and NO_RESULT in texts["figures"]
    assert dict(method["versions"])["Enrichment"] == NO_RESULT
    assert "Enrichment databases" not in dict(method["versions"])
    page = _text(render_html(ctx))
    assert NO_RESULT in page
    for where, text in {**texts, "page": page}.items():
        for phrase in RUN_CLAIMS:
            assert phrase not in text.lower(), (where, phrase)


def test_methods_of_a_deleted_analysis_do_not_print_all_categories():
    """The analysis is gone: its pick, its cut-off and its categories are not known (R-F1)."""
    nova = _honest_nova(enrichment_fdr=None, enrichment_databases=None, enrichment_categories=None)
    text = _text(_page(nova))
    assert "all categories" not in text
    assert "Databases: not recorded" in text and "cut-off not recorded" in text
    assert "hypergeometric" in text  # a GenoLens annoDB run: its test is still known


def test_methods_print_the_categories_recorded_at_run_time():
    nova = _honest_nova(
        enrichment_databases=["GO", "Reactome", "KEGG"], enrichment_categories=["GO_BP", "reactome"]
    )
    text = _text(_page(nova))
    assert "Databases: GO_BP, reactome" in text
    assert "KEGG" not in text and "all categories" not in text  # observed, not the pick
    assert dict(next(p for p in _context(nova)["pages"] if p["kind"] == "method")["versions"])[
        "Enrichment databases"
    ] == "GO_BP, reactome"


def test_uploaded_is_told_by_the_source_not_by_the_p_source_prefix():
    # A deposited table whose p_source was lost is still a deposited table.
    nova = _honest_nova(enrichment_source="uploaded", padj_columns=["qvalue"], enrichment_fdr=None)
    nova["p_source"] = ""
    text = _text(_page(nova))
    assert "column qvalue" in text and "hypergeometric" not in text.lower()
    # And a p_source that merely looks like one does not turn a GenoLens run into a deposit.
    nova = _honest_nova()
    nova["p_source"] = "adjusted p from uploaded tables (column q)"
    text = _text(_page(nova))
    assert "hypergeometric" in text and "deposited" not in text.lower()


def test_a_provenance_without_enrichment_source_infers_nothing():
    nova = _honest_nova()
    del nova["provenance"]["enrichment_source"]
    text = _text(_page(nova))
    assert "hypergeometric" not in text.lower() and "benjamini" not in text.lower()
    assert "all categories" not in text and "stored with the dataset" in text


# ── R-F3: an orphaned analysis is "no longer available", not "imported" ──────────────────


def test_an_orphaned_analysis_is_not_called_imported():
    gone = _text(_page(has_analysis=False, analysis_gone=True, analysis_label=None, params={}))
    assert "The analysis that produced these results is no longer available" in gone
    assert "its methods are not described here" in gone
    assert "imported" not in gone.lower()
    assert "DESeq2" not in gone and "edgeR" not in gone and "limma" not in gone
    ctx = _context(has_analysis=False, analysis_gone=True, analysis_label=None, params={})
    method = next(p for p in ctx["pages"] if p["kind"] == "method")
    assert dict(method["versions"])["Differential expression"] == (
        "analysis no longer available (method not described)"
    )
    # Even when the dataset still records the method that decided the calls, like the LaTeX report.
    named = _text(
        _page(
            has_analysis=False,
            analysis_gone=True,
            analysis_label=None,
            params={},
            de_primary_method="DESeq2",
            de_method_run="all",
        )
    )
    assert "no longer available" in named and "called with DESeq2" not in named
    # Results that were never produced by an analysis keep the imported sentence.
    never = _text(_page(has_analysis=False, analysis_label=None, params={}))
    assert "imported with the dataset" in never and "no longer available" not in never


async def test_the_collected_inputs_name_an_orphaned_analysis():
    """`collect` hands the editorial what the LaTeX methods record: the DEG metadata names an
    analysis that cannot be found (R-F3)."""
    gone = await _collected(_honest_nova(), methods={"analysis_gone": True, "imported": True})
    assert gone.analysis_gone is True and gone.has_analysis is False
    text = _text(render_html(build_editorial_context(gone)))
    assert "no longer available" in text and "imported" not in text.lower()
    for methods in ({}, {"analysis_gone": False, "imported": True}):
        assert (await _collected(_honest_nova(), methods=methods)).analysis_gone is False
