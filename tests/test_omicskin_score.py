"""
OmicSkin score — formula, bands and per-claim scoring of a nova payload.

The reference values come from the validated editorial report (docs/omicskin-report-editorial,
data.py `score()` on the mock nova): Barrier 96, Soothing 94, Firming 91, Regeneration 87,
Anti-ageing 79, Antioxidant 63; Anti-wrinkles, Energising and Healing have no evidence.
"""

import json
from pathlib import Path

import pytest

from app.services.cosmomics_nova import ClaimSubgroupMap, SubgroupTerm, TermHit, build_nova
from app.services.omicskin_score import (
    ALL_CLAIMS,
    BAND_LABELS,
    SATURATION_NL,
    omicskin_band,
    omicskin_score,
    score_claims,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cosmomics_nova_mock.json"


def _reference_score(claim: dict) -> int:
    """docs/omicskin-report-editorial/data.py:score — the formula the report was validated on."""
    return int(round(100 * claim["coherence"] * min(1.0, claim["best_nl"] / 8.0)))


@pytest.mark.parametrize(
    "coherence,best_nl,expected",
    [
        (1.0, 8.0, 100),  # saturation reached exactly
        (1.0, 20.0, 100),  # beyond saturation
        (0.5, 8.0, 50),
        (1.0, 4.0, 50),
        (0.5, 4.0, 25),
        (0.0, 12.0, 0),  # all evidence against the claim
        (1.0, 0.0, 0),
        (0.8, 1.3, 13),  # p = 0.05 sub-group
    ],
)
def test_formula(coherence, best_nl, expected):
    assert omicskin_score(coherence, best_nl) == expected


def test_saturation_constant_is_p_1e_8():
    assert SATURATION_NL == 8.0


def test_inputs_are_clamped():
    assert omicskin_score(1.4, 9) == 100
    assert omicskin_score(-0.2, 9) == 0
    assert omicskin_score(0.9, -3) == 0


@pytest.mark.parametrize(
    "score,band",
    [
        (100, "strong"),
        (75, "strong"),
        (74, "moderate"),
        (50, "moderate"),
        (49, "weak"),
        (25, "weak"),
        (24, "not_supported"),
        (0, "not_supported"),
        (None, "no_evidence"),
    ],
)
def test_bands(score, band):
    assert omicskin_band(score) == band
    assert BAND_LABELS[band]


def test_score_claims_matches_the_validated_report():
    nova = json.loads(FIXTURE.read_text())
    scores = score_claims(nova)

    assert [s.name for s in scores if s.scored] == [
        "Barrier",
        "Soothing",
        "Firming",
        "Regeneration",
        "Anti-ageing",
        "Antioxidant",
    ]
    by_name = {s.name: s for s in scores}
    assert {n: by_name[n].score for n in by_name if by_name[n].scored} == {
        "Barrier": 96,
        "Soothing": 94,
        "Firming": 91,
        "Regeneration": 87,
        "Anti-ageing": 79,
        "Antioxidant": 63,
    }
    for claim in nova["claims"]:
        assert by_name[claim["name"]].score == _reference_score(claim)
    assert by_name["Antioxidant"].band == "moderate"
    assert by_name["Barrier"].label == "Strong support"


def test_claims_absent_from_the_nova_have_no_significant_evidence():
    scores = score_claims(json.loads(FIXTURE.read_text()))
    assert len(scores) == len(ALL_CLAIMS) == 9
    assert {s.name for s in scores} == set(ALL_CLAIMS)
    unscored = [s for s in scores if not s.scored]
    assert sorted(s.name for s in unscored) == ["Anti-wrinkles", "Energising", "Healing"]
    assert all(s.band == "no_evidence" and s.score is None and s.claim is None for s in unscored)
    assert all(s.label == "No significant evidence" for s in unscored)
    # Scored claims come first, highest score first.
    values = [s.score for s in scores if s.scored]
    assert values == sorted(values, reverse=True)
    assert scores.index(unscored[0]) == len(values)


def test_empty_nova_scores_nothing():
    scores = score_claims({"claims": []})
    assert [s.band for s in scores] == ["no_evidence"] * 9


# ── The score reads the best sub-group that supports the claim ───────────────


def _claim_with_a_contrary_best_subgroup() -> dict:
    """The most significant sub-group (p = 1e-10) goes against the claim; the best one going the
    expected way is at p = 1e-4. Coherence = 4 / (4 + 10)."""
    rows = [
        SubgroupTerm("Firming", "Collagen", "GO:0000001", "Collagen", "up"),
        SubgroupTerm("Firming", "Barrier", "GO:0000002", "Barrier", "up"),
    ]
    hits = [
        TermHit("GO:0000001", "Collagen", 1e-4, "up"),
        TermHit("GO:0000002", "Barrier", 1e-10, "down"),
    ]
    return build_nova(hits, ClaimSubgroupMap(rows))


def test_score_uses_the_best_supporting_subgroup():
    nova = _claim_with_a_contrary_best_subgroup()
    claim = nova["claims"][0]
    assert claim["best_nl"] == pytest.approx(10.0)  # the strongest sub-group, against the claim
    assert claim["best_for_nl"] == pytest.approx(4.0)
    (scored,) = [s for s in score_claims(nova) if s.scored]
    # 100 x 4/14 x min(1, 4 / 8): the supporting sub-group's strength, not the contrary one's (29).
    assert scored.score == omicskin_score(claim["coherence"], 4.0) == 14
    assert scored.score != omicskin_score(claim["coherence"], claim["best_nl"])


def test_a_claim_without_a_supporting_subgroup_scores_zero():
    rows = [SubgroupTerm("Firming", "Collagen", "GO:0000001", "Collagen", "up")]
    nova = build_nova([TermHit("GO:0000001", "Collagen", 1e-9, "down")], ClaimSubgroupMap(rows))
    assert nova["claims"][0]["best_for_nl"] is None
    (scored,) = [s for s in score_claims(nova) if s.scored]
    assert scored.score == 0
    assert scored.band == "not_supported"


def test_a_payload_without_best_for_reads_the_supporting_subgroup_off_its_subgroups():
    """The reference mock predates `best_for_nl`: the same figure comes from its sub-groups."""
    nova = _claim_with_a_contrary_best_subgroup()
    del nova["claims"][0]["best_for_nl"], nova["claims"][0]["best_for_subgroup"]
    (scored,) = [s for s in score_claims(nova) if s.scored]
    assert scored.score == 14
