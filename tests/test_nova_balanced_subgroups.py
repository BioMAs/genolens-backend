"""
A balanced sub-group (delta = 0) is neither for nor against a claim.

The payload counts it apart (`n_balanced`), and every consumer of the payload must say the same:
the wheel draws it neutral, outside the hatched "against" arc; the editorial sentence does not name
it among the sub-groups that "go the other way"; the AI prompt lists it as balanced, never as
contradicting. One predicate, `subgroup_side`, decides for all of them.
"""

import math
import re

import pytest

from app.services.ai_interpreter import LocalAIInterpreter
from app.services.cosmomics_nova import (
    ClaimSubgroupMap,
    SubgroupTerm,
    TermHit,
    build_nova,
    subgroup_side,
)
from app.services.cosmomics_nova_svg import CX, CY, _Nova, balance_color, render_nova_svg
from app.services.omicskin_score import score_claims
from app.services.report_editorial_content import claim_statement


def tie_nova() -> dict:
    """One claim with a sub-group for it (A), one against it (B) and one balanced (C).

    C holds a term going the expected way and a term going the other way at the same p-value, so
    its delta is exactly 0 and its balance 0. It is the least significant sub-group, so no
    sentence about the claim has a reason to name it.
    """
    rows = [
        SubgroupTerm("Firming", "Group A", "GO:0000001", "Term A", "up"),
        SubgroupTerm("Firming", "Group B", "GO:0000002", "Term B", "up"),
        SubgroupTerm("Firming", "Group C", "GO:0000003", "Term C", "up"),
        SubgroupTerm("Firming", "Group C", "GO:0000004", "Term D", "up"),
    ]
    hits = [
        TermHit("GO:0000001", "Term A", 1e-6, "up"),
        TermHit("GO:0000002", "Term B", 1e-5, "down"),
        TermHit("GO:0000003", "Term C", 1e-3, "up"),
        TermHit("GO:0000004", "Term D", 1e-3, "down"),
    ]
    return build_nova(hits, ClaimSubgroupMap(rows))


def test_the_fixture_has_one_subgroup_on_each_side():
    claim = tie_nova()["claims"][0]
    sides = {s["name"]: subgroup_side(s) for s in claim["subgroups"]}
    assert sides == {"Group A": "for", "Group B": "against", "Group C": "balanced"}
    assert (claim["n_for"], claim["n_against"], claim["n_balanced"]) == (1, 1, 1)


@pytest.mark.parametrize(
    "delta,side",
    [
        (5.0, "for"),
        (-5.0, "against"),
        (0.0, "balanced"),
        (5e-10, "balanced"),  # float sums of equal evidence need not cancel exactly
        (-5e-10, "balanced"),
        (1e-8, "for"),
        (-1e-8, "against"),
    ],
)
def test_subgroup_side(delta, side):
    assert subgroup_side({"delta": delta}) == side


def test_a_subgroup_without_a_delta_is_balanced():
    assert subgroup_side({}) == "balanced"


# ── The wheel ───────────────────────────────────────────────────────────────


def _bubble_offset(svg: str, fill: str, start: float) -> float:
    """Angle (degrees, from the claim's first edge) of the sub-group bubble drawn in `fill`."""
    found = re.search(
        rf'<circle cx="([\d.]+)" cy="([\d.]+)" r="[\d.]+" fill="{fill}" opacity="0.95" stroke=', svg
    )
    assert found, f"no bubble filled {fill}"
    x, y = float(found.group(1)), float(found.group(2))
    return (math.degrees(math.atan2(y - CY, x - CX)) - start) % 360


def test_a_balanced_subgroup_is_not_drawn_in_the_hatched_against_arc():
    nova = tie_nova()
    svg = render_nova_svg(nova, "light", id_prefix="t")
    band, _ = _Nova(nova, "light", "t", "A", 0.05).layout()
    s0, s1, s2 = band["Firming"]

    tie = _bubble_offset(svg, balance_color(0), s0)
    contrary = _bubble_offset(svg, balance_color(-1), s0)
    assert contrary > s1 - s0, "the sub-group against the claim sits in the hatched arc"
    assert contrary < s2 - s0
    assert tie < s1 - s0, "a balanced sub-group is drawn neutral, outside the hatched arc"


# ── The editorial sentence ──────────────────────────────────────────────────


def test_a_balanced_subgroup_is_not_named_among_those_that_go_the_other_way():
    scores = {s.name: s for s in score_claims(tie_nova())}
    sentence = claim_statement(scores["Firming"], "Treatment", "Control")
    assert "one sub-group (Group B) goes the other way" in sentence
    assert "Group C" not in sentence


# ── The AI prompt ───────────────────────────────────────────────────────────


def test_a_balanced_subgroup_reaches_the_ai_prompt_as_balanced_not_contradicting():
    interpreter = LocalAIInterpreter(base_url="http://127.0.0.1:9/v1")
    prompt = interpreter._build_cosmomics_prompt(tie_nova(), "A_vs_B")
    line = next(line for line in prompt.splitlines() if line.startswith("- Firming"))

    contradicting = re.search(r"contradicting: ([^;]*)", line)
    assert contradicting and "Group B" in contradicting.group(1)
    assert "Group C" not in contradicting.group(1)
    assert "balanced: Group C" in line
    assert "1 against" in line and "1 balanced" in line
