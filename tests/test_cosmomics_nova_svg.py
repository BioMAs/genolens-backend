"""
Server-side SVG of the CosmOmics Pathway Nova (`render_nova_svg`) and its endpoint.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.cosmomics_nova import TermHit
from app.services.cosmomics_nova_svg import balance_color, claim_color, render_nova_svg

FIXTURE = Path(__file__).parent / "fixtures" / "cosmomics_nova_mock.json"


@pytest.fixture(scope="module")
def nova():
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_renders_well_formed_svg(nova, theme):
    svg = render_nova_svg(nova, theme=theme, id_prefix=f"t-{theme}")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.get("viewBox") == "0 130 1000 850"  # wheel-only crop
    for claim in nova["claims"]:
        assert f">{claim['name']}<" in svg
    assert "Not linked to a claim" in svg


def test_every_id_is_prefixed_so_two_novas_can_share_a_page(nova):
    light = render_nova_svg(nova, "light", id_prefix="cover")
    dark = render_nova_svg(nova, "dark", id_prefix="page")
    ids = re.findall(r'\bid="([^"]+)"', light + dark)
    assert ids and len(ids) == len(set(ids))
    assert all(i.startswith(("cover-", "page-")) for i in ids)
    for svg, prefix in ((light, "cover"), (dark, "page")):
        refs = re.findall(r"url\(#([^)]+)\)", svg)
        assert refs and all(r.startswith(f"{prefix}-") for r in refs)


def test_dark_theme_uses_its_own_hatch(nova):
    light = render_nova_svg(nova, "light", id_prefix="a")
    dark = render_nova_svg(nova, "dark", id_prefix="b")
    assert 'fill="#EEF1F4"' in light and 'fill="#1C2A38"' in dark


def test_output_is_deterministic(nova):
    first = render_nova_svg(nova, "light", id_prefix="x")
    assert first == render_nova_svg(nova, "light", id_prefix="x")


def test_terms_are_drawn_as_dots_without_identifiers(nova):
    svg = render_nova_svg(nova, "light", id_prefix="x", wheel_only=False)
    ET.fromstring(svg)
    assert not re.search(r"GO:\d{7}|R-HSA-\d+|hsa\d{5}", svg)


def test_subgroup_labels_are_opt_in(nova):
    best = nova["claims"][0]["best_subgroup"]
    assert f">{best}<" not in render_nova_svg(nova, "light", id_prefix="x")
    assert f">{best}<" in render_nova_svg(nova, "light", id_prefix="x", subgroup_labels=True)


def test_a_long_condition_is_cut_with_an_ellipsis_not_mid_word():
    """The nova names its condition after the numerator of the comparison, whatever its length:
    the disc has room for eight characters and says when the name is longer."""
    long_name = render_nova_svg({"claims": [], "unlinked": [], "condition": "Treatment_12"}, "light")
    assert ">Treatme…<" in long_name
    fits = render_nova_svg({"claims": [], "unlinked": [], "condition": "Treatmen"}, "light")
    assert ">Treatmen<" in fits


def test_empty_payload_renders_a_neutral_wheel():
    svg = render_nova_svg({"claims": [], "unlinked": []}, "light", id_prefix="e")
    ET.fromstring(svg)
    assert "No claim with significant evidence" in svg


def test_labels_are_escaped():
    payload = {
        "claims": [
            {
                "name": "Firming",
                "s_pro": 5.0,
                "s_anti": 0.0,
                "coherence": 1.0,
                "n_for": 1,
                "n_against": 0,
                "best_subgroup": "A <b>&",
                "best_nl": 5.0,
                "subgroups": [
                    {
                        "name": "A <b>&",
                        "nl": 5.0,
                        "delta": 5.0,
                        "balance": 1.0,
                        "obs": "up",
                        "terms": [
                            {"id": "GO:1", "label": "t", "nl": 5.0, "effect": 1, "obs": "up"}
                        ],
                    }
                ],
            }
        ],
        "unlinked": [],
    }
    svg = render_nova_svg(payload, "light", id_prefix="z", center_label="<x>", subgroup_labels=True)
    ET.fromstring(svg)
    assert "<b>" not in svg and "&lt;x&gt;" in svg


def test_colours_match_the_app_palette():
    assert claim_color("Barrier") == ("#47BDD7", "#2A6478")
    # Healing has no fixed colour: djb2 fallback, as novaPalette.ts claimColor().
    assert claim_color("Healing") == ("#66B8E1", "#00659B")
    assert balance_color(1) == "#34B596" and balance_color(-1) == "#FF5F5F"
    assert balance_color(0) == "#C9D1D9"


# ── What the wheel says is what was computed ───────────────────────────────


def _group(name, nl, delta, obs="up", n_terms=1):
    s_pro, s_anti = max(delta, 0.0), max(-delta, 0.0)
    return {
        "name": name,
        "nl": nl,
        "s_pro": s_pro,
        "s_anti": s_anti,
        "delta": delta,
        "balance": delta / (s_pro + s_anti) if s_pro + s_anti else 0.0,
        "obs": obs,
        "terms": [
            {
                "id": f"GO:{i:07d}",
                "label": f"Term {name} {i}",
                "nl": nl / n_terms,
                "effect": 1 if delta > 0 else -1 if delta < 0 else 0,
                "obs": obs,
            }
            for i in range(n_terms)
        ],
    }


def _payload(**overrides):
    groups = [
        _group("Group A", 6.0, 4.0),
        _group("Group B", 4.0, 0.0),
        _group("Group C", 3.0, -2.0),
    ]
    payload = {
        "contrast": "AL-7_vs_Vehicle",
        "condition": "AL-7",
        "alpha": 0.05,
        "claims": [
            {
                "name": "Firming",
                "s_pro": 10.0,
                "s_anti": 2.0,
                "coherence": 0.83,
                "n_for": 1,
                "n_against": 1,
                "n_balanced": 1,
                "best_subgroup": "Group A",
                "best_nl": 6.0,
                "subgroups": groups,
            },
            {
                "name": "Barrier",
                "s_pro": 8.0,
                "s_anti": 0.0,
                "coherence": 1.0,
                "n_for": 1,
                "n_against": 0,
                "n_balanced": 0,
                "best_subgroup": "Group D",
                "best_nl": 8.0,
                "subgroups": [_group("Group D", 8.0, 8.0)],
            },
        ],
        "unlinked": [_group("Group U", 3.0, 0.0)],
        "unlinked_total": 1,
    }
    payload.update(overrides)
    return payload


def test_wheel_has_an_axis_title_and_no_decorative_dots(nova):
    svg = render_nova_svg(nova, "light", id_prefix="x")
    assert "−log10 combined p (Stouffer, unsigned)" in svg
    # Every dot is a term or a sub-group: none is drawn to fill a sector. The only other circles
    # are the significance rings and the centre disc.
    n_terms = sum(len(s["terms"]) for c in nova["claims"] for s in c["subgroups"])
    n_terms += sum(len(s["terms"]) for s in nova["unlinked"])
    n_groups = sum(len(c["subgroups"]) for c in nova["claims"]) + len(nova["unlinked"])
    circles = re.findall(r"<circle ", svg)
    assert n_terms + n_groups < len(circles) <= n_terms + n_groups + 6
    assert 'r="0.70"' not in svg  # the former dust
    for theme in ("light", "dark"):
        assert "−log10 combined p (Stouffer, unsigned)" in render_nova_svg(nova, theme)


def test_zero_against_is_written(nova):
    svg = render_nova_svg(_payload(), "light", id_prefix="x")
    assert "1 sub-group for" in svg and "0 against" in svg
    assert "1 against" in svg and "1 balanced" in svg
    # A claim with no sub-group against it says "0 against" too (the reference mock), but each of
    # them has a contrary term: nothing is said about the absence of one.
    mock = render_nova_svg(nova, "light", id_prefix="y")
    assert "0 against" in mock and "not a confirmation" not in mock
    # A claim with no contrary term at all (S_anti = 0) carries the note: Barrier of _payload().
    note = render_nova_svg(_payload(), "light", id_prefix="n")
    assert "Only terms present in the enrichment results can count against a claim;" in note
    assert "no contrary term is not a confirmation." in note
    # Every claim has a contrary term: nothing to qualify.
    both = {**_payload(), "claims": _payload()["claims"][:1]}
    assert "not a confirmation" not in render_nova_svg(both, "light", id_prefix="z")


def test_subgroups_are_not_called_pathways(nova):
    for payload in (nova, _payload()):
        for wheel_only in (True, False):
            svg = render_nova_svg(payload, "light", id_prefix="x", wheel_only=wheel_only)
            assert " pathways" not in svg and ">1 pathway" not in svg
    assert ">5 sub-groups<" in render_nova_svg(nova, "light", id_prefix="x")
    assert ">1 sub-group<" in render_nova_svg(_payload(), "light", id_prefix="x")


def test_demo_wheel_is_marked(nova):
    demo = {**nova, "provenance": {"demo": True}}
    for theme in ("light", "dark"):
        for wheel_only in (True, False):
            assert ">DEMO<" in render_nova_svg(demo, theme, wheel_only=wheel_only)
    assert ">DEMO<" not in render_nova_svg(nova, "light")
    assert ">DEMO<" not in render_nova_svg({**nova, "provenance": {"demo": False}}, "light")


def test_the_disc_never_names_a_condition_it_was_not_given():
    svg = render_nova_svg({"claims": [], "unlinked": []}, "light", id_prefix="e")
    assert "Active" not in svg
    named = render_nova_svg({"claims": [], "unlinked": [], "condition": "Cond"}, "light")
    assert ">Cond<" in named


def test_a_comparison_without_enrichment_is_not_drawn_as_a_comparison_without_claim():
    empty = {"status": "no_enrichment", "claims": [], "unlinked": []}
    svg = render_nova_svg(empty, "light", id_prefix="e")
    assert "No enrichment result for this comparison" in svg
    assert "No claim with significant evidence" not in svg


def test_a_balanced_subgroup_is_neutral_in_the_claim_cards():
    svg = render_nova_svg(_payload(), "light", id_prefix="p", wheel_only=False)

    def fill_of(text):
        return re.search(rf'fill="(#[0-9A-Fa-f]{{6}})"[^>]*>{text}</text>', svg).group(1)

    ink, against = "#3A4150", "#7D8B9C"
    assert fill_of("Group A") == ink  # for the claim
    assert fill_of("Group B") == ink  # balanced: neither for nor against
    assert fill_of("Group C") == against  # against the claim
    # The balanced sub-group's arrow is not the grey of the evidence against.
    arrows = re.findall(r'fill="(#[0-9A-Fa-f]{6})"[^>]*>(?:↑|↓)</text>', svg)
    assert arrows.count(against) == 1


# ── Endpoint ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_overrides():
    from app.main import app

    yield
    app.dependency_overrides.clear()


def _client(with_module: bool):
    from tests.test_project_access_control import make_client, make_fixtures, make_user

    caller = make_user()
    caller.cosmetics_module_enabled = with_module
    dataset, project = make_fixtures(owner_id=caller.id)
    return make_client(caller=caller, dataset=dataset, project=project, member=None), dataset


async def test_svg_endpoint_requires_the_cosmetics_module():
    client, dataset = _client(with_module=False)
    fetch = AsyncMock(return_value=[])
    with patch("app.services.cosmomics_nova.fetch_term_hits", new=fetch):
        async with client as c:
            resp = await c.get(
                f"/api/v1/datasets/{dataset.id}/comparisons/A_vs_B/cosmomics-nova.svg"
            )
    assert resp.status_code == 403
    fetch.assert_not_awaited()


@pytest.mark.parametrize("theme", ["light", "dark"])
async def test_svg_endpoint_returns_the_wheel(theme):
    client, dataset = _client(with_module=True)
    hits = [
        TermHit(
            "positive regulation of cell adhesion (GO:0045785)",
            "positive regulation of cell adhesion",
            1e-6,
            "up",
        )
    ]
    with patch("app.services.cosmomics_nova.fetch_term_hits", new=AsyncMock(return_value=hits)):
        async with client as c:
            resp = await c.get(
                f"/api/v1/datasets/{dataset.id}/comparisons/P1_vs_control/cosmomics-nova.svg",
                params={"theme": theme},
            )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert resp.headers["cache-control"] == "no-store"
    ET.fromstring(resp.text)


async def test_svg_endpoint_rejects_an_unknown_theme():
    client, dataset = _client(with_module=True)
    async with client as c:
        resp = await c.get(
            f"/api/v1/datasets/{dataset.id}/comparisons/A_vs_B/cosmomics-nova.svg",
            params={"theme": "sepia"},
        )
    assert resp.status_code == 422
