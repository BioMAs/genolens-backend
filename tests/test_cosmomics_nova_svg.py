"""
Server-side SVG of the CosmOmics Pathway Nova (`render_nova_svg`) and its endpoint.
"""

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services import cosmomics_nova_svg as nova_svg
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


# ── R-F1: the poster header says what the p-values are from the enrichment's kind ─────

P_ANNODB = "padj (Benjamini–Hochberg, GenoLens enrichment)"


def _header(payload) -> str:
    """The line under the poster's title (what the p-values are, and α)."""
    svg = render_nova_svg(payload, "light", id_prefix="h", wheel_only=False)
    return " ".join(re.findall(r"<text[^>]*y=\"126.0\"[^>]*>([^<]*)</text>", svg))


def _with_kind(nova, kind, **extra):
    return {
        **nova,
        "p_source": P_ANNODB,  # a stale or hand-made p_source never decides the header
        "provenance": {"enrichment_source": kind, "padj_columns": [], **extra},
    }


def test_the_header_follows_the_enrichment_source_not_the_p_source(nova):
    assert _header(_with_kind(nova, "annodb")) == f"{P_ANNODB} · α = 0.05"
    legacy = _header(_with_kind(nova, "legacy"))
    assert "earlier GenoLens enrichment" in legacy and "not recorded" in legacy
    assert "Benjamini" not in legacy and "α = 0.05" in legacy
    uploaded = _header(_with_kind(nova, "uploaded", padj_columns=["qvalue"]))
    assert uploaded == "adjusted p from uploaded tables (column qvalue) · α = 0.05"
    assert "Benjamini" not in uploaded


def test_the_header_of_a_comparison_with_no_enrichment_says_only_that_none_is_stored(nova):
    for payload in (
        _with_kind(nova, "none"),
        {**_with_kind(nova, "annodb"), "status": "no_enrichment"},
        {**_with_kind(nova, "legacy"), "status": "no_enrichment"},
    ):
        header = _header(payload)
        assert header == "No enrichment result is stored for this comparison"
        assert "Benjamini" not in header and "α" not in header


# ── D6: the labels keep apart from each other and off the wheel, inside the frame ──────────────

SVG = "{http://www.w3.org/2000/svg}"
DISC = nova_svg.R + 4  # the wheel and a 4 px margin: no label comes closer


def _pathway(name, nl, s_pro, s_anti):
    return {
        "name": name,
        "nl": nl,
        "s_pro": s_pro,
        "s_anti": s_anti,
        "delta": s_pro - s_anti,
        "balance": (s_pro - s_anti) / (s_pro + s_anti),
        "obs": "up",
        "terms": [
            {"id": f"GO:{name}", "label": f"Term of {name}", "nl": nl, "effect": 1, "obs": "up"}
        ],
    }


def _wheel_payload(names, weights):
    """One claim per name, sector width ∝ weight (80 % for the claim), three unlinked groups."""
    claims = []
    for n, (name, weight) in enumerate(zip(names, weights), start=1):
        s_pro = 0.8 * weight
        nl = 3 + n % 3
        claims.append(
            {
                "name": name,
                "importance": weight,
                "s_pro": s_pro,
                "s_anti": weight - s_pro,
                "coherence": 0.8,
                "n_for": 1,
                "n_against": 0,
                "best_nl": nl,
                "best_subgroup": f"Pathway {n}",
                "subgroups": [_pathway(f"Pathway {n}", nl, s_pro, weight - s_pro)],
            }
        )
    return {
        "contrast": "Group_A vs Group_B",
        "condition": "Group_A",
        "alpha": 0.05,
        "claims": claims,
        "unlinked": [
            _pathway("Unlinked 1", 4, 5, 1),
            _pathway("Unlinked 2", 3, 4, 1),
            _pathway("Unlinked 3", 2.5, 3, 1),
        ],
        "unlinked_total": 3,
    }


# Three broad claims, then six narrow neighbours of under 4 % of the weight each, 20 characters per
# name: their labels are far wider than their sectors and crowd the left of the wheel.
NINE_WEIGHTS = [30, 25, 20, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4]


def _nine_claims():
    return _wheel_payload([f"Claim {n} long name xx" for n in range(1, 10)], NINE_WEIGHTS)


def _twelve_equal_claims():
    """Twelve claims of about the same weight: the labels crowd the top right and rise past the
    crop, and those of the left crowd the bottom."""
    names = [f"Claim {n} long name xx" for n in range(1, 13)]
    return _wheel_payload(names, [10 - 0.01 * i for i in range(12)])


def _three_broad_claims():
    return _wheel_payload(["Claim 1", "Claim 2", "Claim 3"], [30, 25, 20])


def _with_zero_against_note(payload):
    """The last claim has no contrary term at all: the wheel carries the note about it as well."""
    *others, last = payload["claims"]
    last = {**last, "s_pro": last["s_pro"] + last["s_anti"], "s_anti": 0.0}
    return {**payload, "claims": [*others, last]}


def _drawn(payload):
    wheel = nova_svg._Nova(payload, "light", "t", "", 0.05)
    wheel.draw()
    wheel.annotations()
    return wheel


def _label_boxes(payload):
    return _drawn(payload).label_boxes


def _distance_to_centre(box):
    x0, y0, x1, y1 = box
    return math.hypot(
        max(x0 - nova_svg.CX, 0, nova_svg.CX - x1), max(y0 - nova_svg.CY, 0, nova_svg.CY - y1)
    )


def _apart(a, b):
    return a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3]


def _viewbox(svg):
    return [float(v) for v in ET.fromstring(svg).get("viewBox").split()]


def _label_texts(svg):
    """(text, x, y, anchor) of every claim label line, in drawing order (21 px titles, 12.5 notes)."""
    return [
        (t.text, float(t.get("x")), float(t.get("y")), t.get("text-anchor"))
        for t in ET.fromstring(svg).iter(f"{SVG}text")
        if t.get("font-size") in ("21", "12.5") and not t.text.startswith("Radius")
    ]


def _leaders(svg):
    return [e for e in ET.fromstring(svg).iter(f"{SVG}line") if e.get("stroke-width") == "0.8"]


def test_claim_labels_never_overlap_each_other_or_the_wheel():
    payload = _nine_claims()
    total = sum(NINE_WEIGHTS)
    assert sum(1 for w in NINE_WEIGHTS if w / total < 0.04) == 6
    assert all(len(c["name"]) == 20 for c in payload["claims"])

    boxes = _label_boxes(payload)
    assert len(boxes) == 10  # nine claims and the "not linked" sector
    for i, box in enumerate(boxes):
        assert _distance_to_centre(box) >= DISC - 1e-9
        for other in boxes[i + 1 :]:
            assert _apart(box, other)

    for wheel_only in (True, False):
        x0, y0, width, height = _viewbox(
            render_nova_svg(payload, "light", id_prefix="x", wheel_only=wheel_only)
        )
        for bx0, by0, bx1, by1 in boxes:
            # inside the frame, with the 8 px margin the frame keeps around every label
            assert x0 <= bx0 - 8 and bx1 + 8 <= x0 + width
            assert y0 <= by0 - 8 and by1 + 8 <= y0 + height


def test_a_label_that_stands_free_keeps_its_reference_place():
    """Three broad claims leave their labels clear of each other and of the wheel: they stay exactly
    where the wheel drew them before labels were pushed apart (positions read off the code as it
    stood, before the change). Only "Not linked to a claim" stood over the wheel."""
    svg = render_nova_svg(_three_broad_claims(), "light", id_prefix="x")
    where_they_were = [
        ("Claim 1", 500.0, 199.0, "middle"),
        ("1 sub-group for · 0 against", 500.0, 217.0, "middle"),
        ("Claim 2", 836.0, 728.5, "start"),
        ("1 sub-group for", 836.0, 746.5, "start"),
        ("0 against", 836.0, 762.5, "start"),
        ("Claim 3", 164.0, 854.2, "end"),
        ("1 sub-group for", 164.0, 872.2, "end"),
        ("0 against", 164.0, 888.2, "end"),
    ]
    texts = _label_texts(svg)
    for (text, x, y, anchor), got in zip(where_they_were, texts):
        assert (got[0], got[3]) == (text, anchor)
        assert (got[1], got[2]) == pytest.approx((x, y), abs=0.5)
    assert [t[0] for t in texts[8:]] == ["Not linked to a claim", "3 sub-groups"]
    # None of these three moved: the one leader line is that of "Not linked to a claim", which
    # stood over the grey sector (top at 505) and now ends above it.
    assert len(_leaders(svg)) == 1
    boxes = _label_boxes(_three_broad_claims())
    assert all(_distance_to_centre(box) >= DISC - 1e-9 for box in boxes)
    assert boxes[3][1] == pytest.approx(274.67, abs=0.5)


def test_the_centre_names_the_condition_in_a_title():
    svg = render_nova_svg(
        {"claims": [], "unlinked": [], "condition": "Treatment_C"}, "light", id_prefix="c"
    )
    root = ET.fromstring(svg)
    assert [t.text for t in root.iter(f"{SVG}title")] == ["Treatment_C"]
    assert ">Treatme…<" in svg
    # The disc's text is the cut name alone: the full one lives in the title, nowhere else.
    assert "Treatment_C" not in [t.text for t in root.iter(f"{SVG}text")]
    # The rule is unchanged: whole up to 8 characters, 13 px up to 6 and 11 px above.
    for name, shown, size in (
        ("Sample", "Sample", "13"),
        ("Sample_1", "Sample_1", "11"),
        ("Treatment", "Treatme…", "11"),
    ):
        one = ET.fromstring(
            render_nova_svg({"claims": [], "unlinked": [], "condition": name}, "light")
        )
        centre = [t for t in one.iter(f"{SVG}text") if t.get("font-weight") == "700"]
        assert [(t.text, t.get("font-size")) for t in centre] == [(shown, size)]
        assert [t.text for t in one.iter(f"{SVG}title")] == [name]
    # The title is escaped.
    odd = render_nova_svg({"claims": [], "unlinked": []}, "light", center_label='A&B<c>"d"_long')
    assert "<title>A&amp;B&lt;c&gt;&quot;d&quot;_long</title>" in odd
    assert [t.text for t in ET.fromstring(odd).iter(f"{SVG}title")] == ['A&B<c>"d"_long']
    # A wheel with no name draws the disc and neither text nor title.
    bare = render_nova_svg({"claims": [], "unlinked": [], "condition": ""}, "light", id_prefix="b")
    assert 'r="26"' in bare and "<title" not in bare and 'font-weight="700"' not in bare


def test_the_frame_grows_to_hold_every_label_and_the_cards_follow():
    # Labels that stand inside the crop leave it as the reference drew it.
    free = _three_broad_claims()
    wheel = render_nova_svg(free, "light", id_prefix="x")
    poster = render_nova_svg(free, "light", id_prefix="x", wheel_only=False)
    assert _viewbox(wheel) == [0, 130, 1000, 850]
    assert _viewbox(poster) == [0, 0, 1000, 1196]

    # Narrow neighbours stack their labels below the crop: the frame follows, 8 px past the last.
    crowded = _nine_claims()
    drawn = _drawn(crowded)
    bottom = max(box[3] for box in drawn.label_boxes + drawn.footnote_boxes)
    assert bottom > 980
    svg = render_nova_svg(crowded, "light", id_prefix="x")
    height = math.ceil(bottom + 8) - 130
    assert _viewbox(svg) == [0, 130, 1000, height]
    root = ET.fromstring(svg)
    assert (root.get("width"), root.get("height")) == ("1000", str(height))  # still 1 px per unit

    # Full view: the poster is as much taller; the claim cards and the legend start under the
    # labels and the footnotes.
    extra = math.ceil(bottom + 8) - 980
    full = render_nova_svg(crowded, "light", id_prefix="x", wheel_only=False)
    assert _viewbox(full) == [0, 0, 1000, 1056 + 140 * 3 + extra]
    first_card = float(re.search(r'<rect x="36" y="([\d.]+)"', full).group(1))
    assert first_card == 1016 + extra - 18
    assert first_card > bottom + 8


def test_labels_pushed_above_the_crop_move_the_wheel_down_in_the_poster_not_the_header():
    # The labels crowd the top right and rise past the crop.
    claims = _twelve_equal_claims()
    drawn = _drawn(claims)
    top = min(box[1] for box in drawn.label_boxes)
    bottom = max(box[3] for box in drawn.label_boxes + drawn.footnote_boxes)
    assert top < 130
    above, below = 130 - math.floor(top - 8), math.ceil(bottom + 8) - 980
    assert (above, below) == (45, 68)

    wheel = render_nova_svg(claims, "light", id_prefix="x")
    assert _viewbox(wheel) == [0, 130 - above, 1000, 850 + above + below]

    poster = render_nova_svg(claims, "light", id_prefix="x", wheel_only=False)
    assert _viewbox(poster) == [0, 0, 1000, 1056 + 140 * 3 + above + below]
    # The wheel and what follows move down; the header does not.
    assert f'<g transform="translate(0 {above})">' in poster
    assert poster.index('y="72"') < poster.index("<g transform")
    assert top + above - 8 >= 130  # the highest label clears the header band
    first_card = float(re.search(r'<rect x="36" y="([\d.]+)"', poster).group(1))
    assert first_card == 1016 + above + below - 18 > bottom + above + 8
    # A wheel whose labels stay inside the crop is drawn without the extra group.
    assert "<g transform" not in render_nova_svg(_three_broad_claims(), "light", wheel_only=False)


FOOTNOTES = ("Radius:", "Only terms present", "no contrary term")


def _wheel_offset(svg):
    """How far the poster moved the wheel down (it does when labels rise above the crop)."""
    moved = re.search(r'<g transform="translate\(0 (\d+)\)">', svg)
    return int(moved.group(1)) if moved else 0


def _footnotes_in(svg):
    """The box of every footnote line in the SVG's own coordinates, 0.6 em per character (a little
    wider than the code's own estimate), and its baseline."""
    dy = _wheel_offset(svg)
    notes = []
    for t in ET.fromstring(svg).iter(f"{SVG}text"):
        if t.text and t.text.startswith(FOOTNOTES):
            x, y, size = float(t.get("x")), float(t.get("y")) + dy, float(t.get("font-size"))
            notes.append(((x, y - size, x + 0.6 * size * len(t.text), y + 0.3 * size), y))
    return notes


@pytest.mark.parametrize("wheel_only", [True, False])
@pytest.mark.parametrize("zero_against", [False, True])
@pytest.mark.parametrize("make", [_nine_claims, _twelve_equal_claims])
def test_the_footnotes_move_under_the_labels_that_reach_them(make, zero_against, wheel_only):
    payload = _with_zero_against_note(make()) if zero_against else make()
    svg = render_nova_svg(payload, "light", id_prefix="x", wheel_only=wheel_only)
    notes = _footnotes_in(svg)
    assert len(notes) == (3 if zero_against else 1)
    dy = _wheel_offset(svg)
    labels = [(x0, y0 + dy, x1, y1 + dy) for x0, y0, x1, y1 in _label_boxes(payload)]
    x0, y0, width, height = _viewbox(svg)
    for box, _ in notes:
        assert all(_apart(box, label) for label in labels)
        # inside the frame, with the 8 px margin under the last line
        assert x0 <= box[0] and box[2] <= x0 + width
        assert y0 <= box[1] and box[3] + 8 <= y0 + height
    # The block stands 8 px under the lowest label, its lines at their usual spacing.
    assert notes[0][0][1] == pytest.approx(max(label[3] for label in labels) + 8, abs=0.1)
    assert [y - notes[0][1] for _, y in notes] == pytest.approx([0, 16, 30][: len(notes)], abs=0.1)


def test_footnotes_that_no_label_reaches_stay_where_the_reference_puts_them():
    svg = render_nova_svg(_with_zero_against_note(_three_broad_claims()), "light", id_prefix="x")
    assert [y for _, y in _footnotes_in(svg)] == [944, 960, 974]
    assert _viewbox(svg) == [0, 130, 1000, 850]  # the crop already holds them
    poster = render_nova_svg(_three_broad_claims(), "light", id_prefix="x", wheel_only=False)
    assert [y for _, y in _footnotes_in(poster)] == [944]


def test_a_label_pushed_far_from_its_sector_is_tied_to_it_by_a_leader_line():
    svg = render_nova_svg(_three_broad_claims(), "light", id_prefix="x")
    (line,) = _leaders(svg)  # "Not linked to a claim" is the only label that had to move
    assert line.get("stroke") == "#7D8B9C" and line.get("opacity") == "0.6"
    assert line.get("marker-start") is None and line.get("marker-end") is None
    x1, y1, x2, y2 = (float(line.get(k)) for k in ("x1", "y1", "x2", "y2"))
    # It starts on the rim, 6 px out (the sector's mid angle), and ends on the nearest box point.
    assert math.hypot(x1 - nova_svg.CX, y1 - nova_svg.CY) == pytest.approx(nova_svg.R + 6, abs=0.1)
    box = _label_boxes(_three_broad_claims())[3]
    assert (x2, y2) == pytest.approx(
        (min(max(x1, box[0]), box[2]), min(max(y1, box[1]), box[3])), abs=0.1
    )
    # Drawn under the labels, so their halo keeps the text readable.
    assert svg.index('stroke-width="0.8"') < svg.index('font-size="21"')
    # A leader carries its label's title colour: a claim's own, "Not linked to a claim" in grey.
    crowded = render_nova_svg(_nine_claims(), "light", id_prefix="x")
    titles = [e for e in ET.fromstring(crowded).iter(f"{SVG}text") if e.get("font-size") == "21"]
    leaders = _leaders(crowded)
    moved = [
        title
        for title, box, ref in zip(titles, _label_boxes(_nine_claims()), NINE_BACKEND)
        if abs(box[1] - ref[0]) > 12
    ]
    assert [e.get("stroke") for e in leaders] == [t.get("fill") for t in moved]


def test_a_label_is_pushed_off_the_wheel_by_the_overlap_plus_four():
    half = math.sqrt(DISC**2 - 50**2)  # half the disc's height over the box's nearest column
    below = (nova_svg.CX + 50, nova_svg.CY + 10, nova_svg.CX + 150, nova_svg.CY + 50)
    above = (nova_svg.CX + 50, nova_svg.CY - 50, nova_svg.CX + 150, nova_svg.CY - 10)
    pushed_below, pushed_above = nova_svg._resolve_label_overlaps([below, above])
    assert pushed_below[1] == pytest.approx(nova_svg.CY + half + 4, abs=1e-9)
    assert pushed_above[3] == pytest.approx(nova_svg.CY - half - 4, abs=1e-9)
    assert pushed_below[3] - pushed_below[1] == 40 and pushed_above[3] - pushed_above[1] == 40
    assert pushed_below[0::2] == below[0::2] and pushed_above[0::2] == above[0::2]  # vertical only


def test_an_earlier_label_is_dealt_with_before_the_wheel():
    """`first` ends 2 px inside the chord of the disc over `second`'s column: `second` hits both.
    Past `first` + 4 px it is clear of the disc (top = chord end + 2), so a second push never
    happens. Had the disc come first, `second` would stop at the chord end + 4."""
    cx, cy = nova_svg.CX, nova_svg.CY
    half = math.sqrt(DISC**2 - 300**2)
    first = (cx + 340, cy + 100, cx + 460, cy + half - 2)
    second = (cx + 300, cy + 120, cx + 420, cy + 160)
    placed_first, placed_second = nova_svg._resolve_label_overlaps([first, second])
    assert placed_first == first
    assert placed_second[1] == pytest.approx(cy + half + 2, abs=1e-9)  # = first's bottom + 4
    assert placed_second[3] == pytest.approx(cy + half + 42, abs=1e-9)
    assert _apart(placed_first, placed_second) and _distance_to_centre(placed_second) >= DISC


def test_a_label_that_touches_nothing_stays_put_the_tangent_to_the_wheel_included():
    cx, cy = nova_svg.CX, nova_svg.CY
    tangent = (cx - 60, cy - DISC - 42, cx + 60, cy - DISC)
    far = (cx + DISC + 10, cy - 20, cx + DISC + 130, cy + 20)
    assert nova_svg._resolve_label_overlaps([tangent, far]) == [tangent, far]


def test_a_label_gives_up_after_forty_pushes():
    """A stack of 45 boxes in the way: the label stays where its fortieth push left it (still
    touching the 41st box) instead of looping."""
    stack = [(0.0, 600.0 + 12 * i, 100.0, 610.0 + 12 * i) for i in range(45)]
    label = (0.0, 600.0, 100.0, 640.0)
    *_, placed = nova_svg._resolve_label_overlaps([*stack, label])
    assert placed[1] == 614 + 12 * 39
    assert not _apart(placed, stack[40])


# The frontend's pinned placement of the nine-claim payload (frontend commit aa6adef,
# novaLayout.test.ts): (sector mid angle, reference top, box x0, box x1, final top, leader line
# (x1, y1, x2, y2) or None). Its boxes are 42 px tall: its notes never carry "· 0 against", so no
# label breaks in two lines.
FRONTEND_NINE = [
    (-90.00, 179.00, 380.00, 620.00, 179.00, None),
    (3.97, 565.50, 750.00, 990.00, 780.49, (835.19, 578.26, 835.19, 780.49)),
    (81.22, 892.85, 434.05, 674.05, 892.85, None),
    (121.51, 844.80, 74.99, 314.99, 844.80, None),
    (133.04, 801.73, 18.39, 258.39, 801.73, None),
    (144.24, 749.89, 10.00, 250.00, 890.80, (227.35, 751.37, 227.35, 890.80)),
    (155.10, 692.05, 10.00, 250.00, 936.80, (195.23, 696.47, 195.23, 936.80)),
    (165.63, 628.87, 10.00, 250.00, 982.80, (174.52, 638.40, 174.52, 982.80)),
    (175.82, 566.79, 10.00, 250.00, 1028.80, (164.89, 579.48, 164.89, 1028.80)),
    (198.83, 418.71, 10.00, 262.00, 274.67, (181.99, 446.52, 181.99, 316.67)),
]


def test_the_push_rule_gives_the_frontend_placement_from_the_same_boxes():
    """Same rule on the frontend's own boxes: same final boxes, same leader lines (0.5 px)."""
    boxes = [(x0, ref, x1, ref + 42) for _, ref, x0, x1, _, _ in FRONTEND_NINE]
    placed = nova_svg._resolve_label_overlaps(boxes)
    for (mid, ref, x0, x1, top, leader), box in zip(FRONTEND_NINE, placed):
        assert box == pytest.approx((x0, top, x1, top + 42), abs=0.5)
        assert (abs(box[1] - ref) > 12) == (leader is not None)
        if leader:
            assert nova_svg._leader_line(mid, box) == pytest.approx(leader, abs=0.5)


# What the backend's own wheel comes to for that payload: (reference top, final box, leader line).
# The sector order, the mid angles, the reference tops and the title widths agree with the
# frontend's; the boxes do not, for one reason: the backend's note always reads "N sub-group(s) for
# · 0 against". Where that spills over the wheel (Claims 2, 4 to 9) the label breaks it in two and
# stands 58 px tall, not 42, and Claims 4 and 5 move their right edge to x 250 (the frontend keeps
# 315 and 258). The pushes follow from the taller boxes; the rule itself is tested above.
NINE_BACKEND = [
    (179.00, (380.00, 179.00, 620.00, 221.00), None),
    (565.50, (750.00, 780.49, 990.00, 838.49), (835.19, 578.26, 835.19, 780.49)),
    (892.85, (434.05, 892.85, 674.05, 934.85), None),
    (844.80, (10.00, 844.80, 250.00, 902.80), None),
    (801.73, (10.00, 906.80, 250.00, 964.80), (270.67, 800.57, 250.00, 906.80)),
    (749.89, (10.00, 780.49, 250.00, 838.49), (227.35, 751.37, 227.35, 780.49)),
    (692.05, (10.00, 968.80, 250.00, 1026.80), (195.23, 696.47, 195.23, 968.80)),
    (628.87, (10.00, 1030.80, 250.00, 1088.80), (174.52, 638.40, 174.52, 1030.80)),
    (566.79, (10.00, 1092.80, 250.00, 1150.80), (164.89, 579.48, 164.89, 1092.80)),
    (418.71, (10.00, 274.67, 262.00, 316.67), (181.99, 446.52, 181.99, 316.67)),
]


def test_the_nine_claim_wheel_is_pinned():
    wheel = nova_svg._Nova(_nine_claims(), "light", "t", "", 0.05)
    svg_lines = wheel.draw()
    assert len(wheel.label_boxes) == len(wheel.label_leaders) == len(NINE_BACKEND) == 10
    for (ref, box, leader), got, got_leader in zip(
        NINE_BACKEND, wheel.label_boxes, wheel.label_leaders
    ):
        assert got == pytest.approx(box, abs=0.5)
        assert (abs(got[1] - ref) > 12) == (leader is not None) == (got_leader is not None)
        if leader:
            assert got_leader == pytest.approx(leader, abs=0.5)
    assert sum(1 for line in svg_lines if line.startswith("<line ")) == 7
    # The text moved with its box: the title stands 20 px under its top.
    texts = _label_texts("<svg xmlns='http://www.w3.org/2000/svg'>" + "".join(svg_lines) + "</svg>")
    titles = [t for t in texts if t[0].startswith(("Claim", "Not linked"))]
    for title, box in zip(titles, wheel.label_boxes):
        assert title[2] == pytest.approx(box[1] + 20, abs=0.06)


@pytest.mark.parametrize(
    "moved, has_leader", [(12.0, False), (-12.0, False), (12.01, True), (-12.01, True)]
)
def test_a_label_moved_exactly_twelve_px_has_no_leader_one_moved_just_over_has_one(
    monkeypatch, moved, has_leader
):
    """Nothing is pushed but the first label (top at 179.0, exactly), by `moved`: the threshold is
    read on that distance, up or down, and the label is not free as soon as it is over 12 px."""

    def push_the_first_label(boxes):
        x0, y0, x1, y1 = boxes[0]
        return [(x0, y0 + moved, x1, y1 + moved), *boxes[1:]]

    monkeypatch.setattr(nova_svg, "_resolve_label_overlaps", push_the_first_label)
    wheel = nova_svg._Nova(_three_broad_claims(), "light", "t", "", 0.05)
    drawn = wheel.draw()
    assert (wheel.label_leaders[0] is not None) == has_leader
    assert wheel.label_leaders[1:] == [None, None, None]
    assert sum(1 for line in drawn if line.startswith("<line ")) == int(has_leader)
    title = next(line for line in drawn if ">Claim 1<" in line)
    assert float(re.search(r' y="([\d.]+)"', title).group(1)) == pytest.approx(
        199 + moved, abs=0.06
    )


def _subgroup_labels(svg):
    """The box of every sub-group label (11.5 px), by name, as the code estimates it."""
    boxes = {}
    for t in ET.fromstring(svg).iter(f"{SVG}text"):
        if t.get("font-size") == "11.5":
            x, y, w = float(t.get("x")), float(t.get("y")), len(t.text) * 11.5 * 0.56
            start = t.get("text-anchor") == "start"
            boxes[t.text] = (
                (x - 2, y - 10, x + w + 2, y + 3) if start else (x - w - 2, y - 10, x + 2, y + 3)
            )
    return boxes


def test_subgroup_labels_keep_clear_of_the_claim_labels_where_these_ended_up():
    payload = _nine_claims()
    labels = _label_boxes(payload)
    names = _subgroup_labels(render_nova_svg(payload, "light", subgroup_labels=True))
    assert len(names) == 9
    assert all(_apart(box, label) for box in names.values() for label in labels)


def test_a_subgroup_label_gives_way_to_a_claim_label_that_was_pushed_over_its_place(monkeypatch):
    payload = _three_broad_claims()
    usual = _subgroup_labels(render_nova_svg(payload, "light", subgroup_labels=True))["Pathway 1"]
    # The first claim label ends up right over where that sub-group label goes.
    over = (usual[0] - 1, usual[1] - 1, usual[2] + 1, usual[3] + 1)
    monkeypatch.setattr(nova_svg, "_resolve_label_overlaps", lambda boxes: [over, *boxes[1:]])
    moved = _subgroup_labels(render_nova_svg(payload, "light", subgroup_labels=True))["Pathway 1"]
    assert moved != usual and _apart(moved, over)
