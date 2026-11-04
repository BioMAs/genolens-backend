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
    with patch("app.api.endpoints.cosmomics.fetch_term_hits", new=fetch):
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
    with patch("app.api.endpoints.cosmomics.fetch_term_hits", new=AsyncMock(return_value=hits)):
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
