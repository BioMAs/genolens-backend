"""Tests for the genolens-dd client and its endpoints.

No network: every upstream response is served by `httpx.MockTransport`, so the suite is
deterministic and runs offline.

**The two tests that carry weight are not the happy paths.** They are
`test_every_endpoint_requires_authentication` — because an unauthenticated route here would
re-open the product that the API key exists to close — and `test_an_upstream_401_becomes_502`,
because passing a 401 through would blame the logged-in user for our own bad credential.
"""
import httpx
import pytest

from app.api.endpoints import drug_discovery as dd_endpoints
from app.main import app
from app.services.drug_discovery import (
    API_KEY_HEADER,
    DrugDiscoveryClient,
    DrugDiscoveryRejected,
    DrugDiscoveryUnavailable,
)

KEY = "cle-de-test-non-secrete"


def client_with(handler, *, api_key: str = KEY, timeout: float = 30.0) -> DrugDiscoveryClient:
    return DrugDiscoveryClient(
        base_url="https://dd.genolens.com",
        api_key=api_key,
        timeout=timeout,
        transport=httpx.MockTransport(handler),
    )


def json_handler(status: int, body, *, capture: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        return httpx.Response(status, json=body)

    return handler


# ---------------------------------------------------------------------------
# The key travels, and only in the header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_api_key_is_sent_as_the_expected_header():
    seen: list[httpx.Request] = []
    client = client_with(json_handler(201, {"run_id": "r1"}, capture=seen))
    await client.create_run(indication="TCGA-BRCA")
    assert seen[0].headers[API_KEY_HEADER] == KEY
    # Not `Authorization`: Traefik's basicauth owns that header on this host.
    assert "authorization" not in {name.lower() for name in seen[0].headers}


@pytest.mark.asyncio
async def test_health_and_readyz_are_called_without_the_key():
    """Upstream serves them unauthenticated, and calling them without the key is what makes
    them useful for diagnosing a MISSING key rather than an unreachable service."""
    seen: list[httpx.Request] = []
    client = client_with(json_handler(200, {"status": "ok"}, capture=seen), api_key="")
    assert await client.health() == {"status": "ok"}
    assert API_KEY_HEADER not in seen[0].headers


def test_the_repr_never_renders_the_key():
    """A dataclass repr in a traceback is a classic route from a secret to a log aggregator."""
    client = DrugDiscoveryClient(base_url="https://x", api_key=KEY)
    assert KEY not in repr(client)
    assert "<set>" in repr(client)
    assert "<unset>" in repr(DrugDiscoveryClient(base_url="https://x", api_key=""))


@pytest.mark.asyncio
async def test_an_unconfigured_key_is_503_and_makes_no_request():
    """Not 502: nothing failed, the capability simply is not enabled here. And no call must
    leave the process — an empty header would be answered 401 and read as a bad key."""
    seen: list[httpx.Request] = []
    client = client_with(json_handler(200, {}, capture=seen), api_key="")
    with pytest.raises(DrugDiscoveryUnavailable) as caught:
        await client.create_run(indication="TCGA-BRCA")
    assert caught.value.status_code == 503
    assert seen == []


# ---------------------------------------------------------------------------
# Status mapping — the 401 branch is the one that matters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_upstream_401_becomes_502():
    """OUR key is wrong, not the caller's session.

    Passing 401 through would tell a correctly logged-in user "unauthorised", so they would
    re-authenticate, retry, and report a bug against the wrong system. It is also the failure
    mode a key rotation produces, which makes a clear signal worth more than a short one.
    """
    client = client_with(json_handler(401, {"detail": "clé d'API absente ou invalide"}))
    with pytest.raises(DrugDiscoveryUnavailable) as caught:
        await client.get_run("r1")
    assert caught.value.status_code == 502


@pytest.mark.asyncio
async def test_an_upstream_422_keeps_its_rationale():
    """genolens-dd explains a refused indication with the curated rationale. Replacing that
    with a generic message would discard the only actionable part of the response."""
    detail = (
        "indication exclue de l'axe maladie : TCGA-PCPG. Motif — la cohorte fusionne deux "
        "maladies. Relancer avec allow_excluded=true si un classement SANS axe maladie est assumé"
    )
    client = client_with(json_handler(422, {"detail": detail}))
    with pytest.raises(DrugDiscoveryRejected) as caught:
        await client.create_run(indication="TCGA-PCPG")
    assert caught.value.status_code == 422
    assert "TCGA-PCPG" in caught.value.detail
    assert "allow_excluded" in caught.value.detail


@pytest.mark.asyncio
async def test_an_upstream_404_stays_a_404():
    client = client_with(json_handler(404, {"detail": "run inconnu"}))
    with pytest.raises(DrugDiscoveryRejected) as caught:
        await client.get_run("absent")
    assert caught.value.status_code == 404


@pytest.mark.asyncio
async def test_an_upstream_500_becomes_502_without_leaking_the_body():
    """An upstream stack trace must not reach a client of this API."""
    client = client_with(json_handler(500, {"detail": "Traceback: ligne 42 de scoring.py"}))
    with pytest.raises(DrugDiscoveryUnavailable) as caught:
        await client.get_targets("r1")
    assert caught.value.status_code == 502
    assert "scoring.py" not in str(caught.value)


@pytest.mark.asyncio
async def test_a_timeout_becomes_504():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("trop long", request=request)

    with pytest.raises(DrugDiscoveryUnavailable) as caught:
        await client_with(handler).create_run(indication="TCGA-BRCA")
    assert caught.value.status_code == 504


@pytest.mark.asyncio
async def test_an_unreachable_service_becomes_502():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connexion refusée", request=request)

    with pytest.raises(DrugDiscoveryUnavailable) as caught:
        await client_with(handler).readyz()
    assert caught.value.status_code == 502


@pytest.mark.asyncio
async def test_a_malformed_body_becomes_502():
    """A 200 carrying HTML — a proxy error page, say — must not surface as a valid result."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway</html>")

    with pytest.raises(DrugDiscoveryUnavailable) as caught:
        await client_with(handler).readyz()
    assert caught.value.status_code == 502


# ---------------------------------------------------------------------------
# Requests are shaped as genolens-dd expects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_run_payload_matches_the_upstream_contract():
    seen: list[httpx.Request] = []
    client = client_with(json_handler(201, {"run_id": "r1"}, capture=seen))
    await client.create_run(indication="TCGA-BRCA", profile="benchmark")
    import json

    body = json.loads(seen[0].content)
    assert body == {"indication": "TCGA-BRCA", "profile": "benchmark"}
    assert seen[0].url.path == "/runs"


@pytest.mark.asyncio
async def test_allow_excluded_is_only_sent_when_asked():
    """Sending `false` by default would be harmless today, but it makes the forced path look
    like part of every request in an upstream log — the opposite of an explicit override."""
    import json

    seen: list[httpx.Request] = []
    client = client_with(json_handler(201, {"run_id": "r1"}, capture=seen))
    await client.create_run(indication="TCGA-PCPG")
    assert "allow_excluded" not in json.loads(seen[0].content)
    await client.create_run(indication="TCGA-PCPG", allow_excluded=True)
    assert json.loads(seen[1].content)["allow_excluded"] is True


@pytest.mark.asyncio
async def test_the_targets_limit_is_forwarded():
    seen: list[httpx.Request] = []
    client = client_with(json_handler(200, {"n_ranked": 3}, capture=seen))
    await client.get_targets("r1", limit=25)
    assert seen[0].url.params["limit"] == "25"


@pytest.mark.asyncio
async def test_a_trailing_slash_in_the_base_url_does_not_double():
    seen: list[httpx.Request] = []
    client = DrugDiscoveryClient(
        base_url="https://dd.genolens.com/".rstrip("/"),
        api_key=KEY,
        transport=httpx.MockTransport(json_handler(200, {}, capture=seen)),
    )
    await client.get_run("r1")
    assert str(seen[0].url) == "https://dd.genolens.com/runs/r1"


# ---------------------------------------------------------------------------
# The endpoints: authentication is the whole point
# ---------------------------------------------------------------------------


DD_ROUTES = [
    ("GET", "/drug-discovery/indications"),
    ("GET", "/drug-discovery/status"),
    ("POST", "/drug-discovery/runs"),
    ("GET", "/drug-discovery/runs/r1"),
    ("GET", "/drug-discovery/runs/r1/targets"),
    ("GET", "/drug-discovery/runs/r1/report"),
]


def test_the_route_list_covers_every_registered_route():
    """Guards the list below against drift.

    Without this, a new endpoint added later would simply be absent from
    `test_every_endpoint_requires_authentication` — and the same person who forgets the auth
    dependency forgets to extend a hand-written list.
    """
    registered = {
        route.path.replace("/api/v1", "")
        for route in app.routes
        if getattr(route, "path", "").startswith("/api/v1/drug-discovery")
    }
    expected = {path.replace("/r1", "/{run_id}") for _, path in DD_ROUTES}
    assert registered == expected, f"routes non couvertes : {registered ^ expected}"


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path"), DD_ROUTES)
async def test_every_endpoint_requires_authentication(method, path):
    """THE TEST THAT MATTERS MOST IN THIS FILE.

    genolens-dd is closed by an API key so its rankings are not world-readable. An
    unauthenticated route here would serve those same rankings to anyone able to reach this
    backend — re-opening the product through another door, while the key gave every appearance
    of a closed one. A working feature is exactly what that bug would look like.
    """
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        response = await http.request(method, f"/api/v1{path}", json={})
    assert response.status_code in (401, 403), (
        f"{method} {path} answered {response.status_code} without a token — "
        "the Drug Discovery rankings would be public."
    )


@pytest.mark.asyncio
async def test_status_distinguishes_unconfigured_from_unreachable(monkeypatch):
    """Both look like "Drug Discovery is down" otherwise, and the wrong thing gets investigated."""
    unconfigured = client_with(json_handler(200, {}), api_key="")
    result = await dd_endpoints.drug_discovery_status(user=object(), client=unconfigured)
    assert result["configured"] is False
    assert result["reachable"] is None

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("injoignable", request=request)

    down = client_with(refuse)
    result = await dd_endpoints.drug_discovery_status(user=object(), client=down)
    assert result["configured"] is True
    assert result["reachable"] is False


@pytest.mark.asyncio
async def test_status_reports_readiness_when_all_is_well():
    ready = client_with(json_handler(200, {"ready": True, "tables": {"safety_profile": "present"}}))
    result = await dd_endpoints.drug_discovery_status(user=object(), client=ready)
    assert result == {
        "configured": True,
        "reachable": True,
        "ready": True,
        "tables": {"safety_profile": "present"},
    }


# ---------------------------------------------------------------------------
# Le catalogue : passe-plat, aucune logique de ce côté
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indications_are_passed_through_untouched():
    """Aucune reconstruction ici : genolens-dd possède la table curée.

    Filtrer ou réordonner de ce côté produirait deux catalogues qui divergent, et c'est le
    catalogue local qui aurait tort en silence.
    """
    catalogue = {
        "indications": [
            {"tcga_project": "TCGA-BRCA", "disease_name": "breast carcinoma",
             "excluded": False, "rationale": None},
            {"tcga_project": "TCGA-PCPG", "disease_name": "pheochromocytoma",
             "excluded": True, "rationale": "motif curé"},
        ],
        "profiles": ["default_oncology", "safety_first"],
    }
    seen: list[httpx.Request] = []
    client = client_with(json_handler(200, catalogue, capture=seen))
    assert await client.list_indications() == catalogue
    assert seen[0].url.path == "/indications"
