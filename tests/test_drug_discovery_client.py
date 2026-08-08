"""Tests for the genolens-dd client and its endpoints.

No network: every upstream response is served by `httpx.MockTransport`, so the suite is
deterministic and runs offline.

**The two tests that carry weight are not the happy paths.** They are
`test_every_endpoint_requires_authentication` — because an unauthenticated route here would
re-open the product that the API key exists to close — and `test_an_upstream_401_becomes_502`,
because passing a 401 through would blame the logged-in user for our own bad credential.
"""
import json

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


@pytest.mark.asyncio
async def test_readyz_preserves_the_body_of_a_business_503():
    """genolens-dd's `/readyz` answers 503 with a body when the reference socle is
    incomplete — that is its nominal signal, not an outage, and `tables` says exactly what to
    rebuild. Before this fix, `_interpret` mapped every >=500 onto `DrugDiscoveryUnavailable`
    regardless of body, so this legitimate response was indistinguishable from a crash and its
    `tables` detail was thrown away."""
    body = {"ready": False, "tables": {"safety_profile": "missing"}}
    client = client_with(json_handler(503, body))
    result = await client.readyz()
    assert result == body


@pytest.mark.asyncio
async def test_a_real_upstream_500_from_readyz_still_becomes_unavailable():
    """`preserve_503` only special-cases 503 — a genuine crash upstream must still surface as
    an outage, not be mistaken for an incomplete socle."""
    client = client_with(json_handler(500, {"detail": "Traceback: ligne 12"}))
    with pytest.raises(DrugDiscoveryUnavailable) as caught:
        await client.readyz()
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
    # Mode B. These carry the user's own gene list, so an unauthenticated one would be worse
    # than a public ranking: it would let anyone build a signature from someone else's data.
    ("GET", "/drug-discovery/signature-preview"),
    ("POST", "/drug-discovery/signature-runs"),
    ("GET", "/drug-discovery/runs/r1/signature/s1/report"),
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
    expected = {
        path.replace("/r1", "/{run_id}").replace("/s1", "/{signature_id}")
        for _, path in DD_ROUTES
    }
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
@pytest.mark.parametrize(("method", "path"), DD_ROUTES)
async def test_a_starter_plan_is_refused_on_every_route(method, path):
    """Le verrou vit ici, pas seulement dans l'UI.

    Une garde posée uniquement côté frontend serait cosmétique : un compte STARTER appellerait
    ces routes avec son propre jeton et obtiendrait le classement complet. Le module serait
    « réservé aux plans supérieurs » à l'écran et ouvert à tous par l'API.

    L'authentification elle-même est aussi overridée (`get_current_user`, celui que
    `get_or_create_user` consomme) : sans ça, une requête sans jeton Bearer est rejetée par
    `HTTPBearer` avant même d'atteindre le verrou de plan, et un 403 obtenu de cette façon ne
    prouve rien sur `require_team_plan` — voir `test_every_endpoint_requires_authentication`
    pour ce cas-là. Ici la requête doit être authentifiée pour que seul le plan puisse encore
    la refuser.
    """
    from httpx import ASGITransport, AsyncClient

    from app.api.deps.subscription import get_current_user, get_or_create_user
    from app.core.supabase_auth import SupabaseUser
    from app.models.models import SubscriptionPlan, User, UserRole

    starter = User(
        email="starter@example.com",
        subscription_plan=SubscriptionPlan.STARTER,
        role=UserRole.USER,
    )
    authenticated = SupabaseUser(
        user_id="00000000-0000-0000-0000-000000000099",
        email="starter@example.com",
    )
    app.dependency_overrides[get_or_create_user] = lambda: starter
    app.dependency_overrides[get_current_user] = lambda: authenticated
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http:
            response = await http.request(method, f"/api/v1{path}", json={})
        assert response.status_code == 403, (
            f"{method} {path} a répondu {response.status_code} à un STARTER — "
            "le classement serait accessible hors du plan qui le vend."
        )
        assert "requires a TEAM or ON_PREMISE plan" in response.text, (
            f"{method} {path} a répondu 403 sans nommer l'exigence de plan — "
            f"corps reçu : {response.text!r}. Un 403 dû à l'authentification passerait ce "
            "test à tort ; c'est précisément la confusion qu'il doit détecter."
        )
    finally:
        app.dependency_overrides.clear()


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


@pytest.mark.asyncio
async def test_status_reports_reachable_with_an_incomplete_socle_not_unreachable():
    """A `genolens-dd` that is up but missing a reference table answers `/readyz` with 503 and
    a body — its nominal way of saying "rebuild these tables". Before the fix, that 503 was
    indistinguishable from an outage: `/status` reported `reachable: false` and dropped
    `tables`, sending an operator to check the network on the day a deploy actually needs them
    to check their data."""
    incomplete = client_with(
        json_handler(503, {"ready": False, "tables": {"contrast_disease_normal": "missing"}})
    )
    result = await dd_endpoints.drug_discovery_status(user=object(), client=incomplete)
    assert result == {
        "configured": True,
        "reachable": True,
        "ready": False,
        "tables": {"contrast_disease_normal": "missing"},
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


# ---------------------------------------------------------------------------
# Mode B — the one call that carries the user's own data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_signature_payload_matches_the_upstream_contract():
    """genolens-dd expects `conditions: {name: {genes, replicates}}`, not two parallel dicts.

    Sending the wrong shape would come back as a rule-coded rejection about a missing
    replicate count — a rejection the user cannot act on, since the fault is ours.
    """
    seen: list[httpx.Request] = []
    client = client_with(json_handler(201, {"signature_id": "s1"}, capture=seen))
    await client.submit_signature(
        "r1",
        client_id="proj-1",
        genes_by_condition={"treated": ["EGFR", "ERBB2"], "control": ["TP53"]},
        replicates={"treated": 4, "control": 3},
        seed=1234,
    )
    body = json.loads(seen[0].content)
    assert body["client_id"] == "proj-1"
    assert body["seed"] == 1234
    assert body["conditions"] == {
        "treated": {"genes": ["EGFR", "ERBB2"], "replicates": 4},
        "control": {"genes": ["TP53"], "replicates": 3},
    }
    assert str(seen[0].url).startswith("https://dd.genolens.com/runs/r1/signature")


@pytest.mark.asyncio
async def test_allow_underpowered_is_only_sent_when_asked():
    """Same discipline as `allow_excluded`: a permissive flag always present ends up always
    true, and an underpowered signature would stop being an explicit choice."""
    seen: list[httpx.Request] = []
    client = client_with(json_handler(201, {}, capture=seen))
    await client.submit_signature(
        "r1", client_id="p", genes_by_condition={"t": ["EGFR"]},
        replicates={"t": 3}, seed=1,
    )
    assert "allow_underpowered" not in json.loads(seen[0].content)

    await client.submit_signature(
        "r1", client_id="p", genes_by_condition={"t": ["EGFR"]},
        replicates={"t": 2}, seed=1, allow_underpowered=True,
    )
    assert json.loads(seen[1].content)["allow_underpowered"] is True


@pytest.mark.asyncio
async def test_a_rule_coded_rejection_keeps_its_structured_detail():
    """`rule_id` is the only part of a SIG00x rejection the frontend can route on.

    Flattening it to a string here would force the UI to regex a French sentence, and the
    upstream carries `rule_id` in attributes precisely so nobody has to.
    """
    detail = {"rule_id": "SIG002", "conditions": ["treated"], "message": "…"}
    client = client_with(json_handler(422, {"detail": detail}))
    with pytest.raises(DrugDiscoveryRejected) as raised:
        await client.submit_signature(
            "r1", client_id="p", genes_by_condition={"treated": ["EGFR"]},
            replicates={"treated": 2}, seed=1,
        )
    assert raised.value.status_code == 422
    assert raised.value.detail["rule_id"] == "SIG002"


@pytest.mark.asyncio
async def test_a_run_evicted_between_create_and_submit_stays_a_404():
    """Upstream runs live in memory; a redeploy between the two calls loses one.

    Mapping that to 502 would tell the user the service is broken when the correct action is
    to run again — which is exactly what the frontend's bounded recovery does on a 404.
    """
    client = client_with(json_handler(404, {"detail": "run inconnu : r1"}))
    with pytest.raises(DrugDiscoveryRejected) as raised:
        await client.submit_signature(
            "r1", client_id="p", genes_by_condition={"t": ["EGFR"]},
            replicates={"t": 3}, seed=1,
        )
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_the_signature_report_limit_reaches_the_query_string():
    seen: list[httpx.Request] = []
    client = client_with(json_handler(200, {}, capture=seen))
    await client.get_signature_report("r1", "s1", limit=3)
    assert seen[0].url.params["limit"] == "3"
    assert seen[0].url.path == "/runs/r1/signature/s1/report"
