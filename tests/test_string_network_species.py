"""
The organism the interaction network asks STRING about.

It used to be guessed on the client from gene names — a rat project keyed by symbols went out
as mouse. With `dataset_id`, the route now takes the species the analysis was launched with
(`SelfServiceAnalysis.params['species']`), and the client's guess is only the fallback.
"""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.models import Dataset, SelfServiceAnalysis
from app.services.species import declared_species, species_taxon
from tests.conftest import TEST_PROJECT_ID, TEST_USER_ID, make_fake_supabase_user, make_project

DATASET_ID = UUID("00000000-0000-0000-0000-0000000000e1")
ANALYSIS_ID = UUID("00000000-0000-0000-0000-0000000000e2")
OTHER_USER_ID = UUID("00000000-0000-0000-0000-0000000000e3")

MOUSE, RAT, ZEBRAFISH = 10090, 10116, 7955


def _dataset(metadata):
    dataset = MagicMock(spec=Dataset)
    dataset.id = DATASET_ID
    dataset.project_id = TEST_PROJECT_ID
    dataset.dataset_metadata = metadata
    return dataset


def _analysis(params):
    analysis = MagicMock(spec=SelfServiceAnalysis)
    analysis.id = ANALYSIS_ID
    analysis.params = params
    return analysis


def _db(dataset=None, analysis=None, owner_id=TEST_USER_ID):
    """`get` by model, `scalar` answering the project-access check."""
    db = AsyncMock()

    async def _get(model, _identifier):
        return {Dataset: dataset, SelfServiceAnalysis: analysis}.get(model)

    db.get = AsyncMock(side_effect=_get)
    # Owner → one lookup; anyone else → the project, then no membership.
    db.scalar = AsyncMock(side_effect=[make_project(owner_id=owner_id), None])
    return db


# ── species → taxon ──────────────────────────────────────────────────────────

class TestSpeciesTaxon:
    @pytest.mark.parametrize(
        "name, taxon",
        [
            ("human", 9606),
            ("mouse", MOUSE),
            ("rat", RAT),
            ("zebrafish", ZEBRAFISH),
            ("pig", 9823),
            ("medaka", 8090),
            ("Rattus norvegicus", RAT),
            ("danio_rerio", ZEBRAFISH),
            ("  Mus.musculus ", MOUSE),
            ("rnorvegicus", RAT),
            ("mm39", MOUSE),
            ("hg38", 9606),
            ("10116", RAT),
            (7955, ZEBRAFISH),
        ],
    )
    def test_every_spelling_the_wizard_and_r_scripts_use(self, name, taxon):
        assert species_taxon(name) == taxon

    @pytest.mark.parametrize("value", [None, "", "arabidopsis", "0", 0, -1, True, ["rat"]])
    def test_an_unknown_organism_is_none_not_human(self, value):
        assert species_taxon(value) is None


class TestDeclaredSpecies:
    @pytest.mark.asyncio
    async def test_the_analysis_params_name_the_species(self):
        db = _db(analysis=_analysis({"species": "rat"}))
        dataset = _dataset({"analysis_id": str(ANALYSIS_ID)})
        assert await declared_species(db, dataset) == "rat"

    @pytest.mark.asyncio
    async def test_the_dataset_metadata_wins_over_the_analysis(self):
        db = _db(analysis=_analysis({"species": "rat"}))
        dataset = _dataset({"species": "zebrafish", "analysis_id": str(ANALYSIS_ID)})
        assert await declared_species(db, dataset) == "zebrafish"

    @pytest.mark.asyncio
    async def test_nothing_recorded_is_none(self):
        db = _db(analysis=_analysis({}))
        assert await declared_species(db, _dataset({"analysis_id": str(ANALYSIS_ID)})) is None
        assert await declared_species(db, _dataset(None)) is None

    @pytest.mark.asyncio
    async def test_a_malformed_analysis_id_is_ignored(self):
        db = _db()
        assert await declared_species(db, _dataset({"analysis_id": "not-a-uuid"})) is None
        db.get.assert_not_awaited()


# ── POST /integrations/string/network ────────────────────────────────────────

@pytest_asyncio.fixture
async def network_client():
    """Client with the user and DB overridden, and STRING itself mocked."""
    from app.main import app
    from app.api.deps import get_current_user, get_db

    state = {"db": _db()}

    async def _override_db():
        yield state["db"]

    app.dependency_overrides[get_current_user] = lambda: make_fake_supabase_user()
    app.dependency_overrides[get_db] = _override_db

    async def _network(*, gene_symbols, species, required_score, limit):
        return {"nodes": [], "edges": [], "species": species, "count": 0}

    with patch(
        "app.api.endpoints.integrations.string_service.get_network",
        new=AsyncMock(side_effect=_network),
    ) as get_network:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, state, get_network

    app.dependency_overrides.clear()


def _payload(**extra):
    # The client's guess is mouse: what title-case symbols used to be taken for.
    return {"gene_symbols": ["Trp53", "Cdkn1a"], "species": MOUSE, **extra}


class TestStringNetworkSpecies:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("species, taxon", [("rat", RAT), ("zebrafish", ZEBRAFISH)])
    async def test_the_analysis_species_wins_over_the_client_guess(
        self, network_client, species, taxon
    ):
        client, state, get_network = network_client
        state["db"] = _db(
            dataset=_dataset({"analysis_id": str(ANALYSIS_ID)}),
            analysis=_analysis({"species": species}),
        )

        resp = await client.post(
            "/api/v1/integrations/string/network", json=_payload(dataset_id=str(DATASET_ID))
        )

        assert resp.status_code == 200
        assert get_network.await_args.kwargs["species"] == taxon
        body = resp.json()
        assert body["species"] == taxon
        assert body["species_source"] == "dataset"

    @pytest.mark.asyncio
    async def test_a_species_in_the_dataset_metadata_is_used_too(self, network_client):
        client, state, get_network = network_client
        state["db"] = _db(dataset=_dataset({"species": "Rattus norvegicus"}))

        resp = await client.post(
            "/api/v1/integrations/string/network", json=_payload(dataset_id=str(DATASET_ID))
        )

        assert resp.status_code == 200
        assert get_network.await_args.kwargs["species"] == RAT

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metadata, params",
        [
            # An uploaded DEG file: no analysis behind it.
            ({}, None),
            # An analysis from before the wizard asked for a species.
            ({"analysis_id": str(ANALYSIS_ID)}, {}),
            # A species with no taxon in our map.
            ({"analysis_id": str(ANALYSIS_ID)}, {"species": "arabidopsis"}),
        ],
    )
    async def test_the_client_guess_stays_the_fallback(self, network_client, metadata, params):
        client, state, get_network = network_client
        state["db"] = _db(
            dataset=_dataset(metadata),
            analysis=_analysis(params) if params is not None else None,
        )

        resp = await client.post(
            "/api/v1/integrations/string/network", json=_payload(dataset_id=str(DATASET_ID))
        )

        assert resp.status_code == 200
        assert get_network.await_args.kwargs["species"] == MOUSE
        assert resp.json()["species_source"] == "request"

    @pytest.mark.asyncio
    async def test_without_a_dataset_the_request_species_is_used_as_before(self, network_client):
        client, state, get_network = network_client

        resp = await client.post("/api/v1/integrations/string/network", json=_payload())

        assert resp.status_code == 200
        assert get_network.await_args.kwargs["species"] == MOUSE
        assert resp.json()["species_source"] == "request"
        state["db"].get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_source_survives_a_string_failure(self, network_client):
        client, state, get_network = network_client
        state["db"] = _db(
            dataset=_dataset({"analysis_id": str(ANALYSIS_ID)}),
            analysis=_analysis({"species": "rat"}),
        )
        get_network.side_effect = None
        get_network.return_value = {
            "nodes": [], "edges": [], "species": RAT, "count": 0, "error": "not_found",
        }

        resp = await client.post(
            "/api/v1/integrations/string/network", json=_payload(dataset_id=str(DATASET_ID))
        )

        body = resp.json()
        assert body["error"] == "not_found"
        assert body["species"] == RAT
        assert body["species_source"] == "dataset"

    @pytest.mark.asyncio
    async def test_an_unknown_dataset_is_a_404(self, network_client):
        client, state, get_network = network_client
        state["db"] = _db(dataset=None)

        resp = await client.post(
            "/api/v1/integrations/string/network", json=_payload(dataset_id=str(DATASET_ID))
        )

        assert resp.status_code == 404
        get_network.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_dataset_of_someone_elses_project_is_a_404(self, network_client):
        client, state, get_network = network_client
        state["db"] = _db(
            dataset=_dataset({"analysis_id": str(ANALYSIS_ID)}),
            analysis=_analysis({"species": "rat"}),
            owner_id=OTHER_USER_ID,
        )

        resp = await client.post(
            "/api/v1/integrations/string/network", json=_payload(dataset_id=str(DATASET_ID))
        )

        assert resp.status_code == 404
        get_network.assert_not_awaited()
