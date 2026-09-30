"""
`GET /genes/search` — la recherche de gènes de la palette de commandes.

Ces tests tournent contre un VRAI Postgres, pas contre une session simulée. Ce qu'ils
vérifient — la règle d'accès propriétaire-ou-membre, le LIKE insensible à la casse, le
classement exact-avant-préfixe, la jointure vers l'analyse — vit entièrement dans le SQL :
un mock de `db.execute` renverrait ce qu'on lui dit de renvoyer et ne prouverait rien.
C'est d'ailleurs ainsi que l'ancienne version, qui ne cherchait aucun gène, passait ses
tests.

Chaque test écrit dans une transaction annulée à la fin. La CI fournit la base (migrée,
via `DATABASE_URL`) ; sans base joignable, les tests qui en ont besoin sont SAUTÉS avec
un motif explicite plutôt que de passer à vide.
"""

from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.models.models import (
    Dataset,
    DatasetStatus,
    DatasetType,
    DegGene,
    Project,
    ProjectMember,
    SelfServiceAnalysis,
    SelfServiceAnalysisStatus,
    UserRole,
)
from tests.conftest import make_fake_supabase_user

SEARCH_URL = "/api/v1/genes/search"


# ─────────────────────────────────────────────────────────────────────────────
# Base réelle, une transaction par test
# ─────────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session():
    from app.core.config import settings

    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    try:
        conn = await engine.connect()
    except Exception as exc:  # pragma: no cover - dépend de l'environnement
        await engine.dispose()
        pytest.skip(f"gene search tests need a reachable Postgres at DATABASE_URL ({exc})")

    trans = await conn.begin()
    try:
        await conn.execute(text("SELECT 1 FROM deg_genes LIMIT 0"))
    except Exception as exc:  # pragma: no cover - base non migrée
        await trans.rollback()
        await conn.close()
        await engine.dispose()
        pytest.skip(f"gene search tests need a migrated database ({exc})")

    session = AsyncSession(
        bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
    )
    try:
        yield session
    finally:
        await session.close()
        await trans.rollback()
        await conn.close()
        await engine.dispose()


def client_as(session, user_id: UUID):
    """Client HTTP authentifié comme `user_id`, branché sur la session donnée."""
    from app.api.deps import get_current_user, get_db
    from app.main import app

    async def _db():
        yield session

    app.dependency_overrides[get_current_user] = lambda: make_fake_supabase_user(user_id=user_id)
    app.dependency_overrides[get_db] = _db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    from app.api.deps import get_current_user, get_db
    from app.main import app

    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_db, None)


# ─────────────────────────────────────────────────────────────────────────────
# Jeu de données
# ─────────────────────────────────────────────────────────────────────────────


class World:
    """Trois utilisateurs, trois projets, et des gènes aux noms qui se chevauchent.

    - `owner` possède « Owned » ;
    - `other` possède « Shared », dont `owner` est MEMBRE ;
    - `stranger` possède « Private », auquel `owner` n'a aucun accès.

    Chaque projet porte TP53 : la seule différence entre une fuite et une recherche
    correcte est donc la règle d'accès, pas le nom du gène.
    """

    def __init__(self):
        self.owner, self.other, self.stranger = uuid4(), uuid4(), uuid4()
        self.owned_project = uuid4()
        self.shared_project = uuid4()
        self.private_project = uuid4()
        self.analysis = uuid4()
        self.analysis_dataset = uuid4()
        self.uploaded_dataset = uuid4()
        self.orphan_dataset = uuid4()
        self.shared_dataset = uuid4()
        self.private_dataset = uuid4()
        self.processing_dataset = uuid4()


def _project(pid, owner, name):
    return Project(id=pid, owner_id=owner, name=name)


def _dataset(did, pid, name, *, status=DatasetStatus.READY, metadata=None):
    return Dataset(
        id=did,
        project_id=pid,
        name=name,
        type=DatasetType.DEG,
        status=status,
        column_mapping={},
        dataset_metadata=metadata or {"comparison_name": name},
    )


def _gene(did, comparison, gene_id, symbol, *, log_fc, padj, regulation):
    return DegGene(
        dataset_id=did,
        comparison_name=comparison,
        gene_id=gene_id,
        gene_name=symbol,
        log_fc=log_fc,
        padj=padj,
        regulation=regulation,
    )


@pytest_asyncio.fixture
async def world(db_session):
    w = World()
    db = db_session
    db.add_all(
        [
            _project(w.owned_project, w.owner, "Owned"),
            _project(w.shared_project, w.other, "Shared"),
            _project(w.private_project, w.stranger, "Private"),
        ]
    )
    await db.flush()
    db.add(ProjectMember(project_id=w.shared_project, user_id=w.owner, access_level=UserRole.USER))
    db.add(
        SelfServiceAnalysis(
            id=w.analysis,
            project_id=w.owned_project,
            name="DESeq2 run",
            status=SelfServiceAnalysisStatus.DONE,
            params={},
            result_dataset_ids=[],
            intermediate_dataset_ids={},
            progress_log=[],
            user_id=w.owner,
        )
    )
    db.add_all(
        [
            # Produit par une analyse : la page de comparaison a une URL d'analyse.
            _dataset(
                w.analysis_dataset,
                w.owned_project,
                "KO_vs_WT",
                metadata={"analysis_id": str(w.analysis), "comparison_name": "KO_vs_WT"},
            ),
            # Téléversé directement : pas d'analyse.
            _dataset(w.uploaded_dataset, w.owned_project, "upload"),
            # Pointe vers une analyse supprimée : ne doit pas produire de lien mort.
            _dataset(
                w.orphan_dataset,
                w.owned_project,
                "orphan",
                metadata={"analysis_id": str(uuid4()), "comparison_name": "Old"},
            ),
            _dataset(w.shared_dataset, w.shared_project, "Treated_vs_Ctrl"),
            _dataset(w.private_dataset, w.private_project, "Secret"),
            _dataset(
                w.processing_dataset, w.owned_project, "pending", status=DatasetStatus.PROCESSING
            ),
        ]
    )
    await db.flush()
    db.add_all(
        [
            _gene(
                w.analysis_dataset,
                "KO_vs_WT",
                "ENSG00000141510",
                "TP53",
                log_fc=2.5,
                padj=1e-6,
                regulation="UP",
            ),
            # Préfixe de TP53, et PLUS significatif : l'ordre exact-avant-préfixe doit
            # l'emporter sur la p-value.
            _gene(
                w.analysis_dataset,
                "KO_vs_WT",
                "ENSG00000067369",
                "TP53BP1",
                log_fc=-1.2,
                padj=1e-12,
                regulation="DOWN",
            ),
            _gene(
                w.uploaded_dataset,
                "Drug_vs_Vehicle",
                "ENSG00000141510",
                "TP53",
                log_fc=0.1,
                padj=0.8,
                regulation="NS",
            ),
            # Pas de symbole : cherchable par son identifiant, affiché sous cet identifiant.
            _gene(
                w.uploaded_dataset,
                "Drug_vs_Vehicle",
                "ENSG00000999999",
                None,
                log_fc=1.0,
                padj=0.01,
                regulation="UP",
            ),
            _gene(
                w.orphan_dataset,
                "Old",
                "ENSG00000141510",
                "TP53",
                log_fc=0.4,
                padj=0.3,
                regulation="NS",
            ),
            _gene(
                w.shared_dataset,
                "Treated_vs_Ctrl",
                "ENSG00000141510",
                "TP53",
                log_fc=-3.0,
                padj=1e-4,
                regulation="DOWN",
            ),
            _gene(
                w.private_dataset,
                "Secret",
                "ENSG00000141510",
                "TP53",
                log_fc=5.0,
                padj=1e-30,
                regulation="UP",
            ),
            _gene(
                w.private_dataset,
                "Secret",
                "ENSG00000000001",
                "SECRETGENE",
                log_fc=5.0,
                padj=1e-30,
                regulation="UP",
            ),
            _gene(
                w.processing_dataset,
                "pending",
                "ENSG00000141510",
                "TP53",
                log_fc=9.0,
                padj=1e-40,
                regulation="UP",
            ),
        ]
    )
    await db.flush()
    return w


async def search(session, user_id, **params):
    async with client_as(session, user_id) as client:
        response = await client.get(SEARCH_URL, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def projects_of(body):
    return {r["project_name"] for r in body["results"]}


# ─────────────────────────────────────────────────────────────────────────────
# Accès
# ─────────────────────────────────────────────────────────────────────────────


class TestAccess:

    async def test_owned_project_hit(self, db_session, world):
        body = await search(db_session, world.owner, q="TP53")
        owned = [r for r in body["results"] if r["project_id"] == str(world.owned_project)]
        assert owned, body
        assert all(r["gene_symbol"] in {"TP53", "TP53BP1"} for r in owned)

    async def test_shared_project_hit(self, db_session, world):
        """Un membre trouve les gènes du projet partagé — l'ancienne version l'excluait."""
        body = await search(db_session, world.owner, q="TP53")
        shared = [r for r in body["results"] if r["project_id"] == str(world.shared_project)]
        assert len(shared) == 1
        hit = shared[0]
        assert hit["comparison_name"] == "Treated_vs_Ctrl"
        assert hit["regulation"] == "DOWN"
        assert hit["log_fc"] == pytest.approx(-3.0)
        assert hit["padj"] == pytest.approx(1e-4)

    async def test_non_member_sees_nothing_of_a_private_project(self, db_session, world):
        body = await search(db_session, world.owner, q="TP53")
        assert "Private" not in projects_of(body)
        assert await search(db_session, world.owner, q="SECRETGENE") == {
            "results": [],
            "total": 0,
            "query": "SECRETGENE",
        }

    async def test_user_without_any_project_gets_no_results(self, db_session, world):
        body = await search(db_session, uuid4(), q="TP53")
        assert body["results"] == []
        assert body["total"] == 0

    async def test_the_owner_of_the_shared_project_does_not_see_the_members_own_project(
        self, db_session, world
    ):
        """L'appartenance ne remonte pas : `other` voit « Shared », pas « Owned »."""
        body = await search(db_session, world.other, q="TP53")
        assert projects_of(body) == {"Shared"}

    async def test_project_filter_cannot_open_a_foreign_project(self, db_session, world):
        body = await search(
            db_session, world.owner, q="TP53", project_id=str(world.private_project)
        )
        assert body["results"] == []

    async def test_project_filter_restricts_to_that_project(self, db_session, world):
        body = await search(db_session, world.owner, q="TP53", project_id=str(world.shared_project))
        assert projects_of(body) == {"Shared"}

    async def test_datasets_that_are_not_ready_are_skipped(self, db_session, world):
        body = await search(db_session, world.owner, q="TP53", limit=50)
        assert str(world.processing_dataset) not in {r["dataset_id"] for r in body["results"]}


# ─────────────────────────────────────────────────────────────────────────────
# Correspondance et classement
# ─────────────────────────────────────────────────────────────────────────────


class TestMatching:

    async def test_exact_matches_come_before_prefix_matches(self, db_session, world):
        body = await search(db_session, world.owner, q="TP53")
        symbols = [r["gene_symbol"] for r in body["results"]]
        flags = [r["exact"] for r in body["results"]]
        # TP53BP1 a la meilleure p-value de tout le jeu, mais n'est qu'un préfixe.
        assert symbols[-1] == "TP53BP1"
        assert set(symbols[:-1]) == {"TP53"}
        assert flags == [True] * (len(flags) - 1) + [False]

    async def test_exact_matches_are_ordered_by_adjusted_p_value(self, db_session, world):
        body = await search(db_session, world.owner, q="TP53")
        exact = [r["padj"] for r in body["results"] if r["exact"]]
        assert exact == sorted(exact)
        assert len(exact) == 4  # analyse, téléversé, orphelin, partagé

    async def test_a_prefix_alone_returns_every_gene_starting_with_it(self, db_session, world):
        body = await search(db_session, world.owner, q="TP5")
        assert {r["gene_symbol"] for r in body["results"]} == {"TP53", "TP53BP1"}
        assert not any(r["exact"] for r in body["results"])
        # Sans correspondance exacte, la p-value seule ordonne : TP53BP1 d'abord.
        assert body["results"][0]["gene_symbol"] == "TP53BP1"

    async def test_matching_is_case_insensitive(self, db_session, world):
        lower = await search(db_session, world.owner, q="tp53")
        upper = await search(db_session, world.owner, q="TP53")
        assert lower["results"] == upper["results"]
        assert lower["query"] == "tp53"

    async def test_matches_on_gene_id(self, db_session, world):
        body = await search(db_session, world.owner, q="ensg00000141510")
        assert body["results"]
        assert {r["gene_symbol"] for r in body["results"]} == {"TP53"}
        assert all(r["exact"] and r["gene_id"] == "ENSG00000141510" for r in body["results"])

    async def test_a_gene_without_symbol_is_shown_under_its_id(self, db_session, world):
        body = await search(db_session, world.owner, q="ENSG00000999")
        assert [(r["gene_id"], r["gene_symbol"]) for r in body["results"]] == [
            ("ENSG00000999999", "ENSG00000999999")
        ]

    async def test_like_wildcards_are_literal(self, db_session, world):
        """« T_53 » ne doit pas se lire comme « T, un caractère, 53 »."""
        assert (await search(db_session, world.owner, q="T_53"))["results"] == []
        assert (await search(db_session, world.owner, q="T%"))["results"] == []

    async def test_no_match_returns_an_empty_list_and_never_echoes_the_query(
        self, db_session, world
    ):
        body = await search(db_session, world.owner, q="NOTAGENE")
        assert body == {"results": [], "total": 0, "query": "NOTAGENE"}

    async def test_limit_caps_the_result_count(self, db_session, world):
        body = await search(db_session, world.owner, q="TP53", limit=2)
        assert body["total"] == 2
        assert [r["exact"] for r in body["results"]] == [True, True]


# ─────────────────────────────────────────────────────────────────────────────
# Ce qu'il faut pour construire le lien
# ─────────────────────────────────────────────────────────────────────────────


class TestLinkFields:

    async def test_analysis_is_returned_when_it_exists(self, db_session, world):
        body = await search(db_session, world.owner, q="TP53", limit=50)
        by_dataset = {r["dataset_id"]: r for r in body["results"] if r["gene_symbol"] == "TP53"}

        from_analysis = by_dataset[str(world.analysis_dataset)]
        assert from_analysis["analysis_id"] == str(world.analysis)
        assert from_analysis["analysis_name"] == "DESeq2 run"
        assert from_analysis["comparison_name"] == "KO_vs_WT"

        assert by_dataset[str(world.uploaded_dataset)]["analysis_id"] is None
        # Analyse supprimée : pas de lien vers une page qui n'existe plus.
        assert by_dataset[str(world.orphan_dataset)]["analysis_id"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Validation (sans base)
# ─────────────────────────────────────────────────────────────────────────────


class TestValidation:

    async def test_missing_query_is_rejected(self):
        from unittest.mock import AsyncMock

        async with client_as(AsyncMock(), uuid4()) as client:
            assert (await client.get(SEARCH_URL)).status_code == 422

    async def test_single_character_is_rejected(self):
        """Un caractère ferait correspondre une part énorme de la table."""
        from unittest.mock import AsyncMock

        async with client_as(AsyncMock(), uuid4()) as client:
            assert (await client.get(SEARCH_URL, params={"q": "T"})).status_code == 422

    async def test_limit_above_maximum_is_rejected(self):
        from unittest.mock import AsyncMock

        async with client_as(AsyncMock(), uuid4()) as client:
            response = await client.get(SEARCH_URL, params={"q": "TP53", "limit": 500})
        assert response.status_code == 422

    async def test_blank_query_returns_nothing_without_querying(self):
        from unittest.mock import AsyncMock

        db = AsyncMock()
        async with client_as(db, uuid4()) as client:
            response = await client.get(SEARCH_URL, params={"q": "   "})
        assert response.status_code == 200
        assert response.json()["results"] == []
        db.execute.assert_not_called()
