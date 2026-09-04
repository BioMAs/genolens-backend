"""
Unit tests for enrichment_source — which dataset holds a comparison's enrichment.

The bug this closes: `interpret_comparison` and the chat's pathways tool read
`EnrichmentPathway` under the id they were handed, which on a comparison page is the DEG dataset.
A self-service analysis keeps its enrichment in a separate annoDB ENRICHMENT dataset, and the
worker deliberately skips the legacy Python enrichment that would otherwise write rows under the
DEG id — so the AI and the chat saw **no pathways at all** while the enrichment panel next to them
was full of them.

The rules mirror the client (`useComparisonContext.ts`) on purpose, so the two cannot drift apart
again. These tests pin the mirroring, not just the happy path.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.models.models import DatasetStatus, DatasetType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PROJECT = uuid4()


def make_dataset(
    *,
    name: str = "ds",
    type_=DatasetType.ENRICHMENT,
    status=DatasetStatus.READY,
    project_id=None,
    metadata: dict | None = None,
):
    d = MagicMock()
    d.id = uuid4()
    d.name = name
    d.type = type_
    d.status = status
    d.project_id = project_id or PROJECT
    d.dataset_metadata = metadata
    return d


def make_db(candidates):
    """A db whose single ENRICHMENT query returns `candidates`."""
    db = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = candidates
    db.execute = AsyncMock(return_value=result)
    return db


def deg_dataset(metadata: dict | None = None):
    return make_dataset(
        name="KO_vs_WT", type_=DatasetType.DEG, metadata=metadata
    )


# ─────────────────────────────────────────────────────────────────────────────
# find_enrichment_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestFindEnrichmentDataset:

    @pytest.mark.asyncio
    async def test_matches_on_metadata_comparison_name(self):
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="whatever", metadata={"comparison_name": "KO_vs_WT"})
        db = make_db([enr])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is enr

    @pytest.mark.asyncio
    async def test_matches_on_dataset_name(self):
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="KO_vs_WT", metadata={})
        db = make_db([enr])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is enr

    @pytest.mark.asyncio
    async def test_matches_a_multi_comparison_file(self):
        """An enrichment file that merely lists the comparison among its columns."""
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(
            name="All enrichment",
            metadata={"enrichment_comparisons": ["A_vs_B", "KO_vs_WT"]},
        )
        db = make_db([enr])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is enr

    @pytest.mark.asyncio
    async def test_direct_name_match_beats_a_multi_comparison_file(self):
        """`byName || byComparisons` precedence, as the client applies it."""
        from app.services.enrichment_source import find_enrichment_dataset

        listed = make_dataset(
            name="All enrichment",
            metadata={"enrichment_comparisons": ["KO_vs_WT"]},
        )
        named = make_dataset(name="x", metadata={"comparison_name": "KO_vs_WT"})
        db = make_db([listed, named])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is named

    @pytest.mark.asyncio
    async def test_prefers_a_ready_dataset(self):
        """A failed or half-written duplicate must never be picked over a READY one."""
        from app.services.enrichment_source import find_enrichment_dataset

        failed = make_dataset(
            name="x", status=DatasetStatus.FAILED, metadata={"comparison_name": "KO_vs_WT"}
        )
        ready = make_dataset(
            name="y", status=DatasetStatus.READY, metadata={"comparison_name": "KO_vs_WT"}
        )
        db = make_db([failed, ready])

        found = await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT")
        assert found is ready

    @pytest.mark.asyncio
    async def test_scopes_to_the_same_analysis(self):
        """
        An enrichment file from another analysis sharing the comparison name must not bleed in.

        The client hit this and fixed it the same way ("mélange entre analyses").
        """
        from app.services.enrichment_source import find_enrichment_dataset

        analysis = str(uuid4())
        other = make_dataset(
            name="other-analysis",
            metadata={"comparison_name": "KO_vs_WT", "analysis_id": str(uuid4())},
        )
        mine = make_dataset(
            name="my-analysis",
            metadata={"comparison_name": "KO_vs_WT", "analysis_id": analysis},
        )
        db = make_db([other, mine])

        found = await find_enrichment_dataset(
            db, deg_dataset({"analysis_id": analysis}), "KO_vs_WT"
        )
        assert found is mine

    @pytest.mark.asyncio
    async def test_falls_back_to_project_scope_when_analysis_has_none(self):
        """Analysis scoping narrows only when it would leave something behind."""
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="x", metadata={"comparison_name": "KO_vs_WT"})
        db = make_db([enr])

        found = await find_enrichment_dataset(
            db, deg_dataset({"analysis_id": str(uuid4())}), "KO_vs_WT"
        )
        assert found is enr

    @pytest.mark.asyncio
    async def test_returns_none_when_nothing_matches(self):
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="unrelated", metadata={"comparison_name": "A_vs_B"})
        db = make_db([enr])

        assert await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_project_has_no_enrichment(self):
        from app.services.enrichment_source import find_enrichment_dataset

        db = make_db([])
        assert await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT") is None

    @pytest.mark.asyncio
    async def test_tolerates_null_metadata(self):
        """`dataset_metadata` is nullable; a None must not raise."""
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="KO_vs_WT", metadata=None)
        db = make_db([enr])

        found = await find_enrichment_dataset(db, deg_dataset(None), "KO_vs_WT")
        assert found is enr

    @pytest.mark.asyncio
    async def test_tolerates_non_list_enrichment_comparisons(self):
        """Malformed metadata must not raise — it just fails to match."""
        from app.services.enrichment_source import find_enrichment_dataset

        enr = make_dataset(name="x", metadata={"enrichment_comparisons": "KO_vs_WT"})
        db = make_db([enr])

        assert await find_enrichment_dataset(db, deg_dataset(), "KO_vs_WT") is None


# ─────────────────────────────────────────────────────────────────────────────
# resolve_pathway_dataset_id
# ─────────────────────────────────────────────────────────────────────────────

class TestResolvePathwayDatasetId:

    @pytest.mark.asyncio
    async def test_returns_the_enrichment_dataset_id(self):
        """The regression: the AI must read annoDB pathways, not the empty DEG dataset."""
        from app.services.enrichment_source import resolve_pathway_dataset_id

        enr = make_dataset(name="x", metadata={"comparison_name": "KO_vs_WT"})
        db = make_db([enr])
        deg = deg_dataset({"analysis_id": str(uuid4())})

        assert await resolve_pathway_dataset_id(db, deg, "KO_vs_WT") == enr.id

    @pytest.mark.asyncio
    async def test_falls_back_to_the_deg_dataset(self):
        """
        A plain DEG upload has no ENRICHMENT dataset: the legacy Python enrichment writes its rows
        under the DEG id, and that is the only enrichment it has. Same `?? dataset.id` fallback the
        enrichment panel applies, so the AI reads what the user sees.
        """
        from app.services.enrichment_source import resolve_pathway_dataset_id

        db = make_db([])
        deg = deg_dataset()

        assert await resolve_pathway_dataset_id(db, deg, "KO_vs_WT") == deg.id
