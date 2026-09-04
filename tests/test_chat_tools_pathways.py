"""
Unit tests for the chat's enrichment-pathways tool.

Two bugs are pinned here, both caused by the tool reading the wrong dataset:

1. It queried `ctx.dataset_id` — the DEG dataset on a comparison page — while a self-service
   analysis keeps its enrichment in a separate annoDB ENRICHMENT dataset. The chat answered
   "no pathways" next to a panel full of them.
2. Once it reads annoDB rows, the category vocabulary changes: annoDB writes
   `biological_process`, the legacy Python path wrote `GO:BP`. The tool's own description told the
   model to ask for `GO:BP`, which would then match nothing.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.models.models import DatasetStatus, DatasetType


# ─────────────────────────────────────────────────────────────────────────────
# _match_category — the two vocabularies
# ─────────────────────────────────────────────────────────────────────────────

ANNODB = ["biological_process", "cellular_component", "matrisome", "kegg_pathway"]
LEGACY = ["GO:BP", "GO:MF", "GO:CC"]


class TestMatchCategory:

    def test_exact_match_wins(self):
        from app.services.chat_tools.tools import _match_category
        assert _match_category("matrisome", ANNODB) == "matrisome"

    def test_legacy_go_alias_maps_onto_annodb(self):
        """The regression: a model asking for GO:BP must reach biological_process."""
        from app.services.chat_tools.tools import _match_category
        assert _match_category("GO:BP", ANNODB) == "biological_process"
        assert _match_category("GO:CC", ANNODB) == "cellular_component"

    def test_annodb_name_maps_onto_a_legacy_dataset(self):
        """And the reverse, for datasets still holding legacy rows."""
        from app.services.chat_tools.tools import _match_category
        assert _match_category("biological_process", LEGACY) == "GO:BP"

    def test_case_insensitive(self):
        from app.services.chat_tools.tools import _match_category
        assert _match_category("BIOLOGICAL_PROCESS", ANNODB) == "biological_process"
        assert _match_category("go:bp", ANNODB) == "biological_process"

    def test_unknown_category_returns_none(self):
        """None means 'this dataset has no such category', not 'no filter'."""
        from app.services.chat_tools.tools import _match_category
        assert _match_category("KEGG", ANNODB) is None
        assert _match_category("Reactome", ANNODB) is None

    def test_no_alias_invents_a_category_the_dataset_lacks(self):
        from app.services.chat_tools.tools import _match_category
        assert _match_category("GO:MF", ["biological_process"]) is None


# ─────────────────────────────────────────────────────────────────────────────
# GetEnrichmentPathwaysTool
# ─────────────────────────────────────────────────────────────────────────────

def make_pathway(name="apoptosis", category="biological_process", padj=1e-5):
    p = MagicMock()
    p.pathway_name = name
    p.category = category
    p.padj = padj
    p.gene_count = 12
    p.regulation = "ALL"
    return p


def make_ctx(*, enrichment_datasets, categories, rows, comparison_name="KO_vs_WT"):
    """
    A ToolContext whose db answers the tool's three queries in order:
    the ENRICHMENT lookup, the distinct-categories read, then the rows.
    """
    from app.services.chat_tools.base import ToolContext

    deg = MagicMock()
    deg.id = uuid4()
    deg.name = "KO_vs_WT"
    deg.type = DatasetType.DEG
    deg.status = DatasetStatus.READY
    deg.project_id = uuid4()
    deg.dataset_metadata = {}

    def result_of(values):
        r = MagicMock()
        r.scalars.return_value.all.return_value = values
        return r

    db = MagicMock()
    db.execute = AsyncMock(side_effect=[
        result_of(enrichment_datasets),
        result_of(categories),
        result_of(rows),
    ])

    return ToolContext(
        db=db, dataset=deg, dataset_id=deg.id, project_id=deg.project_id,
        current_user=None, comparison_name=comparison_name,
    ), deg


def make_enrichment_dataset(project_id, comparison_name="KO_vs_WT"):
    e = MagicMock()
    e.id = uuid4()
    e.name = "enr"
    e.type = DatasetType.ENRICHMENT
    e.status = DatasetStatus.READY
    e.project_id = project_id
    e.dataset_metadata = {"comparison_name": comparison_name}
    return e


class TestGetEnrichmentPathwaysTool:

    @pytest.mark.asyncio
    async def test_reads_the_enrichment_dataset_not_the_deg_one(self):
        """
        The regression: pathways must be queried under the annoDB ENRICHMENT dataset's id.

        Asserted on the statement's bound parameters, not on its text — the DEG id would produce
        an equally valid-looking query against the same table.
        """
        from app.services.chat_tools.tools import GetEnrichmentPathwaysTool, PathwaysParams

        ctx, deg = make_ctx(enrichment_datasets=[], categories=[], rows=[])
        enr = make_enrichment_dataset(ctx.project_id)
        ctx.db.execute = AsyncMock(side_effect=[
            _r([enr]), _r(["biological_process"]), _r([make_pathway()]),
        ])

        result = await GetEnrichmentPathwaysTool().execute(ctx, PathwaysParams(top_n=5))

        assert result.summary_for_model["returned"] == 1

        bound = ctx.db.execute.await_args_list[-1].args[0].compile().params.values()
        assert enr.id in bound
        assert deg.id not in bound

    @pytest.mark.asyncio
    async def test_reports_the_available_categories(self):
        """So the model can filter against real values instead of guessing."""
        from app.services.chat_tools.tools import GetEnrichmentPathwaysTool, PathwaysParams

        ctx, _ = make_ctx(
            enrichment_datasets=[],
            categories=["matrisome", "biological_process"],
            rows=[make_pathway()],
        )
        result = await GetEnrichmentPathwaysTool().execute(ctx, PathwaysParams())

        assert result.summary_for_model["available_categories"] == [
            "biological_process", "matrisome"
        ]

    @pytest.mark.asyncio
    async def test_legacy_go_category_still_finds_annodb_rows(self):
        from app.services.chat_tools.tools import GetEnrichmentPathwaysTool, PathwaysParams

        ctx, _ = make_ctx(
            enrichment_datasets=[],
            categories=["biological_process"],
            rows=[make_pathway(category="biological_process")],
        )
        result = await GetEnrichmentPathwaysTool().execute(
            ctx, PathwaysParams(category="GO:BP")
        )

        assert result.summary_for_model["returned"] == 1
        assert "note" not in result.summary_for_model

    @pytest.mark.asyncio
    async def test_unknown_category_explains_itself(self):
        """
        An empty list reads to the model as "this comparison has no pathways". It must instead be
        told the category does not exist here, and which ones do.
        """
        from app.services.chat_tools.tools import GetEnrichmentPathwaysTool, PathwaysParams

        ctx, _ = make_ctx(
            enrichment_datasets=[],
            categories=["biological_process", "matrisome"],
            rows=[make_pathway()],
        )
        result = await GetEnrichmentPathwaysTool().execute(
            ctx, PathwaysParams(category="Reactome")
        )

        summary = result.summary_for_model
        assert summary["returned"] == 0
        assert "Reactome" in summary["note"]
        assert "biological_process" in summary["note"]
        assert "matrisome" in summary["note"]

    @pytest.mark.asyncio
    async def test_falls_back_to_the_scoped_dataset_without_a_comparison(self):
        """No comparison selected: nothing to resolve, so the tool keeps its own dataset id."""
        from app.services.chat_tools.tools import GetEnrichmentPathwaysTool, PathwaysParams

        ctx, deg = make_ctx(enrichment_datasets=[], categories=[], rows=[])
        ctx.comparison_name = None
        # Only two queries now — the ENRICHMENT lookup is skipped.
        ctx.db.execute = AsyncMock(side_effect=[_r([]), _r([])])

        result = await GetEnrichmentPathwaysTool().execute(ctx, PathwaysParams())

        assert result.summary_for_model["returned"] == 0
        assert ctx.db.execute.await_count == 2


def _r(values):
    r = MagicMock()
    r.scalars.return_value.all.return_value = values
    return r
