"""
Unit tests for BookmarksService.

Covers:
- get_bookmarks (with/without gene_symbol filter)
- is_bookmarked (true / false)
- create_bookmark (happy path, duplicate prevention)
- update_bookmark (happy path, not found)
- delete_bookmark (happy path, not found)
- get_gene_lists / create_gene_list / update_gene_list / delete_gene_list
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4, UUID
from tests.conftest import (
    TEST_USER_ID, TEST_PROJECT_ID, TEST_BOOKMARK_ID,
    make_bookmark, make_project,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_result(value):
    """Return a mock that simulates result.scalar() == value."""
    r = MagicMock()
    r.scalar.return_value = value
    return r


def _scalars_all_result(items):
    """Return a mock that simulates result.scalars().all() == items."""
    r = MagicMock()
    r.scalars.return_value.all.return_value = items
    return r


def _scalar_one_or_none_result(value):
    """Return a mock that simulates result.scalar_one_or_none() == value."""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


# ─────────────────────────────────────────────────────────────────────────────
# BookmarksService — Gene Bookmarks
# ─────────────────────────────────────────────────────────────────────────────

class TestGetBookmarks:
    """Tests for BookmarksService.get_bookmarks."""

    @pytest.mark.asyncio
    async def test_get_bookmarks_returns_list(self, mock_db):
        """Should return a list of bookmarks for a user/project."""
        from app.services.bookmarks_service import BookmarksService

        bm1 = make_bookmark(gene_symbol="TP53")
        bm2 = make_bookmark(bookmark_id=uuid4(), gene_symbol="BRCA1")
        mock_db.execute.return_value = _scalars_all_result([bm1, bm2])

        service = BookmarksService()
        result = await service.get_bookmarks(mock_db, TEST_USER_ID, TEST_PROJECT_ID)

        assert len(result) == 2
        assert result[0].gene_symbol == "TP53"
        mock_db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_bookmarks_empty_project(self, mock_db):
        """Should return empty list when no bookmarks exist."""
        from app.services.bookmarks_service import BookmarksService

        mock_db.execute.return_value = _scalars_all_result([])

        service = BookmarksService()
        result = await service.get_bookmarks(mock_db, TEST_USER_ID, TEST_PROJECT_ID)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_bookmarks_filters_by_gene_symbol(self, mock_db):
        """Should apply gene_symbol filter when provided."""
        from app.services.bookmarks_service import BookmarksService

        bm = make_bookmark(gene_symbol="TP53")
        mock_db.execute.return_value = _scalars_all_result([bm])

        service = BookmarksService()
        result = await service.get_bookmarks(
            mock_db, TEST_USER_ID, TEST_PROJECT_ID, gene_symbol="TP53"
        )

        assert len(result) == 1
        assert result[0].gene_symbol == "TP53"


class TestIsBookmarked:
    """Tests for BookmarksService.is_bookmarked."""

    @pytest.mark.asyncio
    async def test_returns_true_when_bookmark_exists(self, mock_db):
        """Should return True when gene is already bookmarked."""
        from app.services.bookmarks_service import BookmarksService

        mock_db.execute.return_value = _scalar_result(1)

        service = BookmarksService()
        result = await service.is_bookmarked(mock_db, TEST_USER_ID, TEST_PROJECT_ID, "TP53")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_bookmarked(self, mock_db):
        """Should return False when gene is not bookmarked."""
        from app.services.bookmarks_service import BookmarksService

        mock_db.execute.return_value = _scalar_result(0)

        service = BookmarksService()
        result = await service.is_bookmarked(mock_db, TEST_USER_ID, TEST_PROJECT_ID, "BRCA1")

        assert result is False


class TestCreateBookmark:
    """Tests for BookmarksService.create_bookmark."""

    @pytest.mark.asyncio
    async def test_creates_bookmark_successfully(self, mock_db):
        """Should create and return a bookmark when gene is not yet bookmarked."""
        from app.services.bookmarks_service import BookmarksService

        # First call: is_bookmarked returns 0 (not bookmarked)
        mock_db.execute.return_value = _scalar_result(0)

        service = BookmarksService()
        result = await service.create_bookmark(
            mock_db,
            user_id=TEST_USER_ID,
            project_id=TEST_PROJECT_ID,
            gene_symbol="TP53",
            notes="Key tumor suppressor",
            tags=["apoptosis", "cancer"],
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()
        mock_db.refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_on_duplicate_bookmark(self, mock_db):
        """Should raise ValueError if gene is already bookmarked."""
        from app.services.bookmarks_service import BookmarksService

        # is_bookmarked returns 1 (already exists)
        mock_db.execute.return_value = _scalar_result(1)

        service = BookmarksService()
        with pytest.raises(ValueError, match="already bookmarked"):
            await service.create_bookmark(
                mock_db,
                user_id=TEST_USER_ID,
                project_id=TEST_PROJECT_ID,
                gene_symbol="TP53",
            )

        mock_db.add.assert_not_called()
        mock_db.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_creates_bookmark_with_defaults(self, mock_db):
        """Tags and is_favorite should have sensible defaults."""
        from app.services.bookmarks_service import BookmarksService
        from app.models.models import GeneBookmark

        mock_db.execute.return_value = _scalar_result(0)

        service = BookmarksService()
        # Just check it doesn't raise; add/commit are called
        await service.create_bookmark(
            mock_db,
            user_id=TEST_USER_ID,
            project_id=TEST_PROJECT_ID,
            gene_symbol="EGFR",
        )

        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.tags == []
        assert added_obj.is_favorite is True


class TestUpdateBookmark:
    """Tests for BookmarksService.update_bookmark."""

    @pytest.mark.asyncio
    async def test_updates_notes_and_tags(self, mock_db):
        """Should update notes and tags on an existing bookmark."""
        from app.services.bookmarks_service import BookmarksService

        bm = make_bookmark()
        mock_db.execute.return_value = _scalar_one_or_none_result(bm)

        service = BookmarksService()
        result = await service.update_bookmark(
            mock_db,
            bookmark_id=TEST_BOOKMARK_ID,
            user_id=TEST_USER_ID,
            notes="Updated notes",
            tags=["updated"],
        )

        assert bm.notes == "Updated notes"
        assert bm.tags == ["updated"]
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_bookmark_not_found(self, mock_db):
        """Should raise ValueError when bookmark doesn't exist or user mismatch."""
        from app.services.bookmarks_service import BookmarksService

        mock_db.execute.return_value = _scalar_one_or_none_result(None)

        service = BookmarksService()
        with pytest.raises(ValueError, match="not found or access denied"):
            await service.update_bookmark(
                mock_db,
                bookmark_id=uuid4(),
                user_id=TEST_USER_ID,
            )

    @pytest.mark.asyncio
    async def test_partial_update_preserves_other_fields(self, mock_db):
        """Updating only color should not change notes or tags."""
        from app.services.bookmarks_service import BookmarksService

        bm = make_bookmark(notes="original note", tags=["tag1"])
        mock_db.execute.return_value = _scalar_one_or_none_result(bm)

        service = BookmarksService()
        await service.update_bookmark(
            mock_db,
            bookmark_id=TEST_BOOKMARK_ID,
            user_id=TEST_USER_ID,
            color="#00FF00",
        )

        # Notes and tags must not be altered
        assert bm.notes == "original note"
        assert bm.tags == ["tag1"]
        assert bm.color == "#00FF00"


class TestDeleteBookmark:
    """Tests for BookmarksService.delete_bookmark."""

    @pytest.mark.asyncio
    async def test_deletes_bookmark_successfully(self, mock_db):
        """Should return True when bookmark exists and belongs to user."""
        from app.services.bookmarks_service import BookmarksService

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = TEST_BOOKMARK_ID
        mock_db.execute.return_value = mock_result

        service = BookmarksService()
        result = await service.delete_bookmark(
            mock_db,
            bookmark_id=TEST_BOOKMARK_ID,
            user_id=TEST_USER_ID,
        )

        assert result is True
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_bookmark_not_found_on_delete(self, mock_db):
        """Should raise ValueError when delete target is missing."""
        from app.services.bookmarks_service import BookmarksService

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = BookmarksService()
        with pytest.raises(ValueError, match="not found or access denied"):
            await service.delete_bookmark(
                mock_db,
                bookmark_id=uuid4(),
                user_id=TEST_USER_ID,
            )

        mock_db.commit.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# BookmarksService — Gene Lists
# ─────────────────────────────────────────────────────────────────────────────

class TestGetGeneLists:
    """Tests for BookmarksService.get_gene_lists."""

    @pytest.mark.asyncio
    async def test_returns_all_lists_for_project(self, mock_db):
        """Should return all gene lists for the project."""
        from app.services.bookmarks_service import BookmarksService
        from app.models.models import GeneList

        gl = MagicMock(spec=GeneList)
        gl.name = "DEGs > 2"
        mock_db.execute.return_value = _scalars_all_result([gl])

        service = BookmarksService()
        result = await service.get_gene_lists(mock_db, TEST_USER_ID, TEST_PROJECT_ID)

        assert len(result) == 1
        assert result[0].name == "DEGs > 2"


class TestCreateGeneList:
    """Tests for BookmarksService.create_gene_list."""

    @pytest.mark.asyncio
    async def test_creates_gene_list(self, mock_db):
        """Should add a new gene list to the DB."""
        from app.services.bookmarks_service import BookmarksService

        service = BookmarksService()
        await service.create_gene_list(
            mock_db,
            user_id=TEST_USER_ID,
            project_id=TEST_PROJECT_ID,
            name="My DEG List",
            genes=["TP53", "BRCA1", "EGFR"],
            description="Upregulated genes",
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gene_count_set_correctly(self, mock_db):
        """gene_count should equal len(genes)."""
        from app.services.bookmarks_service import BookmarksService

        service = BookmarksService()
        await service.create_gene_list(
            mock_db,
            user_id=TEST_USER_ID,
            project_id=TEST_PROJECT_ID,
            name="Short List",
            genes=["BIRC5", "MKI67"],
        )

        added = mock_db.add.call_args[0][0]
        assert added.gene_count == 2
        assert "BIRC5" in added.genes


class TestUpdateGeneList:
    """Tests for BookmarksService.update_gene_list."""

    @pytest.mark.asyncio
    async def test_updates_name_and_genes(self, mock_db):
        """Should update name and genes on an existing list."""
        from app.services.bookmarks_service import BookmarksService
        from app.models.models import GeneList

        gl = MagicMock(spec=GeneList)
        gl.id = uuid4()
        gl.genes = ["TP53"]
        gl.gene_count = 1
        mock_db.execute.return_value = _scalar_one_or_none_result(gl)

        service = BookmarksService()
        await service.update_gene_list(
            mock_db,
            list_id=gl.id,
            user_id=TEST_USER_ID,
            name="Updated Name",
            genes=["TP53", "BRCA1"],
        )

        assert gl.name == "Updated Name"
        assert gl.genes == ["TP53", "BRCA1"]
        assert gl.gene_count == 2
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_list_not_found(self, mock_db):
        """Should raise ValueError when list doesn't exist."""
        from app.services.bookmarks_service import BookmarksService

        mock_db.execute.return_value = _scalar_one_or_none_result(None)

        service = BookmarksService()
        with pytest.raises(ValueError, match="not found or access denied"):
            await service.update_gene_list(
                mock_db,
                list_id=uuid4(),
                user_id=TEST_USER_ID,
                name="Ghost List",
            )


class TestAddGenesToList:
    """Tests for BookmarksService.add_genes_to_list."""

    @pytest.mark.asyncio
    async def test_adds_new_genes_avoiding_duplicates(self, mock_db):
        """Should add only non-existing genes to the list."""
        from app.services.bookmarks_service import BookmarksService
        from app.models.models import GeneList

        gl = MagicMock(spec=GeneList)
        gl.genes = ["TP53", "BRCA1"]
        gl.gene_count = 2
        mock_db.execute.return_value = _scalar_one_or_none_result(gl)

        service = BookmarksService()
        await service.add_genes_to_list(
            mock_db,
            list_id=uuid4(),
            user_id=TEST_USER_ID,
            genes=["BRCA1", "EGFR"],  # BRCA1 is a duplicate
        )

        # Only EGFR should be added
        assert len(gl.genes) == 3
        assert "EGFR" in gl.genes
        assert gl.gene_count == 3
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_commit_when_all_genes_already_present(self, mock_db):
        """Should not commit if no new genes are to be added."""
        from app.services.bookmarks_service import BookmarksService
        from app.models.models import GeneList

        gl = MagicMock(spec=GeneList)
        gl.genes = ["TP53", "EGFR"]
        gl.gene_count = 2
        mock_db.execute.return_value = _scalar_one_or_none_result(gl)

        service = BookmarksService()
        await service.add_genes_to_list(
            mock_db,
            list_id=uuid4(),
            user_id=TEST_USER_ID,
            genes=["TP53", "EGFR"],  # all are duplicates
        )

        mock_db.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_list_not_found(self, mock_db):
        """Should raise ValueError when list doesn't exist."""
        from app.services.bookmarks_service import BookmarksService

        mock_db.execute.return_value = _scalar_one_or_none_result(None)

        service = BookmarksService()
        with pytest.raises(ValueError, match="not found or access denied"):
            await service.add_genes_to_list(
                mock_db,
                list_id=uuid4(),
                user_id=TEST_USER_ID,
                genes=["TP53"],
            )


class TestRemoveGenesFromList:
    """Tests for BookmarksService.remove_genes_from_list."""

    @pytest.mark.asyncio
    async def test_removes_specified_genes(self, mock_db):
        """Should remove listed genes from the gene list."""
        from app.services.bookmarks_service import BookmarksService
        from app.models.models import GeneList

        gl = MagicMock(spec=GeneList)
        gl.genes = ["TP53", "BRCA1", "EGFR"]
        gl.gene_count = 3
        mock_db.execute.return_value = _scalar_one_or_none_result(gl)

        service = BookmarksService()
        await service.remove_genes_from_list(
            mock_db,
            list_id=uuid4(),
            user_id=TEST_USER_ID,
            genes=["BRCA1"],
        )

        assert "BRCA1" not in gl.genes
        assert gl.gene_count == 2
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_removing_absent_gene_leaves_list_unchanged(self, mock_db):
        """Removing a gene not in list should not alter list contents."""
        from app.services.bookmarks_service import BookmarksService
        from app.models.models import GeneList

        gl = MagicMock(spec=GeneList)
        gl.genes = ["TP53", "EGFR"]
        gl.gene_count = 2
        mock_db.execute.return_value = _scalar_one_or_none_result(gl)

        service = BookmarksService()
        await service.remove_genes_from_list(
            mock_db,
            list_id=uuid4(),
            user_id=TEST_USER_ID,
            genes=["BRCA1"],  # not in list
        )

        assert gl.genes == ["TP53", "EGFR"]
        assert gl.gene_count == 2

    @pytest.mark.asyncio
    async def test_raises_when_list_not_found(self, mock_db):
        """Should raise ValueError when list doesn't exist."""
        from app.services.bookmarks_service import BookmarksService

        mock_db.execute.return_value = _scalar_one_or_none_result(None)

        service = BookmarksService()
        with pytest.raises(ValueError, match="not found or access denied"):
            await service.remove_genes_from_list(
                mock_db,
                list_id=uuid4(),
                user_id=TEST_USER_ID,
                genes=["TP53"],
            )


class TestDeleteGeneList:
    """Tests for BookmarksService.delete_gene_list."""

    @pytest.mark.asyncio
    async def test_deletes_list_successfully(self, mock_db):
        """Should return True when list is deleted."""
        from app.services.bookmarks_service import BookmarksService

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = uuid4()
        mock_db.execute.return_value = mock_result

        service = BookmarksService()
        result = await service.delete_gene_list(
            mock_db,
            list_id=uuid4(),
            user_id=TEST_USER_ID,
        )

        assert result is True
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_list_not_found(self, mock_db):
        """Should raise ValueError when list doesn't exist."""
        from app.services.bookmarks_service import BookmarksService

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        service = BookmarksService()
        with pytest.raises(ValueError, match="not found or access denied"):
            await service.delete_gene_list(
                mock_db,
                list_id=uuid4(),
                user_id=TEST_USER_ID,
            )
