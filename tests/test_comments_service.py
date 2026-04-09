"""
Unit tests for CommentsService.

Covers:
- get_comments (with/without filters)
- get_comment_by_id (found / not found)
- get_comment_thread (root + replies)
- create_comment (happy path, missing project, invalid parent)
- update_comment (happy path, permission denied, not found)
- delete_comment (happy path, not found)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from tests.conftest import (
    TEST_USER_ID, TEST_PROJECT_ID, TEST_COMMENT_ID, TEST_PARENT_COMMENT_ID,
    make_comment, make_project,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalars_all(items):
    r = MagicMock()
    r.scalars.return_value.all.return_value = items
    return r


def _scalar_one_or_none(value):
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


# ─────────────────────────────────────────────────────────────────────────────
# get_comments
# ─────────────────────────────────────────────────────────────────────────────

class TestGetComments:

    @pytest.mark.asyncio
    async def test_returns_all_top_level_comments(self, mock_db):
        """Should return top-level comments for a project."""
        from app.services.comments_service import CommentsService

        c1 = make_comment()
        c2 = make_comment(comment_id=uuid4(), content="Another comment")
        mock_db.execute.return_value = _scalars_all([c1, c2])

        service = CommentsService()
        result = await service.get_comments(mock_db, TEST_PROJECT_ID)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_comments(self, mock_db):
        """Should return empty list when project has no comments."""
        from app.services.comments_service import CommentsService

        mock_db.execute.return_value = _scalars_all([])

        service = CommentsService()
        result = await service.get_comments(mock_db, TEST_PROJECT_ID)

        assert result == []

    @pytest.mark.asyncio
    async def test_filters_resolved_comments(self, mock_db):
        """When include_resolved=False, resolved comments should be filtered."""
        from app.services.comments_service import CommentsService

        c_unresolved = make_comment(is_resolved=False)
        mock_db.execute.return_value = _scalars_all([c_unresolved])

        service = CommentsService()
        result = await service.get_comments(
            mock_db, TEST_PROJECT_ID, include_resolved=False
        )
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# get_comment_by_id
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCommentById:

    @pytest.mark.asyncio
    async def test_returns_comment_when_found(self, mock_db):
        """Should return a comment when ID exists."""
        from app.services.comments_service import CommentsService

        c = make_comment()
        mock_db.execute.return_value = _scalar_one_or_none(c)

        service = CommentsService()
        result = await service.get_comment_by_id(mock_db, TEST_COMMENT_ID)

        assert result is c
        assert result.id == TEST_COMMENT_ID

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, mock_db):
        """Should return None when ID does not exist."""
        from app.services.comments_service import CommentsService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        service = CommentsService()
        result = await service.get_comment_by_id(mock_db, uuid4())

        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# get_comment_thread
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCommentThread:

    @pytest.mark.asyncio
    async def test_returns_empty_when_root_not_found(self, mock_db):
        """Should return empty list if root comment doesn't exist."""
        from app.services.comments_service import CommentsService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        service = CommentsService()
        result = await service.get_comment_thread(mock_db, uuid4())

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_root_with_replies(self, mock_db):
        """Should include root comment and all nested replies."""
        from app.services.comments_service import CommentsService

        reply = make_comment(
            comment_id=uuid4(),
            content="Reply",
            parent_id=TEST_COMMENT_ID,
        )
        reply.replies = []  # leaf node

        root = make_comment()
        root.replies = [reply]

        mock_db.execute.return_value = _scalar_one_or_none(root)

        service = CommentsService()
        result = await service.get_comment_thread(mock_db, TEST_COMMENT_ID)

        assert len(result) == 2
        assert result[0] is root
        assert result[1] is reply


# ─────────────────────────────────────────────────────────────────────────────
# create_comment
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateComment:

    @pytest.mark.asyncio
    async def test_creates_comment_successfully(self, mock_db):
        """Happy-path: comment is created when project exists."""
        from app.services.comments_service import CommentsService

        project = make_project()
        # First execute: project lookup → found
        mock_db.execute.side_effect = [
            _scalar_one_or_none(project),   # project exists
        ]

        service = CommentsService()
        await service.create_comment(
            mock_db,
            project_id=TEST_PROJECT_ID,
            user_id=TEST_USER_ID,
            content="This is a comment",
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_project_not_found(self, mock_db):
        """Should raise ValueError when project does not exist."""
        from app.services.comments_service import CommentsService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        service = CommentsService()
        with pytest.raises(ValueError, match="not found"):
            await service.create_comment(
                mock_db,
                project_id=uuid4(),
                user_id=TEST_USER_ID,
                content="Orphan comment",
            )

        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_parent_comment_not_found(self, mock_db):
        """Should raise ValueError when parent comment ID is invalid."""
        from app.services.comments_service import CommentsService

        project = make_project()
        # First call: project found; second call: parent not found
        mock_db.execute.side_effect = [
            _scalar_one_or_none(project),
            _scalar_one_or_none(None),  # parent missing
        ]

        service = CommentsService()
        with pytest.raises(ValueError, match="Parent comment .* not found"):
            await service.create_comment(
                mock_db,
                project_id=TEST_PROJECT_ID,
                user_id=TEST_USER_ID,
                content="Reply",
                parent_id=uuid4(),
            )

    @pytest.mark.asyncio
    async def test_raises_when_parent_belongs_to_different_project(self, mock_db):
        """Should raise ValueError for cross-project reply attempt."""
        from app.services.comments_service import CommentsService

        project = make_project()
        other_project_id = uuid4()
        parent = make_comment(project_id=other_project_id)

        mock_db.execute.side_effect = [
            _scalar_one_or_none(project),
            _scalar_one_or_none(parent),
        ]

        service = CommentsService()
        with pytest.raises(ValueError, match="different project"):
            await service.create_comment(
                mock_db,
                project_id=TEST_PROJECT_ID,
                user_id=TEST_USER_ID,
                content="Cross-project reply",
                parent_id=parent.id,
            )


# ─────────────────────────────────────────────────────────────────────────────
# update_comment
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateComment:

    @pytest.mark.asyncio
    async def test_updates_content_successfully(self, mock_db):
        """Owner can update content."""
        from app.services.comments_service import CommentsService

        c = make_comment()
        mock_db.execute.return_value = _scalar_one_or_none(c)

        service = CommentsService()
        await service.update_comment(
            mock_db,
            comment_id=TEST_COMMENT_ID,
            user_id=TEST_USER_ID,
            content="Updated content",
        )

        assert c.content == "Updated content"
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_comment_not_found(self, mock_db):
        """Should return None when comment doesn't exist."""
        from app.services.comments_service import CommentsService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        service = CommentsService()
        result = await service.update_comment(
            mock_db,
            comment_id=uuid4(),
            user_id=TEST_USER_ID,
            content="Will not be saved",
        )

        assert result is None
        mock_db.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_permission_error_for_non_owner_content_edit(self, mock_db):
        """Only owner can edit comment content."""
        from app.services.comments_service import CommentsService

        other_user = uuid4()
        c = make_comment(user_id=other_user)  # owned by another user
        mock_db.execute.return_value = _scalar_one_or_none(c)

        service = CommentsService()
        with pytest.raises(PermissionError, match="Only comment owner"):
            await service.update_comment(
                mock_db,
                comment_id=TEST_COMMENT_ID,
                user_id=TEST_USER_ID,  # NOT the owner
                content="Stolen edit",
            )

    @pytest.mark.asyncio
    async def test_non_owner_can_resolve_comment(self, mock_db):
        """Any user should be able to resolve a comment (no content change)."""
        from app.services.comments_service import CommentsService

        other_user = uuid4()
        c = make_comment(user_id=other_user, is_resolved=False)
        mock_db.execute.return_value = _scalar_one_or_none(c)

        service = CommentsService()
        # No content= kwarg → no ownership check
        await service.update_comment(
            mock_db,
            comment_id=TEST_COMMENT_ID,
            user_id=TEST_USER_ID,
            is_resolved=True,
        )

        assert c.is_resolved is True


# ─────────────────────────────────────────────────────────────────────────────
# delete_comment
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteComment:
    """Tests for CommentsService.delete_comment."""

    @pytest.mark.asyncio
    async def test_deletes_own_comment(self, mock_db):
        """Owner should be able to delete their comment."""
        from app.services.comments_service import CommentsService

        c = make_comment()
        mock_db.execute.return_value = _scalar_one_or_none(c)

        service = CommentsService()
        result = await service.delete_comment(
            mock_db,
            comment_id=TEST_COMMENT_ID,
            user_id=TEST_USER_ID,
        )

        assert result is True
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_comment_not_found(self, mock_db):
        """Should return False (or raise) when comment ID doesn't exist."""
        from app.services.comments_service import CommentsService

        mock_db.execute.return_value = _scalar_one_or_none(None)

        service = CommentsService()
        result = await service.delete_comment(
            mock_db,
            comment_id=uuid4(),
            user_id=TEST_USER_ID,
        )

        # Service can return False or raise — we check that DB was not committed
        assert result is False or result is None
        mock_db.commit.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# get_comment_count
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCommentCount:

    @pytest.mark.asyncio
    async def test_returns_integer_count(self, mock_db):
        """Should return total comment count for a project."""
        from app.services.comments_service import CommentsService

        result_mock = MagicMock()
        result_mock.scalar.return_value = 42
        mock_db.execute.return_value = result_mock

        service = CommentsService()
        count = await service.get_comment_count(mock_db, TEST_PROJECT_ID)

        assert count == 42
        mock_db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_comments(self, mock_db):
        """Should return 0 when project has no comments."""
        from app.services.comments_service import CommentsService

        result_mock = MagicMock()
        result_mock.scalar.return_value = 0
        mock_db.execute.return_value = result_mock

        service = CommentsService()
        count = await service.get_comment_count(mock_db, TEST_PROJECT_ID)

        assert count == 0

    @pytest.mark.asyncio
    async def test_filters_by_target_id(self, mock_db):
        """Should accept target_id filter without raising."""
        from app.services.comments_service import CommentsService

        result_mock = MagicMock()
        result_mock.scalar.return_value = 5
        mock_db.execute.return_value = result_mock

        service = CommentsService()
        count = await service.get_comment_count(
            mock_db, TEST_PROJECT_ID, target_id="TP53"
        )

        assert count == 5


# ─────────────────────────────────────────────────────────────────────────────
# get_user_comments
# ─────────────────────────────────────────────────────────────────────────────

class TestGetUserComments:

    @pytest.mark.asyncio
    async def test_returns_comments_for_user(self, mock_db):
        """Should return list of comments authored by user."""
        from app.services.comments_service import CommentsService

        c1 = make_comment()
        c2 = make_comment(comment_id=uuid4(), content="Second comment")
        mock_db.execute.return_value = _scalars_all([c1, c2])

        service = CommentsService()
        result = await service.get_user_comments(mock_db, TEST_USER_ID)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_when_user_has_no_comments(self, mock_db):
        """Should return empty list when user has written no comments."""
        from app.services.comments_service import CommentsService

        mock_db.execute.return_value = _scalars_all([])

        service = CommentsService()
        result = await service.get_user_comments(mock_db, TEST_USER_ID)

        assert result == []

    @pytest.mark.asyncio
    async def test_optional_project_filter_accepted(self, mock_db):
        """Should accept project_id filter without raising."""
        from app.services.comments_service import CommentsService

        c = make_comment()
        mock_db.execute.return_value = _scalars_all([c])

        service = CommentsService()
        result = await service.get_user_comments(
            mock_db, TEST_USER_ID, project_id=TEST_PROJECT_ID
        )

        assert len(result) == 1
        mock_db.execute.assert_awaited_once()
