"""
Unit tests for history_service module-level functions.

Covers:
- log_activity: happy path (db.add + db.commit called)
- log_activity: never raises even when db.commit fails
- get_activity_log: returns paginated dict with items/total/limit/offset
- get_activity_log: filters by event_type when provided
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from tests.conftest import TEST_USER_ID, TEST_PROJECT_ID


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_one(value):
    r = MagicMock()
    r.scalar_one.return_value = value
    return r


def _scalars_all(items):
    r = MagicMock()
    r.scalars.return_value.all.return_value = items
    return r


# ─────────────────────────────────────────────────────────────────────────────
# log_activity
# ─────────────────────────────────────────────────────────────────────────────

class TestLogActivity:

    @pytest.mark.asyncio
    async def test_happy_path_calls_add_and_commit(self, mock_db):
        """log_activity should add an entry and commit on success."""
        from app.services.history_service import log_activity
        from app.models.models import ActivityEventType

        await log_activity(
            mock_db,
            project_id=TEST_PROJECT_ID,
            user_id=TEST_USER_ID,
            event_type=ActivityEventType.BOOKMARK_CREATED,
            entity_type="bookmark",
            entity_name="TP53",
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_never_raises_when_commit_fails(self, mock_db):
        """log_activity must swallow exceptions silently (fire-and-forget)."""
        from app.services.history_service import log_activity
        from app.models.models import ActivityEventType

        mock_db.commit.side_effect = RuntimeError("DB down")

        # Should NOT raise
        await log_activity(
            mock_db,
            project_id=TEST_PROJECT_ID,
            user_id=TEST_USER_ID,
            event_type=ActivityEventType.BOOKMARK_CREATED,
        )

    @pytest.mark.asyncio
    async def test_logs_with_metadata(self, mock_db):
        """Extra metadata should be passed to the model."""
        from app.services.history_service import log_activity
        from app.models.models import ActivityEventType, ProjectActivityLog

        await log_activity(
            mock_db,
            project_id=TEST_PROJECT_ID,
            user_id=TEST_USER_ID,
            event_type=ActivityEventType.BOOKMARK_CREATED,
            extra_metadata={"gene": "TP53", "action": "bookmark"},
        )

        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.extra_metadata == {"gene": "TP53", "action": "bookmark"}


# ─────────────────────────────────────────────────────────────────────────────
# get_activity_log
# ─────────────────────────────────────────────────────────────────────────────

class TestGetActivityLog:

    @pytest.mark.asyncio
    async def test_returns_paginated_dict(self, mock_db):
        """get_activity_log should return dict with items/total/limit/offset."""
        from app.services.history_service import get_activity_log
        from app.models.models import ProjectActivityLog

        entry = MagicMock(spec=ProjectActivityLog)
        mock_db.execute.side_effect = [
            _scalar_one(5),       # COUNT query
            _scalars_all([entry]),# items query
        ]

        result = await get_activity_log(mock_db, TEST_PROJECT_ID, limit=10, offset=0)

        assert result["total"] == 5
        assert result["items"] == [entry]
        assert result["limit"] == 10
        assert result["offset"] == 0

    @pytest.mark.asyncio
    async def test_returns_empty_items_when_no_activity(self, mock_db):
        """Should return empty items list when no activity exists."""
        from app.services.history_service import get_activity_log

        mock_db.execute.side_effect = [
            _scalar_one(0),
            _scalars_all([]),
        ]

        result = await get_activity_log(mock_db, TEST_PROJECT_ID)

        assert result["total"] == 0
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_filters_by_event_type(self, mock_db):
        """Should accept event_type_filter without raising."""
        from app.services.history_service import get_activity_log
        from app.models.models import ActivityEventType

        mock_db.execute.side_effect = [
            _scalar_one(2),
            _scalars_all([MagicMock(), MagicMock()]),
        ]

        result = await get_activity_log(
            mock_db,
            TEST_PROJECT_ID,
            event_type_filter=ActivityEventType.BOOKMARK_CREATED,
        )

        assert result["total"] == 2
        assert len(result["items"]) == 2

    @pytest.mark.asyncio
    async def test_respects_limit_and_offset(self, mock_db):
        """Limit and offset should be returned in the response dict."""
        from app.services.history_service import get_activity_log

        mock_db.execute.side_effect = [
            _scalar_one(100),
            _scalars_all([MagicMock() for _ in range(20)]),
        ]

        result = await get_activity_log(mock_db, TEST_PROJECT_ID, limit=20, offset=40)

        assert result["limit"] == 20
        assert result["offset"] == 40
