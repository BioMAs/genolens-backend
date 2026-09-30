"""
Integration tests for Comments API endpoints.

Uses dependency_overrides for auth/db and patches the comments_service singleton.
The fake DB answers the two queries of `assert_project_read_access` (Project,
then ProjectMember), so every test runs the real access check.

Covers:
- Stranger (neither owner nor member) gets 404 on every route, the service is
  never reached, and no mention/reply email is sent
- Member reads, counts, creates, fetches, threads and resolves
- parent_id from another project is rejected
- Author-only edits and deletes map PermissionError to 403 without leaking text
- A former author who lost project access gets 404 on delete
"""
import pytest
import pytest_asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from httpx import AsyncClient, ASGITransport

from tests.conftest import (
    TEST_USER_ID, TEST_PROJECT_ID, TEST_COMMENT_ID,
    make_comment, make_fake_current_user, make_project,
)

OWNER_ID = TEST_USER_ID
MEMBER_ID = UUID("00000000-0000-0000-0000-0000000000aa")
STRANGER_ID = UUID("00000000-0000-0000-0000-0000000000bb")
OTHER_PROJECT_ID = UUID("00000000-0000-0000-0000-0000000000cc")

SVC = "app.api.endpoints.comments.comments_service"
EMAIL = "app.api.endpoints.comments.email_service"


def make_client(*, as_user: UUID, member: bool = False, project_exists: bool = True):
    """Client where TEST_PROJECT_ID is owned by OWNER_ID."""
    from app.main import app
    from app.api.deps.auth import get_current_user
    from app.api.deps.db import get_db

    project = make_project(owner_id=OWNER_ID) if project_exists else None
    membership = (
        SimpleNamespace(project_id=TEST_PROJECT_ID, user_id=as_user) if member else None
    )

    async def _fake_db():
        mock = AsyncMock()

        async def _scalar(stmt):
            entity = stmt.column_descriptions[0]["entity"].__name__
            return project if entity == "Project" else membership

        mock.scalar = AsyncMock(side_effect=_scalar)
        yield mock

    app.dependency_overrides[get_current_user] = lambda: make_fake_current_user(
        user_id=as_user, email=f"{as_user}@example.com"
    )
    app.dependency_overrides[get_db] = _fake_db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app
    from app.api.deps.auth import get_current_user
    from app.api.deps.db import get_db
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_db, None)


def _comment(**kw):
    kw.setdefault("user_id", OWNER_ID)
    return make_comment(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# Stranger: 404 everywhere, nothing reached
# ─────────────────────────────────────────────────────────────────────────────

class TestStrangerIsRejected:

    @pytest.mark.asyncio
    async def test_list_404(self):
        get_comments = AsyncMock(return_value=[_comment()])
        with patch(f"{SVC}.get_comments", new=get_comments):
            async with make_client(as_user=STRANGER_ID) as c:
                resp = await c.get(f"/api/v1/projects/{TEST_PROJECT_ID}/comments")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"
        get_comments.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_count_404(self):
        get_count = AsyncMock(return_value=3)
        with patch(f"{SVC}.get_comment_count", new=get_count):
            async with make_client(as_user=STRANGER_ID) as c:
                resp = await c.get(f"/api/v1/projects/{TEST_PROJECT_ID}/comments/count")
        assert resp.status_code == 404
        get_count.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_404_and_no_email(self):
        create = AsyncMock(return_value=_comment())
        send_mention = AsyncMock()
        send_reply = AsyncMock()
        with patch(f"{SVC}.create_comment", new=create), \
             patch(f"{EMAIL}.send_mention_notification", new=send_mention), \
             patch(f"{EMAIL}.send_reply_notification", new=send_reply), \
             patch("app.api.endpoints.comments.history_service.log_activity", new=AsyncMock()):
            async with make_client(as_user=STRANGER_ID) as c:
                resp = await c.post(
                    f"/api/v1/projects/{TEST_PROJECT_ID}/comments",
                    json={"content": "Hi @victim@example.org, see this"},
                )
        assert resp.status_code == 404
        create.assert_not_awaited()
        send_mention.assert_not_awaited()
        send_reply.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_404(self):
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=_comment())):
            async with make_client(as_user=STRANGER_ID) as c:
                resp = await c.get(f"/api/v1/comments/{TEST_COMMENT_ID}")
        assert resp.status_code == 404
        # Same answer as for a comment that does not exist.
        assert resp.json()["detail"] == "Comment not found"

    @pytest.mark.asyncio
    async def test_thread_404(self):
        get_thread = AsyncMock(return_value=[_comment()])
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=_comment())), \
             patch(f"{SVC}.get_comment_thread", new=get_thread):
            async with make_client(as_user=STRANGER_ID) as c:
                resp = await c.get(f"/api/v1/comments/{TEST_COMMENT_ID}/thread")
        assert resp.status_code == 404
        get_thread.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_patch_resolve_404(self):
        update = AsyncMock(return_value=_comment(is_resolved=True))
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=_comment())), \
             patch(f"{SVC}.update_comment", new=update):
            async with make_client(as_user=STRANGER_ID) as c:
                resp = await c.patch(
                    f"/api/v1/comments/{TEST_COMMENT_ID}", json={"is_resolved": True}
                )
        assert resp.status_code == 404
        update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_404_even_for_former_author(self):
        """The author of a comment who has since lost project access."""
        delete = AsyncMock(return_value=True)
        own_comment = _comment(user_id=STRANGER_ID)
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=own_comment)), \
             patch(f"{SVC}.delete_comment", new=delete):
            async with make_client(as_user=STRANGER_ID) as c:
                resp = await c.delete(f"/api/v1/comments/{TEST_COMMENT_ID}")
        assert resp.status_code == 404
        delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_project_404(self):
        async with make_client(as_user=OWNER_ID, project_exists=False) as c:
            resp = await c.get(f"/api/v1/projects/{uuid4()}/comments")
        assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Member: allowed
# ─────────────────────────────────────────────────────────────────────────────

class TestMemberIsAllowed:

    @pytest.mark.asyncio
    async def test_list_200(self):
        with patch(f"{SVC}.get_comments", new=AsyncMock(return_value=[_comment()])):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.get(f"/api/v1/projects/{TEST_PROJECT_ID}/comments")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    @pytest.mark.asyncio
    async def test_list_invalid_type_400(self):
        async with make_client(as_user=MEMBER_ID, member=True) as c:
            resp = await c.get(
                f"/api/v1/projects/{TEST_PROJECT_ID}/comments", params={"comment_type": "nope"}
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_count_200(self):
        with patch(f"{SVC}.get_comment_count", new=AsyncMock(return_value=7)):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.get(f"/api/v1/projects/{TEST_PROJECT_ID}/comments/count")
        assert resp.status_code == 200
        assert resp.json()["count"] == 7

    @pytest.mark.asyncio
    async def test_owner_count_200(self):
        with patch(f"{SVC}.get_comment_count", new=AsyncMock(return_value=0)):
            async with make_client(as_user=OWNER_ID) as c:
                resp = await c.get(f"/api/v1/projects/{TEST_PROJECT_ID}/comments/count")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_201_sends_mention(self):
        create = AsyncMock(return_value=_comment(user_id=MEMBER_ID, content="Hi @a@example.org"))
        send_mention = AsyncMock()
        with patch(f"{SVC}.create_comment", new=create), \
             patch(f"{EMAIL}.send_mention_notification", new=send_mention), \
             patch("app.api.endpoints.comments.history_service.log_activity", new=AsyncMock()):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.post(
                    f"/api/v1/projects/{TEST_PROJECT_ID}/comments",
                    json={"content": "Hi @a@example.org"},
                )
        assert resp.status_code == 201
        assert create.await_args.kwargs["user_id"] == MEMBER_ID
        send_mention.assert_awaited_once()
        assert send_mention.await_args.kwargs["project_name"] == "Test Project"

    @pytest.mark.asyncio
    async def test_create_rejects_parent_from_other_project(self):
        create = AsyncMock()
        foreign_parent = _comment(comment_id=uuid4(), project_id=OTHER_PROJECT_ID)
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=foreign_parent)), \
             patch(f"{SVC}.create_comment", new=create):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.post(
                    f"/api/v1/projects/{TEST_PROJECT_ID}/comments",
                    json={"content": "reply", "parent_id": str(foreign_parent.id)},
                )
        assert resp.status_code == 400
        create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_rejects_unknown_parent(self):
        create = AsyncMock()
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=None)), \
             patch(f"{SVC}.create_comment", new=create):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.post(
                    f"/api/v1/projects/{TEST_PROJECT_ID}/comments",
                    json={"content": "reply", "parent_id": str(uuid4())},
                )
        assert resp.status_code == 400
        create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_missing_content_422(self):
        async with make_client(as_user=MEMBER_ID, member=True) as c:
            resp = await c.post(f"/api/v1/projects/{TEST_PROJECT_ID}/comments", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_200(self):
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=_comment())):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.get(f"/api/v1/comments/{TEST_COMMENT_ID}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_missing_404(self):
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=None)):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.get(f"/api/v1/comments/{uuid4()}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_thread_200(self):
        root = _comment()
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=root)), \
             patch(f"{SVC}.get_comment_thread", new=AsyncMock(return_value=[root])):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.get(f"/api/v1/comments/{TEST_COMMENT_ID}/thread")
        assert resp.status_code == 200
        assert resp.json()["reply_count"] == 0

    @pytest.mark.asyncio
    async def test_patch_resolve_200(self):
        update = AsyncMock(return_value=_comment(is_resolved=True))
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=_comment())), \
             patch(f"{SVC}.update_comment", new=update):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.patch(
                    f"/api/v1/comments/{TEST_COMMENT_ID}", json={"is_resolved": True}
                )
        assert resp.status_code == 200
        assert update.await_args.kwargs["user_id"] == MEMBER_ID

    @pytest.mark.asyncio
    async def test_patch_content_by_non_author_403_without_leak(self):
        update = AsyncMock(side_effect=PermissionError("internal detail"))
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=_comment())), \
             patch(f"{SVC}.update_comment", new=update):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.patch(
                    f"/api/v1/comments/{TEST_COMMENT_ID}", json={"content": "edit"}
                )
        assert resp.status_code == 403
        assert "internal detail" not in resp.text

    @pytest.mark.asyncio
    async def test_delete_by_author_204(self):
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=_comment())), \
             patch(f"{SVC}.delete_comment", new=AsyncMock(return_value=True)):
            async with make_client(as_user=OWNER_ID) as c:
                resp = await c.delete(f"/api/v1/comments/{TEST_COMMENT_ID}")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_by_non_author_403(self):
        delete = AsyncMock(side_effect=PermissionError("Only comment owner can delete"))
        with patch(f"{SVC}.get_comment_by_id", new=AsyncMock(return_value=_comment())), \
             patch(f"{SVC}.delete_comment", new=delete):
            async with make_client(as_user=MEMBER_ID, member=True) as c:
                resp = await c.delete(f"/api/v1/comments/{TEST_COMMENT_ID}")
        assert resp.status_code == 403
