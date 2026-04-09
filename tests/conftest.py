"""
Shared test fixtures for GenoLens backend tests.

Provides:
- Async DB session mocks (no real DB required for unit tests)
- Reusable UUIDs, model instances, and factory helpers
- AsyncClient for integration tests against the FastAPI app
"""
# ── Environment bootstrap ────────────────────────────────────────────────────
# Must happen BEFORE any app.* import so that pydantic-settings picks up
# the correct LOCAL_STORAGE_PATH and avoids trying to create /app/data
# (Docker-only path) on the developer's machine.
import os
import tempfile

_TEST_STORAGE_DIR = tempfile.mkdtemp(prefix="genolens_test_")
os.environ.setdefault("LOCAL_STORAGE_PATH", _TEST_STORAGE_DIR)
# ────────────────────────────────────────────────────────────────────────────

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
from datetime import datetime

# ─────────────────────────────────────────────
# Constant test IDs (deterministic for fixtures)
# ─────────────────────────────────────────────
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")
TEST_PROJECT_ID = UUID("00000000-0000-0000-0000-000000000002")
TEST_BOOKMARK_ID = UUID("00000000-0000-0000-0000-000000000003")
TEST_COMMENT_ID = UUID("00000000-0000-0000-0000-000000000004")
TEST_PARENT_COMMENT_ID = UUID("00000000-0000-0000-0000-000000000005")
TEST_GENE_LIST_ID = UUID("00000000-0000-0000-0000-000000000006")
TEST_DATASET_ID = UUID("00000000-0000-0000-0000-000000000007")


# ─────────────────────────────────────────────
# Mock DB Session
# ─────────────────────────────────────────────

@pytest.fixture
def mock_db():
    """
    Provides a mocked AsyncSession.
    Callers patch `execute`, `add`, `commit`, `refresh`, `delete` as needed.
    """
    db = AsyncMock()
    db.add = MagicMock()       # synchronous in SQLAlchemy
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()
    return db


# ─────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────

def make_project(
    project_id: UUID = TEST_PROJECT_ID,
    owner_id: UUID = TEST_USER_ID,
    name: str = "Test Project",
):
    """Create a minimal Project mock."""
    from app.models.models import Project
    project = MagicMock(spec=Project)
    project.id = project_id
    project.owner_id = owner_id
    project.name = name
    project.description = None
    project.created_at = datetime.utcnow()
    project.updated_at = datetime.utcnow()
    return project


def make_bookmark(
    bookmark_id: UUID = TEST_BOOKMARK_ID,
    user_id: UUID = TEST_USER_ID,
    project_id: UUID = TEST_PROJECT_ID,
    gene_symbol: str = "TP53",
    notes: str = "Important tumor suppressor",
    tags: list | None = None,
    color: str = "#FF5733",
    is_favorite: bool = True,
):
    """Create a minimal GeneBookmark mock."""
    from app.models.models import GeneBookmark
    bm = MagicMock(spec=GeneBookmark)
    bm.id = bookmark_id
    bm.user_id = user_id
    bm.project_id = project_id
    bm.gene_symbol = gene_symbol
    bm.notes = notes
    bm.tags = tags or ["cancer", "p53"]
    bm.color = color
    bm.is_favorite = is_favorite
    bm.gene_id = None
    bm.extra_data = {}
    bm.created_at = datetime.utcnow()
    bm.updated_at = datetime.utcnow()
    return bm


def make_comment(
    comment_id: UUID = TEST_COMMENT_ID,
    project_id: UUID = TEST_PROJECT_ID,
    user_id: UUID = TEST_USER_ID,
    content: str = "This is a test comment",
    parent_id: UUID | None = None,
    is_resolved: bool = False,
):
    """Create a minimal ProjectComment mock."""
    from app.models.models import ProjectComment, CommentType
    c = MagicMock(spec=ProjectComment)
    c.id = comment_id
    c.project_id = project_id
    c.user_id = user_id
    c.content = content
    c.comment_type = CommentType.GENERAL
    c.parent_id = parent_id
    c.target_id = None
    c.is_resolved = is_resolved
    c.replies = []
    c.parent = None
    c.extra_metadata = {}
    c.created_at = datetime.utcnow()
    c.updated_at = datetime.utcnow()
    return c


def make_gene_list(
    list_id: UUID = TEST_GENE_LIST_ID,
    user_id: UUID = TEST_USER_ID,
    project_id: UUID = TEST_PROJECT_ID,
    name: str = "My Gene List",
    genes: list | None = None,
):
    """Create a minimal GeneList mock."""
    from app.models.models import GeneList
    gl = MagicMock(spec=GeneList)
    gl.id = list_id
    gl.user_id = user_id
    gl.project_id = project_id
    gl.name = name
    gl.genes = genes if genes is not None else ["TP53", "BRCA1"]
    gl.gene_count = len(gl.genes)
    gl.description = ""
    gl.color = "#3B82F6"
    gl.is_public = False
    gl.tags = []
    gl.extra_data = {}
    gl.created_at = datetime.utcnow()
    gl.updated_at = datetime.utcnow()
    return gl


def make_cached_computation(
    dataset_id: UUID = TEST_DATASET_ID,
    computation_type: str = "clustering",
    is_expired: bool = False,
    hit_count: int = 0,
):
    """Create a minimal CachedComputation mock."""
    from app.models.models import CachedComputation
    cc = MagicMock(spec=CachedComputation)
    cc.id = uuid4()
    cc.dataset_id = dataset_id
    cc.computation_type = computation_type
    cc.params_hash = "abc123"
    cc.result_data = {"key": "value"}
    cc.is_expired = is_expired
    cc.hit_count = hit_count
    cc.last_accessed_at = None
    cc.expires_at = None
    cc.created_at = datetime.utcnow()
    return cc


# ── Integration-test auth override ───────────────────────────────────────────

def make_fake_supabase_user(
    user_id: UUID = TEST_USER_ID,
    email: str = "test@example.com",
):
    """Return a SupabaseUser suitable for overriding get_current_user."""
    from app.core.supabase_auth import SupabaseUser
    return SupabaseUser(user_id=user_id, email=email)


# ─────────────────────────────────────────────
# Integration-test client
# ─────────────────────────────────────────────

@pytest_asyncio.fixture
async def async_client():
    """
    Async HTTP client for integration tests (no real DB: 
    individual tests must mock the DB dependency via override_dependencies).
    """
    from httpx import AsyncClient
    from app.main import app
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client
