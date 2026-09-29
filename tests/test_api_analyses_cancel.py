"""
POST /analyses/{id}/cancel — annulation d'une analyse en attente ou en cours.

Avant cette route, le seul chemin était DELETE, qui révoquait la tâche puis
supprimait l'enregistrement : aucune analyse ne passait jamais à CANCELLED, et
le bouton « Cancel » de l'assistant appelait une URL /api/v2 inexistante, sans
jeton.

Couvre :
- le lanceur et un ADMIN du projet peuvent annuler ; statut CANCELLED,
  tâche Celery révoquée avec terminate=True
- un simple membre qui n'a pas lancé l'analyse : 403 ; un inconnu : 403 ;
  analyse introuvable : 404 — et dans ces cas rien n'est révoqué
- une analyse déjà terminée : 409, pas de révocation
- aucune unité de quota : la route ne touche pas `users`, l'analyse annulée
  sort des réservations en vol, et le worker relit le statut sous verrou avant
  de décompter.
"""

import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.models.models import SelfServiceAnalysisStatus
from tests.conftest import TEST_PROJECT_ID, TEST_USER_ID, make_fake_supabase_user, make_project

OWNER_ID = TEST_USER_ID
LAUNCHER_ID = UUID("00000000-0000-0000-0000-0000000000cc")
MEMBER_ID = UUID("00000000-0000-0000-0000-0000000000aa")
STRANGER_ID = UUID("00000000-0000-0000-0000-0000000000bb")
ANALYSIS_ID = uuid4()
TASK_ID = "celery-task-123"


def make_analysis(user_id: UUID = LAUNCHER_ID, status=SelfServiceAnalysisStatus.RUNNING):
    now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=ANALYSIS_ID,
        project_id=TEST_PROJECT_ID,
        name="Demo analysis",
        data_type="transcriptomics",
        status=status,
        matrix_dataset_id=uuid4(),
        samples_dataset_id=uuid4(),
        comparisons_dataset_id=uuid4(),
        params={},
        result_dataset_ids=[],
        intermediate_dataset_ids={},
        celery_task_id=TASK_ID,
        current_step="running_pipeline",
        progress_log=[],
        error_message=None,
        user_id=user_id,
        created_at=now,
        updated_at=now,
    )


class FakeDB:
    """Session factice : scalar() rend `scalars` dans l'ordre ; execute() rend
    `updated_id` (None = l'UPDATE conditionnel n'a touché aucune ligne)."""

    def __init__(self, scalars, updated_id):
        self.scalar = AsyncMock(side_effect=list(scalars))
        result = MagicMock()
        result.scalar = MagicMock(return_value=updated_id)
        self.execute = AsyncMock(return_value=result)
        self.commit = AsyncMock()

        async def _refresh(obj):
            obj.status = SelfServiceAnalysisStatus.CANCELLED
            obj.current_step = "cancelled"

        self.refresh = AsyncMock(side_effect=_refresh)


def make_client(db: FakeDB, *, as_user: UUID):
    from app.api.deps.supabase_deps import get_current_user, get_db
    from app.main import app

    async def _fake_db():
        yield db

    app.dependency_overrides[get_current_user] = lambda: make_fake_supabase_user(user_id=as_user)
    app.dependency_overrides[get_db] = _fake_db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest_asyncio.fixture(autouse=True)
async def _clear_overrides():
    yield
    from app.main import app

    app.dependency_overrides.clear()


@pytest.fixture
def revoke():
    from app.worker.celery_app import celery_app

    control = MagicMock()
    with patch.object(celery_app, "control", control):
        yield control.revoke


def _tables_written(db: FakeDB) -> set:
    return {call.args[0].table.name for call in db.execute.await_args_list}


async def _post_cancel(db: FakeDB, as_user: UUID):
    async with make_client(db, as_user=as_user) as c:
        return await c.post(f"/api/v1/analyses/{ANALYSIS_ID}/cancel")


# ── Qui peut annuler ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_launcher_cancels_running_analysis(revoke):
    project = make_project(owner_id=OWNER_ID)
    member = SimpleNamespace(project_id=TEST_PROJECT_ID, user_id=LAUNCHER_ID)
    db = FakeDB([make_analysis(user_id=LAUNCHER_ID), project, member], updated_id=ANALYSIS_ID)
    resp = await _post_cancel(db, LAUNCHER_ID)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "CANCELLED"
    revoke.assert_called_once_with(TASK_ID, terminate=True)
    db.commit.assert_awaited_once()

    stmt = db.execute.await_args.args[0]
    assert stmt.table.name == "self_service_analyses"
    assert SelfServiceAnalysisStatus.CANCELLED in stmt.compile().params.values()
    # UPDATE conditionnel : ne repeint pas une analyse déjà terminée.
    assert "status IN" in str(stmt)


@pytest.mark.asyncio
async def test_pending_analysis_can_be_cancelled(revoke):
    analysis = make_analysis(user_id=OWNER_ID, status=SelfServiceAnalysisStatus.PENDING)
    db = FakeDB([analysis, make_project(owner_id=OWNER_ID)], updated_id=ANALYSIS_ID)
    resp = await _post_cancel(db, OWNER_ID)

    assert resp.status_code == 200, resp.text
    revoke.assert_called_once_with(TASK_ID, terminate=True)


@pytest.mark.asyncio
async def test_project_admin_can_cancel_someone_elses_analysis(revoke):
    project = make_project(owner_id=OWNER_ID)
    membership = SimpleNamespace(project_id=TEST_PROJECT_ID, user_id=MEMBER_ID)
    admin_row = SimpleNamespace(project_id=TEST_PROJECT_ID, user_id=MEMBER_ID)
    db = FakeDB([make_analysis(), project, membership, admin_row], updated_id=ANALYSIS_ID)
    resp = await _post_cancel(db, MEMBER_ID)

    assert resp.status_code == 200, resp.text
    revoke.assert_called_once_with(TASK_ID, terminate=True)


@pytest.mark.asyncio
async def test_project_owner_can_cancel_a_members_analysis(revoke):
    db = FakeDB([make_analysis(), make_project(owner_id=OWNER_ID)], updated_id=ANALYSIS_ID)
    resp = await _post_cancel(db, OWNER_ID)

    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_plain_member_cannot_cancel_someone_elses_analysis(revoke):
    project = make_project(owner_id=OWNER_ID)
    membership = SimpleNamespace(project_id=TEST_PROJECT_ID, user_id=MEMBER_ID)
    db = FakeDB([make_analysis(), project, membership, None], updated_id=ANALYSIS_ID)
    resp = await _post_cancel(db, MEMBER_ID)

    assert resp.status_code == 403
    db.execute.assert_not_awaited()
    revoke.assert_not_called()


@pytest.mark.asyncio
async def test_non_member_is_refused(revoke):
    db = FakeDB([make_analysis(), make_project(owner_id=OWNER_ID), None], updated_id=ANALYSIS_ID)
    resp = await _post_cancel(db, STRANGER_ID)

    assert resp.status_code in (403, 404)
    db.execute.assert_not_awaited()
    db.commit.assert_not_awaited()
    revoke.assert_not_called()


@pytest.mark.asyncio
async def test_missing_analysis_404(revoke):
    resp = await _post_cancel(FakeDB([None], updated_id=None), OWNER_ID)

    assert resp.status_code == 404
    revoke.assert_not_called()


@pytest.mark.asyncio
async def test_finished_analysis_409_and_not_revoked(revoke):
    """Le worker a fini entre la lecture et l'UPDATE : zéro ligne touchée."""
    analysis = make_analysis(user_id=OWNER_ID, status=SelfServiceAnalysisStatus.DONE)
    db = FakeDB([analysis, make_project(owner_id=OWNER_ID)], updated_id=None)
    resp = await _post_cancel(db, OWNER_ID)

    assert resp.status_code == 409
    db.commit.assert_not_awaited()
    revoke.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_failure_still_cancels():
    """Broker injoignable : le statut CANCELLED suffit, le worker s'arrêtera."""
    from app.worker.celery_app import celery_app

    db = FakeDB([make_analysis(user_id=OWNER_ID), make_project(owner_id=OWNER_ID)], ANALYSIS_ID)
    control = MagicMock()
    control.revoke.side_effect = ConnectionError("broker down")
    with patch.object(celery_app, "control", control):
        resp = await _post_cancel(db, OWNER_ID)

    assert resp.status_code == 200
    assert resp.json()["status"] == "CANCELLED"


# ── Quota ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_consumes_no_quota(revoke):
    db = FakeDB([make_analysis(user_id=OWNER_ID), make_project(owner_id=OWNER_ID)], ANALYSIS_ID)
    resp = await _post_cancel(db, OWNER_ID)

    assert resp.status_code == 200
    assert _tables_written(db) == {"self_service_analyses"}


@pytest.mark.asyncio
async def test_cancelled_analysis_releases_its_in_flight_reservation():
    """`_in_flight_analyses` réserve une unité par analyse PENDING/RUNNING ; une
    analyse annulée ne doit plus en bloquer une."""
    from app.api.endpoints.analyses import _in_flight_analyses

    db = AsyncMock()
    db.scalar = AsyncMock(return_value=0)
    await _in_flight_analyses(db, OWNER_ID)
    params = db.scalar.await_args.args[0].compile().params
    reserved = {s for v in params.values() if isinstance(v, (list, tuple)) for s in v}
    assert SelfServiceAnalysisStatus.CANCELLED not in reserved
    assert {SelfServiceAnalysisStatus.PENDING, SelfServiceAnalysisStatus.RUNNING} <= reserved


@pytest.mark.asyncio
async def test_worker_cancel_check_reads_status_under_lock():
    from sqlalchemy.dialects import postgresql

    from app.worker.tasks import _analysis_cancelled

    db = AsyncMock()
    db.scalar = AsyncMock(return_value=SelfServiceAnalysisStatus.CANCELLED)
    assert await _analysis_cancelled(db, ANALYSIS_ID, lock=True) is True
    sql = str(db.scalar.await_args.args[0].compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in sql

    db.scalar = AsyncMock(return_value=SelfServiceAnalysisStatus.RUNNING)
    assert await _analysis_cancelled(db, ANALYSIS_ID, lock=True) is False


def test_worker_checks_cancellation_before_counting_quota():
    """La boucle du worker (400 lignes, sous-processus R) n'est pas testable en
    isolation. On épingle l'ordre : le contrôle d'annulation sous verrou vient
    avant le passage à DONE et avant le décompte de quota."""
    from app.worker import tasks

    src = inspect.getsource(tasks.run_self_service_analysis)
    check = src.index("_analysis_cancelled(db, UUID(analysis_id), lock=True)")
    assert check < src.index("analysis.status = SelfServiceAnalysisStatus.DONE")
    assert check < src.index("_count_pipeline_analysis(quota_user")
