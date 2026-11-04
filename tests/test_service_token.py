"""
Scoped service tokens (`glst_…`) and the `use_ai` switch of the editorial report.

A service token is the credential of a machine client (the analysis pipeline) instead of a
user's JWT. It acts as its owner — licence, plan, modules and quota apply unchanged — on the
routes its scopes open, and is refused everywhere else. Only its sha256 is stored.

The API tests run the real authentication chain (`get_current_user` → `get_or_create_user` →
`require_active_account` → the route's gates) against an in-memory SQLite database. Nothing
hosted is called: the Supabase role lookup and the LLM are replaced by fakes that fail the test
if they are reached.
"""

import hashlib
import io
import json
import re
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import jwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, text

from app.models.models import (
    Dataset,
    DatasetStatus,
    DatasetType,
    Project,
    ProjectMember,
    SubscriptionPlan,
    User,
    UserRole,
    UserStatus,
)
from app.models.report_job import ReportJob, ReportJobStatus

API = "/api/v1"
COMPARISON = "KO_vs_WT"
TOKEN_RE = re.compile(r"glst_[A-Za-z0-9_-]{43}")
ALL_SCOPES = ("projects:create", "datasets:upload", "reports:editorial")
EDITORIAL = {"comparison_name": COMPARISON, "template": "editorial"}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def sessions():
    """An in-memory SQLite database holding the tables these routes touch."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.schema import CreateTable

    from app.models.service_token import ServiceToken

    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    tables = (User, ServiceToken, Project, ProjectMember, Dataset, ReportJob)
    try:
        async with engine.begin() as conn:
            for model in tables:
                # Table only: some Postgres index names collide in SQLite's single namespace.
                await conn.execute(CreateTable(model.__table__))
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.fixture
def no_hosted_service():
    """Fail on any Supabase role lookup; never record login events (they open a real session)."""

    async def _supabase(*_args, **_kwargs):
        raise AssertionError("Supabase was called")

    with (
        patch("app.core.supabase_auth.fetch_user_role_from_supabase", new=_supabase),
        patch("app.middleware.login_tracking._record_login_event", new=AsyncMock()),
    ):
        yield


@pytest_asyncio.fixture
async def client(sessions, no_hosted_service):
    """The app on the SQLite database, with the real authentication chain and a valid licence."""
    from app.api.deps.license import require_active_license
    from app.db.session import get_db
    from app.main import app

    async def _get_db():
        async with sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # The real chain: no override left by another module, nor conftest's default account.
    app.dependency_overrides.clear()
    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[require_active_license] = lambda: None
    with ExitStack() as stack:
        stack.enter_context(
            patch("app.api.endpoints.datasets.process_dataset_upload.delay", new=MagicMock())
        )
        stack.enter_context(
            patch(
                "app.api.endpoints.datasets.storage_service.upload_file",
                new=AsyncMock(return_value="projects/raw/table.tsv"),
            )
        )
        stack.enter_context(
            patch("app.api.endpoints.datasets.history_service.log_activity", new=AsyncMock())
        )
        stack.enter_context(
            patch(
                "app.worker.tasks.report_task.generate_comparison_report_editorial.delay",
                new=MagicMock(return_value=SimpleNamespace(id="editorial-task")),
            )
        )
        stack.enter_context(
            patch(
                "app.worker.tasks.report_task.generate_comparison_report.delay",
                new=MagicMock(return_value=SimpleNamespace(id="latex-task")),
            )
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
            yield c
    app.dependency_overrides.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def seed_owner(sessions, **fields) -> User:
    """An account of the fictitious organisation the pipeline deposits under."""
    values = dict(
        id=uuid4(),
        email=f"pipeline-{uuid4().hex[:8]}@example.test",
        role=UserRole.USER,
        subscription_plan=SubscriptionPlan.TEAM,
        status=UserStatus.ACTIVE,
        cosmetics_module_enabled=True,
        report_customization_module_enabled=False,
        is_demo=False,
    )
    values.update(fields)
    async with sessions() as s:
        user = User(**values)
        s.add(user)
        await s.commit()
        return user


async def make_token(sessions, owner: User, scopes=ALL_SCOPES, name="pipeline") -> tuple[UUID, str]:
    from app.scripts.service_token import create_service_token

    async with sessions() as s:
        row, token = await create_service_token(s, owner=owner.email, name=name, scopes=scopes)
    return row.id, token


async def seed_dataset(sessions, owner: User) -> UUID:
    """A READY DEG dataset in a project of `owner`."""
    async with sessions() as s:
        project = Project(id=uuid4(), name="TST01 v1", owner_id=owner.id)
        dataset = Dataset(
            id=uuid4(),
            project_id=project.id,
            name="TST01 DEG",
            type=DatasetType.DEG,
            status=DatasetStatus.READY,
            dataset_metadata={},
            column_mapping={},
        )
        s.add_all([project, dataset])
        await s.commit()
        return dataset.id


async def jobs_of(sessions, dataset_id: UUID) -> list[ReportJob]:
    async with sessions() as s:
        rows = await s.execute(
            select(ReportJob).where(ReportJob.dataset_id == dataset_id).order_by(ReportJob.id)
        )
        return list(rows.scalars())


async def upload(c, token, project_id, dataset_type="DEG"):
    return await c.post(
        f"{API}/datasets/upload",
        headers=bearer(token),
        data={
            "project_id": project_id,
            "name": "TST01 C01",
            "dataset_type": dataset_type,
            "comparison_name": COMPARISON,
        },
        files={"file": ("C01.tsv", b"gene_id\tlog2FoldChange\tpadj\nG1\t1.5\t0.01\n")},
    )


async def run_command(sessions, *argv) -> tuple[int, str]:
    from app.scripts.service_token import run

    out, err = io.StringIO(), io.StringIO()
    code = await run(list(argv), session_factory=sessions, out=out, err=err)
    return code, out.getvalue() + err.getvalue()


# ── The token acts as its owner on its scoped routes ─────────────────────────


async def test_service_token_creates_a_project_and_uploads(client, sessions):
    owner = await seed_owner(sessions)
    _, token = await make_token(sessions, owner)

    created = await client.post(
        f"{API}/projects/", headers=bearer(token), json={"name": "TST01 v1", "description": "test"}
    )
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]

    deposited = await upload(client, token, project_id)
    assert deposited.status_code == 201, deposited.text
    dataset_id = deposited.json()["dataset_id"]

    polled = await client.get(f"{API}/datasets/{dataset_id}", headers=bearer(token))
    assert polled.status_code == 200, polled.text
    assert polled.json()["status"] == "PENDING"

    async with sessions() as s:
        project = await s.get(Project, UUID(project_id))
        assert project.owner_id == owner.id
        dataset = await s.get(Dataset, UUID(dataset_id))
        assert dataset.project_id == project.id
        # The owner's DEG quota counted the deposit, as for any of their uploads.
        assert (await s.get(User, owner.id)).analyses_used_this_month == 1


async def test_service_token_is_stored_hashed(sessions):
    owner = await seed_owner(sessions)

    code, printed = await run_command(
        sessions, "create", "--owner", owner.email, "--name", "pipeline"
    )
    assert code == 0, printed
    shown = TOKEN_RE.findall(printed)
    assert len(shown) == 1, printed  # printed once, at creation
    token = shown[0]
    assert printed.count(token) == 1

    async with sessions() as s:
        rows = (await s.execute(text("SELECT * FROM service_tokens"))).mappings().all()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["token_sha256"] == hashlib.sha256(token.encode()).hexdigest()
    assert not any(token in str(value) for value in row.values())
    assert json.loads(row["scopes"]) == list(ALL_SCOPES)
    assert row["revoked_at"] is None and row["last_used_at"] is None

    code, listing = await run_command(sessions, "list")
    assert code == 0, listing
    assert str(row["id"]).replace("-", "") in listing.replace("-", "")
    assert "pipeline" in listing and owner.email in listing
    assert token not in listing and row["token_sha256"] not in listing


async def test_revoked_service_token_is_refused(client, sessions):
    owner = await seed_owner(sessions)
    token_id, token = await make_token(sessions, owner)

    first = await client.post(f"{API}/projects/", headers=bearer(token), json={"name": "TST01"})
    assert first.status_code == 201, first.text
    async with sessions() as s:
        from app.models.service_token import ServiceToken

        assert (await s.get(ServiceToken, token_id)).last_used_at is not None

    code, printed = await run_command(sessions, "revoke", str(token_id))
    assert code == 0, printed

    again = await client.post(f"{API}/projects/", headers=bearer(token), json={"name": "TST02"})
    assert again.status_code == 401, again.text
    assert token not in again.text
    polled = await client.get(f"{API}/datasets/{uuid4()}", headers=bearer(token))
    assert polled.status_code == 401

    unknown = "glst_" + "A" * 43
    assert (
        await client.post(f"{API}/projects/", headers=bearer(unknown), json={"name": "TST03"})
    ).status_code == 401


async def test_service_token_is_refused_outside_its_scopes(client, sessions):
    owner = await seed_owner(sessions)
    _, token = await make_token(sessions, owner)
    real_decode = jwt.decode
    decode = MagicMock(side_effect=real_decode)

    with patch("jwt.decode", new=decode):
        created = await client.post(
            f"{API}/projects/", headers=bearer(token), json={"name": "TST01"}
        )
        assert created.status_code == 201, created.text
        project_id = created.json()["id"]
        dataset_id = (await upload(client, token, project_id)).json()["dataset_id"]

        refused = {
            "delete the project": await client.delete(
                f"{API}/projects/{project_id}", headers=bearer(token)
            ),
            "list the users (admin)": await client.get(f"{API}/admin/users", headers=bearer(token)),
            "list the projects": await client.get(f"{API}/projects/", headers=bearer(token)),
            "read the account": await client.get(f"{API}/users/me", headers=bearer(token)),
            "delete the dataset": await client.delete(
                f"{API}/datasets/{dataset_id}", headers=bearer(token)
            ),
            "the LaTeX report": await client.post(
                f"{API}/datasets/{dataset_id}/report",
                headers=bearer(token),
                params={"comparison_name": COMPARISON, "template": "latex"},
            ),
            "the report, format left out": await client.post(
                f"{API}/datasets/{dataset_id}/report",
                headers=bearer(token),
                params={"comparison_name": COMPARISON},
            ),
            "the status of any format": await client.get(
                f"{API}/datasets/{dataset_id}/report/status",
                headers=bearer(token),
                params={"comparison_name": COMPARISON},
            ),
            "the download of the LaTeX report": await client.get(
                f"{API}/datasets/{dataset_id}/report/download",
                headers=bearer(token),
                params={"comparison_name": COMPARISON, "template": "latex"},
            ),
            "the analysis report": await client.get(
                f"{API}/analyses/{uuid4()}/report/status", headers=bearer(token)
            ),
        }
        for what, resp in refused.items():
            assert resp.status_code == 403, f"{what}: {resp.status_code} {resp.text}"
            assert token not in resp.text

    async with sessions() as s:
        assert await s.get(Project, UUID(project_id)) is not None  # still there
    # A service token is never parsed as a JWT (auth dependencies and middlewares alike).
    decoded = [str(call.args[0]) for call in decode.call_args_list if call.args]
    assert not [value for value in decoded if value.startswith("glst_")]

    from app.middleware.rate_limit import get_user_identifier

    request = SimpleNamespace(headers=bearer(token), client=SimpleNamespace(host="127.0.0.1"))
    with patch("jwt.decode", new=decode):
        decode.reset_mock()
        assert get_user_identifier(request) == "ip:127.0.0.1"
        decode.assert_not_called()

    # A token without `reports:editorial` cannot ask for the editorial report.
    _, limited = await make_token(
        sessions, owner, scopes=("projects:create", "datasets:upload"), name="no-report"
    )
    resp = await client.post(
        f"{API}/datasets/{dataset_id}/report", headers=bearer(limited), params=EDITORIAL
    )
    assert resp.status_code == 403, resp.text
    assert "reports:editorial" in resp.text
    assert await jobs_of(sessions, UUID(dataset_id)) == []


async def test_service_token_inherits_the_owner_gates(client, sessions):
    from app.api.deps.license import require_active_license
    from app.core.license import LicenseStatus
    from app.main import app

    # Licence absent: the application refuses the token like any caller.
    owner = await seed_owner(sessions)
    _, token = await make_token(sessions, owner)
    app.dependency_overrides.pop(require_active_license)
    with patch(
        "app.api.deps.license.get_license_status",
        return_value=LicenseStatus(valid=False, error="no license configured"),
    ):
        resp = await client.post(f"{API}/projects/", headers=bearer(token), json={"name": "TST01"})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["error"] == "license_expired"
    app.dependency_overrides[require_active_license] = lambda: None

    # Owner without the cosmetics module: no editorial report.
    plain = await seed_owner(sessions, cosmetics_module_enabled=False)
    _, plain_token = await make_token(sessions, plain)
    dataset_id = await seed_dataset(sessions, plain)
    resp = await client.post(
        f"{API}/datasets/{dataset_id}/report", headers=bearer(plain_token), params=EDITORIAL
    )
    assert resp.status_code == 403, resp.text
    assert "cosmetics module" in resp.text

    # Owner on the STARTER plan: no PDF report at all.
    starter = await seed_owner(sessions, subscription_plan=SubscriptionPlan.STARTER)
    _, starter_token = await make_token(sessions, starter)
    dataset_id = await seed_dataset(sessions, starter)
    resp = await client.post(
        f"{API}/datasets/{dataset_id}/report", headers=bearer(starter_token), params=EDITORIAL
    )
    assert resp.status_code == 403, resp.text
    assert "TEAM or ON_PREMISE" in resp.text

    # Owner whose DEG quota is spent: the deposit is refused.
    spent = await seed_owner(
        sessions,
        analyses_quota_override=1,
        analyses_used_this_month=1,
        quota_reset_at=datetime.now(timezone.utc),
    )
    _, spent_token = await make_token(sessions, spent)
    project = await client.post(f"{API}/projects/", headers=bearer(spent_token), json={"name": "Q"})
    assert project.status_code == 201, project.text
    resp = await upload(client, spent_token, project.json()["id"])
    assert resp.status_code == 429, resp.text

    # An account that is not ACTIVE gets no token from the command...
    for inactive in (UserStatus.PENDING, UserStatus.SUSPENDED):
        account = await seed_owner(sessions, status=inactive)
        code, printed = await run_command(
            sessions, "create", "--owner", account.email, "--name", "pipeline"
        )
        assert code == 1 and "not active" in printed, printed
        assert not TOKEN_RE.search(printed)
    async with sessions() as s:
        issued = await s.scalar(text("SELECT count(*) FROM service_tokens"))

    # ...and a token whose owner is no longer ACTIVE is refused on every scoped route, before
    # any project, upload, quota unit or report job. A pending invitation stays pending: the
    # token never activates it, and never changes the stored role.
    for inactive in (UserStatus.PENDING, UserStatus.SUSPENDED):
        account = await seed_owner(sessions, role=UserRole.ANALYST)
        _, account_token = await make_token(sessions, account)
        dataset_id = await seed_dataset(sessions, account)
        async with sessions() as s:
            (await s.get(User, account.id)).status = inactive
            await s.commit()
            project_id = (await s.get(Dataset, dataset_id)).project_id
        report = f"{API}/datasets/{dataset_id}/report"
        auth = bearer(account_token)
        calls = [
            ("create a project", lambda: client.post(f"{API}/projects/", headers=auth,
                                                     json={"name": "TST09"})),
            ("upload", lambda: upload(client, account_token, str(project_id))),
            ("poll the dataset", lambda: client.get(f"{API}/datasets/{dataset_id}", headers=auth)),
            ("ask for the report", lambda: client.post(report, headers=auth, params=EDITORIAL,
                                                       json={"use_ai": False})),
            ("poll the report", lambda: client.get(f"{report}/status", headers=auth,
                                                   params=EDITORIAL)),
            ("download the report", lambda: client.get(f"{report}/download", headers=auth,
                                                       params=EDITORIAL)),
        ]
        for what, call in calls:
            resp = await call()
            assert resp.status_code == 403, f"{inactive.value}, {what}: {resp.text}"
            detail = resp.json()["detail"]
            assert detail["error"] == "account_inactive", detail
            assert detail["status"] == inactive.value
            assert account_token not in resp.text
        async with sessions() as s:
            stored = await s.get(User, account.id)
            assert stored.status == inactive and stored.role == UserRole.ANALYST
            assert stored.analyses_used_this_month == 0
            owned = await s.scalar(
                select(func.count()).select_from(Project).where(Project.owner_id == account.id)
            )
            assert owned == 1  # the seeded project only
        assert await jobs_of(sessions, dataset_id) == []
    async with sessions() as s:
        assert await s.scalar(text("SELECT count(*) FROM service_tokens")) == issued + 2


# ── The editorial report: use_ai and branding ────────────────────────────────


async def test_editorial_without_ai_calls_no_llm(tmp_path):
    from app.services.report_editorial_content import build_editorial_context
    from app.services.report_editorial_service import render_html, report_editorial_service
    from app.services.report_latex_service import report_latex_service
    from tests.test_report_editorial import ENGLISH_AI, FIXTURE, _inputs

    nova = json.loads(FIXTURE.read_text())
    seen: list[str] = []
    collect_calls: list[dict] = []

    async def fake_collect(db, dataset_id, comparison_name, work_dir, generate_ai=False,
                           make_figures=True, include_ai=True):
        collect_calls.append({"include_ai": include_ai, "generate_ai": generate_ai})
        comp = {"test": "AL-7", "control": "Vehicle", "degs": [], "n_up": 0, "n_down": 0,
                "enr_up": [], "enr_down": []}
        if include_ai:
            comp["ai_interpretation"] = ENGLISH_AI  # what the cached read would return
        return {"comparisons": [comp], "expression": {}, "qc_rows": [], "params": {},
                "project": SimpleNamespace(name="Fictitious study"), "analysis": None}

    def recording_db():
        db = MagicMock()

        async def _scalar(stmt):
            seen.append(str(stmt))
            return None

        async def _execute(stmt):
            seen.append(str(stmt))
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            return result

        db.scalar = AsyncMock(side_effect=_scalar)
        db.execute = AsyncMock(side_effect=_execute)
        db.get = AsyncMock(return_value=SimpleNamespace(id=uuid4(), project_id=uuid4()))
        return db

    comparison_text = AsyncMock(return_value=("generated", "model"))
    interpreter = MagicMock()
    with (
        patch.object(report_latex_service, "collect_comparison_data", new=fake_collect),
        patch.object(report_latex_service, "_load_branding", new=AsyncMock(return_value=None)),
        patch(
            "app.services.report_editorial_service.compute_nova",
            new=AsyncMock(return_value=(nova, [])),
        ),
        patch("app.services.ai_interpreter.generate_comparison_text", new=comparison_text),
        patch("app.services.ai_interpreter.LocalAIInterpreter", new=interpreter),
    ):
        # Even when generation is asked for, use_ai=False wins.
        inputs = await report_editorial_service.collect(
            recording_db(), uuid4(), "AL-7_vs_Vehicle", str(tmp_path),
            requested_by=uuid4(), generate_ai=True, use_ai=False,
        )
        comparison_text.assert_not_called()
        interpreter.assert_not_called()
        assert collect_calls == [{"include_ai": False, "generate_ai": False}]
        assert not [s for s in seen if "ai_interpretations" in s or "cosmetic_interpretations" in s]
        assert inputs.use_ai is False
        assert inputs.comparison_ai is None and inputs.nova_ai is None

        # The probe works: with use_ai the cached nova reading is read.
        seen.clear()
        await report_editorial_service.collect(
            recording_db(), uuid4(), "AL-7_vs_Vehicle", str(tmp_path), use_ai=True,
        )
        assert [s for s in seen if "cosmetic_interpretations" in s]

    # The LaTeX collector skips the cached interpretation when asked to.
    for include_ai in (False, True):
        seen.clear()
        db = recording_db()
        empty = MagicMock()
        empty.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(side_effect=lambda stmt: seen.append(str(stmt)) or empty)
        await report_latex_service._build_comparison(
            db, SimpleNamespace(id=uuid4(), dataset_metadata={}, name="DEG"), [], {},
            str(tmp_path), False, comparison_name="AL-7_vs_Vehicle", make_figures=False,
            include_ai=include_ai,
        )
        read = [s for s in seen if "ai_interpretations" in s]
        assert bool(read) is include_ai

    # No AI text and no AI label in the rendered report, even if a text were handed over.
    page = render_html(build_editorial_context(_inputs(use_ai=False)))
    assert re.search(r"\bAI\b", page) is None, re.findall(r".{60}\bAI\b.{60}", page)[:3]
    assert "AI-generated" not in page and "interpretation model" not in page.lower()
    assert "cornified-envelope genes" not in page and "<b>barrier</b>" not in page
    assert "Top genes across conditions" in page  # the heatmap page stays


async def test_editorial_service_leaves_out_the_owner_branding(tmp_path):
    from app.services.report_editorial_content import build_editorial_context
    from app.services.report_editorial_service import render_html, report_editorial_service
    from app.services.report_latex_service import report_latex_service
    from tests.test_report_editorial import FIXTURE

    nova = json.loads(FIXTURE.read_text())
    saved = {
        "institute_name": "Fictitious Skin Institute",
        "institute_address": "1 Example Street",
        "cover_info": {"prepared_by": "A. Tester"},
        "logo_file": None,
    }

    async def fake_collect(*_args, **_kwargs):
        comp = {"test": "AL-7", "control": "Vehicle", "degs": [], "n_up": 0, "n_down": 0,
                "enr_up": [], "enr_down": []}
        return {"comparisons": [comp], "expression": {}, "qc_rows": [], "params": {},
                "project": SimpleNamespace(name="Fictitious study"), "analysis": None}

    db = MagicMock()
    db.scalar = AsyncMock(return_value=None)
    empty = MagicMock()
    empty.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=empty)
    db.get = AsyncMock(return_value=SimpleNamespace(id=uuid4(), project_id=uuid4()))
    load_branding = AsyncMock(return_value=saved)
    with (
        patch.object(report_latex_service, "collect_comparison_data", new=fake_collect),
        patch.object(report_latex_service, "_load_branding", new=load_branding),
        patch(
            "app.services.report_editorial_service.compute_nova",
            new=AsyncMock(return_value=(nova, [])),
        ),
    ):
        plain = await report_editorial_service.collect(
            db, uuid4(), "AL-7_vs_Vehicle", str(tmp_path), requested_by=uuid4(),
            use_ai=False, use_branding=False,
        )
        load_branding.assert_not_called()
        assert plain.branding is None and "prepared_by" not in plain.cover_info

        # Control: by default the requester's saved branding is loaded and printed.
        branded = await report_editorial_service.collect(
            db, uuid4(), "AL-7_vs_Vehicle", str(tmp_path), requested_by=uuid4(), use_ai=False,
        )
        load_branding.assert_awaited_once()
        assert branded.branding == saved

    assert "Fictitious Skin Institute" not in render_html(build_editorial_context(plain))
    assert "Fictitious Skin Institute" in render_html(build_editorial_context(branded))


async def test_use_ai_is_read_without_the_customization_module(client, sessions):
    owner = await seed_owner(sessions, report_customization_module_enabled=False)
    _, token = await make_token(sessions, owner)
    dataset_id = await seed_dataset(sessions, owner)

    resp = await client.post(
        f"{API}/datasets/{dataset_id}/report",
        headers=bearer(token),
        params=EDITORIAL,
        json={"use_ai": False},
    )
    assert resp.status_code == 202, resp.text
    (job,) = await jobs_of(sessions, dataset_id)
    assert job.use_ai is False and job.template == "editorial"


async def test_a_flip_of_use_ai_never_reuses_the_other_job(client, sessions):
    owner = await seed_owner(sessions)
    _, token = await make_token(sessions, owner)
    dataset_id = await seed_dataset(sessions, owner)

    async def ask(use_ai):
        resp = await client.post(
            f"{API}/datasets/{dataset_id}/report",
            headers=bearer(token),
            params=EDITORIAL,
            json={"use_ai": use_ai},
        )
        assert resp.status_code == 202, resp.text
        return resp.json()["job_id"]

    with_ai = await ask(True)
    without_ai = await ask(False)
    assert without_ai != with_ai
    assert await ask(False) == without_ai  # an in-progress job of the same setting is reused
    assert await ask(True) == with_ai
    jobs = {str(j.id): j.use_ai for j in await jobs_of(sessions, dataset_id)}
    assert jobs == {with_ai: True, without_ai: False}


async def test_service_token_report_ignores_the_owner_branding(client, sessions):
    from app.worker.tasks import report_task

    owner = await seed_owner(sessions, report_customization_module_enabled=True)
    token_id, token = await make_token(sessions, owner)
    dataset_id = await seed_dataset(sessions, owner)

    resp = await client.post(
        f"{API}/datasets/{dataset_id}/report",
        headers=bearer(token),
        params=EDITORIAL,
        json={"use_ai": False, "conclusion": "x", "cover_info": {"prepared_by": "A. Tester"}},
    )
    assert resp.status_code == 202, resp.text
    (job,) = await jobs_of(sessions, dataset_id)
    assert job.service_token_id == token_id
    assert job.cover_info is None and job.conclusion is None

    # The worker renders it without the owner's saved branding, and without AI.
    render = AsyncMock(return_value=b"%PDF-1.7")
    with (
        patch(
            "app.services.report_editorial_service.report_editorial_service.render_comparison_pdf",
            new=render,
        ),
        patch(
            "app.services.report_latex_service.report_latex_service.is_demo_requester",
            new=AsyncMock(return_value=False),
        ),
    ):
        await report_task._render_editorial(MagicMock(), job)
        human = SimpleNamespace(**{**job.__dict__, "service_token_id": None, "use_ai": True})
        await report_task._render_editorial(MagicMock(), human)
    assert render.await_args_list[0].kwargs["use_branding"] is False
    assert render.await_args_list[0].kwargs["use_ai"] is False
    assert render.await_args_list[1].kwargs["use_branding"] is True
    assert render.await_args_list[1].kwargs["use_ai"] is True

    # A later job of the owner, branded, is never what the token polls or downloads.
    async with sessions() as s:
        mine = await s.get(ReportJob, job.id)
        mine.status, mine.pdf_path = ReportJobStatus.DONE, "token.pdf"
        s.add(
            ReportJob(
                id=uuid4(), project_id=mine.project_id, dataset_id=dataset_id,
                comparison_name=COMPARISON, template="editorial", status=ReportJobStatus.DONE,
                pdf_path="owner.pdf", requested_by=owner.id,
                created_at=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
        )
        await s.commit()
    status = await client.get(
        f"{API}/datasets/{dataset_id}/report/status", headers=bearer(token), params=EDITORIAL
    )
    assert status.status_code == 200, status.text
    assert status.json()["id"] == str(job.id) and status.json()["use_ai"] is False
    with patch(
        "app.services.storage.storage_service.download_file",
        new=AsyncMock(side_effect=lambda path: path.encode()),
    ):
        pdf = await client.get(
            f"{API}/datasets/{dataset_id}/report/download", headers=bearer(token), params=EDITORIAL
        )
    assert pdf.status_code == 200, pdf.text
    assert pdf.content == b"token.pdf"


# ── JWT users ─────────────────────────────────────────────────────────────────


async def test_jwt_users_are_unchanged(client, sessions):
    from app.core.config import settings

    user = await seed_owner(sessions)
    login = jwt.encode(
        {
            "sub": str(user.id),
            "email": user.email,
            "aud": "authenticated",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        settings.SUPABASE_JWT_SECRET,
        algorithm="HS256",
    )
    with patch(
        "app.core.supabase_auth.get_cached_user_role", new=AsyncMock(return_value="user")
    ):
        created = await client.post(f"{API}/projects/", headers=bearer(login), json={"name": "J"})
        assert created.status_code == 201, created.text
        listed = await client.get(f"{API}/projects/", headers=bearer(login))
        assert listed.status_code == 200, listed.text

        dataset_id = await seed_dataset(sessions, user)
        # No body: the report keeps its AI, as before.
        resp = await client.post(
            f"{API}/datasets/{dataset_id}/report", headers=bearer(login), params=EDITORIAL
        )
        assert resp.status_code == 202, resp.text
        (job,) = await jobs_of(sessions, dataset_id)
        assert job.use_ai is True and job.service_token_id is None
        async with sessions() as s:  # done (SQLite reads back naive times: skip the stale check)
            (await s.get(ReportJob, job.id)).status = ReportJobStatus.DONE
            await s.commit()
        # The LaTeX format and the status of any format stay open to a user.
        latex = await client.get(
            f"{API}/datasets/{dataset_id}/report/status",
            headers=bearer(login),
            params={"comparison_name": COMPARISON},
        )
        assert latex.status_code == 200, latex.text

        forged = jwt.encode(
            {"sub": str(user.id), "email": user.email, "aud": "authenticated"},
            "not-the-secret",
            algorithm="HS256",
        )
        assert (
            await client.get(f"{API}/projects/", headers=bearer(forged))
        ).status_code == 401
