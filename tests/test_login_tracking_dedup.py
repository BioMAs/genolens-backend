"""
The login-tracking middleware must not open a DB session per request.

Before 2026-10-12 every authenticated request scheduled `_record_login_event`,
which opened its own session just to find out it had nothing to insert. A page
firing 50 API calls added 50 sessions to a 15-connection pool that was already
saturated. The per-process memo below lets only the first request of each dedup
window reach the database.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import login_tracking
from app.middleware.login_tracking import LoginTrackingMiddleware, _should_record

WINDOW = login_tracking._DEDUP_WINDOW_MINUTES * 60


@pytest.fixture(autouse=True)
def _fresh_memo():
    login_tracking._last_recorded.clear()
    yield
    login_tracking._last_recorded.clear()


def test_should_record_once_per_window():
    assert _should_record("u1", now=1000.0) is True
    assert _should_record("u1", now=1000.0 + 1) is False
    assert _should_record("u1", now=1000.0 + WINDOW - 1) is False
    assert _should_record("u1", now=1000.0 + WINDOW) is True


def test_should_record_is_per_user():
    assert _should_record("u1", now=1000.0) is True
    assert _should_record("u2", now=1000.0) is True


def test_a_burst_of_requests_records_once(monkeypatch):
    recorder = AsyncMock()
    monkeypatch.setattr(login_tracking, "_record_login_event", recorder)
    monkeypatch.setattr(login_tracking, "_extract_user_id", lambda _auth: "user-1")

    app = FastAPI()
    app.add_middleware(LoginTrackingMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    with TestClient(app) as client:
        for _ in range(20):
            assert client.get("/ping", headers={"Authorization": "Bearer x"}).status_code == 200

    assert recorder.await_count == 1
    recorder.assert_awaited_with("user-1")
