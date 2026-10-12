"""
Per-process cache of the Supabase role read by `verify_supabase_token`.

Every authenticated request goes through `verify_supabase_token`, which reads
`profiles.role` from Supabase. It used to open a brand-new httpx client for that
(fresh TLS handshake, 30 s timeout) on every call, so a page firing 20-50 API calls
paid 20-50 round trips, and a slow Supabase could hold each request for 30 s. Not
the cause of the 2026-10-12 pool deadlock, but it widened every request in the
burst that triggered it.

What these tests pin:
- a burst of concurrent requests for one user costs ONE Supabase call;
- a cached role expires after the TTL, so a promotion or demotion lands within it;
- a failed lookup still falls back to USER for that request, and is NOT cached;
- `invalidate_cached_role` (called by the admin endpoints) applies a change at once,
  even against a lookup that was already in flight;
- a request cancelled mid-burst does not cancel the lookup the others wait on;
- one shared client, with a short timeout.

Supabase is replaced by an `httpx.MockTransport`, injected by intercepting
`httpx.AsyncClient` exactly as `test_admin_role_update.py` does.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import jwt
import pytest

from app.core import supabase_auth
from app.core.config import settings
from app.models.models import UserRole

USER_ID = UUID("00000000-0000-0000-0000-0000000000a1")
OTHER_ID = UUID("00000000-0000-0000-0000-0000000000a2")

verify = supabase_auth.verify_supabase_token


def make_token(user_id: UUID = USER_ID) -> str:
    payload = {
        "sub": str(user_id),
        "email": f"{user_id}@example.com",
        "aud": "authenticated",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return jwt.encode(payload, settings.SUPABASE_JWT_SECRET, algorithm="HS256")


class FakeProfiles:
    """Stands in for Supabase's `/rest/v1/profiles`, counting the calls that reach it."""

    def __init__(self):
        self.roles: dict = {USER_ID: "admin", OTHER_ID: "user"}
        self.calls: list = []
        self.clients: list = []  # kwargs of every httpx.AsyncClient built
        self.fail_with = None  # an HTTP status, an exception, or "no-profile-row"
        self._gate: Optional[asyncio.Event] = None

    def hold_next_call(self) -> asyncio.Event:
        """Park the next call after it has read its answer; set the event to release it."""
        self._gate = asyncio.Event()
        return self._gate

    async def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/v1/profiles"
        user_id = UUID(request.url.params["id"].removeprefix("eq."))
        self.calls.append(user_id)

        # The answer is read when the request arrives, as a real database would.
        failure, role = self.fail_with, self.roles.get(user_id)
        gate, self._gate = self._gate, None
        if gate is not None:
            await gate.wait()

        if isinstance(failure, Exception):
            raise failure
        if isinstance(failure, int):
            return httpx.Response(failure, text="upstream error")
        if failure == "no-profile-row" or role is None:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[{"role": role}])


@pytest.fixture
def profiles(monkeypatch):
    fake = FakeProfiles()
    real_client = httpx.AsyncClient

    def build(*args, **kwargs):
        fake.clients.append(dict(kwargs))
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(supabase_auth.httpx, "AsyncClient", build)
    # Fresh per-process state for every test.
    monkeypatch.setattr(supabase_auth, "_role_cache", {}, raising=False)
    monkeypatch.setattr(supabase_auth, "_role_inflight", {}, raising=False)
    monkeypatch.setattr(supabase_auth, "_http_client", None, raising=False)
    return fake


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock the test moves by hand. `now[0]` is the current time."""
    now = [1_000.0]
    monkeypatch.setattr(supabase_auth, "_monotonic", lambda: now[0], raising=False)
    return now


async def _until(condition, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.001)


async def test_a_burst_for_one_user_costs_one_supabase_call(profiles):
    release = profiles.hold_next_call()
    token = make_token()

    burst = [asyncio.create_task(verify(token)) for _ in range(30)]
    await _until(lambda: len(profiles.calls) >= 1)
    await asyncio.sleep(0.02)  # every request of the burst has now reached the lookup
    release.set()
    users = await asyncio.gather(*burst)

    assert len(profiles.calls) == 1
    assert {u.role for u in users} == {UserRole.ADMIN}


async def test_a_cached_role_is_served_until_the_ttl_then_refreshed(profiles, clock):
    ttl = supabase_auth.ROLE_CACHE_TTL_SECONDS
    # The promise made to admins: a promotion or demotion lands within a minute.
    assert 0 < ttl <= 60
    token = make_token()

    assert (await verify(token)).role is UserRole.ADMIN

    profiles.roles[USER_ID] = "user"  # demoted in Supabase, behind the cache's back
    clock[0] += ttl - 1
    assert (await verify(token)).role is UserRole.ADMIN
    assert len(profiles.calls) == 1

    clock[0] += 2
    assert (await verify(token)).role is UserRole.USER
    assert len(profiles.calls) == 2


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(503, id="http-error"),
        pytest.param(httpx.ConnectError("unreachable"), id="network-error"),
        pytest.param(httpx.ReadTimeout("too slow"), id="timeout"),
        pytest.param("no-profile-row", id="no-profile-row"),
    ],
)
async def test_a_failed_lookup_falls_back_to_user_and_is_not_cached(profiles, failure):
    token = make_token()

    profiles.fail_with = failure
    assert (await verify(token)).role is UserRole.USER  # the existing fallback, unchanged

    # Had the failure been cached, this admin would be locked out for a whole TTL.
    profiles.fail_with = None
    assert (await verify(token)).role is UserRole.ADMIN
    assert len(profiles.calls) == 2


async def test_invalidating_a_role_applies_on_the_next_request(profiles):
    token = make_token()
    assert (await verify(token)).role is UserRole.ADMIN

    profiles.roles[USER_ID] = "user"
    supabase_auth.invalidate_cached_role(USER_ID)

    assert (await verify(token)).role is UserRole.USER
    assert len(profiles.calls) == 2


async def test_a_lookup_in_flight_cannot_undo_an_invalidation(profiles):
    """
    The stale answer of a read that started before the admin's change must not be
    cached over it: that would re-grant a revoked role for a full TTL.
    """
    release = profiles.hold_next_call()
    token = make_token()

    before = asyncio.create_task(verify(token))
    await _until(lambda: len(profiles.calls) == 1)  # in flight, answer read: "admin"

    profiles.roles[USER_ID] = "user"  # the admin demotes...
    supabase_auth.invalidate_cached_role(USER_ID)  # ...and the endpoint invalidates

    # Must not join the stale lookup (still parked): joining it would hang, so bound it.
    after = await asyncio.wait_for(verify(token), timeout=2)
    release.set()
    await before

    assert after.role is UserRole.USER
    assert (await verify(token)).role is UserRole.USER
    assert len(profiles.calls) == 2


async def test_a_cancelled_request_does_not_cancel_the_shared_lookup(profiles):
    release = profiles.hold_next_call()
    token = make_token()

    leaver = asyncio.create_task(verify(token))
    stayer = asyncio.create_task(verify(token))
    await _until(lambda: len(profiles.calls) == 1)
    await asyncio.sleep(0.01)  # both are now waiting on the lookup

    leaver.cancel()  # e.g. the browser navigated away mid-burst
    release.set()

    assert (await stayer).role is UserRole.ADMIN
    assert len(profiles.calls) == 1
    with pytest.raises(asyncio.CancelledError):
        await leaver


async def test_lookups_share_one_http_client(profiles):
    assert (await verify(make_token(USER_ID))).role is UserRole.ADMIN
    assert (await verify(make_token(OTHER_ID))).role is UserRole.USER

    assert len(profiles.calls) == 2
    assert len(profiles.clients) == 1


async def test_the_shared_client_gives_up_well_before_thirty_seconds(profiles):
    await verify(make_token())

    timeout = httpx.Timeout(profiles.clients[0]["timeout"])
    # Every phase bounded, so even connect + read stays within 10 s.
    for phase in (timeout.connect, timeout.read, timeout.write, timeout.pool):
        assert phase is not None and phase <= 5


async def test_shutdown_closes_the_shared_client(profiles):
    await verify(make_token())
    client = supabase_auth._http_client
    assert client is not None and not client.is_closed

    await supabase_auth.close_supabase_http_client()

    assert client.is_closed
    # A later lookup (e.g. a script reusing the module) gets a fresh client.
    await verify(make_token(OTHER_ID))
    assert len(profiles.clients) == 2


async def test_the_app_lifespan_closes_the_shared_client_on_shutdown(profiles, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "close_db", AsyncMock())  # no database in unit tests
    await verify(make_token())
    client = supabase_auth._http_client

    async with main.lifespan(main.app):
        assert not client.is_closed

    assert client.is_closed
