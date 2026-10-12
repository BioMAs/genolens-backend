"""
Tripwire: a request must never open two database sessions.

On 2026-10-12 the production API deadlocked its connection pool. Two distinct
`get_db` functions existed — `app.db.session.get_db` and a copy in
`app.api.deps.supabase_deps`. FastAPI caches a dependency per callable, not per
name, so a route that reached both (every route, once `require_active_account`
pulled in `get_or_create_user` at the router level) held one connection for the
user lookup and then waited for a second one for its own queries. Fifteen such
requests in flight — one page of the DEG table firing its bookmark checks — took
the whole pool (5 + 10 overflow), and every waiter timed out at 30 s with a 500.

This file walks the assembled route table and fails if any route's dependency
tree reaches more than one session provider.
"""

from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute

from app.api import deps as api_deps
from app.api.deps import supabase_deps
from app.db import session as db_session
from app.main import app


def _session_providers(route: APIRoute) -> set:
    """Every distinct `get_db`-named callable reachable from this route."""
    found = set()
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if getattr(dep.call, "__name__", "") == "get_db":
            found.add(dep.call)
        stack.extend(dep.dependencies)
    return found


def test_there_is_a_single_get_db():
    """Every import path must hand out the same function object."""
    assert api_deps.get_db is db_session.get_db
    assert supabase_deps.get_db is db_session.get_db


def test_no_route_opens_two_sessions():
    offenders = sorted(
        f"{sorted(r.methods)} {r.path}"
        for r in app.routes
        if isinstance(r, APIRoute) and len(_session_providers(r)) > 1
    )
    assert not offenders, (
        "These routes resolve two different get_db dependencies, so each request "
        "holds two pool connections at once and the pool can deadlock under load. "
        "Import get_db from app.db.session (or app.api.deps):\n  " + "\n  ".join(offenders)
    )


def test_the_tripwire_can_actually_fail():
    """Guards the guard: two distinct providers on one route must be detected."""

    async def get_db():  # a second, distinct provider with the same name
        yield None

    probe = FastAPI()

    @probe.get("/probe")
    async def _probe(a=Depends(db_session.get_db), b=Depends(get_db)):
        return {}

    route = next(r for r in probe.routes if isinstance(r, APIRoute) and r.path == "/probe")
    assert len(_session_providers(route)) == 2
