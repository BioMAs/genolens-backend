"""
Database dependency — re-exports get_db from app.db.session so that
endpoints can import from a consistent location (app.api.deps.db).
"""
from app.db.session import get_db  # noqa: F401

__all__ = ["get_db"]
