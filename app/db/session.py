"""
Database session management using SQLAlchemy.
"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

# Create Async Engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
)

# Create Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database session.

    The ONLY session provider: every import path (app.api.deps, app.api.deps.db,
    app.api.deps.supabase_deps) re-exports this object. FastAPI caches a
    dependency per callable, so a second copy would give a route two sessions —
    two pool connections held at once — and deadlock the pool under concurrent
    load (see tests/test_single_db_session_per_request.py).

    Commits when the request succeeds, rolls back when it raises.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db() -> None:
    """
    Close database connections.
    """
    await engine.dispose()


