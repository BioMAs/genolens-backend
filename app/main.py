"""
GenoLens Next - FastAPI Application Entry Point
A flexible, data-centric bioinformatics SaaS platform for transcriptomics analysis.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.db.session import close_db
from app.api.endpoints import projects, datasets, admin, users, ontology, enrichment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Manages startup and shutdown events.
    """
    # Startup
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"🌍 Environment: {settings.ENVIRONMENT}")
    print(f"📊 Database: Connected")
    print(f"🔄 Celery: Worker configured")

    yield

    # Shutdown
    print("🛑 Shutting down...")
    await close_db()
    print("✅ Database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    GenoLens Next - A flexible, data-centric bioinformatics platform.

    ## Features

    - **Bring Your Own Data (BYOD)**: Upload processed transcriptomics data
    - **Hybrid Storage**: Metadata in PostgreSQL, data in Parquet files
    - **Lazy Loading**: Query data on-demand with filters
    - **Background Processing**: Celery-powered async data conversion
    - **Supabase Integration**: Auth and Storage

    ## Architecture

    - FastAPI for high-performance async API
    - SQLAlchemy (async) with Alembic migrations
    - Celery + Redis for background tasks
    - Pandas + PyArrow for data processing
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully."""
    if settings.is_development:
        # In development, return full error details
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    else:
        # In production, return generic error
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )


# Root Endpoint
@app.get("/", tags=["Health"])
async def root():
    """API root endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


@app.get("/db-test", tags=["Health"])
async def database_test():
    """Test database connectivity (development only)."""
    from app.db.session import AsyncSessionLocal
    from sqlalchemy import text

    try:
        async with AsyncSessionLocal() as session:
            # Test simple query
            result = await session.execute(text("SELECT COUNT(*) as count FROM projects"))
            projects_count = result.scalar()

            result = await session.execute(text("SELECT COUNT(*) as count FROM datasets"))
            datasets_count = result.scalar()

            return {
                "status": "database_connected",
                "projects": projects_count,
                "datasets": datasets_count,
                "environment": settings.ENVIRONMENT
            }
    except Exception as e:
        return {
            "status": "database_error",
            "error": str(e),
            "type": type(e).__name__
        }


# Include API Routers
app.include_router(
    projects.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    datasets.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    admin.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    users.router,
    prefix=settings.API_V1_PREFIX + "/users",
    tags=["users"]
)

app.include_router(
    ontology.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    enrichment.router,
    prefix=settings.API_V1_PREFIX,
)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development
    )
