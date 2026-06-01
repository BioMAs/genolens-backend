"""
GenoLens Next - FastAPI Application Entry Point
A flexible, data-centric bioinformatics SaaS platform for transcriptomics analysis.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.db.session import close_db
from app.api.endpoints import projects, datasets, admin, admin_deployment, users, ontology, enrichment, bookmarks, genes, comments, history, integrations, license, billing, reports, analyses
from app.middleware import SecurityHeadersMiddleware, limiter, LoginTrackingMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Manages startup and shutdown events.
    """
    # Startup
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"🌍 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🌐 CORS Origins: {settings.CORS_ORIGINS}")
    logger.info(f"📊 Database: Connected")
    logger.info(f"🔄 Celery: Worker configured")

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    await close_db()
    logger.info("✅ Database connections closed")


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

# Security Headers Middleware
app.add_middleware(SecurityHeadersMiddleware)

# Login Tracking Middleware (deduplicated, non-blocking)
app.add_middleware(LoginTrackingMiddleware)

# Rate Limiter State (required by slowapi)
app.state.limiter = limiter


# Exception Handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded. Please try again later.",
            "retry_after": exc.detail
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    # Convert Pydantic errors to JSON-serializable format
    errors = []
    for error in exc.errors():
        error_dict = {
            "loc": list(error.get("loc", [])),
            "msg": str(error.get("msg", "")),
            "type": str(error.get("type", ""))
        }
        # Handle ValueError context if present
        if "ctx" in error:
            error_dict["ctx"] = {k: str(v) for k, v in error["ctx"].items()}
        errors.append(error_dict)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": errors,
            "body": str(exc.body) if exc.body else None
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully."""
    # Always log the full traceback so we can debug production issues
    logger.exception(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )

    response_content = {}
    
    if settings.is_development:
        # In development, return full error details
        response_content = {
            "detail": str(exc),
            "type": type(exc).__name__
        }
    else:
        # In production, return generic error
        response_content = {"detail": "Internal server error"}
    
    # Always include CORS headers on errors
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_content,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
        }
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

app.include_router(
    bookmarks.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    genes.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    comments.router,
    prefix=settings.API_V1_PREFIX,
    tags=["comments"]
)

app.include_router(
    history.router,
    prefix=settings.API_V1_PREFIX,
    tags=["history"]
)

app.include_router(
    integrations.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    license.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    admin_deployment.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    billing.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    reports.router,
    prefix=settings.API_V1_PREFIX,
)

app.include_router(
    analyses.router,
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
