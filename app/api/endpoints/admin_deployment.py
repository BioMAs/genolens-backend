"""
Admin endpoints for on-premise deployment management.

Allows Scilicium admins to:
  - Register target servers (host, SSH key, repo path)
  - Store encrypted deployment secrets per server
  - Trigger deployments (dispatched as Celery tasks)
  - Monitor deployment logs and history

All endpoints require ADMIN or SCILICIUM_ADMIN role.
"""
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.api.deps import get_db, require_admin
from app.core.supabase_auth import SupabaseUser
from app.core.encryption import encrypt, decrypt
from app.models.models import ServerConfig, DeploymentSecret, DeploymentJob, DeploymentStatus
from app.schemas.deployment import (
    ServerConfigCreate, ServerConfigUpdate, ServerConfigResponse,
    SecretUpsert, SecretKeyResponse,
    DeploymentTrigger, DeploymentJobResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin - Deployment"])


# =============================================================================
# Server configuration
# =============================================================================

@router.get("/servers", response_model=list[ServerConfigResponse])
async def list_servers(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all registered deployment target servers."""
    result = await db.execute(select(ServerConfig).order_by(ServerConfig.created_at.desc()))
    servers = result.scalars().all()
    return [_server_to_response(s) for s in servers]


@router.post("/servers", response_model=ServerConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_server(
    payload: ServerConfigCreate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Register a new deployment target server."""
    server = ServerConfig(
        name=payload.name,
        host=payload.host,
        ssh_port=payload.ssh_port,
        ssh_user=payload.ssh_user,
        repo_path=payload.repo_path,
        health_url=payload.health_url,
        notes=payload.notes,
        ssh_key_encrypted=encrypt(payload.ssh_key) if payload.ssh_key else None,
    )
    db.add(server)
    await db.commit()
    await db.refresh(server)
    return _server_to_response(server)


@router.put("/servers/{server_id}", response_model=ServerConfigResponse)
async def update_server(
    server_id: UUID,
    payload: ServerConfigUpdate,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing server configuration."""
    server = await _get_server_or_404(db, server_id)

    if payload.name is not None:
        server.name = payload.name
    if payload.host is not None:
        server.host = payload.host
    if payload.ssh_port is not None:
        server.ssh_port = payload.ssh_port
    if payload.ssh_user is not None:
        server.ssh_user = payload.ssh_user
    if payload.repo_path is not None:
        server.repo_path = payload.repo_path
    if payload.health_url is not None:
        server.health_url = payload.health_url
    if payload.notes is not None:
        server.notes = payload.notes
    if payload.ssh_key is not None:
        server.ssh_key_encrypted = encrypt(payload.ssh_key)

    await db.commit()
    await db.refresh(server)
    return _server_to_response(server)


@router.delete("/servers/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_server(
    server_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a server and all its secrets and deployment history."""
    server = await _get_server_or_404(db, server_id)
    await db.delete(server)
    await db.commit()


# =============================================================================
# Deployment secrets
# =============================================================================

@router.get("/servers/{server_id}/secrets", response_model=list[SecretKeyResponse])
async def list_secret_keys(
    server_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List secret keys for a server (names only — values are never returned)."""
    await _get_server_or_404(db, server_id)
    result = await db.execute(
        select(DeploymentSecret)
        .where(DeploymentSecret.server_id == server_id)
        .order_by(DeploymentSecret.key)
    )
    rows = result.scalars().all()
    return [SecretKeyResponse(key=r.key, updated_at=r.updated_at) for r in rows]


@router.put("/servers/{server_id}/secrets", response_model=list[SecretKeyResponse])
async def upsert_secrets(
    server_id: UUID,
    secrets: list[SecretUpsert],
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Create or update secrets for a server.
    Pass the full list of key/value pairs you want to persist.
    Empty-value entries are skipped (useful for placeholder rows in the UI).
    """
    await _get_server_or_404(db, server_id)

    # Load existing secrets for this server
    result = await db.execute(
        select(DeploymentSecret).where(DeploymentSecret.server_id == server_id)
    )
    existing: dict[str, DeploymentSecret] = {r.key: r for r in result.scalars().all()}

    for item in secrets:
        if not item.value:
            continue  # skip empty values
        if item.key in existing:
            existing[item.key].value_encrypted = encrypt(item.value)
        else:
            db.add(DeploymentSecret(
                server_id=server_id,
                key=item.key,
                value_encrypted=encrypt(item.value),
            ))

    await db.commit()

    result2 = await db.execute(
        select(DeploymentSecret)
        .where(DeploymentSecret.server_id == server_id)
        .order_by(DeploymentSecret.key)
    )
    rows = result2.scalars().all()
    return [SecretKeyResponse(key=r.key, updated_at=r.updated_at) for r in rows]


# =============================================================================
# Deployment jobs
# =============================================================================

@router.post("/deployments/trigger", response_model=DeploymentJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_deployment(
    payload: DeploymentTrigger,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Trigger an on-premise deployment. Returns immediately; follow progress via GET /deployments/{id}."""
    await _get_server_or_404(db, payload.server_id)

    valid_services = {"backend", "license", "git_pull"}
    bad = [s for s in payload.services if s not in valid_services]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown services: {bad}")
    if not payload.services:
        raise HTTPException(status_code=400, detail="At least one service must be selected")

    job = DeploymentJob(
        server_id=payload.server_id,
        services=payload.services,
        skip_build=payload.skip_build,
        status=DeploymentStatus.PENDING,
        triggered_by=str(current_user.user_id),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Dispatch Celery task
    from app.worker.tasks.deployment_task import run_deployment
    run_deployment.delay(str(job.id))

    logger.info("Deployment job %s triggered by user %s for server %s", job.id, current_user.user_id, payload.server_id)
    return await _job_to_response(db, job)


@router.get("/deployments/{job_id}", response_model=DeploymentJobResponse)
async def get_deployment(
    job_id: UUID,
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get status and logs for a specific deployment job."""
    result = await db.execute(select(DeploymentJob).where(DeploymentJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Deployment job not found")
    return await _job_to_response(db, job)


@router.get("/deployments", response_model=list[DeploymentJobResponse])
async def list_deployments(
    current_user: SupabaseUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
):
    """List the most recent deployment jobs."""
    result = await db.execute(
        select(DeploymentJob)
        .order_by(DeploymentJob.created_at.desc())
        .limit(limit)
    )
    jobs = result.scalars().all()
    return [await _job_to_response(db, j) for j in jobs]


# =============================================================================
# Helpers
# =============================================================================

async def _get_server_or_404(db: AsyncSession, server_id: UUID) -> ServerConfig:
    result = await db.execute(select(ServerConfig).where(ServerConfig.id == server_id))
    server = result.scalar_one_or_none()
    if not server:
        raise HTTPException(status_code=404, detail="Server configuration not found")
    return server


def _server_to_response(server: ServerConfig) -> ServerConfigResponse:
    return ServerConfigResponse(
        id=server.id,
        name=server.name,
        host=server.host,
        ssh_port=server.ssh_port,
        ssh_user=server.ssh_user,
        repo_path=server.repo_path,
        health_url=server.health_url,
        notes=server.notes,
        has_ssh_key=bool(server.ssh_key_encrypted),
        created_at=server.created_at,
        updated_at=server.updated_at,
    )


async def _job_to_response(db: AsyncSession, job: DeploymentJob) -> DeploymentJobResponse:
    server_name: Optional[str] = None
    if job.server_id:
        result = await db.execute(select(ServerConfig.name).where(ServerConfig.id == job.server_id))
        server_name = result.scalar_one_or_none()

    return DeploymentJobResponse(
        id=job.id,
        server_id=job.server_id,
        server_name=server_name,
        services=job.services or [],
        skip_build=job.skip_build,
        status=job.status.value if hasattr(job.status, "value") else job.status,
        logs=job.logs,
        triggered_by=job.triggered_by,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
    )
