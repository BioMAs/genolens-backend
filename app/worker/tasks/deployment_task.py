"""
Celery task: SSH-based on-premise deployment.

Reproduces the logic of scripts/deploy/deploy-backend.sh via paramiko so that
the admin UI can trigger a full deployment without shell access.

Security contract:
- SSH private keys and env var values are decrypted in memory only.
- Decrypted secrets are NEVER written to DeploymentJob.logs.
- The only network egress is the SSH connection to the target host.
"""
import asyncio
import io
import logging
import time
from datetime import datetime, timezone
from uuid import UUID

import paramiko
from sqlalchemy import select, update

from app.auth.keys import generate_private_key_b64
from app.auth.mfa import generate_encryption_key
from app.worker.celery_app import celery_app
from app.db.session import AsyncSessionLocal
from app.models.models import DeploymentJob, DeploymentStatus, ServerConfig, DeploymentSecret
from app.core.encryption import decrypt, encrypt

logger = logging.getLogger(__name__)


# Every long-running backend service in docker-compose.prod.yml, restarted after a
# build. postgres and redis are left alone: their images do not change on deploy.
# `beat` (the Celery scheduler) was missing, so the daily expiry warnings and the
# monthly quota reset never ran. tests/test_deploy_services.py keeps this list in
# step with the compose file.
RESTARTED_BACKEND_SERVICES = ["api", "worker", "r-worker", "beat"]

# Services that require a Docker build step
SERVICES_WITH_BUILD = {"backend", "license"}

# Expected variables for each service — used to filter secrets
SERVICE_ENV_KEYS = {
    "backend": [
        "POSTGRES_PASSWORD",
        "ADMIN_API_KEY",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LICENSE_DOMAIN",
        "BACKEND_DOMAIN",
        # Own authentication (docs/AUTH.md): no hosted provider, so no SUPABASE_*.
        "AUTH_JWT_PRIVATE_KEYS",
        "AUTH_ENCRYPTION_KEYS",
        "AUTH_COOKIE_DOMAIN",
    ],
    "license": ["ADMIN_API_KEY", "LICENSE_DOMAIN"],
}

# The two secrets of an instance's own authentication, generated for each server the first time its
# backend is deployed and stored with the server's other secrets from then on (see
# `ensure_auth_secrets`). They are never regenerated: a new signing key signs every user out, and a
# new encryption key makes every stored second-factor secret unreadable.
GENERATED_AUTH_SECRETS = ("AUTH_JWT_PRIVATE_KEYS", "AUTH_ENCRYPTION_KEYS")


def _run_ssh_command(channel, command: str, timeout: int = 600) -> tuple[str, int]:
    """Execute a command on an already-open SSH transport channel."""
    channel.exec_command(command)
    stdout_data, stderr_data = b"", b""
    deadline = time.time() + timeout

    while True:
        if channel.recv_ready():
            stdout_data += channel.recv(4096)
        if channel.recv_stderr_ready():
            stderr_data += channel.recv_stderr(4096)
        if channel.exit_status_ready():
            # Drain remaining output
            while channel.recv_ready():
                stdout_data += channel.recv(4096)
            while channel.recv_stderr_ready():
                stderr_data += channel.recv_stderr(4096)
            break
        if time.time() > deadline:
            channel.close()
            return "TIMEOUT", -1
        time.sleep(0.5)

    exit_code = channel.recv_exit_status()
    output = (stdout_data + stderr_data).decode("utf-8", errors="replace").strip()
    return output, exit_code


def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="app.worker.tasks.deployment_task.run_deployment")
def run_deployment(self, job_id: str) -> dict:
    """
    Execute an on-premise deployment over SSH.

    Steps per service:
      backend/license: git pull → write .env → docker compose build → alembic migrate → rolling restart → health check
      ai:              git pull → docker compose pull → restart

    Args:
        job_id: UUID of the DeploymentJob row to execute.
    """
    async def _deploy():
        async with AsyncSessionLocal() as db:
            # ------------------------------------------------------------------
            # Load job + server
            # ------------------------------------------------------------------
            job_result = await db.execute(select(DeploymentJob).where(DeploymentJob.id == job_id))
            job = job_result.scalar_one_or_none()
            if not job:
                logger.error("DeploymentJob %s not found", job_id)
                return {"status": "error", "detail": "job not found"}

            server_result = await db.execute(select(ServerConfig).where(ServerConfig.id == job.server_id))
            server = server_result.scalar_one_or_none()
            if not server:
                await _fail(db, job, "Server configuration not found")
                return {"status": "error"}

            secrets_result = await db.execute(
                select(DeploymentSecret).where(DeploymentSecret.server_id == server.id)
            )
            secret_rows = secrets_result.scalars().all()
            # Decrypt all secrets into a plain dict (memory-only, never logged)
            env_vars: dict[str, str] = {}
            for row in secret_rows:
                try:
                    env_vars[row.key] = decrypt(row.value_encrypted)
                except Exception:
                    await _fail(db, job, f"Failed to decrypt secret '{row.key}'")
                    return {"status": "error"}

            # The instance's own authentication keys: generated the first time, kept ever after.
            # Stored (in a session of their own, so a failure cannot upset this one) BEFORE the
            # .env is written, so that no .env ever holds a key the next deploy cannot find.
            generated_keys: list[str] = []
            if "backend" in (job.services or []):
                try:
                    async with AsyncSessionLocal() as secrets_db:
                        generated_keys = await ensure_auth_secrets(secrets_db, server.id, env_vars)
                except Exception:
                    await _fail(db, job, "Failed to store the generated authentication keys")
                    return {"status": "error"}

            # Mark running
            await db.execute(
                update(DeploymentJob)
                .where(DeploymentJob.id == job_id)
                .values(status=DeploymentStatus.RUNNING, started_at=datetime.now(timezone.utc), logs="")
            )
            await db.commit()

            # ------------------------------------------------------------------
            # Open SSH connection
            # ------------------------------------------------------------------
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            try:
                connect_kwargs: dict = dict(
                    hostname=server.host,
                    port=server.ssh_port,
                    username=server.ssh_user,
                    timeout=30,
                )
                if server.ssh_key_encrypted:
                    pem = decrypt(server.ssh_key_encrypted)
                    pkey = paramiko.RSAKey.from_private_key(io.StringIO(pem))
                    connect_kwargs["pkey"] = pkey
                else:
                    connect_kwargs["look_for_keys"] = True

                ssh.connect(**connect_kwargs)
            except Exception as exc:
                await _fail(db, job, f"SSH connection failed: {exc}")
                return {"status": "error"}

            # ------------------------------------------------------------------
            # Helper: run a command and append output to logs
            # ------------------------------------------------------------------
            async def exec_step(label: str, cmd: str, allow_fail: bool = False) -> bool:
                await _append_log(db, job_id, f"\n--> {label}")
                transport = ssh.get_transport()
                channel = transport.open_session()
                output, code = _run_ssh_command(channel, cmd)
                channel.close()
                if output:
                    await _append_log(db, job_id, output)
                if code != 0 and not allow_fail:
                    await _fail(db, job, f"Step '{label}' failed (exit {code})")
                    return False
                return True

            # ------------------------------------------------------------------
            # Deploy each requested service
            # ------------------------------------------------------------------
            repo = server.repo_path
            services = job.services or []

            try:
                # 1. Git pull (always — main repo)
                ok = await exec_step("git pull", f"cd {repo} && git pull origin main")
                if not ok:
                    ssh.close()
                    return {"status": "error"}

                # 2. Per-service steps
                for svc in services:
                    await _append_log(db, job_id, f"\n==> Deploying service: {svc}")

                    if svc == "backend":
                        backend_dir = f"{repo}/backend"
                        if generated_keys:
                            await _append_log(
                                db, job_id,
                                f"--> generated for this server and stored with its secrets: "
                                f"{', '.join(generated_keys)} (values not logged)",
                            )
                        env_content = _build_backend_env(env_vars, server)
                        sftp = ssh.open_sftp()
                        _sftp_write(sftp, f"{backend_dir}/.env", env_content)
                        sftp.close()
                        await _append_log(db, job_id, "--> .env written (secrets not logged)")

                        if not job.skip_build:
                            ok = await exec_step(
                                "docker compose build",
                                f"cd {backend_dir} && docker compose -f docker-compose.prod.yml build"
                            )
                            if not ok:
                                ssh.close()
                                return {"status": "error"}

                        ok = await exec_step(
                            "alembic upgrade head",
                            f"cd {backend_dir} && docker compose -f docker-compose.prod.yml run --rm api alembic upgrade head"
                        )
                        if not ok:
                            ssh.close()
                            return {"status": "error"}

                        for container in RESTARTED_BACKEND_SERVICES:
                            ok = await exec_step(
                                f"restart {container}",
                                f"cd {backend_dir} && docker compose -f docker-compose.prod.yml up -d --no-deps {container}"
                            )
                            if not ok:
                                ssh.close()
                                return {"status": "error"}
                            await asyncio.sleep(10)

                    elif svc == "license":
                        license_dir = f"{repo}/license-service"
                        if not job.skip_build:
                            ok = await exec_step(
                                "docker compose build (license)",
                                f"cd {license_dir} && docker compose build"
                            )
                            if not ok:
                                ssh.close()
                                return {"status": "error"}

                        ok = await exec_step(
                            "restart license",
                            f"cd {license_dir} && docker compose up -d"
                        )
                        if not ok:
                            ssh.close()
                            return {"status": "error"}

                    elif svc == "git_pull":
                        # Update submodules to their latest remote commits.
                        # The main repo git pull already ran above; this step
                        # propagates the update to backend/frontend/admin-front submodules.
                        ok = await exec_step(
                            "git submodule update",
                            f"cd {repo} && git submodule update --remote --merge"
                        )
                        if not ok:
                            ssh.close()
                            return {"status": "error"}

                # 3. Health check
                if server.health_url:
                    ok = await exec_step(
                        "health check",
                        f"curl --fail --silent --show-error --max-time 15 '{server.health_url}'"
                    )
                    if not ok:
                        ssh.close()
                        return {"status": "error"}

                ssh.close()
                await db.execute(
                    update(DeploymentJob)
                    .where(DeploymentJob.id == job_id)
                    .values(status=DeploymentStatus.SUCCESS, finished_at=datetime.now(timezone.utc))
                )
                await _append_log(db, job_id, "\n==> Deployment completed successfully.")
                await db.commit()
                return {"status": "success"}

            except Exception as exc:
                ssh.close()
                await _fail(db, job, f"Unexpected error: {exc}")
                return {"status": "error"}

    return run_async(_deploy())


async def _append_log(db, job_id: str, line: str):
    """Append a line to the job's logs column."""
    from sqlalchemy import text
    await db.execute(
        text("UPDATE deployment_jobs SET logs = COALESCE(logs, '') || :line WHERE id = :id"),
        {"line": line + "\n", "id": str(job_id)},
    )
    await db.commit()


async def _fail(db, job: DeploymentJob, reason: str):
    await db.execute(
        update(DeploymentJob)
        .where(DeploymentJob.id == job.id)
        .values(
            status=DeploymentStatus.FAILED,
            finished_at=datetime.now(timezone.utc),
            logs=(job.logs or "") + f"\n[ERROR] {reason}\n",
        )
    )
    await db.commit()
    logger.error("Deployment job %s failed: %s", job.id, reason)


def _missing_auth_secrets(env: dict[str, str]) -> dict[str, str]:
    """Freshly generated values for the authentication secrets `env` does not hold.

    A secret `env` already holds (given by the operator, or kept from an earlier deploy) is never
    replaced: only the absent ones are generated.
    """
    generators = {
        "AUTH_JWT_PRIVATE_KEYS": generate_private_key_b64,
        "AUTH_ENCRYPTION_KEYS": generate_encryption_key,
    }
    return {name: generators[name]() for name in GENERATED_AUTH_SECRETS if not env.get(name)}


async def ensure_auth_secrets(db, server_id, env: dict[str, str]) -> list[str]:
    """Make sure the server holds its authentication keys; returns the names that were created.

    Whatever the server's secrets already hold stays as it is. A missing key is generated, stored
    encrypted with the server's other secrets (`deployment_secrets`, the same rows the admin UI
    writes), committed, and only then added to `env` (in place): a `.env` is never written with a
    key that the next deploy could not find. Values are never logged; only names are returned.
    """
    generated = _missing_auth_secrets(env)
    if not generated:
        return []
    for name, value in generated.items():
        db.add(DeploymentSecret(server_id=server_id, key=name, value_encrypted=encrypt(value)))
    await db.commit()
    env.update(generated)
    return list(generated)


def _build_backend_env(env: dict[str, str], server) -> str:
    """Build the .env file content for the backend service.

    An on-premise instance has no hosted identity provider: it runs its own authentication
    (`AUTH_MODE=local`, docs/AUTH.md), so the file carries the `AUTH_*` settings, no `SUPABASE_*`.
    `run_deployment` stores the instance's keys before calling this (`ensure_auth_secrets`), so they
    are the same at every deploy. A key still missing here is generated on the spot, which is only
    right for a caller that keeps what it writes: it would otherwise sign every user out.
    """
    env = {**env, **_missing_auth_secrets(env)}
    license_domain = env.get("LICENSE_DOMAIN", "")
    backend_domain = env.get("BACKEND_DOMAIN", "")
    return f"""DATABASE_URL=postgresql+asyncpg://postgres:{env.get('POSTGRES_PASSWORD', '')}@postgres:5432/genolens
POSTGRES_USER=postgres
POSTGRES_PASSWORD={env.get('POSTGRES_PASSWORD', '')}
POSTGRES_DB=genolens
REDIS_URL=redis://redis:6379/0
AUTH_MODE=local
AUTH_LOCAL_ENABLED=true
AUTH_ISSUER=https://{backend_domain}
AUTH_COOKIE_DOMAIN={env.get('AUTH_COOKIE_DOMAIN', '')}
AUTH_JWT_PRIVATE_KEYS={env['AUTH_JWT_PRIVATE_KEYS']}
AUTH_ENCRYPTION_KEYS={env['AUTH_ENCRYPTION_KEYS']}
ENVIRONMENT=production
LOCAL_STORAGE_PATH=/app/data
APP_URL=https://app.genolens.com
LLM_BASE_URL={env.get('LLM_BASE_URL', '')}
LLM_API_KEY={env.get('LLM_API_KEY', '')}
LLM_MODEL={env.get('LLM_MODEL', 'google/gemma-4-E4B-it')}
LICENSE_SERVER_URL=https://{license_domain}
LICENSE_ADMIN_API_KEY={env.get('ADMIN_API_KEY', '')}
CORS_ORIGINS=["https://app.genolens.com","https://admin.genolens.com","https://{backend_domain}","http://localhost:3000"]
"""


def _sftp_write(sftp, remote_path: str, content: str):
    """Write a string to a remote file via SFTP."""
    with sftp.open(remote_path, "w") as f:
        f.write(content)
