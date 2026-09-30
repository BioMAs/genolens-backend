"""
The admin deploy task must restart every backend service of docker-compose.prod.yml.

`beat` (the Celery scheduler) was added to the production compose file but not to
the lists of services the deploys build and restart, so it was never started: the
daily expiry warnings and the monthly quota reset silently never ran. This keeps
`RESTARTED_BACKEND_SERVICES` in step with the compose file.

The GitHub Actions deploy (.github/workflows/ci.yml) carries the same list, but
`.github/` is excluded from the image these tests run in, so it cannot be checked
here — keep the two in step by hand.
"""
import re
from pathlib import Path

from app.worker.tasks.deployment_task import RESTARTED_BACKEND_SERVICES

COMPOSE = Path(__file__).resolve().parent.parent / "docker-compose.prod.yml"

# Stateful infrastructure: pinned upstream images, never rebuilt or restarted by a deploy.
INFRA = {"postgres", "redis"}


def _compose_services() -> set[str]:
    """Top-level keys under `services:`, without a YAML dependency."""
    services: set[str] = set()
    in_services = False
    for line in COMPOSE.read_text().splitlines():
        if re.match(r"^\S", line):
            in_services = line.startswith("services:")
            continue
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if in_services and match:
            services.add(match.group(1))
    return services


def test_compose_file_is_parsed():
    """Guard against a vacuous pass if the compose layout changes."""
    assert {"api", "worker", "postgres"} <= _compose_services()


def test_deploy_restarts_every_backend_service():
    assert set(RESTARTED_BACKEND_SERVICES) == _compose_services() - INFRA


def test_beat_is_restarted():
    assert "beat" in RESTARTED_BACKEND_SERVICES
