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

import pytest

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


# --------------------------------------------------------------------------------------------
# The AUTH_* variables of the self-hosted authentication (docs/AUTH.md)
# --------------------------------------------------------------------------------------------

AUTH_VARIABLES = {
    "AUTH_MODE",
    "AUTH_LOCAL_ENABLED",
    "AUTH_ISSUER",
    "AUTH_JWT_PRIVATE_KEYS",
    "AUTH_ENCRYPTION_KEYS",
    "AUTH_COOKIE_DOMAIN",
    "AUTH_COOKIE_SECURE",
}
# Services whose process loads the application settings: all of them pass the SUPABASE_* ones.
# The r-worker signs nobody in and holds no key: it only needs to agree on the mode.
KEY_HOLDERS = ("api", "worker", "beat")
MODE_ONLY = ("r-worker",)


def _environment(service: str) -> dict[str, str]:
    """The `- NAME=value` entries of a service's `environment:` list, without a YAML dependency."""
    values: dict[str, str] = {}
    current = None
    in_environment = False
    for line in COMPOSE.read_text().splitlines():
        top = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if top:
            current, in_environment = top.group(1), False
            continue
        if current != service:
            continue
        if re.match(r"^    environment:\s*$", line):
            in_environment = True
            continue
        if in_environment and re.match(r"^    \S", line):
            in_environment = False
        entry = re.match(r"^      - ([A-Z][A-Z0-9_]*)=(.*)$", line)
        if in_environment and entry:
            values[entry.group(1)] = entry.group(2)
    return values


def test_compose_environment_is_parsed():
    """Guard against a vacuous pass if the compose layout changes."""
    assert _environment("api")["ENVIRONMENT"] == "production"


@pytest.mark.parametrize("service", KEY_HOLDERS)
def test_the_auth_settings_reach_the_services_that_hold_keys(service):
    assert AUTH_VARIABLES <= set(_environment(service))


@pytest.mark.parametrize("service", MODE_ONLY)
def test_the_services_that_only_load_the_settings_get_the_mode(service):
    environment = _environment(service)
    assert {"AUTH_MODE", "AUTH_LOCAL_ENABLED"} <= set(environment)
    assert not {"AUTH_JWT_PRIVATE_KEYS", "AUTH_ENCRYPTION_KEYS"} & set(environment)


@pytest.mark.parametrize("service", KEY_HOLDERS + MODE_ONLY)
def test_every_auth_variable_has_a_default_so_a_missing_one_cannot_break_boot(service):
    """`${NAME}` with NAME unset reaches the container as an empty string, which overrides the
    default of the code (an empty AUTH_MODE is not a valid mode, an empty issuer signs tokens for
    nobody). Each entry must be `${NAME:-default}`."""
    for name, value in _environment(service).items():
        if name.startswith("AUTH_"):
            assert re.fullmatch(r"\$\{%s:-[^}]*\}" % name, value), f"{service}: {name}={value}"


@pytest.mark.parametrize("service", KEY_HOLDERS + MODE_ONLY)
def test_the_compose_defaults_keep_production_on_the_hosted_provider(service):
    from app.core.config import Settings

    environment = _environment(service)
    assert environment["AUTH_MODE"] == "${AUTH_MODE:-supabase}"
    assert environment["AUTH_LOCAL_ENABLED"] == "${AUTH_LOCAL_ENABLED:-false}"
    assert Settings.model_fields["AUTH_MODE"].default == "supabase"
    assert Settings.model_fields["AUTH_LOCAL_ENABLED"].default is False


@pytest.mark.parametrize("service", KEY_HOLDERS)
def test_the_other_compose_defaults_are_the_defaults_of_the_code(service):
    from app.core.config import Settings

    environment = _environment(service)
    assert environment["AUTH_COOKIE_SECURE"] == "${AUTH_COOKIE_SECURE:-true}"
    assert Settings.model_fields["AUTH_COOKIE_SECURE"].default is True
    # No key and no cookie domain by default: nothing is read until the local authentication is on.
    assert environment["AUTH_JWT_PRIVATE_KEYS"] == "${AUTH_JWT_PRIVATE_KEYS:-}"
    assert environment["AUTH_ENCRYPTION_KEYS"] == "${AUTH_ENCRYPTION_KEYS:-}"
    assert environment["AUTH_COOKIE_DOMAIN"] == "${AUTH_COOKIE_DOMAIN:-}"
    issuer_default = re.fullmatch(r"\$\{AUTH_ISSUER:-(.+)\}", environment["AUTH_ISSUER"])
    assert issuer_default and issuer_default.group(1).startswith("https://")


def test_every_service_that_gets_the_hosted_provider_also_gets_the_mode():
    for service in _compose_services() - INFRA:
        environment = _environment(service)
        if "SUPABASE_URL" in environment:
            assert "AUTH_MODE" in environment, f"{service} loads settings without the mode"


# --------------------------------------------------------------------------------------------
# APP_URL: the host of every link an e-mail carries
# --------------------------------------------------------------------------------------------

APP_URL_ENTRY = "${APP_URL:-https://app.genolens.com}"


def test_the_services_that_send_e_mail_know_the_address_of_the_app():
    """The compose file has no `env_file`: a setting it does not pass is the code's default, and
    the default of APP_URL is `http://localhost:3000`, so every link of an e-mail (sign-up
    verification, password reset, invitation) pointed to localhost in production."""
    senders = [s for s in sorted(_compose_services() - INFRA) if "SMTP_HOST" in _environment(s)]
    assert {"api", "worker", "beat"} <= set(senders)
    for service in senders:
        assert _environment(service).get("APP_URL") == APP_URL_ENTRY, service


# --------------------------------------------------------------------------------------------
# SUPPORT_EMAIL: the address the security e-mails tell the reader to contact
# --------------------------------------------------------------------------------------------

SUPPORT_EMAIL_DEFAULT = "support@scilicium.com"
SUPPORT_EMAIL_ENTRY = "${SUPPORT_EMAIL:-%s}" % SUPPORT_EMAIL_DEFAULT


def test_the_services_that_send_e_mail_know_the_support_address():
    """Configurable (a real, monitored mailbox before the local authentication opens), with
    today's address as the default in the compose file and in the code alike: production keeps
    sending exactly what it sent."""
    from app.core.config import Settings

    senders = [s for s in sorted(_compose_services() - INFRA) if "SMTP_HOST" in _environment(s)]
    assert {"api", "worker", "beat"} <= set(senders)
    for service in senders:
        assert _environment(service).get("SUPPORT_EMAIL") == SUPPORT_EMAIL_ENTRY, service
    assert Settings.model_fields["SUPPORT_EMAIL"].default == SUPPORT_EMAIL_DEFAULT


def test_the_r_worker_sends_no_e_mail():
    """It runs the R analyses and the reports, neither of which mails anyone: it has no SMTP
    settings, hence no need for APP_URL. If it ever gets them, it needs APP_URL too (above)."""
    assert "SMTP_HOST" not in _environment("r-worker")
