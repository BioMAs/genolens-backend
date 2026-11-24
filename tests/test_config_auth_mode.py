"""
The hosted provider's settings are required only when that provider is the one in use.

`SUPABASE_URL`, `SUPABASE_KEY` and `SUPABASE_JWT_SECRET` used to be required whatever happened.
An instance running its own authentication (`AUTH_MODE=local`, the on-premise template) has no
hosted provider, so it must boot without them; production (`AUTH_MODE=supabase`, the default)
must keep refusing to boot without them, with a message that names what is missing.

The settings are built with `_env_file=None` and the three variables removed from the process
environment, so the result does not depend on the developer's `.env` or shell.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings

HOSTED = ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_JWT_SECRET")


@pytest.fixture(autouse=True)
def _no_hosted_provider_in_the_environment(monkeypatch):
    for name in (*HOSTED, "AUTH_MODE"):
        monkeypatch.delenv(name, raising=False)


def _hosted(**overrides):
    values = {
        "SUPABASE_URL": "https://placeholder.supabase.example.test",
        "SUPABASE_KEY": "anon-placeholder",
        "SUPABASE_JWT_SECRET": "jwt-placeholder",
    }
    values.update(overrides)
    return values


def test_the_default_mode_is_the_hosted_provider():
    assert Settings.model_fields["AUTH_MODE"].default == "supabase"


def test_supabase_mode_with_the_three_settings_boots():
    settings = Settings(_env_file=None, AUTH_MODE="supabase", **_hosted())
    assert settings.SUPABASE_URL == "https://placeholder.supabase.example.test"
    assert settings.SUPABASE_KEY == "anon-placeholder"
    assert settings.SUPABASE_JWT_SECRET == "jwt-placeholder"


def test_the_default_mode_still_requires_them():
    """Production sets nothing but the three variables: the default must keep behaving as before."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    message = str(excinfo.value)
    for name in HOSTED:
        assert name in message


@pytest.mark.parametrize("missing", HOSTED)
def test_supabase_mode_refuses_to_boot_without_one_of_them(missing):
    values = _hosted()
    del values[missing]
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, AUTH_MODE="supabase", **values)
    message = str(excinfo.value)
    assert missing in message
    assert "AUTH_MODE" in message
    # The ones that are set are not blamed.
    for name in HOSTED:
        if name != missing:
            assert name not in message


def test_the_error_lists_every_missing_setting_at_once():
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, AUTH_MODE="supabase", SUPABASE_KEY="anon-placeholder")
    message = str(excinfo.value)
    assert "SUPABASE_URL" in message
    assert "SUPABASE_JWT_SECRET" in message
    assert "SUPABASE_KEY" not in message


def test_local_mode_boots_without_the_hosted_provider():
    settings = Settings(_env_file=None, AUTH_MODE="local")
    assert settings.AUTH_MODE == "local"
    assert settings.SUPABASE_URL is None
    assert settings.SUPABASE_KEY is None
    assert settings.SUPABASE_JWT_SECRET is None


def test_local_mode_accepts_the_empty_values_a_compose_file_passes():
    """`SUPABASE_URL=${SUPABASE_URL}` with the variable unset reaches the container as ''."""
    settings = Settings(
        _env_file=None,
        AUTH_MODE="local",
        SUPABASE_URL="",
        SUPABASE_KEY="",
        SUPABASE_JWT_SECRET="",
    )
    assert settings.AUTH_MODE == "local"


def test_supabase_mode_keeps_accepting_what_it_accepted_before():
    """The check is for an absent setting, as `Field(...)` was: a value that is present is not
    second-guessed, so a deployment that boots today keeps booting."""
    settings = Settings(
        _env_file=None,
        AUTH_MODE="supabase",
        SUPABASE_URL="",
        SUPABASE_KEY="",
        SUPABASE_JWT_SECRET="",
    )
    assert settings.SUPABASE_URL == ""


def test_local_mode_still_takes_the_hosted_settings_when_they_are_given():
    """The migration window: local mode with the provider still configured (migration script)."""
    settings = Settings(_env_file=None, AUTH_MODE="local", **_hosted())
    assert settings.SUPABASE_URL == "https://placeholder.supabase.example.test"


def test_the_mode_comes_from_the_environment_like_any_setting(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "local")
    assert Settings(_env_file=None).AUTH_MODE == "local"
    monkeypatch.setenv("AUTH_MODE", "supabase")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_the_error_never_prints_a_value_the_environment_holds():
    """A refused boot ends in the logs: the environment it was refused for (API keys, secrets) must
    not be printed with it."""
    sentinel = "sentinel-secret-must-not-be-printed"
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            _env_file=None, AUTH_MODE="supabase", LLM_API_KEY=sentinel, STRIPE_SECRET_KEY=sentinel
        )
    assert sentinel not in str(excinfo.value)
    assert "SUPABASE_URL" in str(excinfo.value)


# --------------------------------------------------------------------------------------------
# Where the optional settings are read: an instance without the hosted provider refuses its tokens
# --------------------------------------------------------------------------------------------


def test_without_the_hosted_provider_none_of_its_tokens_is_valid(monkeypatch):
    """`AUTH_MODE=local` without SUPABASE_JWT_SECRET: a token is refused (401), never decoded with a
    missing key, and the hosted-provider helpers say what is wrong instead of failing obscurely."""
    import asyncio
    from datetime import datetime, timedelta, timezone

    import jwt
    from fastapi import HTTPException

    from app.core import security
    from app.core.config import settings
    from app.core.supabase_auth import verify_supabase_token

    forged = jwt.encode(
        {
            "sub": "00000000-0000-0000-0000-000000000001",
            "email": "someone@example.test",
            "aud": "authenticated",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        "x" * 32,
        algorithm="HS256",
    )
    monkeypatch.setattr(settings, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(settings, "SUPABASE_KEY", None)

    with pytest.raises(HTTPException) as excinfo:
        security.verify_token(forged)
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "Could not validate credentials"
    assert asyncio.run(verify_supabase_token(forged)) is None
    with pytest.raises(RuntimeError, match="SUPABASE_JWT_SECRET"):
        security.create_access_token({"sub": "x"})

    from app.auth.directory import _provider_headers

    with pytest.raises(RuntimeError, match="SUPABASE_KEY"):
        _provider_headers()


def test_with_the_hosted_provider_the_token_checks_are_unchanged(monkeypatch):
    import asyncio
    from datetime import datetime, timedelta, timezone

    import jwt

    from app.core import security
    from app.core.config import settings
    from app.core.supabase_auth import verify_supabase_token

    monkeypatch.setattr(settings, "SUPABASE_JWT_SECRET", "a-secret-of-at-least-32-characters-ok")
    token = security.create_access_token({"sub": "00000000-0000-0000-0000-000000000001"})
    assert security.verify_token(token).sub == "00000000-0000-0000-0000-000000000001"

    wrong = jwt.encode(
        {
            "sub": "00000000-0000-0000-0000-000000000001",
            "aud": "authenticated",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        "another-secret-of-at-least-32-chars",
        algorithm="HS256",
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        security.verify_token(wrong)
    assert excinfo.value.status_code == 401
    assert asyncio.run(verify_supabase_token(wrong)) is None
