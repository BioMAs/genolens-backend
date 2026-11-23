"""
Tests of the self-hosted authentication keys and access tokens (`app/auth/keys.py`,
`app/auth/tokens.py`).

Keys are generated here, never committed. Nothing hosted is called.
"""

import base64
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

import app.main as app_main
from app.auth.keys import (
    SigningKey,
    generate_private_key_b64,
    jwks,
    load_signing_keys,
    reset_key_cache,
)
from app.auth.mfa import generate_encryption_key, load_cipher, reset_encryption_cache
from app.auth.tokens import (
    ACCESS_TOKEN_TTL,
    AUDIENCE,
    InvalidAccessToken,
    decode_access_token,
    hash_token,
    is_local_token,
    issue_access_token,
    new_opaque_token,
)
from app.core.config import settings


@pytest.fixture(autouse=True)
def _clean_key_cache():
    """No key (ephemeral or not) leaks from one test to the next: signing or encryption."""
    reset_key_cache()
    reset_encryption_cache()
    yield
    reset_key_cache()
    reset_encryption_cache()


def _claims(**overrides):
    now = datetime.now(UTC)
    claims = {
        "iss": settings.AUTH_ISSUER,
        "aud": AUDIENCE,
        "sub": str(uuid4()),
        "sid": str(uuid4()),
        "amr": ["pwd"],
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
    }
    claims.update(overrides)
    return claims


def _sign(claims: dict, key: SigningKey, kid=True) -> str:
    """A token signed by hand, to craft the ones `issue_access_token` would never emit."""
    headers = {"kid": key.kid} if kid else {}
    return jwt.encode(claims, key.private_key, algorithm="ES256", headers=headers)


# ── Settings ─────────────────────────────────────────────────────────────────


def test_settings_defaults_leave_the_hosted_provider_in_charge():
    assert settings.AUTH_MODE == "supabase"
    assert settings.AUTH_LOCAL_ENABLED is False
    assert settings.AUTH_COOKIE_SECURE is True
    assert settings.AUTH_COOKIE_DOMAIN is None
    assert settings.AUTH_ENCRYPTION_KEYS == ""
    assert settings.SUPABASE_DB_URL is None


# ── Tokens ───────────────────────────────────────────────────────────────────


def test_round_trip(local_keys):
    uid, sid = uuid4(), uuid4()
    claims = decode_access_token(issue_access_token(uid, sid, ["pwd", "otp"]))
    assert (claims.sub, claims.sid, claims.amr) == (uid, sid, ["pwd", "otp"])


def test_token_lifetime_and_header(local_keys):
    now = datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)
    token = issue_access_token(uuid4(), uuid4(), ["pwd"], now=now)
    header = jwt.get_unverified_header(token)
    assert header["alg"] == "ES256"
    assert header["kid"] == load_signing_keys()[0].kid  # the first key signs
    payload = jwt.decode(token, options={"verify_signature": False})
    assert payload["iss"] == settings.AUTH_ISSUER
    assert payload["aud"] == AUDIENCE == "genolens"
    assert payload["iat"] == int(now.timestamp())
    assert payload["exp"] - payload["iat"] == int(ACCESS_TOKEN_TTL.total_seconds()) == 15 * 60


def test_decoded_expiry_is_an_aware_datetime(local_keys):
    claims = decode_access_token(issue_access_token(uuid4(), uuid4(), []))
    assert claims.exp.tzinfo is not None
    assert claims.exp > datetime.now(UTC)


def test_rejects_expired_wrong_aud_wrong_iss_and_hs256(local_keys):
    old = issue_access_token(
        uuid4(), uuid4(), ["pwd"], now=datetime.now(UTC) - timedelta(minutes=16)
    )
    supabase_like = jwt.encode(
        {"sub": str(uuid4()), "aud": "authenticated"}, "secret", algorithm="HS256"
    )
    for bad in (old, supabase_like, "garbage"):
        with pytest.raises(InvalidAccessToken):
            decode_access_token(bad)
    assert not is_local_token(supabase_like) and is_local_token(
        issue_access_token(uuid4(), uuid4(), [])
    )


def test_rejects_wrong_audience(local_keys):
    token = _sign(_claims(aud="authenticated"), load_signing_keys()[0])
    with pytest.raises(InvalidAccessToken):
        decode_access_token(token)


def test_rejects_wrong_issuer(local_keys):
    token = _sign(_claims(iss="https://elsewhere.example.test"), load_signing_keys()[0])
    with pytest.raises(InvalidAccessToken):
        decode_access_token(token)


def test_rejects_token_issued_for_another_issuer(local_keys, monkeypatch):
    token = issue_access_token(uuid4(), uuid4(), ["pwd"])
    monkeypatch.setattr(settings, "AUTH_ISSUER", "https://other-deployment.example.test")
    with pytest.raises(InvalidAccessToken):
        decode_access_token(token)


@pytest.mark.parametrize("missing", ["iss", "aud", "exp", "sub", "sid"])
def test_rejects_token_missing_a_required_claim(local_keys, missing):
    claims = _claims()
    del claims[missing]
    with pytest.raises(InvalidAccessToken):
        decode_access_token(_sign(claims, load_signing_keys()[0]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"sub": "not-a-uuid"},
        {"sid": "not-a-uuid"},
        {"amr": "pwd"},
        {"amr": [1, 2]},
        {"amr": None},
    ],
)
def test_rejects_malformed_claims(local_keys, overrides):
    with pytest.raises(InvalidAccessToken):
        decode_access_token(_sign(_claims(**overrides), load_signing_keys()[0]))


def test_rejects_alg_none_and_hmac_with_a_known_kid(local_keys):
    key = load_signing_keys()[0]
    unsigned = jwt.encode(_claims(), None, algorithm="none", headers={"kid": key.kid})
    hmac_signed = jwt.encode(
        _claims(), key.public_jwk["x"] * 2, algorithm="HS256", headers={"kid": key.kid}
    )
    for bad in (unsigned, hmac_signed):
        with pytest.raises(InvalidAccessToken):
            decode_access_token(bad)
        assert not is_local_token(bad)


def test_rejects_other_ec_algorithms(local_keys):
    key = load_signing_keys()[0]
    p384 = ec.generate_private_key(ec.SECP384R1())
    token = jwt.encode(_claims(), p384, algorithm="ES384", headers={"kid": key.kid})
    with pytest.raises(InvalidAccessToken):
        decode_access_token(token)
    assert not is_local_token(token)


def test_rejects_unknown_or_missing_kid(local_keys):
    stranger = load_signing_keys()[0]
    foreign_private = ec.generate_private_key(ec.SECP256R1())
    foreign = SigningKey(kid="0" * 16, private_key=foreign_private, public_jwk={})
    for token in (
        _sign(_claims(), foreign),  # kid not configured
        _sign(_claims(), stranger, kid=False),  # no kid at all
        jwt.encode(_claims(), foreign_private, algorithm="ES256", headers={"kid": stranger.kid}),
    ):  # a configured kid, but signed by another key
        with pytest.raises(InvalidAccessToken):
            decode_access_token(token)


def test_rejects_tampered_payload(local_keys):
    token = issue_access_token(uuid4(), uuid4(), ["pwd"])
    header, payload, signature = token.split(".")
    forged = jwt.utils.base64url_encode(
        b'{"sub":"%s","sid":"%s","amr":["pwd"],"aud":"genolens","iss":"%s","exp":9999999999}'
        % (str(uuid4()).encode(), str(uuid4()).encode(), settings.AUTH_ISSUER.encode())
    ).decode()
    with pytest.raises(InvalidAccessToken):
        decode_access_token(f"{header}.{forged}.{signature}")


def test_is_local_token_does_not_raise_on_junk(local_keys):
    for junk in ("", "garbage", "a.b.c", "....", "é"):
        assert is_local_token(junk) is False


def test_rotation_second_key_still_verifies(local_keys):
    first, second = load_signing_keys()
    assert first.kid != second.kid
    # A token signed by key #2 (as if it had been issued before the rotation) verifies...
    old_token = _sign(_claims(), second)
    assert decode_access_token(old_token).amr == ["pwd"]
    # ... and so does one signed by key #1 (the current signing key).
    assert decode_access_token(issue_access_token(uuid4(), uuid4(), [])).amr == []
    # Once key #2 is dropped from the setting, its tokens are refused.
    only_first = settings.AUTH_JWT_PRIVATE_KEYS.split(",")[0]
    settings.AUTH_JWT_PRIVATE_KEYS = only_first  # restored by the fixture's monkeypatch
    reset_key_cache()
    assert [k.kid for k in load_signing_keys()] == [first.kid]
    with pytest.raises(InvalidAccessToken):
        decode_access_token(old_token)


def test_opaque_tokens_and_hash():
    first, second = new_opaque_token(), new_opaque_token()
    assert first != second
    assert len(first) >= 43  # secrets.token_urlsafe(32)
    assert hash_token("abc") == hashlib.sha256(b"abc").hexdigest()
    assert hash_token(first) != hash_token(second)
    assert len(hash_token(first)) == 64


# ── Keys ─────────────────────────────────────────────────────────────────────


def test_jwks_lists_every_public_key(local_keys):
    assert {k["kid"] for k in jwks()["keys"]} == {k.kid for k in load_signing_keys()}
    assert all("d" not in k for k in jwks()["keys"])  # never the private part


def test_jwk_shape_and_kid_derivation(local_keys):
    keys = load_signing_keys()
    assert len(jwks()["keys"]) == 2
    for key in keys:
        public = key.private_key.public_key()
        der = public.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        assert key.kid == hashlib.sha256(der).hexdigest()[:16]
        jwk = key.public_jwk
        assert (jwk["kty"], jwk["crv"], jwk["alg"], jwk["use"], jwk["kid"]) == (
            "EC",
            "P-256",
            "ES256",
            "sig",
            key.kid,
        )
        numbers = public.public_numbers()
        for coordinate, value in (("x", numbers.x), ("y", numbers.y)):
            raw = base64.urlsafe_b64decode(jwk[coordinate] + "=" * (-len(jwk[coordinate]) % 4))
            assert len(raw) == 32  # fixed width, whatever the leading zero bytes
            assert int.from_bytes(raw, "big") == value
        assert set(jwk) == {"kty", "crv", "x", "y", "kid", "alg", "use"}


def test_published_jwks_verify_a_token(local_keys):
    """A third party holding only the JWKS can verify what the service issues."""
    token = issue_access_token(uuid4(), uuid4(), ["pwd"])
    kid = jwt.get_unverified_header(token)["kid"]
    jwk = next(k for k in jwks()["keys"] if k["kid"] == kid)
    public_key = jwt.PyJWK(jwk).key
    payload = jwt.decode(token, public_key, algorithms=["ES256"], audience=AUDIENCE)
    assert payload["iss"] == settings.AUTH_ISSUER


def test_load_signing_keys_is_cached(local_keys):
    assert load_signing_keys() is load_signing_keys()
    reset_key_cache()
    assert load_signing_keys() is not None


def test_generated_key_is_pkcs8_pem_in_base64():
    encoded = generate_private_key_b64()
    pem = base64.b64decode(encoded)
    assert pem.startswith(b"-----BEGIN PRIVATE KEY-----")
    key = serialization.load_pem_private_key(pem, password=None)
    assert isinstance(key, ec.EllipticCurvePrivateKey) and key.curve.name == "secp256r1"
    assert generate_private_key_b64() != encoded


def test_first_configured_key_signs_and_whitespace_is_ignored(monkeypatch):
    one, two = generate_private_key_b64(), generate_private_key_b64()
    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", f" {two} ,, {one} ,")
    keys = load_signing_keys()
    assert len(keys) == 2
    token = issue_access_token(uuid4(), uuid4(), [])
    assert jwt.get_unverified_header(token)["kid"] == keys[0].kid


def test_duplicate_key_is_listed_once(monkeypatch):
    one = generate_private_key_b64()
    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", f"{one},{one}")
    assert len(load_signing_keys()) == 1


def test_development_without_keys_uses_one_ephemeral_key_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", "")
    with caplog.at_level(logging.WARNING, logger="app.auth.keys"):
        keys = load_signing_keys()
        assert load_signing_keys() is keys  # generated once per process, not per call
    assert len(keys) == 1
    assert sum("ephemeral" in r.getMessage().lower() for r in caplog.records) == 1
    # It is a working key: a token issued with it verifies.
    assert decode_access_token(issue_access_token(uuid4(), uuid4(), [])).amr == []


@pytest.mark.parametrize("environment", ["production", "PRODUCTION"])
def test_production_without_keys_fails(monkeypatch, environment):
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)
    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", "")
    with pytest.raises(RuntimeError, match="AUTH_JWT_PRIVATE_KEYS is required in production"):
        load_signing_keys()
    # The failure is not cached as a key: it fails again, and no ephemeral key was kept.
    with pytest.raises(RuntimeError):
        load_signing_keys()


def test_production_with_only_blank_entries_fails(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", " , ,")
    with pytest.raises(RuntimeError, match="required in production"):
        load_signing_keys()


@pytest.mark.parametrize("environment", ["development", "production"])
def test_invalid_key_is_refused_without_echoing_it(monkeypatch, environment):
    secret_looking = "bm90LWEta2V5LW1hdGVyaWFs"  # base64 of a non-key
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)
    monkeypatch.setattr(
        settings, "AUTH_JWT_PRIVATE_KEYS", f"{generate_private_key_b64()},{secret_looking}"
    )
    with pytest.raises(RuntimeError) as excinfo:
        load_signing_keys()
    assert secret_looking not in str(excinfo.value)
    assert "AUTH_JWT_PRIVATE_KEYS" in str(excinfo.value)


def test_non_p256_key_is_refused(monkeypatch):
    p384 = ec.generate_private_key(ec.SECP384R1())
    pem = p384.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", base64.b64encode(pem).decode())
    with pytest.raises(RuntimeError, match="P-256"):
        load_signing_keys()


# ── Application start-up ─────────────────────────────────────────────────────


@pytest.fixture
def quiet_lifespan(monkeypatch):
    """The lifespan without its hosted shutdown calls (database engine, HTTP client)."""

    async def _noop():
        return None

    monkeypatch.setattr(app_main, "close_db", _noop)
    monkeypatch.setattr(app_main, "close_supabase_http_client", _noop)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, enabled",
    [("local", False), ("supabase", True), ("local", True)],
)
async def test_startup_fails_fast_in_production_without_keys(
    monkeypatch, quiet_lifespan, mode, enabled
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", "")
    monkeypatch.setattr(settings, "AUTH_MODE", mode)
    monkeypatch.setattr(settings, "AUTH_LOCAL_ENABLED", enabled)
    with pytest.raises(RuntimeError, match="AUTH_JWT_PRIVATE_KEYS is required in production"):
        async with app_main.lifespan(app_main.app):
            pytest.fail("the application must not start")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, enabled",
    [("local", False), ("supabase", True), ("local", True)],
)
async def test_startup_fails_fast_in_production_without_encryption_keys(
    monkeypatch, quiet_lifespan, local_keys, mode, enabled
):
    """Signing keys are there; the encryption keys (second-factor secrets, recovery-code hashes)
    are not: the application must not start, rather than fail at the first enrolment."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", "")
    monkeypatch.setattr(settings, "AUTH_MODE", mode)
    monkeypatch.setattr(settings, "AUTH_LOCAL_ENABLED", enabled)
    with pytest.raises(RuntimeError, match="AUTH_ENCRYPTION_KEYS is required in production"):
        async with app_main.lifespan(app_main.app):
            pytest.fail("the application must not start")


@pytest.mark.asyncio
async def test_startup_fails_fast_on_a_malformed_encryption_key(
    monkeypatch, quiet_lifespan, local_keys
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "AUTH_MODE", "local")
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", f"{generate_encryption_key()},nope")
    with pytest.raises(RuntimeError, match="AUTH_ENCRYPTION_KEYS entry 2"):
        async with app_main.lifespan(app_main.app):
            pytest.fail("the application must not start")


@pytest.mark.asyncio
async def test_startup_in_production_with_both_sets_of_keys(
    monkeypatch, quiet_lifespan, local_keys, enc_keys
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "AUTH_MODE", "local")
    async with app_main.lifespan(app_main.app):
        pass


@pytest.mark.asyncio
async def test_startup_loads_keys_when_local_auth_is_on(
    monkeypatch, quiet_lifespan, local_keys, enc_keys
):
    monkeypatch.setattr(settings, "AUTH_MODE", "local")
    signing_spy = MagicMock(wraps=load_signing_keys)
    cipher_spy = MagicMock(wraps=load_cipher)
    monkeypatch.setattr(app_main, "load_signing_keys", signing_spy)
    monkeypatch.setattr(app_main, "load_cipher", cipher_spy)
    async with app_main.lifespan(app_main.app):
        pass
    signing_spy.assert_called_once_with()
    cipher_spy.assert_called_once_with()


@pytest.mark.asyncio
async def test_startup_is_unchanged_with_the_default_settings(monkeypatch, quiet_lifespan):
    """AUTH_MODE=supabase and AUTH_LOCAL_ENABLED=False: no key is read, none is required."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "AUTH_JWT_PRIVATE_KEYS", "")
    monkeypatch.setattr(settings, "AUTH_MODE", "supabase")
    monkeypatch.setattr(settings, "AUTH_LOCAL_ENABLED", False)
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", "")
    spy = MagicMock()
    cipher_spy = MagicMock()
    monkeypatch.setattr(app_main, "load_signing_keys", spy)
    monkeypatch.setattr(app_main, "load_cipher", cipher_spy)
    async with app_main.lifespan(app_main.app):
        pass
    spy.assert_not_called()
    cipher_spy.assert_not_called()
