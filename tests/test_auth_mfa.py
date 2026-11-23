"""
Tests of the TOTP second factor, the encryption of its secret at rest and the recovery codes
(`app/auth/mfa.py`).

Keys and secrets are generated here, never committed. Nothing hosted is called.
"""

import base64
import hashlib
import logging
import re
from datetime import UTC, datetime, timedelta, timezone
from urllib.parse import parse_qs, unquote, urlparse

import pyotp
import pytest
from cryptography.fernet import Fernet, InvalidToken

from app.auth import mfa
from app.auth.mfa import (
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
    generate_encryption_key,
    generate_recovery_codes,
    hash_recovery_code,
    load_cipher,
    new_totp_secret,
    normalize_recovery_code,
    provisioning_uri,
    reset_encryption_cache,
    verify_totp,
)
from app.core.config import settings

# 12:00:00 UTC is on a 30-second boundary (86400 is a multiple of 30): T0 is the first second of
# its step, T0 + 29 s the last one.
T0 = datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)
STEP0 = int(T0.timestamp()) // 30


@pytest.fixture(autouse=True)
def _fresh_cipher_cache():
    """No cipher, ephemeral or not, leaks from one test to another."""
    reset_encryption_cache()
    yield
    reset_encryption_cache()


def _code_for_step(secret: str, step: int) -> str:
    return pyotp.TOTP(secret).generate_otp(step)


# --------------------------------------------------------------------------------------------
# TOTP secret and provisioning URI
# --------------------------------------------------------------------------------------------


def test_new_totp_secret_is_32_base32_characters_of_160_bits():
    secret = new_totp_secret()
    assert re.fullmatch(r"[A-Z2-7]{32}", secret)
    assert len(base64.b32decode(secret)) == 20  # 160 bits, no padding needed
    assert len({new_totp_secret() for _ in range(50)}) == 50


def test_provisioning_uri_names_the_issuer_and_the_account():
    secret = new_totp_secret()
    uri = provisioning_uri(secret, "ada@example.test")
    parsed = urlparse(uri)
    assert (parsed.scheme, parsed.netloc) == ("otpauth", "totp")
    assert unquote(parsed.path) == "/GenoLens:ada@example.test"
    query = parse_qs(parsed.query)
    assert query["secret"] == [secret]
    assert query["issuer"] == ["GenoLens"]
    # SHA-1, 6 digits and 30 seconds are what authenticator apps assume: not spelled out.
    assert not {"algorithm", "digits", "period"} & set(query)


def test_provisioning_uri_escapes_the_account_name():
    uri = provisioning_uri(new_totp_secret(), "a+b c@example.test")
    parsed = urlparse(uri)
    assert unquote(parsed.path) == "/GenoLens:a+b c@example.test"
    assert " " not in uri and "+" not in parsed.path


def test_provisioning_uri_secret_generates_the_codes_of_the_app():
    secret = new_totp_secret()
    from_uri = pyotp.parse_uri(provisioning_uri(secret, "ada@example.test"))
    assert from_uri.at(T0) == pyotp.TOTP(secret).at(T0)


# --------------------------------------------------------------------------------------------
# verify_totp
# --------------------------------------------------------------------------------------------


def test_totp_window_and_replay():
    s = new_totp_secret()
    code = pyotp.TOTP(s).at(T0)
    step = verify_totp(s, code, None, now=T0)
    assert step == int(T0.timestamp()) // 30
    assert verify_totp(s, code, step, now=T0) is None  # replay
    assert verify_totp(s, code, None, now=T0 + timedelta(seconds=30)) == step  # +1 step tolerated
    assert verify_totp(s, code, None, now=T0 + timedelta(seconds=90)) is None  # outside window


def test_window_is_one_step_either_side_and_returns_the_step_of_the_code():
    s = new_totp_secret()
    for offset in (-1, 0, 1):
        code = _code_for_step(s, STEP0 + offset)
        assert verify_totp(s, code, None, now=T0) == STEP0 + offset
    for offset in (-2, 2, -10, 10):
        assert verify_totp(s, _code_for_step(s, STEP0 + offset), None, now=T0) is None


def test_window_follows_the_instant_not_the_step_start():
    s = new_totp_secret()
    code = _code_for_step(s, STEP0)
    # T0 + 29 s is still step STEP0; T0 + 30 s is STEP0 + 1; +59 s is its last second.
    assert verify_totp(s, code, None, now=T0 + timedelta(seconds=29)) == STEP0
    assert verify_totp(s, code, None, now=T0 + timedelta(seconds=59)) == STEP0
    assert verify_totp(s, code, None, now=T0 + timedelta(seconds=60)) is None
    # Mirror image: the code of the next step is valid from the first second of the step before.
    nxt = _code_for_step(s, STEP0 + 1)
    assert verify_totp(s, nxt, None, now=T0) == STEP0 + 1
    assert verify_totp(s, nxt, None, now=T0 + timedelta(seconds=29)) == STEP0 + 1
    assert verify_totp(s, nxt, None, now=T0 - timedelta(seconds=1)) is None  # two steps early


def test_wrong_code_is_refused():
    s = new_totp_secret()
    good = {_code_for_step(s, STEP0 + offset) for offset in (-1, 0, 1)}
    wrong = next(f"{n:06d}" for n in range(1_000_000) if f"{n:06d}" not in good)
    assert verify_totp(s, wrong, None, now=T0) is None


def test_replay_guard_rejects_any_step_not_after_the_last_used_one():
    s = new_totp_secret()
    previous, current, following = (_code_for_step(s, STEP0 + k) for k in (-1, 0, 1))
    # Last accepted step is the current one: nothing at or before it comes back.
    assert verify_totp(s, previous, STEP0, now=T0) is None
    assert verify_totp(s, current, STEP0, now=T0) is None
    # ...but the next step's code does, and reports its own step.
    assert verify_totp(s, following, STEP0, now=T0) == STEP0 + 1
    # A last-used step in the future (clock moved back) refuses everything in the window.
    assert verify_totp(s, following, STEP0 + 1, now=T0) is None
    assert verify_totp(s, following, STEP0 + 5, now=T0) is None
    # A last-used step that is old is no obstacle.
    assert verify_totp(s, current, STEP0 - 7, now=T0) == STEP0
    # Zero is a real step number, not "none".
    assert verify_totp(s, current, 0, now=T0) == STEP0


def test_code_with_space_is_accepted():
    s = new_totp_secret()
    c = pyotp.TOTP(s).at(T0)
    assert verify_totp(s, f"{c[:3]} {c[3:]}", None, now=T0) is not None


@pytest.mark.parametrize(
    "decorate",
    [
        lambda c: f" {c}",
        lambda c: f"{c} ",
        lambda c: f"\t{c}\n",
        lambda c: f"{c[:2]}  {c[2:4]}\t{c[4:]}",
        lambda c: f"{c[:3]} {c[3:]}",  # no-break space, as some phones paste it
        lambda c: "\n".join(c),
    ],
)
def test_whitespace_around_and_inside_the_code_is_ignored(decorate):
    s = new_totp_secret()
    c = pyotp.TOTP(s).at(T0)
    assert verify_totp(s, decorate(c), None, now=T0) == STEP0


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " ",
        "12345",
        "1234567",
        "abcdef",
        "12345a",
        "12-345",
        "+12345",
        "١٢٣٤٥٦",  # six Arabic-Indic digits: str.isdigit() says yes, a TOTP code is ASCII
        "１２３４５６",  # six full-width digits
        "12²456",  # superscript two
        "000000\x00",
    ],
)
def test_malformed_codes_are_refused_without_raising(bad):
    assert verify_totp(new_totp_secret(), bad, None, now=T0) is None


def test_non_string_code_is_refused_without_raising():
    s = new_totp_secret()
    assert verify_totp(s, None, None, now=T0) is None  # type: ignore[arg-type]
    assert verify_totp(s, 123456, None, now=T0) is None  # type: ignore[arg-type]
    assert verify_totp(s, b"123456", None, now=T0) is None  # type: ignore[arg-type]


def test_naive_now_is_read_as_utc():
    s = new_totp_secret()
    code = _code_for_step(s, STEP0)
    assert verify_totp(s, code, None, now=T0.replace(tzinfo=None)) == STEP0


def test_aware_now_in_another_zone_is_the_same_instant():
    s = new_totp_secret()
    code = _code_for_step(s, STEP0)
    paris = T0.astimezone(timezone(timedelta(hours=1)))
    assert verify_totp(s, code, None, now=paris) == STEP0


def test_now_defaults_to_the_clock():
    s = new_totp_secret()
    step = verify_totp(s, pyotp.TOTP(s).now(), None)
    assert step is not None
    assert abs(step - int(datetime.now(UTC).timestamp()) // 30) <= 1


def test_every_step_of_the_window_is_compared_whatever_matches(monkeypatch):
    """No early exit: how long the answer takes does not say which step (if any) matched."""
    calls = []
    real = mfa.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(mfa.hmac, "compare_digest", spy)
    s = new_totp_secret()
    for offset in (-1, 0, 1, 9):
        calls.clear()
        verify_totp(s, _code_for_step(s, STEP0 + offset), None, now=T0)
        assert len(calls) == 3, f"offset {offset}"


def test_when_several_steps_match_the_highest_unused_one_is_returned(monkeypatch):
    """Two steps of the window can produce the same code (about one in a million). The highest
    one is reported, so that storing it as `last_used_step` also shuts out replays at the lower
    ones."""
    monkeypatch.setattr(pyotp.TOTP, "generate_otp", lambda self, step: "424242")
    s = new_totp_secret()
    assert verify_totp(s, "424242", None, now=T0) == STEP0 + 1
    assert verify_totp(s, "424242", STEP0 - 1, now=T0) == STEP0 + 1
    assert verify_totp(s, "424242", STEP0, now=T0) == STEP0 + 1
    assert verify_totp(s, "424242", STEP0 + 1, now=T0) is None


def test_a_corrupt_secret_is_an_error_not_a_refusal():
    # A secret that is not base32 is a data problem; it must not read as "wrong code".
    with pytest.raises(ValueError):  # binascii.Error
        verify_totp("not base32 !", "123456", None, now=T0)


# --------------------------------------------------------------------------------------------
# Recovery codes
# --------------------------------------------------------------------------------------------


def test_recovery_codes_format_and_normalization():
    codes = generate_recovery_codes()
    assert len(codes) == 10 == len(set(codes))
    assert all(re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{5}-[0-9A-HJKMNP-TV-Z]{5}", c) for c in codes)
    assert hash_recovery_code(codes[0].lower().replace("-", "")) == hash_recovery_code(codes[0])


def test_recovery_codes_use_the_crockford_alphabet_only():
    alphabet = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
    seen = set()
    for code in generate_recovery_codes(200):
        seen |= set(code.replace("-", ""))
    assert seen <= alphabet
    assert not seen & set("ILOU")  # the four letters Crockford leaves out
    assert len(seen) > 28  # 2000 draws touch (almost) every symbol: not a degenerate generator


def test_recovery_codes_count_and_distinctness():
    assert len(generate_recovery_codes(3)) == 3
    assert generate_recovery_codes(0) == []
    many = generate_recovery_codes(500)
    assert len(many) == len(set(many)) == 500
    assert set(generate_recovery_codes()).isdisjoint(generate_recovery_codes())


def test_recovery_codes_are_drawn_from_secrets_not_random(monkeypatch):
    drawn = []
    real = mfa.secrets.choice

    def spy(seq):
        drawn.append(seq)
        return real(seq)

    monkeypatch.setattr(mfa.secrets, "choice", spy)
    generate_recovery_codes(2)
    assert len(drawn) == 2 * 10


def test_negative_count_is_refused():
    with pytest.raises(ValueError):
        generate_recovery_codes(-1)


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("abcde-fghjk", "ABCDEFGHJK"),
        ("ABCDE-FGHJK", "ABCDEFGHJK"),
        ("  abcde fghjk  ", "ABCDEFGHJK"),
        ("abcde - fghjk", "ABCDEFGHJK"),
        ("ab-cd-ef-gh-jk", "ABCDEFGHJK"),
        ("\tAbCdE\nfGhJk\n", "ABCDEFGHJK"),
        ("", ""),
    ],
)
def test_normalize_recovery_code(typed, expected):
    assert normalize_recovery_code(typed) == expected


def test_hash_recovery_code_is_sha256_hex_of_the_normalized_code():
    expected = hashlib.sha256(b"ABCDEFGHJK").hexdigest()
    assert hash_recovery_code("abcde-fghjk") == expected
    assert hash_recovery_code("ABCDE FGHJK") == expected
    assert re.fullmatch(r"[0-9a-f]{64}", expected)
    assert hash_recovery_code("ABCDE-FGHJL") != expected


def test_hashes_of_distinct_codes_are_distinct_and_never_the_code():
    codes = generate_recovery_codes()
    hashes = [hash_recovery_code(c) for c in codes]
    assert len(set(hashes)) == len(codes)
    assert not set(hashes) & set(codes)


# --------------------------------------------------------------------------------------------
# Secret encryption
# --------------------------------------------------------------------------------------------


def test_secret_encryption_round_trip_and_rotation(enc_keys, monkeypatch):
    k1, k2 = enc_keys
    assert decrypt_secret(encrypt_secret("ABC")) == "ABC"

    # A secret written when k2 was the only key...
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", k2)
    reset_encryption_cache()
    old_token = encrypt_secret("ABC")
    assert Fernet(k2).decrypt(old_token.encode()) == b"ABC"

    # ...still decrypts once k1 became the primary and k2 the previous key...
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", f"{k1},{k2}")
    reset_encryption_cache()
    assert decrypt_secret(old_token) == "ABC"

    # ...while every new encryption uses k1 and k1 alone.
    new_token = encrypt_secret("ABC")
    assert Fernet(k1).decrypt(new_token.encode()) == b"ABC"
    with pytest.raises(InvalidToken):
        Fernet(k2).decrypt(new_token.encode())
    assert decrypt_secret(new_token) == "ABC"

    # Once k2 is dropped, what it alone protected is gone, loudly.
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", k1)
    reset_encryption_cache()
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(old_token)
    assert decrypt_secret(new_token) == "ABC"


def test_encrypted_secret_is_not_the_secret_and_is_never_the_same_twice(enc_keys):
    secret = new_totp_secret()
    token = encrypt_secret(secret)
    assert isinstance(token, str)
    assert secret not in token
    assert encrypt_secret(secret) != token  # Fernet draws a fresh IV each time
    assert decrypt_secret(token) == secret


@pytest.mark.parametrize("plain", ["", "ABC", "é€✓ unicode", "A" * 5000])
def test_round_trip_of_unusual_secrets(enc_keys, plain):
    assert decrypt_secret(encrypt_secret(plain)) == plain


def test_decrypting_garbage_raises_a_clear_error_never_garbage(enc_keys):
    with pytest.raises(SecretDecryptionError):
        decrypt_secret("not a fernet token")
    with pytest.raises(SecretDecryptionError):
        decrypt_secret("")
    with pytest.raises(SecretDecryptionError):
        decrypt_secret("é€✓")  # non-ASCII text is not even base64
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(None)  # type: ignore[arg-type]


def test_token_encrypted_by_an_unknown_key_is_refused(enc_keys):
    foreign = Fernet(generate_encryption_key().encode()).encrypt(b"ABC").decode()
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(foreign)


def test_tampered_token_is_refused(enc_keys):
    token = encrypt_secret("ABC")
    raw = bytearray(base64.urlsafe_b64decode(token))
    raw[-1] ^= 0x01
    tampered = base64.urlsafe_b64encode(bytes(raw)).decode()
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(tampered)
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(token[:-4])


def test_decryption_error_does_not_leak_the_token_or_a_key(enc_keys):
    k1, k2 = enc_keys
    token = Fernet(generate_encryption_key().encode()).encrypt(b"top-secret-seed").decode()
    with pytest.raises(SecretDecryptionError) as caught:
        decrypt_secret(token)
    text = f"{caught.value!s} {caught.value!r}"
    assert token not in text and "top-secret-seed" not in text
    assert k1 not in text and k2 not in text


def test_generate_encryption_key_is_a_usable_fernet_key():
    key = generate_encryption_key()
    assert isinstance(key, str)
    assert Fernet(key.encode()).decrypt(Fernet(key.encode()).encrypt(b"x")) == b"x"
    assert generate_encryption_key() != key


# --------------------------------------------------------------------------------------------
# Encryption keys: configuration policy (mirrors the signing keys)
# --------------------------------------------------------------------------------------------


def test_empty_setting_outside_production_uses_one_ephemeral_key_and_warns_once(
    monkeypatch, caplog
):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", "")
    with caplog.at_level(logging.WARNING, logger="app.auth.mfa"):
        token = encrypt_secret("ABC")
        assert decrypt_secret(token) == "ABC"  # same key on the second call: it is cached
        assert decrypt_secret(encrypt_secret("DEF")) == "DEF"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "AUTH_ENCRYPTION_KEYS" in warnings[0].getMessage()
    assert "ephemeral" in warnings[0].getMessage()

    # A new process (cache dropped) has a new key: what the old one wrote is unreadable.
    reset_encryption_cache()
    with pytest.raises(SecretDecryptionError):
        decrypt_secret(token)


@pytest.mark.parametrize("environment", ["production", "PRODUCTION"])
def test_production_without_keys_fails(monkeypatch, environment):
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", "")
    with pytest.raises(RuntimeError, match="AUTH_ENCRYPTION_KEYS is required in production"):
        encrypt_secret("ABC")
    with pytest.raises(RuntimeError, match="AUTH_ENCRYPTION_KEYS is required in production"):
        decrypt_secret("whatever")
    # The failure is not cached as a key: it fails again, and no ephemeral key was kept.
    with pytest.raises(RuntimeError):
        load_cipher()


def test_production_with_only_blank_entries_fails(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", " , ,")
    with pytest.raises(RuntimeError, match="required in production"):
        load_cipher()


def test_production_with_keys_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", generate_encryption_key())
    with caplog.at_level(logging.WARNING, logger="app.auth.mfa"):
        assert decrypt_secret(encrypt_secret("ABC")) == "ABC"
    assert not caplog.records


def test_blank_entries_and_surrounding_spaces_are_ignored(monkeypatch):
    k1, k2 = generate_encryption_key(), generate_encryption_key()
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", f" , {k1} ,, {k2}, ")
    token = encrypt_secret("ABC")
    assert Fernet(k1.encode()).decrypt(token.encode()) == b"ABC"  # first non-blank key encrypts
    assert decrypt_secret(Fernet(k2.encode()).encrypt(b"XYZ").decode()) == "XYZ"


def test_the_same_key_listed_twice_is_harmless(monkeypatch):
    key = generate_encryption_key()
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", f"{key},{key}")
    assert decrypt_secret(encrypt_secret("ABC")) == "ABC"


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-key",
        "short",
        base64.urlsafe_b64encode(b"x" * 31).decode(),  # decodes, but not 32 bytes
        base64.urlsafe_b64encode(b"x" * 33).decode(),
        "é" * 44,
    ],
)
def test_a_bad_key_is_a_configuration_error_naming_its_position_not_its_value(monkeypatch, bad):
    good = generate_encryption_key()
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", f"{good},{bad}")
    with pytest.raises(RuntimeError) as caught:
        encrypt_secret("ABC")
    message = str(caught.value)
    assert "AUTH_ENCRYPTION_KEYS" in message and "entry 2" in message
    assert bad not in message and good not in message
    # Not cached: the next call fails again rather than using a half-loaded set of keys.
    with pytest.raises(RuntimeError):
        encrypt_secret("ABC")


def test_cache_is_kept_until_reset(monkeypatch):
    k1, k2 = generate_encryption_key(), generate_encryption_key()
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", k1)
    first = load_cipher()
    assert load_cipher() is first
    monkeypatch.setattr(settings, "AUTH_ENCRYPTION_KEYS", k2)
    assert load_cipher() is first  # read once per process
    reset_encryption_cache()
    assert load_cipher() is not first
    assert decrypt_secret(Fernet(k2.encode()).encrypt(b"ABC").decode()) == "ABC"


def test_default_setting_is_empty():
    assert type(settings).model_fields["AUTH_ENCRYPTION_KEYS"].default == ""
