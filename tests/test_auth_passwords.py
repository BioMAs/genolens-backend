"""Password hashing and policy of the local authentication (no database, no network)."""

import unicodedata
from unittest.mock import MagicMock

import bcrypt
import pytest
from argon2 import PasswordHasher

from app.auth import passwords
from app.auth.passwords import (
    DUMMY_HASH,
    PasswordCheck,
    PasswordPolicyError,
    hash_password,
    validate_new_password,
    verify_password,
)

# What Supabase stored: Go bcrypt, `$2a$`, cost 10, over the raw UTF-8 bytes of the password.
LEGACY = bcrypt.hashpw("Café-été-2026!".encode(), bcrypt.gensalt(10, prefix=b"2a")).decode()


def test_legacy_supabase_hash_verifies_and_upgrades():
    check = verify_password("Café-été-2026!", LEGACY)
    assert check.ok and check.upgraded_hash.startswith("$argon2id$")
    assert verify_password("Café-été-2026!", check.upgraded_hash) == PasswordCheck(True, None)


@pytest.mark.parametrize("prefix", [b"2a", b"2b"])
def test_bcrypt_prefixes_are_upgraded(prefix):
    stored = bcrypt.hashpw(b"s3cret-passphrase", bcrypt.gensalt(4, prefix=prefix)).decode()
    check = verify_password("s3cret-passphrase", stored)
    assert check.ok and check.upgraded_hash.startswith("$argon2id$")


def test_bcrypt_2y_prefix_is_upgraded():
    stored = bcrypt.hashpw(b"s3cret-passphrase", bcrypt.gensalt(4)).decode()
    stored = "$2y$" + stored[4:]
    check = verify_password("s3cret-passphrase", stored)
    assert check.ok and check.upgraded_hash.startswith("$argon2id$")


def test_wrong_or_oversized_password_never_raises():  # Review Focus 1
    assert not verify_password("x" * 200, LEGACY).ok
    assert not verify_password("wrong", LEGACY).ok
    assert not verify_password("anything", None).ok
    assert not verify_password("anything", "not-a-hash").ok


def test_wrong_password_offers_no_upgrade():
    assert verify_password("wrong", LEGACY) == PasswordCheck(False, None)
    assert verify_password("wrong", hash_password("right-passphrase")) == PasswordCheck(False, None)


def test_malformed_bcrypt_hash_never_raises():
    # Short or truncated `$2a$` strings make the bcrypt extension panic (a BaseException that an
    # `except Exception` does not catch), so the format has to be checked before bcrypt sees it.
    for broken in ("$2a$10$short", "$2a$", "$2b$10$", "$2y$99$" + "a" * 53, "$2a$10$" + "é" * 53):
        assert verify_password("anything", broken) == PasswordCheck(False, None)


def test_bcrypt_never_truncates_to_72_bytes():
    # bcrypt only reads the first 72 bytes; accepting "<72 bytes>anything" for the hash of
    # "<72 bytes>" would turn a long password into a shorter one.
    stored = bcrypt.hashpw(b"x" * 72, bcrypt.gensalt(4, prefix=b"2a")).decode()
    assert verify_password("x" * 72, stored).ok
    assert not verify_password("x" * 72 + "y", stored).ok
    assert not verify_password("x" * 200, stored).ok


def test_oversized_password_still_pays_for_the_bcrypt_check(monkeypatch):
    # Refusing a >72-byte password before running bcrypt would answer faster than the other
    # attempts, and so tell an account with an inherited hash from one without.
    spy = MagicMock(wraps=bcrypt.checkpw)
    monkeypatch.setattr(passwords.bcrypt, "checkpw", spy)
    assert not verify_password("x" * 200, LEGACY).ok
    spy.assert_called_once()


def test_password_is_not_unicode_normalised():
    nfc = unicodedata.normalize("NFC", "Café-été-2026!")
    nfd = unicodedata.normalize("NFD", "Café-été-2026!")
    assert nfc != nfd
    stored = bcrypt.hashpw(nfc.encode(), bcrypt.gensalt(4, prefix=b"2a")).decode()
    assert verify_password(nfc, stored).ok
    assert not verify_password(nfd, stored).ok
    argon = hash_password(nfc)
    assert verify_password(nfc, argon).ok
    assert not verify_password(nfd, argon).ok


def test_hash_password_is_argon2id_and_salted():
    first, second = hash_password("correct horse"), hash_password("correct horse")
    assert first.startswith("$argon2id$")
    assert first != second
    assert verify_password("correct horse", first) == PasswordCheck(True, None)


def test_argon2_hash_with_outdated_parameters_is_rehashed():
    weak = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash("correct horse")
    check = verify_password("correct horse", weak)
    assert check.ok and check.upgraded_hash.startswith("$argon2id$")
    assert check.upgraded_hash != weak
    assert verify_password("correct horse", check.upgraded_hash) == PasswordCheck(True, None)


def test_missing_hash_is_verified_against_the_dummy_hash(monkeypatch):
    assert DUMMY_HASH.startswith("$argon2id$")
    # PasswordHasher uses __slots__, so its methods cannot be patched: wrap the whole instance.
    spy = MagicMock(wraps=passwords._hasher)
    monkeypatch.setattr(passwords, "_hasher", spy)
    assert verify_password("anything", None) == PasswordCheck(False, None)
    spy.verify.assert_called_once_with(DUMMY_HASH, "anything")


@pytest.mark.parametrize(
    "pw,code",
    [
        ("short", "too_short"),
        ("x" * 129, "too_long"),
        ("unbelievable", "too_common"),  # 12 characters, in the embedded 10k list
        ("me@example.test", "same_as_email"),
    ],
)
def test_policy(pw, code):
    with pytest.raises(PasswordPolicyError) as e:
        validate_new_password(pw, "me@example.test")
    assert e.value.code == code


def test_policy_accepts_long_passphrase():
    validate_new_password("correct horse battery staple", "me@example.test")


def test_policy_length_bounds_are_inclusive():
    validate_new_password("a1b2c3d4e5f6", "me@example.test")  # 12
    validate_new_password("a1" * 64, "me@example.test")  # 128
    with pytest.raises(PasswordPolicyError) as short:
        validate_new_password("a1b2c3d4e5f", "me@example.test")  # 11
    assert short.value.code == "too_short"


def test_policy_common_password_check_ignores_case():
    with pytest.raises(PasswordPolicyError) as e:
        validate_new_password("UnBelievable", "me@example.test")
    assert e.value.code == "too_common"


def test_policy_email_check_ignores_case():
    with pytest.raises(PasswordPolicyError) as e:
        validate_new_password("Me@Example.TEST", "me@example.test")
    assert e.value.code == "same_as_email"


def test_policy_checks_length_before_email_and_common_list():
    # The e-mail and "password" are both refused for their length first.
    with pytest.raises(PasswordPolicyError) as e:
        validate_new_password("me@x.test", "me@x.test")
    assert e.value.code == "too_short"
    with pytest.raises(PasswordPolicyError) as e:
        validate_new_password("password", "me@example.test")
    assert e.value.code == "too_short"


def test_policy_error_is_a_value_error_carrying_its_code():
    error = PasswordPolicyError("too_short")
    assert isinstance(error, ValueError)
    assert error.code == "too_short"


def test_common_password_list_ships_with_the_package_and_is_loaded_once():
    common = passwords._common_passwords()
    assert isinstance(common, frozenset)
    assert 9_000 <= len(common) <= 11_000
    assert "unbelievable" in common
    assert passwords._common_passwords() is common
