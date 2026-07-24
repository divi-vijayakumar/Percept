"""Tests for the immutable judge-context store (PERCEPT-SDK-P06-001).

Exercises the fail-closed contract: exact-bytes verification, digest normalization, size/decode
guards, path safety, and the immutability guarantee (a resolved artifact is retained in memory and
is unaffected by later mutation of the underlying file).
"""
from __future__ import annotations

import hashlib
import os

import pytest

from percept.judge_context import (
    DEFAULT_MAX_BYTES,
    JUDGE_STATIC_CONTEXT_KEY,
    JudgeContextDecodeError,
    JudgeContextError,
    JudgeContextHashMismatch,
    JudgeContextNotFound,
    JudgeContextOversize,
    JudgeContextStore,
    JudgeContextStoreError,
    ReservedContextKeyError,
    ResolvedJudgeContext,
)


def _write(root, name: str, data: bytes) -> str:
    (root / name).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_happy_path_bytes_text_and_sha(tmp_path):
    payload = "grade strictly against the rubric".encode("utf-8")
    digest = _write(tmp_path, "rubric.txt", payload)
    store = JudgeContextStore.from_dir(tmp_path)

    resolved = store.resolve("rubric.txt", digest)

    assert isinstance(resolved, ResolvedJudgeContext)
    assert resolved.ref == "rubric.txt"
    assert resolved.sha256 == digest
    assert resolved.content_bytes == payload
    assert resolved.decoded_text == "grade strictly against the rubric"
    assert resolved.media_type == "text/plain; charset=utf-8"


def test_sha_computed_over_exact_bytes(tmp_path):
    payload = b"exact bytes only \xe2\x9c\x93"  # includes a multibyte char
    digest = _write(tmp_path, "art", payload)
    store = JudgeContextStore.from_dir(tmp_path)

    resolved = store.resolve("art", digest)

    assert hashlib.sha256(resolved.content_bytes).hexdigest() == resolved.sha256


def test_expected_digest_with_sha256_prefix_normalizes(tmp_path):
    digest = _write(tmp_path, "a", b"hello")
    store = JudgeContextStore.from_dir(tmp_path)

    resolved = store.resolve("a", f"sha256:{digest}")

    assert resolved.sha256 == digest


def test_expected_digest_uppercase_normalizes(tmp_path):
    digest = _write(tmp_path, "a", b"hello")
    store = JudgeContextStore.from_dir(tmp_path)

    resolved = store.resolve("a", digest.upper())

    assert resolved.sha256 == digest  # returned digest is bare lowercase


def test_expected_digest_prefixed_and_uppercase_normalizes(tmp_path):
    digest = _write(tmp_path, "a", b"hello")
    store = JudgeContextStore.from_dir(tmp_path)

    resolved = store.resolve("a", f"SHA256:{digest.upper()}")

    assert resolved.sha256 == digest


def test_hash_mismatch_reports_both_digests(tmp_path):
    _write(tmp_path, "a", b"hello")
    store = JudgeContextStore.from_dir(tmp_path)
    wrong = "0" * 64

    with pytest.raises(JudgeContextHashMismatch) as exc:
        store.resolve("a", wrong)

    actual = hashlib.sha256(b"hello").hexdigest()
    msg = str(exc.value)
    assert wrong in msg
    assert actual in msg


def test_oversize_raises(tmp_path):
    _write(tmp_path, "big", b"0123456789")
    store = JudgeContextStore.from_dir(tmp_path, max_bytes=4)

    with pytest.raises(JudgeContextOversize):
        # digest never gets checked — size guard fires first
        store.resolve("big", "0" * 64)


def test_at_limit_is_allowed(tmp_path):
    payload = b"1234"
    digest = _write(tmp_path, "exact", payload)
    store = JudgeContextStore.from_dir(tmp_path, max_bytes=4)

    resolved = store.resolve("exact", digest)

    assert resolved.content_bytes == payload


def test_invalid_utf8_raises_decode_error(tmp_path):
    payload = b"\xff\xfe\x00bad"
    digest = _write(tmp_path, "binary", payload)
    store = JudgeContextStore.from_dir(tmp_path)

    with pytest.raises(JudgeContextDecodeError):
        store.resolve("binary", digest)


def test_missing_ref_raises_not_found(tmp_path):
    store = JudgeContextStore.from_dir(tmp_path)

    with pytest.raises(JudgeContextNotFound):
        store.resolve("absent", "0" * 64)


@pytest.mark.parametrize("ref", ["../x", "a/b", "..", "", "a\\b", "foo/../bar", "bad ref", "sha:*"])
def test_bad_ref_grammar_raises_store_error(tmp_path, ref):
    store = JudgeContextStore.from_dir(tmp_path)

    with pytest.raises(JudgeContextStoreError):
        store.resolve(ref, "0" * 64)


def test_symlink_ref_raises_store_error(tmp_path):
    target = tmp_path / "real.txt"
    target.write_bytes(b"secret")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    store = JudgeContextStore.from_dir(tmp_path)

    with pytest.raises(JudgeContextStoreError):
        store.resolve("link.txt", hashlib.sha256(b"secret").hexdigest())


def test_symlink_escaping_root_raises_store_error(tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_bytes(b"escape")
    root = tmp_path / "store"
    root.mkdir()
    (root / "link.txt").symlink_to(outside)
    store = JudgeContextStore.from_dir(root)

    with pytest.raises(JudgeContextStoreError):
        store.resolve("link.txt", hashlib.sha256(b"escape").hexdigest())


def test_directory_as_ref_raises_store_error(tmp_path):
    (tmp_path / "adir").mkdir()
    store = JudgeContextStore.from_dir(tmp_path)

    with pytest.raises(JudgeContextStoreError):
        store.resolve("adir", "0" * 64)


def test_immutability_after_mutation(tmp_path):
    payload = b"original context"
    digest = _write(tmp_path, "art", payload)
    store = JudgeContextStore.from_dir(tmp_path)

    resolved = store.resolve("art", digest)

    # Mutate the underlying file after the fact.
    (tmp_path / "art").write_bytes(b"tampered context")

    # The already-resolved artifact is untouched — bytes retained in memory.
    assert resolved.content_bytes == payload
    assert resolved.decoded_text == "original context"
    assert resolved.sha256 == digest

    # And a fresh resolve against the original digest now fails closed.
    with pytest.raises(JudgeContextHashMismatch):
        store.resolve("art", digest)


def test_immutability_after_deletion(tmp_path):
    payload = b"context"
    digest = _write(tmp_path, "art", payload)
    store = JudgeContextStore.from_dir(tmp_path)

    resolved = store.resolve("art", digest)
    os.remove(tmp_path / "art")

    assert resolved.content_bytes == payload
    with pytest.raises(JudgeContextNotFound):
        store.resolve("art", digest)


def test_reserved_context_key_constant_value():
    assert JUDGE_STATIC_CONTEXT_KEY == "judge_static_context"


def test_default_max_bytes_value():
    assert DEFAULT_MAX_BYTES == 1_048_576


def test_error_taxonomy_inheritance():
    for subclass in (
        JudgeContextNotFound,
        JudgeContextHashMismatch,
        JudgeContextOversize,
        JudgeContextDecodeError,
        JudgeContextStoreError,
        ReservedContextKeyError,
    ):
        assert issubclass(subclass, JudgeContextError)
    assert issubclass(JudgeContextError, Exception)


def test_resolved_context_is_frozen(tmp_path):
    digest = _write(tmp_path, "a", b"x")
    store = JudgeContextStore.from_dir(tmp_path)
    resolved = store.resolve("a", digest)

    with pytest.raises((AttributeError, TypeError)):
        resolved.sha256 = "changed"  # type: ignore[misc]
