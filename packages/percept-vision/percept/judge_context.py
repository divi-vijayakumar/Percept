"""Immutable judge-context binding — a static context artifact bound by ref + SHA-256.

The judge (the LLM-as-evaluator step) may be handed a *static context* artifact: a fixed
blob of bytes — a rubric, a policy, a reference sheet — that must be identical every time it is
verified. This module is the store that resolves such an artifact and proves it, byte-for-byte,
against a caller-supplied digest.

The contract, and the whole point of the module:

* **exact-bytes verification** — the bytes are read once, the SHA-256 is computed over *those*
  exact bytes, and the returned artifact retains both the verified bytes and the text decoded from
  them. We never hash one representation and return another.
* **fail-closed** — any deviation (missing artifact, digest mismatch, oversize, bad decode, unsafe
  path) raises; there is no partial or best-effort result.
* **no live retrieval** — the store is a local, path-safe reader over a single directory. It never
  touches the network, ever; the artifact is static bytes on disk.

Deliberately self-contained: the public API takes primitive strings only and imports nothing from
``percept.contracts``. Pure stdlib.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

#: Reserved key under which verified static context is injected into a judge call's context.
#: A caller-supplied context that already contains this key is a bug — see
#: :class:`ReservedContextKeyError`.
JUDGE_STATIC_CONTEXT_KEY = "judge_static_context"

#: Reserved key under which the concern's currently OPEN episode is injected into a judge call's
#: context. Its value is the bounded, pixel-free text built by
#: :func:`percept.episode_memory.episode_context_text`, so the judge is told what it already
#: established about this episode instead of re-deciding it from one frame with no memory.
#: Like :data:`JUDGE_STATIC_CONTEXT_KEY`, a caller-supplied context already carrying this key is a
#: bug — see :class:`ReservedContextKeyError`.
JUDGE_OPEN_EPISODE_KEY = "judge_open_episode"

#: Default upper bound on artifact size. Static judge context is small by design; a large blob is a
#: configuration mistake, not a valid artifact.
DEFAULT_MAX_BYTES = 1_048_576

# Ref grammar: a bare filename token — no path separators, no "..", non-empty. Anything else is a
# store-config error, never a lookup miss.
_ALLOWED_REF_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._@-"
)


class JudgeContextError(Exception):
    """Base for every judge-context failure. Callers may catch this to fail closed uniformly."""


class JudgeContextNotFound(JudgeContextError):
    """The ref names a well-formed artifact that does not exist in the store."""


class JudgeContextHashMismatch(JudgeContextError):
    """The artifact's actual SHA-256 does not match the expected digest. Reports both digests."""


class JudgeContextOversize(JudgeContextError):
    """The artifact exceeds the store's ``max_bytes`` bound."""


class JudgeContextDecodeError(JudgeContextError):
    """The verified bytes are not valid UTF-8; the artifact cannot be presented as text."""


class JudgeContextStoreError(JudgeContextError):
    """A store-configuration problem: bad ref grammar, traversal, symlink, or non-regular file."""


class ReservedContextKeyError(JudgeContextError):
    """A caller's context already contains one of this module's reserved keys
    (:data:`JUDGE_STATIC_CONTEXT_KEY`, :data:`JUDGE_OPEN_EPISODE_KEY`).

    Defined here so the reserved keys and the error that guards them have a single home; raised by
    the later injection step, not by this module's store.
    """


@dataclass(frozen=True)
class ResolvedJudgeContext:
    """A verified static context artifact.

    Retains both the exact ``content_bytes`` the digest was computed over and the ``decoded_text``
    derived from them. Frozen and self-contained — unaffected by any later change to the file it
    came from.
    """

    ref: str
    sha256: str  # bare 64-hex lowercase
    content_bytes: bytes
    decoded_text: str
    media_type: str


def _normalize_expected_sha256(expected: str) -> str:
    """Strip an optional ``sha256:`` prefix and lowercase. Returns the bare hex for comparison."""
    value = expected.strip()
    if value.lower().startswith("sha256:"):
        value = value[len("sha256:") :]
    return value.strip().lower()


class JudgeContextStore:
    """Local, path-safe, static-bytes-only artifact store. No network, ever.

    Construct via :meth:`from_dir`. Each :meth:`resolve` maps a bare ref to a file directly under the
    store root, verifies its bytes against an expected digest, and returns an immutable
    :class:`ResolvedJudgeContext`.
    """

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._root = root
        self._max_bytes = max_bytes

    @classmethod
    def from_dir(cls, root: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> "JudgeContextStore":
        """Open a store rooted at ``root`` (resolved to an absolute, symlink-free canonical path)."""
        return cls(Path(root).resolve(), max_bytes=max_bytes)

    def _validate_ref(self, ref: str) -> None:
        """Enforce the ref grammar. Raises :class:`JudgeContextStoreError` on any violation."""
        if not ref:
            raise JudgeContextStoreError("empty ref")
        if ref == ".." or "/" in ref or "\\" in ref or os.sep in ref:
            raise JudgeContextStoreError(f"ref contains a path separator: {ref!r}")
        if not set(ref) <= _ALLOWED_REF_CHARS:
            raise JudgeContextStoreError(f"ref has characters outside the allowed grammar: {ref!r}")

    def resolve(
        self,
        ref: str,
        expected_sha256: str,
        *,
        media_type: str = "text/plain; charset=utf-8",
    ) -> ResolvedJudgeContext:
        """Resolve, verify, and return a static context artifact — fail-closed at every step.

        Order is exact and load-bearing: validate the ref grammar → resolve the path and reject any
        unsafe target (symlink, escape, non-regular file) → read the bytes ONCE → enforce
        ``max_bytes`` → compute SHA-256 over those exact bytes → compare against the normalized
        expected digest → strict UTF-8 decode → return the artifact retaining the verified bytes and
        the decoded text.
        """
        self._validate_ref(ref)

        candidate = self._root / ref

        # Path safety, all against the *lexical* candidate (never following a symlink):
        # 1. the target must not itself be a symlink;
        # 2. its real path must stay within the store root;
        # 3. it must be a regular file (not a directory, socket, device, ...).
        try:
            lst = os.lstat(candidate)
        except FileNotFoundError:
            raise JudgeContextNotFound(f"no such judge-context artifact: {ref!r}") from None
        except OSError as exc:
            raise JudgeContextStoreError(f"cannot stat judge-context artifact {ref!r}: {exc}") from exc

        if os.path.islink(candidate):
            raise JudgeContextStoreError(f"judge-context ref is a symlink: {ref!r}")

        real = candidate.resolve()
        if real != self._root and self._root not in real.parents:
            raise JudgeContextStoreError(f"judge-context ref escapes the store root: {ref!r}")

        if not stat.S_ISREG(lst.st_mode):
            raise JudgeContextStoreError(f"judge-context ref is not a regular file: {ref!r}")

        # Read the exact bytes ONCE — everything downstream verifies and returns *these* bytes.
        try:
            content_bytes = candidate.read_bytes()
        except FileNotFoundError:
            raise JudgeContextNotFound(f"no such judge-context artifact: {ref!r}") from None
        except OSError as exc:
            raise JudgeContextStoreError(f"cannot read judge-context artifact {ref!r}: {exc}") from exc

        if len(content_bytes) > self._max_bytes:
            raise JudgeContextOversize(
                f"judge-context artifact {ref!r} is {len(content_bytes)} bytes, "
                f"over the {self._max_bytes}-byte limit"
            )

        actual_sha256 = hashlib.sha256(content_bytes).hexdigest()
        expected = _normalize_expected_sha256(expected_sha256)
        if actual_sha256 != expected:
            raise JudgeContextHashMismatch(
                f"judge-context artifact {ref!r} hash mismatch: "
                f"expected {expected}, got {actual_sha256}"
            )

        try:
            decoded_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JudgeContextDecodeError(
                f"judge-context artifact {ref!r} is not valid UTF-8: {exc}"
            ) from exc

        return ResolvedJudgeContext(
            ref=ref,
            sha256=actual_sha256,
            content_bytes=content_bytes,
            decoded_text=decoded_text,
            media_type=media_type,
        )
