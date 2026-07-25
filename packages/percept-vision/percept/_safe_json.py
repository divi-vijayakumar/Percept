"""Private, bounded JSON input for local runtime and evaluation artifacts.

This module is deliberately not part of :mod:`percept`'s public API.  It turns a path into a
stable descriptor-backed snapshot before parsing JSON:

* only a regular file with exactly one link is accepted;
* the final path may not be a symlink;
* opening is non-blocking and uses ``O_NOFOLLOW`` where the platform provides it;
* descriptor and path identity/metadata must agree before and after the bounded read; and
* UTF-8 is strict and duplicate object keys are rejected at every nesting depth.

The default 8 MiB limit is intentionally conservative for control-plane manifests and evidence
receipts.  Large media and bulk datasets must be streamed by their purpose-built readers instead
of being accidentally materialized through this helper.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAFE_JSON_MAX_BYTES = 8 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


class SafeJSONInputError(ValueError):
    """A controlled local-input failure.

    ``phase`` lets a caller preserve its own public distinction between filesystem/encoding
    failures and JSON syntax failures without exposing this private exception type.
    """

    def __init__(self, message: str, *, phase: str, code: str = "invalid") -> None:
        super().__init__(message)
        self.phase = phase
        self.code = code


@dataclass(frozen=True)
class SafeInputReceipt:
    """Identity and content receipt for the snapshot that was actually parsed."""

    path: str
    canonical_path: str
    byte_count: int
    sha256: str
    device: int
    inode: int
    modified_time_ns: int


@dataclass(frozen=True)
class SafeJSONResult:
    """A parsed value paired with the descriptor-backed input receipt."""

    value: Any
    receipt: SafeInputReceipt


def _input_error(message: str, *, code: str = "invalid") -> SafeJSONInputError:
    return SafeJSONInputError(message, phase="input", code=code)


def _parse_error(message: str) -> SafeJSONInputError:
    return SafeJSONInputError(message, phase="parse")


def _path_text(path: str | Path) -> tuple[str, str]:
    try:
        raw = os.fspath(path)
    except (TypeError, ValueError) as exc:
        raise _input_error(f"invalid input path: {exc}") from exc
    if not isinstance(raw, str):
        raise _input_error("input path must resolve to text")
    return raw, raw


def _lstat(path: str, *, stage: str) -> os.stat_result:
    try:
        return os.lstat(path)
    except (OSError, ValueError) as exc:
        raise _input_error(f"{stage} path inspection failed: {exc}") from exc


def _canonical_path(path: str, *, stage: str) -> str:
    try:
        return os.path.realpath(os.path.abspath(path), strict=True)
    except (OSError, ValueError) as exc:
        raise _input_error(f"{stage} canonical-path resolution failed: {exc}") from exc


def _validate_regular_file(
    info: os.stat_result,
    *,
    stage: str,
    require_single_link: bool = True,
) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise _input_error(f"{stage} input is not a regular file")
    if require_single_link and info.st_nlink != 1:
        raise _input_error(f"{stage} input must have exactly one hard link")


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _snapshot(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _open_flags() -> int:
    flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_NONBLOCK", "O_NOFOLLOW"):
        flags |= int(getattr(os, name, 0))
    return flags


def _read_bounded(fd: int, *, max_bytes: int) -> bytes:
    data = bytearray()
    while True:
        remaining = max_bytes + 1 - len(data)
        if remaining <= 0:
            raise _input_error(f"input exceeds the {max_bytes}-byte limit")
        try:
            chunk = os.read(fd, min(_READ_CHUNK_BYTES, remaining))
        except OSError as exc:
            raise _input_error(f"input read failed: {exc}") from exc
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if len(data) > max_bytes:
            raise _input_error(f"input exceeds the {max_bytes}-byte limit")


def _read_snapshot(
    path: str | Path,
    *,
    max_bytes: int,
) -> tuple[bytes, SafeInputReceipt]:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise _input_error("max_bytes must be a positive integer")

    raw_path, display_path = _path_text(path)
    before_path = _lstat(raw_path, stage="initial")
    _validate_regular_file(before_path, stage="initial")
    canonical_path = _canonical_path(raw_path, stage="initial")
    before_canonical = _lstat(canonical_path, stage="initial canonical")
    _validate_regular_file(before_canonical, stage="initial canonical")
    if _identity(before_path) != _identity(before_canonical):
        raise _input_error("input canonical path does not identify the requested file")
    if before_path.st_size > max_bytes:
        raise _input_error(f"input exceeds the {max_bytes}-byte limit")

    try:
        fd = os.open(raw_path, _open_flags())
    except (OSError, ValueError) as exc:
        raise _input_error(f"input open failed: {exc}") from exc

    try:
        try:
            before_fd = os.fstat(fd)
        except OSError as exc:
            raise _input_error(f"opened input inspection failed: {exc}") from exc
        _validate_regular_file(before_fd, stage="opened")
        if before_fd.st_size > max_bytes:
            raise _input_error(f"input exceeds the {max_bytes}-byte limit")

        after_open_path = _lstat(raw_path, stage="post-open")
        _validate_regular_file(after_open_path, stage="post-open")
        after_open_canonical_path = _canonical_path(raw_path, stage="post-open")
        after_open_canonical = _lstat(after_open_canonical_path, stage="post-open canonical")
        _validate_regular_file(after_open_canonical, stage="post-open canonical")
        if not (
            _identity(before_path)
            == _identity(before_fd)
            == _identity(after_open_path)
            == _identity(after_open_canonical)
        ):
            raise _input_error("input path was replaced while opening")
        if canonical_path != after_open_canonical_path:
            raise _input_error("input canonical path changed while opening")
        if _snapshot(before_path) != _snapshot(before_fd):
            raise _input_error("input changed while opening")
        if _snapshot(before_fd) != _snapshot(after_open_path):
            raise _input_error("input path changed while opening")

        payload = _read_bounded(fd, max_bytes=max_bytes)
        if not payload:
            raise _input_error("input is empty", code="empty")

        try:
            after_fd = os.fstat(fd)
        except OSError as exc:
            raise _input_error(f"post-read input inspection failed: {exc}") from exc
        # A rename/unlink can reduce this still-open descriptor's link count to zero.  Inspect the
        # final path first so that removal/replacement becomes the controlled, accurate failure;
        # the path and full snapshot checks below still reject any hard link added during the read.
        _validate_regular_file(after_fd, stage="post-read", require_single_link=False)
        after_read_path = _lstat(raw_path, stage="post-read")
        _validate_regular_file(after_read_path, stage="post-read")
        after_read_canonical_path = _canonical_path(raw_path, stage="post-read")
        after_read_canonical = _lstat(after_read_canonical_path, stage="post-read canonical")
        _validate_regular_file(after_read_canonical, stage="post-read canonical")
        if _identity(after_fd) != _identity(after_read_path):
            raise _input_error("input path was replaced while it was being read")
        if canonical_path != after_read_canonical_path:
            raise _input_error("input canonical path changed while it was being read")
        if _identity(after_fd) != _identity(after_read_canonical):
            raise _input_error("input canonical path was replaced while it was being read")
        if _snapshot(before_fd) != _snapshot(after_fd):
            raise _input_error("input changed while it was being read")
        if _snapshot(after_fd) != _snapshot(after_read_path):
            raise _input_error("input path changed while it was being read")
        if _snapshot(after_fd) != _snapshot(after_read_canonical):
            raise _input_error("input canonical path changed while it was being read")
        if len(payload) != after_fd.st_size:
            raise _input_error("input length changed while it was being read")
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    return payload, SafeInputReceipt(
        path=display_path,
        canonical_path=canonical_path,
        byte_count=len(payload),
        sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        device=after_fd.st_dev,
        inode=after_fd.st_ino,
        modified_time_ns=after_fd.st_mtime_ns,
    )


def _decode_utf8(payload: bytes) -> str:
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _input_error(f"input is not valid UTF-8: {exc}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _parse_error(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> Any:
    raise _parse_error(f"non-finite JSON number {value!r} is unsupported")


def _parse_json(text: str, *, context: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_finite,
        )
    except SafeJSONInputError:
        raise
    except (ValueError, RecursionError) as exc:
        raise _parse_error(f"invalid JSON in {context}: {exc}") from exc


def load_json_file(
    path: str | Path,
    *,
    max_bytes: int = SAFE_JSON_MAX_BYTES,
) -> SafeJSONResult:
    """Read one bounded, stable UTF-8 JSON document."""

    payload, receipt = _read_snapshot(path, max_bytes=max_bytes)
    return SafeJSONResult(_parse_json(_decode_utf8(payload), context="document"), receipt)


def load_jsonl_file(
    path: str | Path,
    *,
    max_bytes: int = SAFE_JSON_MAX_BYTES,
) -> SafeJSONResult:
    """Read bounded UTF-8 JSONL, rejecting duplicate keys within every row."""

    payload, receipt = _read_snapshot(path, max_bytes=max_bytes)
    text = _decode_utf8(payload)
    rows: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            rows.append(_parse_json(line, context=f"JSONL row {line_number}"))
    return SafeJSONResult(rows, receipt)
