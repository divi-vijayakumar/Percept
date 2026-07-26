"""Matched-Watch execution receipts over synthetic fixtures or frozen real video.

Both arms execute the shipped resolver/compiler/runtime path.  Real-video manifests bind a closed
MP4, corpus index, and offline oracle to exact decoded evidence coordinates; the oracle proves
mechanics only.  No receipt produced here is provider-measured or launch-qualifying.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field as dc_field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from percept import Goal, Percept, build_info
from percept._safe_json import (
    SAFE_JSON_MAX_BYTES,
    SafeInputReceipt,
    SafeJSONInputError,
    SafeJSONResult,
    _decode_utf8,
    _parse_json,
    load_json_file,
)
from percept.adapters import Verdict
from percept.config import Settings
from percept.contracts.watchplan import WatchPlan
from percept.domain_pack import DomainPack, get_domain_pack, watch_template
from percept.harness.tier0 import gray_frame
from percept.packs_default import FACILITY_OBSERVATION_V1
from percept.resolver import ConfigResolver
from percept.runtime.envelope import SourceEnvelope
from percept.runtime.worker import RuntimeHealthEvent, RuntimeStats, run
from percept.sourceprofile import SourceProfile
from percept.contracts import judgment_event_id
from percept.tracing import PlanIdentity, content_hash
from percept.tunables import REGISTRY
from percept.watchplan_apply import apply_watch_plan

__all__ = [
    "MatchedWatchInputError",
    "ProviderCallReceipt",
    "CandidateReceipt",
    "ArmReceipt",
    "MatchedWatchReceipt",
    "LoadedMatchedWatchReceipt",
    "load_matched_watch_manifest",
    "load_matched_watch_receipt",
    "run_matched_watch",
    "run_matched_watch_async",
    "validate_matched_arms",
    "derive_viability_rows",
    "main",
]

_SCHEMA = "percept.eval.matched_watch/v1"
_REAL_SCHEMA = "percept.eval.matched_watch.real_video/v1"
_MEDIA_SCHEMA = "percept.eval.matched_watch.synthetic_video/v1"
_RECEIPT_SCHEMA = "percept.eval.matched_watch.receipt/v1"
_RECEIPT_SCHEMA_V2 = "percept.eval.matched_watch.receipt/v2"
_CORPUS_SCHEMA = "percept.eval.matched_watch.corpus/v1"
_ORACLE_SCHEMA = "percept.eval.matched_watch.offline_oracle/v1"
_ARCHETYPE = "physical_scene/package_appeared"
_TEMPLATE = "facility_observation@1/package_appeared"
_PACK = "facility_observation@1"
_EVIDENCE_CLASSES = ("scripted_contract", "admission_only")
_SHA_PATHS = {
    "Settings.edge.motion_gate",
    "Settings.edge.motion_gate_block_max",
    "Settings.edge.motion_gate_p95",
    "SourceProfile.parameters.salience_threshold",
}

_MANIFEST_KEYS = {
    "schema",
    "case_id",
    "archetype",
    "split",
    "media",
    "replay",
    "watch",
    "judge",
    "adaptive",
    "events",
    "evidence_classification",
}
_REAL_MANIFEST_KEYS = _MANIFEST_KEYS | {"corpus", "oracle"}
_MEDIA_REF_KEYS = {"path", "sha256"}
_REAL_MEDIA_REF_KEYS = {"path", "sha256", "media_type"}
_ARTIFACT_REF_KEYS = {"path", "sha256"}
_REPLAY_KEYS = {"fps", "timebase", "source_id"}
_WATCH_KEYS = {"template_ref", "parameters", "domain_pack"}
_JUDGE_KEYS = {"adapter", "model", "prompt_sha256"}
_REAL_JUDGE_KEYS = _JUDGE_KEYS | {"judge_mode"}
_ADAPTIVE_KEYS = {"operating_point", "salience_stat", "threshold"}
_EVENT_KEYS = {
    "id",
    "eligible_at_s",
    "window_start_s",
    "window_end_s",
    "review_authority",
    "label_digest",
}
_MEDIA_KEYS = {"schema", "width", "height", "frames"}
_FRAME_KEYS = {"t", "fill", "token", "from", "to"}
_VERDICT_KEYS = {"satisfied", "confidence", "evidence", "error"}
_CORPUS_KEYS = {"schema", "corpus_id", "version", "cases", "authorities"}
_CORPUS_CASE_KEYS = {
    "case_id",
    "split",
    "source_group_id",
    "media_sha256",
    "oracle_sha256",
}
_AUTHORITY_NAMES = (
    "license",
    "label_review",
    "provider_transmission",
    "preregistration",
)
_ORACLE_KEYS = {"schema", "candidates"}
_ORACLE_CANDIDATE_KEYS = {"sequence", "t", "frame_sha256", "token", "verdicts"}
_SAFE_INPUT_RECEIPT_KEYS = {
    "path",
    "canonical_path",
    "byte_count",
    "sha256",
    "device",
    "inode",
    "modified_time_ns",
}
_INPUT_ARTIFACT_KEYS = {"role"} | _SAFE_INPUT_RECEIPT_KEYS
_PERSISTED_RECEIPT_KEYS = {
    "schema",
    "manifest",
    "manifest_input_receipt",
    "media_input_receipt",
    "watch_plan",
    "watch_plan_sha256",
    "judge_identity",
    "event_windows",
    "evidence_classification",
    "package_version",
    "build_provenance",
    "baseline",
    "adaptive",
    "comparable",
    "comparability_blockers",
    "launch_qualified",
    "qualification_reason",
    "content_sha256",
    "evidence_sha256",
    "origin_evidence_sha256",
    "run_timestamp",
}
_PERSISTED_RECEIPT_V2_KEYS = _PERSISTED_RECEIPT_KEYS | {
    "input_artifacts",
    "authority_artifact_presence",
}
_ARM_RECEIPT_KEYS = {
    "arm",
    "source_profile",
    "resolved_watch",
    "apply_receipt",
    "compiled_binding",
    "judge_identity",
    "media_identity",
    "replay_identity",
    "configuration_sha256",
    "candidate_timestamps",
    "candidates",
    "incidents",
    "asks",
    "traces",
    "health",
    "runtime_stats",
    "runtime_dispatches",
    "provider_call_attempts",
    "provider_call_successes",
    "provider_call_failures",
    "valid",
    "blockers",
}
_CANDIDATE_RECEIPT_KEYS = {
    "index",
    "t",
    "edge_decision",
    "admitted",
    "runtime_dispatched",
    "provider_calls",
    "decision",
    "transition_outcome",
    "incident_ids",
}
_CANDIDATE_RECEIPT_V2_KEYS = _CANDIDATE_RECEIPT_KEYS | {
    "source_sequence",
    "frame_sha256",
}
_PROVIDER_CALL_RECEIPT_KEYS = {
    "sequence",
    "candidate_index",
    "candidate_t",
    "predicate",
    "condition_sha256",
    "status",
    "satisfied",
    "confidence",
    "evidence",
    "error",
}
_PROVIDER_CALL_RECEIPT_V2_KEYS = _PROVIDER_CALL_RECEIPT_KEYS | {
    "oracle_lookup_sha256",
}
_SOURCE_PROFILE_KEYS = {
    "name",
    "modalities",
    "temporal_unit",
    "admission_strategy",
    "evidence_strategy",
    "identity_strategy",
    "default_timebase",
    "parameters",
    "stability",
    "notes",
}
_RESOLVED_WATCH_KEYS = {
    "plan",
    "plan_hash",
    "source_profile",
    "domain_pack",
    "components",
    "effective",
    "provenance",
    "execution_policy",
    "optimization_lock_hash",
    "component_snapshot_hash",
}
_SETTING_ORIGIN_KEYS = {"value", "owner_layer", "source_id", "reason", "validated_by"}
_REGISTERED_TUNABLE_PATHS = frozenset(spec.path for spec in REGISTRY)
_RESOLVED_SOURCE_PROFILE_KEYS = {
    "SourceProfile.name",
    "SourceProfile.modalities",
    "SourceProfile.temporal_unit",
    "SourceProfile.admission_strategy",
    "SourceProfile.evidence_strategy",
    "SourceProfile.identity_strategy",
    "SourceProfile.default_timebase",
    "SourceProfile.stability",
    "SourceProfile.notes",
}
_RESOLVED_COMPONENT_KEYS = {"providers", "catalog_component_hash"}
_RESOLVED_PROVIDER_KEYS = {"vision", "conductor", "specialist", "stt", "tts"}
_APPLY_RECEIPT_KEYS = {
    "status",
    "watch_ids",
    "accepted_fields",
    "coerced_fields",
    "dropped_fields",
    "warnings",
    "fallback_used",
    "judge_requested",
    "judge_resolved",
    "judge_context_ref",
    "judge_context_sha256",
    "plan_revision",
}
_COMPILED_BINDING_KEYS = {
    "goal_id",
    "timebase",
    "coverage_floor",
    "watchspec_kwargs",
    "components",
}
_JUDGE_IDENTITY_KEYS = {"adapter", "model", "prompt_sha256"}
_JUDGE_IDENTITY_V2_KEYS = _JUDGE_IDENTITY_KEYS | {"judge_mode"}
_MEDIA_IDENTITY_KEYS = {"sha256", "byte_count", "schema", "width", "height"}
_MEDIA_IDENTITY_V2_KEYS = _MEDIA_IDENTITY_KEYS | {"decoder_identity"}
_DECODER_IDENTITY_KEYS = {
    "library",
    "version",
    "build_sha256",
    "source_adapter",
    "frame_encoding",
}
_INCIDENT_RECEIPT_KEYS = {
    "id",
    "concern_id",
    "ts",
    "message",
    "entity_id",
    "reason",
    "watchplan_id",
    "judgment_event_id",
    "evidence_origin",
}
_ASK_RECEIPT_KEYS = {"goal_id", "action", "text"}
_TRACE_RECEIPT_KEYS = {"trace_id", "stage", "t", "goal_id", "data"}
_HEALTH_RECEIPT_KEYS = {"kind", "ts", "data"}
_EDGE_DECISION_KEYS = {
    "threshold",
    "motion",
    "structural",
    "semantic",
    "escalate",
    "coast_admit",
    "goal_ids",
    "admission_causes",
}
_TRACE_DATA_KEYS = {
    "edge.decision": _EDGE_DECISION_KEYS,
    "source.frame": set(),
    "transition.decision": {
        "armed",
        "enriched",
        "entity_id",
        "fired_latch",
        "focus_entity_id",
        "from_confidence",
        "from_evidence",
        "from_truth",
        "identity_policy",
        "latch_key",
        "latch_key_source",
        "outcome",
        "strip_len",
        "to_confidence",
        "to_evidence",
        "to_truth",
    },
    "judgment.event": {
        "action",
        "dispatch_seq",
        "entity_id",
        "evidence_origin",
        "judgment_event_id",
    },
    "judge.started": {"temporal", "roi"},
    "judge.finished": {"satisfied", "confidence"},
    "gate.classified": {
        "truth",
        "action",
        "reason",
        "gate_reason",
        "entity_id",
        "satisfied",
        "confidence",
        "freshness",
        "hold_s",
    },
    "shadow.fold": {"subject", "fold"},
    "incident.created": {
        "entity_id",
        "evidence_origin",
        "from_boundary",
        "incident_id",
        "judgment_event_id",
    },
}
_HEALTH_DATA_KEYS = {
    "runtime_started": {"candidate_count", "media", "replay", "source_kind"},
    "judge_dispatch": {"mode"},
    "effective_config": {"coast_enabled", "fps", "video_timebase"},
    "escalation": {"cause", "source"},
    "incident": {"entity_id", "goal_id", "incident_id"},
}
_RUNTIME_ENDED_DATA_KEYS = {
    "admission_by_cause",
    "admission_by_goal",
    "admitted",
    "audio_wakes",
    "coast_admits",
    "fixed_admits",
    "fixed_opportunities",
    "frames_seen",
    "goal_admissions",
    "incidents",
    "judge_calls",
    "perceive_dispatches",
    "signal_admits",
    "source_coverage",
}
_HEALTH_DATA_KEYS["runtime_ended"] = _RUNTIME_ENDED_DATA_KEYS
_HEALTH_DATA_KEYS["zero_admission_coverage"] = _RUNTIME_ENDED_DATA_KEYS | {
    "threshold",
    "stat",
    "coverage_floor",
    "floor_unmet",
}
_BUILD_PROVENANCE_KEYS = {
    "package_version",
    "git_commit",
    "contract_version",
    "build_timestamp",
}
_QUALIFICATION_REASON = (
    "L1 scripted/admission evidence proves execution mechanics only; "
    "provider-measured release qualification is unavailable"
)
_CLI_DESCRIPTION = """Preregistered L1 matched-Watch execution receipts.

This module is deliberately narrow.  It executes the shipped
``facility_observation@1/package_appeared`` transition Watch through the real resolver/compiler and
runtime worker for an exhaustive sampled baseline and one adaptive operating point.  The bundled
judge is scripted and proves contract mechanics only; no receipt produced here is provider-measured
or launch-qualifying.
"""


class MatchedWatchInputError(ValueError):
    """A matched replay input or execution receipt is unsafe, incomplete, or incomparable."""


def _fail(message: str) -> None:
    raise MatchedWatchInputError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    return dict(value)


def _exact(value: Mapping[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    _require(not missing, f"{context} is missing required fields {missing!r}")
    _require(not extra, f"{context} has unsupported fields {extra!r}")


def _text(value: Any, context: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{context} must be non-empty text")
    result = value.strip()
    _require(len(result) <= 4096, f"{context} exceeds the 4096-character limit")
    return result


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{context} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise MatchedWatchInputError(f"{context} must be a finite number") from exc
    _require(math.isfinite(result), f"{context} must be a finite number")
    if minimum is not None:
        _require(result >= minimum, f"{context} must be >= {minimum}")
    return result


def _sha(value: Any, context: str) -> str:
    result = _text(value, context)
    _require(
        result.startswith("sha256:")
        and len(result) == 71
        and all(c in "0123456789abcdef" for c in result[7:]),
        f"{context} must be sha256:<64 lowercase hex>",
    )
    return result


def _canonical_sha(value: Any) -> str:
    return f"sha256:{content_hash(value)}"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _json_shape(value: Any) -> Any:
    return _thaw(_freeze(value))


def _safe_receipt_dict(receipt: SafeInputReceipt) -> dict[str, Any]:
    return {
        "path": receipt.path,
        "canonical_path": receipt.canonical_path,
        "byte_count": receipt.byte_count,
        "sha256": receipt.sha256,
        "device": receipt.device,
        "inode": receipt.inode,
        "modified_time_ns": receipt.modified_time_ns,
    }


def _artifact_receipt(role: str, receipt: SafeInputReceipt) -> Mapping[str, Any]:
    return MappingProxyType({"role": role, **_safe_receipt_dict(receipt)})


def _resolve_contained_path(raw: Any, *, root: Path, context: str) -> Path:
    """Resolve one v2 artifact path without allowing an absolute path or directory escape."""

    text = _text(raw, context)
    candidate = Path(text)
    _require(not candidate.is_absolute(), f"{context} must be relative to its manifest")
    _require(
        all(part not in {"", ".", ".."} for part in candidate.parts),
        f"{context} must use a normalized relative path",
    )
    try:
        canonical_root = root.resolve(strict=True)
        lexical = canonical_root / candidate
        canonical = lexical.resolve(strict=True)
        contained = os.path.commonpath((str(canonical_root), str(canonical))) == str(canonical_root)
        cursor = canonical_root
        for part in candidate.parts:
            cursor = cursor / part
            _require(
                not stat.S_ISLNK(os.lstat(cursor).st_mode),
                f"{context} may not traverse a symlink",
            )
    except (OSError, ValueError) as exc:
        raise MatchedWatchInputError(f"{context} cannot be safely resolved: {exc}") from exc
    _require(contained, f"{context} escapes its manifest directory")
    # Carry the already-inspected canonical location forward.  If an attacker renames one of the
    # lexical parents and replaces it with a symlink after this check, subsequent descriptor opens
    # still target this canonical location (which will either remain the same file or fail closed).
    return Path(canonical)


def _open_artifact_no_symlinks(path: Path) -> int:
    """Open every path component by descriptor so a directory swap cannot escape containment."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    _require(
        absolute.is_absolute()
        and len(parts) >= 2
        and all(part not in {"", ".", ".."} for part in parts[1:]),
        "unsafe binary artifact: path must be normalized and absolute",
    )
    _require(
        os.open in getattr(os, "supports_dir_fd", set())
        and bool(getattr(os, "O_NOFOLLOW", 0))
        and bool(getattr(os, "O_DIRECTORY", 0)),
        "unsafe binary artifact: platform lacks descriptor-relative no-symlink open",
    )
    directory_flags = (
        os.O_RDONLY
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
        | int(getattr(os, "O_DIRECTORY", 0))
    )
    file_flags = (
        os.O_RDONLY
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    directory_fd = -1
    try:
        directory_fd = os.open(parts[0], directory_flags)
        for part in parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(parts[-1], file_flags, dir_fd=directory_fd)
    except OSError as exc:
        raise MatchedWatchInputError(
            f"unsafe binary artifact: symlink-safe open failed: {exc}"
        ) from exc
    finally:
        if directory_fd >= 0:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def _normalize_platform_root_alias(path: Path) -> Path:
    """Normalize only operating-system root aliases before the no-symlink descriptor walk."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    for alias_text in ("/var", "/tmp", "/etc"):
        alias = Path(alias_text)
        try:
            if not alias.is_symlink():
                continue
            relative = absolute.relative_to(alias)
            return alias.resolve(strict=True) / relative
        except (OSError, ValueError):
            continue
    return absolute


def _read_artifact_snapshot(
    path: Path,
    *,
    max_bytes: int,
    capture: bool,
    sink_fd: int | None = None,
) -> tuple[bytes | None, SafeInputReceipt]:
    """Read one stable file through a symlink-safe descriptor walk."""

    _require(
        isinstance(max_bytes, int)
        and not isinstance(max_bytes, bool)
        and max_bytes > 0,
        "unsafe binary artifact: max_bytes must be positive",
    )
    display_path = str(path)
    normalized_path = _normalize_platform_root_alias(path)
    try:
        initial_canonical = str(normalized_path.resolve(strict=True))
    except OSError as exc:
        raise MatchedWatchInputError(
            f"unsafe binary artifact: initial canonical inspection failed: {exc}"
        ) from exc
    # macOS exposes ordinary paths such as /var through /private/var. Only that operating-system
    # root alias is normalized; every caller-controlled component remains subject to O_NOFOLLOW.
    fd = _open_artifact_no_symlinks(normalized_path)
    try:
        try:
            opened = os.fstat(fd)
        except OSError as exc:
            raise MatchedWatchInputError(
                f"unsafe binary artifact: opened inspection failed: {exc}"
            ) from exc
        _require(
            stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1,
            "unsafe binary artifact: input is not a single-link regular file",
        )
        _require(
            0 < opened.st_size <= max_bytes,
            "unsafe binary artifact: invalid or excessive byte size",
        )
        before_snapshot = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
        )
        digest = hashlib.sha256()
        payload = bytearray() if capture else None
        byte_count = 0
        while True:
            try:
                chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - byte_count))
            except OSError as exc:
                raise MatchedWatchInputError(
                    f"unsafe binary artifact: read failed: {exc}"
                ) from exc
            if not chunk:
                break
            byte_count += len(chunk)
            _require(byte_count <= max_bytes, "unsafe binary artifact exceeds byte limit")
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
            if sink_fd is not None:
                unwritten = memoryview(chunk)
                while unwritten:
                    try:
                        written = os.write(sink_fd, unwritten)
                    except OSError as exc:
                        raise MatchedWatchInputError(
                            f"unsafe binary artifact: private snapshot write failed: {exc}"
                        ) from exc
                    _require(written > 0, "unsafe binary artifact: private snapshot write stalled")
                    unwritten = unwritten[written:]
        try:
            after = os.fstat(fd)
        except OSError as exc:
            raise MatchedWatchInputError(
                f"unsafe binary artifact: post-read inspection failed: {exc}"
            ) from exc
        after_snapshot = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        try:
            final_path = os.lstat(normalized_path)
            final_canonical = str(normalized_path.resolve(strict=True))
            canonical_info = os.lstat(final_canonical)
        except OSError as exc:
            raise MatchedWatchInputError(
                f"unsafe binary artifact: post-read inspection failed: {exc}"
            ) from exc
        _require(
            before_snapshot == after_snapshot
            and (final_path.st_dev, final_path.st_ino) == (after.st_dev, after.st_ino)
            and (canonical_info.st_dev, canonical_info.st_ino)
            == (after.st_dev, after.st_ino)
            and initial_canonical == final_canonical
            and final_path.st_nlink == 1
            and final_path.st_size == byte_count,
            "unsafe binary artifact changed while it was being read",
        )
    finally:
        os.close(fd)
    return (
        bytes(payload) if payload is not None else None,
        SafeInputReceipt(
            path=display_path,
            canonical_path=final_canonical,
            byte_count=byte_count,
            sha256=f"sha256:{digest.hexdigest()}",
            device=after.st_dev,
            inode=after.st_ino,
            modified_time_ns=after.st_mtime_ns,
        ),
    )


def _snapshot_binary(
    path: Path,
    *,
    max_bytes: int = 64 * 1024 * 1024 * 1024,
) -> SafeInputReceipt:
    """Hash one stable regular-file snapshot without materializing media in memory."""

    _payload, receipt = _read_artifact_snapshot(
        path,
        max_bytes=max_bytes,
        capture=False,
    )
    return receipt


def _load_json_snapshot(
    path: Path,
    *,
    max_bytes: int = SAFE_JSON_MAX_BYTES,
) -> SafeJSONResult:
    """Parse strict JSON from the same descriptor-backed, symlink-safe snapshot."""

    payload, receipt = _read_artifact_snapshot(
        path,
        max_bytes=max_bytes,
        capture=True,
    )
    assert payload is not None
    try:
        value = _parse_json(_decode_utf8(payload), context="document")
    except SafeJSONInputError:
        raise
    return SafeJSONResult(value, receipt)


@contextmanager
def _private_decoder_snapshot(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> Any:
    """Yield an unlinked, read-only descriptor path containing the exact recorded MP4 bytes."""

    with tempfile.TemporaryDirectory(prefix="percept-matched-media-") as private_root:
        private_path = Path(private_root) / "snapshot.mp4"
        output_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0))
        )
        output_fd = os.open(private_path, output_flags, 0o600)
        try:
            _payload, observed = _read_artifact_snapshot(
                path,
                max_bytes=64 * 1024 * 1024 * 1024,
                capture=False,
                sink_fd=output_fd,
            )
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        _require(
            _safe_receipt_dict(observed)
            == {key: expected[key] for key in _SAFE_INPUT_RECEIPT_KEYS},
            "media artifact changed before its immutable decoder snapshot",
        )
        os.chmod(private_path, 0o400)
        snapshot_fd = os.open(
            private_path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        os.unlink(private_path)
        descriptor_path = next(
            (
                candidate
                for candidate in (
                    f"/dev/fd/{snapshot_fd}",
                    f"/proc/self/fd/{snapshot_fd}",
                )
                if os.path.exists(candidate)
            ),
            None,
        )
        try:
            _require(
                descriptor_path is not None,
                "real-video matched evaluation requires descriptor-path decoding",
            )
            yield descriptor_path
        finally:
            os.close(snapshot_fd)


def _assert_recorded_artifact_identity(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> None:
    """Check path and opened inode identity without rehashing a large media file."""

    normalized = _normalize_platform_root_alias(path)
    fd = _open_artifact_no_symlinks(normalized)
    try:
        opened = os.fstat(fd)
        lexical = os.lstat(normalized)
        canonical = normalized.resolve(strict=True)
        canonical_info = os.lstat(canonical)
    except OSError as exc:
        raise MatchedWatchInputError(
            f"media artifact identity cannot be inspected: {exc}"
        ) from exc
    finally:
        os.close(fd)
    _require(
        stat.S_ISREG(opened.st_mode)
        and opened.st_nlink == 1
        and (opened.st_dev, opened.st_ino)
        == (int(expected["device"]), int(expected["inode"]))
        == (lexical.st_dev, lexical.st_ino)
        == (canonical_info.st_dev, canonical_info.st_ino)
        and opened.st_size == int(expected["byte_count"])
        and opened.st_mtime_ns == int(expected["modified_time_ns"])
        and str(canonical) == str(expected["canonical_path"]),
        "media artifact identity changed while the immutable snapshot was decoded",
    )


async def _monitor_recorded_media(
    path: Path,
    *,
    expected: Mapping[str, Any],
    stop: asyncio.Event,
    violation: list[BaseException],
) -> None:
    """Observe the recorded path while OpenCV consumes its immutable private snapshot."""

    while not stop.is_set() and not violation:
        try:
            _assert_recorded_artifact_identity(path, expected=expected)
        except BaseException as exc:  # noqa: BLE001 - preserve the exact fail-closed cause
            violation.append(exc)
            return
        await asyncio.sleep(0)


@dataclass(frozen=True)
class _VerdictSpec:
    satisfied: bool
    confidence: float
    evidence: str
    error: str | None


@dataclass(frozen=True)
class _FrameSpec:
    index: int
    t: float
    fill: int
    token: str
    from_verdict: _VerdictSpec
    to_verdict: _VerdictSpec


@dataclass(frozen=True)
class _OracleFrameSpec:
    index: int
    t: float
    frame_sha256: str
    token: str
    verdicts: Mapping[str, _VerdictSpec]


@dataclass(frozen=True)
class _EventSpec:
    id: str
    eligible_at_s: float
    start: float
    end: float
    review_authority: str
    label_digest: str


@dataclass(frozen=True)
class _Manifest:
    document: Mapping[str, Any]
    input_receipt: SafeInputReceipt
    media_receipt: SafeInputReceipt
    case_id: str
    split: str
    fps: float
    source_id: str
    watch_parameters: Mapping[str, Any]
    judge_identity: Mapping[str, str]
    operating_point: str
    salience_stat: str
    threshold: float
    events: tuple[_EventSpec, ...]
    frames: tuple[_FrameSpec, ...]
    width: int
    height: int
    evidence_classification: str
    schema: str = _SCHEMA
    template_ref: str = _TEMPLATE
    domain_pack: DomainPack = FACILITY_OBSERVATION_V1
    plan_route: str = "transition"
    judge_mode: str = "scripted_contract"
    media_type: str = "application/json"
    oracle_frames: tuple[_OracleFrameSpec, ...] = ()
    input_artifacts: tuple[Mapping[str, Any], ...] = ()
    authority_artifact_presence: Mapping[str, bool] = dc_field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class _JudgePayload:
    index: int
    token: str
    t: float

    def __str__(self) -> str:
        return self.token


@dataclass(frozen=True)
class ProviderCallReceipt:
    sequence: int
    candidate_index: int
    candidate_t: float
    predicate: str
    condition_sha256: str
    status: str
    satisfied: bool | None
    confidence: float | None
    evidence: str | None
    error: str | None
    oracle_lookup_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.oracle_lookup_sha256 is None:
            result.pop("oracle_lookup_sha256")
        return result


@dataclass(frozen=True)
class CandidateReceipt:
    index: int
    t: float
    edge_decision: Mapping[str, Any]
    admitted: bool
    runtime_dispatched: bool
    provider_calls: tuple[ProviderCallReceipt, ...]
    decision: str | None
    transition_outcome: str | None
    incident_ids: tuple[str, ...]
    source_sequence: int | None = None
    frame_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "index": self.index,
            "t": self.t,
            "edge_decision": _thaw(self.edge_decision),
            "admitted": self.admitted,
            "runtime_dispatched": self.runtime_dispatched,
            "provider_calls": [item.to_dict() for item in self.provider_calls],
            "decision": self.decision,
            "transition_outcome": self.transition_outcome,
            "incident_ids": list(self.incident_ids),
        }
        if self.source_sequence is not None or self.frame_sha256 is not None:
            result["source_sequence"] = self.source_sequence
            result["frame_sha256"] = self.frame_sha256
        return result


@dataclass(frozen=True)
class ArmReceipt:
    arm: str
    source_profile: Mapping[str, Any]
    resolved_watch: Mapping[str, Any]
    apply_receipt: Mapping[str, Any]
    compiled_binding: Mapping[str, Any]
    judge_identity: Mapping[str, str]
    media_identity: Mapping[str, Any]
    replay_identity: Mapping[str, Any]
    configuration_sha256: str
    candidate_timestamps: tuple[float, ...]
    candidates: tuple[CandidateReceipt, ...]
    incidents: tuple[Mapping[str, Any], ...]
    asks: tuple[Mapping[str, Any], ...]
    traces: tuple[Mapping[str, Any], ...]
    health: tuple[Mapping[str, Any], ...]
    runtime_stats: Mapping[str, Any]
    runtime_dispatches: int
    provider_call_attempts: int
    provider_call_successes: int
    provider_call_failures: int
    valid: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "source_profile": _thaw(self.source_profile),
            "resolved_watch": _thaw(self.resolved_watch),
            "apply_receipt": _thaw(self.apply_receipt),
            "compiled_binding": _thaw(self.compiled_binding),
            "judge_identity": _thaw(self.judge_identity),
            "media_identity": _thaw(self.media_identity),
            "replay_identity": _thaw(self.replay_identity),
            "configuration_sha256": self.configuration_sha256,
            "candidate_timestamps": list(self.candidate_timestamps),
            "candidates": [item.to_dict() for item in self.candidates],
            "incidents": [_thaw(item) for item in self.incidents],
            "asks": [_thaw(item) for item in self.asks],
            "traces": [_thaw(item) for item in self.traces],
            "health": [_thaw(item) for item in self.health],
            "runtime_stats": _thaw(self.runtime_stats),
            "runtime_dispatches": self.runtime_dispatches,
            "provider_call_attempts": self.provider_call_attempts,
            "provider_call_successes": self.provider_call_successes,
            "provider_call_failures": self.provider_call_failures,
            "valid": self.valid,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class MatchedWatchReceipt:
    manifest: Mapping[str, Any]
    manifest_input_receipt: Mapping[str, Any]
    media_input_receipt: Mapping[str, Any]
    watch_plan: Mapping[str, Any]
    watch_plan_sha256: str
    judge_identity: Mapping[str, str]
    event_windows: tuple[Mapping[str, Any], ...]
    evidence_classification: str
    package_version: str
    build_provenance: Mapping[str, str]
    baseline: ArmReceipt
    adaptive: ArmReceipt
    comparable: bool
    comparability_blockers: tuple[str, ...]
    run_timestamp: str
    origin_evidence_sha256: str
    receipt_schema: str = _RECEIPT_SCHEMA
    input_artifacts: tuple[Mapping[str, Any], ...] = ()
    authority_artifact_presence: Mapping[str, bool] = dc_field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def launch_qualified(self) -> bool:
        return False

    def content_dict(self) -> dict[str, Any]:
        result = {
            "schema": self.receipt_schema,
            "manifest": _thaw(self.manifest),
            "manifest_input_receipt": _thaw(self.manifest_input_receipt),
            "media_input_receipt": _thaw(self.media_input_receipt),
            "watch_plan": _thaw(self.watch_plan),
            "watch_plan_sha256": self.watch_plan_sha256,
            "judge_identity": _thaw(self.judge_identity),
            "event_windows": [_thaw(item) for item in self.event_windows],
            "evidence_classification": self.evidence_classification,
            "package_version": self.package_version,
            "build_provenance": _thaw(self.build_provenance),
            "baseline": self.baseline.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "comparable": self.comparable,
            "comparability_blockers": list(self.comparability_blockers),
            "launch_qualified": False,
            "qualification_reason": _QUALIFICATION_REASON,
        }
        if self.receipt_schema == _RECEIPT_SCHEMA_V2:
            result["input_artifacts"] = [_thaw(item) for item in self.input_artifacts]
            result["authority_artifact_presence"] = _thaw(
                self.authority_artifact_presence
            )
        return result

    def hash_content_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        for key in ("manifest_input_receipt", "media_input_receipt"):
            full = result[key]
            result[key] = {"sha256": full["sha256"], "byte_count": full["byte_count"]}
        if "input_artifacts" in result:
            result["input_artifacts"] = [
                {
                    "role": item["role"],
                    "sha256": item["sha256"],
                    "byte_count": item["byte_count"],
                }
                for item in result["input_artifacts"]
            ]
        for arm_name in ("baseline", "adaptive"):
            arm = result[arm_name]
            arm["incidents"] = [
                {
                    key: value
                    for key, value in incident.items()
                    if key not in {"id", "judgment_event_id"}
                }
                for incident in arm["incidents"]
            ]
            for candidate in arm["candidates"]:
                candidate["incident_count"] = len(candidate.pop("incident_ids"))
            # Raw traces contain real local timing and generated correlation ids. Their deterministic
            # semantic projection is already bound above as candidates/provider calls/incidents/stats.
            arm.pop("traces")
            arm.pop("health")
            arm["runtime_stats"].pop("last_wall_t", None)
        return result

    def evidence_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["run_timestamp"] = self.run_timestamp
        return result

    @property
    def content_sha256(self) -> str:
        return _canonical_sha(self.hash_content_dict())

    @property
    def evidence_sha256(self) -> str:
        return _canonical_sha(self.evidence_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["content_sha256"] = self.content_sha256
        result["evidence_sha256"] = self.evidence_sha256
        result["origin_evidence_sha256"] = self.origin_evidence_sha256
        result["run_timestamp"] = self.run_timestamp
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class LoadedMatchedWatchReceipt:
    """One safely snapshotted receipt and its independently reconstructed projections."""

    receipt: MatchedWatchReceipt
    input_receipt: SafeInputReceipt
    viability_rows: tuple[Mapping[str, Any], Mapping[str, Any]]


def _strict_int(value: Any, context: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{context} must be an exact integer",
    )
    _require(value >= minimum, f"{context} must be >= {minimum}")
    return value


def _strict_float(value: Any, context: str) -> float:
    _require(isinstance(value, float), f"{context} must be an exact JSON floating-point number")
    _require(math.isfinite(value), f"{context} must be finite")
    return value


def _strict_bool(value: Any, context: str) -> bool:
    _require(isinstance(value, bool), f"{context} must be boolean")
    return value


def _strict_string(value: Any, context: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str), f"{context} must be text")
    _require(allow_empty or bool(value), f"{context} must be non-empty text")
    return value


def _strict_json(value: Any, context: str) -> Any:
    """Freeze an exact JSON value while refusing values outside the persisted wire model."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        _require(math.isfinite(value), f"{context} must not contain non-finite numbers")
        return value
    if isinstance(value, list):
        return tuple(_strict_json(item, f"{context}[{index}]") for index, item in enumerate(value))
    if isinstance(value, dict):
        return MappingProxyType(
            {
                key: _strict_json(item, f"{context}.{key}")
                for key, item in value.items()
            }
        )
    _fail(f"{context} contains a non-JSON value")


def _strict_object(value: Any, context: str, *, keys: set[str] | None = None) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{context} must be an object")
    if keys is not None:
        _exact(value, keys, context)
    result = _strict_json(value, context)
    assert isinstance(result, Mapping)
    return result


def _strict_object_tuple(
    value: Any,
    context: str,
    *,
    keys: set[str] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    _require(isinstance(value, list), f"{context} must be a list")
    return tuple(
        _strict_object(item, f"{context}[{index}]", keys=keys)
        for index, item in enumerate(value)
    )


def _parse_safe_input_receipt_mapping(value: Any, context: str) -> Mapping[str, Any]:
    doc = _object(value, context)
    _exact(doc, _SAFE_INPUT_RECEIPT_KEYS, context)
    _strict_string(doc["path"], f"{context}.path")
    _strict_string(doc["canonical_path"], f"{context}.canonical_path")
    _sha(doc["sha256"], f"{context}.sha256")
    _strict_int(doc["byte_count"], f"{context}.byte_count", minimum=1)
    _strict_int(doc["device"], f"{context}.device")
    _strict_int(doc["inode"], f"{context}.inode")
    _strict_int(doc["modified_time_ns"], f"{context}.modified_time_ns")
    return _strict_object(doc, context, keys=_SAFE_INPUT_RECEIPT_KEYS)


def _parse_provider_call_receipt(
    value: Any,
    context: str,
    *,
    v2: bool = False,
) -> ProviderCallReceipt:
    doc = _object(value, context)
    _exact(
        doc,
        _PROVIDER_CALL_RECEIPT_V2_KEYS if v2 else _PROVIDER_CALL_RECEIPT_KEYS,
        context,
    )
    sequence = _strict_int(doc["sequence"], f"{context}.sequence", minimum=1)
    candidate_index = _strict_int(doc["candidate_index"], f"{context}.candidate_index")
    candidate_t = _strict_float(doc["candidate_t"], f"{context}.candidate_t")
    predicate = _strict_string(doc["predicate"], f"{context}.predicate")
    allowed_predicates = {"from", "to", "condition"} if v2 else {"from", "to"}
    _require(
        predicate in allowed_predicates,
        f"{context}.predicate must be one of {sorted(allowed_predicates)!r}",
    )
    condition_sha256 = _sha(doc["condition_sha256"], f"{context}.condition_sha256")
    status = _strict_string(doc["status"], f"{context}.status")
    _require(status in {"success", "error"}, f"{context}.status must be 'success' or 'error'")
    satisfied = doc["satisfied"]
    confidence = doc["confidence"]
    evidence = doc["evidence"]
    error = doc["error"]
    _require(satisfied is None or isinstance(satisfied, bool), f"{context}.satisfied must be boolean or null")
    if confidence is not None:
        _strict_float(confidence, f"{context}.confidence")
        _require(0.0 <= confidence <= 1.0, f"{context}.confidence must be in [0, 1]")
    _require(evidence is None or isinstance(evidence, str), f"{context}.evidence must be text or null")
    _require(error is None or isinstance(error, str), f"{context}.error must be text or null")
    if status == "success":
        _require(
            satisfied is not None and confidence is not None and evidence is not None and error is None,
            f"{context} successful call has inconsistent result fields",
        )
    else:
        _require(
            satisfied is None and confidence is None and evidence is None and bool(error),
            f"{context} failed call has inconsistent result fields",
        )
    oracle_lookup_sha256 = doc.get("oracle_lookup_sha256")
    if v2:
        oracle_lookup_sha256 = _sha(
            oracle_lookup_sha256,
            f"{context}.oracle_lookup_sha256",
        )
    return ProviderCallReceipt(
        sequence,
        candidate_index,
        candidate_t,
        predicate,
        condition_sha256,
        status,
        satisfied,
        confidence,
        evidence,
        error,
        oracle_lookup_sha256,
    )


def _parse_candidate_receipt(
    value: Any,
    context: str,
    *,
    v2: bool = False,
) -> CandidateReceipt:
    doc = _object(value, context)
    _exact(
        doc,
        _CANDIDATE_RECEIPT_V2_KEYS if v2 else _CANDIDATE_RECEIPT_KEYS,
        context,
    )
    index = _strict_int(doc["index"], f"{context}.index")
    t = _strict_float(doc["t"], f"{context}.t")
    raw_edge = _object(doc["edge_decision"], f"{context}.edge_decision")
    _exact(raw_edge, _EDGE_DECISION_KEYS, f"{context}.edge_decision")
    for key in ("threshold", "motion", "structural", "semantic"):
        _strict_float(raw_edge[key], f"{context}.edge_decision.{key}")
    for key in ("escalate", "coast_admit"):
        _strict_bool(raw_edge[key], f"{context}.edge_decision.{key}")
    for key in ("goal_ids", "admission_causes"):
        _require(isinstance(raw_edge[key], list), f"{context}.edge_decision.{key} must be a list")
        for item_index, item in enumerate(raw_edge[key]):
            _strict_string(item, f"{context}.edge_decision.{key}[{item_index}]")
    edge = _strict_object(raw_edge, f"{context}.edge_decision", keys=_EDGE_DECISION_KEYS)
    admitted = _strict_bool(doc["admitted"], f"{context}.admitted")
    dispatched = _strict_bool(doc["runtime_dispatched"], f"{context}.runtime_dispatched")
    _require(isinstance(doc["provider_calls"], list), f"{context}.provider_calls must be a list")
    calls = tuple(
        _parse_provider_call_receipt(
            item,
            f"{context}.provider_calls[{call_index}]",
            v2=v2,
        )
        for call_index, item in enumerate(doc["provider_calls"])
    )
    decision = doc["decision"]
    _require(
        decision is None or decision in {"fire", "ask", "silent", "missing"},
        f"{context}.decision is unsupported",
    )
    transition = doc["transition_outcome"]
    _require(
        transition is None or isinstance(transition, str),
        f"{context}.transition_outcome must be text or null",
    )
    raw_ids = doc["incident_ids"]
    _require(isinstance(raw_ids, list), f"{context}.incident_ids must be a list")
    incident_ids = tuple(
        _strict_string(item, f"{context}.incident_ids[{item_index}]")
        for item_index, item in enumerate(raw_ids)
    )
    _require(len(incident_ids) == len(set(incident_ids)), f"{context}.incident_ids contains duplicates")
    _require(dispatched == (decision is not None), f"{context} dispatch and decision fields disagree")
    source_sequence = None
    frame_sha256 = None
    if v2:
        source_sequence = _strict_int(
            doc["source_sequence"],
            f"{context}.source_sequence",
        )
        frame_sha256 = _sha(doc["frame_sha256"], f"{context}.frame_sha256")
    return CandidateReceipt(
        index,
        t,
        edge,
        admitted,
        dispatched,
        calls,
        decision,
        transition,
        incident_ids,
        source_sequence,
        frame_sha256,
    )


def _same_json_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _parse_resolved_watch(value: Any, context: str) -> Mapping[str, Any]:
    doc = _object(value, context)
    _exact(doc, _RESOLVED_WATCH_KEYS, context)
    raw_plan = _object(doc["plan"], f"{context}.plan")
    try:
        canonical_plan = WatchPlan.from_dict(raw_plan).to_dict()
    except (TypeError, ValueError) as exc:
        raise MatchedWatchInputError(f"{context}.plan is not a valid WatchPlan: {exc}") from exc
    _require(canonical_plan == raw_plan, f"{context}.plan is not canonical")
    source_profile = _object(doc["source_profile"], f"{context}.source_profile")
    _exact(source_profile, _RESOLVED_SOURCE_PROFILE_KEYS, f"{context}.source_profile")
    domain_pack = _object(doc["domain_pack"], f"{context}.domain_pack")
    _exact(
        domain_pack,
        {"id", "version", "values", "components", "policy_clamps"},
        f"{context}.domain_pack",
    )
    _strict_string(domain_pack["id"], f"{context}.domain_pack.id")
    _strict_string(domain_pack["version"], f"{context}.domain_pack.version")
    domain_values = _object(domain_pack["values"], f"{context}.domain_pack.values")
    unknown_domain_paths = sorted(set(domain_values) - _REGISTERED_TUNABLE_PATHS)
    _require(
        not unknown_domain_paths,
        f"{context}.domain_pack.values contains unregistered paths {unknown_domain_paths!r}",
    )
    raw_components = domain_pack["components"]
    _require(isinstance(raw_components, list), f"{context}.domain_pack.components must be a list")
    for index, raw_component in enumerate(raw_components):
        component_context = f"{context}.domain_pack.components[{index}]"
        component = _object(raw_component, component_context)
        _exact(component, {"ref", "strength", "reason"}, component_context)
        for key in ("ref", "strength", "reason"):
            _strict_string(component[key], f"{component_context}.{key}")
    clamps = _object(domain_pack["policy_clamps"], f"{context}.domain_pack.policy_clamps")
    unsupported_clamps = sorted(
        set(clamps) - {"false_negative_cost_min", "retention_max_days"}
    )
    _require(
        not unsupported_clamps,
        f"{context}.domain_pack.policy_clamps has unsupported fields {unsupported_clamps!r}",
    )
    _require(doc["execution_policy"] is None, f"{context}.execution_policy must be null")
    components = _object(doc["components"], f"{context}.components")
    _exact(components, _RESOLVED_COMPONENT_KEYS, f"{context}.components")
    providers = _object(components["providers"], f"{context}.components.providers")
    _exact(providers, _RESOLVED_PROVIDER_KEYS, f"{context}.components.providers")
    for name, identity in providers.items():
        _require(
            identity is None or isinstance(identity, str),
            f"{context}.components.providers[{name!r}] must be text or null",
        )
    _strict_string(
        components["catalog_component_hash"],
        f"{context}.components.catalog_component_hash",
    )
    effective = _object(doc["effective"], f"{context}.effective")
    provenance = _object(doc["provenance"], f"{context}.provenance")
    _require(
        set(effective) == set(provenance),
        f"{context}.effective and provenance must have exactly the same paths",
    )
    unknown_paths = sorted(set(effective) - _REGISTERED_TUNABLE_PATHS)
    _require(not unknown_paths, f"{context} contains unregistered tunable paths {unknown_paths!r}")
    for path, raw_origin in provenance.items():
        origin_context = f"{context}.provenance[{path!r}]"
        origin = _object(raw_origin, origin_context)
        _exact(origin, _SETTING_ORIGIN_KEYS, origin_context)
        _strict_string(origin["owner_layer"], f"{origin_context}.owner_layer")
        _strict_string(origin["source_id"], f"{origin_context}.source_id")
        _require(isinstance(origin["reason"], str), f"{origin_context}.reason must be text")
        _require(
            origin["validated_by"] is None or isinstance(origin["validated_by"], str),
            f"{origin_context}.validated_by must be text or null",
        )
        _require(
            _same_json_value(origin["value"], effective[path]),
            f"{origin_context}.value disagrees with effective[{path!r}]",
        )
    _strict_string(doc["plan_hash"], f"{context}.plan_hash")
    _strict_string(doc["component_snapshot_hash"], f"{context}.component_snapshot_hash")
    _require(
        doc["optimization_lock_hash"] is None
        or isinstance(doc["optimization_lock_hash"], str),
        f"{context}.optimization_lock_hash must be text or null",
    )
    return _strict_object(doc, context, keys=_RESOLVED_WATCH_KEYS)


def _parse_source_profile_receipt(value: Any, context: str) -> Mapping[str, Any]:
    doc = _object(value, context)
    _exact(doc, _SOURCE_PROFILE_KEYS, context)
    for key in (
        "name",
        "temporal_unit",
        "admission_strategy",
        "evidence_strategy",
        "identity_strategy",
        "default_timebase",
        "stability",
        "notes",
    ):
        _require(isinstance(doc[key], str), f"{context}.{key} must be text")
    _require(
        isinstance(doc["modalities"], list)
        and all(isinstance(item, str) and item for item in doc["modalities"]),
        f"{context}.modalities must be a list of non-empty strings",
    )
    parameters = _object(doc["parameters"], f"{context}.parameters")
    _exact(
        parameters,
        {"salience_stat", "salience_threshold"},
        f"{context}.parameters",
    )
    _require(
        parameters["salience_stat"] in {"mean", "block_max", "p95"},
        f"{context}.parameters.salience_stat is unsupported",
    )
    _strict_float(parameters["salience_threshold"], f"{context}.parameters.salience_threshold")
    return _strict_object(doc, context, keys=_SOURCE_PROFILE_KEYS)


def _parse_apply_receipt(value: Any, context: str) -> Mapping[str, Any]:
    doc = _object(value, context)
    _exact(doc, _APPLY_RECEIPT_KEYS, context)
    _require(
        doc["status"] in {"applied", "applied_with_warnings"},
        f"{context}.status is unsupported for a valid matched receipt",
    )
    for key in ("watch_ids", "accepted_fields"):
        _require(isinstance(doc[key], list), f"{context}.{key} must be a list")
        for index, item in enumerate(doc[key]):
            _strict_string(item, f"{context}.{key}[{index}]")
    for key in ("dropped_fields", "coerced_fields", "warnings"):
        _require(isinstance(doc[key], list), f"{context}.{key} must be a list")
        for index, raw_item in enumerate(doc[key]):
            item_context = f"{context}.{key}[{index}]"
            item = _object(raw_item, item_context)
            _exact(item, {"path", "code", "goal_id", "message"}, item_context)
            for field in ("path", "code", "message"):
                _strict_string(item[field], f"{item_context}.{field}")
            _require(
                item["goal_id"] is None or isinstance(item["goal_id"], str),
                f"{item_context}.goal_id must be text or null",
            )
    _require(
        doc["plan_revision"] is None
        or (isinstance(doc["plan_revision"], int) and not isinstance(doc["plan_revision"], bool)),
        f"{context}.plan_revision must be an integer or null",
    )
    for key in (
        "judge_requested",
        "judge_resolved",
        "judge_context_ref",
        "judge_context_sha256",
    ):
        _require(
            doc[key] is None or isinstance(doc[key], str),
            f"{context}.{key} must be text or null",
        )
    _strict_bool(doc["fallback_used"], f"{context}.fallback_used")
    return _strict_object(doc, context, keys=_APPLY_RECEIPT_KEYS)


def _parse_compiled_binding(value: Any, context: str) -> Mapping[str, Any]:
    doc = _object(value, context)
    _exact(doc, _COMPILED_BINDING_KEYS, context)
    _strict_string(doc["goal_id"], f"{context}.goal_id")
    _strict_string(doc["timebase"], f"{context}.timebase")
    _require(
        doc["coverage_floor"] is None
        or (
            isinstance(doc["coverage_floor"], (int, float))
            and not isinstance(doc["coverage_floor"], bool)
            and math.isfinite(float(doc["coverage_floor"]))
        ),
        f"{context}.coverage_floor must be a finite number or null",
    )
    kwargs = _object(doc["watchspec_kwargs"], f"{context}.watchspec_kwargs")
    _exact(
        kwargs,
        {"threshold", "salience_stat", "ego_compensate", "stat_thresholds"},
        f"{context}.watchspec_kwargs",
    )
    _strict_float(kwargs["threshold"], f"{context}.watchspec_kwargs.threshold")
    _require(
        kwargs["salience_stat"] in {"mean", "block_max", "p95"},
        f"{context}.watchspec_kwargs.salience_stat is unsupported",
    )
    _strict_bool(kwargs["ego_compensate"], f"{context}.watchspec_kwargs.ego_compensate")
    thresholds = _object(
        kwargs["stat_thresholds"],
        f"{context}.watchspec_kwargs.stat_thresholds",
    )
    _exact(
        thresholds,
        {"mean", "block_max", "p95"},
        f"{context}.watchspec_kwargs.stat_thresholds",
    )
    for stat_name, threshold in thresholds.items():
        _strict_float(
            threshold,
            f"{context}.watchspec_kwargs.stat_thresholds.{stat_name}",
        )
    components = _object(doc["components"], f"{context}.components")
    _exact(components, _RESOLVED_PROVIDER_KEYS, f"{context}.components")
    for name, identity in components.items():
        _require(
            identity is None or isinstance(identity, str),
            f"{context}.components[{name!r}] must be text or null",
        )
    return _strict_object(doc, context, keys=_COMPILED_BINDING_KEYS)


def _parse_arm_receipt(
    value: Any,
    context: str,
    expected_arm: str,
    *,
    v2: bool = False,
) -> ArmReceipt:
    doc = _object(value, context)
    _exact(doc, _ARM_RECEIPT_KEYS, context)
    _require(doc["arm"] == expected_arm, f"{context}.arm must be {expected_arm!r}")
    source_profile = _parse_source_profile_receipt(
        doc["source_profile"],
        f"{context}.source_profile",
    )
    resolved_watch = _parse_resolved_watch(
        doc["resolved_watch"],
        f"{context}.resolved_watch",
    )
    apply_receipt = _parse_apply_receipt(
        doc["apply_receipt"],
        f"{context}.apply_receipt",
    )
    compiled_binding = _parse_compiled_binding(
        doc["compiled_binding"],
        f"{context}.compiled_binding",
    )
    judge_identity = _strict_object(
        doc["judge_identity"],
        f"{context}.judge_identity",
        keys=_JUDGE_IDENTITY_V2_KEYS if v2 else _JUDGE_IDENTITY_KEYS,
    )
    media_identity = _strict_object(
        doc["media_identity"],
        f"{context}.media_identity",
        keys=_MEDIA_IDENTITY_V2_KEYS if v2 else _MEDIA_IDENTITY_KEYS,
    )
    if v2:
        decoder = _object(
            media_identity["decoder_identity"],
            f"{context}.media_identity.decoder_identity",
        )
        _exact(
            decoder,
            _DECODER_IDENTITY_KEYS,
            f"{context}.media_identity.decoder_identity",
        )
        for key in ("library", "version", "source_adapter", "frame_encoding"):
            _strict_string(
                decoder[key],
                f"{context}.media_identity.decoder_identity.{key}",
            )
        _sha(
            decoder["build_sha256"],
            f"{context}.media_identity.decoder_identity.build_sha256",
        )
    replay_identity = _strict_object(
        doc["replay_identity"], f"{context}.replay_identity", keys=_REPLAY_KEYS
    )
    configuration_sha256 = _sha(doc["configuration_sha256"], f"{context}.configuration_sha256")
    raw_timestamps = doc["candidate_timestamps"]
    _require(isinstance(raw_timestamps, list), f"{context}.candidate_timestamps must be a list")
    timestamps = tuple(
        _strict_float(item, f"{context}.candidate_timestamps[{index}]")
        for index, item in enumerate(raw_timestamps)
    )
    _require(isinstance(doc["candidates"], list), f"{context}.candidates must be a list")
    candidates = tuple(
        _parse_candidate_receipt(
            item,
            f"{context}.candidates[{index}]",
            v2=v2,
        )
        for index, item in enumerate(doc["candidates"])
    )
    incidents = _strict_object_tuple(
        doc["incidents"], f"{context}.incidents", keys=_INCIDENT_RECEIPT_KEYS
    )
    for index, incident in enumerate(incidents):
        _strict_float(incident["ts"], f"{context}.incidents[{index}].ts")
    asks = _strict_object_tuple(doc["asks"], f"{context}.asks", keys=_ASK_RECEIPT_KEYS)
    traces = _strict_object_tuple(doc["traces"], f"{context}.traces", keys=_TRACE_RECEIPT_KEYS)
    for index, trace in enumerate(traces):
        stage = trace["stage"]
        _require(
            isinstance(stage, str) and stage in _TRACE_DATA_KEYS,
            f"{context}.traces[{index}].stage is unsupported",
        )
        trace_data = _object(trace["data"], f"{context}.traces[{index}].data")
        _exact(trace_data, _TRACE_DATA_KEYS[stage], f"{context}.traces[{index}].data")
        _require(
            trace["t"] is None or isinstance(trace["t"], float),
            f"{context}.traces[{index}].t must be an exact JSON floating-point number or null",
        )
    health = _strict_object_tuple(doc["health"], f"{context}.health", keys=_HEALTH_RECEIPT_KEYS)
    for index, item in enumerate(health):
        kind = item["kind"]
        _require(
            isinstance(kind, str) and kind in _HEALTH_DATA_KEYS,
            f"{context}.health[{index}].kind is unsupported",
        )
        health_data = _object(item["data"], f"{context}.health[{index}].data")
        _exact(health_data, _HEALTH_DATA_KEYS[kind], f"{context}.health[{index}].data")
        _require(
            item["ts"] is None or isinstance(item["ts"], float),
            f"{context}.health[{index}].ts must be an exact JSON floating-point number or null",
        )
    runtime_stats = _strict_object(
        doc["runtime_stats"],
        f"{context}.runtime_stats",
        keys=set(asdict(RuntimeStats())),
    )
    for key, item in runtime_stats.items():
        stat_context = f"{context}.runtime_stats.{key}"
        if key in {"effective_config", "admission_by_goal", "admission_by_cause"}:
            _require(isinstance(item, Mapping), f"{stat_context} must be an object")
            continue
        if key in {"last_media_t", "last_wall_t"}:
            _require(
                item is None or isinstance(item, float),
                f"{stat_context} must be an exact JSON floating-point number or null",
            )
            continue
        _strict_int(item, stat_context)
    count_fields = (
        "runtime_dispatches",
        "provider_call_attempts",
        "provider_call_successes",
        "provider_call_failures",
    )
    counts = {
        key: _strict_int(doc[key], f"{context}.{key}")
        for key in count_fields
    }
    valid = _strict_bool(doc["valid"], f"{context}.valid")
    raw_blockers = doc["blockers"]
    _require(isinstance(raw_blockers, list), f"{context}.blockers must be a list")
    blockers = tuple(
        _strict_string(item, f"{context}.blockers[{index}]")
        for index, item in enumerate(raw_blockers)
    )
    return ArmReceipt(
        expected_arm,
        source_profile,
        resolved_watch,
        apply_receipt,
        compiled_binding,
        judge_identity,
        media_identity,
        replay_identity,
        configuration_sha256,
        timestamps,
        candidates,
        incidents,
        asks,
        traces,
        health,
        runtime_stats,
        counts["runtime_dispatches"],
        counts["provider_call_attempts"],
        counts["provider_call_successes"],
        counts["provider_call_failures"],
        valid,
        blockers,
    )


def _plan_route(plan: WatchPlan) -> str:
    route = str(plan.explicit.get("Goal.state.type", "watch"))
    _require(
        route in {"watch", "transition"},
        "matched-Watch evaluation supports only ordinary and transition Watch templates",
    )
    if route == "transition":
        _require(
            isinstance(plan.explicit.get("Goal.state.from"), str)
            and isinstance(plan.explicit.get("Goal.state.to"), str),
            "transition Watch template must render exact from/to predicates",
        )
    return route


def _parse_persisted_real_manifest(doc: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    _exact(doc, _REAL_MANIFEST_KEYS, context)
    _require(doc["schema"] == _REAL_SCHEMA, f"{context}.schema must be {_REAL_SCHEMA!r}")
    _strict_string(doc["case_id"], f"{context}.case_id")
    _strict_string(doc["archetype"], f"{context}.archetype")
    _strict_string(doc["split"], f"{context}.split")
    media = _object(doc["media"], f"{context}.media")
    _exact(media, _REAL_MEDIA_REF_KEYS, f"{context}.media")
    _strict_string(media["path"], f"{context}.media.path")
    _sha(media["sha256"], f"{context}.media.sha256")
    _require(media["media_type"] == "video/mp4", f"{context}.media.media_type must be 'video/mp4'")
    replay = _object(doc["replay"], f"{context}.replay")
    _exact(replay, _REPLAY_KEYS, f"{context}.replay")
    _strict_float(replay["fps"], f"{context}.replay.fps")
    _require(replay["timebase"] == "media", f"{context}.replay.timebase must be 'media'")
    _strict_string(replay["source_id"], f"{context}.replay.source_id")
    watch = _object(doc["watch"], f"{context}.watch")
    _exact(watch, _WATCH_KEYS, f"{context}.watch")
    template_ref = _strict_string(watch["template_ref"], f"{context}.watch.template_ref")
    pack_name = _strict_string(watch["domain_pack"], f"{context}.watch.domain_pack")
    _require(
        template_ref.startswith(pack_name + "/"),
        f"{context}.watch.template_ref must belong to its domain_pack",
    )
    parameters = _strict_object(watch["parameters"], f"{context}.watch.parameters")
    try:
        _plan_route(watch_template(template_ref, parameters))
    except (KeyError, TypeError, ValueError) as exc:
        raise MatchedWatchInputError(f"{context}.watch cannot render: {exc}") from exc
    judge = _object(doc["judge"], f"{context}.judge")
    _exact(judge, _REAL_JUDGE_KEYS, f"{context}.judge")
    for key in ("adapter", "model"):
        _strict_string(judge[key], f"{context}.judge.{key}")
    _sha(judge["prompt_sha256"], f"{context}.judge.prompt_sha256")
    _require(
        judge["judge_mode"] == "offline_oracle",
        f"{context}.judge.judge_mode must be 'offline_oracle'",
    )
    adaptive = _object(doc["adaptive"], f"{context}.adaptive")
    _exact(adaptive, _ADAPTIVE_KEYS, f"{context}.adaptive")
    _strict_string(adaptive["operating_point"], f"{context}.adaptive.operating_point")
    _require(
        adaptive["salience_stat"] in {"mean", "block_max", "p95"},
        f"{context}.adaptive.salience_stat is unsupported",
    )
    _strict_float(adaptive["threshold"], f"{context}.adaptive.threshold")
    raw_events = doc["events"]
    _require(isinstance(raw_events, list) and raw_events, f"{context}.events must be non-empty")
    for index, raw_event in enumerate(raw_events):
        event_context = f"{context}.events[{index}]"
        event = _object(raw_event, event_context)
        _exact(event, _EVENT_KEYS, event_context)
        for key in ("id", "review_authority"):
            _strict_string(event[key], f"{event_context}.{key}")
        for key in ("eligible_at_s", "window_start_s", "window_end_s"):
            _strict_float(event[key], f"{event_context}.{key}")
        _sha(event["label_digest"], f"{event_context}.label_digest")
    for ref_name in ("corpus", "oracle"):
        ref = _object(doc[ref_name], f"{context}.{ref_name}")
        _exact(ref, _ARTIFACT_REF_KEYS, f"{context}.{ref_name}")
        _strict_string(ref["path"], f"{context}.{ref_name}.path")
        _sha(ref["sha256"], f"{context}.{ref_name}.sha256")
    _require(
        doc["evidence_classification"] == "admission_only",
        f"{context}.evidence_classification must be 'admission_only'",
    )
    return _strict_object(doc, context, keys=_REAL_MANIFEST_KEYS)


def _parse_persisted_manifest(value: Any) -> Mapping[str, Any]:
    context = "receipt.manifest"
    doc = _object(value, context)
    if doc.get("schema") == _REAL_SCHEMA:
        return _parse_persisted_real_manifest(doc, context)
    _exact(doc, _MANIFEST_KEYS, context)
    _require(doc["schema"] == _SCHEMA, f"{context}.schema must be {_SCHEMA!r}")
    _strict_string(doc["case_id"], f"{context}.case_id")
    _require(doc["archetype"] == _ARCHETYPE, f"{context}.archetype must be {_ARCHETYPE!r}")
    _strict_string(doc["split"], f"{context}.split")
    media = _object(doc["media"], f"{context}.media")
    _exact(media, _MEDIA_REF_KEYS, f"{context}.media")
    _strict_string(media["path"], f"{context}.media.path")
    _sha(media["sha256"], f"{context}.media.sha256")
    replay = _object(doc["replay"], f"{context}.replay")
    _exact(replay, _REPLAY_KEYS, f"{context}.replay")
    _strict_float(replay["fps"], f"{context}.replay.fps")
    _require(replay["timebase"] == "media", f"{context}.replay.timebase must be 'media'")
    _strict_string(replay["source_id"], f"{context}.replay.source_id")
    watch = _object(doc["watch"], f"{context}.watch")
    _exact(watch, _WATCH_KEYS, f"{context}.watch")
    _require(watch["template_ref"] == _TEMPLATE, f"{context}.watch.template_ref mismatch")
    _require(watch["domain_pack"] == _PACK, f"{context}.watch.domain_pack mismatch")
    _strict_object(watch["parameters"], f"{context}.watch.parameters")
    judge = _object(doc["judge"], f"{context}.judge")
    _exact(judge, _JUDGE_KEYS, f"{context}.judge")
    for key in ("adapter", "model"):
        _strict_string(judge[key], f"{context}.judge.{key}")
    _sha(judge["prompt_sha256"], f"{context}.judge.prompt_sha256")
    adaptive = _object(doc["adaptive"], f"{context}.adaptive")
    _exact(adaptive, _ADAPTIVE_KEYS, f"{context}.adaptive")
    _strict_string(adaptive["operating_point"], f"{context}.adaptive.operating_point")
    _require(
        adaptive["salience_stat"] in {"mean", "block_max", "p95"},
        f"{context}.adaptive.salience_stat is unsupported",
    )
    _strict_float(adaptive["threshold"], f"{context}.adaptive.threshold")
    raw_events = doc["events"]
    _require(isinstance(raw_events, list) and raw_events, f"{context}.events must be a non-empty list")
    for index, raw_event in enumerate(raw_events):
        event_context = f"{context}.events[{index}]"
        event = _object(raw_event, event_context)
        _exact(event, _EVENT_KEYS, event_context)
        for key in ("id", "review_authority"):
            _strict_string(event[key], f"{event_context}.{key}")
        for key in ("eligible_at_s", "window_start_s", "window_end_s"):
            _strict_float(event[key], f"{event_context}.{key}")
        _sha(event["label_digest"], f"{event_context}.label_digest")
    _require(
        doc["evidence_classification"] in _EVIDENCE_CLASSES,
        f"{context}.evidence_classification is unsupported",
    )
    return _strict_object(doc, context, keys=_MANIFEST_KEYS)


def _parse_verdict(value: Any, context: str) -> _VerdictSpec:
    doc = _object(value, context)
    _exact(doc, _VERDICT_KEYS, context)
    _require(isinstance(doc["satisfied"], bool), f"{context}.satisfied must be boolean")
    confidence = _number(doc["confidence"], f"{context}.confidence", minimum=0.0)
    _require(confidence <= 1.0, f"{context}.confidence must be <= 1")
    evidence = doc["evidence"]
    _require(isinstance(evidence, str), f"{context}.evidence must be text")
    error = doc["error"]
    _require(error is None or isinstance(error, str), f"{context}.error must be text or null")
    return _VerdictSpec(doc["satisfied"], confidence, evidence, error)


def _load_real_manifest(
    loaded: Any,
    doc: Mapping[str, Any],
) -> _Manifest:
    _exact(doc, _REAL_MANIFEST_KEYS, "manifest")
    case_id = _text(doc["case_id"], "manifest.case_id")
    _text(doc["archetype"], "manifest.archetype")
    split = _text(doc["split"], "manifest.split")
    manifest_root = Path(loaded.receipt.canonical_path).parent

    media_ref = _object(doc["media"], "manifest.media")
    _exact(media_ref, _REAL_MEDIA_REF_KEYS, "manifest.media")
    _require(media_ref["media_type"] == "video/mp4", "manifest.media.media_type must be 'video/mp4'")
    media_path = _resolve_contained_path(
        media_ref["path"],
        root=manifest_root,
        context="manifest.media.path",
    )
    _require(media_path.suffix.lower() == ".mp4", "real-video media path must end in .mp4")
    media_receipt = _snapshot_binary(media_path)
    _require(
        media_receipt.sha256 == _sha(media_ref["sha256"], "manifest.media.sha256"),
        "manifest media digest does not match the descriptor-backed MP4 snapshot",
    )

    corpus_ref = _object(doc["corpus"], "manifest.corpus")
    _exact(corpus_ref, _ARTIFACT_REF_KEYS, "manifest.corpus")
    corpus_path = _resolve_contained_path(
        corpus_ref["path"],
        root=manifest_root,
        context="manifest.corpus.path",
    )
    try:
        corpus_loaded = _load_json_snapshot(corpus_path)
    except SafeJSONInputError as exc:
        raise MatchedWatchInputError(f"unsafe matched-Watch corpus index: {exc}") from exc
    _require(
        corpus_loaded.receipt.sha256 == _sha(corpus_ref["sha256"], "manifest.corpus.sha256"),
        "manifest corpus digest does not match its descriptor-backed snapshot",
    )
    corpus = _object(corpus_loaded.value, "corpus")
    _exact(corpus, _CORPUS_KEYS, "corpus")
    _require(corpus["schema"] == _CORPUS_SCHEMA, f"corpus.schema must be {_CORPUS_SCHEMA!r}")
    _text(corpus["corpus_id"], "corpus.corpus_id")
    _text(corpus["version"], "corpus.version")
    raw_cases = corpus["cases"]
    _require(isinstance(raw_cases, list) and raw_cases, "corpus.cases must be non-empty")
    case_ids: set[str] = set()
    media_owners: dict[str, str] = {}
    oracle_owners: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    active_case: Mapping[str, Any] | None = None
    for index, raw_case in enumerate(raw_cases):
        context = f"corpus.cases[{index}]"
        case = _object(raw_case, context)
        _exact(case, _CORPUS_CASE_KEYS, context)
        indexed_case_id = _text(case["case_id"], f"{context}.case_id")
        _require(indexed_case_id not in case_ids, f"duplicate corpus case id {indexed_case_id!r}")
        case_ids.add(indexed_case_id)
        indexed_split = _text(case["split"], f"{context}.split")
        group = _text(case["source_group_id"], f"{context}.source_group_id")
        media_sha = _sha(case["media_sha256"], f"{context}.media_sha256")
        oracle_sha = _sha(case["oracle_sha256"], f"{context}.oracle_sha256")
        owner = media_owners.setdefault(media_sha, indexed_case_id)
        _require(
            owner == indexed_case_id,
            f"duplicate media digest is assigned to cases {owner!r} and {indexed_case_id!r}",
        )
        oracle_owner = oracle_owners.setdefault(oracle_sha, indexed_case_id)
        _require(
            oracle_owner == indexed_case_id,
            "duplicate offline-oracle digest is assigned to cases "
            f"{oracle_owner!r} and {indexed_case_id!r}; case/split evidence must be independent",
        )
        prior_split = group_splits.setdefault(group, indexed_split)
        _require(
            prior_split == indexed_split,
            f"source group {group!r} crosses splits {prior_split!r} and {indexed_split!r}",
        )
        if indexed_case_id == case_id:
            active_case = MappingProxyType(
                {
                    **case,
                    "media_sha256": media_sha,
                    "oracle_sha256": oracle_sha,
                }
            )
    _require(active_case is not None, f"case {case_id!r} is absent from its corpus index")
    _require(active_case["split"] == split, "manifest split disagrees with its corpus index")
    _require(
        active_case["media_sha256"] == media_receipt.sha256,
        "manifest media does not match the indexed case",
    )

    authorities = _object(corpus["authorities"], "corpus.authorities")
    _exact(authorities, set(_AUTHORITY_NAMES), "corpus.authorities")
    authority_presence: dict[str, bool] = {}
    authority_artifacts: list[Mapping[str, Any]] = []
    corpus_root = Path(corpus_loaded.receipt.canonical_path).parent
    for name in _AUTHORITY_NAMES:
        raw_ref = authorities[name]
        if raw_ref is None:
            authority_presence[name] = False
            continue
        ref = _object(raw_ref, f"corpus.authorities.{name}")
        _exact(ref, _ARTIFACT_REF_KEYS, f"corpus.authorities.{name}")
        authority_path = _resolve_contained_path(
            ref["path"],
            root=corpus_root,
            context=f"corpus.authorities.{name}.path",
        )
        authority_receipt = _snapshot_binary(authority_path, max_bytes=64 * 1024 * 1024)
        _require(
            authority_receipt.sha256
            == _sha(ref["sha256"], f"corpus.authorities.{name}.sha256"),
            f"corpus authority artifact {name!r} digest mismatch",
        )
        authority_presence[name] = True
        authority_artifacts.append(_artifact_receipt(f"authority:{name}", authority_receipt))

    oracle_ref = _object(doc["oracle"], "manifest.oracle")
    _exact(oracle_ref, _ARTIFACT_REF_KEYS, "manifest.oracle")
    oracle_path = _resolve_contained_path(
        oracle_ref["path"],
        root=manifest_root,
        context="manifest.oracle.path",
    )
    try:
        oracle_loaded = _load_json_snapshot(oracle_path)
    except SafeJSONInputError as exc:
        raise MatchedWatchInputError(f"unsafe matched-Watch offline oracle: {exc}") from exc
    expected_oracle_sha = _sha(oracle_ref["sha256"], "manifest.oracle.sha256")
    _require(
        oracle_loaded.receipt.sha256 == expected_oracle_sha,
        "manifest oracle digest does not match its descriptor-backed snapshot",
    )
    _require(
        active_case["oracle_sha256"] == expected_oracle_sha,
        "manifest oracle does not match the indexed case",
    )

    watch = _object(doc["watch"], "manifest.watch")
    _exact(watch, _WATCH_KEYS, "manifest.watch")
    template_ref = _text(watch["template_ref"], "manifest.watch.template_ref")
    pack_name = _text(watch["domain_pack"], "manifest.watch.domain_pack")
    _require(
        template_ref.startswith(pack_name + "/"),
        "manifest.watch.template_ref must belong to manifest.watch.domain_pack",
    )
    parameters = _object(watch["parameters"], "manifest.watch.parameters")
    try:
        domain_pack = get_domain_pack(pack_name)
        plan = watch_template(template_ref, parameters)
        route = _plan_route(plan)
    except (KeyError, TypeError, ValueError) as exc:
        raise MatchedWatchInputError(f"manifest Watch template cannot render: {exc}") from exc
    expected_conditions: dict[str, str]
    if route == "transition":
        expected_conditions = {
            _canonical_sha(str(plan.explicit["Goal.state.from"])): "from",
            _canonical_sha(str(plan.explicit["Goal.state.to"])): "to",
        }
    else:
        expected_conditions = {_canonical_sha(plan.condition): "condition"}

    oracle = _object(oracle_loaded.value, "oracle")
    _exact(oracle, _ORACLE_KEYS, "oracle")
    _require(oracle["schema"] == _ORACLE_SCHEMA, f"oracle.schema must be {_ORACLE_SCHEMA!r}")
    raw_candidates = oracle["candidates"]
    _require(
        isinstance(raw_candidates, list) and 0 < len(raw_candidates) <= 100_000,
        "oracle.candidates must contain 1..100000 decoded candidates",
    )
    oracle_frames: list[_OracleFrameSpec] = []
    prior_t: float | None = None
    for index, raw_candidate in enumerate(raw_candidates):
        context = f"oracle.candidates[{index}]"
        candidate = _object(raw_candidate, context)
        _exact(candidate, _ORACLE_CANDIDATE_KEYS, context)
        sequence = candidate["sequence"]
        _require(
            isinstance(sequence, int) and not isinstance(sequence, bool) and sequence == index,
            f"{context}.sequence must be the contiguous zero-based source sequence",
        )
        candidate_t = _number(candidate["t"], f"{context}.t", minimum=0.0)
        _require(
            prior_t is None or candidate_t > prior_t,
            "oracle candidate timestamps must be strictly increasing",
        )
        prior_t = candidate_t
        frame_sha = _sha(candidate["frame_sha256"], f"{context}.frame_sha256")
        verdict_doc = _object(candidate["verdicts"], f"{context}.verdicts")
        _require(
            set(verdict_doc) == set(expected_conditions),
            f"{context}.verdicts must contain exactly the rendered Watch predicate hashes",
        )
        verdicts = {
            condition_sha: _parse_verdict(
                verdict_doc[condition_sha],
                f"{context}.verdicts[{condition_sha!r}]",
            )
            for condition_sha in sorted(verdict_doc)
        }
        oracle_frames.append(
            _OracleFrameSpec(
                index,
                candidate_t,
                frame_sha,
                _text(candidate["token"], f"{context}.token"),
                MappingProxyType(verdicts),
            )
        )

    replay = _object(doc["replay"], "manifest.replay")
    _exact(replay, _REPLAY_KEYS, "manifest.replay")
    fps = _number(replay["fps"], "manifest.replay.fps", minimum=0.000001)
    _require(replay["timebase"] == "media", "manifest.replay.timebase must be 'media'")
    source_id = _text(replay["source_id"], "manifest.replay.source_id")

    judge = _object(doc["judge"], "manifest.judge")
    _exact(judge, _REAL_JUDGE_KEYS, "manifest.judge")
    judge_identity = {
        "adapter": _text(judge["adapter"], "manifest.judge.adapter"),
        "model": _text(judge["model"], "manifest.judge.model"),
        "prompt_sha256": _sha(judge["prompt_sha256"], "manifest.judge.prompt_sha256"),
        "judge_mode": _text(judge["judge_mode"], "manifest.judge.judge_mode"),
    }
    _require(
        judge_identity["adapter"] == "offline_oracle/v1"
        and judge_identity["judge_mode"] == "offline_oracle",
        "real-video mechanics require adapter 'offline_oracle/v1' and judge_mode 'offline_oracle'",
    )

    adaptive = _object(doc["adaptive"], "manifest.adaptive")
    _exact(adaptive, _ADAPTIVE_KEYS, "manifest.adaptive")
    salience_stat = _text(adaptive["salience_stat"], "manifest.adaptive.salience_stat")
    _require(salience_stat in {"mean", "block_max", "p95"}, "unsupported adaptive salience_stat")
    threshold = _number(adaptive["threshold"], "manifest.adaptive.threshold", minimum=0.0)
    _require(threshold <= 1.0, "manifest.adaptive.threshold must be <= 1")

    raw_events = doc["events"]
    _require(isinstance(raw_events, list) and raw_events, "manifest.events must be non-empty")
    events: list[_EventSpec] = []
    event_ids: set[str] = set()
    for index, raw_event in enumerate(raw_events):
        context = f"manifest.events[{index}]"
        event = _object(raw_event, context)
        _exact(event, _EVENT_KEYS, context)
        event_id = _text(event["id"], f"{context}.id")
        _require(event_id not in event_ids, f"duplicate event id {event_id!r}")
        event_ids.add(event_id)
        start = _number(event["window_start_s"], f"{context}.window_start_s", minimum=0.0)
        end = _number(event["window_end_s"], f"{context}.window_end_s", minimum=0.0)
        eligible = _number(event["eligible_at_s"], f"{context}.eligible_at_s", minimum=0.0)
        _require(start <= eligible <= end, f"{context}.eligible_at_s must be inside its window")
        events.append(
            _EventSpec(
                event_id,
                eligible,
                start,
                end,
                _text(event["review_authority"], f"{context}.review_authority"),
                _sha(event["label_digest"], f"{context}.label_digest"),
            )
        )
    for candidate in oracle_frames:
        owners = [event.id for event in events if event.start <= candidate.t <= event.end]
        _require(
            len(owners) <= 1,
            f"candidate t={candidate.t} belongs to multiple event windows {owners!r}",
        )
    for event in events:
        _require(
            any(event.start <= candidate.t <= event.end for candidate in oracle_frames),
            f"event {event.id!r} contains zero oracle candidates",
        )
    for index, left in enumerate(events):
        for right in events[index + 1 :]:
            _require(
                left.end < right.start or right.end < left.start,
                f"event windows {left.id!r} and {right.id!r} overlap",
            )

    classification = _text(
        doc["evidence_classification"],
        "manifest.evidence_classification",
    )
    _require(
        classification == "admission_only",
        "real-video offline-oracle evidence_classification must be 'admission_only'",
    )
    artifacts = [
        _artifact_receipt("manifest", loaded.receipt),
        _artifact_receipt("corpus", corpus_loaded.receipt),
        _artifact_receipt("media", media_receipt),
        _artifact_receipt("oracle", oracle_loaded.receipt),
        *authority_artifacts,
    ]
    identities = [
        (item["device"], item["inode"])
        for item in artifacts
    ]
    _require(
        len(identities) == len(set(identities)),
        "real-video input artifact files must be distinct",
    )
    return _Manifest(
        MappingProxyType(dict(doc)),
        loaded.receipt,
        media_receipt,
        case_id,
        split,
        fps,
        source_id,
        MappingProxyType(dict(parameters)),
        MappingProxyType(judge_identity),
        _text(adaptive["operating_point"], "manifest.adaptive.operating_point"),
        salience_stat,
        threshold,
        tuple(events),
        (),
        0,
        0,
        classification,
        _REAL_SCHEMA,
        template_ref,
        domain_pack,
        route,
        "offline_oracle",
        "video/mp4",
        tuple(oracle_frames),
        tuple(sorted(artifacts, key=lambda item: item["role"])),
        MappingProxyType(authority_presence),
    )


def load_matched_watch_manifest(path: str | Path) -> _Manifest:
    """Load and freeze a strict manifest plus its safely snapshotted synthetic media."""

    try:
        loaded = _load_json_snapshot(Path(path))
    except (SafeJSONInputError, MatchedWatchInputError) as exc:
        raise MatchedWatchInputError(f"unsafe matched-Watch manifest: {exc}") from exc
    doc = _object(loaded.value, "manifest")
    if doc.get("schema") == _REAL_SCHEMA:
        return _load_real_manifest(loaded, doc)
    _exact(doc, _MANIFEST_KEYS, "manifest")
    _require(doc["schema"] == _SCHEMA, f"manifest.schema must be {_SCHEMA!r}")
    case_id = _text(doc["case_id"], "manifest.case_id")
    _require(doc["archetype"] == _ARCHETYPE, f"manifest.archetype must be {_ARCHETYPE!r}")
    split = _text(doc["split"], "manifest.split")

    media_ref = _object(doc["media"], "manifest.media")
    _exact(media_ref, _MEDIA_REF_KEYS, "manifest.media")
    expected_media_sha = _sha(media_ref["sha256"], "manifest.media.sha256")
    media_path = Path(_text(media_ref["path"], "manifest.media.path"))
    if not media_path.is_absolute():
        media_path = Path(loaded.receipt.canonical_path).parent / media_path
    try:
        media_loaded = load_json_file(media_path)
    except SafeJSONInputError as exc:
        raise MatchedWatchInputError(f"unsafe matched-Watch media: {exc}") from exc
    _require(
        media_loaded.receipt.sha256 == expected_media_sha,
        "manifest media digest does not match the descriptor-backed media snapshot",
    )
    media = _object(media_loaded.value, "media")
    _exact(media, _MEDIA_KEYS, "media")
    _require(media["schema"] == _MEDIA_SCHEMA, f"media.schema must be {_MEDIA_SCHEMA!r}")
    _require(
        isinstance(media["width"], int) and not isinstance(media["width"], bool),
        "media.width must be an exact integer",
    )
    _require(
        isinstance(media["height"], int) and not isinstance(media["height"], bool),
        "media.height must be an exact integer",
    )
    width, height = media["width"], media["height"]
    _require(1 <= width <= 512 and 1 <= height <= 512, "media dimensions exceed the 512x512 L1 limit")
    _require(width * height <= 262_144, "media pixel budget exceeds 262144")

    raw_frames = media["frames"]
    _require(
        isinstance(raw_frames, list) and 0 < len(raw_frames) <= 10_000,
        "media.frames must contain 1..10000 candidates",
    )
    _require(
        len(raw_frames) * width * height <= 4_000_000,
        "synthetic replay exceeds the 4000000 total-pixel allocation budget",
    )
    frames: list[_FrameSpec] = []
    prior_t: float | None = None
    for index, raw in enumerate(raw_frames):
        context = f"media.frames[{index}]"
        frame = _object(raw, context)
        _exact(frame, _FRAME_KEYS, context)
        t = _number(frame["t"], f"{context}.t", minimum=0.0)
        _require(prior_t is None or t > prior_t, "media frame timestamps must be strictly increasing")
        prior_t = t
        fill = frame["fill"]
        _require(
            isinstance(fill, int) and not isinstance(fill, bool) and 0 <= fill <= 255,
            f"{context}.fill must be an integer in [0, 255]",
        )
        frames.append(
            _FrameSpec(
                index,
                t,
                fill,
                _text(frame["token"], f"{context}.token"),
                _parse_verdict(frame["from"], f"{context}.from"),
                _parse_verdict(frame["to"], f"{context}.to"),
            )
        )

    replay = _object(doc["replay"], "manifest.replay")
    _exact(replay, _REPLAY_KEYS, "manifest.replay")
    fps = _number(replay["fps"], "manifest.replay.fps", minimum=0.000001)
    _require(replay["timebase"] == "media", "manifest.replay.timebase must be 'media'")
    source_id = _text(replay["source_id"], "manifest.replay.source_id")

    watch = _object(doc["watch"], "manifest.watch")
    _exact(watch, _WATCH_KEYS, "manifest.watch")
    _require(watch["template_ref"] == _TEMPLATE, f"manifest.watch.template_ref must be {_TEMPLATE!r}")
    _require(watch["domain_pack"] == _PACK, f"manifest.watch.domain_pack must be {_PACK!r}")
    parameters = _object(watch["parameters"], "manifest.watch.parameters")

    judge = _object(doc["judge"], "manifest.judge")
    _exact(judge, _JUDGE_KEYS, "manifest.judge")
    judge_identity = {
        "adapter": _text(judge["adapter"], "manifest.judge.adapter"),
        "model": _text(judge["model"], "manifest.judge.model"),
        "prompt_sha256": _sha(judge["prompt_sha256"], "manifest.judge.prompt_sha256"),
    }
    _require(
        judge_identity["adapter"] == "scripted_transition/v1",
        "L1 matched-Watch manifests require adapter 'scripted_transition/v1'",
    )

    adaptive = _object(doc["adaptive"], "manifest.adaptive")
    _exact(adaptive, _ADAPTIVE_KEYS, "manifest.adaptive")
    stat = _text(adaptive["salience_stat"], "manifest.adaptive.salience_stat")
    _require(stat in {"mean", "block_max", "p95"}, "unsupported adaptive salience_stat")
    threshold = _number(adaptive["threshold"], "manifest.adaptive.threshold", minimum=0.0)
    _require(threshold <= 1.0, "manifest.adaptive.threshold must be <= 1")

    raw_events = doc["events"]
    _require(isinstance(raw_events, list) and raw_events, "manifest.events must be non-empty")
    events: list[_EventSpec] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_events):
        context = f"manifest.events[{index}]"
        event = _object(raw, context)
        _exact(event, _EVENT_KEYS, context)
        event_id = _text(event["id"], f"{context}.id")
        _require(event_id not in ids, f"duplicate event id {event_id!r}")
        ids.add(event_id)
        start = _number(event["window_start_s"], f"{context}.window_start_s", minimum=0.0)
        end = _number(event["window_end_s"], f"{context}.window_end_s", minimum=0.0)
        eligible = _number(event["eligible_at_s"], f"{context}.eligible_at_s", minimum=0.0)
        _require(start <= eligible <= end, f"{context}.eligible_at_s must be inside its window")
        events.append(
            _EventSpec(
                event_id,
                eligible,
                start,
                end,
                _text(event["review_authority"], f"{context}.review_authority"),
                _sha(event["label_digest"], f"{context}.label_digest"),
            )
        )
    for frame in frames:
        owners = [event.id for event in events if event.start <= frame.t <= event.end]
        _require(len(owners) <= 1, f"candidate t={frame.t} belongs to multiple event windows {owners!r}")
    for event in events:
        _require(
            any(event.start <= frame.t <= event.end for frame in frames),
            f"event {event.id!r} contains zero enumerated candidates",
        )
    for index, left in enumerate(events):
        for right in events[index + 1 :]:
            _require(
                left.end < right.start or right.end < left.start,
                f"event windows {left.id!r} and {right.id!r} overlap",
            )

    evidence_classification = _text(
        doc["evidence_classification"], "manifest.evidence_classification"
    )
    _require(
        evidence_classification in _EVIDENCE_CLASSES,
        f"evidence classification must be one of {_EVIDENCE_CLASSES!r}",
    )
    return _Manifest(
        MappingProxyType(doc),
        loaded.receipt,
        media_loaded.receipt,
        case_id,
        split,
        fps,
        source_id,
        MappingProxyType(parameters),
        MappingProxyType(judge_identity),
        _text(adaptive["operating_point"], "manifest.adaptive.operating_point"),
        stat,
        threshold,
        tuple(events),
        tuple(frames),
        width,
        height,
        evidence_classification,
    )


class _RecordingTransitionJudge:
    SUPPORTS_FUSED_AUDIO_INPUT = True

    def __init__(self, frames: Sequence[_FrameSpec], from_condition: str, to_condition: str) -> None:
        self._frames = tuple(frames)
        self._from = from_condition
        self._to = to_condition
        self.calls: list[ProviderCallReceipt] = []

    async def detect(self, frame: Any, goals: Any) -> list[Any]:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any) -> Verdict:
        if not isinstance(frame, _JudgePayload):
            raise TypeError("scripted transition judge requires the bound candidate payload")
        spec = self._frames[frame.index]
        if condition == self._from:
            predicate, answer = "from", spec.from_verdict
        elif condition == self._to:
            predicate, answer = "to", spec.to_verdict
        else:
            predicate, answer = "unknown", _VerdictSpec(False, 0.0, "", "unexpected condition")
        sequence = len(self.calls) + 1
        if answer.error is not None:
            self.calls.append(
                ProviderCallReceipt(
                    sequence,
                    frame.index,
                    frame.t,
                    predicate,
                    _canonical_sha(condition),
                    "error",
                    None,
                    None,
                    None,
                    answer.error,
                )
            )
            raise RuntimeError(answer.error)
        self.calls.append(
            ProviderCallReceipt(
                sequence,
                frame.index,
                frame.t,
                predicate,
                _canonical_sha(condition),
                "success",
                answer.satisfied,
                answer.confidence,
                answer.evidence,
                None,
            )
        )
        return Verdict(answer.satisfied, answer.confidence, answer.evidence)


@dataclass(frozen=True)
class _DecodedCoordinate:
    source_sequence: int
    t: float
    frame_sha256: str


class _OracleCoordinateTracker:
    def __init__(self) -> None:
        self.current: _DecodedCoordinate | None = None
        self.seen: list[_DecodedCoordinate] = []
        self.width: int | None = None
        self.height: int | None = None

    def bind(self, coordinate: _DecodedCoordinate, envelope: SourceEnvelope) -> None:
        width = envelope.format.width
        height = envelope.format.height
        _require(
            isinstance(width, int)
            and width > 0
            and isinstance(height, int)
            and height > 0,
            "runtime decoder did not report positive frame dimensions",
        )
        if self.width is None:
            self.width, self.height = width, height
        _require(
            (width, height) == (self.width, self.height),
            "runtime decoder changed frame dimensions within one case",
        )
        self.current = coordinate
        self.seen.append(coordinate)


def _decoded_frame_sha256(frame: Any) -> str:
    _require(
        isinstance(frame, tuple)
        and len(frame) == 2
        and isinstance(frame[0], str)
        and isinstance(frame[1], bytes),
        "real-video decoder must supply the canonical (media_type, bytes) judge frame",
    )
    return f"sha256:{hashlib.sha256(frame[1]).hexdigest()}"


def _decoder_identity() -> Mapping[str, str]:
    try:
        import cv2  # noqa: PLC0415 - runtime-extra identity is evaluated only for real video
    except ImportError as exc:
        raise MatchedWatchInputError(
            "real-video matched evaluation requires OpenCV; "
            "install 'percept-vision[runtime]'"
        ) from exc
    try:
        build = cv2.getBuildInformation()
        version = str(cv2.__version__)
    except Exception as exc:  # noqa: BLE001 - normalize optional decoder failures
        raise MatchedWatchInputError(f"cannot identify the active OpenCV decoder: {exc}") from exc
    _require(bool(build) and bool(version), "active OpenCV decoder identity is incomplete")
    return MappingProxyType(
        {
            "library": "opencv",
            "version": version,
            "build_sha256": f"sha256:{hashlib.sha256(build.encode('utf-8')).hexdigest()}",
            "source_adapter": "percept.runtime.source.open_source/v1",
            "frame_encoding": "cv2.imencode:image/jpeg:backend-default",
        }
    )


class _RecordingOfflineOracleJudge:
    """Offline mechanics judge keyed by exact decoded evidence, never by event labels."""

    SUPPORTS_FUSED_AUDIO_INPUT = True

    def __init__(
        self,
        manifest: _Manifest,
        tracker: _OracleCoordinateTracker,
        predicates: Mapping[str, str],
    ) -> None:
        self._manifest = manifest
        self._tracker = tracker
        self._predicates = dict(predicates)
        self._frames = {item.index: item for item in manifest.oracle_frames}
        self.calls: list[ProviderCallReceipt] = []
        self._seen_lookups: set[tuple[int, str]] = set()

    async def detect(self, frame: Any, goals: Any) -> list[Any]:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any) -> Verdict:
        coordinate = self._tracker.current
        if coordinate is None:
            raise RuntimeError("offline oracle has no current decoded source coordinate")
        actual_frame_sha = _decoded_frame_sha256(frame)
        if actual_frame_sha != coordinate.frame_sha256:
            raise RuntimeError("offline oracle frame bytes disagree with the current source coordinate")
        spec = self._frames.get(coordinate.source_sequence)
        if spec is None:
            raise RuntimeError("offline oracle has no candidate for the decoded source sequence")
        if not math.isclose(spec.t, coordinate.t, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError("offline oracle source timestamp mismatch")
        if spec.frame_sha256 != coordinate.frame_sha256:
            raise RuntimeError("offline oracle frame digest mismatch")
        condition_sha = _canonical_sha(condition)
        predicate = self._predicates.get(condition_sha)
        if predicate is None:
            raise RuntimeError("offline oracle received an undeclared Watch predicate")
        answer = spec.verdicts.get(condition_sha)
        if answer is None:
            raise RuntimeError("offline oracle candidate is missing the rendered Watch predicate")
        lookup_key = (coordinate.source_sequence, condition_sha)
        if lookup_key in self._seen_lookups:
            raise RuntimeError("offline oracle predicate was consumed more than once for one candidate")
        self._seen_lookups.add(lookup_key)
        lookup_sha = _canonical_sha(
            {
                "case_id": self._manifest.case_id,
                "source_sequence": coordinate.source_sequence,
                "t": coordinate.t,
                "frame_sha256": coordinate.frame_sha256,
                "predicate_sha256": condition_sha,
            }
        )
        sequence = len(self.calls) + 1
        if answer.error is not None:
            self.calls.append(
                ProviderCallReceipt(
                    sequence,
                    coordinate.source_sequence,
                    coordinate.t,
                    predicate,
                    condition_sha,
                    "error",
                    None,
                    None,
                    None,
                    answer.error,
                    lookup_sha,
                )
            )
            raise RuntimeError(answer.error)
        self.calls.append(
            ProviderCallReceipt(
                sequence,
                coordinate.source_sequence,
                coordinate.t,
                predicate,
                condition_sha,
                "success",
                answer.satisfied,
                answer.confidence,
                answer.evidence,
                None,
                lookup_sha,
            )
        )
        return Verdict(answer.satisfied, answer.confidence, answer.evidence)


async def _real_video_source(
    manifest: _Manifest,
    tracker: _OracleCoordinateTracker,
) -> Any:
    """Decode the frozen MP4 through the shipped runtime source adapter without changing envelopes."""

    media_artifact = next(
        item for item in manifest.input_artifacts if item["role"] == "media"
    )
    media_path = Path(media_artifact["canonical_path"])
    try:
        import cv2  # noqa: F401, PLC0415 - fail before starting the decoder thread
        from percept.runtime.source import open_source
    except ImportError as exc:
        raise MatchedWatchInputError(
            "real-video matched evaluation requires the runtime decoder; "
            "install 'percept-vision[runtime]'"
        ) from exc
    expected = manifest.oracle_frames
    count = 0
    try:
        with _private_decoder_snapshot(media_path, expected=media_artifact) as decoder_path:
            monitor_stop = asyncio.Event()
            monitor_violations: list[BaseException] = []
            monitor_task = asyncio.create_task(
                _monitor_recorded_media(
                    media_path,
                    expected=media_artifact,
                    stop=monitor_stop,
                    violation=monitor_violations,
                )
            )
            try:
                async for envelope in open_source(
                    decoder_path,
                    fps=manifest.fps,
                    # The shipped capture adapter is bounded/latest-wins. Keep its production-sized queue
                    # instead of buffering an entire corpus case in JPEG memory; any overflow becomes an
                    # observable source-sequence gap and this lossless evaluator fails the arm closed.
                    maxsize=64,
                    reconnects=0,
                    timebase="media",
                ):
                    if monitor_violations:
                        raise monitor_violations[0]
                    _require(
                        isinstance(envelope, SourceEnvelope),
                        "runtime decoder returned a non-envelope",
                    )
                    _require(
                        envelope.modality == "video",
                        "runtime decoder returned non-video evidence",
                    )
                    _require(
                        count < len(expected),
                        "runtime decoder produced more candidates than the oracle",
                    )
                    spec = expected[count]
                    _require(
                        envelope.sequence == spec.index,
                        "runtime decoder source sequence disagrees with the oracle",
                    )
                    decoded_t = envelope.event_time
                    _require(
                        math.isclose(decoded_t, spec.t, rel_tol=0.0, abs_tol=1e-6),
                        "runtime decoder source timestamp disagrees with the oracle",
                    )
                    frame_sha = _decoded_frame_sha256(envelope.payload)
                    _require(
                        frame_sha == spec.frame_sha256,
                        "runtime decoder frame digest disagrees with the oracle",
                    )
                    tracker.bind(
                        _DecodedCoordinate(envelope.sequence, decoded_t, frame_sha),
                        envelope,
                    )
                    count += 1
                    # The canonical SourceEnvelope and both gate/judge payloads pass through unchanged.
                    yield envelope
            finally:
                monitor_stop.set()
                await monitor_task
            if monitor_violations:
                raise monitor_violations[0]
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise MatchedWatchInputError(
                "real-video matched evaluation requires OpenCV; "
                "install 'percept-vision[runtime]'"
            ) from exc
        raise
    _require(count == len(expected), "runtime decoder produced fewer candidates than the oracle")
    after = _snapshot_binary(media_path)
    _require(
        after.sha256 == manifest.media_receipt.sha256
        and after.byte_count == manifest.media_receipt.byte_count
        and after.device == manifest.media_receipt.device
        and after.inode == manifest.media_receipt.inode
        and after.modified_time_ns == manifest.media_receipt.modified_time_ns,
        "MP4 artifact changed during matched execution",
    )


def _revalidate_manifest_artifacts(manifest: _Manifest, *, phase: str) -> None:
    """Require every closed v2 input to retain its descriptor/content identity."""

    if manifest.schema != _REAL_SCHEMA:
        return
    for recorded in manifest.input_artifacts:
        role = str(recorded["role"])
        max_bytes = (
            64 * 1024 * 1024 * 1024
            if role == "media"
            else 64 * 1024 * 1024
            if role.startswith("authority:")
            else SAFE_JSON_MAX_BYTES
        )
        observed = _snapshot_binary(
            Path(str(recorded["path"])),
            max_bytes=max_bytes,
        )
        expected = {key: recorded[key] for key in _SAFE_INPUT_RECEIPT_KEYS}
        _require(
            _safe_receipt_dict(observed) == expected,
            f"{phase}: input artifact {role!r} changed after its frozen snapshot",
        )


def _profile(arm: str, stat: str, threshold: float) -> SourceProfile:
    return SourceProfile(
        name=f"matched-watch-{stat}",
        modalities=frozenset({"video"}),
        temporal_unit="frame",
        admission_strategy="pixel-change",
        evidence_strategy="frame",
        identity_strategy="scene",
        default_timebase="media",
        parameters={"salience_stat": stat, "salience_threshold": threshold},
        notes="preregistered matched-Watch L1 execution profile",
    )


def _trace_dict(event: Any) -> dict[str, Any]:
    return {
        "trace_id": event.trace_id,
        "stage": event.stage,
        "t": event.t,
        "goal_id": event.goal_id,
        "data": dict(event.data),
    }


def _health_dict(
    event: RuntimeHealthEvent,
    manifest: _Manifest,
    *,
    media_identity: Mapping[str, Any] | None = None,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    data = dict(event.data or {})
    if event.kind == "runtime_started":
        # Runtime emits repr(source), which contains every synthetic pixel. Store a compact,
        # reconstructable identity at the receipt boundary instead of retaining or later masking it.
        data = {
            "source_kind": (
                "real_video_offline_oracle"
                if manifest.schema == _REAL_SCHEMA
                else "synthetic_video"
            ),
            "media": _thaw(media_identity)
            if media_identity is not None
            else {
                "sha256": manifest.media_receipt.sha256,
                "byte_count": manifest.media_receipt.byte_count,
                "schema": _MEDIA_SCHEMA,
                "width": manifest.width,
                "height": manifest.height,
            },
            "replay": {
                "fps": manifest.fps,
                "timebase": "media",
                "source_id": manifest.source_id,
            },
            "candidate_count": (
                candidate_count if candidate_count is not None else len(manifest.frames)
            ),
        }
    return {"kind": event.kind, "ts": event.ts, "data": data}


def _incident_dict(incident: Any) -> dict[str, Any]:
    return {
        "id": incident.id,
        "concern_id": incident.concern_id,
        "ts": float(incident.ts),
        "message": incident.message,
        "entity_id": incident.entity_id,
        "reason": incident.reason,
        "watchplan_id": incident.watchplan_id,
        "judgment_event_id": getattr(incident, "judgment_event_id", None),
        "evidence_origin": getattr(incident, "evidence_origin", None),
    }


def _profile_dict(profile: SourceProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "modalities": sorted(profile.modalities),
        "temporal_unit": profile.temporal_unit,
        "admission_strategy": profile.admission_strategy,
        "evidence_strategy": profile.evidence_strategy,
        "identity_strategy": profile.identity_strategy,
        "default_timebase": profile.default_timebase,
        "parameters": dict(profile.parameters),
        "stability": profile.stability,
        "notes": profile.notes,
    }


async def _run_arm(manifest: _Manifest, plan: Any, arm: str) -> ArmReceipt:
    threshold = 0.0 if arm == "baseline" else manifest.threshold
    profile = _profile(arm, manifest.salience_stat, threshold)
    tracker: _OracleCoordinateTracker | None = None
    if manifest.schema == _REAL_SCHEMA:
        tracker = _OracleCoordinateTracker()
        if manifest.plan_route == "transition":
            predicate_conditions = {
                _canonical_sha(str(plan.explicit["Goal.state.from"])): "from",
                _canonical_sha(str(plan.explicit["Goal.state.to"])): "to",
            }
        else:
            predicate_conditions = {_canonical_sha(plan.condition): "condition"}
        adapter: Any = _RecordingOfflineOracleJudge(
            manifest,
            tracker,
            predicate_conditions,
        )
    else:
        from_condition = str(plan.explicit["Goal.state.from"])
        to_condition = str(plan.explicit["Goal.state.to"])
        adapter = _RecordingTransitionJudge(manifest.frames, from_condition, to_condition)
    base_settings = Settings()
    edge_updates: dict[str, Any] = {"salience_stat": manifest.salience_stat}
    edge_updates[
        {
            "mean": "motion_gate",
            "block_max": "motion_gate_block_max",
            "p95": "motion_gate_p95",
        }[manifest.salience_stat]
    ] = threshold
    runner_settings = replace(
        base_settings,
        edge=replace(base_settings.edge, **edge_updates),
    )
    core = Percept.create(
        vision=adapter,
        settings=runner_settings,
        discover_plugins=False,
        mode="offline",
    )
    resolver = ConfigResolver()
    resolved = resolver.resolve(
        plan,
        source_profile=profile,
        domain_pack=manifest.domain_pack,
        settings=core.settings,
    )
    apply_receipt = apply_watch_plan(
        core,
        plan,
        resolver=resolver,
        source_profile=profile,
        domain_pack=manifest.domain_pack,
        settings=core.settings,
    )
    _require(
        apply_receipt.status in {"applied", "applied_with_warnings"},
        f"{arm} WatchPlan apply failed: {apply_receipt.to_dict()!r}",
    )
    _require(len(apply_receipt.watch_ids) == 1, f"{arm} armed an unexpected Watch set")
    goal_id = apply_receipt.watch_ids[0]
    bound = core._watch_runtime_bindings.get(goal_id)
    _require(bound is not None, f"{arm} compiled runtime binding is missing")
    compiled = bound[0]

    stats = RuntimeStats()
    traces: list[Any] = []
    health: list[RuntimeHealthEvent] = []
    asks: list[Any] = []
    if manifest.schema == _REAL_SCHEMA:
        assert tracker is not None
        source: Any = _real_video_source(manifest, tracker)
        expected_candidate_count = len(manifest.oracle_frames)
    else:
        source = [
            (
                frame.t,
                gray_frame(manifest.width, manifest.height, frame.fill),
                _JudgePayload(frame.index, frame.token, frame.t),
            )
            for frame in manifest.frames
        ]
        expected_candidate_count = len(manifest.frames)
    incidents = [
        incident
        async for incident in run(
            core,
            source=source,
            fps=manifest.fps,
            gate_source=manifest.source_id,
            timebase="media",
            judge_dispatch="inline",
            stats=stats,
            on_trace=traces.append,
            on_health=health.append,
            on_ask=asks.append,
            regate_on_goal_change=False,
            coverage_floor=compiled.coverage_floor,
        )
    ]

    edge = [item for item in traces if item.stage == "edge.decision" and item.goal_id is None]
    if manifest.plan_route == "transition":
        decisions = [
            item
            for item in traces
            if item.stage == "transition.decision" and item.goal_id == goal_id
        ]
    else:
        decisions = [
            item
            for item in traces
            if item.stage == "gate.classified" and item.goal_id == goal_id
        ]
    if manifest.schema == _REAL_SCHEMA:
        assert tracker is not None
        candidate_ts = tuple(item.t for item in tracker.seen)
        candidate_specs: Sequence[Any] = manifest.oracle_frames
    else:
        candidate_ts = tuple(frame.t for frame in manifest.frames)
        candidate_specs = manifest.frames
    blockers: list[str] = []
    if len(edge) != expected_candidate_count:
        blockers.append("edge trace count does not equal candidate count")
    edge_by_t = {float(item.t): item for item in edge if item.t is not None}
    if tuple(sorted(edge_by_t)) != candidate_ts:
        blockers.append("edge trace timestamps do not equal enumerated candidate timestamps")
    decision_by_t = {float(item.t): item for item in decisions if item.t is not None}
    incident_by_t: dict[float, list[Any]] = {}
    for incident in incidents:
        incident_by_t.setdefault(float(incident.ts), []).append(incident)
    ask_by_t: dict[float, list[Any]] = {}
    for ask in asks:
        ask_t = getattr(ask, "ts", None)
        if isinstance(ask_t, (int, float)):
            ask_by_t.setdefault(float(ask_t), []).append(ask)
    calls_by_index: dict[int, list[ProviderCallReceipt]] = {}
    for call in adapter.calls:
        calls_by_index.setdefault(call.candidate_index, []).append(call)

    candidates: list[CandidateReceipt] = []
    for frame in candidate_specs:
        edge_event = edge_by_t.get(frame.t)
        edge_data = dict(edge_event.data) if edge_event is not None else {}
        admitted = goal_id in tuple(edge_data.get("goal_ids", ()))
        decision_trace = decision_by_t.get(frame.t)
        dispatched = decision_trace is not None
        frame_calls = tuple(calls_by_index.get(frame.index, ()))
        frame_incidents = tuple(item.id for item in incident_by_t.get(frame.t, ()))
        if dispatched:
            if frame_incidents:
                decision = "fire"
            elif ask_by_t.get(frame.t):
                decision = "ask"
            elif (
                manifest.plan_route == "transition"
                and decision_trace.data.get("outcome") == "judge_degraded"
            ) or any(
                call.status != "success" for call in frame_calls
            ):
                decision = "missing"
            elif manifest.plan_route == "watch":
                decision = str(decision_trace.data.get("action", "silent"))
            else:
                decision = "silent"
        else:
            decision = None
        candidates.append(
            CandidateReceipt(
                frame.index,
                frame.t,
                _freeze(edge_data),
                admitted,
                dispatched,
                frame_calls,
                decision,
                (
                    str(decision_trace.data.get("outcome"))
                    if manifest.plan_route == "transition" and decision_trace is not None
                    else None
                ),
                frame_incidents,
                frame.index if manifest.schema == _REAL_SCHEMA else None,
                frame.frame_sha256 if manifest.schema == _REAL_SCHEMA else None,
            )
        )

    admitted_count = sum(item.admitted for item in candidates)
    dispatch_count = sum(item.runtime_dispatched for item in candidates)
    failures = sum(item.status != "success" for item in adapter.calls)
    successes = len(adapter.calls) - failures
    if stats.frames != len(candidates):
        blockers.append("RuntimeStats.frames does not equal candidate count")
    if stats.goal_admissions != admitted_count or stats.escalations != admitted_count:
        blockers.append("RuntimeStats admission counts do not equal edge traces")
    if stats.perceive_dispatches != dispatch_count or stats.judge_calls != dispatch_count:
        blockers.append("RuntimeStats dispatch counts do not equal transition traces")
    if dispatch_count != admitted_count:
        blockers.append("admitted candidates and inline runtime dispatches differ")
    expected_calls_per_dispatch = 2 if manifest.plan_route == "transition" else 1
    if len(adapter.calls) != expected_calls_per_dispatch * dispatch_count:
        blockers.append("Watch judgment subcall count is incomplete")
    if failures:
        blockers.append("provider subcall failure")
    if stats.frames_dropped or stats.sink_errors:
        blockers.append("runtime dropped frames or reported sink errors")
    if (
        stats.judge_superseded
        or stats.judge_revision_dropped
        or stats.judge_source_discontinuity_dropped
    ):
        blockers.append("async/revision work loss is forbidden")
    if stats.source_excluded or stats.source_discontinuities or stats.source_missing_sequences:
        blockers.append("source exclusion, gap, or discontinuity is forbidden")
    if any(item.kind == "runtime_error" for item in health):
        blockers.append("runtime_error health event")
    if arm == "baseline":
        if threshold != 0.0:
            blockers.append("baseline threshold is not exactly 0.0")
        if admitted_count != len(candidates) or dispatch_count != len(candidates):
            blockers.append("baseline is not exhaustive over sampled candidates")
        if stats.coast_admits or stats.fixed_admits or stats.audio_wakes:
            blockers.append("baseline used a forbidden non-signal admission path")
    if len(candidate_ts) != expected_candidate_count:
        blockers.append("decoded candidate count does not equal the oracle")
    if manifest.schema == _REAL_SCHEMA:
        assert tracker is not None
        if tracker.width is None or tracker.height is None:
            blockers.append("runtime decoder reported no frame dimensions")

    resolved_dict = resolved.to_dict()
    compiled_dict = {
        "goal_id": compiled.goal.id,
        "timebase": compiled.timebase,
        "coverage_floor": compiled.coverage_floor,
        "watchspec_kwargs": dict(compiled.watchspec_kwargs),
        "components": dict(compiled.components),
    }
    config_sha = _canonical_sha(
        {
            "effective": _json_shape(resolved_dict["effective"]),
            "source_profile": _json_shape(_profile_dict(profile)),
            "compiled_binding": _json_shape(compiled_dict),
        }
    )
    if manifest.schema == _REAL_SCHEMA:
        assert tracker is not None
        media_identity = {
            "sha256": manifest.media_receipt.sha256,
            "byte_count": manifest.media_receipt.byte_count,
            "schema": manifest.media_type,
            "width": tracker.width,
            "height": tracker.height,
            "decoder_identity": _thaw(_decoder_identity()),
        }
    else:
        media_identity = {
            "sha256": manifest.media_receipt.sha256,
            "byte_count": manifest.media_receipt.byte_count,
            "schema": _MEDIA_SCHEMA,
            "width": manifest.width,
            "height": manifest.height,
        }
    return ArmReceipt(
        arm,
        _freeze(_profile_dict(profile)),
        _freeze(resolved_dict),
        _freeze(apply_receipt.to_dict()),
        _freeze(compiled_dict),
        _freeze(manifest.judge_identity),
        _freeze(media_identity),
        _freeze(
            {
                "fps": manifest.fps,
                "timebase": "media",
                "source_id": manifest.source_id,
            }
        ),
        config_sha,
        candidate_ts,
        tuple(candidates),
        tuple(_freeze(_incident_dict(item)) for item in incidents),
        tuple(
            _freeze(
                {
                    "goal_id": getattr(item, "goal_id", None),
                    "action": getattr(item, "action", None),
                    "text": getattr(item, "text", None),
                }
            )
            for item in asks
        ),
        tuple(_freeze(_trace_dict(item)) for item in traces),
        tuple(
            _freeze(
                _health_dict(
                    item,
                    manifest,
                    media_identity=media_identity,
                    candidate_count=len(candidates),
                )
            )
            for item in health
        ),
        _freeze(asdict(stats)),
        dispatch_count,
        len(adapter.calls),
        successes,
        failures,
        not blockers,
        tuple(blockers),
    )


def _sanitized_effective(arm: ArmReceipt) -> dict[str, Any]:
    effective = dict(arm.resolved_watch["effective"])
    stat = arm.source_profile["parameters"]["salience_stat"]
    active = {
        "mean": "Settings.edge.motion_gate",
        "block_max": "Settings.edge.motion_gate_block_max",
        "p95": "Settings.edge.motion_gate_p95",
    }[stat]
    effective[active] = "<matched-arm-threshold>"
    return effective


def _sanitized_profile(arm: ArmReceipt) -> dict[str, Any]:
    profile = _thaw(arm.source_profile)
    profile["parameters"].pop("salience_threshold", None)
    return profile


def _sanitized_binding(arm: ArmReceipt) -> dict[str, Any]:
    binding = _thaw(arm.compiled_binding)
    kwargs = binding["watchspec_kwargs"]
    stat = arm.source_profile["parameters"]["salience_stat"]
    kwargs["stat_thresholds"][stat] = "<matched-arm-threshold>"
    if stat == "mean":
        kwargs["threshold"] = "<matched-arm-threshold>"
    return binding


def _binding_consistent(arm: ArmReceipt) -> bool:
    profile = arm.source_profile
    stat = profile["parameters"]["salience_stat"]
    threshold = float(profile["parameters"]["salience_threshold"])
    kwargs = arm.compiled_binding["watchspec_kwargs"]
    effective = arm.resolved_watch["effective"]
    threshold_path = {
        "mean": "Settings.edge.motion_gate",
        "block_max": "Settings.edge.motion_gate_block_max",
        "p95": "Settings.edge.motion_gate_p95",
    }[stat]
    return (
        kwargs["salience_stat"] == stat
        and float(kwargs["stat_thresholds"][stat]) == threshold
        and effective["Settings.edge.salience_stat"] == stat
        and float(effective[threshold_path]) == threshold
    )


def validate_matched_arms(
    baseline: ArmReceipt,
    adaptive: ArmReceipt,
    *,
    watch_plan_sha256: str | None = None,
    expected_media_sha256: str | None = None,
) -> tuple[str, ...]:
    """Return every comparability blocker; an empty tuple means the arms are matched."""

    blockers: list[str] = []
    if baseline.arm != "baseline" or adaptive.arm != "adaptive":
        blockers.append("receipt arms must be baseline then adaptive")
    if baseline.candidate_timestamps != adaptive.candidate_timestamps:
        blockers.append("candidate timestamp sequences differ")
    if baseline.media_identity != adaptive.media_identity:
        blockers.append("decoded media identities differ")
    if baseline.replay_identity != adaptive.replay_identity:
        blockers.append("replay identities differ")
    baseline_frames = tuple(
        (item.source_sequence, item.t, item.frame_sha256)
        for item in baseline.candidates
    )
    adaptive_frames = tuple(
        (item.source_sequence, item.t, item.frame_sha256)
        for item in adaptive.candidates
    )
    if baseline_frames != adaptive_frames:
        blockers.append("decoded candidate evidence differs")
    baseline_plan = baseline.resolved_watch.get("plan")
    adaptive_plan = adaptive.resolved_watch.get("plan")
    if baseline_plan != adaptive_plan:
        blockers.append("rendered WatchPlan differs")
    if baseline.resolved_watch.get("plan_hash") != adaptive.resolved_watch.get("plan_hash"):
        blockers.append("resolved plan hashes differ")
    if baseline.resolved_watch.get("component_snapshot_hash") != adaptive.resolved_watch.get(
        "component_snapshot_hash"
    ):
        blockers.append("component snapshots differ")
    if _sanitized_effective(baseline) != _sanitized_effective(adaptive):
        blockers.append("arms differ outside approved admission configuration")
    if _sanitized_profile(baseline) != _sanitized_profile(adaptive):
        blockers.append("SourceProfiles differ outside approved admission configuration")
    if _sanitized_binding(baseline) != _sanitized_binding(adaptive):
        blockers.append("compiled bindings differ outside approved admission configuration")
    if baseline.judge_identity != adaptive.judge_identity:
        blockers.append("judge adapter/model/prompt identities differ")
    if not _binding_consistent(baseline) or not _binding_consistent(adaptive):
        blockers.append("SourceProfile/resolved/compiled admission bindings are inconsistent")
    if float(baseline.source_profile["parameters"]["salience_threshold"]) != 0.0:
        blockers.append("baseline compiled admission threshold is not exactly 0.0")
    if (
        watch_plan_sha256 is not None
        and f"sha256:{baseline.resolved_watch.get('plan_hash')}" != watch_plan_sha256
    ):
        blockers.append("resolved WatchPlan does not match the receipt WatchPlan identity")
    if expected_media_sha256 is not None:
        if not expected_media_sha256.startswith("sha256:"):
            blockers.append("media identity is malformed")
        elif (
            baseline.media_identity.get("sha256") != expected_media_sha256
            or adaptive.media_identity.get("sha256") != expected_media_sha256
        ):
            blockers.append("arm media identity does not match the expected media")
    blockers.extend(f"baseline: {item}" for item in baseline.blockers)
    blockers.extend(f"adaptive: {item}" for item in adaptive.blockers)
    return tuple(blockers)


async def run_matched_watch_async(path: str | Path) -> MatchedWatchReceipt:
    manifest = load_matched_watch_manifest(path)
    _revalidate_manifest_artifacts(manifest, phase="before baseline")
    canonical_plan = watch_template(manifest.template_ref, manifest.watch_parameters)
    # Fresh render + core per arm. Sequential replay keeps append order deterministic and prevents
    # any shallow nested state in a future WatchPlan field from becoming a cross-arm channel.
    baseline = await _run_arm(
        manifest,
        watch_template(manifest.template_ref, manifest.watch_parameters),
        "baseline",
    )
    _revalidate_manifest_artifacts(manifest, phase="between matched arms")
    adaptive = await _run_arm(
        manifest,
        watch_template(manifest.template_ref, manifest.watch_parameters),
        "adaptive",
    )
    _revalidate_manifest_artifacts(manifest, phase="after adaptive")
    plan_dict = canonical_plan.to_dict()
    plan_sha = _canonical_sha(plan_dict)
    blockers = validate_matched_arms(
        baseline,
        adaptive,
        watch_plan_sha256=plan_sha,
        expected_media_sha256=manifest.media_receipt.sha256,
    )
    provenance = build_info().to_dict()
    package_version = provenance["package_version"]
    events = tuple(
        MappingProxyType(
            {
                "id": event.id,
                "eligible_at_s": event.eligible_at_s,
                "window_start_s": event.start,
                "window_end_s": event.end,
                "review_authority": event.review_authority,
                "label_digest": event.label_digest,
            }
        )
        for event in manifest.events
    )
    receipt = MatchedWatchReceipt(
        _freeze(dict(manifest.document)),
        _freeze(_safe_receipt_dict(manifest.input_receipt)),
        _freeze(_safe_receipt_dict(manifest.media_receipt)),
        _freeze(plan_dict),
        plan_sha,
        _freeze(manifest.judge_identity),
        events,
        manifest.evidence_classification,
        package_version,
        _freeze(provenance),
        baseline,
        adaptive,
        not blockers,
        blockers,
        datetime.now(timezone.utc).isoformat(),
        "",
        _RECEIPT_SCHEMA_V2 if manifest.schema == _REAL_SCHEMA else _RECEIPT_SCHEMA,
        manifest.input_artifacts,
        manifest.authority_artifact_presence,
    )
    return replace(receipt, origin_evidence_sha256=receipt.evidence_sha256)


def run_matched_watch(path: str | Path) -> MatchedWatchReceipt:
    """Synchronous entry point for the finite L1 replay."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_matched_watch_async(path))
    raise RuntimeError("run_matched_watch() cannot be called from an active event loop; await the async API")


def _event_rows(receipt: MatchedWatchReceipt, arm: ArmReceipt) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    fire_ids: list[str] = []
    known_incidents = {item["id"]: item for item in arm.incidents}
    aligned_incidents: set[str] = set()
    for event in receipt.event_windows:
        candidates = [
            item
            for item in arm.candidates
            if event["window_start_s"] <= item.t <= event["window_end_s"]
        ]
        _require(candidates, f"event {event['id']!r} contains zero enumerated candidates")
        fires = [
            incident
            for incident in arm.incidents
            if event["window_start_s"] <= float(incident["ts"]) <= event["window_end_s"]
        ]
        _require(len(fires) <= 1, f"event {event['id']!r} has multiple aligned incidents")
        _require(
            not any(str(item["id"]) in aligned_incidents for item in fires),
            "one incident aligned to multiple event windows",
        )
        asks = [item for item in candidates if item.decision == "ask"]
        missing = [item for item in candidates if item.decision == "missing"]
        if fires:
            outcome = "fire"
            fire_ids.append(str(event["id"]))
            aligned_incidents.update(str(item["id"]) for item in fires)
        elif asks:
            outcome = "ask"
        elif missing:
            outcome = "missing"
        else:
            outcome = "silent"
        events.append(
            {
                "id": event["id"],
                "candidate_windows": len(candidates),
                "admitted_candidates": sum(item.admitted for item in candidates),
                "outcome": outcome,
            }
        )
    for incident_id in known_incidents:
        if incident_id not in aligned_incidents:
            fire_ids.append(incident_id)
    return events, fire_ids


def _validate_v2_arm_reconstruction(
    receipt: MatchedWatchReceipt,
    arm: ArmReceipt,
    *,
    manifest: Mapping[str, Any],
    route: str,
) -> None:
    """Reconstruct deterministic runtime claims from the closed real-video evidence."""

    candidates = arm.candidates
    dispatched = [candidate for candidate in candidates if candidate.runtime_dispatched]
    admitted = [candidate for candidate in candidates if candidate.admitted]
    goal_id = str(arm.compiled_binding["goal_id"])
    source_id = str(manifest["replay"]["source_id"])

    expected_admission_by_goal: dict[str, int] = {}
    expected_admission_by_cause: dict[str, int] = {}
    expected_escalations: list[tuple[float, dict[str, Any]]] = []
    signal_admits = 0
    fixed_admits = 0
    coast_admits = 0
    fixed_opportunities = 0
    goal_admissions = 0
    for candidate in candidates:
        decision = _thaw(candidate.edge_decision)
        candidate_goal_ids = tuple(str(item) for item in decision["goal_ids"])
        causes = tuple(str(item) for item in decision["admission_causes"])
        _require(
            len(candidate_goal_ids) == len(causes)
            and candidate.admitted == (goal_id in candidate_goal_ids)
            and candidate.runtime_dispatched == candidate.admitted,
            f"{arm.arm} candidate admission/dispatch state is not reconstructable",
        )
        if not candidate.admitted:
            continue
        goal_admissions += len(candidate_goal_ids)
        for admitted_goal, cause in zip(candidate_goal_ids, causes):
            expected_admission_by_goal[admitted_goal] = (
                expected_admission_by_goal.get(admitted_goal, 0) + 1
            )
            expected_admission_by_cause[cause] = (
                expected_admission_by_cause.get(cause, 0) + 1
            )
            if cause == "fixed":
                fixed_opportunities += 1
        cause_set = set(causes)
        if bool(decision["escalate"]) and cause_set - {"fixed"}:
            signal_admits += 1
            health_cause = "motion"
        elif bool(decision["escalate"]) and cause_set == {"fixed"}:
            fixed_admits += 1
            health_cause = "fixed"
        elif bool(decision["coast_admit"]):
            coast_admits += 1
            health_cause = "coast"
        else:
            _fail(f"{arm.arm} admitted candidate has no reconstructable admission cause")
        expected_escalations.append(
            (candidate.t, {"source": source_id, "cause": health_cause})
        )

    stats = _thaw(arm.runtime_stats)
    exact_stats = {
        "frames": len(candidates),
        "frames_dropped": 0,
        "escalations": len(admitted),
        "judge_calls": len(dispatched),
        "fires": len(arm.incidents),
        "asks": len(arm.asks),
        "incidents": len(arm.incidents),
        "sink_errors": 0,
        "audio_chunks": 0,
        "audio_wakes": 0,
        "signal_admits": signal_admits,
        "fixed_admits": fixed_admits,
        "coast_admits": coast_admits,
        "boundaries": 0,
        "utterances": 0,
        "preroll_frames_dropped": 0,
        "judge_superseded": 0,
        "judge_revision_dropped": 0,
        "judge_source_discontinuity_dropped": 0,
        "judge_source_discontinuity_active_dropped": 0,
        "judge_source_discontinuity_pending_dropped": 0,
        "source_observed": len(candidates),
        "source_excluded": 0,
        "source_discontinuities": 0,
        "source_missing_sequences": 0,
        "fixed_opportunities": fixed_opportunities,
        "goal_admissions": goal_admissions,
        "admission_by_goal": expected_admission_by_goal,
        "admission_by_cause": expected_admission_by_cause,
        "perceive_dispatches": len(dispatched),
    }
    _require(
        all(stats.get(key) == value for key, value in exact_stats.items()),
        f"{arm.arm} RuntimeStats contain a counter that cannot be reconstructed",
    )
    expected_last_media_t = candidates[-1].t if candidates else None
    _require(
        stats["last_media_t"] == expected_last_media_t
        and isinstance(stats["last_wall_t"], float)
        and math.isfinite(stats["last_wall_t"])
        and stats["last_wall_t"] >= 0.0,
        f"{arm.arm} RuntimeStats source clocks are invalid",
    )

    dispatch_health = [item for item in arm.health if item["kind"] == "judge_dispatch"]
    _require(
        len(dispatch_health) == 1
        and dispatch_health[0]["ts"] is None
        and _thaw(dispatch_health[0]["data"]) == {"mode": "inline"},
        f"{arm.arm} judge-dispatch health is not the executed inline topology",
    )
    observed_escalations = [
        (float(item["ts"]), _thaw(item["data"]))
        for item in arm.health
        if item["kind"] == "escalation"
    ]
    _require(
        observed_escalations == expected_escalations,
        f"{arm.arm} escalation health does not reconstruct from edge admissions",
    )
    source_frames = [item for item in arm.traces if item["stage"] == "source.frame"]
    _require(
        [_thaw(item) for item in source_frames]
        == [
            {
                "trace_id": f"{source_id}@{candidate.t}",
                "stage": "source.frame",
                "t": candidate.t,
                "goal_id": None,
                "data": {},
            }
            for candidate in dispatched
        ],
        f"{arm.arm} source-frame trace identity does not reconstruct from dispatches",
    )
    if route == "transition":
        transition_by_t = {
            float(item["t"]): item
            for item in arm.traces
            if item["stage"] == "transition.decision"
        }
        for candidate in dispatched:
            trace_data = _thaw(transition_by_t[candidate.t]["data"])
            if any(call.status != "success" for call in candidate.provider_calls):
                _require(
                    all(
                        trace_data[key] is None
                        for key in (
                            "from_confidence",
                            "from_evidence",
                            "from_truth",
                            "to_confidence",
                            "to_evidence",
                            "to_truth",
                        )
                    ),
                    f"{arm.arm} degraded transition trace retained an unexecuted verdict claim",
                )
                continue
            for call in candidate.provider_calls:
                _require(
                    trace_data[f"{call.predicate}_confidence"] == call.confidence
                    and trace_data[f"{call.predicate}_evidence"] == call.evidence
                    and trace_data[f"{call.predicate}_truth"]
                    == ("yes" if call.satisfied else "no"),
                    f"{arm.arm} transition trace verdict does not reconstruct "
                    "from offline-oracle calls",
                )

    expected_effective_config = {
        "fps": {
            "requested_fps": float(arm.replay_identity["fps"]),
            "effective_fps": float(arm.replay_identity["fps"]),
            "reason": "profile-default",
        },
        "video_timebase": {
            "requested": "media",
            "resolved": "media",
            "reason": "explicit-override",
        },
        "audio_timebase": None,
        "coast_enabled": True,
    }
    effective_health = [item for item in arm.health if item["kind"] == "effective_config"]
    _require(
        stats["effective_config"] == expected_effective_config
        and len(effective_health) == 1
        and effective_health[0]["ts"] is None
        and _thaw(effective_health[0]["data"])
        == {
            "fps": {**expected_effective_config["fps"], "overridden": False},
            "video_timebase": expected_effective_config["video_timebase"],
            "coast_enabled": True,
        },
        f"{arm.arm} effective runtime configuration does not reconstruct from the compiled replay",
    )

    plan = _thaw(receipt.watch_plan)
    expected_plan_id = PlanIdentity.from_goals(
        [Goal(id=goal_id, condition=str(plan["condition"]), say=str(plan["condition"]))],
        revision=1,
    ).plan_hash
    dispatch_ordinals = {
        candidate.index: ordinal
        for ordinal, candidate in enumerate(dispatched, start=1)
    }
    expected_incident_health: list[tuple[float, dict[str, Any]]] = []
    judgment_traces = [
        item for item in arm.traces if item["stage"] == "judgment.event"
    ]
    incident_traces = [
        item for item in arm.traces if item["stage"] == "incident.created"
    ]
    incident_by_id = {str(item["id"]): item for item in arm.incidents}
    for candidate in candidates:
        for incident_id in candidate.incident_ids:
            incident = incident_by_id[str(incident_id)]
            dispatch_seq = dispatch_ordinals[candidate.index]
            expected_entity = "scene" if route == "transition" else None
            expected_incident_origin = "transition" if route == "transition" else None
            expected_trace_origin = "transition" if route == "transition" else "judged"
            expected_reason = None if route == "transition" else "rising_edge"
            expected_judgment_id = judgment_event_id(
                goal_id,
                expected_entity,
                dispatch_seq,
            )
            _require(
                incident["concern_id"] == goal_id
                and float(incident["ts"]) == candidate.t
                and incident["message"] == plan["condition"]
                and incident["entity_id"] == expected_entity
                and incident["reason"] == expected_reason
                and incident["watchplan_id"] == expected_plan_id
                and incident["judgment_event_id"] == expected_judgment_id
                and incident["evidence_origin"] == expected_incident_origin,
                f"{arm.arm} Incident fields do not reconstruct from the compiled Watch",
            )
            matching_judgments = [
                item
                for item in judgment_traces
                if item["data"]["judgment_event_id"] == expected_judgment_id
            ]
            _require(
                len(matching_judgments) == 1
                and matching_judgments[0]["trace_id"] == f"{source_id}@{candidate.t}"
                and matching_judgments[0]["t"] == candidate.t
                and matching_judgments[0]["goal_id"] == goal_id
                and _thaw(matching_judgments[0]["data"])
                == {
                    "action": "fire",
                    "dispatch_seq": dispatch_seq,
                    "entity_id": expected_entity,
                    "judgment_event_id": expected_judgment_id,
                    "evidence_origin": expected_trace_origin,
                },
                f"{arm.arm} judgment trace does not reconstruct from its FIRE candidate",
            )
            matching_incident_traces = [
                item
                for item in incident_traces
                if item["data"]["incident_id"] == incident_id
            ]
            _require(
                len(matching_incident_traces) == 1
                and matching_incident_traces[0]["trace_id"] == f"{source_id}@{candidate.t}"
                and matching_incident_traces[0]["t"] == candidate.t
                and matching_incident_traces[0]["goal_id"] == goal_id
                and _thaw(matching_incident_traces[0]["data"])
                == {
                    "incident_id": incident_id,
                    "from_boundary": False,
                    "entity_id": expected_entity,
                    "judgment_event_id": expected_judgment_id,
                    "evidence_origin": expected_incident_origin,
                },
                f"{arm.arm} incident trace does not reconstruct from its delivered Incident",
            )
            expected_incident_health.append(
                (
                    candidate.t,
                    {
                        "incident_id": incident_id,
                        "goal_id": goal_id,
                        "entity_id": expected_entity,
                    },
                )
            )
    observed_incident_health = [
        (float(item["ts"]), _thaw(item["data"]))
        for item in arm.health
        if item["kind"] == "incident"
    ]
    _require(
        observed_incident_health == expected_incident_health,
        f"{arm.arm} incident health does not reconstruct from delivered Incidents",
    )
    ended = [item for item in arm.health if item["kind"] == "runtime_ended"]
    _require(
        len(ended) == 1
        and ended[0]["ts"] is None
        and _thaw(ended[0]["data"])
        == {
            "frames_seen": len(candidates),
            "admitted": len(admitted),
            "signal_admits": signal_admits,
            "fixed_admits": fixed_admits,
            "coast_admits": coast_admits,
            "audio_wakes": 0,
            "fixed_opportunities": fixed_opportunities,
            "goal_admissions": goal_admissions,
            "admission_by_goal": expected_admission_by_goal,
            "admission_by_cause": expected_admission_by_cause,
            "perceive_dispatches": len(dispatched),
            "judge_calls": len(dispatched),
            "incidents": len(arm.incidents),
            "source_coverage": {
                "video": {
                    "observed": len(candidates),
                    "excluded": 0,
                    "discontinuities": 0,
                    "missing_sequences": 0,
                },
                "audio": {
                    "observed": 0,
                    "excluded": 0,
                    "discontinuities": 0,
                    "missing_sequences": 0,
                },
            },
        },
        f"{arm.arm} runtime-ended health does not reconstruct from executed candidates",
    )


def _validate_receipt(receipt: MatchedWatchReceipt) -> None:
    """Reconstruct the receipt's cross-field invariants from its raw append-only records."""

    manifest = _thaw(receipt.manifest)
    is_v2 = manifest.get("schema") == _REAL_SCHEMA
    _exact(
        manifest,
        _REAL_MANIFEST_KEYS if is_v2 else _MANIFEST_KEYS,
        "receipt.manifest",
    )
    _require(
        manifest["schema"] == (_REAL_SCHEMA if is_v2 else _SCHEMA),
        "receipt manifest schema mismatch",
    )
    if not is_v2:
        _require(manifest["archetype"] == _ARCHETYPE, "receipt archetype mismatch")
    _require(manifest["replay"]["timebase"] == "media", "receipt replay timebase mismatch")
    if not is_v2:
        _require(manifest["watch"]["template_ref"] == _TEMPLATE, "receipt Watch template mismatch")
        _require(manifest["watch"]["domain_pack"] == _PACK, "receipt DomainPack mismatch")
    template_ref = manifest["watch"]["template_ref"]
    pack_name = manifest["watch"]["domain_pack"]
    _require(
        template_ref.startswith(pack_name + "/"),
        "receipt Watch template does not belong to its DomainPack",
    )
    rendered = watch_template(template_ref, manifest["watch"]["parameters"]).to_dict()
    route = _plan_route(WatchPlan.from_dict(rendered))
    _require(rendered == _thaw(receipt.watch_plan), "receipt WatchPlan render mismatch")
    _require(_canonical_sha(rendered) == receipt.watch_plan_sha256, "receipt WatchPlan identity mismatch")
    _require(manifest["judge"] == _thaw(receipt.judge_identity), "receipt judge identity mismatch")
    _require(manifest["events"] == _thaw(receipt.event_windows), "receipt event windows mismatch")
    _require(
        manifest["media"]["sha256"] == receipt.media_input_receipt["sha256"],
        "receipt media identity mismatch",
    )
    provenance = _thaw(receipt.build_provenance)
    _require(
        receipt.package_version == provenance.get("package_version")
        and set(provenance)
        == {"package_version", "git_commit", "contract_version", "build_timestamp"},
        "receipt build provenance mismatch",
    )
    _require(provenance == build_info().to_dict(), "receipt build provenance is not the active build")
    if is_v2:
        _require(
            receipt.receipt_schema == _RECEIPT_SCHEMA_V2,
            "real-video receipt must use receipt/v2",
        )
        _require(
            set(receipt.authority_artifact_presence) == set(_AUTHORITY_NAMES)
            and all(
                isinstance(value, bool)
                for value in receipt.authority_artifact_presence.values()
            ),
            "receipt authority artifact presence has an invalid shape",
        )
        artifact_roles = [item["role"] for item in receipt.input_artifacts]
        _require(
            artifact_roles == sorted(artifact_roles)
            and len(artifact_roles) == len(set(artifact_roles))
            and {"manifest", "corpus", "media", "oracle"}.issubset(artifact_roles),
            "receipt input artifacts are incomplete, duplicated, or noncanonical",
        )
        for name in _AUTHORITY_NAMES:
            _require(
                receipt.authority_artifact_presence[name]
                == (f"authority:{name}" in artifact_roles),
                f"receipt authority artifact presence disagrees for {name!r}",
            )
    else:
        _require(
            receipt.receipt_schema == _RECEIPT_SCHEMA
            and not receipt.input_artifacts
            and not receipt.authority_artifact_presence,
            "synthetic receipt carries v2-only artifact state",
        )

    for arm in (receipt.baseline, receipt.adaptive):
        _require(
            arm.valid == (not arm.blockers),
            f"{arm.arm} cached validity disagrees with its blockers",
        )
        expected_replay = {
            "fps": float(manifest["replay"]["fps"]),
            "timebase": "media",
            "source_id": manifest["replay"]["source_id"],
        }
        _require(_thaw(arm.replay_identity) == expected_replay, "arm replay identity mismatch")
        _require(
            arm.media_identity["sha256"] == receipt.media_input_receipt["sha256"],
            "arm media identity mismatch",
        )
        if is_v2:
            _require(
                _thaw(arm.media_identity["decoder_identity"])
                == _thaw(_decoder_identity()),
                "arm decoder identity does not match the active reconstruction environment",
            )
        _require(arm.judge_identity == receipt.judge_identity, "arm judge identity mismatch")
        _require(
            arm.apply_receipt["status"] in {"applied", "applied_with_warnings"}
            and tuple(arm.apply_receipt["watch_ids"]) == (arm.compiled_binding["goal_id"],),
            "apply receipt does not bind the compiled goal",
        )
        _require(
            content_hash(_thaw(arm.resolved_watch["components"]))
            == arm.resolved_watch["component_snapshot_hash"],
            "arm component snapshot hash mismatch",
        )
        expected_resolved_source = {
            path: _thaw(arm.source_profile[path.removeprefix("SourceProfile.")])
            for path in _RESOLVED_SOURCE_PROFILE_KEYS
        }
        _require(
            _thaw(arm.resolved_watch["source_profile"]) == expected_resolved_source,
            "arm resolved SourceProfile snapshot mismatch",
        )
        active_pack = get_domain_pack(pack_name)
        expected_domain_pack = {
            "id": active_pack.id,
            "version": active_pack.version,
            "values": dict(active_pack.values),
            "components": [
                component.to_dict() for component in active_pack.components
            ],
            "policy_clamps": dict(active_pack.policy_clamps),
        }
        _require(
            _thaw(arm.resolved_watch["domain_pack"]) == expected_domain_pack,
            "arm resolved DomainPack snapshot mismatch",
        )
        _require(
            f"sha256:{arm.resolved_watch['plan_hash']}" == receipt.watch_plan_sha256,
            "arm resolved plan hash mismatch",
        )
        _require(
            _thaw(arm.resolved_watch["plan"]) == _thaw(receipt.watch_plan),
            "arm resolved WatchPlan does not exactly match the receipt WatchPlan",
        )
        candidates = arm.candidates
        _require(
            [item.index for item in candidates] == list(range(len(candidates))),
            "candidate indices are not exact and contiguous",
        )
        if is_v2:
            _require(
                [item.source_sequence for item in candidates]
                == list(range(len(candidates)))
                and all(
                    isinstance(item.frame_sha256, str)
                    and item.frame_sha256.startswith("sha256:")
                    for item in candidates
                ),
                "real-video candidate source sequence/frame identity is incomplete",
            )
            for candidate in candidates:
                for call in candidate.provider_calls:
                    expected_lookup = _canonical_sha(
                        {
                            "case_id": manifest["case_id"],
                            "source_sequence": candidate.source_sequence,
                            "t": candidate.t,
                            "frame_sha256": candidate.frame_sha256,
                            "predicate_sha256": call.condition_sha256,
                        }
                    )
                    _require(
                        call.oracle_lookup_sha256 == expected_lookup,
                        "offline-oracle lookup identity mismatch",
                    )
        _require(
            tuple(item.t for item in candidates) == arm.candidate_timestamps,
            "candidate timestamp sequence mismatch",
        )
        edge = [item for item in arm.traces if item["stage"] == "edge.decision"]
        decision_traces = [
            item
            for item in arm.traces
            if item["stage"]
            == ("transition.decision" if route == "transition" else "gate.classified")
        ]
        source_frames = [item for item in arm.traces if item["stage"] == "source.frame"]
        dispatched = [item for item in candidates if item.runtime_dispatched]
        started = [item for item in arm.health if item["kind"] == "runtime_started"]
        _require(len(started) == 1, "runtime_started health record is missing or duplicated")
        _require(
            _thaw(started[0]["data"])
            == {
                "source_kind": (
                    "real_video_offline_oracle" if is_v2 else "synthetic_video"
                ),
                "media": _thaw(arm.media_identity),
                "replay": _thaw(arm.replay_identity),
                "candidate_count": len(candidates),
            },
            "runtime_started source identity disagrees with media/replay receipt",
        )
        _require(
            [item["t"] for item in edge] == [item.t for item in candidates],
            "edge trace coverage/timestamps mismatch",
        )
        _require(
            [_thaw(item["data"]) for item in edge]
            == [_thaw(item.edge_decision) for item in candidates],
            "candidate edge records disagree with traces",
        )
        _require(
            [item["t"] for item in decision_traces] == [item.t for item in dispatched]
            and [item["t"] for item in source_frames] == [item.t for item in dispatched],
            "transition/source trace coverage mismatch",
        )
        if route == "transition":
            _require(
                [item["data"]["outcome"] for item in decision_traces]
                == [item.transition_outcome for item in dispatched],
                "candidate transition outcomes disagree with traces",
            )
        else:
            _require(
                [item["data"]["action"] for item in decision_traces]
                == [item.decision for item in dispatched],
                "candidate ordinary-Watch decisions disagree with traces",
            )

        incident_ids = [str(item["id"]) for item in arm.incidents]
        linked = [
            (candidate, incident_id)
            for candidate in candidates
            for incident_id in candidate.incident_ids
        ]
        _require(
            len(incident_ids) == len(set(incident_ids)) == len(linked)
            and sorted(incident_ids) == sorted(incident_id for _, incident_id in linked),
            "incident IDs are not in one-to-one candidate ownership",
        )
        incidents = {str(item["id"]): item for item in arm.incidents}
        for candidate, incident_id in linked:
            _require(
                float(incidents[incident_id]["ts"]) == candidate.t,
                "incident timestamp does not match its candidate",
            )
        incident_traces = [item for item in arm.traces if item["stage"] == "incident.created"]
        judgment_traces = [item for item in arm.traces if item["stage"] == "judgment.event"]
        _require(
            len(incident_traces) == len(judgment_traces) == len(arm.incidents),
            "judgment/incident trace completeness mismatch",
        )
        _require(
            sorted(str(item["data"]["incident_id"]) for item in incident_traces)
            == sorted(incident_ids),
            "incident trace IDs disagree with delivered incidents",
        )
        for incident in arm.incidents:
            _require(
                any(
                    trace["data"]["judgment_event_id"] == incident["judgment_event_id"]
                    and trace["t"] == incident["ts"]
                    for trace in judgment_traces
                ),
                "JudgmentEvent trace does not link to delivered Incident",
            )

        fire_decisions = sum(item.decision == "fire" for item in dispatched)
        ask_decisions = sum(item.decision == "ask" for item in dispatched)
        missing_decisions = sum(item.decision == "missing" for item in dispatched)
        _require(
            fire_decisions == len(arm.incidents) == arm.runtime_stats["fires"]
            and ask_decisions == len(arm.asks) == arm.runtime_stats["asks"],
            "FIRE/ASK decisions disagree with runtime stats and outputs",
        )
        for candidate in dispatched:
            failed_calls = [
                call for call in candidate.provider_calls if call.status != "success"
            ]
            has_error = bool(failed_calls)
            _require(
                (candidate.decision == "missing") == has_error,
                "missing decision disagrees with provider outcomes",
            )
            _require(
                len(failed_calls) <= 1
                and (
                    not failed_calls
                    or candidate.provider_calls[-1] is failed_calls[0]
                ),
                "one failed Watch predicate must short-circuit the remaining predicate calls",
            )
            _require(
                (candidate.decision == "fire") == bool(candidate.incident_ids),
                "FIRE decision disagrees with incident link",
            )
        _require(
            missing_decisions == arm.provider_call_failures,
            "missing decisions disagree with provider failures",
        )
        if is_v2:
            _validate_v2_arm_reconstruction(
                receipt,
                arm,
                manifest=manifest,
                route=route,
            )

        ended = [item for item in arm.health if item["kind"] == "runtime_ended"]
        _require(len(ended) == 1, "runtime_ended health record is missing or duplicated")
        ended_data = ended[0]["data"]
        _require(
            ended_data["frames_seen"] == arm.runtime_stats["frames"]
            and ended_data["admitted"] == arm.runtime_stats["escalations"]
            and ended_data["perceive_dispatches"] == arm.runtime_stats["perceive_dispatches"]
            and ended_data["judge_calls"] == arm.runtime_stats["judge_calls"]
            and ended_data["incidents"] == arm.runtime_stats["incidents"],
            "runtime_ended health disagrees with RuntimeStats",
        )
        zero_admission = [
            item for item in arm.health if item["kind"] == "zero_admission_coverage"
        ]
        expects_zero_admission = (
            arm.runtime_stats["frames"] > 0 and arm.runtime_stats["escalations"] == 0
        )
        _require(
            len(zero_admission) == int(expects_zero_admission),
            "zero_admission_coverage cardinality disagrees with observed admission counts",
        )
        if zero_admission:
            kinds = [item["kind"] for item in arm.health]
            zero_index = kinds.index("zero_admission_coverage")
            ended_index = kinds.index("runtime_ended")
            _require(
                zero_index + 1 == ended_index,
                "zero_admission_coverage must immediately precede runtime_ended",
            )
            zero_data = _thaw(zero_admission[0]["data"])
            shared = {
                key: zero_data[key]
                for key in _RUNTIME_ENDED_DATA_KEYS
            }
            _require(
                shared == _thaw(ended_data),
                "zero_admission_coverage counters/source coverage disagree with runtime_ended",
            )
            expected_threshold = float(
                arm.source_profile["parameters"]["salience_threshold"]
            )
            expected_stat = arm.source_profile["parameters"]["salience_stat"]
            compiled_threshold = arm.compiled_binding["watchspec_kwargs"]["stat_thresholds"][
                expected_stat
            ]
            _require(
                float(zero_data["threshold"]) == expected_threshold
                == float(compiled_threshold)
                and zero_data["stat"] == expected_stat,
                "zero_admission_coverage threshold/stat disagree with the effective "
                "SourceProfile/compiled binding",
            )
            coverage_floor = arm.compiled_binding["coverage_floor"]
            _require(
                zero_data["coverage_floor"] == coverage_floor,
                "zero_admission_coverage floor disagrees with the compiled binding",
            )
            _require(
                isinstance(zero_data["floor_unmet"], bool)
                and zero_data["floor_unmet"]
                == (
                    coverage_floor is not None
                    and arm.runtime_stats["escalations"] < coverage_floor
                ),
                "zero_admission_coverage floor state is inconsistent",
            )
        _require(
            len([item for item in arm.health if item["kind"] == "escalation"])
            == sum(item.admitted for item in candidates),
            "escalation health coverage mismatch",
        )
        effective = [item for item in arm.health if item["kind"] == "effective_config"]
        _require(len(effective) == 1, "effective_config health record is missing or duplicated")
        _require(
            float(effective[0]["data"]["fps"]["requested_fps"]) == arm.replay_identity["fps"]
            and effective[0]["data"]["video_timebase"]["resolved"] == arm.replay_identity["timebase"],
            "observed effective config disagrees with replay identity",
        )

    adaptive_doc = manifest["adaptive"]
    _require(
        receipt.adaptive.source_profile["parameters"]["salience_stat"]
        == adaptive_doc["salience_stat"]
        and float(receipt.adaptive.source_profile["parameters"]["salience_threshold"])
        == float(adaptive_doc["threshold"]),
        "adaptive manifest/profile binding mismatch",
    )
    recomputed_blockers = validate_matched_arms(
        receipt.baseline,
        receipt.adaptive,
        watch_plan_sha256=receipt.watch_plan_sha256,
        expected_media_sha256=receipt.media_input_receipt.get("sha256"),
    )
    _require(
        receipt.comparability_blockers == recomputed_blockers
        and receipt.comparable == (not recomputed_blockers),
        "cached receipt comparable state disagrees with recomputed blockers",
    )


def derive_viability_rows(receipt: MatchedWatchReceipt) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the v3 viability rows exclusively from an immutable execution receipt."""

    if receipt.receipt_schema == _RECEIPT_SCHEMA_V2:
        _revalidate_v2_input_artifacts(receipt)
    _validate_receipt(receipt)
    _require(receipt.evidence_classification in _EVIDENCE_CLASSES, "unsupported evidence classification")
    manifest_media = _object(receipt.manifest.get("media"), "receipt.manifest.media")
    _require(
        manifest_media.get("sha256") == receipt.media_input_receipt.get("sha256"),
        "receipt media identity does not match its manifest",
    )
    _require(
        _canonical_sha(_thaw(receipt.watch_plan)) == receipt.watch_plan_sha256,
        "receipt WatchPlan identity mismatch",
    )
    _require(
        _thaw(receipt.manifest.get("judge")) == _thaw(receipt.judge_identity),
        "receipt judge identity does not match its manifest",
    )
    manifest_events = _thaw(receipt.manifest.get("events"))
    _require(manifest_events == _thaw(receipt.event_windows), "receipt event windows do not match manifest")
    recomputed_blockers = validate_matched_arms(
        receipt.baseline,
        receipt.adaptive,
        watch_plan_sha256=receipt.watch_plan_sha256,
        expected_media_sha256=receipt.media_input_receipt.get("sha256"),
    )
    _require(not recomputed_blockers, f"matched arms are not comparable: {recomputed_blockers!r}")
    rows: list[dict[str, Any]] = []
    for arm in (receipt.baseline, receipt.adaptive):
        _require(arm.judge_identity == receipt.judge_identity, "arm judge identity mismatch")
        recomputed_config = _canonical_sha(
            {
                "effective": _thaw(arm.resolved_watch["effective"]),
                "source_profile": _thaw(arm.source_profile),
                "compiled_binding": _thaw(arm.compiled_binding),
            }
        )
        _require(recomputed_config == arm.configuration_sha256, "arm configuration hash mismatch")
        stats = arm.runtime_stats
        admitted_count = sum(candidate.admitted for candidate in arm.candidates)
        dispatch_count = sum(candidate.runtime_dispatched for candidate in arm.candidates)
        _require(
            stats.get("frames") == len(arm.candidates)
            and stats.get("goal_admissions") == admitted_count
            and stats.get("escalations") == admitted_count
            and stats.get("perceive_dispatches") == dispatch_count
            and stats.get("judge_calls") == dispatch_count
            and arm.runtime_dispatches == dispatch_count,
            "RuntimeStats/runtime dispatch receipt mismatch",
        )
        incident_ids = {str(item["id"]) for item in arm.incidents}
        candidate_incident_ids = {
            incident_id
            for candidate in arm.candidates
            for incident_id in candidate.incident_ids
        }
        _require(
            incident_ids == candidate_incident_ids,
            "incident records and candidate FIRE links disagree",
        )
        _require(
            tuple(candidate.t for candidate in arm.candidates) == arm.candidate_timestamps,
            "candidate timestamps disagree with arm receipt",
        )
        route = _plan_route(WatchPlan.from_dict(_thaw(receipt.watch_plan)))
        if route == "transition":
            expected_predicates = ["from", "to"]
            expected_hashes = [
                _canonical_sha(receipt.watch_plan["explicit"]["Goal.state.from"]),
                _canonical_sha(receipt.watch_plan["explicit"]["Goal.state.to"]),
            ]
        else:
            expected_predicates = ["condition"]
            expected_hashes = [_canonical_sha(receipt.watch_plan["condition"])]
        for candidate in arm.candidates:
            _require(
                (candidate.decision == "fire") == bool(candidate.incident_ids),
                "candidate decision and incident links disagree",
            )
        observed_calls = [
            call
            for candidate in arm.candidates
            for call in candidate.provider_calls
        ]
        _require(
            [call.sequence for call in observed_calls] == list(range(1, len(observed_calls) + 1)),
            "provider call sequence is incomplete or reordered",
        )
        for candidate in arm.candidates:
            if candidate.runtime_dispatched:
                _require(candidate.admitted, "runtime dispatched a non-admitted candidate")
                _require(
                    [call.predicate for call in candidate.provider_calls]
                    == expected_predicates,
                    "Watch dispatch does not carry its exact predicate calls",
                )
                _require(
                    [call.condition_sha256 for call in candidate.provider_calls]
                    == expected_hashes,
                    "provider condition hashes do not match the WatchPlan",
                )
                _require(
                    all(
                        call.candidate_index == candidate.index
                        and call.candidate_t == candidate.t
                        and call.status == "success"
                        for call in candidate.provider_calls
                    ),
                    "provider subcall provenance/status mismatch",
                )
            else:
                _require(
                    not candidate.provider_calls,
                    "non-dispatched candidate carries provider subcalls",
                )
        _require(
            len(observed_calls) == arm.provider_call_attempts
            and sum(call.status == "success" for call in observed_calls)
            == arm.provider_call_successes
            and sum(call.status != "success" for call in observed_calls)
            == arm.provider_call_failures,
            "provider subcall aggregates do not match append-only records",
        )
        events, fire_ids = _event_rows(receipt, arm)
        decisions = {key: 0 for key in ("fire", "ask", "silent", "missing")}
        for candidate in arm.candidates:
            if candidate.runtime_dispatched:
                _require(candidate.decision in decisions, "runtime dispatch is missing a decision")
                decisions[str(candidate.decision)] += 1
        _require(
            sum(decisions.values()) == arm.runtime_dispatches,
            "decision counts do not sum to runtime dispatches",
        )
        rows.append(
            {
                "case_id": receipt.manifest["case_id"],
                "archetype": receipt.manifest["archetype"],
                "operating_point": receipt.manifest["adaptive"]["operating_point"],
                # The viability manifest fingerprints the adaptive operating point; the baseline's
                # own executable configuration remains independently bound in the arm receipt.
                "configuration_sha256": receipt.adaptive.configuration_sha256,
                "arm": arm.arm,
                "evidence_kind": "synthetic",
                "candidate_windows_total": len(arm.candidates),
                "candidate_windows_admitted": sum(item.admitted for item in arm.candidates),
                "runtime_dispatches": arm.runtime_dispatches,
                "judge_calls": arm.provider_call_attempts,
                "decisions": decisions,
                "event_windows": events,
                "fire_event_ids": fire_ids,
            }
        )
    baseline, adaptive = rows
    _require(
        baseline["candidate_windows_admitted"] == baseline["candidate_windows_total"]
        and baseline["runtime_dispatches"] == baseline["candidate_windows_total"],
        "derived baseline is not exhaustive",
    )
    _require(
        receipt.origin_evidence_sha256 == receipt.evidence_sha256,
        "receipt evidence origin seal mismatch",
    )
    return baseline, adaptive


def _revalidate_v2_input_artifacts(receipt: MatchedWatchReceipt) -> None:
    """Reopen every v2 source artifact and require its exact recorded descriptor snapshot."""

    _require(
        receipt.receipt_schema == _RECEIPT_SCHEMA_V2,
        "v2 artifact validation requires receipt/v2",
    )
    by_role = {str(item["role"]): item for item in receipt.input_artifacts}
    _require(
        len(by_role) == len(receipt.input_artifacts),
        "receipt input artifacts contain duplicate roles",
    )
    for role, recorded in by_role.items():
        path = Path(str(recorded["path"]))
        try:
            if role in {"manifest", "corpus", "oracle"}:
                observed = _load_json_snapshot(path).receipt
            else:
                observed = _snapshot_binary(path)
        except (SafeJSONInputError, MatchedWatchInputError) as exc:
            raise MatchedWatchInputError(
                f"receipt input artifact {role!r} cannot be revalidated: {exc}"
            ) from exc
        expected = {key: recorded[key] for key in _SAFE_INPUT_RECEIPT_KEYS}
        _require(
            _safe_receipt_dict(observed) == expected,
            f"receipt input artifact {role!r} no longer matches its recorded snapshot",
        )
    manifest = _thaw(receipt.manifest)
    _require(
        {key: by_role["manifest"][key] for key in _SAFE_INPUT_RECEIPT_KEYS}
        == _thaw(receipt.manifest_input_receipt),
        "manifest artifact and manifest_input_receipt disagree",
    )
    _require(
        {key: by_role["media"][key] for key in _SAFE_INPUT_RECEIPT_KEYS}
        == _thaw(receipt.media_input_receipt),
        "media artifact and media_input_receipt disagree",
    )
    _require(
        by_role["media"]["sha256"] == manifest["media"]["sha256"]
        and by_role["corpus"]["sha256"] == manifest["corpus"]["sha256"]
        and by_role["oracle"]["sha256"] == manifest["oracle"]["sha256"],
        "receipt manifest artifact references disagree with input artifacts",
    )
    for name in _AUTHORITY_NAMES:
        _require(
            receipt.authority_artifact_presence[name]
            == (f"authority:{name}" in by_role),
            f"receipt authority artifact presence disagrees for {name!r}",
        )
    reconstructed = load_matched_watch_manifest(by_role["manifest"]["path"])
    _require(
        reconstructed.schema == _REAL_SCHEMA
        and _thaw(reconstructed.document) == manifest
        and [_thaw(item) for item in reconstructed.input_artifacts]
        == [_thaw(item) for item in receipt.input_artifacts]
        and _thaw(reconstructed.authority_artifact_presence)
        == _thaw(receipt.authority_artifact_presence),
        "receipt no longer reconstructs from its closed real-video manifest and corpus",
    )
    oracle_by_sequence = {
        item.index: item for item in reconstructed.oracle_frames
    }
    _require(
        len(oracle_by_sequence) == len(reconstructed.oracle_frames),
        "reconstructed offline oracle contains duplicate source sequences",
    )
    plan = WatchPlan.from_dict(_thaw(receipt.watch_plan))
    route = _plan_route(plan)
    if route == "transition":
        predicates = (
            ("from", _canonical_sha(plan.explicit["Goal.state.from"])),
            ("to", _canonical_sha(plan.explicit["Goal.state.to"])),
        )
    else:
        predicates = (("condition", _canonical_sha(plan.condition)),)
    for arm in (receipt.baseline, receipt.adaptive):
        _require(
            [
                (candidate.source_sequence, candidate.t, candidate.frame_sha256)
                for candidate in arm.candidates
            ]
            == [
                (frame.index, frame.t, frame.frame_sha256)
                for frame in reconstructed.oracle_frames
            ],
            f"{arm.arm} decoded candidate evidence disagrees with the frozen offline oracle",
        )
        expected_sequence = 1
        for candidate in arm.candidates:
            if not candidate.runtime_dispatched:
                _require(
                    not candidate.provider_calls,
                    "non-dispatched real-video candidate carries provider calls",
                )
                continue
            oracle_frame = oracle_by_sequence[candidate.source_sequence]
            expected_calls: list[dict[str, Any]] = []
            for predicate, predicate_sha in predicates:
                verdict = oracle_frame.verdicts[predicate_sha]
                lookup_sha = _canonical_sha(
                    {
                        "case_id": reconstructed.case_id,
                        "source_sequence": candidate.source_sequence,
                        "t": candidate.t,
                        "frame_sha256": candidate.frame_sha256,
                        "predicate_sha256": predicate_sha,
                    }
                )
                expected_calls.append(
                    ProviderCallReceipt(
                        expected_sequence,
                        candidate.index,
                        candidate.t,
                        predicate,
                        predicate_sha,
                        "error" if verdict.error is not None else "success",
                        None if verdict.error is not None else verdict.satisfied,
                        None if verdict.error is not None else verdict.confidence,
                        None if verdict.error is not None else verdict.evidence,
                        verdict.error,
                        lookup_sha,
                    ).to_dict()
                )
                expected_sequence += 1
                if verdict.error is not None:
                    break
            _require(
                [call.to_dict() for call in candidate.provider_calls]
                == expected_calls,
                f"{arm.arm} provider calls disagree with the frozen offline oracle",
            )


def load_matched_watch_receipt(path: str | Path) -> LoadedMatchedWatchReceipt:
    """Safely reconstruct and revalidate one persisted matched-Watch receipt.

    The receipt file's descriptor-backed snapshot is retained independently from the provenance
    embedded by the runner. No cached row, validity, comparability, or hash field is trusted.
    """

    try:
        loaded = load_json_file(path)
    except SafeJSONInputError as exc:
        raise MatchedWatchInputError(f"unsafe persisted matched-Watch receipt: {exc}") from exc
    try:
        doc = _object(loaded.value, "receipt")
        receipt_schema = doc.get("schema")
        v2 = receipt_schema == _RECEIPT_SCHEMA_V2
        _exact(
            doc,
            _PERSISTED_RECEIPT_V2_KEYS if v2 else _PERSISTED_RECEIPT_KEYS,
            "receipt",
        )
        _require(
            receipt_schema in {_RECEIPT_SCHEMA, _RECEIPT_SCHEMA_V2},
            "receipt.schema is unsupported",
        )
        manifest = _parse_persisted_manifest(doc["manifest"])
        _require(
            v2 == (manifest["schema"] == _REAL_SCHEMA),
            "receipt and embedded manifest schema generations disagree",
        )
        manifest_input = _parse_safe_input_receipt_mapping(
            doc["manifest_input_receipt"], "receipt.manifest_input_receipt"
        )
        media_input = _parse_safe_input_receipt_mapping(
            doc["media_input_receipt"], "receipt.media_input_receipt"
        )
        raw_watch_plan = _object(doc["watch_plan"], "receipt.watch_plan")
        try:
            canonical_watch_plan = WatchPlan.from_dict(raw_watch_plan).to_dict()
        except (TypeError, ValueError) as exc:
            raise MatchedWatchInputError(
                f"receipt.watch_plan is not a valid WatchPlan wire document: {exc}"
            ) from exc
        _require(
            canonical_watch_plan == raw_watch_plan,
            "receipt.watch_plan is not the exact canonical WatchPlan wire form",
        )
        watch_plan = _strict_object(canonical_watch_plan, "receipt.watch_plan")
        watch_plan_sha256 = _sha(doc["watch_plan_sha256"], "receipt.watch_plan_sha256")
        judge_identity = _strict_object(
            doc["judge_identity"],
            "receipt.judge_identity",
            keys=_JUDGE_IDENTITY_V2_KEYS if v2 else _JUDGE_IDENTITY_KEYS,
        )
        event_windows = _strict_object_tuple(
            doc["event_windows"], "receipt.event_windows", keys=_EVENT_KEYS
        )
        evidence_classification = _strict_string(
            doc["evidence_classification"], "receipt.evidence_classification"
        )
        _require(
            evidence_classification in _EVIDENCE_CLASSES,
            "receipt.evidence_classification is unsupported",
        )
        package_version = _strict_string(doc["package_version"], "receipt.package_version")
        build_provenance = _strict_object(
            doc["build_provenance"],
            "receipt.build_provenance",
            keys=_BUILD_PROVENANCE_KEYS,
        )
        baseline = _parse_arm_receipt(
            doc["baseline"],
            "receipt.baseline",
            "baseline",
            v2=v2,
        )
        adaptive = _parse_arm_receipt(
            doc["adaptive"],
            "receipt.adaptive",
            "adaptive",
            v2=v2,
        )
        comparable = _strict_bool(doc["comparable"], "receipt.comparable")
        raw_blockers = doc["comparability_blockers"]
        _require(
            isinstance(raw_blockers, list),
            "receipt.comparability_blockers must be a list",
        )
        comparability_blockers = tuple(
            _strict_string(item, f"receipt.comparability_blockers[{index}]")
            for index, item in enumerate(raw_blockers)
        )
        _require(doc["launch_qualified"] is False, "receipt.launch_qualified must be false")
        _require(
            doc["qualification_reason"] == _QUALIFICATION_REASON,
            "receipt.qualification_reason does not match the fail-closed contract",
        )
        cached_content_sha256 = _sha(doc["content_sha256"], "receipt.content_sha256")
        cached_evidence_sha256 = _sha(doc["evidence_sha256"], "receipt.evidence_sha256")
        origin_evidence_sha256 = _sha(
            doc["origin_evidence_sha256"], "receipt.origin_evidence_sha256"
        )
        run_timestamp = _strict_string(doc["run_timestamp"], "receipt.run_timestamp")
        try:
            parsed_timestamp = datetime.fromisoformat(run_timestamp)
        except ValueError as exc:
            raise MatchedWatchInputError("receipt.run_timestamp must be ISO-8601") from exc
        _require(
            parsed_timestamp.tzinfo is not None,
            "receipt.run_timestamp must include an explicit timezone",
        )
        if v2:
            raw_artifacts = doc["input_artifacts"]
            _require(
                isinstance(raw_artifacts, list) and raw_artifacts,
                "receipt.input_artifacts must be a non-empty list",
            )
            input_artifacts: tuple[Mapping[str, Any], ...] = tuple(
                _strict_object(
                    item,
                    f"receipt.input_artifacts[{index}]",
                    keys=_INPUT_ARTIFACT_KEYS,
                )
                for index, item in enumerate(raw_artifacts)
            )
            for index, artifact in enumerate(input_artifacts):
                _strict_string(artifact["role"], f"receipt.input_artifacts[{index}].role")
                _parse_safe_input_receipt_mapping(
                    {key: artifact[key] for key in _SAFE_INPUT_RECEIPT_KEYS},
                    f"receipt.input_artifacts[{index}]",
                )
            presence_doc = _object(
                doc["authority_artifact_presence"],
                "receipt.authority_artifact_presence",
            )
            _exact(
                presence_doc,
                set(_AUTHORITY_NAMES),
                "receipt.authority_artifact_presence",
            )
            for name, value in presence_doc.items():
                _strict_bool(value, f"receipt.authority_artifact_presence.{name}")
            authority_presence: Mapping[str, bool] = _strict_object(
                presence_doc,
                "receipt.authority_artifact_presence",
                keys=set(_AUTHORITY_NAMES),
            )
        else:
            input_artifacts = ()
            authority_presence = MappingProxyType({})
        receipt = MatchedWatchReceipt(
            manifest,
            manifest_input,
            media_input,
            watch_plan,
            watch_plan_sha256,
            judge_identity,
            event_windows,
            evidence_classification,
            package_version,
            build_provenance,
            baseline,
            adaptive,
            comparable,
            comparability_blockers,
            run_timestamp,
            origin_evidence_sha256,
            receipt_schema,
            input_artifacts,
            authority_presence,
        )
        if v2:
            _revalidate_v2_input_artifacts(receipt)
        _require(
            cached_content_sha256 == receipt.content_sha256,
            "receipt cached content hash disagrees with reconstruction",
        )
        _require(
            cached_evidence_sha256 == receipt.evidence_sha256,
            "receipt cached evidence hash disagrees with reconstruction",
        )
        _require(
            origin_evidence_sha256 == cached_evidence_sha256,
            "receipt evidence origin seal mismatch",
        )
        rows = derive_viability_rows(receipt)
        _require(
            receipt.to_dict() == doc,
            "receipt wire document does not exactly match reconstructed DTOs",
        )
    except MatchedWatchInputError:
        raise
    except (AssertionError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MatchedWatchInputError(
            f"persisted matched-Watch receipt reconstruction failed: {exc}"
        ) from exc
    return LoadedMatchedWatchReceipt(
        receipt,
        loaded.receipt,
        (
            _freeze(rows[0]),
            _freeze(rows[1]),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=_CLI_DESCRIPTION)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        receipt = run_matched_watch(args.manifest)
        payload = receipt.to_json()
    except MatchedWatchInputError as exc:
        parser.error(str(exc))
    if args.out:
        try:
            with Path(args.out).open("x", encoding="utf-8") as stream:
                stream.write(payload + "\n")
        except FileExistsError:
            parser.error("--out is append-only and already exists; choose a new receipt path")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
