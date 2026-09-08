"""Strict, receiver-independent raw-VLM baseline for the frozen qualification corpus.

This module deliberately contains no HTTP implementation.  Its default CLI is offline preflight;
the doubly-confirmed paid mode imports a sibling wrapper around Percept's shipped provider backend.
Callers can inject an offline provider, keeping schema, scoring, identity, crash, and cost-accounting
tests deterministic and preventing an accidental paid run from a normal CLI invocation.

The comparison unit is one frozen corpus case: the exact verified source bytes plus exactly one
unchanged goal condition are sent in one provider call.  Provider output is retained verbatim and
must satisfy the declared temporal JSON schema.  Invalid or out-of-bounds timestamps reject the
whole response; they are never clipped into a plausible-looking prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


# Direct script execution names this module ``__main__``.  The paid adapter imports the sibling by
# its canonical name, so without this alias Python executes the file a second time and creates a
# distinct ``ProviderResponse`` class.  Bind the canonical name to the already-running CLI module;
# response validation remains an exact isinstance check against the one shared class.
if __name__ == "__main__":
    sys.modules["raw_vlm_baseline"] = sys.modules[__name__]


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_CORPUS = HERE / "corpus.v1.2.json"
DEFAULT_AUTHORING = REPO / "eval/primitive-authoring/corpus-v1.2.frozen.json"

RESULT_KIND = "percept-raw-vlm-baseline"
RESULT_VERSION = 1
RESPONSE_SCHEMA = "percept.raw-vlm-temporal-events/1"
PERCEPT_EVENT_CHAIN_SCHEMA = "percept.native-event-chain/1"
DEFAULT_PROMPT_VERSION = "raw-vlm-temporal-events-v1"
DEFAULT_POINT_TOLERANCE_S = 1.0
CHECKPOINT_KIND = "percept-raw-vlm-paid-checkpoint"
CHECKPOINT_VERSION = 1
PAID_RESULT_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_ENV_NAMES = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_AISTUDIO_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
)
DEFAULT_PROMPT_TEMPLATE = """You are evaluating one video against one watch condition.

Condition (preserve its meaning exactly):
{condition}

The source-clock duration is {duration_s:.9f} seconds. Return JSON only, with exactly this shape:
{{"events":[{{"start_s":0.0,"end_s":1.0}}]}}

Each item is one distinct occurrence during which the condition is true. Timestamps are seconds on
the source clock, must be finite, and must satisfy 0 <= start_s <= end_s <= duration. Use an empty
events list when the condition never occurs. Do not merge distinct repeated occurrences.
"""


class BaselineInputError(ValueError):
    """Frozen input, provider output, or accounting metadata is not scoreable as declared."""


class PaidBudgetExceeded(RuntimeError):
    """The next physical provider attempt cannot fit inside an immutable run cap."""


@dataclass(frozen=True)
class PricingStamp:
    """Immutable provider/model price snapshot used for this comparison artifact."""

    backend: str
    model: str
    pricing_version: str
    pricing_source: str
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float

    def __post_init__(self) -> None:
        for name in ("backend", "model", "pricing_version", "pricing_source"):
            if not str(getattr(self, name)).strip():
                raise BaselineInputError(f"pricing stamp {name} must be non-empty")
        for name in (
            "input_usd_per_million_tokens",
            "output_usd_per_million_tokens",
        ):
            value = getattr(self, name)
            if not _number(value) or float(value) < 0:
                raise BaselineInputError(f"pricing stamp {name} must be finite and >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "version": self.pricing_version,
            "source": self.pricing_source,
            "currency": "USD",
            "input_usd_per_million_tokens": float(self.input_usd_per_million_tokens),
            "output_usd_per_million_tokens": float(self.output_usd_per_million_tokens),
        }


@dataclass(frozen=True)
class PromptStamp:
    version: str = DEFAULT_PROMPT_VERSION
    template: str = DEFAULT_PROMPT_TEMPLATE

    def __post_init__(self) -> None:
        if not self.version.strip() or not self.template.strip():
            raise BaselineInputError("prompt version and template must be non-empty")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, str]:
        return {"version": self.version, "sha256": self.sha256, "template": self.template}


@dataclass(frozen=True)
class ProviderResponse:
    """Provider's exact textual response and measured token usage, with no parsing hidden inside."""

    raw_response: str
    input_tokens: int | None
    output_tokens: int | None
    response_schema: str = RESPONSE_SCHEMA
    reported_cost_usd: float | None = None


@dataclass(frozen=True)
class BuildStamp:
    """Exact package build that produced a qualification artifact."""

    package_version: str
    git_commit: str
    contract_version: str
    build_timestamp: str

    def __post_init__(self) -> None:
        for name in ("package_version", "git_commit", "contract_version", "build_timestamp"):
            if not str(getattr(self, name)).strip():
                raise BaselineInputError(f"build stamp {name} must be non-empty")

    @classmethod
    def installed(cls) -> "BuildStamp":
        from percept._build import build_info

        return cls(**build_info().to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "contract_version": self.contract_version,
            "build_timestamp": self.build_timestamp,
        }


@dataclass(frozen=True)
class ProfileStamp:
    """Pinned runtime/profile identity; the caller hashes the exact profile bytes."""

    name: str
    sha256: str
    source: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.source.strip():
            raise BaselineInputError("profile name and source must be non-empty")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise BaselineInputError("profile sha256 must be 64 lowercase hexadecimal characters")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256, "source": self.source}


@dataclass(frozen=True)
class PaidRunPolicy:
    """Immutable physical-call and dollar caps for one paid comparator run."""

    max_usd: float
    max_calls: int
    conservative_usd_per_call: float
    request_timeout_s: float

    def __post_init__(self) -> None:
        if not _number(self.max_usd) or self.max_usd <= 0:
            raise BaselineInputError("max_usd must be finite and positive")
        if type(self.max_calls) is not int or self.max_calls <= 0:
            raise BaselineInputError("max_calls must be a positive integer")
        if (
            not _number(self.conservative_usd_per_call)
            or self.conservative_usd_per_call <= 0
            or self.conservative_usd_per_call > self.max_usd
        ):
            raise BaselineInputError(
                "conservative_usd_per_call must be positive and no greater than max_usd"
            )
        if not _number(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise BaselineInputError("request_timeout_s must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_usd": float(self.max_usd),
            "max_calls": self.max_calls,
            "conservative_usd_per_call": float(self.conservative_usd_per_call),
            "attempts_per_call": 1,
            "request_timeout_s": float(self.request_timeout_s),
        }


@dataclass(frozen=True)
class ComparisonProtocolStamp:
    """Immutable evidence/chunk policy so a whole-clip lane is never labelled chunking."""

    mode: str = "whole_frozen_case"
    chunk_duration_s: float | None = None
    chunk_overlap_s: float | None = None
    source_clock_offset_mapping: str | None = None
    global_dedup_scoring: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "whole_frozen_case":
            if any(
                value is not None
                for value in (
                    self.chunk_duration_s,
                    self.chunk_overlap_s,
                    self.source_clock_offset_mapping,
                    self.global_dedup_scoring,
                )
            ):
                raise BaselineInputError(
                    "whole_frozen_case must leave every chunk-policy field null"
                )
            return
        if self.mode != "chunked":
            raise BaselineInputError(
                "comparison protocol mode must be whole_frozen_case or chunked"
            )
        if not _number(self.chunk_duration_s) or float(self.chunk_duration_s) <= 0:
            raise BaselineInputError("chunked mode requires positive chunk_duration_s")
        if (
            not _number(self.chunk_overlap_s)
            or float(self.chunk_overlap_s) < 0
            or float(self.chunk_overlap_s) >= float(self.chunk_duration_s)
        ):
            raise BaselineInputError(
                "chunked mode requires 0 <= chunk_overlap_s < chunk_duration_s"
            )
        if not str(self.source_clock_offset_mapping or "").strip():
            raise BaselineInputError("chunked mode requires source-clock offset mapping")
        if not str(self.global_dedup_scoring or "").strip():
            raise BaselineInputError("chunked mode requires global one-to-one de-dup/scoring")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "chunk_duration_s": self.chunk_duration_s,
            "chunk_overlap_s": self.chunk_overlap_s,
            "source_clock_offset_mapping": self.source_clock_offset_mapping,
            "global_dedup_scoring": self.global_dedup_scoring,
        }


@dataclass(frozen=True)
class RawVlmRequest:
    """One verified source and one exact corpus condition; ``media_bytes`` are the pinned bytes."""

    case_id: str
    condition: str
    source_path: str
    source_sha256: str
    duration_s: float
    media_bytes: bytes
    prompt: str
    prompt_sha256: str
    backend: str
    model: str


class RawVlmProvider(Protocol):
    def complete(self, request: RawVlmRequest) -> ProviderResponse:
        """Return one raw response for one source/condition request."""


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    dataset: str
    goal_kind: str
    condition: str
    source_path: Path
    declared_source_path: str
    source_sha256: str
    duration_s: float
    duration_source: str
    positives: tuple[tuple[float, float], ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "goal_kind": self.goal_kind,
            "condition": self.condition,
            "source_path": self.declared_source_path,
            "source_sha256": self.source_sha256,
            "duration_s": self.duration_s,
            "duration_source": self.duration_source,
            "positives": [list(span) for span in self.positives],
        }


@dataclass(frozen=True)
class FrozenCorpus:
    corpus_path: Path
    corpus_sha256: str
    corpus_version: str
    authoring_path: Path
    authoring_sha256: str
    cases: tuple[FrozenCase, ...]

    @property
    def source_stream_s(self) -> float:
        """Physical source time, deduplicated by the verified source-content hash."""
        durations: dict[str, float] = {}
        for case in self.cases:
            prior = durations.setdefault(case.source_sha256, case.duration_s)
            if not math.isclose(prior, case.duration_s, rel_tol=0.0, abs_tol=1e-6):
                raise BaselineInputError(
                    f"source {case.source_sha256} has inconsistent durations {prior} and "
                    f"{case.duration_s}"
                )
        return sum(durations.values())

    @property
    def scheduled_watch_s(self) -> float:
        """Source-goal exposure time: one armed watch for every corpus case."""
        return sum(case.duration_s for case in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.corpus_path),
            "sha256": self.corpus_sha256,
            "version": self.corpus_version,
            "authoring_path": str(self.authoring_path),
            "authoring_sha256": self.authoring_sha256,
            "n_cases": len(self.cases),
            "n_unique_conditions": len({case.condition for case in self.cases}),
            "n_unique_sources": len({case.source_sha256 for case in self.cases}),
            "source_bytes_verified": True,
            "goal_conditions_cross_verified": True,
        }


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _container_duration_s(path: Path) -> float:
    """Use the existing bench convention: MP4 frame count divided by FPS metadata."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by real preflight, not fixture tests
        raise BaselineInputError(
            f"{path}: corpus omits duration_s and OpenCV is unavailable for container metadata"
        ) from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise BaselineInputError(f"cannot open media for duration metadata: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not _number(fps) or not _number(frames) or fps <= 0 or frames <= 0:
        raise BaselineInputError(
            f"invalid duration metadata for {path}: frames={frames}, fps={fps}"
        )
    return frames / fps


def _resolve_source_path(declared: str, repo_root: Path) -> Path:
    path = Path(declared)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def preflight_frozen_corpus(
    corpus_path: str | Path = DEFAULT_CORPUS,
    authoring_path: str | Path = DEFAULT_AUTHORING,
    *,
    repo_root: str | Path = REPO,
    duration_resolver: Callable[[Path], float] = _container_duration_s,
) -> FrozenCorpus:
    """Verify authoring pin, exact goal set, every source byte hash, and every time bound."""
    corpus_path = Path(corpus_path).resolve()
    authoring_path = Path(authoring_path).resolve()
    repo_root = Path(repo_root).resolve()
    corpus_bytes = corpus_path.read_bytes()
    authoring_bytes = authoring_path.read_bytes()
    corpus_sha = _sha256_bytes(corpus_bytes)
    corpus_doc = json.loads(corpus_bytes)
    authoring_doc = json.loads(authoring_bytes)

    pin = authoring_doc.get("corpus") or {}
    if pin.get("sha256") != corpus_sha:
        raise BaselineInputError(
            f"authoring manifest pins corpus {pin.get('sha256')!r}, found {corpus_sha!r}"
        )
    if str(pin.get("version")) != str(corpus_doc.get("corpus_version")):
        raise BaselineInputError("authoring and stage corpus versions disagree")

    raw_cases = corpus_doc.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BaselineInputError("frozen corpus must contain a non-empty cases list")
    if pin.get("n_cases") != len(raw_cases) or corpus_doc.get("n_cases") != len(raw_cases):
        raise BaselineInputError("frozen corpus case-count pins disagree")

    authoring_entries = authoring_doc.get("entries")
    if not isinstance(authoring_entries, dict):
        raise BaselineInputError("authoring manifest entries must be an object")
    if len(authoring_entries) != pin.get("n_unique_goals"):
        raise BaselineInputError("authoring entry count disagrees with its unique-goal pin")
    authored_conditions = {entry.get("goal_text") for entry in authoring_entries.values()}
    corpus_conditions = {case.get("goal_text") for case in raw_cases}
    if authored_conditions != corpus_conditions or None in corpus_conditions:
        raise BaselineInputError("authoring and stage corpus goal conditions differ")
    if pin.get("n_unique_goals") != len(corpus_conditions):
        raise BaselineInputError("authoring unique-goal count disagrees with exact corpus goal set")

    seen_ids: set[str] = set()
    cases: list[FrozenCase] = []
    for raw in raw_cases:
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise BaselineInputError(f"case_id must be non-empty and unique: {case_id!r}")
        seen_ids.add(case_id)
        condition = raw.get("goal_text")
        if not isinstance(condition, str) or not condition.strip():
            raise BaselineInputError(f"{case_id}: goal_text must be a non-empty exact string")
        declared_path = raw.get("video_path")
        expected_sha = raw.get("video_sha256")
        if not isinstance(declared_path, str) or not isinstance(expected_sha, str):
            raise BaselineInputError(f"{case_id}: video path and sha256 are required")
        source_path = _resolve_source_path(declared_path, repo_root)
        if not source_path.is_file():
            raise BaselineInputError(f"{case_id}: pinned source is missing: {source_path}")
        actual_sha = _sha256_file(source_path)
        if actual_sha != expected_sha:
            raise BaselineInputError(
                f"{case_id}: source sha256 mismatch: expected {expected_sha}, found {actual_sha}"
            )

        declared_duration = raw.get("duration_s")
        if declared_duration is None:
            duration_s = float(duration_resolver(source_path))
            duration_source = "container_frame_count_div_fps"
        else:
            duration_s = float(declared_duration) if _number(declared_duration) else math.nan
            duration_source = "frozen_corpus"
        if not _number(duration_s) or duration_s <= 0:
            raise BaselineInputError(f"{case_id}: duration must be finite and positive")

        raw_positives = raw.get("positives")
        if not isinstance(raw_positives, list):
            raise BaselineInputError(f"{case_id}: positives must be a list")
        positives: list[tuple[float, float]] = []
        for index, span in enumerate(raw_positives):
            if not isinstance(span, list) or len(span) != 2 or not all(_number(x) for x in span):
                raise BaselineInputError(
                    f"{case_id}: positives[{index}] must be two finite numbers"
                )
            start_s, end_s = float(span[0]), float(span[1])
            if start_s < 0 or end_s < start_s or end_s > duration_s:
                raise BaselineInputError(
                    f"{case_id}: positives[{index}] is out of source bounds: {span!r} / {duration_s}"
                )
            positives.append((start_s, end_s))

        cases.append(
            FrozenCase(
                case_id=case_id,
                dataset=str(raw.get("dataset") or ""),
                goal_kind=str(raw.get("goal_kind") or ""),
                condition=condition,
                source_path=source_path,
                declared_source_path=declared_path,
                source_sha256=expected_sha,
                duration_s=duration_s,
                duration_source=duration_source,
                positives=tuple(positives),
            )
        )

    frozen = FrozenCorpus(
        corpus_path=corpus_path,
        corpus_sha256=corpus_sha,
        corpus_version=str(corpus_doc.get("corpus_version")),
        authoring_path=authoring_path,
        authoring_sha256=_sha256_bytes(authoring_bytes),
        cases=tuple(cases),
    )
    # Resolve cross-case duplicate-source contradictions during preflight, not after paid calls.
    _ = frozen.source_stream_s
    return frozen


def parse_temporal_response(
    raw_response: str,
    duration_s: float,
) -> tuple[tuple[float, float], ...]:
    """Parse the exact response schema and reject any malformed or out-of-bounds event."""
    if not isinstance(raw_response, str):
        raise BaselineInputError("raw provider response must be text")
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BaselineInputError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"events"}:
        raise BaselineInputError("response root must contain exactly the 'events' field")
    events = payload["events"]
    if not isinstance(events, list):
        raise BaselineInputError("response 'events' must be a list")
    parsed: list[tuple[float, float]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {"start_s", "end_s"}:
            raise BaselineInputError(
                f"events[{index}] must contain exactly start_s and end_s"
            )
        start, end = event["start_s"], event["end_s"]
        if not _number(start) or not _number(end):
            raise BaselineInputError(f"events[{index}] timestamps must be finite numbers")
        start_s, end_s = float(start), float(end)
        if start_s < 0 or end_s < start_s or end_s > duration_s:
            raise BaselineInputError(
                f"events[{index}] is out of bounds: [{start_s}, {end_s}] / {duration_s}"
            )
        parsed.append((start_s, end_s))
    return tuple(parsed)


def parse_percept_event_chain_response(
    raw_response: str,
    duration_s: float,
) -> tuple[tuple[float, float], ...]:
    """Strictly score the shipped Percept event-chain wire response.

    Percept's adapter parser intentionally only protects cognition.  This evaluator adds the
    qualification boundary that the source duration supplies: every native-video timestamp must
    remain on the source clock.  A provider response is never repaired or clipped for scoring.
    """
    if not isinstance(raw_response, str):
        raise BaselineInputError("raw provider response must be text")
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BaselineInputError(f"response is not valid JSON: {exc}") from exc
    allowed_root = {
        "satisfied", "confidence", "evidence", "action_present", "phase", "events"
    }
    if not isinstance(payload, dict) or "events" not in payload:
        raise BaselineInputError("event-chain response root must contain 'events'")
    unexpected = set(payload) - allowed_root
    if unexpected:
        raise BaselineInputError(
            f"event-chain response contains unexpected fields: {sorted(unexpected)!r}"
        )
    events = payload["events"]
    if not isinstance(events, list) or len(events) > 64:
        raise BaselineInputError("event-chain 'events' must be a list of at most 64 items")
    parsed: list[tuple[float, float]] = []
    prior_start = -1.0
    for index, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {"t_start", "t_end", "label"}:
            raise BaselineInputError(
                f"events[{index}] must contain exactly t_start, t_end, and label"
            )
        start = event["t_start"]
        raw_end = event["t_end"]
        if not _number(start) or (raw_end is not None and not _number(raw_end)):
            raise BaselineInputError(f"events[{index}] timestamps must be finite numbers")
        if not isinstance(event["label"], str):
            raise BaselineInputError(f"events[{index}].label must be text")
        start_s = float(start)
        end_s = start_s if raw_end is None else float(raw_end)
        if (
            start_s < 0
            or start_s < prior_start
            or end_s < start_s
            or end_s > duration_s
        ):
            raise BaselineInputError(
                f"events[{index}] is unordered or out of bounds: "
                f"[{start_s}, {end_s}] / {duration_s}"
            )
        prior_start = start_s
        parsed.append((start_s, end_s))
    return tuple(parsed)


def _expanded_ground_truth(
    span: tuple[float, float], duration_s: float, point_tolerance_s: float
) -> tuple[float, float]:
    start, end = span
    if start == end:
        return max(0.0, start - point_tolerance_s), min(duration_s, end + point_tolerance_s)
    return span


def _spans_match(
    prediction: tuple[float, float],
    truth: tuple[float, float],
    *,
    truth_was_point: bool,
) -> bool:
    p_start, p_end = prediction
    t_start, t_end = truth
    if truth_was_point:
        # A point label's declared +/- tolerance is a closed matching window.
        if p_start == p_end:
            return t_start <= p_start <= t_end
        return max(p_start, t_start) <= min(p_end, t_end)
    if p_start == p_end:
        return t_start <= p_start < t_end
    return max(p_start, t_start) < min(p_end, t_end)


def score_temporal_events(
    positives: Sequence[tuple[float, float]],
    predictions: Sequence[tuple[float, float]],
    *,
    duration_s: float,
    point_tolerance_s: float = DEFAULT_POINT_TOLERANCE_S,
) -> dict[str, Any]:
    """Maximum-cardinality, one-to-one occurrence matching on the source clock.

    A long prediction spanning several annotations can claim only one.  A duplicate prediction is
    a false positive.  Zero-width ground-truth instants alone receive the declared tolerance.
    """
    if not _number(point_tolerance_s) or point_tolerance_s < 0:
        raise BaselineInputError("point_tolerance_s must be finite and >= 0")
    original_truths = [tuple(span) for span in positives]
    truths = [
        _expanded_ground_truth(tuple(span), duration_s, float(point_tolerance_s))
        for span in original_truths
    ]
    preds = [tuple(span) for span in predictions]
    adjacency = [
        [
            truth_index
            for truth_index, truth in enumerate(truths)
            if _spans_match(
                pred,
                truth,
                truth_was_point=(
                    original_truths[truth_index][0] == original_truths[truth_index][1]
                ),
            )
        ]
        for pred in preds
    ]

    # Kuhn augmenting paths yield a maximum-cardinality bipartite match, unlike a local greedy
    # choice that can consume the only valid occurrence for a later, narrower prediction.
    prediction_for_truth: dict[int, int] = {}

    def augment(prediction_index: int, seen_truths: set[int]) -> bool:
        for truth_index in adjacency[prediction_index]:
            if truth_index in seen_truths:
                continue
            seen_truths.add(truth_index)
            prior = prediction_for_truth.get(truth_index)
            if prior is None or augment(prior, seen_truths):
                prediction_for_truth[truth_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(preds)):
        augment(prediction_index, set())

    pairs = sorted(
        (
            (prediction_index, truth_index)
            for truth_index, prediction_index in prediction_for_truth.items()
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    true_positives = len(pairs)
    false_positives = len(preds) - true_positives
    false_negatives = len(truths) - true_positives
    return {
        "matching_rule": "maximum-cardinality one-to-one temporal overlap",
        "point_ground_truth_tolerance_s": float(point_tolerance_s),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ground_truth_occurrences": len(truths),
        "predicted_occurrences": len(preds),
        "precision": true_positives / len(preds) if preds else None,
        "recall": true_positives / len(truths) if truths else None,
        "matched_pairs": [
            {"prediction_index": prediction_index, "ground_truth_index": truth_index}
            for prediction_index, truth_index in pairs
        ],
    }


def _usage_issue(response: ProviderResponse) -> str | None:
    for name in ("input_tokens", "output_tokens"):
        value = getattr(response, name)
        if type(value) is not int or value < 0:
            return f"{name} must be a measured non-negative integer"
    return None


def _call_cost(response: ProviderResponse, pricing: PricingStamp) -> float | None:
    if _usage_issue(response) is not None:
        return None
    assert response.input_tokens is not None and response.output_tokens is not None
    return (
        response.input_tokens * pricing.input_usd_per_million_tokens
        + response.output_tokens * pricing.output_usd_per_million_tokens
    ) / 1_000_000.0


class PaidBudgetLedger:
    """Crash-conservative budget object compatible with shipped Percept backends.

    A bound is committed at ``begin_attempt`` and the checkpoint callback runs before control is
    returned to the backend.  Unpriced and failed requests retain that full commitment: an absent
    billing receipt can never make a possibly-billed attempt disappear from the cap.
    """

    def __init__(
        self,
        policy: PaidRunPolicy,
        pricing: PricingStamp,
        *,
        on_reserve: Callable[[dict[str, Any]], None] | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.pricing = pricing
        self.on_reserve = on_reserve
        self.plan = {"price_snapshot": pricing.to_dict()}
        self.attempts_per_call = 1
        self.request_timeout_s = float(policy.request_timeout_s)
        self._next_id = 1
        self._physical_attempts = 0
        self._failed_attempts = 0
        self._metered_attempts = 0
        self._unpriced_attempts = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._committed_usd = 0.0
        self._billed_usd = 0.0
        self._inflight: dict[int, float] = {}
        if snapshot is not None:
            self._restore(snapshot)

    def _restore(self, snapshot: dict[str, Any]) -> None:
        expected = {
            "max_usd": float(self.policy.max_usd),
            "max_calls": self.policy.max_calls,
            "conservative_usd_per_call": float(self.policy.conservative_usd_per_call),
        }
        for name, expected_value in expected.items():
            if snapshot.get(name) != expected_value:
                raise BaselineInputError(f"checkpoint budget {name} is incompatible")
        integer_fields = (
            "next_reservation_id", "physical_attempts", "failed_attempts",
            "metered_attempts", "unpriced_attempts", "input_tokens", "output_tokens",
        )
        for name in integer_fields:
            value = snapshot.get(name)
            if type(value) is not int or value < 0:
                raise BaselineInputError(f"checkpoint budget {name} is invalid")
        self._next_id = snapshot["next_reservation_id"]
        self._physical_attempts = snapshot["physical_attempts"]
        self._failed_attempts = snapshot["failed_attempts"]
        self._metered_attempts = snapshot["metered_attempts"]
        self._unpriced_attempts = snapshot["unpriced_attempts"]
        self._input_tokens = snapshot["input_tokens"]
        self._output_tokens = snapshot["output_tokens"]
        for name in ("committed_usd", "billed_usd"):
            if not _number(snapshot.get(name)) or float(snapshot[name]) < 0:
                raise BaselineInputError(f"checkpoint budget {name} is invalid")
        self._committed_usd = float(snapshot["committed_usd"])
        self._billed_usd = float(snapshot["billed_usd"])
        raw_inflight = snapshot.get("inflight")
        if not isinstance(raw_inflight, list):
            raise BaselineInputError("checkpoint budget inflight must be a list")
        for item in raw_inflight:
            if (
                not isinstance(item, dict)
                or type(item.get("reservation_id")) is not int
                or not _number(item.get("reserved_usd"))
                or not math.isclose(
                    float(item.get("reserved_usd")),
                    float(self.policy.conservative_usd_per_call),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise BaselineInputError("checkpoint contains an invalid reservation")
            self._inflight[item["reservation_id"]] = float(item["reserved_usd"])
        if self._next_id != self._physical_attempts + 1:
            raise BaselineInputError("checkpoint reservation sequence is inconsistent")
        settled_and_inflight = (
            self._failed_attempts
            + self._metered_attempts
            + self._unpriced_attempts
            + len(self._inflight)
        )
        if settled_and_inflight != self._physical_attempts:
            raise BaselineInputError("checkpoint attempt accounting is inconsistent")
        expected_commitment = self._billed_usd + float(
            self._failed_attempts + self._unpriced_attempts + len(self._inflight)
        ) * float(self.policy.conservative_usd_per_call)
        if not math.isclose(
            self._committed_usd, expected_commitment, rel_tol=0.0, abs_tol=1e-10
        ):
            raise BaselineInputError("checkpoint dollar accounting is inconsistent")
        if self._physical_attempts > self.policy.max_calls:
            raise BaselineInputError("checkpoint exceeds immutable call cap")
        if self._committed_usd > self.policy.max_usd + 1e-12:
            raise BaselineInputError("checkpoint exceeds immutable dollar cap")

    def begin_attempt(self) -> dict[str, Any]:
        bound = float(self.policy.conservative_usd_per_call)
        if self._physical_attempts >= self.policy.max_calls:
            raise PaidBudgetExceeded("maximum physical provider calls reached")
        if self._committed_usd + bound > self.policy.max_usd + 1e-12:
            raise PaidBudgetExceeded(
                "next conservative per-call reservation would exceed maximum USD"
            )
        reservation = {"reservation_id": self._next_id, "reserved_usd": bound}
        self._next_id += 1
        self._physical_attempts += 1
        self._committed_usd += bound
        self._inflight[reservation["reservation_id"]] = bound
        if self.on_reserve is not None:
            # Synchronous by contract: the crash-safe reservation is durable before transport.
            self.on_reserve(dict(reservation))
        return reservation

    def _finish(self, reservation: dict[str, Any]) -> float:
        reservation_id = reservation.get("reservation_id")
        if type(reservation_id) is not int or reservation_id not in self._inflight:
            raise BaselineInputError("unknown or already-completed budget reservation")
        return self._inflight.pop(reservation_id)

    @staticmethod
    def _tokens(usage: dict[str, Any]) -> tuple[int, int]:
        values: list[int] = []
        for name in ("input_tokens", "output_tokens"):
            value = usage.get(name, 0)
            if type(value) is not int or value < 0:
                raise BaselineInputError(f"provider usage {name} must be a non-negative integer")
            values.append(value)
        return values[0], values[1]

    def complete_unpriced_attempt(
        self, reservation: dict[str, Any], usage: dict[str, Any]
    ) -> None:
        self._finish(reservation)
        input_tokens, output_tokens = self._tokens(usage)
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._unpriced_attempts += 1

    def complete_attempt(self, reservation: dict[str, Any], usage: dict[str, Any]) -> None:
        bound = self._finish(reservation)
        input_tokens, output_tokens = self._tokens(usage)
        cost = usage.get("cost_usd")
        if not _number(cost) or float(cost) < 0:
            raise BaselineInputError("provider cost_usd must be finite and non-negative")
        cost = float(cost)
        if cost > bound + 1e-12:
            # Keep the reservation committed. The declared bound was not conservative.
            self._inflight[reservation["reservation_id"]] = bound
            raise PaidBudgetExceeded("provider billed cost exceeded declared per-call bound")
        self._committed_usd -= bound - cost
        self._billed_usd += cost
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._metered_attempts += 1

    def fail_attempt(self, reservation: dict[str, Any]) -> None:
        self._finish(reservation)
        self._failed_attempts += 1

    def recover_inflight_as_failed(self) -> int:
        count = len(self._inflight)
        self._failed_attempts += count
        self._inflight.clear()
        return count

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_usd": float(self.policy.max_usd),
            "max_calls": self.policy.max_calls,
            "conservative_usd_per_call": float(self.policy.conservative_usd_per_call),
            "attempts_per_call": 1,
            "request_timeout_s": float(self.policy.request_timeout_s),
            "next_reservation_id": self._next_id,
            "physical_attempts": self._physical_attempts,
            "failed_attempts": self._failed_attempts,
            "metered_attempts": self._metered_attempts,
            "unpriced_attempts": self._unpriced_attempts,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "committed_usd": self._committed_usd,
            "billed_usd": self._billed_usd,
            "inflight": [
                {"reservation_id": key, "reserved_usd": value}
                for key, value in sorted(self._inflight.items())
            ],
        }


def run_baseline(
    corpus: FrozenCorpus,
    provider: RawVlmProvider,
    pricing: PricingStamp,
    *,
    prompt: PromptStamp = PromptStamp(),
    point_tolerance_s: float = DEFAULT_POINT_TOLERANCE_S,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one injected-provider call per case and return an auditable comparison artifact."""
    calls: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    known_cost = 0.0
    all_usage_complete = True

    for case in corpus.cases:
        media_bytes = case.source_path.read_bytes()
        execution_sha = _sha256_bytes(media_bytes)
        if execution_sha != case.source_sha256:
            raise BaselineInputError(
                f"{case.case_id}: source bytes changed after preflight; refusing provider call"
            )
        rendered_prompt = prompt.template.format(
            condition=case.condition,
            duration_s=case.duration_s,
        )
        request = RawVlmRequest(
            case_id=case.case_id,
            condition=case.condition,
            source_path=case.declared_source_path,
            source_sha256=case.source_sha256,
            duration_s=case.duration_s,
            media_bytes=media_bytes,
            prompt=rendered_prompt,
            prompt_sha256=_sha256_bytes(rendered_prompt.encode("utf-8")),
            backend=pricing.backend,
            model=pricing.model,
        )
        started = clock()
        response: ProviderResponse | None = None
        provider_error: str | None = None
        try:
            response = provider.complete(request)
        except Exception as exc:  # noqa: BLE001 - failures are measured misses, not run loss
            provider_error = f"{type(exc).__name__}: {exc}"
        if response is not None and not isinstance(response, ProviderResponse):
            provider_error = (
                "TypeError: provider must return raw_vlm_baseline.ProviderResponse, got "
                f"{type(response).__name__}"
            )
            response = None
        elapsed_s = max(0.0, clock() - started)

        parsed: tuple[tuple[float, float], ...] = ()
        schema_error: str | None = None
        usage_error: str | None = None
        raw_response: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        cost_usd: float | None = None
        if response is not None:
            raw_response = response.raw_response
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            try:
                parsed = parse_temporal_response(raw_response, case.duration_s)
            except BaselineInputError as exc:
                schema_error = str(exc)
            usage_error = _usage_issue(response)
            cost_usd = _call_cost(response, pricing)
            if usage_error is None:
                assert input_tokens is not None
                assert output_tokens is not None
                assert cost_usd is not None
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                known_cost += cost_usd
            else:
                all_usage_complete = False
        else:
            all_usage_complete = False
            usage_error = "provider produced no measurable response"

        scoring = score_temporal_events(
            case.positives,
            parsed if schema_error is None and provider_error is None else (),
            duration_s=case.duration_s,
            point_tolerance_s=point_tolerance_s,
        )
        calls.append(
            {
                "case": case.public_dict(),
                "request": {
                    "goals_in_call": 1,
                    "condition": case.condition,
                    "source_sha256": execution_sha,
                    "source_bytes": len(media_bytes),
                    "prompt_sha256": request.prompt_sha256,
                },
                "response": {
                    "schema": RESPONSE_SCHEMA,
                    "status": (
                        "provider_error" if provider_error is not None
                        else "rejected" if schema_error is not None
                        else "accepted"
                    ),
                    "raw_response": raw_response,
                    "raw_response_retained": response is not None,
                    "parsed_events": (
                        [list(span) for span in parsed] if schema_error is None else []
                    ),
                    "schema_error": schema_error,
                    "provider_error": provider_error,
                },
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "complete": usage_error is None and response is not None,
                    "error": usage_error,
                    "estimated_cost_usd": cost_usd,
                },
                "provider_elapsed_s": elapsed_s,
                "receiver_boundary_involved": False,
                "score": scoring,
            }
        )

    tp = sum(call["score"]["true_positives"] for call in calls)
    fp = sum(call["score"]["false_positives"] for call in calls)
    fn = sum(call["score"]["false_negatives"] for call in calls)
    predicted = tp + fp
    ground_truth = tp + fn
    source_hours = corpus.source_stream_s / 3600.0
    scheduled_hours = corpus.scheduled_watch_s / 3600.0
    total_cost = known_cost if all_usage_complete else None
    return {
        "kind": RESULT_KIND,
        "version": RESULT_VERSION,
        "comparison_scope": "receiver-independent raw provider temporal output",
        "not_measured": [
            "webhook delivery",
            "receiver arrival or acknowledgement",
            "Percept runtime action latency",
            "all-in hosted cost",
        ],
        "corpus": corpus.to_dict(),
        "comparison_protocol": ComparisonProtocolStamp().to_dict(),
        "execution": {
            "calls": len(calls),
            "goal_call_policy": "exactly one frozen goal condition per provider call",
            "default_goals_per_call": 1,
            "source_bytes_rehashed_immediately_before_each_call": True,
            "raw_responses_retained": True,
        },
        "provider": {"backend": pricing.backend, "model": pricing.model},
        "prompt": prompt.to_dict(),
        "pricing": pricing.to_dict(),
        "calls": calls,
        "receiver_independent_quality": {
            "boundary": "accepted raw provider response before any Percept cognition or receiver",
            "unit": "independent_ground_truth_occurrence",
            "matching_rule": "maximum-cardinality one-to-one temporal overlap",
            "point_ground_truth_tolerance_s": float(point_tolerance_s),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "ground_truth_occurrences": ground_truth,
            "predicted_occurrences": predicted,
            "precision": tp / predicted if predicted else None,
            "recall": tp / ground_truth if ground_truth else None,
            "accepted_calls": sum(c["response"]["status"] == "accepted" for c in calls),
            "rejected_calls": sum(c["response"]["status"] == "rejected" for c in calls),
            "provider_error_calls": sum(c["response"]["status"] == "provider_error" for c in calls),
        },
        "economy": {
            "scope": "declared provider token price only; not all-in hosting cost",
            "input_tokens": total_input_tokens if all_usage_complete else None,
            "output_tokens": total_output_tokens if all_usage_complete else None,
            "usage_complete_for_all_calls": all_usage_complete,
            "estimated_cost_usd": total_cost,
            "known_cost_lower_bound_usd": known_cost,
            "source_stream_hours": source_hours,
            "source_stream_hour_definition": "unique verified source-content seconds / 3600",
            "scheduled_watch_hours": scheduled_hours,
            "scheduled_watch_hour_definition": (
                "sum of source seconds for every armed source-goal case / 3600"
            ),
            "cost_usd_per_source_stream_hour": (
                total_cost / source_hours if total_cost is not None and source_hours > 0 else None
            ),
            "cost_usd_per_scheduled_watch_hour": (
                total_cost / scheduled_hours
                if total_cost is not None and scheduled_hours > 0
                else None
            ),
        },
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _credential_values() -> tuple[str, ...]:
    # Values remain in memory only and are never included in an artifact, even under their env name.
    return tuple(
        value for name in _CREDENTIAL_ENV_NAMES if (value := os.environ.get(name))
    )


def _redact_text(value: str | None, secrets: Sequence[str]) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    redacted = value
    changed = False
    for secret in secrets:
        if secret and secret in redacted:
            redacted = redacted.replace(secret, "[REDACTED_CREDENTIAL]")
            changed = True
    return redacted, changed


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _qualification_summary(
    *,
    enabled: bool,
    contract_path: str | Path | None,
    approved_contract_sha256: str | None,
) -> dict[str, Any]:
    if not enabled:
        if contract_path is not None or approved_contract_sha256 is not None:
            raise BaselineInputError(
                "qualification contract arguments require qualification_mode=True"
            )
        return {"mode": "exploratory", "approved_launch_contract": None}
    if contract_path is None or approved_contract_sha256 is None:
        raise BaselineInputError(
            "qualification mode requires contract_path and approved_contract_sha256"
        )
    expected = str(approved_contract_sha256).lower()
    if not _SHA256_RE.fullmatch(expected):
        raise BaselineInputError("approved contract sha256 must be 64 hexadecimal characters")
    resolved = Path(contract_path).resolve()
    actual = _sha256_file(resolved)
    if actual != expected:
        raise BaselineInputError(
            f"approved launch contract hash mismatch: expected {expected}, found {actual}"
        )
    try:
        import qualification_contract

        _document, summary = qualification_contract.load(resolved, require_approved=True)
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise BaselineInputError(f"launch qualification contract rejected: {exc}") from exc
    if summary.get("sha256") != expected:
        raise BaselineInputError("qualification validator returned a different contract hash")
    return {"mode": "qualification", "approved_launch_contract": summary}


def _paid_compatibility(
    corpus: FrozenCorpus,
    pricing: PricingStamp,
    prompt: PromptStamp,
    build: BuildStamp,
    profile: ProfileStamp,
    policy: PaidRunPolicy,
    protocol: ComparisonProtocolStamp,
    qualification: dict[str, Any],
    point_tolerance_s: float,
) -> dict[str, Any]:
    return {
        "corpus": {
            **corpus.to_dict(),
            "ordered_cases": [case.public_dict() for case in corpus.cases],
        },
        "provider": {"backend": pricing.backend, "model": pricing.model},
        "pricing": pricing.to_dict(),
        "prompt": prompt.to_dict(),
        "build": build.to_dict(),
        "profile": profile.to_dict(),
        "budget_policy": policy.to_dict(),
        "comparison_protocol": protocol.to_dict(),
        "qualification": qualification,
        "point_ground_truth_tolerance_s": float(point_tolerance_s),
    }


def _case_request(
    case: FrozenCase, pricing: PricingStamp, prompt: PromptStamp
) -> RawVlmRequest:
    media_bytes = case.source_path.read_bytes()
    execution_sha = _sha256_bytes(media_bytes)
    if execution_sha != case.source_sha256:
        raise BaselineInputError(
            f"{case.case_id}: source bytes changed after preflight; refusing provider call"
        )
    rendered_prompt = prompt.template.format(
        condition=case.condition,
        duration_s=case.duration_s,
    )
    return RawVlmRequest(
        case_id=case.case_id,
        condition=case.condition,
        source_path=case.declared_source_path,
        source_sha256=case.source_sha256,
        duration_s=case.duration_s,
        media_bytes=media_bytes,
        prompt=rendered_prompt,
        prompt_sha256=_sha256_bytes(rendered_prompt.encode("utf-8")),
        backend=pricing.backend,
        model=pricing.model,
    )


def _score_paid_response(
    *,
    case: FrozenCase,
    request: RawVlmRequest,
    response: ProviderResponse | None,
    provider_error: str | None,
    pricing: PricingStamp,
    point_tolerance_s: float,
    elapsed_s: float,
    secrets: Sequence[str],
    attempt_reserved: bool,
) -> dict[str, Any]:
    raw_response: str | None = None
    schema_error: str | None = None
    usage_error: str | None = None
    parsed: tuple[tuple[float, float], ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    response_schema = response.response_schema if response is not None else None
    credential_redacted = False
    if response is not None:
        raw_response, credential_redacted = _redact_text(response.raw_response, secrets)
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
        usage_error = _usage_issue(response)
        cost_usd = (
            response.reported_cost_usd
            if response.reported_cost_usd is not None
            else _call_cost(response, pricing)
        )
        try:
            if response_schema == RESPONSE_SCHEMA:
                parsed = parse_temporal_response(response.raw_response, case.duration_s)
            elif response_schema == PERCEPT_EVENT_CHAIN_SCHEMA:
                parsed = parse_percept_event_chain_response(response.raw_response, case.duration_s)
            else:
                raise BaselineInputError(
                    f"provider declared unsupported response schema {response_schema!r}"
                )
        except BaselineInputError as exc:
            schema_error, changed = _redact_text(str(exc), secrets)
            credential_redacted = credential_redacted or changed
    else:
        usage_error = "provider produced no measurable response"
    provider_error, changed = _redact_text(provider_error, secrets)
    credential_redacted = credential_redacted or changed
    scoring = score_temporal_events(
        case.positives,
        parsed if schema_error is None and provider_error is None else (),
        duration_s=case.duration_s,
        point_tolerance_s=point_tolerance_s,
    )
    return {
        "case": case.public_dict(),
        "request": {
            "goals_in_call": 1,
            "condition": case.condition,
            "source_sha256": request.source_sha256,
            "source_bytes": len(request.media_bytes),
            "prompt_sha256": request.prompt_sha256,
        },
        "response": {
            "schema": response_schema,
            "status": (
                "provider_error" if provider_error is not None
                else "rejected" if schema_error is not None
                else "accepted"
            ),
            "raw_response": raw_response,
            "raw_response_retained": response is not None,
            "raw_response_exact": response is not None and not credential_redacted,
            "credential_redacted": credential_redacted,
            "parsed_events": [list(span) for span in parsed] if schema_error is None else [],
            "schema_error": schema_error,
            "provider_error": provider_error,
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "complete": usage_error is None and response is not None,
            "error": usage_error,
            "estimated_or_reported_cost_usd": cost_usd,
        },
        "budget_reservation_observed_before_transport": attempt_reserved,
        "provider_elapsed_s": elapsed_s,
        "receiver_boundary_involved": False,
        "score": scoring,
    }


def _recovered_unknown_call(
    case: FrozenCase,
    pricing: PricingStamp,
    prompt: PromptStamp,
    point_tolerance_s: float,
) -> dict[str, Any]:
    request = _case_request(case, pricing, prompt)
    return _score_paid_response(
        case=case,
        request=request,
        response=None,
        provider_error=(
            "RecoveredUnknownOutcome: process stopped after paid reservation; "
            "request is not replayed and is scored as a miss"
        ),
        pricing=pricing,
        point_tolerance_s=point_tolerance_s,
        elapsed_s=0.0,
        secrets=(),
        attempt_reserved=True,
    )


def run_paid_baseline(
    corpus: FrozenCorpus,
    provider: RawVlmProvider,
    pricing: PricingStamp,
    *,
    policy: PaidRunPolicy,
    build: BuildStamp,
    profile: ProfileStamp,
    checkpoint_path: str | Path,
    prompt: PromptStamp = PromptStamp(),
    protocol: ComparisonProtocolStamp = ComparisonProtocolStamp(),
    point_tolerance_s: float = DEFAULT_POINT_TOLERANCE_S,
    qualification_mode: bool = False,
    contract_path: str | Path | None = None,
    approved_contract_sha256: str | None = None,
    resume: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run the paid lane with exact identity, admission, crash, and scoring controls.

    This runner currently implements a whole-frozen-case native-video comparator.  Chunking is a
    distinct protocol because it changes provider-call count, temporal mapping, de-duplication, and
    cost.  A caller cannot silently label this implementation as a chunked baseline.
    """
    if protocol.mode != "whole_frozen_case":
        raise BaselineInputError(
            "this runner implements whole_frozen_case only; a chunked runner must apply the "
            "declared offsets and global one-to-one de-dup/scoring before it may use that label"
        )
    if not _number(point_tolerance_s) or point_tolerance_s < 0:
        raise BaselineInputError("point_tolerance_s must be finite and >= 0")
    if getattr(provider, "backend", None) != pricing.backend:
        raise BaselineInputError("provider backend does not match the explicit pricing backend")
    if getattr(provider, "model", None) != pricing.model:
        raise BaselineInputError("provider model does not match the explicit pricing model")
    if getattr(provider, "supports_native_video", None) is not True:
        raise BaselineInputError("paid provider must declare native-video transport support")
    if getattr(provider, "budget_reservation_before_transport", None) is not True:
        raise BaselineInputError("paid provider must reserve budget immediately before transport")
    configure_budget = getattr(provider, "configure_request_budget", None)
    if not callable(configure_budget):
        raise BaselineInputError("paid provider must accept the comparator request budget")

    qualification = _qualification_summary(
        enabled=qualification_mode,
        contract_path=contract_path,
        approved_contract_sha256=approved_contract_sha256,
    )
    compatibility = _paid_compatibility(
        corpus,
        pricing,
        prompt,
        build,
        profile,
        policy,
        protocol,
        qualification,
        float(point_tolerance_s),
    )
    compatibility_sha256 = _canonical_sha256(compatibility)
    secrets = _credential_values()
    serialized_identity = json.dumps(compatibility, sort_keys=True, ensure_ascii=False)
    if any(secret in serialized_identity for secret in secrets):
        raise BaselineInputError("run provenance contains a credential value; refusing artifact")

    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.exists() and not resume:
        raise BaselineInputError(
            f"checkpoint already exists; pass resume=True or choose a new path: {checkpoint_path}"
        )
    if resume and not checkpoint_path.exists():
        raise BaselineInputError(f"resume checkpoint does not exist: {checkpoint_path}")

    if resume:
        try:
            state = json.loads(checkpoint_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BaselineInputError(f"cannot read paid checkpoint: {exc}") from exc
        if state.get("kind") != CHECKPOINT_KIND or state.get("version") != CHECKPOINT_VERSION:
            raise BaselineInputError("checkpoint kind/version is incompatible")
        if (
            state.get("compatibility_sha256") != compatibility_sha256
            or state.get("compatibility") != compatibility
        ):
            raise BaselineInputError("checkpoint compatibility differs from this exact run")
        calls = state.get("calls")
        next_index = state.get("next_case_index")
        if not isinstance(calls, list) or type(next_index) is not int:
            raise BaselineInputError("checkpoint progress is malformed")
        if next_index < 0 or next_index > len(corpus.cases) or len(calls) != next_index:
            raise BaselineInputError("checkpoint case progress is inconsistent")
        expected_prefix = [case.case_id for case in corpus.cases[:next_index]]
        actual_prefix = [((call.get("case") or {}).get("case_id")) for call in calls]
        if actual_prefix != expected_prefix:
            raise BaselineInputError("checkpoint calls are not the exact ordered corpus prefix")
        if isinstance(state.get("result"), dict) and state.get("status") == "complete":
            return state["result"]
    else:
        state = {
            "kind": CHECKPOINT_KIND,
            "version": CHECKPOINT_VERSION,
            "compatibility_sha256": compatibility_sha256,
            "compatibility": compatibility,
            "status": "initialized",
            "next_case_index": 0,
            "pending_call": None,
            "calls": [],
            "budget": None,
            "result": None,
        }
        calls = state["calls"]
        next_index = 0

    current_index = next_index

    def checkpoint_reservation(reservation: dict[str, Any]) -> None:
        state["status"] = "provider_attempt_reserved"
        state["pending_call"] = {
            "case_index": current_index,
            "case_id": corpus.cases[current_index].case_id,
            "reservation": reservation,
        }
        state["budget"] = ledger.snapshot()
        _atomic_json(checkpoint_path, state)

    ledger = PaidBudgetLedger(
        policy,
        pricing,
        on_reserve=checkpoint_reservation,
        snapshot=state.get("budget") if resume else None,
    )

    pending = state.get("pending_call")
    if pending is not None:
        if (
            next_index >= len(corpus.cases)
            or not isinstance(pending, dict)
            or pending.get("case_index") != next_index
            or pending.get("case_id") != corpus.cases[next_index].case_id
            or len(ledger.snapshot()["inflight"]) != 1
        ):
            raise BaselineInputError("checkpoint pending reservation is inconsistent")
        if ledger.recover_inflight_as_failed() != 1:
            raise BaselineInputError("checkpoint must recover exactly one pending attempt")
        calls.append(
            _recovered_unknown_call(
                corpus.cases[next_index], pricing, prompt, float(point_tolerance_s)
            )
        )
        next_index += 1
        current_index = next_index
        state["next_case_index"] = next_index
        state["pending_call"] = None
        state["status"] = "resumed_after_unknown_paid_outcome"
        state["budget"] = ledger.snapshot()
        _atomic_json(checkpoint_path, state)
    elif ledger.snapshot()["inflight"]:
        raise BaselineInputError("checkpoint contains inflight budget without a pending case")

    configure_budget(ledger)
    terminal_status: str | None = None
    for index in range(next_index, len(corpus.cases)):
        current_index = index
        case = corpus.cases[index]
        request = _case_request(case, pricing, prompt)
        before_attempts = ledger.snapshot()["physical_attempts"]
        started = clock()
        response: ProviderResponse | None = None
        provider_error: str | None = None
        try:
            response = provider.complete(request)
        except PaidBudgetExceeded:
            if ledger.snapshot()["physical_attempts"] != before_attempts:
                raise
            terminal_status = "budget_exhausted"
            state["status"] = terminal_status
            state["pending_call"] = None
            state["budget"] = ledger.snapshot()
            _atomic_json(checkpoint_path, state)
            break
        except Exception as exc:  # noqa: BLE001 - paid failures stay in denominators
            provider_error = f"{type(exc).__name__}: {exc}"
        elapsed_s = max(0.0, clock() - started)
        if response is not None and not isinstance(response, ProviderResponse):
            provider_error = (
                "TypeError: provider must return raw_vlm_baseline.ProviderResponse, got "
                f"{type(response).__name__}"
            )
            response = None
        after = ledger.snapshot()
        attempt_delta = after["physical_attempts"] - before_attempts
        if attempt_delta not in (0, 1):
            raise BaselineInputError("provider used more than one physical attempt for one case")
        if attempt_delta == 0 and provider_error is None:
            raise BaselineInputError("provider returned without reserving a physical attempt")
        if after["inflight"]:
            raise BaselineInputError("provider returned without settling its budget reservation")

        calls.append(
            _score_paid_response(
                case=case,
                request=request,
                response=response,
                provider_error=provider_error,
                pricing=pricing,
                point_tolerance_s=float(point_tolerance_s),
                elapsed_s=elapsed_s,
                secrets=secrets,
                attempt_reserved=attempt_delta == 1,
            )
        )
        next_index = index + 1
        state["next_case_index"] = next_index
        state["pending_call"] = None
        state["budget"] = ledger.snapshot()
        state["status"] = "running"
        _atomic_json(checkpoint_path, state)
        if attempt_delta == 0:
            # A local/provider-configuration failure may be safely recorded, but without a
            # reservation we cannot prove that retrying would not cause unledgered transport.
            terminal_status = "provider_failed_before_reservation"
            break

    complete_corpus = next_index == len(corpus.cases)
    if terminal_status is None:
        terminal_status = "complete" if complete_corpus else "stopped"
    tp = sum(call["score"]["true_positives"] for call in calls)
    fp = sum(call["score"]["false_positives"] for call in calls)
    scored_fn = sum(call["score"]["false_negatives"] for call in calls)
    unprocessed = corpus.cases[next_index:]
    unprocessed_fn = sum(len(case.positives) for case in unprocessed)
    fn = scored_fn + unprocessed_fn
    predicted = tp + fp
    ground_truth = tp + fn
    budget_snapshot = ledger.snapshot()
    response_usage_complete = complete_corpus and all(
        call["usage"]["complete"] for call in calls
    )
    known_response_cost = sum(
        float(call["usage"]["estimated_or_reported_cost_usd"] or 0.0) for call in calls
    )
    source_hours = corpus.source_stream_s / 3600.0
    scheduled_hours = corpus.scheduled_watch_s / 3600.0
    result = {
        "kind": RESULT_KIND,
        "version": PAID_RESULT_VERSION,
        "status": terminal_status,
        "complete_corpus": complete_corpus,
        "comparison_scope": "receiver-independent paid raw provider temporal output",
        "not_measured": [
            "webhook delivery", "receiver arrival or acknowledgement",
            "Percept runtime action latency", "all-in hosted cost",
        ],
        "compatibility_sha256": compatibility_sha256,
        **compatibility,
        "execution": {
            "completed_or_recovered_cases": next_index,
            "unprocessed_cases": [case.case_id for case in unprocessed],
            "goal_call_policy": "exactly one frozen goal condition per provider call",
            "source_bytes_rehashed_immediately_before_each_call": True,
            "raw_responses_checkpointed_after_each_settled_call": True,
            "retries": 0,
        },
        "calls": calls,
        "receiver_independent_quality": {
            "boundary": "accepted raw provider response before Percept cognition or receiver",
            "unit": "independent_ground_truth_occurrence",
            "matching_rule": "maximum-cardinality one-to-one temporal overlap",
            "point_ground_truth_tolerance_s": float(point_tolerance_s),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "unprocessed_ground_truth_counted_as_false_negatives": unprocessed_fn,
            "ground_truth_occurrences": ground_truth,
            "predicted_occurrences": predicted,
            "precision": tp / predicted if predicted else None,
            "recall": tp / ground_truth if ground_truth else None,
            "accepted_calls": sum(c["response"]["status"] == "accepted" for c in calls),
            "rejected_calls": sum(c["response"]["status"] == "rejected" for c in calls),
            "provider_error_calls": sum(
                c["response"]["status"] == "provider_error" for c in calls
            ),
        },
        "economy": {
            "scope": "declared provider price only; not all-in hosting cost",
            "budget": budget_snapshot,
            "usage_complete_for_full_corpus": response_usage_complete,
            "estimated_or_reported_cost_usd": (
                known_response_cost if response_usage_complete else None
            ),
            "known_response_cost_lower_bound_usd": known_response_cost,
            "crash_conservative_committed_usd": budget_snapshot["committed_usd"],
            "source_stream_hours": source_hours,
            "scheduled_watch_hours": scheduled_hours,
            "committed_usd_per_source_stream_hour": (
                budget_snapshot["committed_usd"] / source_hours if source_hours else None
            ),
            "committed_usd_per_scheduled_watch_hour": (
                budget_snapshot["committed_usd"] / scheduled_hours if scheduled_hours else None
            ),
        },
    }
    state["status"] = terminal_status
    state["next_case_index"] = next_index
    state["pending_call"] = None
    state["budget"] = budget_snapshot
    state["result"] = result
    _atomic_json(checkpoint_path, state)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strict raw-VLM comparison. Default mode is offline preflight and never imports or "
            "calls a provider; paid execution requires two explicit opt-ins and complete caps."
        )
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--authoring", type=Path, default=DEFAULT_AUTHORING)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute-paid", action="store_true")
    parser.add_argument("--confirm-paid-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend")
    parser.add_argument("--model")
    parser.add_argument("--pricing-version")
    parser.add_argument("--pricing-source")
    parser.add_argument("--input-usd-per-million-tokens", type=float)
    parser.add_argument("--output-usd-per-million-tokens", type=float)
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--conservative-usd-per-call", type=float)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--profile-name")
    parser.add_argument("--profile-sha256")
    parser.add_argument("--profile-source")
    parser.add_argument("--qualification-mode", action="store_true")
    parser.add_argument("--qualification-contract", type=Path)
    parser.add_argument("--approved-contract-sha256")
    args = parser.parse_args(argv)
    corpus = preflight_frozen_corpus(
        args.corpus,
        args.authoring,
        repo_root=args.repo_root,
    )
    if args.execute_paid:
        if not args.confirm_paid_run:
            parser.error("--execute-paid requires --confirm-paid-run")
        required = {
            "--backend": args.backend,
            "--model": args.model,
            "--pricing-version": args.pricing_version,
            "--pricing-source": args.pricing_source,
            "--input-usd-per-million-tokens": args.input_usd_per_million_tokens,
            "--output-usd-per-million-tokens": args.output_usd_per_million_tokens,
            "--max-usd": args.max_usd,
            "--max-calls": args.max_calls,
            "--conservative-usd-per-call": args.conservative_usd_per_call,
            "--request-timeout-s": args.request_timeout_s,
            "--checkpoint": args.checkpoint,
            "--profile-name": args.profile_name,
            "--profile-sha256": args.profile_sha256,
            "--profile-source": args.profile_source,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("paid execution requires: " + ", ".join(missing))
        if args.qualification_mode and (
            args.qualification_contract is None or args.approved_contract_sha256 is None
        ):
            parser.error(
                "--qualification-mode requires --qualification-contract and "
                "--approved-contract-sha256"
            )
        assert args.checkpoint is not None
        if args.out is not None and args.out.resolve() == args.checkpoint.resolve():
            parser.error("--out and --checkpoint must be different paths")
        pricing = PricingStamp(
            backend=args.backend,
            model=args.model,
            pricing_version=args.pricing_version,
            pricing_source=args.pricing_source,
            input_usd_per_million_tokens=args.input_usd_per_million_tokens,
            output_usd_per_million_tokens=args.output_usd_per_million_tokens,
        )
        policy = PaidRunPolicy(
            max_usd=args.max_usd,
            max_calls=args.max_calls,
            conservative_usd_per_call=args.conservative_usd_per_call,
            request_timeout_s=args.request_timeout_s,
        )
        profile = ProfileStamp(
            name=args.profile_name,
            sha256=args.profile_sha256,
            source=args.profile_source,
        )
        # Importing the provider (and therefore any vendor SDK) is impossible in default mode.
        from raw_vlm_percept_adapter import PerceptBackendProvider

        provider = PerceptBackendProvider(backend=args.backend, model=args.model)
        report = run_paid_baseline(
            corpus,
            provider,
            pricing,
            policy=policy,
            build=BuildStamp.installed(),
            profile=profile,
            checkpoint_path=args.checkpoint,
            qualification_mode=args.qualification_mode,
            contract_path=args.qualification_contract,
            approved_contract_sha256=args.approved_contract_sha256,
            resume=args.resume,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out is not None:
            args.out.write_text(rendered)
        else:
            print(rendered, end="")
        return 0

    report = {
        "kind": "percept-raw-vlm-baseline-preflight",
        "version": 1,
        "provider_called": False,
        "comparison_protocol": ComparisonProtocolStamp().to_dict(),
        "corpus": corpus.to_dict(),
        "source_stream_hours": corpus.source_stream_s / 3600.0,
        "scheduled_watch_hours": corpus.scheduled_watch_s / 3600.0,
        "duration_sources": {
            source: sum(case.duration_source == source for case in corpus.cases)
            for source in sorted({case.duration_source for case in corpus.cases})
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
