#!/usr/bin/env python3
"""Offline counterfactual replay for saved per-verdict evaluation artifacts.

This tool intentionally imports no Percept runtime or provider adapter.  Its input is the
``verdict_rows`` already persisted by ``live_run.py``; it replays only the cognition boundary
(``truth=yes|no|unknown``) through candidate rising-edge policies.  It therefore answers a narrow
question: *would a different episode-exit latch have changed incidents for the exact same paid
judgments?*  It does not claim that those judgments are correct on unseen media.

Example::

    python eval/stage-validation/verdict_policy_replay.py \
      eval/stage-validation/results/single-clip-corrected-3p1-flash-lite-20260802.json

The JSON report is written to stdout.  ``--output`` writes the same report to a file.  No network,
credential, video decoder, or provider call is used.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Sequence


REPORT_SCHEMA = "percept.offline-verdict-policy-replay/1"
MATCHING_RULE = "maximum-cardinality one-to-one temporal overlap"
DEFAULT_OFF_DWELL_S = (2.0, 5.0)
TRUTHS = frozenset({"yes", "no", "unknown"})


class ReplayInputError(ValueError):
    """The saved artifact cannot support a deterministic verdict replay."""


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _format_seconds(value: float) -> str:
    return f"{value:g}".replace(".", "p")


@dataclass(frozen=True)
class ReplayPolicy:
    """A rising-edge latch plus one mutually exclusive episode-exit rule.

    ``false_count=1`` is today's on-false behavior: one confident ``no`` closes the episode and
    re-arms the edge.  ``false_count>1`` requires consecutive confident-false rows.  ``off_dwell_s``
    instead requires an uninterrupted confident-false run spanning that much source time.

    Unknown/ASK rows never arm or close an episode.  They break a *pending* exit proof because a
    count described as consecutive, or a dwell described as sustained, cannot bridge unsupported
    evidence.
    """

    name: str
    require_false_first: bool
    false_count: int | None = 1
    off_dwell_s: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy name must be non-empty")
        modes = int(self.false_count is not None) + int(self.off_dwell_s is not None)
        if modes != 1:
            raise ValueError("policy must declare exactly one exit rule")
        if self.false_count is not None and (
            type(self.false_count) is not int or self.false_count < 1
        ):
            raise ValueError("false_count must be a positive integer")
        if self.off_dwell_s is not None and (
            not _finite_number(self.off_dwell_s) or float(self.off_dwell_s) <= 0
        ):
            raise ValueError("off_dwell_s must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_false_first": self.require_false_first,
            "exit_rule": (
                {"kind": "consecutive_confident_false", "count": self.false_count}
                if self.false_count is not None
                else {"kind": "sustained_confident_false", "dwell_s": self.off_dwell_s}
            ),
            "unknown_behavior": "preserve_latch_and_break_pending_exit_proof",
        }


def default_policies(off_dwells_s: Sequence[float] = DEFAULT_OFF_DWELL_S) -> tuple[ReplayPolicy, ...]:
    """The launch-feasibility matrix requested for the saved mozzarella stream."""
    policies = [
        ReplayPolicy("current", require_false_first=False, false_count=1),
        ReplayPolicy("false_first", require_false_first=True, false_count=1),
        ReplayPolicy("false_first_exit_k2", require_false_first=True, false_count=2),
        ReplayPolicy("false_first_exit_k3", require_false_first=True, false_count=3),
    ]
    for dwell in off_dwells_s:
        dwell = float(dwell)
        policies.append(
            ReplayPolicy(
                f"false_first_off_dwell_{_format_seconds(dwell)}s",
                require_false_first=True,
                false_count=None,
                off_dwell_s=dwell,
            )
        )
    return tuple(policies)


def _validated_rows(rows: object, *, case_id: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ReplayInputError(f"{case_id}: verdict_rows must be a non-empty list")
    validated: list[dict[str, Any]] = []
    previous_t = -math.inf
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ReplayInputError(f"{case_id}: verdict_rows[{index}] must be an object")
        t = raw.get("media_ts")
        truth = raw.get("truth")
        if not _finite_number(t) or float(t) < 0:
            raise ReplayInputError(f"{case_id}: verdict_rows[{index}].media_ts is invalid")
        t = float(t)
        if t < previous_t:
            raise ReplayInputError(f"{case_id}: verdict_rows must be ordered by media_ts")
        if truth not in TRUTHS:
            raise ReplayInputError(
                f"{case_id}: verdict_rows[{index}].truth must be yes, no, or unknown"
            )
        validated.append({**raw, "media_ts": t, "truth": truth})
        previous_t = t
    return validated


def _validated_intervals(raw: object, *, case_id: str, media_end_s: float) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        raise ReplayInputError(f"{case_id}: gt_event_intervals_raw must be a list")
    intervals: list[tuple[float, float]] = []
    prior_start = -math.inf
    for index, span in enumerate(raw):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ReplayInputError(f"{case_id}: GT interval {index} must be [start, end]")
        start, end = span
        if not _finite_number(start) or not _finite_number(end):
            raise ReplayInputError(f"{case_id}: GT interval {index} must be finite")
        start, end = float(start), float(end)
        if start < prior_start or start < 0 or end < start or end > media_end_s:
            raise ReplayInputError(
                f"{case_id}: GT interval {index} is unordered or out of bounds: {span}"
            )
        intervals.append((start, end))
        prior_start = start
    return intervals


def extract_cases(document: object, *, selected_case_id: str | None = None) -> list[dict[str, Any]]:
    """Extract replayable case objects from a case JSON or a full ``live_run`` result."""
    if not isinstance(document, dict):
        raise ReplayInputError("artifact root must be an object")
    if "verdict_rows" in document:
        candidates = [document]
    else:
        groups = document.get("cases")
        if not isinstance(groups, dict):
            raise ReplayInputError("artifact has neither verdict_rows nor a cases object")
        candidates = []
        for group in groups.values():
            if isinstance(group, list):
                candidates.extend(row for row in group if isinstance(row, dict))

    if selected_case_id is not None:
        candidates = [row for row in candidates if str(row.get("case_id")) == selected_case_id]
        if not candidates:
            raise ReplayInputError(f"case_id not found: {selected_case_id}")
    replayable = [row for row in candidates if row.get("verdict_rows") is not None]
    if not replayable:
        raise ReplayInputError("artifact contains no cases with verdict_rows")
    return replayable


def simulate_fires(rows: Sequence[dict[str, Any]], policy: ReplayPolicy) -> list[float]:
    """Replay one verdict stream and return emitted source-clock timestamps."""
    armed = not policy.require_false_first
    episode_open = False
    pending_false_count = 0
    pending_false_since: float | None = None
    fires: list[float] = []

    def clear_exit_candidate() -> None:
        nonlocal pending_false_count, pending_false_since
        pending_false_count = 0
        pending_false_since = None

    for row in rows:
        t = float(row["media_ts"])
        truth = str(row["truth"])

        if truth == "unknown":
            clear_exit_candidate()
            continue

        if truth == "yes":
            clear_exit_candidate()
            if armed:
                fires.append(t)
                armed = False
                episode_open = True
            elif policy.require_false_first and not episode_open:
                # Already true at startup: latch the unreported episode shut until a proven exit.
                episode_open = True
            continue

        # Confident false.  When no episode is open, one such row is the false-first baseline;
        # exit hysteresis is intentionally only an episode-*closure* policy.
        if not episode_open:
            armed = True
            clear_exit_candidate()
            continue

        should_close = False
        if policy.false_count is not None:
            pending_false_count += 1
            should_close = pending_false_count >= policy.false_count
        else:
            if pending_false_since is None:
                pending_false_since = t
            should_close = t - pending_false_since >= float(policy.off_dwell_s)

        if should_close:
            episode_open = False
            armed = True
            clear_exit_candidate()

    return fires


def _spans_match(prediction_t: float, truth: tuple[float, float]) -> bool:
    start, end = truth
    # Duration GT follows the existing strict scorer's half-open interval contract.  A point label
    # is exact here because the replay artifact declares no separate point tolerance.
    return prediction_t == start if start == end else start <= prediction_t < end


def strict_score(
    truths: Sequence[tuple[float, float]], predictions: Sequence[float]
) -> dict[str, Any]:
    """Maximum-cardinality, one-to-one strict score; duplicate fires are false positives."""
    adjacency = [
        [truth_index for truth_index, truth in enumerate(truths) if _spans_match(t, truth)]
        for t in predictions
    ]
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

    for prediction_index in range(len(predictions)):
        augment(prediction_index, set())

    pairs = sorted(
        (
            (prediction_index, truth_index)
            for truth_index, prediction_index in prediction_for_truth.items()
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    tp = len(pairs)
    fp = len(predictions) - tp
    fn = len(truths) - tp
    return {
        "matching_rule": MATCHING_RULE,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "ground_truth_occurrences": len(truths),
        "predicted_occurrences": len(predictions),
        "precision": tp / len(predictions) if predictions else None,
        "recall": tp / len(truths) if truths else None,
        "matched_pairs": [
            {"prediction_index": prediction_index, "ground_truth_index": truth_index}
            for prediction_index, truth_index in pairs
        ],
    }


def first_correct_onsets(
    case_id: str,
    truths: Sequence[tuple[float, float]],
    predictions: Sequence[float],
    *,
    timely_s: float,
) -> list[dict[str, Any]]:
    """Retain the earliest correct fire per occurrence; later duplicates never skew latency."""
    result: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(truths):
        hits = [t for t in predictions if _spans_match(t, (start, end))]
        first = min(hits) if hits else None
        latency = None if first is None else round(first - start, 6)
        result.append(
            {
                "case_id": case_id,
                "ground_truth_index": index,
                "ground_truth_interval": [start, end],
                "first_correct_fire_ts": first,
                "onset_to_first_correct_fire_s": latency,
                "timely_within_s": timely_s,
                "timely": latency is not None and latency <= timely_s,
            }
        )
    return result


def replay_case(case: dict[str, Any], policies: Sequence[ReplayPolicy], *, timely_s: float) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "<unnamed-case>")
    media_end = case.get("media_end_s")
    if not _finite_number(media_end) or float(media_end) <= 0:
        raise ReplayInputError(f"{case_id}: media_end_s must be finite and positive")
    media_end = float(media_end)
    rows = _validated_rows(case.get("verdict_rows"), case_id=case_id)
    truths = _validated_intervals(
        case.get("gt_event_intervals_raw"), case_id=case_id, media_end_s=media_end
    )

    by_policy: dict[str, Any] = {}
    for policy in policies:
        fires = simulate_fires(rows, policy)
        onsets = first_correct_onsets(case_id, truths, fires, timely_s=timely_s)
        by_policy[policy.name] = {
            "parameters": policy.to_dict(),
            "fire_timestamps": fires,
            "score": strict_score(truths, fires),
            "first_correct_onsets": onsets,
            "timely_true_positives": sum(1 for row in onsets if row["timely"]),
            "timely_recall": (
                sum(1 for row in onsets if row["timely"]) / len(onsets) if onsets else None
            ),
        }
    return {
        "case_id": case_id,
        "media_end_s": media_end,
        "verdict_rows": len(rows),
        "ground_truth_intervals": [[a, b] for a, b in truths],
        "recorded_fire_timestamps": [
            float(row["media_ts"]) for row in rows if row.get("action") == "fire"
        ],
        "policies": by_policy,
    }


def build_report(
    document: object,
    *,
    source: str,
    policies: Sequence[ReplayPolicy],
    selected_case_id: str | None = None,
    timely_s: float = 2.0,
) -> dict[str, Any]:
    if not _finite_number(timely_s) or float(timely_s) < 0:
        raise ReplayInputError("timely_s must be finite and non-negative")
    cases = [
        replay_case(case, policies, timely_s=float(timely_s))
        for case in extract_cases(document, selected_case_id=selected_case_id)
    ]

    aggregate: dict[str, Any] = {}
    for policy in policies:
        scores = [case["policies"][policy.name]["score"] for case in cases]
        onsets = [
            onset
            for case in cases
            for onset in case["policies"][policy.name]["first_correct_onsets"]
        ]
        tp = sum(score["true_positives"] for score in scores)
        fp = sum(score["false_positives"] for score in scores)
        fn = sum(score["false_negatives"] for score in scores)
        predicted = tp + fp
        gt = tp + fn
        timely = sum(1 for row in onsets if row["timely"])
        latencies = [
            row["onset_to_first_correct_fire_s"]
            for row in onsets
            if row["onset_to_first_correct_fire_s"] is not None
        ]
        aggregate[policy.name] = {
            "parameters": policy.to_dict(),
            "score": {
                "matching_rule": MATCHING_RULE,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "ground_truth_occurrences": gt,
                "predicted_occurrences": predicted,
                "precision": tp / predicted if predicted else None,
                "recall": tp / gt if gt else None,
            },
            "first_correct_onsets": onsets,
            "first_correct_onset_latency_s": {
                "n": len(latencies),
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
            "timely_within_s": float(timely_s),
            "timely_true_positives": timely,
            "timely_recall": timely / gt if gt else None,
        }

    return {
        "schema": REPORT_SCHEMA,
        "mode": "offline_saved_verdict_replay_no_provider",
        "source": source,
        "limitations": [
            "Counterfactuals hold the saved VLM verdict stream fixed; they measure latch policy only.",
            "No claim is made about unseen-video perception quality or provider nondeterminism.",
            "Strict scoring permits one TP per raw GT occurrence; duplicate fires are false positives.",
        ],
        "cases": cases,
        "aggregate": aggregate,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=pathlib.Path, help="saved JSON containing verdict_rows")
    parser.add_argument("--case-id", help="replay one case from a multi-case artifact")
    parser.add_argument(
        "--off-dwell-s",
        type=float,
        action="append",
        dest="off_dwells_s",
        help="sustained-false dwell variant (repeatable; defaults: 2 and 5 seconds)",
    )
    parser.add_argument("--timely-s", type=float, default=2.0, help="timely-recall threshold")
    parser.add_argument("--output", type=pathlib.Path, help="optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = json.loads(args.artifact.read_text(encoding="utf-8"))
        report = build_report(
            document,
            source=str(args.artifact.resolve()),
            policies=default_policies(args.off_dwells_s or DEFAULT_OFF_DWELL_S),
            selected_case_id=args.case_id,
            timely_s=args.timely_s,
        )
    except (OSError, json.JSONDecodeError, ReplayInputError, ValueError) as exc:
        print(f"verdict-policy-replay: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
