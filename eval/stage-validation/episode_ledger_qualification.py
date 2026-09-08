"""Fail-closed scorer for real-world episode creation and context-ledger reliability.

This lane is deliberately separate from notification precision/recall.  It consumes a normalized,
fully-labelled source-clock artifact produced by a real ``percept.run`` soak plus its durable archive.
It measures whether Percept created the right episode boundaries and whether the revision/action
ledgers remain structurally trustworthy.  It never treats an in-memory probe episode as runtime
evidence, and it refuses headline precision when prediction retention or annotation coverage is
incomplete.

The input contract is JSON and intentionally small.  Required top-level fields are:

``provenance``
    ``build_commit``, ``run_id``, ``clock_domain='source_seconds'``,
    ``archive_complete=true`` and ``archive_drops=0``.
``watch_windows``
    Fully-labelled ``source_id``/``goal_id`` intervals with non-empty site, camera and day cluster
    ids.  Every truth and prediction must land inside one such window.
``ground_truth_episodes`` / ``predicted_episodes``
    Normalized episode rows.  Matching is exact on source+goal and maximum-cardinality one-to-one on
    temporal overlap; subject identity is scored *after* matching so an identity error cannot hide as
    a different boundary error.
``incidents``
    Receiver-observed logical incidents.  Revision ids must be unique and each ``supersedes`` link
    must point backward to a previously observed incident id.
``ledger``
    Runtime counters in their native units.  Admission and judge-terminal conservation are checked
    exactly; no residual is rounded away.

The scorer does not choose launch bars.  It emits eligibility plus auditable counts and distributions
for the qualification report to compare with an approved contract.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

KIND = "percept-episode-ledger-qualification"
VERSION = 1
TERMINAL_REVISION_KINDS = frozenset({"closed", "expired", "retracted"})
REVISION_KINDS = frozenset(
    {"opened", "updated", "confirmed", "retracted", "closed", "expired", "unobserved_gap"}
)


class QualificationInputError(ValueError):
    """The artifact is malformed or cannot support a qualification claim."""


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise QualificationInputError(f"{name} must be a non-negative integer")
    return value


def _text(row: Mapping[str, Any], name: str, context: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise QualificationInputError(f"{context}.{name} must be a non-empty string")
    return value


def _time(row: Mapping[str, Any], name: str, context: str, *, optional: bool = False) -> float | None:
    value = row.get(name)
    if optional and value is None:
        return None
    if not _number(value):
        raise QualificationInputError(f"{context}.{name} must be a finite number")
    return float(value)


def _rows(document: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    value = document.get(name)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise QualificationInputError(f"{name} must be a list of objects")
    return value


def _contains(window: Mapping[str, Any], start: float, end: float) -> bool:
    return float(window["start_s"]) <= start and end <= float(window["end_s"])


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _union_duration(spans: Sequence[tuple[float, float]]) -> float:
    ordered = sorted(spans)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}

    def nearest_rank(q: float) -> float:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    return {
        "n": len(ordered),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _maximum_matching(adjacency: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Deterministic maximum-cardinality bipartite matching (prediction, truth)."""
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

    for prediction_index in range(len(adjacency)):
        augment(prediction_index, set())
    return sorted(
        ((prediction_index, truth_index) for truth_index, prediction_index in prediction_for_truth.items()),
        key=lambda pair: (pair[0], pair[1]),
    )


def _normalize_windows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        context = f"watch_windows[{index}]"
        start = _time(row, "start_s", context)
        end = _time(row, "end_s", context)
        assert start is not None and end is not None
        if end <= start:
            raise QualificationInputError(f"{context} must have end_s > start_s")
        if row.get("annotation_complete") is not True:
            raise QualificationInputError(f"{context}.annotation_complete must be true")
        out.append(
            {
                "source_id": _text(row, "source_id", context),
                "goal_id": _text(row, "goal_id", context),
                "site_id": _text(row, "site_id", context),
                "camera_id": _text(row, "camera_id", context),
                "day_id": _text(row, "day_id", context),
                "start_s": start,
                "end_s": end,
            }
        )
    if not out:
        raise QualificationInputError("watch_windows must not be empty")
    return out


def _normalize_episodes(
    rows: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    *,
    predicted: bool,
) -> list[dict[str, Any]]:
    label = "predicted_episodes" if predicted else "ground_truth_episodes"
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        context = f"{label}[{index}]"
        episode_id = _text(row, "episode_id", context)
        if episode_id in seen:
            raise QualificationInputError(f"duplicate {label} episode_id {episode_id!r}")
        seen.add(episode_id)
        source_id = _text(row, "source_id", context)
        goal_id = _text(row, "goal_id", context)
        start = _time(row, "start_s", context)
        end = _time(row, "end_s", context)
        assert start is not None and end is not None
        if end <= start:
            raise QualificationInputError(f"{context} must have end_s > start_s")
        candidates = [
            window
            for window in windows
            if window["source_id"] == source_id
            and window["goal_id"] == goal_id
            and _contains(window, start, end)
        ]
        if len(candidates) != 1:
            raise QualificationInputError(
                f"{context} must be contained by exactly one fully-labelled watch window; got {len(candidates)}"
            )
        normalized = {
            "episode_id": episode_id,
            "source_id": source_id,
            "goal_id": goal_id,
            "start_s": start,
            "end_s": end,
            "subject_id": row.get("subject_id") if not predicted else row.get("subject_key"),
            "cluster": {
                "site_id": candidates[0]["site_id"],
                "camera_id": candidates[0]["camera_id"],
                "day_id": candidates[0]["day_id"],
            },
        }
        if normalized["subject_id"] is not None and not isinstance(normalized["subject_id"], str):
            raise QualificationInputError(f"{context} subject identity must be a string or null")
        if predicted:
            normalized["status"] = _text(row, "status", context)
            terminal_at = _time(row, "t_end", context, optional=True)
            if terminal_at is not None and not math.isclose(terminal_at, end, abs_tol=1e-9):
                raise QualificationInputError(f"{context}.t_end must equal normalized end_s")
            normalized["terminal"] = terminal_at is not None
            revisions = row.get("revisions")
            if not isinstance(revisions, list) or not all(isinstance(revision, dict) for revision in revisions):
                raise QualificationInputError(f"{context}.revisions must be a list of objects")
            normalized["revisions"] = revisions
            normalized["observations"] = _count(row.get("observations"), f"{context}.observations")
            normalized["judged_observations"] = _count(
                row.get("judged_observations"), f"{context}.judged_observations"
            )
        out.append(normalized)
    return out


def _revision_defects(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    for episode in predictions:
        eid = str(episode["episode_id"])
        end = float(episode["end_s"])
        if episode["judged_observations"] > episode["observations"]:
            defects.append({"episode_id": eid, "kind": "judged_observations_exceed_observations"})
        revisions = episode["revisions"]
        if revisions and revisions[0].get("kind") != "opened":
            defects.append({"episode_id": eid, "kind": "revision_chain_does_not_start_opened"})
        prior_t: float | None = None
        for index, revision in enumerate(revisions):
            kind = revision.get("kind")
            t = revision.get("t")
            if kind not in REVISION_KINDS:
                defects.append({"episode_id": eid, "kind": "invalid_revision_kind", "index": index})
            if kind == "confirmed":
                defects.append({"episode_id": eid, "kind": "confirmation_appended_as_revision", "index": index})
            if not _number(t):
                defects.append({"episode_id": eid, "kind": "invalid_revision_time", "index": index})
                continue
            t = float(t)
            if t < float(episode["start_s"]) or t > end:
                defects.append({"episode_id": eid, "kind": "revision_outside_episode", "index": index})
            if prior_t is not None and t < prior_t:
                defects.append({"episode_id": eid, "kind": "revision_time_reversal", "index": index})
            prior_t = t
        terminal = bool(episode["terminal"])
        if terminal and not revisions:
            defects.append({"episode_id": eid, "kind": "terminal_episode_without_revisions"})
        if terminal and revisions and revisions[-1].get("kind") not in TERMINAL_REVISION_KINDS:
            defects.append({"episode_id": eid, "kind": "terminal_status_without_terminal_revision"})
        if not terminal and revisions and revisions[-1].get("kind") in TERMINAL_REVISION_KINDS:
            defects.append({"episode_id": eid, "kind": "open_status_after_terminal_revision"})
    return defects


def _incident_defects(
    rows: Sequence[Mapping[str, Any]], predicted_episode_ids: set[str]
) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    incident_ids: set[str] = set()
    revision_ids: set[str] = set()
    next_revision_by_episode: dict[str, int] = {}
    prior_ts = -math.inf
    for index, row in enumerate(rows):
        context = f"incidents[{index}]"
        incident_id = _text(row, "incident_id", context)
        revision_id = _text(row, "revision_id", context)
        received = _time(row, "receiver_ack_monotonic_s", context)
        assert received is not None
        episode_id, separator, revision_number_text = revision_id.rpartition("#r")
        if not separator or not revision_number_text.isdigit() or int(revision_number_text) < 1:
            defects.append({"incident_id": incident_id, "kind": "invalid_revision_id"})
        else:
            revision_number = int(revision_number_text)
            if episode_id not in predicted_episode_ids:
                defects.append({"incident_id": incident_id, "kind": "orphan_episode_revision"})
            expected = next_revision_by_episode.get(episode_id, 1)
            if revision_number != expected:
                defects.append(
                    {
                        "incident_id": incident_id,
                        "kind": "revision_sequence_gap_or_reversal",
                        "expected": expected,
                        "observed": revision_number,
                    }
                )
            next_revision_by_episode[episode_id] = max(expected, revision_number + 1)
        if incident_id in incident_ids:
            defects.append({"incident_id": incident_id, "kind": "duplicate_incident_id"})
        if revision_id in revision_ids:
            defects.append({"incident_id": incident_id, "kind": "duplicate_revision_id"})
        supersedes = row.get("supersedes")
        if supersedes is not None and supersedes not in incident_ids:
            defects.append({"incident_id": incident_id, "kind": "dangling_or_forward_supersedes"})
        if received < prior_ts:
            defects.append({"incident_id": incident_id, "kind": "receiver_order_reversal"})
        prior_ts = received
        incident_ids.add(incident_id)
        revision_ids.add(revision_id)
    return defects


def _ledger_result(ledger: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "admissions",
        "launched_batches",
        "scheduler_receipted",
        "judge_started",
        "judge_finished",
        "judge_degraded",
        "gate_classified",
    )
    values = {name: _count(ledger.get(name), f"ledger.{name}") for name in names}
    admission_residual = values["admissions"] - values["launched_batches"] - values["scheduler_receipted"]
    judge_terminal_residual = values["judge_started"] - values["judge_finished"] - values["judge_degraded"]
    gate_residual = values["judge_finished"] - values["gate_classified"]
    return {
        **values,
        "admission_residual": admission_residual,
        "judge_terminal_residual": judge_terminal_residual,
        "gate_residual": gate_residual,
        "balances": admission_residual == judge_terminal_residual == gate_residual == 0,
    }


def qualify(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and score one normalized real-world episode/ledger artifact."""
    if not isinstance(document, Mapping):
        raise QualificationInputError("artifact root must be an object")
    if document.get("kind") != KIND or document.get("version") != VERSION:
        raise QualificationInputError(f"artifact must declare kind={KIND!r}, version={VERSION}")
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise QualificationInputError("provenance must be an object")
    for name in ("build_commit", "run_id"):
        _text(provenance, name, "provenance")
    if provenance.get("clock_domain") != "source_seconds":
        raise QualificationInputError("provenance.clock_domain must be 'source_seconds'")
    if provenance.get("archive_complete") is not True:
        raise QualificationInputError("provenance.archive_complete must be true")
    if _count(provenance.get("archive_drops"), "provenance.archive_drops") != 0:
        raise QualificationInputError("provenance.archive_drops must be zero")

    windows = _normalize_windows(_rows(document, "watch_windows"))
    truths = _normalize_episodes(
        _rows(document, "ground_truth_episodes"), windows, predicted=False
    )
    predictions = _normalize_episodes(
        _rows(document, "predicted_episodes"), windows, predicted=True
    )

    adjacency: list[list[int]] = []
    for prediction in predictions:
        candidates = [
            truth_index
            for truth_index, truth in enumerate(truths)
            if prediction["source_id"] == truth["source_id"]
            and prediction["goal_id"] == truth["goal_id"]
            and _overlap(
                (float(prediction["start_s"]), float(prediction["end_s"])),
                (float(truth["start_s"]), float(truth["end_s"])),
            )
            > 0
        ]
        # Prefer the greatest overlap, then stable truth index, without sacrificing cardinality.
        candidates.sort(
            key=lambda truth_index: (
                -_overlap(
                    (float(prediction["start_s"]), float(prediction["end_s"])),
                    (float(truths[truth_index]["start_s"]), float(truths[truth_index]["end_s"])),
                ),
                truth_index,
            )
        )
        adjacency.append(candidates)
    pairs = _maximum_matching(adjacency)

    starts: list[float] = []
    ends: list[float] = []
    subject_matches = subject_switches = subject_unknown = 0
    matched_rows: list[dict[str, Any]] = []
    for prediction_index, truth_index in pairs:
        prediction, truth = predictions[prediction_index], truths[truth_index]
        start_error = float(prediction["start_s"]) - float(truth["start_s"])
        end_error = float(prediction["end_s"]) - float(truth["end_s"])
        starts.append(abs(start_error))
        ends.append(abs(end_error))
        predicted_subject, truth_subject = prediction["subject_id"], truth["subject_id"]
        if predicted_subject is None or truth_subject is None:
            subject_unknown += 1
            disposition = "unknown"
        elif predicted_subject == truth_subject:
            subject_matches += 1
            disposition = "match"
        else:
            subject_switches += 1
            disposition = "switch"
        matched_rows.append(
            {
                "prediction_episode_id": prediction["episode_id"],
                "ground_truth_episode_id": truth["episode_id"],
                "start_error_s": start_error,
                "end_error_s": end_error,
                "subject_disposition": disposition,
                "cluster": truth["cluster"],
            }
        )

    true_positives = len(pairs)
    false_positives = len(predictions) - true_positives
    false_negatives = len(truths) - true_positives
    revision_defects = _revision_defects(predictions)
    incident_defects = _incident_defects(
        _rows(document, "incidents"),
        {str(prediction["episode_id"]) for prediction in predictions},
    )
    ledger = document.get("ledger")
    if not isinstance(ledger, Mapping):
        raise QualificationInputError("ledger must be an object")
    ledger_result = _ledger_result(ledger)

    split_truths = sum(
        1 for truth_index in range(len(truths))
        if sum(truth_index in candidates for candidates in adjacency) > 1
    )
    merged_predictions = sum(1 for candidates in adjacency if len(candidates) > 1)
    watch_hours = sum(float(row["end_s"]) - float(row["start_s"]) for row in windows) / 3600.0
    source_spans: dict[str, list[tuple[float, float]]] = {}
    for row in windows:
        source_spans.setdefault(str(row["source_id"]), []).append(
            (float(row["start_s"]), float(row["end_s"]))
        )
    source_hours = sum(_union_duration(spans) for spans in source_spans.values()) / 3600.0
    clusters = {
        (row["site_id"], row["camera_id"], row["day_id"])
        for row in windows
    }
    reliable = not revision_defects and not incident_defects and ledger_result["balances"]
    return {
        "kind": "percept-episode-ledger-qualification-result",
        "version": 1,
        "eligible": True,
        "matching_rule": "maximum-cardinality one-to-one temporal overlap, exact source+goal",
        "exposure": {
            "fully_labelled_scheduled_watch_hours": watch_hours,
            "fully_labelled_source_hours": source_hours,
            "watch_windows": len(windows),
            "independent_site_camera_day_clusters": len(clusters),
        },
        "episode_creation": {
            "ground_truth_episodes": len(truths),
            "predicted_episodes": len(predictions),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": true_positives / len(predictions) if predictions else None,
            "recall": true_positives / len(truths) if truths else None,
            "false_episodes_per_fully_labelled_source_hour": (
                false_positives / source_hours if source_hours else None
            ),
            "false_episodes_per_fully_labelled_scheduled_watch_hour": (
                false_positives / watch_hours if watch_hours else None
            ),
            "potential_split_truths": split_truths,
            "potential_merged_predictions": merged_predictions,
            "start_boundary_absolute_error_s": _quantiles(starts),
            "end_boundary_absolute_error_s": _quantiles(ends),
            "matched_pairs": matched_rows,
        },
        "identity": {
            "matched_subject_ids": subject_matches,
            "subject_id_switches": subject_switches,
            "subject_id_unknown": subject_unknown,
            "switch_rate_among_known": (
                subject_switches / (subject_matches + subject_switches)
                if subject_matches + subject_switches
                else None
            ),
        },
        "context_ledger": {
            "reliable": reliable,
            "revision_defects": revision_defects,
            "incident_defects": incident_defects,
            "runtime_conservation": ledger_result,
        },
        "provenance": dict(provenance),
    }


def load_and_qualify(path: str | Path) -> dict[str, Any]:
    return qualify(json.loads(Path(path).read_text()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="normalized real-world JSON artifact")
    parser.add_argument("--out", type=Path, help="write result JSON here (stdout when omitted)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = load_and_qualify(args.artifact)
    except (OSError, json.JSONDecodeError, QualificationInputError) as exc:
        parser.exit(2, f"episode-ledger qualification refused: {exc}\n")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.write_text(rendered)
    return 0


__all__ = ["KIND", "VERSION", "QualificationInputError", "qualify", "load_and_qualify"]


if __name__ == "__main__":
    raise SystemExit(main())
