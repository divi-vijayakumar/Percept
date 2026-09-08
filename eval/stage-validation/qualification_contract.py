"""Fail-closed launch-bar contract validation shared by paid qualification entry points."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

KIND = "percept-launch-qualification-contract"


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _rate(value: Any) -> bool:
    return _number(value) and 0.0 < float(value) <= 1.0


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def issues(document: dict[str, Any], *, require_approved: bool) -> list[str]:
    if not isinstance(document, dict):
        return ["contract root must be a JSON object"]
    out: list[str] = []
    if document.get("kind") != KIND:
        out.append(f"kind must be {KIND!r}")
    if document.get("version") != 1:
        out.append("version must be 1")
    if require_approved:
        if document.get("status") != "approved":
            out.append("status must be 'approved' for a qualification run")
        if not str(document.get("approved_by") or "").strip():
            out.append("approved_by is required for a qualification run")
        if not str(document.get("approved_at") or "").strip():
            out.append("approved_at is required for a qualification run")

    quality = _mapping(document.get("quality"))
    recall_target = quality.get("recall_target")
    if not _rate(recall_target) or float(recall_target) < 0.99:
        out.append("recall_target must be at least 0.99")
    recall_confidence = quality.get("recall_confidence")
    if not _rate(recall_confidence) or float(recall_confidence) < 0.95:
        out.append("recall_confidence must be at least 0.95")
    if quality.get("recall_unit") != "independent_ground_truth_occurrence":
        out.append("recall_unit must be independent_ground_truth_occurrence")
    if not _rate(quality.get("precision_target")):
        out.append("precision_target must be selected in (0,1]")
    precision_confidence = quality.get("precision_confidence")
    if not _rate(precision_confidence) or float(precision_confidence) < 0.95:
        out.append("precision_confidence must be at least 0.95")
    if quality.get("precision_unit") != "receiver_observed_logical_notification":
        out.append("precision_unit must be receiver_observed_logical_notification")
    far = quality.get("false_alerts_per_fully_labelled_negative_source_hour")
    if not _number(far) or float(far) < 0:
        out.append("false-alert rate per fully labelled negative source-hour must be selected")
    required_misses = {
        "no_receiver_arrival", "no_successful_receiver_ack", "budget_suppressed_judgment"
    }
    if not required_misses.issubset(set(quality.get("misses_include") or [])):
        out.append("receiver and budget-suppressed misses must remain in recall denominators")

    latency = _mapping(document.get("latency"))
    if latency.get("primary") != (
        "ground_truth_occurrence_onset_to_successful_webhook_receiver_ack"
    ):
        out.append("primary latency must end at successful webhook receiver ACK")
    deadline = latency.get("timely_deadline_s")
    if not _number(deadline) or float(deadline) <= 0:
        out.append("timely_deadline_s must be selected and positive")

    economy = _mapping(document.get("economy"))
    ceiling = economy.get("absolute_ceiling_usd")
    if not _number(ceiling) or not 0.0 < float(ceiling) <= 0.10:
        out.append("absolute cost ceiling must be positive and at most $0.10")
    if economy.get("denominator") != "source_stream_hour":
        out.append("cost denominator must be source_stream_hour")
    if economy.get("scope") != "all_in":
        out.append("cost scope must be all_in")

    stats = _mapping(document.get("statistics"))
    unadjusted_n = stats.get("minimum_zero_miss_independent_occurrences_per_unadjusted_cell")
    if type(unadjusted_n) is not int or unadjusted_n < 299:
        out.append("unadjusted zero-miss minimum must be at least 299")
    family_n = stats.get("minimum_zero_miss_independent_occurrences_per_12_cell_family")
    if type(family_n) is not int or family_n < 546:
        out.append("12-cell family zero-miss minimum must be at least 546 per cell")

    cells = _mapping(document.get("supported_primary_cells"))
    if not cells.get("mechanics") or not cells.get("source_classes"):
        out.append("supported mechanics and source classes must be frozen")
    return out


def load(path: str | Path, *, require_approved: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = Path(path).resolve()
    document = json.loads(resolved.read_text())
    defects = issues(document, require_approved=require_approved)
    if defects:
        raise ValueError("launch qualification contract is incomplete:\n  - " + "\n  - ".join(defects))
    summary = {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "status": document["status"],
        "approved_by": document.get("approved_by"),
        "approved_at": document.get("approved_at"),
        "recall_target": document["quality"]["recall_target"],
        "precision_target": document["quality"]["precision_target"],
        "timely_deadline_s": document["latency"]["timely_deadline_s"],
        "cost_ceiling_usd_per_source_hour": document["economy"]["absolute_ceiling_usd"],
    }
    return document, summary
