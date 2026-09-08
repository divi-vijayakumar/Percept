"""A paid run cannot call itself launch qualification until product bars are approved."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import qualification_contract as QC


def proposed() -> dict:
    return json.loads(Path(__file__).with_name("launch-gates.proposed.json").read_text())


def approved() -> dict:
    doc = copy.deepcopy(proposed())
    doc["status"] = "approved"
    doc["approved_by"] = "launch-owner"
    doc["approved_at"] = "2026-08-02T00:00:00Z"
    doc["quality"]["precision_target"] = 0.99
    doc["quality"]["false_alerts_per_fully_labelled_negative_source_hour"] = 0.01
    doc["latency"]["timely_deadline_s"] = 2.0
    return doc


def test_proposed_contract_refuses_qualification_for_every_open_decision():
    defects = QC.issues(proposed(), require_approved=True)
    assert "status must be 'approved' for a qualification run" in defects
    assert "precision_target must be selected in (0,1]" in defects
    assert "false-alert rate per fully labelled negative source-hour must be selected" in defects
    assert "timely_deadline_s must be selected and positive" in defects


def test_complete_approved_contract_passes():
    assert QC.issues(approved(), require_approved=True) == []


def test_cost_and_denominator_cannot_be_relaxed_silently():
    doc = approved()
    doc["economy"].update(absolute_ceiling_usd=0.11, denominator="watch_hour", scope="provider")
    defects = QC.issues(doc, require_approved=True)
    assert "absolute cost ceiling must be positive and at most $0.10" in defects
    assert "cost denominator must be source_stream_hour" in defects
    assert "cost scope must be all_in" in defects


def test_load_stamps_exact_approved_bytes(tmp_path):
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(approved()))
    _doc, summary = QC.load(path, require_approved=True)
    assert summary["status"] == "approved"
    assert len(summary["sha256"]) == 64


def test_load_fails_loud_on_proposed_file():
    path = Path(__file__).with_name("launch-gates.proposed.json")
    with pytest.raises(ValueError, match="contract is incomplete"):
        QC.load(path, require_approved=True)


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("quality",), "not-an-object", "recall_target must be at least 0.99"),
        (("quality", "recall_target"), "0.99", "recall_target must be at least 0.99"),
        (
            ("economy", "absolute_ceiling_usd"),
            -1.0,
            "absolute cost ceiling must be positive and at most $0.10",
        ),
        (
            ("statistics", "minimum_zero_miss_independent_occurrences_per_unadjusted_cell"),
            "299",
            "unadjusted zero-miss minimum must be at least 299",
        ),
        (("quality", "recall_target"), 10**1000, "recall_target must be at least 0.99"),
    ],
)
def test_malformed_or_type_coerced_values_fail_closed(path, value, expected):
    doc = approved()
    target = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert expected in QC.issues(doc, require_approved=True)


def test_non_object_root_fails_closed_without_attribute_error():
    assert QC.issues([], require_approved=True) == ["contract root must be a JSON object"]
