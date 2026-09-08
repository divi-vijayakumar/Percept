"""The supported v1.2 frozen plan is complete, reproducible, and explicitly human-adjudicated."""

from __future__ import annotations

import json

import migrate_v1_2 as migration
from percept.conductor import AuthoredPrimitive, authoring_key, goal_kind_of


def payload() -> dict:
    return migration.migrate_payload(
        json.loads(migration.DEFAULT_SOURCE.read_text()),
        json.loads(migration.DEFAULT_CORPUS.read_text()),
        json.loads(migration.DEFAULT_ADJUDICATION.read_text()),
    )


def test_supported_plan_is_complete_and_mechanics_valid_without_provider_calls():
    out = payload()
    corpus = json.loads(migration.DEFAULT_CORPUS.read_text())
    assert out["migration"]["provider_calls"] == 0
    assert out["scorecard"]["hit"] == out["scorecard"]["n"] == 96
    assert out["scorecard"]["mismatches"] == []
    assert len(out["entries"]) == len({case["goal_text"] for case in corpus["cases"]})
    for case in corpus["cases"]:
        record = out["entries"][authoring_key(case["goal_text"], model=out["model"])]
        assert goal_kind_of(AuthoredPrimitive.from_record(record)) == case["goal_kind"]


def test_only_one_conductor_record_is_human_overridden():
    out = payload()
    overridden = [row for row in out["entries"].values()
                  if row.get("origin") == "human_adjudication"]
    assert len(overridden) == 1
    assert overridden[0]["goal_text"].startswith("Alert me whenever you see a group")
    assert overridden[0]["behavior_type"] == "transition"
    assert out["migration"]["overrides"][0]["reason_code"] == (
        "whenever_visible_is_rising_transition"
    )


def test_checked_in_artifact_is_byte_identical():
    assert migration.DEFAULT_OUT.read_text() == migration.render(
        migration.DEFAULT_SOURCE, migration.DEFAULT_CORPUS, migration.DEFAULT_ADJUDICATION
    )
