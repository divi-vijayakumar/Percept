"""The v1.2 qualification corpus is an auditable narrowing, never a label-shaped repair."""

from __future__ import annotations

import json

import build_corpus_v1_2 as migration


def payload() -> dict:
    source = json.loads(migration.DEFAULT_SOURCE.read_text())
    adjudication = json.loads(migration.DEFAULT_ADJUDICATION.read_text())
    return migration.derive(source, adjudication, source_sha=migration.sha256(migration.DEFAULT_SOURCE))


def test_exact_adjudicated_population_and_reasons():
    out = payload()
    audit = out["mechanic_adjudication"]
    assert out["corpus_version"] == "1.2"
    assert out["n_cases"] == 96
    assert len(audit["corrected"]) == 4
    assert len(audit["excluded"]) == 4
    assert {row["reason_code"] for row in audit["excluded"]} == {
        "human_boundary_review_required",
        "unsupported_audio_triggered_snapshot_count",
        "unsupported_sequential_instruction",
        "unsupported_open_ended_narration",
    }
    assert out["qualification_scope"]["statistical_launch_eligibility"] is False


def test_media_pins_and_positive_intervals_are_never_changed():
    source = json.loads(migration.DEFAULT_SOURCE.read_text())
    before = {row["case_id"]: row for row in source["cases"]}
    out = payload()
    for case in out["cases"]:
        prior = before[case["case_id"]]
        assert case["video_path"] == prior["video_path"]
        assert case["video_sha256"] == prior["video_sha256"]
        assert case["positives"] == prior["positives"]
        assert case["duration_s"] == prior["duration_s"]


def test_only_the_four_adjudicated_goal_kinds_change():
    source = json.loads(migration.DEFAULT_SOURCE.read_text())
    before = {row["case_id"]: row["goal_kind"] for row in source["cases"]}
    out = payload()
    changed = {
        row["case_id"]: (before[row["case_id"]], row["goal_kind"])
        for row in out["cases"]
        if before[row["case_id"]] != row["goal_kind"]
    }
    assert changed == {row["case_id"]: ("transition", "duration")
                       for row in out["mechanic_adjudication"]["corrected"]}


def test_checked_in_artifact_is_byte_identical():
    assert migration.DEFAULT_OUT.read_text() == migration.render(
        migration.DEFAULT_SOURCE, migration.DEFAULT_ADJUDICATION
    )
