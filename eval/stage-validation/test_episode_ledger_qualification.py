"""Offline tests for the real-world episode/context-ledger qualification lane."""

from __future__ import annotations

import copy

import pytest

import episode_ledger_qualification as ELQ


def artifact():
    return {
        "kind": ELQ.KIND,
        "version": 1,
        "provenance": {
            "build_commit": "abc123",
            "run_id": "soak-1",
            "clock_domain": "source_seconds",
            "archive_complete": True,
            "archive_drops": 0,
        },
        "watch_windows": [
            {
                "source_id": "camera-1",
                "goal_id": "door",
                "site_id": "site-a",
                "camera_id": "camera-1",
                "day_id": "2026-08-01",
                "start_s": 0.0,
                "end_s": 100.0,
                "annotation_complete": True,
            }
        ],
        "ground_truth_episodes": [
            {
                "episode_id": "truth-1",
                "source_id": "camera-1",
                "goal_id": "door",
                "start_s": 10.0,
                "end_s": 20.0,
                "subject_id": "person-a",
            },
            {
                "episode_id": "truth-2",
                "source_id": "camera-1",
                "goal_id": "door",
                "start_s": 40.0,
                "end_s": 50.0,
                "subject_id": "person-b",
            },
        ],
        "predicted_episodes": [
            {
                "episode_id": "door:1",
                "source_id": "camera-1",
                "goal_id": "door",
                "start_s": 11.0,
                "end_s": 21.0,
                "t_end": 21.0,
                "status": "closed",
                "subject_key": "person-a",
                "observations": 4,
                "judged_observations": 3,
                "revisions": [
                    {"kind": "opened", "t": 11.0},
                    {"kind": "closed", "t": 21.0},
                ],
            },
            {
                "episode_id": "door:2",
                "source_id": "camera-1",
                "goal_id": "door",
                "start_s": 39.0,
                "end_s": 49.0,
                "t_end": 49.0,
                "status": "closed",
                "subject_key": "wrong-person",
                "observations": 2,
                "judged_observations": 2,
                "revisions": [
                    {"kind": "opened", "t": 39.0},
                    {"kind": "closed", "t": 49.0},
                ],
            },
        ],
        "incidents": [
            {
                "incident_id": "incident-1",
                "revision_id": "door:1#r1",
                "supersedes": None,
                "receiver_ack_monotonic_s": 1000.0,
            },
            {
                "incident_id": "incident-2",
                "revision_id": "door:1#r2",
                "supersedes": "incident-1",
                "receiver_ack_monotonic_s": 1001.0,
            },
        ],
        "ledger": {
            "admissions": 12,
            "launched_batches": 10,
            "scheduler_receipted": 2,
            "judge_started": 20,
            "judge_finished": 18,
            "judge_degraded": 2,
            "gate_classified": 18,
        },
    }


def test_scores_boundaries_identity_and_balanced_ledgers():
    result = ELQ.qualify(artifact())

    assert result["eligible"] is True
    assert result["episode_creation"]["precision"] == 1.0
    assert result["episode_creation"]["recall"] == 1.0
    assert result["exposure"]["fully_labelled_source_hours"] == 100.0 / 3600.0
    assert result["episode_creation"]["false_episodes_per_fully_labelled_source_hour"] == 0.0
    assert result["episode_creation"]["start_boundary_absolute_error_s"]["p95"] == 1.0
    assert result["identity"]["subject_id_switches"] == 1
    assert result["identity"]["switch_rate_among_known"] == 0.5
    assert result["context_ledger"]["reliable"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc["provenance"].update(archive_complete=False), "archive_complete"),
        (lambda doc: doc["provenance"].update(archive_drops=1), "archive_drops"),
        (lambda doc: doc["watch_windows"][0].update(annotation_complete=False), "annotation_complete"),
        (lambda doc: doc["provenance"].update(clock_domain="wall"), "clock_domain"),
    ],
)
def test_refuses_incomplete_or_cross_clock_evidence(mutate, message):
    doc = artifact()
    mutate(doc)
    with pytest.raises(ELQ.QualificationInputError, match=message):
        ELQ.qualify(doc)


def test_maximum_cardinality_matching_avoids_greedy_loss():
    doc = artifact()
    doc["ground_truth_episodes"] = [
        {**doc["ground_truth_episodes"][0], "episode_id": "wide", "start_s": 10.0, "end_s": 30.0},
        {**doc["ground_truth_episodes"][1], "episode_id": "narrow", "start_s": 20.0, "end_s": 22.0},
    ]
    doc["predicted_episodes"][0].update(start_s=19.0, end_s=21.0, t_end=21.0)
    doc["predicted_episodes"][0]["revisions"][-1]["t"] = 21.0
    doc["predicted_episodes"][1].update(start_s=11.0, end_s=12.0, t_end=12.0)
    doc["predicted_episodes"][1]["revisions"] = [
        {"kind": "opened", "t": 11.0}, {"kind": "closed", "t": 12.0}
    ]

    result = ELQ.qualify(doc)
    assert result["episode_creation"]["true_positives"] == 2


def test_reports_split_merge_revision_supersession_and_conservation_defects():
    doc = artifact()
    extra = copy.deepcopy(doc["predicted_episodes"][0])
    extra.update(episode_id="door:3", start_s=12.0, end_s=19.0, t_end=19.0)
    extra["revisions"] = [{"kind": "opened", "t": 12.0}, {"kind": "closed", "t": 19.0}]
    doc["predicted_episodes"].append(extra)
    doc["predicted_episodes"][0]["revisions"] = [
        {"kind": "closed", "t": 21.0},
        {"kind": "confirmed", "t": 11.0},
    ]
    doc["incidents"][1]["supersedes"] = "missing"
    doc["incidents"][1]["revision_id"] = "missing-episode#r3"
    doc["ledger"]["gate_classified"] = 17

    result = ELQ.qualify(doc)
    assert result["episode_creation"]["potential_split_truths"] == 1
    assert result["context_ledger"]["reliable"] is False
    assert result["context_ledger"]["revision_defects"]
    assert result["context_ledger"]["incident_defects"]
    assert result["context_ledger"]["runtime_conservation"]["gate_residual"] == 1


def test_prediction_outside_fully_labelled_window_is_rejected_not_silently_false_positive():
    doc = artifact()
    doc["predicted_episodes"][0].update(start_s=99.0, end_s=101.0, t_end=101.0)
    with pytest.raises(ELQ.QualificationInputError, match="exactly one fully-labelled"):
        ELQ.qualify(doc)
