"""Long-horizon manifest compatibility and fail-closed preflight."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import long_run as LRUN  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parents[2]
INJECTED = (
    ROOT
    / "Product-Execution"
    / "4-probes"
    / "longhorizon-soak"
    / "sources"
    / "inject_2h_v1.manifest.json"
)


def test_relocated_injected_fixture_is_consumable_by_the_long_runner():
    manifest = LRUN.normalize_long_manifest(json.loads(INJECTED.read_text()))

    assert manifest["source_manifest_kind"] == "longhorizon-injected-source"
    assert manifest["clock"]["measured_container_duration_s"] == 7197.932
    assert manifest["n_goals"] == 19
    assert sum(g["n_occurrences"] for g in manifest["goals"]) == 128
    assert {g["goal_kind"] for g in manifest["goals"]} == {"duration", "count"}
    assert LRUN.long_manifest_issues(manifest) == []


def test_unlabelled_background_is_explicitly_ineligible_for_precision():
    manifest = LRUN.normalize_long_manifest(json.loads(INJECTED.read_text()))
    assert manifest["precision_eligible"] is False
    assert "unlabelled" in manifest["precision_ineligible_reason"]


def test_out_of_bounds_occurrence_fails_preflight():
    manifest = LRUN.normalize_long_manifest(json.loads(INJECTED.read_text()))
    manifest["goals"][0]["positives"] = [[0.0, 8000.0]]
    issues = LRUN.long_manifest_issues(manifest)
    assert any("outside [0,7197.932]" in issue for issue in issues)


def test_retained_l2_manifest_remains_compatible():
    path = ROOT / "eval" / "stage-validation" / "long_sources" / "long_run_all.manifest.json"
    manifest = LRUN.normalize_long_manifest(json.loads(path.read_text()))
    assert manifest["source_manifest_kind"] == "stage-validation-single-long-source"
    assert LRUN.long_manifest_issues(manifest) == []


def test_channel_tracks_exact_last_yielded_media_for_partial_cost_denominator():
    ch = LRUN.Channels(anchor_media=2.0, last_media=62.0, frames_yielded=901)
    observed_s = min(120.0, max(0.0, ch.last_media - ch.anchor_media) + 1.0 / 15.0)
    assert observed_s == 60.0 + 1.0 / 15.0


def test_long_quality_is_occurrence_deduplicated_and_keeps_misses_in_timely_recall():
    armed = [
        {
            "goal_id": "g1",
            "goal_kind": "duration",
            "dataset": "fixture",
            "positives": [[10.0, 20.0], [30.0, 40.0]],
        }
    ]
    incidents = [
        {"id": "first", "goal_id": "g1", "ts": 12.0, "observed_since_anchor_s": 11.0},
        {"id": "repeat", "goal_id": "g1", "ts": 15.0, "observed_since_anchor_s": 17.0},
        {"id": "outside", "goal_id": "g1", "ts": 50.0, "observed_since_anchor_s": 51.0},
    ]

    result = LRUN.score_long_quality(
        armed,
        incidents,
        anchor_media=0.0,
        precision_eligible=True,
        precision_ineligible_reason=None,
    )
    pooled = result["pooled"]
    assert pooled["occurrences_matched"] == 1
    assert pooled["occurrences_missed"] == 1
    assert pooled["occurrence_recall"] == 0.5
    assert pooled["onset_to_incident_s"]["n"] == 1
    assert pooled["onset_to_incident_s"]["p50"] == 1.0
    assert pooled["timely_recall"]["2s"] == 0.5
    assert pooled["precision_per_emission"] == 2 / 3
    assert result["occurrences"][0]["emissions_in_occurrence"] == 2


def test_unlabelled_background_refuses_precision_but_still_scores_recall():
    result = LRUN.score_long_quality(
        [{"goal_id": "g", "goal_kind": "count", "dataset": "x", "positives": [[1.0, 2.0]]}],
        [{"id": "i", "goal_id": "g", "ts": 1.5, "observed_since_anchor_s": 1.7}],
        anchor_media=0.0,
        precision_eligible=False,
        precision_ineligible_reason="background unlabelled",
    )
    assert result["pooled"]["occurrence_recall"] == 1.0
    assert result["pooled"]["precision_per_emission"] is None
    assert result["pooled"]["precision_ineligible_reason"] == "background unlabelled"


def test_long_quality_separates_sdk_recall_from_webhook_receiver_recall():
    incidents = [
        {
            "id": "delivered",
            "goal_id": "g",
            "ts": 1.5,
            "observed_since_anchor_s": 1.7,
            "webhook": {"receiver_arrival_since_anchor_s": 1.9, "attempts": 1},
        },
        {
            "id": "lost",
            "goal_id": "g",
            "ts": 3.5,
            "observed_since_anchor_s": 3.7,
            "webhook": None,
        },
    ]
    result = LRUN.score_long_quality(
        [{
            "goal_id": "g",
            "goal_kind": "duration",
            "dataset": "x",
            "positives": [[1.0, 2.0], [3.0, 4.0]],
        }],
        incidents,
        anchor_media=0.0,
        precision_eligible=True,
        precision_ineligible_reason=None,
        delivery_enabled=True,
    )
    assert result["pooled"]["occurrence_recall"] == 1.0
    assert result["receiver"]["occurrence_recall"] == 0.5
    assert result["receiver"]["onset_to_webhook_arrival_s"]["p50"] == 0.9
    assert result["receiver"]["timely_recall"]["2s"] == 0.5
