"""Focused no-provider tests for ``verdict_policy_replay.py``.

Run: ``python -m pytest eval/stage-validation/test_verdict_policy_replay.py -q``
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verdict_policy_replay as VPR  # noqa: E402


def verdict(t: float, truth: str, *, action: str = "silent") -> dict:
    return {"media_ts": t, "truth": truth, "action": action}


def case(rows: list[dict], gt: list[list[float]] | None = None) -> dict:
    return {
        "case_id": "fixture",
        "media_end_s": 20.0,
        "gt_event_intervals_raw": gt if gt is not None else [[5.0, 10.0]],
        "verdict_rows": rows,
    }


def policy(name: str, *, false_first: bool = True, k: int = 1) -> VPR.ReplayPolicy:
    return VPR.ReplayPolicy(name, require_false_first=false_first, false_count=k)


def test_current_fires_at_first_true_while_false_first_requires_a_seen_false():
    rows = [verdict(1, "yes"), verdict(2, "yes"), verdict(3, "no"), verdict(4, "yes")]
    assert VPR.simulate_fires(rows, policy("current", false_first=False)) == [1.0, 4.0]
    assert VPR.simulate_fires(rows, policy("false-first")) == [4.0]


def test_k2_and_k3_require_consecutive_confident_false_exit_evidence():
    rows = [
        verdict(0, "no"),
        verdict(1, "yes"),       # first fire
        verdict(2, "no"),
        verdict(3, "yes"),       # resets the K-run: no re-fire
        verdict(4, "no"),
        verdict(5, "no"),
        verdict(6, "yes"),       # K=2 re-fires, K=3 does not
        verdict(7, "no"),
        verdict(8, "no"),
        verdict(9, "no"),
        verdict(10, "yes"),      # both have now proven an exit
    ]
    assert VPR.simulate_fires(rows, policy("k2", k=2)) == [1.0, 6.0, 10.0]
    assert VPR.simulate_fires(rows, policy("k3", k=3)) == [1.0, 10.0]


def test_unknown_never_closes_and_breaks_pending_count_or_dwell():
    rows = [
        verdict(0, "no"),
        verdict(1, "yes"),
        verdict(2, "no"),
        verdict(4, "unknown"),
        verdict(6, "no"),
        verdict(7, "yes"),
        verdict(8, "no"),
        verdict(11, "no"),
        verdict(12, "yes"),
    ]
    k2 = policy("k2", k=2)
    dwell = VPR.ReplayPolicy(
        "dwell", require_false_first=True, false_count=None, off_dwell_s=2.0
    )
    assert VPR.simulate_fires(rows, k2) == [1.0, 12.0]
    assert VPR.simulate_fires(rows, dwell) == [1.0, 12.0]


def test_strict_score_counts_duplicate_inside_fires_and_outside_fires_as_fp():
    score = VPR.strict_score([(5.0, 10.0)], [1.0, 6.0, 7.0, 12.0])
    assert score["true_positives"] == 1
    assert score["false_positives"] == 3
    assert score["false_negatives"] == 0
    assert score["precision"] == pytest.approx(0.25)
    assert score["recall"] == pytest.approx(1.0)


def test_first_correct_onset_keeps_earliest_hit_not_later_duplicates():
    rows = VPR.first_correct_onsets("c", [(5.0, 10.0)], [6.5, 7.0, 8.0], timely_s=2.0)
    assert rows == [
        {
            "case_id": "c",
            "ground_truth_index": 0,
            "ground_truth_interval": [5.0, 10.0],
            "first_correct_fire_ts": 6.5,
            "onset_to_first_correct_fire_s": 1.5,
            "timely_within_s": 2.0,
            "timely": True,
        }
    ]


def test_unknown_only_stream_cannot_manufacture_a_baseline_or_incident():
    rows = [verdict(0, "unknown"), verdict(1, "yes")]
    assert VPR.simulate_fires(rows, policy("false-first")) == []


def test_full_artifact_shape_is_extracted_and_aggregated_without_core_imports():
    document = {"cases": {"reflex": [case([verdict(0, "no"), verdict(6, "yes")])]}}
    policies = (policy("false-first"),)
    report = VPR.build_report(document, source="memory", policies=policies, timely_s=2.0)
    assert report["mode"] == "offline_saved_verdict_replay_no_provider"
    assert report["cases"][0]["verdict_rows"] == 2
    assert report["aggregate"]["false-first"]["score"] == {
        "matching_rule": VPR.MATCHING_RULE,
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "ground_truth_occurrences": 1,
        "predicted_occurrences": 1,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert report["aggregate"]["false-first"]["first_correct_onset_latency_s"] == {
        "n": 1,
        "min": 1.0,
        "max": 1.0,
    }


def test_mozzarella_saved_rows_reproduce_recorded_false_first_fires_and_score():
    artifact = HERE / "results" / "single-clip-corrected-3p1-flash-lite-20260802.json"
    if not artifact.exists():
        pytest.skip("local paid artifact is not present")
    document = json.loads(artifact.read_text(encoding="utf-8"))
    selected = VPR.extract_cases(document)[0]
    replay = VPR.replay_case(selected, VPR.default_policies(), timely_s=2.0)

    simulated = replay["policies"]["false_first"]["fire_timestamps"]
    assert simulated == replay["recorded_fire_timestamps"]
    score = replay["policies"]["false_first"]["score"]
    assert (score["true_positives"], score["false_positives"], score["false_negatives"]) == (
        1,
        9,
        0,
    )
    onset = replay["policies"]["false_first"]["first_correct_onsets"][0]
    assert onset["first_correct_fire_ts"] == 18.0
    assert onset["onset_to_first_correct_fire_s"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    "bad_rows",
    [
        [verdict(2, "no"), verdict(1, "yes")],
        [verdict(1, "maybe")],
        [{"media_ts": None, "truth": "no"}],
    ],
)
def test_bad_streams_fail_loudly(bad_rows):
    with pytest.raises(VPR.ReplayInputError):
        VPR.replay_case(case(bad_rows), (policy("p"),), timely_s=2.0)
