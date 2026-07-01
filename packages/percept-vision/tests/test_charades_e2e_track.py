from pathlib import Path

from eval.bench.track_charades_e2e import (
    ActionGoal,
    VideoRun,
    _candidate_metrics,
    _interval_metrics,
    _sparse_miou,
)


def test_charades_interval_metrics_score_fires_inside_segments():
    run = VideoRun(
        video_id="v1",
        path=Path("v1.mp4"),
        length_s=10.0,
        goals=[
            ActionGoal("g1", "someone is holding a vacuum", "holding a vacuum", [(2.0, 4.0)]),
            ActionGoal("g2", "someone is tidying something", "tidying something", [(6.0, 8.0)]),
        ],
        fires=[("g1", 2.5, 0.9), ("g1", 9.0, 0.9), ("g2", 6.5, 0.8)],
    )

    metrics = _interval_metrics([run])

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 2 / 3


def test_charades_candidate_metrics_use_all_segments_and_escalations():
    run = VideoRun(
        video_id="v1",
        path=Path("v1.mp4"),
        length_s=10.0,
        goals=[
            ActionGoal("g1", "someone is opening a door", "opening a door", [(1.0, 2.0)]),
            ActionGoal("g2", "someone is running somewhere", "running somewhere", [(7.0, 8.0)]),
        ],
        escalations=[1.5, 5.0],
    )

    metrics = _candidate_metrics([run])

    assert metrics["candidate_recall"] == 0.5
    assert metrics["candidate_precision_proxy"] == 0.5
    assert metrics["candidate_frames"] == 2


def test_charades_sparse_miou_is_diagnostic_from_candidate_verdicts():
    run = VideoRun(
        video_id="v1",
        path=Path("v1.mp4"),
        length_s=10.0,
        goals=[ActionGoal("g1", "someone is holding a blanket", "holding a blanket", [(2.0, 6.0)])],
        verdicts={
            "someone is holding a blanket": [
                (1.0, False, 0.9),
                (2.0, True, 0.9),
                (4.0, True, 0.9),
                (6.0, False, 0.9),
            ]
        },
    )

    assert _sparse_miou([run]) == 1.0
