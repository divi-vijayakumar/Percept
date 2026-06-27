from eval.bench.derived_interaction import DerivedCue, cue_metrics, time_metrics
from eval.bench import track_cuespeak, track_timespeak
from eval.bench import derived_interaction


def test_derived_cue_metrics_score_precision_recall_and_duplicates():
    cues = [
        DerivedCue("pt", "open door", (1.0, 3.0), (1.5, 2.0)),
        DerivedCue("pt", "close door", (5.0, 7.0), (9.0,)),
        DerivedCue("pt", "wave", (10.0, 12.0), ()),
    ]

    metrics = cue_metrics(cues)

    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1 / 3
    assert metrics["fires_per_episode"] == 1.0
    assert metrics["duplicate_rate"] == 1 / 3


def test_derived_time_metrics_score_strict_window():
    cues = [
        DerivedCue("pt", "on time", (10.0, 15.0), (10.5,)),
        DerivedCue("pt", "late", (20.0, 25.0), (24.0,)),
        DerivedCue("pt", "early", (30.0, 35.0), (29.0,)),
    ]

    metrics = time_metrics(cues, omega_s=2.0)

    assert metrics["timing_accuracy"] == 1 / 3
    assert metrics["p_pauc"] == 25.0
    assert metrics["early_fire_rate"] == 1 / 3


def test_cuespeak_prefers_e2e_artifacts(monkeypatch):
    monkeypatch.setattr(
        track_cuespeak,
        "cues_from_latest_e2e",
        lambda: ([DerivedCue("pt", "open door", (1.0, 3.0), (1.5,))], ["run/results/pt.json"], 4),
    )

    result = track_cuespeak.score(live=True, n=1)

    assert result.cells[("e2e", "precision")] == 1.0
    assert result.cells[("e2e", "recall")] == 1.0
    assert result.diagnostic is True


def test_timespeak_prefers_e2e_artifacts(monkeypatch):
    monkeypatch.setattr(
        track_timespeak,
        "cues_from_latest_e2e",
        lambda: ([DerivedCue("pt", "open door", (1.0, 3.0), (1.5,))], ["run/results/pt.json"], 4),
    )

    result = track_timespeak.score(live=True, n=1)

    assert result.cells[("e2e", "timing_accuracy")] == 1.0
    assert result.cells[("e2e", "p-pauc")] == 75.0
    assert result.diagnostic is True


def test_derived_reader_prefers_current_run_dir(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True)
    (result_dir / "pt-action-e2e.json").write_text(
        """
        {
          "track": "pt-action-e2e",
          "calls": 1,
          "cells": {},
          "coverage": [{"dataset": "pt", "expected": 1, "observed": 1}],
          "notes": "per_event=[{'label':'current','window':[1,2],'fires':[1.5]}]"
        }
        """
    )
    monkeypatch.setenv("PERCEPT_BENCH_RUN_DIR", str(run_dir))
    monkeypatch.setattr(derived_interaction, "_latest_result", derived_interaction._latest_result)

    cues, refs, calls = derived_interaction.cues_from_latest_e2e()

    assert any(c.label == "current" for c in cues)
    assert any("pt-action-e2e.json" in ref for ref in refs)
    assert calls >= 1
