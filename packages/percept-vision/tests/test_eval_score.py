"""Scoring on hand-built fixtures: TP/FP/FN, stage attribution, funnel, cost, bootstrap."""

from __future__ import annotations

import pytest

from percept.plans import ClipRun, LabelError, Score, evaluate, load_labels
from percept.plans import score as sc
from percept.plans.labels import Occurrence
from percept.plans.score import Drop, Fire, Look, Prediction


def labels():
    return load_labels({
        "conditions": [{"label": "a", "text": "a begins"}, {"label": "b", "text": "b begins"},
                       {"label": "s", "text": "s holds", "kind": "state"}],
        "clips": [{"file": "x.mp4", "occurrences": [{"label": "a", "start": 100, "end": 105},
                                                     {"label": "a", "start": 500, "end": 505},
                                                     {"label": "a", "start": 2000, "end": 2005}]},
                  {"file": "y.mp4", "occurrences": [{"label": "a", "start": 10, "end": 12},
                                                     {"label": "a", "start": 30, "end": 31},
                                                     {"label": "a", "start": 50, "end": 51},
                                                     {"label": "a", "start": 70, "end": 71},
                                                     {"label": "s", "start": 5, "end": 40}]}]})


def p(label, t0, t1=None, decided=None, frames=()):
    t1 = t0 if t1 is None else t1
    return Prediction(label, t0, t1, t1 if decided is None else decided, 0.9, tuple(frames))


def runs():
    x = ClipRun("x.mp4", 1800, cost_usd=0.3, compute_s=12.0,
                predictions=[p("a", 101, decided=103), p("a", 102, decided=104), p("b", 900)],
                looks=[Look("a", 499, 501, "no")])
    y = ClipRun("videos/y.mp4", 1800, cost_usd=0.1, compute_s=4.0,
                predictions=[p("a", 9, 11), p("s", 6)],
                looks=[Look("a", 9, 11, "yes", None, "detect", 0.9), Look("a", 29, 31, "yes", None, "detect", 0.9),
                       Look("a", 29, 31, "no", None, "verify"), Look("a", 49, 51, "unknown", "timeout"),
                       Look("s", 6, 6, "yes", None, "detect", 0.9)],
                drops=[Drop("a", 69, 72)], fires=[Fire("a", 10.0)])
    return [x, y]


def test_counts_metrics_and_cost():
    s = evaluate(labels(), runs(), tol=1.0)
    assert isinstance(s, Score)
    # x: a@100 matched (one dup), a@500 missed, a@2000 beyond the 1800 s watched; b: one false alert
    # y: a@10 matched, a@30/50/70 missed; s@5 matched
    assert (s.occurrences, s.tp, s.fp, s.fn, s.duplicates, s.predictions) == (7, 3, 1, 4, 1, 5)
    assert s.recall == pytest.approx(3 / 7) and s.precision == pytest.approx(3 / 5)
    assert s.hours == 1.0 and s.false_per_hour == 1.0
    assert s.false_per_100_occurrences == pytest.approx(100 / 7)
    assert s.cost_usd == pytest.approx(0.4) and s.cost_per_true_event == pytest.approx(0.4 / 3)
    assert s.compute_s == 16.0 and s.compute_s_per_true_event == pytest.approx(16 / 3)
    assert s.misses_by_stage == {"trigger_never_fired": 0, "budget_dropped": 1, "judge_failed": 1,
                                 "judge_said_no": 1, "cooldown_absorbed": 0, "verify_rejected": 1,
                                 "decide_suppressed": 0}
    assert sum(s.misses_by_stage.values()) == s.fn
    assert s.latency_p50_s == 1.0           # latencies 3 (x), 1 (y, a), 1 (y, s)
    f = [s.funnel[k] for k in sc.FUNNEL]
    assert f == sorted(f, reverse=True) and f[0] == 7 and f[-1] == 1
    assert s.funnel_matched_outside == 2     # no look at x a@100; y a@10 never verified
    assert s.funnel_not_applicable == ()


def test_cooldown_absorbed_and_ceiling():
    occs = [Occurrence("a", a, b) for a, b in ((10, 12), (20, 22), (60, 62))]
    looks = [Look("a", 19, 21, "yes", None, "detect", 0.9)]
    assert sc.attribute(occs[1], looks, [], 1.0, reports=(10.5,), cooldown=30.0) == "cooldown_absorbed"
    assert sc.attribute(occs[1], looks, [], 1.0, reports=(10.5,), cooldown=5.0) == "decide_suppressed"
    assert sc.reachable(occs, 30.0, 1.0) == 2 and sc.reachable(occs, 0.0, 1.0) == 3


def test_bootstrap_is_deterministic():
    a = evaluate(labels(), runs(), tol=1.0, seed=7, n=200)
    b = evaluate(labels(), list(runs()), tol=1.0, seed=7, n=200)
    assert a.ci95 == b.ci95 and set(a.ci95) == set(sc.CI_METRICS)
    lo, hi = a.ci95["recall"]
    assert 0.0 <= lo <= hi <= 1.0


def test_runs_must_name_labelled_clips_once():
    with pytest.raises(LabelError):
        evaluate(labels(), [ClipRun("z.mp4", 10)])
    with pytest.raises(LabelError):
        evaluate(labels(), [ClipRun("x.mp4", 10), ClipRun("other/x.mp4", 10)])
    empty = evaluate(labels(), [])
    assert empty.recall is None and empty.false_per_hour is None and empty.ci95["recall"] is None
