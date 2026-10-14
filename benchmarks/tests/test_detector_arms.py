"""The tracker arms: tracker ids, the zone-entry rule, and both arms end to end (offline)."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("scipy")
pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from benchmarks import datasets, judging, run, tracking  # noqa: E402
from benchmarks.tracker import Tracker  # noqa: E402
from benchmarks.cases import CondSpec, Occurrence, VideoCase  # noqa: E402


def box(cx, cy, w=0.1, h=0.3, s=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s]


def test_tracker_ids_are_stable_and_a_returning_object_gets_a_new_id():
    tr = Tracker(0.5, 0.1, 0.5, track_buffer=3)
    ids = []
    for k in range(10):     # two people walking in opposite directions
        out = tr.update(np.array([box(0.2 + 0.02 * k, 0.5), box(0.8 - 0.02 * k, 0.4)]), k / 5)
        ids.append(sorted(t.track_id for t in out))
    assert all(i == ids[1] for i in ids[1:]) and len(ids[1]) == 2
    for k in range(5):      # both gone longer than the buffer
        tr.update(np.zeros((0, 5)), 2 + k / 5)
    tr.update(np.array([box(0.5, 0.5)]), 4.0)
    back = tr.update(np.array([box(0.5, 0.5)]), 4.2)
    assert len(back) == 1 and back[0].track_id not in ids[1]


def test_tracker_low_score_boxes_keep_a_track_but_never_start_one():
    tr = Tracker(0.5, 0.1, 0.5)
    tr.update(np.array([box(0.3, 0.5)]), 0.0)
    first = tr.update(np.array([box(0.31, 0.5)]), 0.2)[0].track_id
    kept = tr.update(np.array([box(0.32, 0.5, s=0.3)]), 0.4)
    assert [t.track_id for t in kept] == [first]
    fresh = Tracker(0.5, 0.1, 0.5)
    assert not fresh.update(np.array([box(0.3, 0.5, s=0.3)]), 0.0) and not fresh.tracks


ZONE = (((0.4, 0.0), (1.0, 0.0), (1.0, 1.0), (0.4, 1.0)),)


def test_entering_from_outside_is_one_event():
    rule = tracking.EnterRule(ZONE)
    fired = [rule.step(1, (x, 0.5)) for x in (0.2, 0.3, 0.38, 0.45, 0.6, 0.3, 0.5)]
    assert fired == [False, False, False, True, False, False, False]    # once per track, even re-entering


def test_standing_inside_is_no_event_but_appearing_at_the_edge_and_walking_in_is():
    rule = tracking.EnterRule(ZONE)
    assert not any(rule.step(1, (0.7 + 0.001 * k, 0.5)) for k in range(20))
    assert [rule.step(2, (x, 0.5)) for x in (0.97, 0.95, 0.92)] == [False, False, True]


def test_crossing_twice_within_the_cooldown_is_one_decision(monkeypatch, tmp_path):
    case = _enter_case(tmp_path)
    monkeypatch.setattr(tracking, "detect", _scripted_people)
    r = _run(tracking.run_track_rule(case, judging.Timelines(), zones={"enter": ZONE}, cooldown_s=30.0, fps=5.0))
    assert len(r.fires) == 2 and len(r.preds) == 1 and r.cost_usd == 0 and r.compute["frames"] > 0
    assert r.preds[0].frames == (r.fires[0][1],) and r.unsupported == ["door"]


def _scripted_people(image, device="cpu"):
    """Two people walking into the zone from the left, at 2 s and 5 s (decoded from the frame's
    brightness marker written by ``_enter_case``)."""
    t = image[0, 0, 0] / 20.0
    out = []
    for start in (2.0, 5.0):
        if start <= t < start + 3.0:
            out.append(box(0.1 + 0.15 * (t - start), 0.5))
    return np.array(out).reshape(-1, 5)


def _enter_case(tmp_path, seconds=8.0, fps=10):
    import cv2
    path = tmp_path / "enter.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for n in range(int(seconds * fps)):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:8, :8] = int(round(n / fps * 20))     # the time, for the scripted detector
        writer.write(frame)
    writer.release()
    from benchmarks.cases import Segment
    conds = (CondSpec("enter", "a person walks in", "event"), CondSpec("door", "a door opens", "event"))
    return VideoCase("CAMX-enter", "local", (Segment(str(path), 0.0, seconds),), conds,
                     (Occurrence("enter", 2.0, 4.0), Occurrence("enter", 5.0, 7.0)), synthetic=True)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_decide_arm_delivers_through_percept(monkeypatch, tmp_path):
    case = _enter_case(tmp_path)
    monkeypatch.setattr(tracking, "detect", _scripted_people)
    guard = judging.CostGuard(judging.OracleJudge([case], judging.Timelines()), None)
    r = _run(tracking.run_track_decide(case, guard, judging.Timelines(), tmp_path / "t.jsonl",
                                       zones={"enter": ZONE}, rearm_s=30.0, speed=4.0, fps=5.0))
    assert r.unsupported == ["door"] and len(r.fires) == 2 and r.cost_usd == 0 and not r.error
    # one met per tracked person (rearm is per subject), each verified by the rule judge
    assert len(r.preds) == 2 and all(p.frames for p in r.preds)
    assert all(lk.answer == "yes" for lk in r.looks)
    assert guard.requests == 0      # the harness judge was never called


def _plan(directory):
    directory.mkdir(parents=True)
    plan = {"schema_version": "percept.watch/v1alpha1", "plan_id": "synthetic-zone", "revision": 1,
            "sources": [{"id": "synthetic", "kind": "video"}],
            "concerns": [{"id": "enter", "source": "synthetic", "description": "A person walks in.",
                          "judgment": {"route": "judge", "zones": [[[0.4, 0], [1, 0], [1, 1], [0.4, 1]]]},
                          "decision": {"kind": "event", "rearm_s": 30.0},
                          "attention": {"any": [{"on_change": {"threshold": 0.1}}]}}],
            "budget": {"max_requests_per_min": 24}}
    (directory / "synthetic.json").write_text(json.dumps(plan))


def test_both_arms_run_end_to_end_on_the_synthetic_dataset(tmp_path):
    """Both arms through the harness on the CI clips, with a plan for their zones."""
    _plan(tmp_path / "plans" / "tiny")
    out = tmp_path / "run"
    code = run.main(["--dataset", "synthetic", "--limit", "1", "--arms",
                     "chunk5,detector_track_rule,detector_track_decide", "--speed", "4", "--out", str(out),
                     "--detector-plans", str(tmp_path / "plans" / "tiny")])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert all(len(set(d.values())) == 1 for d in manifest["frame_digests"].values())
    assert manifest["tracker_config"]["detector_track_decide"]["max_requests_per_min"] == tracking.RULE_BUDGET
    table = json.loads((out / "metrics.json").read_text())
    for arm in ("detector_track_rule", "detector_track_decide"):
        assert table[arm]["cost_usd"] == 0 and table[arm]["footprint"]["detector_cpu_s"] >= 0
    _, results, _ = run.read_raw(out / "raw" / "detector_track_rule.jsonl")
    # no condition there is an entry: every one is unsupported and the detector is never run
    assert results[0].compute["frames"] == 0 and len(results[0].unsupported) == len(datasets.SYNTH)


def test_the_real_detector_runs_and_finds_no_person_in_a_blank_frame():
    try:
        out = tracking.detect(np.full((240, 320, 3), 127, np.uint8))
    except OSError as e:        # the weights are downloaded on first use
        pytest.skip(f"detector weights unavailable: {e}")
    assert out.shape[1] == 5 and not (out[:, 4] >= tracking.SCORE).any()


def test_zones_come_from_the_plan_matching_the_case(tmp_path):
    _plan(tmp_path / "tiny")
    args = run.parser().parse_args(["--detector-plans", str(tmp_path / "tiny")])
    case = datasets.synthetic(tmp_path / "clips", clips=1)[0]
    zones = run.detector_zones(case, "detector_track_rule", args, None)
    assert set(zones) == {"enter"} and len(zones["enter"]) == 1


def test_missing_dependencies_exit_with_the_install_hint(monkeypatch):
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                        lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name)) if name == "torch"
                        else real(name, *a, **k))
    with pytest.raises(SystemExit, match="requirements-detector.txt"):
        tracking.require()
