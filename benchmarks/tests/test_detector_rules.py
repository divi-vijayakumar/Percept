"""Which conditions the tracker arms can express, and where their zones come from: no detector,
torch or scipy needed, so these run wherever the benchmark tests do."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("cv2")

from benchmarks import datasets, run, tracking   # noqa: E402


def case(*ids, dataset="local", case_id="x.G341"):
    return SimpleNamespace(case_id=case_id, dataset=dataset, conditions=[SimpleNamespace(id=i) for i in ids])


def test_entry_is_recognised_by_its_local_label_and_its_meva_activity():
    assert set(tracking.supported(case("enter", "door"))) == {"enter"}
    assert set(tracking.supported(case(*datasets.MEVA_PERSON, "person_opens_facility_door"))) == set(datasets.MEVA_PERSON)
    assert tracking.supported(case("door", "person_exits_scene_through_structure")) == {}


def _plan(directory, zones=True):
    directory.mkdir(parents=True)
    judgment = {"route": "judge", **({"zones": [[[0.4, 0], [1, 0], [1, 1], [0.4, 1]]]} if zones else {})}
    plan = {"schema_version": "percept.watch/v1alpha1", "plan_id": "p", "revision": 1,
            "sources": [{"id": "G341", "kind": "video"}],
            "concerns": [{"id": "enter", "source": "G341", "description": "A person walks in.",
                          "judgment": judgment, "decision": {"kind": "event", "rearm_s": 30.0}}],
            "budget": {"max_requests_per_min": 24}}
    (directory / "G341.json").write_text(json.dumps(plan))


def test_a_tracker_arm_never_falls_back_to_the_whole_frame_on_real_footage(tmp_path):
    bare = run.parser().parse_args(["--dataset", "local:x"])
    with pytest.raises(ValueError, match="needs --detector-plans"):
        run.detector_zones(case("enter"), "detector_track_rule", bare, None)
    assert run.detector_zones(case("enter", dataset="synthetic"), "detector_track_rule", bare, None) == {}
    _plan(tmp_path / "nozone", zones=False)
    args = run.parser().parse_args(["--dataset", "local:x", "--detector-plans", str(tmp_path / "nozone")])
    with pytest.raises(ValueError, match="gives no zones for"):
        run.detector_zones(case("enter"), "detector_track_rule", args, None)
    _plan(tmp_path / "zone")
    args = run.parser().parse_args(["--dataset", "local:x", "--detector-plans", str(tmp_path / "zone")])
    assert len(run.detector_zones(case("enter"), "detector_track_rule", args, None)["enter"]) == 1


def test_the_tracker_arms_are_always_paced_and_never_bounded_for_fail_fast():
    args = run.parser().parse_args(["--dataset", "local:x"])
    assert run.tracker_config("detector_track_rule", args)["pace"] == args.speed
    real = SimpleNamespace(case_id="x.G341", duration=300.0, conditions=[SimpleNamespace(id="enter")])
    assert run.look_bound("detector_track_rule", [real], args, None) is None


def _box(cx, cy, w=0.1, h=0.3, s=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s]


def test_a_track_seen_once_is_dropped_and_a_briefly_lost_one_keeps_its_id():
    pytest.importorskip("scipy")
    import numpy as np
    from benchmarks.tracker import Tracker
    once = Tracker(0.5, 0.1, 0.5, track_buffer=3)
    once.update(np.zeros((0, 5)), 0.0)                                   # (on the first frame, as in
    assert once.update(np.array([_box(0.3, 0.5)]), 0.2) == []           # the footage, tracks start confirmed)
    once.update(np.zeros((0, 5)), 0.4)                                   # tentative and missed
    assert not once.tracks                                               # never confirmed: gone
    tr = Tracker(0.5, 0.1, 0.5, track_buffer=3)
    for k in range(3):
        out = tr.update(np.array([_box(0.3 + 0.01 * k, 0.5)]), k / 5)
    first = out[0].track_id
    for k in range(3, 5):                                                # lost for 0.4 s, within the buffer
        assert tr.update(np.zeros((0, 5)), k / 5) == []
    back = tr.update(np.array([_box(0.35, 0.5)]), 1.0)
    assert [t.track_id for t in back] == [first]
