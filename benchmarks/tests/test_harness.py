"""CI-safe checks of the harness: generated clips only, no datasets, no keys, no network."""

import asyncio
import json
import unittest

try:
    import cv2  # noqa: F401
except ImportError:  # the harness needs the video extra
    raise unittest.SkipTest('benchmarks need OpenCV: pip install -r benchmarks/requirements.txt')

import numpy as np
import pytest

from percept import JudgeItem, JudgeRequest, Condition
from benchmarks import arms, datasets, judging, metrics, run
from benchmarks.arms import ArmResult, Look, Pred
from benchmarks.cases import CondSpec, Occurrence, Segment, VideoCase


def pred(t0, t1=None, decided=None, cid="a"):
    t1 = t0 if t1 is None else t1
    return Pred(cid, t0, t1, t1 if decided is None else decided, 0.9)


def watched(arm, case_id, seconds, **kw):
    """An ArmResult for a case the arm actually watched."""
    return ArmResult(arm, case_id, digest="1:x", seconds=seconds, **kw)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    return datasets.synthetic(tmp_path_factory.mktemp("clips"), clips=1)


def oracle_guard(cases, cap=None, latency=0.0):
    timelines = judging.Timelines()
    oracle = judging.OracleJudge(cases, timelines, fn_rate=0.0, fp_rate=0.0, latency_s=latency)
    return judging.CostGuard(oracle, cap), timelines


# --------------------------------------------------------------------------- matching


def test_matching_is_one_to_one_with_duplicates_and_false_events():
    occs = [Occurrence("a", 10, 12), Occurrence("a", 11, 20)]
    m = metrics.match([pred(10.5), pred(11.5), pred(11.8), pred(50)], occs, tol=0.0)
    assert {i: p.t0 for i, p in m["matched"].items()} == {0: 10.5, 1: 11.5}
    assert [p.t0 for p in m["dups"]] == [11.8] and [p.t0 for p in m["false"]] == [50]
    assert m["missed"] == []


def test_matching_tolerance_and_evidence_intervals():
    occ = [Occurrence("a", 10, 12)]
    assert len(metrics.match([pred(13.9)], occ, tol=2.0)["matched"]) == 1
    assert len(metrics.match([pred(14.1)], occ, tol=2.0)["false"]) == 1
    # a long chunk whose evidence covers the onset matches even though it ends much later
    assert len(metrics.match([pred(0, 30)], occ, tol=0.0)["matched"]) == 1


def test_no_credit_for_deciding_before_the_onset():
    occ = [Occurrence("a", 10, 12)]
    assert len(metrics.match([pred(8.0, 9.4)], occ, tol=2.0)["false"]) == 1      # ends before onset - 0.5
    assert len(metrics.match([pred(8.0, 9.6)], occ, tol=2.0)["matched"]) == 1


# --------------------------------------------------------------------------- metrics


def test_metric_formulas():
    conds = (CondSpec("a", "a begins"), CondSpec("b", "b begins"))
    case = VideoCase("x", "t", (Segment("unused.mp4", 0, 1800),), conds,
                     (Occurrence("a", 100, 105), Occurrence("a", 500, 505)))
    r = watched("arm", "x", 1800, cost_usd=0.3, requests=3)
    r.preds = [pred(101, decided=103), pred(102, decided=104), pred(900, cid="b")]
    r.looks = [Look("a", 499, 501, "no", None)]
    s = metrics.score([r], {"x": case}, tol=1.0)
    assert s["recall"] == 0.5 and s["precision"] == pytest.approx(1 / 3) and s["recall_ceiling"] == 1.0
    assert s["false_per_hour"] == 2.0 and s["false_on_absent_conditions"] == 1
    assert s["duplicates_per_occurrence"] == 1.0 and s["latency_p50_s"] == 3.0
    assert s["cost_per_hour"] == pytest.approx(0.6) and s["cost_per_true_event"] == pytest.approx(0.3)
    assert s["misses_by_stage"]["judge_said_no"] == 1 and sum(s["misses_by_stage"].values()) == 1


def test_skipped_and_cut_short_cases_leave_the_denominators():
    conds = (CondSpec("a", "a"),)
    occs = (Occurrence("a", 100, 105), Occurrence("a", 2000, 2005))
    cases = {k: VideoCase(k, "t", (Segment("u.mp4", 0, 3600),), conds, occs) for k in ("x", "y", "z")}
    results = [watched("arm", "x", 1800, preds=[pred(101)]),          # cut short at 1800 s by the cap
               ArmResult("arm", "y", aborted=True),                     # never started
               ArmResult("arm", "z", error="RuntimeError: boom")]
    s = metrics.score(results, cases, tol=1.0)
    assert s["cases"] == 1 and s["source_hours"] == 0.5 and s["occurrences"] == 1 and s["recall"] == 1.0
    assert [k["case_id"] for k in s["skipped"]] == ["y", "z"] and "boom" in s["skipped"][1]["reason"]


def test_unsupported_conditions_are_not_scored():
    case = VideoCase("x", "t", (Segment("unused.mp4", 0, 3600),), (CondSpec("a", "a"),), (Occurrence("a", 1, 2),))
    s = metrics.score([watched("d", "x", 3600, unsupported=["a"])], {"x": case}, tol=1.0)
    assert s["occurrences"] == 0 and s["recall"] is None and s["unsupported"] == ["x/a"]


@pytest.mark.parametrize("looks, drops, stage", [
    ([], [], "trigger_never_fired"),
    ([], [("a", 0, 60)], "budget_dropped"),
    ([Look("a", 9, 11, "unknown", "budget")], [], "budget_dropped"),
    ([Look("a", 9, 11, "unknown", "timeout")], [], "judge_failed"),
    ([Look("a", 9, 11, "no", None), Look("a", 10, 12, "unknown", "timeout")], [], "judge_said_no"),
    ([Look("a", 9, 11, "no", None), Look("a", 9, 11, "yes", None, "detect")], [], "decide_suppressed"),
    ([Look("a", 9, 11, "yes", None), Look("a", 9, 11, "no", None, "verify")], [], "verify_rejected"),
    ([Look("a", 30, 34, "yes", None)], [], "trigger_never_fired"),     # a look elsewhere doesn't count
])
def test_stage_attribution(looks, drops, stage):
    assert metrics.attribute(Occurrence("a", 10, 12), looks, drops, tol=1.0) == stage


def test_cooldown_absorbed_and_the_recall_ceiling():
    occs = [Occurrence("a", 10, 12), Occurrence("a", 20, 22), Occurrence("a", 60, 62)]
    looks = [Look("a", 19, 21, "yes", None)]
    assert metrics.attribute(occs[1], looks, [], 1.0, reports=(10.5,), cooldown=30.0) == "cooldown_absorbed"
    assert metrics.attribute(occs[1], looks, [], 1.0, reports=(10.5,), cooldown=5.0) == "decide_suppressed"
    assert metrics.reachable(occs, 30.0, 1.0) == 2 and metrics.reachable(occs, 0.0, 1.0) == 3


# --------------------------------------------------------------------------- the spend cap


def test_cost_cap_aborts_cleanly_and_is_never_exceeded(clips):
    guard, timelines = oracle_guard(clips, cap=0.0008)
    result = asyncio.run(arms.run_chunk(clips[0], guard, timelines, 2.0))
    assert guard.tripped.is_set() and guard.refused >= 1 and result.aborted
    assert 0 < guard.spent <= 0.0008 and guard.held == 0
    assert result.seconds < clips[0].duration


def test_cost_cap_stops_the_percept_arm(clips, tmp_path):
    guard, timelines = oracle_guard(clips, cap=0.0008, latency=0.05)
    guard.inner.realtime = True
    result = asyncio.run(arms.run_percept(clips[0], guard, timelines, tmp_path / "t.jsonl", speed=4.0,
                                          max_requests_per_min=600))
    assert guard.tripped.is_set() and result.aborted and 0 < guard.spent <= 0.0008
    assert result.seconds < clips[0].duration


def test_a_cap_needs_a_price(clips):
    oracle = judging.OracleJudge(clips, judging.Timelines(), price=None)
    with pytest.raises(ValueError):
        judging.CostGuard(oracle, 1.0)


# --------------------------------------------------------------------------- arms and the oracle


def _request(case, timelines, frames, purpose="detect"):
    item = JudgeItem(Condition("red_square_appears", "a red square appears", kind="event"))
    return JudgeRequest("r", case.case_id, 0, (item,), tuple(f[1] for f in frames),
                        (frames[0][0], frames[-1][0]), purpose)


def test_oracle_answers_from_the_sampled_frames_not_the_interval(clips):
    case, timelines = clips[0], judging.Timelines()
    timelines.set(case.case_id, 0.0)
    red = next(o for o in case.occurrences if o.label == "red square appears")
    images = [np.full((4, 4, 3), i, np.uint8) for i in range(3)]
    before, after = red.start - 3.0, red.start + 4.0          # an interval spanning the onset ...
    for t, image in zip((before, after), images):
        timelines.note(case.case_id, t, image)
    oracle = judging.OracleJudge(clips, timelines, fn_rate=0.0, fp_rate=0.0)
    sparse = _request(case, timelines, [(before, images[0]), (after, images[1])])
    assert oracle._answer(sparse)["red_square_appears"][0] == "no"     # ... but no frame shows it
    timelines.note(case.case_id, red.start + 0.4, images[2])
    dense = _request(case, timelines, [(before, images[0]), (red.start + 0.4, images[2]), (after, images[1])])
    assert oracle._answer(dense)["red_square_appears"][0] == "yes"


def test_oracle_noise_depends_on_the_evidence_only(clips):
    case, timelines = clips[0], judging.Timelines()
    timelines.set(case.case_id, 0.0)
    image = np.zeros((4, 4, 3), np.uint8)
    timelines.note(case.case_id, 1.0, image)
    oracle = judging.OracleJudge(clips, timelines, fn_rate=0.5, fp_rate=0.5)
    answers = {oracle._answer(_request(case, timelines, [(1.0, image)], p))["red_square_appears"]
               for p in ("detect", "verify")}
    assert len(answers) == 1


def test_baseline_arms_see_identical_evidence(clips):
    guard, timelines = oracle_guard(clips)
    case = clips[0]
    results = asyncio.run(_baselines(case, guard, timelines))
    digests = {r.digest for r in results}
    assert len(digests) == 1 and not next(iter(digests)).startswith("0:")
    assert results[1].unsupported == [] and results[0].preds      # motion expresses every synthetic condition
    assert all(r.seconds == case.duration and r.cooldown_s == arms.COOLDOWN_S for r in results)


async def _baselines(case, guard, timelines):
    return [await arms.run_chunk(case, guard, timelines, 5.0),
            await arms.run_detector(case, guard, timelines, then_vlm=False),
            await arms.run_detector(case, guard, timelines, then_vlm=True)]


def test_long_sparse_negatives_never_appear_in_distractors():
    vocab = [f"l{i}" for i in range(12)]
    pool = [VideoCase(f"c{i}", "t", (Segment(f"c{i}.mp4", 0, 20),), (), (Occurrence(vocab[i], 2, 4),))
            for i in range(12)]
    for case in datasets.long_sparse(pool, occupancies=(0.1, 0.3), length_s=200, negatives=6, seed=3):
        shown = {o.label for s in case.segments for c in pool if c.segments[0].path == s.path and not s.hold
                 for o in c.occurrences}
        present = {o.label for o in case.occurrences}
        negatives = {c.label for c in case.conditions} - present
        assert negatives and not negatives & shown


def test_a_failing_case_is_recorded_not_fatal(clips, tmp_path, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("decoder exploded")
    monkeypatch.setattr(arms, "run_chunk", boom)
    out = tmp_path / "run"
    assert run.main(["--dataset", "synthetic", "--limit", "1", "--arms", "chunk5,detector_debounce",
                     "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["skipped"][0]["arm"] == "chunk5" and "decoder exploded" in manifest["skipped"][0]["reason"]
    assert not manifest["publishable"] and "some cases were skipped" in manifest["not_publishable"]


def test_unreadable_media_fails_the_case_not_quiet_footage(clips, tmp_path):
    guard, timelines = oracle_guard(clips)
    bad = tmp_path / "corrupt.avi"
    bad.write_bytes(b"not a video")
    case = VideoCase("bad", "t", (Segment(str(bad), 0, 20),), clips[0].conditions, ())
    with pytest.raises(OSError, match="cannot open"):
        asyncio.run(arms.run_chunk(case, guard, timelines, 5.0))


def test_missing_media_stops_the_run(clips, tmp_path):
    gone = VideoCase("gone", "t", (Segment(str(tmp_path / "unmounted.avi"), 0, 20),), clips[0].conditions, ())
    guard, timelines = oracle_guard([gone])
    args = run.parser().parse_args(["--dataset", "synthetic", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="media files are missing"):
        asyncio.run(run.run_arm("chunk5", [gone], guard, timelines, tmp_path / "o", args, {}))


def test_dry_run_end_to_end(tmp_path):
    out = tmp_path / "run"
    code = run.main(["--dataset", "synthetic", "--limit", "1", "--arms", "percept,percept-matched,chunk5",
                     "--speed", "3", "--oracle-latency", "0.2", "--out", str(out), "--max-cost-usd", "1",
                     "--percept-rearm", "30"])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["arms"] == ["percept", "percept-matched", "chunk5"] and not manifest["aborted"]
    assert all(len(set(d.values())) == 1 for d in manifest["frame_digests"].values())
    assert manifest["cases"][0]["media"] and manifest["percept"]["package_sha256"]
    config = manifest["monitor_config"]
    assert config["percept"]["max_requests_per_min"] == 6 and config["percept-matched"]["max_requests_per_min"] == 12
    assert config["percept"]["attempts_reserved_per_request"] == 2 and config["percept"]["condition"]["rearm_s"] == 30
    assert manifest["harness"]["clip_frames"] == arms.CLIP_FRAMES and not manifest["publishable"]
    table = json.loads((out / "metrics.json").read_text())
    assert table["percept"]["requests"] > 0 and table["percept"]["recall"] is not None
    p = table["percept"]
    assert p["unintended_loss_total"] == 0 and p["unintended_loss"]["handoff_dropped"] == 0
    assert set(p["intentional_sampling"]) == {"harness_fps", "monitor_fps_cap"}
    for key in ("budget_drops", "trigger_never_fired", "verify_rejected", "unverified", "uncertain", "watched_s",
                "cases_skipped", "cost_per_hour", "latency_p95_s", "false_per_hour"):
        assert key in p
    label = "percept (product defaults; rearm standardised to 30 s)"
    assert p["label"] == label and config["percept"]["label"] == label
    assert config["percept"]["changed_from_product_defaults"]["condition.rearm_s"]["product_default"] == 120
    text = (out / "report.md").read_text()
    for heading in ("Dry run", "Diagnostic, not publishable", "Publishable: no", "## Every arm", "recall ceiling",
                    "## Losses and verification", "## Misses by stage", "Actual spend $", f"| {label} |",
                    "## Footprint", f"## Where {label} loses", "## Where percept-matched (max_requests_per_min 12; "
                    "rearm standardised to 30 s) loses",
                    "Identical evidence: yes", "only 1.0x results are publishable", "2 reserved per request"):
        assert heading in text
    assert (out / "traces" / "percept" / "synthetic-0.jsonl").exists()


def _meva_run(**arm):
    manifest = {"judge": {"spec": "gemini:m"}, "dataset": "meva", "speed": 1.0, "aborted": False}
    s = dict(skipped=[], footprint={"max_push_lag_s": 0.1}, intentional_sampling={}, unintended_loss={},
             unintended_loss_total=0)
    s.update(arm)
    return manifest, {"percept": s}


def test_intentional_sampling_is_publishable_unintended_loss_is_not():
    heavy = {"harness_fps": 900_000, "monitor_fps_cap": 50_000}
    assert run.not_publishable(*_meva_run(intentional_sampling=heavy)) == []
    why = run.not_publishable(*_meva_run(intentional_sampling=heavy, unintended_loss={"handoff_dropped": 3},
                                         unintended_loss_total=3))
    assert len(why) == 1 and "unintended_loss" in why[0]
    assert run.not_publishable(*_meva_run(footprint={"max_push_lag_s": 2.5}))       # the lag limit still holds


def test_loss_accounting_adds_up_per_arm():
    case = VideoCase("x", "t", (Segment("u.mp4", 0, 60),), (CondSpec("a", "a"),), ())
    r = [watched("percept", "x", 60, intentional_sampling={"harness_fps": 1500, "monitor_fps_cap": 0},
                 unintended_loss={"decode_gaps": 2, "handoff_dropped": 1, "late_frames": 0},
                 budget_dropped=4, unverified=1, uncertain=2)]
    s = metrics.score(r * 2, {"x": case}, tol=1.0)
    assert s["intentional_sampling"] == {"harness_fps": 3000, "monitor_fps_cap": 0}
    assert s["unintended_loss_total"] == 6 and s["budget_drops"] == 8 and s["unverified"] == 2
    assert s["watched_s"] == 120 and s["cases_skipped"] == 0


def test_meva_activity_lines_parse():
    line = ("- {'act': {'act2': {'person_picks_up_object': 1.0}, 'actors': [{'id1': 2, 'timespan': "
            "[{'tsr0': [919, 938]}]}], 'id2': 1, 'src_status': 'good', 'timespan': [{'tsr0': [915, 940]}]}}")
    assert datasets.meva_activities("- { meta: x }\n" + line, fps=10.0) == \
        [Occurrence("person_picks_up_object", 91.5, 94.0)]
    empty = line.replace("person_picks_up_object", "empty_37")
    assert datasets.meva_activities(empty, fps=10.0) == []


def test_baseline_reply_limit_grows_with_the_batch(clips):
    many = VideoCase("m", "t", clips[0].segments, tuple(CondSpec(f"cond_{i}", f"c {i}") for i in range(25)), ())
    guard, timelines = oracle_guard([many])
    sent, judge = [], guard.judge

    async def spy(request):
        sent.append(request)
        return await judge(request)
    guard.judge = spy
    asyncio.run(arms.run_chunk(many, guard, timelines, 5.0))
    assert sent and all(r.max_output_tokens > 400 for r in sent)    # 400 truncated a 25-condition reply


def test_configs():
    e1 = json.loads((run.Path(run.__file__).parent / "configs" / "e1.json").read_text())
    assert e1["dataset"] == "meva" and e1["cooldown"] == e1["percept_rearm"] == 30.0
    assert "percept-matched" in e1["arms"]
    with pytest.raises(SystemExit, match="scaffold"):
        run.main(["--config", str(run.Path(run.__file__).parent / "configs" / "e2.json")])


def test_dry_run_reports_an_abort(tmp_path):
    out = tmp_path / "run"
    code = run.main(["--dataset", "synthetic", "--limit", "2", "--arms", "chunk5,chunk10", "--out", str(out),
                     "--max-cost-usd", "0.0005", "--parallel", "1"])
    assert code == 3
    assert "Aborted: spend cap reached" in (out / "report.md").read_text()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["spent_usd"] <= 0.0005 and manifest["skipped"]
    table = json.loads((out / "metrics.json").read_text())
    assert table["chunk10"]["cases"] == 0 and table["chunk10"]["source_hours"] == 0
    assert table["chunk10"]["cases_skipped"] == 2
    assert sum(s["cost_usd"] for s in table.values()) == pytest.approx(manifest["spent_usd"])   # partial spend kept
