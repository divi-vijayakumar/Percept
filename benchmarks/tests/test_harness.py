"""CI-safe checks of the harness: generated clips only, no datasets, no keys, no network."""

import asyncio
import json
import unittest

try:
    import cv2  # noqa: F401
except ImportError:  # the harness needs the video extra
    raise unittest.SkipTest('benchmarks need OpenCV: pip install "percept-vision[video]"')

import pytest

from benchmarks import arms, datasets, judging, metrics, run
from benchmarks.arms import ArmResult, Look, Pred
from benchmarks.cases import CondSpec, Occurrence, Segment, VideoCase


def pred(t0, t1=None, decided=None, cid="a"):
    t1 = t0 if t1 is None else t1
    return Pred(cid, t0, t1, t1 if decided is None else decided, 0.9)


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


# --------------------------------------------------------------------------- metrics


def test_metric_formulas():
    conds = (CondSpec("a", "a begins"), CondSpec("b", "b begins"))
    case = VideoCase("x", "t", (Segment("unused.mp4", 0, 1800),), conds,
                     (Occurrence("a", 100, 105), Occurrence("a", 500, 505)))
    r = ArmResult("arm", "x", cost_usd=0.3, requests=3)
    r.preds = [pred(101, decided=103), pred(102, decided=104), pred(900, cid="b")]
    r.looks = [Look("a", 499, 501, "no", None)]
    s = metrics.score([r], {"x": case}, tol=1.0)
    assert s["recall"] == 0.5 and s["precision"] == pytest.approx(1 / 3)
    assert s["false_per_hour"] == 2.0 and s["false_on_absent_conditions"] == 1
    assert s["duplicates_per_occurrence"] == 1.0 and s["latency_p50_s"] == 3.0
    assert s["cost_per_hour"] == pytest.approx(0.6) and s["cost_per_true_event"] == pytest.approx(0.3)
    assert s["misses_by_stage"]["judge_said_no"] == 1 and sum(s["misses_by_stage"].values()) == 1


def test_unsupported_conditions_are_not_scored():
    case = VideoCase("x", "t", (Segment("unused.mp4", 0, 3600),), (CondSpec("a", "a"),), (Occurrence("a", 1, 2),))
    s = metrics.score([ArmResult("d", "x", unsupported=["a"])], {"x": case}, tol=1.0)
    assert s["occurrences"] == 0 and s["recall"] is None and s["unsupported"] == ["x/a"]


@pytest.mark.parametrize("looks, drops, stage", [
    ([], [], "trigger_never_fired"),
    ([], [("a", 0, 60)], "budget_dropped"),
    ([Look("a", 9, 11, "unknown", "budget")], [], "budget_dropped"),
    ([Look("a", 9, 11, "unknown", "timeout")], [], "judge_failed"),
    ([Look("a", 9, 11, "no", None), Look("a", 10, 12, "unknown", "timeout")], [], "judge_said_no"),
    ([Look("a", 9, 11, "no", None), Look("a", 9, 11, "yes", None, "detect")], [], "decide_suppressed"),
    ([Look("a", 30, 34, "yes", None)], [], "trigger_never_fired"),     # a look elsewhere doesn't count
])
def test_stage_attribution(looks, drops, stage):
    assert metrics.attribute(Occurrence("a", 10, 12), looks, drops, tol=1.0) == stage


# --------------------------------------------------------------------------- the spend cap


def test_cost_cap_aborts_cleanly_and_is_never_exceeded(clips):
    guard, timelines = oracle_guard(clips, cap=0.0008)
    result = asyncio.run(arms.run_chunk(clips[0], guard, timelines, 2.0))
    assert guard.tripped.is_set() and guard.refused >= 1 and result.aborted
    assert 0 < guard.spent <= 0.0008 and guard.held == 0
    assert any(lk.failed == "budget" for lk in result.looks)


def test_a_cap_needs_a_price(clips):
    oracle = judging.OracleJudge(clips, judging.Timelines(), price=None)
    with pytest.raises(ValueError):
        judging.CostGuard(oracle, 1.0)


# --------------------------------------------------------------------------- arms and the oracle


def test_oracle_answers_onsets_from_ground_truth(clips):
    oracle = judging.OracleJudge(clips, judging.Timelines(), fn_rate=0.0, fp_rate=0.0)
    case = clips[0]
    red = next(o for o in case.occurrences if o.label == "red square appears")
    assert oracle.truth(case.case_id, red.label, red.start - 1, red.start + 1)
    assert not oracle.truth(case.case_id, red.label, red.start + 2.5, red.start + 3)
    assert not oracle.truth(case.case_id, "green square appears", 0, case.duration)


def test_baseline_arms_see_identical_evidence(clips):
    guard, timelines = oracle_guard(clips)
    case = clips[0]
    results = asyncio.run(_baselines(case, guard, timelines))
    digests = {r.digest for r in results}
    assert len(digests) == 1 and not next(iter(digests)).startswith("0:")
    assert results[1].unsupported == [] and results[0].preds      # motion expresses every synthetic condition


async def _baselines(case, guard, timelines):
    return [await arms.run_chunk(case, guard, timelines, 5.0),
            await arms.run_detector(case, guard, timelines, then_vlm=False),
            await arms.run_detector(case, guard, timelines, then_vlm=True)]


def test_dry_run_end_to_end(tmp_path):
    out = tmp_path / "run"
    code = run.main(["--dataset", "synthetic", "--limit", "1", "--arms", "percept,chunk5,detector_then_vlm",
                     "--speed", "3", "--oracle-latency", "0.2", "--out", str(out), "--max-cost-usd", "1"])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["arms"] == ["percept", "chunk5", "detector_then_vlm"] and not manifest["aborted"]
    assert all(len(set(d.values())) == 1 for d in manifest["frame_digests"].values())
    assert manifest["cases"][0]["media"] and manifest["percept"]["package_sha256"]
    table = json.loads((out / "metrics.json").read_text())
    assert table["percept"]["requests"] > 0 and table["percept"]["recall"] is not None
    text = (out / "report.md").read_text()
    for heading in ("Dry run", "## Every arm", "## Misses by stage", "## Footprint", "## Where Percept loses",
                    "Identical evidence: yes", "only 1.0x results are publishable"):
        assert heading in text
    assert (out / "traces" / "synthetic-0.jsonl").exists()


def test_meva_activity_lines_parse():
    line = ("- {'act': {'act2': {'person_picks_up_object': 1.0}, 'actors': [{'id1': 2, 'timespan': "
            "[{'tsr0': [919, 938]}]}], 'id2': 1, 'src_status': 'good', 'timespan': [{'tsr0': [915, 940]}]}}")
    assert datasets.meva_activities("- { meta: x }\n" + line, fps=10.0) == \
        [Occurrence("person_picks_up_object", 91.5, 94.0)]


def test_e2_config_is_a_scaffold():
    with pytest.raises(SystemExit, match="scaffold"):
        run.main(["--config", str(run.Path(run.__file__).parent / "configs" / "e2.json")])


def test_dry_run_reports_an_abort(tmp_path):
    out = tmp_path / "run"
    code = run.main(["--dataset", "synthetic", "--limit", "1", "--arms", "chunk5,chunk10", "--out", str(out),
                     "--max-cost-usd", "0.0005"])
    assert code == 3
    assert "Aborted: spend cap reached" in (out / "report.md").read_text()
    assert json.loads((out / "manifest.json").read_text())["spent_usd"] <= 0.0005
