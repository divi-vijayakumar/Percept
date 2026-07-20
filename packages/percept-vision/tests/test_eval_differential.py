"""Card D3 — `percept eval`, the paired differential A/B over client fixtures.

The properties under test are the card's own promises: paired (not two independent runs), honest
about scope (ADR-006 shapes), silent about pass/fail without client bars, and refusing to score a
configuration that moved after the bars were declared.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from percept.eval.differential import (
    Bar,
    BarsDocument,
    ConfigDriftError,
    EvalInputError,
    OverallVerdict,
    ScopeVerdict,
    load_bars,
    load_fixtures,
    load_plan,
    render_text,
    run_differential,
    scope_for_goal_shape,
)

ANN = "eligibility-rubric-v1"


# ---------------------------------------------------------------------------
# fixture-tree builders
# ---------------------------------------------------------------------------


def _episode(eid: str, start: float, end: float, lateness: float = 1.0) -> dict:
    return {
        "episode_id": eid,
        "event_start_ts": start,
        "decision_eligible_ts": start,
        "episode_end_ts": end,
        "allowed_lateness_s": lateness,
        "annotation_version": ANN,
        "annotator_or_fixture_id": "fixture-gen",
    }


def _event(key: str, ts: float, seq: int = 0, action: str = "fire") -> dict:
    return {
        "key": key,
        "evidence_seq": seq,
        "action": action,
        "timing": {"decision_emitted_ts": ts},
    }


def _trace(run_id: str, *, cost: float, input_tokens: float, judge_completed: int) -> dict:
    return {
        "manifest": {"run_id": run_id},
        "records": [],
        "summary": {
            "run_id": run_id,
            "judge_completed": judge_completed,
            "cost": {"input_tokens": input_tokens, "output_tokens": 0.0, "usd": cost},
        },
    }


def write_fixture(
    root: Path,
    fixture_id: str,
    *,
    goal_shape: str,
    episodes: list[dict],
    baseline_events: list[dict],
    candidate_events: list[dict],
    baseline_cost: float | None = None,
    candidate_cost: float | None = None,
) -> None:
    d = root / fixture_id
    (d / "baseline").mkdir(parents=True)
    (d / "candidate").mkdir(parents=True)
    (d / "ground_truth.json").write_text(
        json.dumps({"goal_shape": goal_shape, "episodes": episodes})
    )
    (d / "baseline" / "events.json").write_text(json.dumps(baseline_events))
    (d / "candidate" / "events.json").write_text(json.dumps(candidate_events))
    if baseline_cost is not None:
        (d / "baseline" / "trace.json").write_text(
            json.dumps(_trace(f"{fixture_id}-b", cost=baseline_cost, input_tokens=1000,
                              judge_completed=10))
        )
    if candidate_cost is not None:
        (d / "candidate" / "trace.json").write_text(
            json.dumps(_trace(f"{fixture_id}-c", cost=candidate_cost, input_tokens=1400,
                              judge_completed=12))
        )


def write_plan(path: Path, *, condition: str, model: str, thresholds: dict | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "goals": [{"id": "g1", "condition": condition, "say": "heads up"}],
                "components": [
                    {"role": "vision", "name": "ocr", "version": "1.2.0", "model": model},
                    {"role": "judge", "name": "claude", "version": "0.9", "model": model},
                ],
                "thresholds": thresholds or {"confidence": 0.7, "dwell_s": 1.0},
            }
        )
    )
    return path


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """Four fixtures: 3 candidate-better episodes/points + 1 unsupported (correction) shape."""
    root = tmp_path / "fixtures"
    root.mkdir()
    for i, name in enumerate(["doc_a", "doc_b", "doc_c"]):
        write_fixture(
            root,
            name,
            goal_shape="episode",
            episodes=[_episode("e1", 10.0, 12.0), _episode("e2", 30.0, 32.0)],
            # baseline finds only the first episode
            baseline_events=[_event("e1", 11.0)],
            # candidate finds both
            candidate_events=[_event("e1", 11.0), _event("e2", 31.0, seq=1)],
            baseline_cost=0.10 + i * 0.01,
            candidate_cost=0.14 + i * 0.01,
        )
    write_fixture(
        root,
        "call_retraction",
        goal_shape="correction",
        episodes=[_episode("c1", 5.0, 6.0)],
        baseline_events=[],
        candidate_events=[_event("c1", 5.5)],
    )
    (root / "splits.json").write_text(
        json.dumps({"train": [], "dev": ["doc_c"], "test": ["doc_a", "doc_b", "call_retraction"]})
    )
    return root


@pytest.fixture()
def arms(tmp_path: Path):
    b = load_plan(write_plan(tmp_path / "baseline.json", condition="invoice total visible",
                             model="sonnet"), name="baseline")
    c = load_plan(write_plan(tmp_path / "candidate.json", condition="invoice total visible via OCR",
                             model="sonnet"), name="candidate")
    return b, c


# ---------------------------------------------------------------------------
# scope honesty (ADR-006)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["point", "episode"])
def test_point_and_episode_shapes_are_supported(shape):
    assert scope_for_goal_shape(shape).verdict is ScopeVerdict.SUPPORTED


def test_recurrence_is_only_partially_supported():
    a = scope_for_goal_shape("recurrence")
    assert a.verdict is ScopeVerdict.PARTIALLY_SUPPORTED
    assert a.scorable  # partial still scores, but the report says what is missing
    assert "RELATION" in a.reason


@pytest.mark.parametrize(
    "shape",
    ["supersession", "correction", "settled_interval", "speaker_attributed_claim",
     "document_identity_change"],
)
def test_adr006_shapes_are_unsupported_not_silently_scored(shape):
    a = scope_for_goal_shape(shape)
    assert a.verdict is ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE
    assert not a.scorable
    assert "ADR-006" in a.reason


def test_unknown_shape_is_unsupported_never_optimistically_supported():
    a = scope_for_goal_shape("holographic_vibes")
    assert a.verdict is ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE


def test_undeclared_shape_is_unsupported():
    assert scope_for_goal_shape("").verdict is ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------


def test_plan_hash_is_content_addressed_and_differs_between_arms(arms):
    baseline, candidate = arms
    assert baseline.plan_hash and candidate.plan_hash
    assert baseline.plan_hash != candidate.plan_hash


def test_identical_plans_hash_identically(tmp_path):
    a = load_plan(write_plan(tmp_path / "a.json", condition="x", model="m"), name="baseline")
    b = load_plan(write_plan(tmp_path / "b.json", condition="x", model="m"), name="candidate")
    assert a.plan_hash == b.plan_hash


def test_report_carries_adapter_and_model_versions(tree, arms):
    baseline, candidate = arms
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=BarsDocument(source="(none supplied)"))
    text = render_text(report)
    assert baseline.plan_hash in text and candidate.plan_hash in text
    assert "version=1.2.0" in text and "model=sonnet" in text


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------


def test_deltas_are_paired_and_exclude_unsupported_and_non_held_out(tree, arms):
    baseline, candidate = arms
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=BarsDocument(source="(none supplied)"))
    scored = {p.fixture.fixture_id for p in report.scored_pairs}
    assert scored == {"doc_a", "doc_b"}  # doc_c is dev; call_retraction is an unsupported shape
    recall = report.delta("recall")
    assert recall.n_pairs == 2
    assert recall.mean_delta == pytest.approx(0.5)  # 0.5 -> 1.0 on both


def test_unsupported_fixture_is_still_reported(tree, arms):
    baseline, candidate = arms
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=BarsDocument(source="(none supplied)"))
    pair = next(p for p in report.pairs if p.fixture.fixture_id == "call_retraction")
    assert not pair.scored
    assert pair.fixture.scope.verdict is ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE
    assert "UNSUPPORTED_FOR_THIS_GOAL_SHAPE" in render_text(report)


def test_cost_delta_comes_from_trace_summary(tree, arms):
    baseline, candidate = arms
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=BarsDocument(source="(none supplied)"))
    cost = report.delta("cost_usd")
    assert cost.mean_delta == pytest.approx(0.04)
    assert report.delta("judge_calls").mean_delta == pytest.approx(2.0)


def test_missing_trace_yields_no_cost_delta_rather_than_a_zero(tmp_path, arms):
    root = tmp_path / "fx"
    root.mkdir()
    for name in ("f1", "f2"):
        write_fixture(root, name, goal_shape="point",
                      episodes=[_episode("e1", 1.0, 2.0)],
                      baseline_events=[], candidate_events=[_event("e1", 1.5)])
    (root / "splits.json").write_text(json.dumps({"train": [], "dev": [], "test": ["f1", "f2"]}))
    baseline, candidate = arms
    report = run_differential(fixtures_root=root, baseline=baseline, candidate=candidate,
                              bars=BarsDocument(source="(none supplied)"))
    cost = report.delta("cost_usd")
    assert cost.mean_delta is None and cost.n_pairs == 0
    assert "—" in render_text(report)


def test_confidence_interval_is_deterministic_and_reported(tree, arms):
    baseline, candidate = arms
    kw = dict(fixtures_root=tree, baseline=baseline, candidate=candidate,
              bars=BarsDocument(source="(none supplied)"))
    a = run_differential(**kw).delta("recall")
    b = run_differential(**kw).delta("recall")
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert a.ci_low is not None and a.ci_low <= a.mean_delta <= a.ci_high


def test_single_pair_gets_no_interval_and_says_so(tmp_path, arms):
    root = tmp_path / "fx"
    root.mkdir()
    write_fixture(root, "only", goal_shape="point", episodes=[_episode("e1", 1.0, 2.0)],
                  baseline_events=[], candidate_events=[_event("e1", 1.5)])
    (root / "splits.json").write_text(json.dumps({"train": [], "dev": [], "test": ["only"]}))
    baseline, candidate = arms
    report = run_differential(fixtures_root=root, baseline=baseline, candidate=candidate,
                              bars=BarsDocument(source="(none supplied)"))
    d = report.delta("recall")
    assert d.n_pairs == 1 and d.ci_low is None
    assert "anecdote" in d.note


def test_scoring_uses_the_shared_5a_matcher_semantics(tmp_path, arms):
    """A duplicate fire inside one episode is a false fire, not extra recall (Viability §5a)."""
    root = tmp_path / "fx"
    root.mkdir()
    for name in ("f1", "f2"):
        write_fixture(root, name, goal_shape="episode", episodes=[_episode("e1", 10.0, 12.0)],
                      baseline_events=[_event("e1", 11.0)],
                      candidate_events=[_event("e1", 11.0), _event("e1", 11.5, seq=1)])
    (root / "splits.json").write_text(json.dumps({"train": [], "dev": [], "test": ["f1", "f2"]}))
    baseline, candidate = arms
    report = run_differential(fixtures_root=root, baseline=baseline, candidate=candidate,
                              bars=BarsDocument(source="(none supplied)"))
    assert report.delta("recall").mean_delta == pytest.approx(0.0)
    assert report.delta("false_fires").mean_delta == pytest.approx(1.0)
    assert report.delta("precision").mean_delta == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# bars: client-supplied, never invented
# ---------------------------------------------------------------------------


def test_without_bars_no_pass_fail_is_invented(tree, arms):
    baseline, candidate = arms
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=load_bars(None))
    assert report.verdict is OverallVerdict.NOT_ASSESSED
    assert report.bar_results == ()
    assert "does not invent" in report.verdict_reason
    assert "no pass/fail invented" in render_text(report)


def test_supplied_bars_produce_a_verdict(tree, arms, tmp_path):
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({
        "declared_by": "client-team",
        "bars": [{"metric": "recall_delta", "direction": "min", "value": 0.2}],
    }))
    baseline, candidate = arms
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=load_bars(bars_path))
    assert report.verdict is OverallVerdict.PASS
    assert report.bar_results[0].observed == pytest.approx(0.5)


def test_failing_bar_is_reported_as_fail(tree, arms, tmp_path):
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({
        "bars": [{"metric": "cost_usd_delta", "direction": "max", "value": 0.0}],
    }))
    baseline, candidate = arms
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=load_bars(bars_path))
    assert report.verdict is OverallVerdict.FAIL
    assert report.bar_results[0].passed is False


def test_bar_requiring_ci_to_exclude_zero_fails_when_it_does_not(tmp_path, arms):
    root = tmp_path / "fx"
    root.mkdir()
    # candidate helps on one fixture and hurts on another -> CI straddles zero
    write_fixture(root, "f1", goal_shape="point", episodes=[_episode("e1", 1.0, 2.0)],
                  baseline_events=[], candidate_events=[_event("e1", 1.5)])
    write_fixture(root, "f2", goal_shape="point", episodes=[_episode("e1", 1.0, 2.0)],
                  baseline_events=[_event("e1", 1.5)], candidate_events=[])
    (root / "splits.json").write_text(json.dumps({"train": [], "dev": [], "test": ["f1", "f2"]}))
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({"bars": [
        {"metric": "recall_delta", "direction": "min", "value": -1.0,
         "require_ci_excludes_zero": True}
    ]}))
    baseline, candidate = arms
    report = run_differential(fixtures_root=root, baseline=baseline, candidate=candidate,
                              bars=load_bars(bars_path))
    assert report.verdict is OverallVerdict.FAIL
    assert "includes zero" in report.bar_results[0].note


def test_bars_are_not_applied_without_an_explicit_held_out_split(tmp_path, arms):
    root = tmp_path / "fx"
    root.mkdir()
    for name in ("f1", "f2"):
        write_fixture(root, name, goal_shape="point", episodes=[_episode("e1", 1.0, 2.0)],
                      baseline_events=[], candidate_events=[_event("e1", 1.5)])
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({"bars": [
        {"metric": "recall_delta", "direction": "min", "value": 0.0}
    ]}))
    baseline, candidate = arms
    report = run_differential(fixtures_root=root, baseline=baseline, candidate=candidate,
                              bars=load_bars(bars_path))
    assert report.verdict is OverallVerdict.NOT_ASSESSED
    assert "tune-to-green" in report.verdict_reason
    # ...but the numbers are still there
    assert report.delta("recall").mean_delta == pytest.approx(1.0)


def test_bars_not_applied_when_every_held_out_fixture_is_unsupported(tmp_path, arms):
    root = tmp_path / "fx"
    root.mkdir()
    write_fixture(root, "sup", goal_shape="supersession", episodes=[_episode("e1", 1.0, 2.0)],
                  baseline_events=[], candidate_events=[_event("e1", 1.5)])
    (root / "splits.json").write_text(json.dumps({"train": [], "dev": [], "test": ["sup"]}))
    bars_path = tmp_path / "bars.json"
    bars_path.write_text(json.dumps({"bars": [
        {"metric": "recall_delta", "direction": "min", "value": 0.0}
    ]}))
    baseline, candidate = arms
    report = run_differential(fixtures_root=root, baseline=baseline, candidate=candidate,
                              bars=load_bars(bars_path))
    assert report.verdict is OverallVerdict.NOT_ASSESSED
    assert "scorable goal shape" in report.verdict_reason


def test_unknown_bar_metric_is_rejected():
    with pytest.raises(EvalInputError):
        Bar(metric="vibes", direction="min", value=1.0)


# ---------------------------------------------------------------------------
# no tune-to-green
# ---------------------------------------------------------------------------


def test_registration_hash_covers_plans_split_thresholds_and_bars(tree, arms, tmp_path):
    baseline, candidate = arms
    bars = load_bars(None)
    first = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate, bars=bars)
    h = first.registration.registration_hash

    # same config -> same hash
    again = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate, bars=bars)
    assert again.registration.registration_hash == h

    # moved threshold -> different hash
    tuned = load_plan(write_plan(tmp_path / "tuned.json", condition="invoice total visible via OCR",
                                 model="sonnet", thresholds={"confidence": 0.4, "dwell_s": 1.0}),
                      name="candidate")
    moved = run_differential(fixtures_root=tree, baseline=baseline, candidate=tuned, bars=bars)
    assert moved.registration.registration_hash != h


def test_config_drift_after_bars_declared_refuses_to_score(tree, arms, tmp_path):
    baseline, candidate = arms
    bars = load_bars(None)
    declared = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                                bars=bars).registration.registration_hash

    tuned = load_plan(write_plan(tmp_path / "tuned.json", condition="invoice total visible via OCR",
                                 model="sonnet", thresholds={"confidence": 0.1}),
                      name="candidate")
    with pytest.raises(ConfigDriftError) as exc:
        run_differential(fixtures_root=tree, baseline=baseline, candidate=tuned, bars=bars,
                         expected_registration_hash=declared)
    assert "REFUSING TO SCORE" in str(exc.value)


def test_matching_registration_hash_scores_normally(tree, arms):
    baseline, candidate = arms
    bars = load_bars(None)
    h = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                         bars=bars).registration.registration_hash
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=bars, expected_registration_hash=h)
    assert report.delta("recall").n_pairs == 2


def test_bootstrap_seed_is_derived_from_the_registration_not_wall_time(tree, arms):
    baseline, candidate = arms
    reg = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                           bars=load_bars(None)).registration
    assert reg.seed == int(reg.registration_hash[:12], 16)


# ---------------------------------------------------------------------------
# split reporting
# ---------------------------------------------------------------------------


def test_exact_held_out_split_is_reported(tree, arms):
    baseline, candidate = arms
    report = run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                              bars=load_bars(None))
    split = report.registration.split
    assert split.declared
    assert split.held_out == ("call_retraction", "doc_a", "doc_b")
    assert split.exploratory == ("doc_c",)
    assert "held-out (test)" in render_text(report)


def test_split_naming_an_unknown_fixture_is_an_error(tree, arms):
    (tree / "splits.json").write_text(json.dumps({"train": [], "dev": [], "test": ["ghost"]}))
    baseline, candidate = arms
    with pytest.raises(EvalInputError):
        run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                         bars=load_bars(None))


def test_overlapping_splits_are_rejected_by_the_shared_loader(tree, arms):
    from percept.eval import SplitError

    (tree / "splits.json").write_text(
        json.dumps({"train": [], "dev": ["doc_a"], "test": ["doc_a"]})
    )
    baseline, candidate = arms
    with pytest.raises(SplitError):
        run_differential(fixtures_root=tree, baseline=baseline, candidate=candidate,
                         bars=load_bars(None))


# ---------------------------------------------------------------------------
# loading / CLI
# ---------------------------------------------------------------------------


def test_fixture_without_episodes_is_rejected(tmp_path):
    root = tmp_path / "fx"
    (root / "empty").mkdir(parents=True)
    (root / "empty" / "ground_truth.json").write_text(json.dumps({"goal_shape": "point"}))
    with pytest.raises(EvalInputError):
        load_fixtures(root)


def test_plan_without_goals_is_rejected(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"goals": []}))
    with pytest.raises(EvalInputError):
        load_plan(p, name="baseline")


def test_yaml_plan_loads_when_pyyaml_is_available(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "plan.yaml"
    p.write_text("goals:\n  - id: g1\n    condition: door is open\n    say: heads up\n")
    assert load_plan(p, name="baseline").plan_hash


def test_cli_reports_without_bars_and_exits_zero(tree, arms, tmp_path, capsys):
    from percept.eval.differential import main

    out = tmp_path / "report.json"
    code = main([
        "--baseline", arms[0].plan_path, "--candidate", arms[1].plan_path,
        "--fixtures", str(tree), "--json", str(out),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "VERDICT: NOT_ASSESSED" in printed
    payload = json.loads(out.read_text())
    assert payload["schema"] == "percept.eval.differential/1"
    assert payload["pre_registration"]["registration_hash"]
    assert len(payload["fixtures"]) == 4


def test_cli_exits_one_when_a_client_bar_fails(tree, arms, tmp_path):
    from percept.eval.differential import main

    bars = tmp_path / "bars.json"
    bars.write_text(json.dumps({"bars": [
        {"metric": "recall_delta", "direction": "min", "value": 0.9}
    ]}))
    code = main([
        "--baseline", arms[0].plan_path, "--candidate", arms[1].plan_path,
        "--fixtures", str(tree), "--bars", str(bars),
    ])
    assert code == 1


def test_cli_exits_two_when_it_refuses(tree, arms, capsys):
    from percept.eval.differential import main

    code = main([
        "--baseline", arms[0].plan_path, "--candidate", arms[1].plan_path,
        "--fixtures", str(tree), "--expect-registration-hash", "deadbeef",
    ])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
