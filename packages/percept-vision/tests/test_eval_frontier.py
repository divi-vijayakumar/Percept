"""Card RB-FRONTIER — the operating-frontier report (§3.2) emitted from the eval harness.

Two families of test here, and they check different kinds of promise:

1. **The math is the costing doc's math.** ``docs/reference/eval-hours-costing.md`` tabulates the
   exact Garwood/Poisson factors and the Clopper-Pearson episode counts. Those tables ARE the
   oracle — every cell is asserted. If :mod:`percept.eval.exact_ci` ever drifts from the doc, the
   published bar and the published bound stop meaning the same thing, which is the exact failure
   this suite exists to prevent.

2. **The report is honest under thin evidence.** A point estimate of 1.00 recall over 3 episodes
   must NOT produce a passing claim; an archetype with no evidence must produce a printed row, not
   an absent one; degraded time must reduce coverage without shrinking any rate denominator;
   abstentions must survive into the ledger.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from percept.contracts import GroundTruth, JudgmentEvent, JudgmentTiming
from percept.eval.differential import build_parser, main
from percept.eval.exact_ci import (
    clopper_pearson_lower,
    poisson_factor,
    poisson_upper_bound,
    regularized_incomplete_beta,
    required_exposure,
    required_positives,
)
from percept.eval.frontier import (
    CORE_GATING_ARCHETYPES,
    AbstentionLedger,
    EvidenceVerdict,
    Exposure,
    FrontierBars,
    FrontierInputError,
    FrontierVerdict,
    build_frontier,
    evidence_from_fixture,
    load_frontier_evidence,
    render_text,
)
from percept.harness.golden_eval import GroundTruthEpisode
from percept.tracing import TraceSummary

ANN = "eligibility-rubric-v1"


# ---------------------------------------------------------------------------
# 1. The exact math IS the costing doc's math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k,expected", [(0, 2.996), (1, 4.744), (2, 6.296)])
def test_poisson_factor_matches_costing_doc_table(k, expected):
    """§1.2 factor(k) = chi2_{0.95, 2(k+1)} / 2."""
    assert poisson_factor(k) == pytest.approx(expected, abs=5e-4)


def test_rule_of_three_is_the_zero_event_bound():
    """§1.1: zero events in T hours supports 'rate <= 3/T', no tighter, however clean the run felt."""
    assert poisson_factor(0) == pytest.approx(-math.log(0.05), abs=1e-9)
    assert poisson_upper_bound(0, 6.0) == pytest.approx(2.9957 / 6.0, abs=1e-4)


@pytest.mark.parametrize(
    "bar,expected",
    [(2.0, (1.5, 2.4, 3.1)), (1.0, (3.0, 4.7, 6.3)), (0.5, (6.0, 9.5, 12.6)),
     (0.2, (15.0, 23.7, 31.5))],
)
def test_required_hours_matches_costing_doc_decision_table(bar, expected):
    """§2 decision table, exact (the doc rounds 3.148 up to 3.2 in one cell; we assert exact)."""
    got = tuple(round(required_exposure(bar, k), 1) for k in (0, 1, 2))
    assert got == expected


@pytest.mark.parametrize(
    "bar,expected",
    [(0.95, (59, 93, 124)), (0.90, (29, 46, 61)), (0.85, (19, 30, 40)),
     (0.80, (14, 22, 30)), (0.75, (11, 18, 23))],
)
def test_required_episodes_matches_costing_doc_recall_table(bar, expected):
    """§3 decision table — Clopper-Pearson exact, one-sided 95% lower bound."""
    assert tuple(required_positives(bar, m) for m in (0, 1, 2)) == expected


def test_ten_for_ten_only_supports_point_seven_four():
    """§1.3's worked cautionary case: a perfect 10/10 is not a 0.90 claim."""
    assert clopper_pearson_lower(10, 10) == pytest.approx(0.7411, abs=5e-4)
    assert clopper_pearson_lower(29, 29) == pytest.approx(0.9019, abs=5e-4)
    assert clopper_pearson_lower(28, 28) == pytest.approx(0.8985, abs=5e-4)


def test_all_hits_collapses_to_the_closed_form():
    for n in (1, 3, 7, 29, 100):
        assert clopper_pearson_lower(n, n) == pytest.approx(0.05 ** (1.0 / n), abs=1e-9)


def test_zero_trials_has_no_bound_not_a_vacuous_one():
    assert clopper_pearson_lower(0, 0) is None
    assert clopper_pearson_lower(0, 5) == 0.0
    assert poisson_upper_bound(0, 0.0) is None


def test_incomplete_beta_is_a_cdf():
    assert regularized_incomplete_beta(3, 4, 0.0) == 0.0
    assert regularized_incomplete_beta(3, 4, 1.0) == 1.0
    prev = -1.0
    for i in range(11):
        v = regularized_incomplete_beta(3, 4, i / 10)
        assert v >= prev
        prev = v


def test_exact_bound_is_not_the_normal_approximation():
    """At n=29 all-hits the Wald interval returns 1.0 (zero width) — the exact bound does not."""
    exact = clopper_pearson_lower(29, 29)
    assert exact < 1.0
    assert exact == pytest.approx(0.9019, abs=1e-3)


def test_recall_min_of_one_is_unreachable_by_any_finite_evidence():
    assert required_positives(1.0, 0) is None


# ---------------------------------------------------------------------------
# 2. Evidence construction reuses golden_eval.score
# ---------------------------------------------------------------------------


def _ep(eid: str, start: float, end: float, lateness: float = 1.0) -> GroundTruthEpisode:
    return GroundTruthEpisode(
        episode_id=eid,
        ground_truth=GroundTruth(
            event_start_ts=start,
            decision_eligible_ts=start,
            annotation_version=ANN,
            annotator_or_fixture_id="fixture-gen",
        ),
        episode_end_ts=end,
        allowed_lateness_s=lateness,
    )


def _ev(key: str, ts: float, seq: int = 0, action: str = "fire") -> JudgmentEvent:
    t = JudgmentTiming(
        ingest_observed_ts=ts, admitted_ts=ts, judge_started_ts=ts,
        judge_completed_ts=ts, decision_emitted_ts=ts,
    )
    return JudgmentEvent(key=key, evidence_seq=seq, action=action, timing=t)


def _evidence(**kw):
    defaults = dict(
        fixture_id="fx",
        archetype="state",
        episodes=(_ep("e1", 10, 12), _ep("e2", 30, 32)),
        events=(_ev("e1", 11),),
        exposure=Exposure(scheduled_watch_hours=6.0),
    )
    defaults.update(kw)
    return evidence_from_fixture(**defaults)


def test_evidence_counts_come_from_the_one_matcher():
    e = _evidence(events=(_ev("e1", 11), _ev("e1", 11.5, 1), _ev("x", 100, 2)))
    assert e.episodes == 2
    assert e.fire_matches == 1
    assert e.fire_misses == 1
    assert e.duplicate_fires == 1  # second fire inside e1's window
    assert e.unmatched_false_fires == 1  # the fire at t=100 matches no window


def test_assisted_recall_counts_asks_but_fire_recall_does_not():
    e = _evidence(events=(_ev("e1", 11), _ev("e2", 31, 1, action="ask")))
    assert e.fire_matches == 1
    assert e.assisted_matches == 2
    assert e.ledger.asks == 1
    assert e.ledger.fires == 1


def test_ask_is_never_scored_as_a_false_fire():
    e = _evidence(events=(_ev("noise", 500, 0, action="ask"),))
    assert e.unmatched_false_fires == 0
    assert e.ledger.asks == 1


def test_unresolved_judge_work_and_degraded_time_come_from_trace_summary():
    summary = TraceSummary(
        run_id="r", watch_duration_s=3600.0, source_healthy_s=1800.0,
        judge_degraded=2, judge_dropped=1, gate_outcomes={"suppressed": 4},
    )
    e = _evidence(exposure=Exposure(scheduled_watch_hours=1.0), summary=summary)
    assert e.ledger.unresolved == 3
    assert e.ledger.suppressed == 4
    assert e.exposure.degraded_hours == pytest.approx(0.5)
    assert e.exposure.coverage == pytest.approx(0.5)


def test_archetype_is_required():
    with pytest.raises(FrontierInputError):
        _evidence(archetype="")


# ---------------------------------------------------------------------------
# 3. Denominator discipline
# ---------------------------------------------------------------------------


def test_degraded_time_reduces_coverage_but_not_the_rate_denominator():
    ex = Exposure(scheduled_watch_hours=6.0, degraded_hours=1.5)
    assert ex.coverage == pytest.approx(0.75)
    assert ex.healthy_watch_hours == pytest.approx(4.5)
    # The bound uses the SCHEDULED hours, not the healthy ones — a blind watch still owes its time.
    e = _evidence(exposure=ex, events=(_ev("junk", 9999),))
    report = build_frontier([e])
    ff = report.pooled.false_fire_rate
    assert ff.bound == pytest.approx(poisson_factor(1) / 6.0)
    assert ff.bound != pytest.approx(poisson_factor(1) / 4.5)


def test_degraded_cannot_exceed_scheduled():
    with pytest.raises(FrontierInputError):
        Exposure(scheduled_watch_hours=1.0, degraded_hours=2.0)


def test_per_source_hour_is_withheld_without_a_stated_concurrency():
    e = _evidence(exposure=Exposure(scheduled_watch_hours=6.0, source_hours=2.0))
    report = build_frontier([e])
    assert report.pooled.exposure.mean_active_watches is None
    assert report.pooled.false_fires_per_source_hour is None
    assert any("underspecified" in n for n in report.notes)


def test_both_denominators_published_when_concurrency_is_stated():
    e = _evidence(
        exposure=Exposure(scheduled_watch_hours=6.0, source_hours=2.0, active_watches=6.0),
        events=(_ev("e1", 11), _ev("junk", 9999)),
    )
    report = build_frontier([e])
    assert report.pooled.exposure.mean_active_watches == pytest.approx(3.0)
    assert report.pooled.false_fires_per_source_hour == pytest.approx(0.5)
    payload = report.to_dict()
    assert payload["evaluated_hours"]["scheduled_watch_hours"] == pytest.approx(6.0)
    assert payload["evaluated_hours"]["source_hours"] == pytest.approx(2.0)
    assert payload["denominator_discipline"]["primary"] == "per scheduled watch-hour"


# ---------------------------------------------------------------------------
# 4. Honesty: the bound is the claim
# ---------------------------------------------------------------------------


def _six_archetype_evidence(*, episodes_per: int, misses: int = 0, hours: float = 6.0):
    out = []
    for arch in CORE_GATING_ARCHETYPES:
        eps = tuple(_ep(f"{arch}-{i}", 10 * i, 10 * i + 2) for i in range(1, episodes_per + 1))
        hits = eps[: episodes_per - misses]
        events = tuple(_ev(e.episode_id, e.ground_truth.decision_eligible_ts + 0.5, i)
                       for i, e in enumerate(hits))
        out.append(
            evidence_from_fixture(
                fixture_id=f"fx-{arch}",
                archetype=arch,
                episodes=eps,
                events=events,
                exposure=Exposure(scheduled_watch_hours=hours, source_hours=hours,
                                  active_watches=hours),
            )
        )
    return out


def test_a_perfect_point_estimate_on_thin_evidence_does_not_pass():
    """3-for-3 is recall 1.00 as a point estimate and >= 0.368 as a claim. It must FAIL a 0.90 bar."""
    bars = FrontierBars(recall_min=0.90, false_fire_max=0.5)
    report = build_frontier(_six_archetype_evidence(episodes_per=3), bars=bars)
    row = report.archetype("state")
    assert row.recall.point == pytest.approx(1.0)
    assert row.recall.bound < 0.90
    assert row.recall.verdict is EvidenceVerdict.INSUFFICIENT
    assert report.verdict is FrontierVerdict.FAIL
    # ... and it says what it would take.
    assert row.recall.required_evidence == 29
    assert "29" in row.recall.reason


def test_sufficient_evidence_passes_and_quotes_the_bound():
    bars = FrontierBars(recall_min=0.90, false_fire_max=0.5)
    report = build_frontier(_six_archetype_evidence(episodes_per=29, hours=6.0), bars=bars)
    row = report.archetype("absence")
    assert row.recall.bound == pytest.approx(0.9019, abs=1e-3)
    assert row.recall.verdict is EvidenceVerdict.SUFFICIENT
    assert row.false_fire_rate.bound == pytest.approx(2.9957 / 6.0, abs=1e-3)
    assert row.false_fire_rate.verdict is EvidenceVerdict.SUFFICIENT
    assert report.verdict is FrontierVerdict.PASS


def test_one_false_fire_makes_the_same_bar_more_expensive_not_impossible():
    bars = FrontierBars(false_fire_max=0.5, recall_min=0.90)
    ev = _six_archetype_evidence(episodes_per=29, hours=6.0)
    dirty = evidence_from_fixture(
        fixture_id="fx-noise", archetype="state",
        episodes=(_ep("z1", 10, 12),), events=(_ev("z1", 11), _ev("junk", 99999)),
        exposure=Exposure(scheduled_watch_hours=0.0001),
    )
    report = build_frontier([*ev, dirty], bars=bars)
    row = report.archetype("state")
    assert row.unmatched_false_fires == 1
    assert row.false_fire_rate.verdict is EvidenceVerdict.INSUFFICIENT
    assert row.false_fire_rate.required_evidence == pytest.approx(9.49, abs=0.02)
    assert report.verdict is FrontierVerdict.FAIL


def test_zero_episodes_is_no_evidence_not_recall_one():
    e = _evidence(episodes=(), events=())
    report = build_frontier([e], bars=FrontierBars(recall_min=0.90))
    row = report.archetype("state")
    assert row.recall.point is None
    assert row.recall.bound is None
    assert row.recall.verdict is EvidenceVerdict.NO_EVIDENCE
    assert "1.0" in row.recall.reason
    assert report.verdict is FrontierVerdict.NOT_ASSESSED


def test_zero_scheduled_hours_yields_no_false_fire_bound():
    e = _evidence(exposure=Exposure(scheduled_watch_hours=0.0))
    report = build_frontier([e], bars=FrontierBars(false_fire_max=0.5))
    row = report.archetype("state")
    assert row.false_fire_rate.bound is None
    assert row.false_fire_rate.verdict is EvidenceVerdict.NO_EVIDENCE
    assert row.false_fire_rate.required_evidence == pytest.approx(5.99, abs=0.02)


def test_no_bars_reports_bounds_and_asserts_nothing():
    report = build_frontier(_six_archetype_evidence(episodes_per=29))
    assert report.verdict is FrontierVerdict.NOT_ASSESSED
    assert report.pooled.recall.bound is not None
    assert report.pooled.recall.verdict is EvidenceVerdict.NOT_ASSESSED


# ---------------------------------------------------------------------------
# 5. Completeness: every archetype gets a row; abstentions stay visible
# ---------------------------------------------------------------------------


def test_every_core_archetype_gets_a_row_even_with_no_evidence():
    report = build_frontier([_evidence()], bars=FrontierBars(recall_min=0.9))
    assert [a.archetype for a in report.archetypes] == list(CORE_GATING_ARCHETYPES)
    assert set(report.missing_archetypes) == set(CORE_GATING_ARCHETYPES) - {"state"}
    assert report.verdict is FrontierVerdict.NOT_ASSESSED
    assert "cherry-picked" in report.verdict_reason
    text = render_text(report)
    for arch in CORE_GATING_ARCHETYPES:
        assert arch in text
    assert "NO EVIDENCE" in text


def test_non_core_archetypes_are_reported_not_dropped():
    report = build_frontier([_evidence(archetype="document_change")])
    assert report.archetype("document_change") is not None
    assert any("Non-core" in n for n in report.notes)


def test_abstentions_survive_into_the_ledger_and_the_render():
    summary = TraceSummary(run_id="r", watch_duration_s=3600.0, source_healthy_s=3600.0,
                           judge_degraded=2, judge_dropped=1, gate_outcomes={"suppressed": 5})
    e = _evidence(
        events=(_ev("e1", 11), _ev("e2", 31, 1, action="ask"), _ev("q", 40, 2, action="silent")),
        exposure=Exposure(scheduled_watch_hours=2.0),
        summary=summary,
    )
    report = build_frontier([e])
    led = report.pooled.ledger
    assert (led.fires, led.asks, led.silents, led.suppressed, led.unresolved) == (1, 1, 1, 5, 3)
    assert led.ask_rate == pytest.approx(1 / 3)
    assert report.pooled.asks_per_scheduled_watch_hour == pytest.approx(0.5)
    assert led.abstentions == 10
    text = render_text(report)
    assert "unresolved judge work=3" in text
    assert "ABSTENTION LEDGER" in text
    assert any("never produced a decision" in n for n in report.notes)


def test_duplicate_fires_per_episode_is_reported():
    e = _evidence(events=(_ev("e1", 11), _ev("e1", 11.2, 1), _ev("e1", 11.4, 2)))
    report = build_frontier([e])
    assert report.pooled.duplicate_fires == 2
    assert report.pooled.duplicate_fires_per_episode == pytest.approx(1.0)


def test_ledger_addition_is_componentwise():
    a = AbstentionLedger(fires=1, asks=2, silents=3, suppressed=4, unresolved=5)
    assert (a + a).abstentions == 2 * a.abstentions
    assert (a + a).fires == 2


def test_render_contains_every_required_section():
    report = build_frontier(
        _six_archetype_evidence(episodes_per=29),
        bars=FrontierBars(recall_min=0.90, false_fire_max=0.5),
    )
    text = render_text(report)
    for needle in (
        "OPERATING FRONTIER",
        "EVALUATED HOURS — BOTH DENOMINATORS",
        "scheduled watch-hours",
        "source-hours",
        "fire_recall",
        "assisted_recall",
        "false_fires_per_scheduled_watch_hour",
        "coverage",
        "ASK rate",
        "duplicate FIREs per episode",
        "ABSTENTION LEDGER",
        "PER-ARCHETYPE",
        "WHAT THE EVIDENCE SUPPORTS",
        "VERDICT:",
    ):
        assert needle in text, needle


def test_json_payload_is_serialisable_and_complete():
    report = build_frontier(
        _six_archetype_evidence(episodes_per=29),
        bars=FrontierBars(recall_min=0.90, false_fire_max=0.5),
    )
    payload = json.loads(report.to_json())
    assert payload["schema"] == "percept.eval.frontier/1"
    assert len(payload["per_archetype"]) == len(CORE_GATING_ARCHETYPES)
    assert payload["bars"]["false_fire_max_per_scheduled_watch_hour"] == 0.5
    assert payload["pooled"]["bounds"]["fire_recall"]["bound"] is not None


def test_bars_reject_an_unreachable_recall_bar():
    with pytest.raises(FrontierInputError):
        FrontierBars(recall_min=1.0)
    with pytest.raises(FrontierInputError):
        FrontierBars(false_fire_max=0.0)


# ---------------------------------------------------------------------------
# 6. Wiring through the EXISTING `percept eval` surface (no second CLI)
# ---------------------------------------------------------------------------


def _write_frontier_fixture(root: Path, fixture_id: str, archetype: str, *, n: int, hours: float):
    d = root / fixture_id
    (d / "candidate").mkdir(parents=True)
    (d / "baseline").mkdir(parents=True)
    episodes = [
        {
            "episode_id": f"{fixture_id}-{i}",
            "event_start_ts": 10.0 * i,
            "decision_eligible_ts": 10.0 * i,
            "episode_end_ts": 10.0 * i + 2,
            "allowed_lateness_s": 1.0,
            "annotation_version": ANN,
            "annotator_or_fixture_id": "fixture-gen",
        }
        for i in range(1, n + 1)
    ]
    events = [
        {"key": e["episode_id"], "evidence_seq": i, "action": "fire",
         "timing": {"decision_emitted_ts": e["decision_eligible_ts"] + 0.5}}
        for i, e in enumerate(episodes)
    ]
    (d / "ground_truth.json").write_text(json.dumps({
        "goal_shape": "episode",
        "archetype": archetype,
        "exposure": {"scheduled_watch_hours": hours, "source_hours": hours, "active_watches": hours},
        "episodes": episodes,
    }))
    (d / "candidate" / "events.json").write_text(json.dumps(events))
    (d / "baseline" / "events.json").write_text(json.dumps(events[:-1] or []))


def _write_plans(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "baseline.json"
    cand = tmp_path / "candidate.json"
    base.write_text(json.dumps({"goals": [{"id": "g", "condition": "c", "say": "s"}]}))
    cand.write_text(json.dumps({"goals": [{"id": "g", "condition": "c2", "say": "s"}]}))
    return base, cand


@pytest.fixture()
def frontier_tree(tmp_path: Path) -> Path:
    root = tmp_path / "fixtures"
    root.mkdir()
    for arch in CORE_GATING_ARCHETYPES:
        _write_frontier_fixture(root, f"fx-{arch}", arch, n=29, hours=6.0)
    (root / "splits.json").write_text(json.dumps({
        "train": [], "dev": [], "test": [f"fx-{a}" for a in CORE_GATING_ARCHETYPES],
    }))
    return root


def test_load_frontier_evidence_reads_the_shared_fixture_tree(frontier_tree: Path):
    evidence = load_frontier_evidence(frontier_tree)
    assert len(evidence) == len(CORE_GATING_ARCHETYPES)
    report = build_frontier(evidence, bars=FrontierBars(recall_min=0.90, false_fire_max=0.5))
    assert report.verdict is FrontierVerdict.PASS
    assert report.pooled.exposure.scheduled_watch_hours == pytest.approx(36.0)


def test_frontier_refuses_a_fixture_without_exposure(tmp_path: Path):
    root = tmp_path / "fx"
    d = root / "one" / "candidate"
    d.mkdir(parents=True)
    (root / "one" / "ground_truth.json").write_text(json.dumps({
        "goal_shape": "episode", "archetype": "state",
        "episodes": [{
            "episode_id": "a", "event_start_ts": 1.0, "decision_eligible_ts": 1.0,
            "episode_end_ts": 2.0, "annotation_version": ANN, "annotator_or_fixture_id": "f",
        }],
    }))
    (d / "events.json").write_text("[]")
    with pytest.raises(FrontierInputError) as exc:
        load_frontier_evidence(root, held_out_only=False)
    assert "exposure" in str(exc.value)


def test_frontier_refuses_a_fixture_without_an_archetype(tmp_path: Path):
    root = tmp_path / "fx"
    d = root / "one" / "candidate"
    d.mkdir(parents=True)
    (root / "one" / "ground_truth.json").write_text(json.dumps({
        "goal_shape": "episode",
        "exposure": {"scheduled_watch_hours": 1.0},
        "episodes": [{
            "episode_id": "a", "event_start_ts": 1.0, "decision_eligible_ts": 1.0,
            "episode_end_ts": 2.0, "annotation_version": ANN, "annotator_or_fixture_id": "f",
        }],
    }))
    (d / "events.json").write_text("[]")
    with pytest.raises(FrontierInputError) as exc:
        load_frontier_evidence(root, held_out_only=False)
    assert "archetype" in str(exc.value)


def test_percept_eval_frontier_flag_is_on_the_existing_parser():
    """RB-EVAL-CLI overlap: ONE entry point. --frontier hangs off `percept eval`, not a new CLI."""
    args = build_parser().parse_args(
        ["--baseline", "b", "--candidate", "c", "--fixtures", "f", "--frontier"]
    )
    assert args.frontier is True
    assert args.frontier_arm == "candidate"


def test_percept_eval_frontier_end_to_end(tmp_path: Path, frontier_tree: Path, capsys):
    base, cand = _write_plans(tmp_path)
    bars = tmp_path / "frontier-bars.json"
    bars.write_text(json.dumps({"false_fire_max": 0.5, "recall_min": 0.90}))
    out_json = tmp_path / "frontier.json"
    rc = main([
        "--baseline", str(base), "--candidate", str(cand), "--fixtures", str(frontier_tree),
        "--frontier", "--frontier-bars", str(bars), "--frontier-json", str(out_json),
    ])
    text = capsys.readouterr().out
    assert "OPERATING FRONTIER" in text
    assert "paired differential" in text  # the differential report is still there
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["verdict"] == "PASS"
    assert payload["bars"]["recall_min"] == 0.90


def test_percept_eval_frontier_fails_loudly_on_thin_evidence(tmp_path: Path, capsys):
    root = tmp_path / "fixtures"
    root.mkdir()
    for arch in CORE_GATING_ARCHETYPES:
        _write_frontier_fixture(root, f"fx-{arch}", arch, n=3, hours=6.0)
    (root / "splits.json").write_text(json.dumps({
        "train": [], "dev": [], "test": [f"fx-{a}" for a in CORE_GATING_ARCHETYPES],
    }))
    base, cand = _write_plans(tmp_path)
    bars = tmp_path / "bars.json"
    bars.write_text(json.dumps({"recall_min": 0.90}))
    rc = main([
        "--baseline", str(base), "--candidate", str(cand), "--fixtures", str(root),
        "--frontier", "--frontier-bars", str(bars),
    ])
    text = capsys.readouterr().out
    assert rc == 1
    assert "VERDICT: FAIL" in text
    assert "would require" in text
