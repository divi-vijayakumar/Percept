"""Card D1 — ``percept doctor``: a plan-aware tier-0 dry run that spends ZERO judge calls.

The bars these tests hold:

* the three salience statistics come from ONE pass and are bit-identical to ``motion_score``;
* each statistic is scored against ITS OWN calibrated threshold — never one shared cutoff;
* a non-admit is attributed to a NAMED rule, and "unexplained" never appears;
* zero admits is reported loudly but is not treated as a runtime alarm;
* no judge is ever constructed or called;
* the funnel is read off ``percept.tracing.TraceSummary``, not recounted.
"""

from __future__ import annotations

import json
import random

import pytest

from percept.adapters import Goal
from percept.doctor import (
    ALTERNATIVE_STAT_WOULD_ADMIT,
    EXIT_INVALID_CONFIG,
    EXIT_OK,
    EXIT_PLAN_UNOBSERVABLE,
    EXIT_SOURCE_FAILURE,
    EXIT_ZERO_COVERAGE,
    SIGNAL_UNAVAILABLE,
    SOURCE_PROFILE_SELECTION_AMBIGUOUS,
    ZERO_ADMITS_ON_FIXTURE,
    DoctorConfigError,
    main,
    render_report,
    run_doctor,
)
from percept.harness.tier0 import SALIENCE_STATS, Frame, motion_score, salience_stats

W, H = 48, 36
GRID = (8, 6)          # the gate's block partition: 6x6-pixel blocks over a 48x36 gate frame
BLOCK_PX = (W // GRID[0]) * (H // GRID[1])


def _blank() -> Frame:
    return Frame(w=W, h=H, pixels=tuple([0] * (W * H)))


def _bright_blocks(n: int, value: int = 255) -> Frame:
    """A frame with the first ``n`` top-left blocks lit — localized salience, tiny whole-frame mean."""
    px = [0] * (W * H)
    bw, bh = W // GRID[0], H // GRID[1]
    for b in range(n):
        bx = (b % GRID[0]) * bw
        by = (b // GRID[0]) * bh
        for y in range(by, by + bh):
            for x in range(bx, bx + bw):
                px[y * W + x] = value
    return Frame(w=W, h=H, pixels=tuple(px))


def _quiet_stream(n: int = 12) -> list[tuple[float, Frame]]:
    return [(float(i), _blank()) for i in range(n)]


def _localized_stream(n: int = 12, blocks: int = 4) -> list[tuple[float, Frame]]:
    """Alternating blank / four-lit-blocks: block_max and p95 clear their cutoffs, the mean does not."""
    lit = _bright_blocks(blocks)
    return [(float(i), _blank() if i % 2 == 0 else lit) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. one pass, three stats — and they match the gate's own numbers exactly
# ---------------------------------------------------------------------------


def test_salience_stats_are_bit_identical_to_motion_score():
    rnd = random.Random(20260720)
    for _ in range(8):
        a = Frame(w=W, h=H, pixels=tuple(rnd.randrange(256) for _ in range(W * H)))
        b = Frame(w=W, h=H, pixels=tuple(rnd.randrange(256) for _ in range(W * H)))
        stats = salience_stats(a, b)
        assert set(stats) == set(SALIENCE_STATS)
        for stat in SALIENCE_STATS:
            assert stats[stat] == motion_score(a, b, stat=stat), stat


def test_salience_stats_no_predecessor_is_all_zero():
    assert salience_stats(None, _blank()) == {s: 0.0 for s in SALIENCE_STATS}


def test_salience_stats_respect_a_region_like_motion_score():
    from percept.adapters import Box

    region = Box(x=0.0, y=0.0, w=0.5, h=0.5)
    a, b = _blank(), _bright_blocks(4)
    stats = salience_stats(a, b, region=region)
    for stat in SALIENCE_STATS:
        assert stats[stat] == motion_score(a, b, region=region, stat=stat), stat


def test_localized_fixture_separates_the_three_scales():
    """The whole point of three thresholds: one event, three very different value scales."""
    stats = salience_stats(_blank(), _bright_blocks(4))
    assert stats["block_max"] == pytest.approx(1.0)
    assert stats["p95"] == pytest.approx(1.0)
    assert stats["mean"] == pytest.approx(4 * BLOCK_PX / (W * H))
    assert stats["mean"] < 0.10 < stats["p95"]  # below the mean cutoff, far above the p95 cutoff


# ---------------------------------------------------------------------------
# 2. zero admits is DIAGNOSTIC — reported loud, exit 2, no runtime alarm
# ---------------------------------------------------------------------------


def test_quiet_scene_reports_zero_admits_loudly_and_exits_2():
    report = run_doctor(frames=_quiet_stream(), goals=["someone walks in"])
    assert report.exit_code == EXIT_ZERO_COVERAGE
    assert report.has(ZERO_ADMITS_ON_FIXTURE)
    finding = next(f for f in report.findings if f.code == ZERO_ADMITS_ON_FIXTURE)
    # It must NOT be presented as a failure: a still scene saving every model call is the product.
    assert finding.severity == "warn"
    assert "not automatically a failure" in finding.detail
    assert report.summary.observed == 12
    assert report.summary.admitted == 0


def test_doctor_adds_no_runtime_health_alarm():
    """Doctor is a dry run: it must not reach into the runtime health channel at all."""
    src = (__import__("pathlib").Path(__import__("percept.doctor", fromlist=["x"]).__file__)).read_text()
    assert "on_health" not in src
    assert "RuntimeHealthEvent" not in src


# ---------------------------------------------------------------------------
# 3. never one threshold for three stats
# ---------------------------------------------------------------------------


def test_each_stat_is_scored_against_its_own_effective_threshold():
    report = run_doctor(frames=_localized_stream(), goals=["a cell changes"])
    rows = {s.stat: s for s in report.goals[0].stats}
    assert set(rows) == set(SALIENCE_STATS)
    assert rows["mean"].effective_threshold == pytest.approx(0.10)
    assert rows["block_max"].effective_threshold == pytest.approx(0.20)
    assert rows["p95"].effective_threshold == pytest.approx(0.15)
    # three DISTINCT cutoffs, and the plan panel carries the same three
    assert len({r.effective_threshold for r in rows.values()}) == 3
    assert report.plan.stat_thresholds == {"mean": 0.10, "block_max": 0.20, "p95": 0.15}


def test_alternative_statistics_admit_different_evidence():
    report = run_doctor(frames=_localized_stream(n=12), goals=["a cell changes"])
    rows = {s.stat: s for s in report.goals[0].stats}
    assert rows["mean"].active is True          # deployment default salience_stat is the mean
    assert rows["mean"].admitted == 0
    assert rows["block_max"].admitted > 0
    assert rows["p95"].admitted > 0
    assert report.has(ALTERNATIVE_STAT_WOULD_ADMIT)
    assert rows["block_max"].first_admit_ts is not None


def test_block_max_profile_makes_the_localized_event_admit():
    """The same fixture, the same frames — only the selected eye changes the outcome."""
    report = run_doctor(
        frames=_localized_stream(n=12), goals=["a cell changes"],
        goal_temporality="stative", camera_mount="fixed", start_state="start_anchored",
    )
    assert report.plan.profile_name == "fixed-cam-localized"
    assert report.goals[0].active_stat == "block_max"
    assert report.goals[0].admitted > 0
    assert report.exit_code == EXIT_OK
    assert not report.has(ZERO_ADMITS_ON_FIXTURE)


# ---------------------------------------------------------------------------
# 4. WHICH RULE suppressed each non-admit
# ---------------------------------------------------------------------------


def test_non_admits_are_attributed_to_named_rules():
    report = run_doctor(frames=_quiet_stream(6), goals=["someone walks in"])
    supp = report.goals[0].suppressions
    assert supp["no_predecessor_frame"] == 1          # frame 0 has nothing to diff against
    assert supp["below_threshold"] == 5
    assert sum(supp.values()) == 6
    assert "unexplained" not in supp


def test_cadence_suppression_is_named_not_confused_with_quiet():
    goal = Goal(
        id="g0", condition="a cell changes", say="a cell changed",
        state={"salience_stat": "block_max", "cadence": {"min_judge_interval": 4.0}},
    )
    report = run_doctor(frames=_localized_stream(n=10), goals=[goal])
    supp = report.goals[0].suppressions
    assert supp.get("cadence_not_elapsed", 0) > 0
    assert report.goals[0].admitted > 0
    assert supp.get("unexplained", 0) == 0
    # the cadence-suppressed frames are NOT quiet ones — the two buckets are distinct
    assert report.goals[0].stats[1].max_value == pytest.approx(1.0)


def test_every_frame_lands_in_exactly_one_bucket():
    report = run_doctor(frames=_localized_stream(n=11), goals=["a cell changes"])
    g = report.goals[0]
    assert g.admitted + sum(g.suppressions.values()) == report.source.frames


# ---------------------------------------------------------------------------
# 5. plan / profile / signals / timebase
# ---------------------------------------------------------------------------


def test_plan_panel_answers_what_was_compiled():
    report = run_doctor(frames=_quiet_stream(3), goals=["someone walks in", "a phone appears"])
    assert report.plan.goal_ids == ("g0", "g1")
    assert report.plan.plan_hash and len(report.plan.plan_hash) == 64
    assert len(report.plan.conditions) == 2
    # Card D7 landed: doctor names the SOURCE profile, and says whether it was chosen or defaulted.
    assert report.plan.source_profile == "physical_scene"
    assert report.plan.source_profile_origin == "default"
    assert report.plan.source_profile_detail["admission_tuned"] is True


def test_admission_signal_availability_is_reported():
    report = run_doctor(frames=_quiet_stream(3), goals=["someone walks in"])
    assert report.plan.signals["motion"] is True
    assert report.plan.signals["structural"] is False
    assert report.plan.signals["semantic"] is False
    assert report.plan.signals["audio"] is False


def test_source_profile_recommendation_surfaces_in_panel_and_json(monkeypatch):
    """select_source_profile's ambiguous-source recommendation is surfaced three ways: a
    structured PlanDiagnosis field, the FINDINGS list (unchanged, pre-existing behavior), and now
    a dedicated line right under 'source profile' in the rendered PLAN panel."""
    import percept.sourceprofile as sourceprofile_mod

    def fake_select(**kwargs):
        return sourceprofile_mod.SourceProfileSelection(
            name="physical_scene", reason="default", recommend=True
        )

    monkeypatch.setattr(sourceprofile_mod, "select_source_profile", fake_select)
    report = run_doctor(frames=_quiet_stream(3), goals=["someone walks in"])

    assert report.plan.source_profile_recommended is True
    assert report.plan.to_dict()["source_profile"]["recommended_review"] is True
    assert report.has(SOURCE_PROFILE_SELECTION_AMBIGUOUS)

    rendered = render_report(report)
    assert "RECOMMENDATION" in rendered
    assert SOURCE_PROFILE_SELECTION_AMBIGUOUS in rendered


def test_goal_gated_on_an_unfed_channel_is_plan_unobservable():
    goal = Goal(id="g0", condition="the slide changes", say="the slide changed", state={"cadence": {"gate_on": "semantic"}})
    report = run_doctor(frames=_localized_stream(n=8), goals=[goal])
    assert report.exit_code == EXIT_PLAN_UNOBSERVABLE
    assert report.has(SIGNAL_UNAVAILABLE)
    assert report.goals[0].observable is False
    assert report.goals[0].suppressions.get("signal_unavailable", 0) > 0


def test_timebase_is_reported_for_an_in_memory_stream():
    report = run_doctor(frames=_quiet_stream(3), goals=["x happens"])
    assert report.source.timebase_resolved == "media"
    assert report.source.timebase_reason == "deterministic-stream"


# ---------------------------------------------------------------------------
# 6. the funnel comes from D8a — not recounted here
# ---------------------------------------------------------------------------


def test_funnel_is_the_tracing_summary():
    report = run_doctor(frames=_localized_stream(n=10), goals=["a cell changes"],
                        goal_temporality="stative", camera_mount="fixed", start_state="start_anchored")
    s = report.summary
    assert s.observed == 10
    assert s.admitted == report.goals[0].admitted
    assert s.judge_started == 0 and s.judge_completed == 0
    # savings is never fabricated: no actual call -> no measured mean input-token cost
    assert s.estimated_saved_tokens is None
    assert s.savings_text() == "—"
    assert s.estimated_baseline_calls == 10
    assert report.to_dict()["funnel"] == s.to_dict()


def test_trace_artifact_round_trips(tmp_path):
    from percept.tracing import load_trace

    out = tmp_path / "trace.json"
    report = run_doctor(frames=_localized_stream(n=8), goals=["a cell changes"],
                        trace_out=str(out))
    manifest, records, summary = load_trace(out.read_text())
    assert manifest.driver == "doctor"
    assert summary.observed == 8
    assert report.trace_path == str(out)
    assert records


# ---------------------------------------------------------------------------
# 7. ZERO JUDGE SPEND
# ---------------------------------------------------------------------------


def test_doctor_spends_no_judge_call(monkeypatch):
    """Poison the judging path: a doctor run that touches it fails loudly here."""
    import percept.agent as agent

    def _boom(*a, **kw):  # pragma: no cover - the point is that it is NEVER called
        raise AssertionError("percept doctor constructed an agent / spent a judge call")

    monkeypatch.setattr(agent.Percept, "__init__", _boom, raising=True)
    monkeypatch.setattr(agent.Percept, "perceive_judged", _boom, raising=True)

    report = run_doctor(frames=_localized_stream(n=10), goals=["a cell changes"])
    assert report.judge_calls == 0
    assert report.summary.judge_started == 0
    assert report.summary.judge_completed == 0
    assert report.summary.cost_usd == 0.0
    assert report.summary.usage_reported_calls == 0


def test_doctor_module_never_imports_a_vision_backend():
    from pathlib import Path

    import percept.doctor as doctor

    src = Path(doctor.__file__).read_text()
    for forbidden in ("percept.backends", "from .backends", "perceive_judged(", "OPENROUTER"):
        assert forbidden not in src, forbidden


# ---------------------------------------------------------------------------
# 8. exit codes + CLI
# ---------------------------------------------------------------------------


def test_source_failure_exits_3(tmp_path):
    missing = tmp_path / "nope.mp4"
    report = run_doctor(source=str(missing), goals=["someone walks in"], fps=1.0)
    assert report.exit_code == EXIT_SOURCE_FAILURE
    assert report.source.opened is False or report.source.frames == 0


def test_invalid_configuration_raises_and_the_cli_exits_5(capsys):
    with pytest.raises(DoctorConfigError):
        run_doctor(frames=_quiet_stream(2), goals=[])
    with pytest.raises(DoctorConfigError):
        run_doctor(frames=_quiet_stream(2), goals=["x"], timebase="sundial")
    with pytest.raises(DoctorConfigError):
        run_doctor(goals=["x"])  # neither source= nor frames=
    assert main(["--source", "whatever.mp4"]) == EXIT_INVALID_CONFIG
    assert "at least one --goal" in capsys.readouterr().err


def test_cli_json_is_machine_readable(tmp_path, capsys):
    missing = tmp_path / "nope.mp4"
    code = main(["--source", str(missing), "--goal", "someone walks in", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_SOURCE_FAILURE
    assert payload["exit_code"] == EXIT_SOURCE_FAILURE
    assert payload["judge_calls"] == 0
    assert payload["schema"] == "percept.doctor/1"
    assert {"source", "plan", "goals", "funnel", "findings"} <= set(payload)


def test_render_shows_all_three_stats_with_their_own_thresholds():
    report = run_doctor(frames=_localized_stream(n=8), goals=["a cell changes"])
    text = report.render()
    assert "PERCEPT DOCTOR" in text
    for stat in SALIENCE_STATS:
        assert stat in text
    assert "0.100" in text and "0.200" in text and "0.150" in text
    assert "WHICH RULE SUPPRESSED" in text
    assert "FUNNEL" in text
    assert "judge calls spent: 0" in text


def test_console_entry_point_is_declared():
    from pathlib import Path

    import percept

    pyproject = Path(percept.__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():  # pragma: no cover - installed wheel, not a source checkout
        pytest.skip("no pyproject.toml alongside the package (installed distribution)")
    assert 'percept-doctor = "percept.doctor:main"' in pyproject.read_text()


# ---------------------------------------------------------------------------
# Card X3 — the doctor's trace manifest quotes an identity that includes the cutoffs.
#
# A dry run at motion_gate=0.10 and one at 0.20 are two different configurations. If both stamped
# the same plan_hash, a client comparing two doctor traces at 2am could not prove which one they
# were looking at.
# ---------------------------------------------------------------------------


def _doctor_plan_hash(settings, tmp_path, tag: str) -> str:
    from percept.tracing import load_trace

    out = tmp_path / f"{tag}.json"
    run_doctor(frames=_localized_stream(n=8), goals=["a cell changes"],
               settings=settings, trace_out=str(out))
    manifest, _records, _summary = load_trace(out.read_text())
    return manifest.plan.plan_hash


def test_doctor_plan_hash_moves_with_the_resolved_cutoff(tmp_path):
    from dataclasses import replace

    from percept.config import Settings

    base = Settings()
    tuned = replace(base, edge=replace(base.edge, motion_gate=0.42))
    assert _doctor_plan_hash(base, tmp_path, "base") != _doctor_plan_hash(tuned, tmp_path, "tuned")
    # ...and the same settings twice is the same identity (content addressing, not a nonce)
    assert _doctor_plan_hash(base, tmp_path, "base2") == _doctor_plan_hash(base, tmp_path, "base3")


def test_doctor_plan_identity_carries_the_policy_it_hashed(tmp_path):
    from percept.tracing import load_trace

    out = tmp_path / "trace.json"
    run_doctor(frames=_localized_stream(n=8), goals=["a cell changes"], trace_out=str(out))
    manifest, _records, _summary = load_trace(out.read_text())
    policy = manifest.plan.policy_dict
    assert policy["motion_gate"] == pytest.approx(0.10)
    assert policy["motion_gate_block_max"] == pytest.approx(0.20)
    assert policy["motion_gate_p95"] == pytest.approx(0.15)
    assert policy["salience_stat"] == "mean"
    assert "require_false_first" in policy and "refractory_s" in policy
