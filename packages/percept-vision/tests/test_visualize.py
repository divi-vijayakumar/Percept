"""Card D9 — the native visualizer.

What these tests hold to:

* the metrics panel is READ from :class:`percept.tracing.TraceSummary` (doctor, eval and the
  visualizer must not invent three funnels),
* a GAP shows where the decision died — an empty stage lane is rendered, not omitted,
* the per-stat counterfactual reports each statistic against ITS OWN threshold,
* **untrusted trace text is escaped** (verdict evidence is model output — the XSS regression below
  puts real markup in it),
* media embedding is opt-in and OFF by default,
* the page opens from ``file://`` with ZERO network requests, and
* the matcher is NOT reimplemented — the drift lane calls ``golden_eval.score``.
"""

from __future__ import annotations

import json
import re

import pytest

from percept.harness.golden_eval import GroundTruthEpisode
from percept.contracts import GroundTruth
from percept.tracing import (
    ADMISSION_STAGE,
    OBSERVATION_STAGE,
    STAGE_GATE_CLASSIFIED,
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    STAGE_JUDGE_USAGE,
    STAGE_SOURCE_HEALTH,
    BaselineAssumption,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
)
from percept.visualize import (
    DEFERRED_TO_ADR_006,
    VisualizeError,
    VizOptions,
    build_model,
    esc,
    load_visualizable,
    render_html,
    visualize_trace,
)

# ---------------------------------------------------------------------------
# Fixtures — recorders driven exactly the way doctor/the runtime drive them
# ---------------------------------------------------------------------------


def _manifest(**kw):
    kw.setdefault("labels", {
        "stat_thresholds": {"mean": 0.02, "block_max": 0.08, "p95": 0.05},
        "salience_stat": "mean",
    })
    kw.setdefault("source", "fixture.mp4")
    kw.setdefault("policy", TracePolicy.diagnostic())
    kw.setdefault("baseline", BaselineAssumption.every_observation(10))
    return RunTraceManifest.create(driver="doctor", **kw)


def _silent_run():
    """The real client shape: frames arrive, the source is healthy, NOTHING is admitted."""
    rec = TraceRecorder(_manifest())
    for i in range(10):
        rec.record(
            OBSERVATION_STAGE, source_ts=float(i), trace_id=f"fixture@{i}",
            escalate=False, mean=0.004, block_max=0.031, p95=0.019, threshold=0.02,
        )
    rec.close()
    return rec


def _firing_run(evidence="a person entered the frame"):
    rec = TraceRecorder(_manifest())
    for i in range(6):
        admitted = i in (2, 4)
        rec.record(
            OBSERVATION_STAGE, source_ts=float(i), trace_id=f"fixture@{i}",
            escalate=admitted, mean=0.09 if admitted else 0.004,
            block_max=0.2 if admitted else 0.03, p95=0.15 if admitted else 0.02, threshold=0.02,
        )
        if not admitted:
            continue
        tid = f"fixture@{i}"
        rec.record(ADMISSION_STAGE, source_ts=float(i), trace_id=tid)
        rec.record(STAGE_JUDGE_STARTED, source_ts=float(i), trace_id=tid, goal_id="g1")
        rec.record(STAGE_JUDGE_FINISHED, source_ts=float(i), trace_id=tid, goal_id="g1",
                   satisfied=True, confidence=0.91, evidence=evidence)
        rec.record(STAGE_JUDGE_USAGE, source_ts=float(i), trace_id=tid, goal_id="g1",
                   input_tokens=1200, output_tokens=90, cost_usd=0.004)
        rec.record(STAGE_GATE_CLASSIFIED, source_ts=float(i), trace_id=tid, goal_id="g1",
                   truth="yes", action="fire" if i == 2 else "silent",
                   reason="fire: rising_edge" if i == 2 else "silent: already_fired_or_not_rising",
                   entity_id="person-7")
        if i == 2:
            rec.record(STAGE_INCIDENT_CREATED, source_ts=float(i), trace_id=tid, goal_id="g1",
                       incident_id="inc-1", entity_id="person-7")
    rec.close()
    return rec


def _render(rec, **kw):
    return render_html(rec.manifest, list(rec.records), rec.summary(), **kw)


# ---------------------------------------------------------------------------
# 1. Numbers come from TraceSummary — never recomputed
# ---------------------------------------------------------------------------


def test_metrics_panel_is_read_from_trace_summary():
    rec = _firing_run()
    summary = rec.summary()
    page = _render(rec)
    assert f"{summary.observed}" in page
    assert f"${summary.cost_usd:.4f}" in page
    assert f"{summary.watch_duration_s:.2f}s" in page
    assert esc(summary.savings_text(), limit=400) in page


def test_model_does_not_recount_the_funnel():
    rec = _firing_run()
    summary = rec.summary()
    model = build_model(rec.manifest, list(rec.records), summary)
    assert model.summary is summary  # the SAME object doctor and eval print


# ---------------------------------------------------------------------------
# 2. A gap shows where the decision died
# ---------------------------------------------------------------------------


def test_every_stage_lane_is_drawn_even_when_empty():
    model = build_model(_silent_run().manifest, list(_silent_run().records), _silent_run().summary())
    keys = [lane.key for lane in model.lanes]
    for expected in ("source", "health", "admission", "judge", "gate", "incidents", "context", "cost"):
        assert expected in keys, f"stage lane {expected} must be drawn even when empty"


def test_silent_run_says_the_decision_died_at_admission():
    rec = _silent_run()
    model = build_model(rec.manifest, list(rec.records), rec.summary())
    admission = next(lane for lane in model.lanes if lane.key == "admission")
    assert "DECISION DIED HERE" in admission.note
    judge = next(lane for lane in model.lanes if lane.key == "judge")
    assert judge.items == ()
    assert "no judge call was ever made" in judge.note
    page = _render(rec)
    assert "No gate decision was ever recorded." in page


def test_health_lane_latches_an_unrecovered_degradation():
    rec = TraceRecorder(_manifest())
    rec.record(OBSERVATION_STAGE, source_ts=0.0, trace_id="fixture@0", escalate=False, mean=0.0)
    rec.record(STAGE_SOURCE_HEALTH, source_ts=1.0, state="degraded", kind="capture_plan_unmet")
    rec.record(OBSERVATION_STAGE, source_ts=9.0, trace_id="fixture@9", escalate=False, mean=0.0)
    rec.close()
    model = build_model(rec.manifest, list(rec.records), rec.summary())
    health = next(lane for lane in model.lanes if lane.key == "health")
    assert len(health.items) == 1
    assert "never recovered" in health.items[0].label
    assert health.items[0].t1 == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# 3. The per-stat counterfactual — "here is the knob"
# ---------------------------------------------------------------------------


def test_counterfactual_scores_each_stat_against_its_own_threshold():
    rec = _silent_run()
    model = build_model(rec.manifest, list(rec.records), rec.summary())
    rows = {r.stat: r for r in model.counterfactual}
    assert set(rows) == {"mean", "block_max", "p95"}
    assert rows["mean"].threshold == 0.02 and rows["block_max"].threshold == 0.08
    # every stat sat under HALF its own cutoff — the real client screen fixture
    for row in rows.values():
        assert row.would_admit == 0
        assert row.ratio is not None and row.ratio < 0.5
        assert "never close" in row.verdict
    assert rows["mean"].active is True


def test_counterfactual_reports_an_alternative_stat_that_would_have_admitted():
    rec = TraceRecorder(_manifest())
    for i in range(4):
        rec.record(OBSERVATION_STAGE, source_ts=float(i), trace_id=f"fixture@{i}",
                   escalate=False, mean=0.005, block_max=0.12, p95=0.01)
    rec.close()
    model = build_model(rec.manifest, list(rec.records), rec.summary())
    rows = {r.stat: r for r in model.counterfactual}
    assert rows["mean"].would_admit == 0
    assert rows["block_max"].would_admit == 4  # 0.12 >= its own 0.08 cutoff
    assert "would admit 4/4" in rows["block_max"].verdict


def test_counterfactual_never_invents_a_threshold_it_was_not_given():
    manifest = RunTraceManifest.create(source="x", driver="doctor", policy=TracePolicy.diagnostic())
    rec = TraceRecorder(manifest)
    rec.record(OBSERVATION_STAGE, source_ts=0.0, trace_id="x@0", escalate=False, mean=0.3)
    rec.close()
    model = build_model(rec.manifest, list(rec.records), rec.summary())
    row = next(r for r in model.counterfactual if r.stat == "mean")
    assert row.threshold is None
    assert row.ratio is None
    assert "not recorded" in row.verdict
    assert "not recorded" in render_html(rec.manifest, list(rec.records), rec.summary())


def test_decision_chain_carries_the_per_frame_counterfactual():
    rec = _firing_run()
    model = build_model(rec.manifest, list(rec.records), rec.summary())
    fire = next(c for c in model.chains if c.action == "fire")
    stats = {r.stat: r for r in fire.counterfactual}
    assert stats["mean"].would_admit == 1
    labels = [label for label, _ in fire.stages]
    assert labels[0].startswith("tier-0")
    assert any("VLM verdict" in x for x in labels)
    assert labels[-1] == "gate"


# ---------------------------------------------------------------------------
# 4. SECURITY — the trace is untrusted input
# ---------------------------------------------------------------------------


XSS = '<img src=x onerror="alert(1)"><script>alert(2)</script></dd></dl>"\'&'


def test_xss_in_verdict_evidence_is_escaped_everywhere():
    """A model can emit markup. Verdict evidence is model output. Regression for exactly that."""
    rec = _firing_run(evidence=XSS)
    page = _render(rec)
    assert "<script>alert(2)</script>" not in page
    assert '<img src=x onerror="' not in page   # no live attribute can be forged
    assert 'onerror="' not in page              # the quote is escaped, so the handler cannot close
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in page  # rendered as visible TEXT instead
    # exactly one <script> element on the page: our own fixed, data-free one
    assert page.count("<script>") == 1


def test_xss_in_gate_reason_and_goal_id_is_escaped():
    rec = TraceRecorder(_manifest())
    rec.record(OBSERVATION_STAGE, source_ts=0.0, trace_id="t", escalate=True, mean=0.5)
    rec.record(STAGE_GATE_CLASSIFIED, source_ts=0.0, trace_id="t", goal_id=XSS,
               action="fire", reason=XSS)
    rec.close()
    page = _render(rec)
    assert 'onerror="' not in page
    assert "<script>alert(2)</script>" not in page


def test_no_trace_text_reaches_javascript():
    """The inline script is a FIXED constant — there is no JSON payload to break out of."""
    rec = _firing_run(evidence=XSS)
    page = _render(rec)
    script = page.split("<script>")[1].split("</script>")[0]
    assert "alert(1)" not in script
    assert "fixture" not in script
    assert "person-7" not in script
    assert "JSON.parse" not in script


def test_esc_strips_control_characters_and_caps_length():
    assert "\x00" not in esc("a\x00b")
    out = esc("x" * 5000, limit=100)
    assert "truncated" in out and len(out) < 400


def test_media_is_not_embedded_by_default():
    rec = _firing_run()
    rec.manifest  # sanity
    page = render_html(
        rec.manifest, list(rec.records), rec.summary(),
        media=[("frame-001.jpg", "image/jpeg", b"\xff\xd8\xff\xe0secret-pixels")],
    )
    assert "data:image/jpeg;base64," not in page
    assert "Media embedding is OFF" in page
    assert "img-src 'none'" in page


def test_media_embedding_requires_an_explicit_opt_in():
    rec = _firing_run()
    page = render_html(
        rec.manifest, list(rec.records), rec.summary(),
        options=VizOptions(include_media=True),
        media=[("frame-001.jpg", "image/jpeg", b"\xff\xd8\xff\xe0")],
    )
    assert "data:image/jpeg;base64," in page
    assert "img-src data:" in page


def test_media_key_in_record_data_is_withheld_by_default():
    manifest = _manifest(policy=TracePolicy(level="full", include_media=True))
    rec = TraceRecorder(manifest)
    rec.record(OBSERVATION_STAGE, source_ts=0.0, trace_id="t", escalate=True,
               mean=0.5, image="RAW-CLIENT-PIXELS")
    rec.close()
    page = _render(rec)
    assert "RAW-CLIENT-PIXELS" not in page
    assert "media withheld" in page


def test_a_hostile_media_type_cannot_forge_a_data_uri():
    rec = _firing_run()
    page = render_html(
        rec.manifest, list(rec.records), rec.summary(),
        options=VizOptions(include_media=True),
        media=[("x", 'image/svg+xml";onload="alert(1)', b"<svg/>")],
    )
    assert 'onload="' not in page
    assert "data:image/svg+xml" not in page
    assert "data:application/octet-stream;base64," in page


def test_size_limit_refuses_to_write_an_enormous_page():
    rec = _firing_run()
    with pytest.raises(VisualizeError):
        render_html(rec.manifest, list(rec.records), rec.summary(),
                    options=VizOptions(max_html_bytes=500))


def test_lane_truncation_is_reported_not_silent():
    rec = _silent_run()
    model = build_model(rec.manifest, list(rec.records), rec.summary(),
                        options=VizOptions(max_lane_items=3))
    admission = next(lane for lane in model.lanes if lane.key == "admission")
    assert len(admission.items) == 3
    assert admission.truncated == 7
    page = _render(rec, options=VizOptions(max_lane_items=3))
    assert "7 items truncated" in page


# ---------------------------------------------------------------------------
# 5. file:// with ZERO network requests
# ---------------------------------------------------------------------------


_ATTR_RE = re.compile(r'\b(?:src|href|action|data|poster|srcset|background)\s*=\s*"([^"]*)"')


def test_page_makes_no_network_request_of_any_kind():
    rec = _firing_run(evidence="see https://evil.example/steal for details")
    page = render_html(
        rec.manifest, list(rec.records), rec.summary(),
        options=VizOptions(include_media=True),
        media=[("f.jpg", "image/jpeg", b"\xff\xd8")],
    )
    for value in _ATTR_RE.findall(page):
        assert value.startswith("data:") or value.startswith("#") or value == "", (
            f"remote reference in an attribute: {value!r}"
        )
    assert "<link" not in page
    assert "@import" not in page
    assert "url(http" not in page
    assert "<script src" not in page
    assert "fetch(" not in page and "XMLHttpRequest" not in page and "WebSocket" not in page
    # the untrusted URI still appears — as ESCAPED TEXT, never as a fetchable attribute
    assert "https://evil.example/steal" in page


def test_source_uri_is_text_only_and_never_auto_fetched():
    manifest = _manifest(source="https://client.example/private/accounting.mp4")
    rec = TraceRecorder(manifest)
    rec.record(OBSERVATION_STAGE, source_ts=0.0, trace_id="t", escalate=False, mean=0.0)
    rec.close()
    page = _render(rec)
    assert "never fetched" in page
    assert 'src="https://client.example' not in page


def test_csp_is_strict_and_pins_the_script_by_hash():
    rec = _firing_run()
    page = _render(rec)
    csp = re.search(r'Content-Security-Policy" content="([^"]+)"', page).group(1)
    assert "default-src 'none'" in csp
    assert "script-src 'sha256-" in csp
    assert "connect-src 'none'" in csp
    assert "base-uri 'none'" in csp
    assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]


def test_page_body_never_scrolls_sideways_and_lanes_scroll_inside_their_own_container():
    page = _render(_firing_run())
    assert "body{margin:0" in page and "overflow-x:hidden" in page
    assert ".scroller{overflow-x:auto" in page


def test_keyboard_navigation_between_firings_is_wired():
    page = _render(_firing_run())
    assert "ArrowRight" in page and "ArrowLeft" in page
    assert 'data-firing="d0"' in page
    assert 'data-chain-for="d0"' in page


# ---------------------------------------------------------------------------
# 6. Ground-truth drift — the §5a matcher is CALLED, not reimplemented
# ---------------------------------------------------------------------------


def _episode(eid, start, eligible, end, lateness=0.0):
    return GroundTruthEpisode(
        episode_id=eid,
        ground_truth=GroundTruth(
            event_start_ts=start, decision_eligible_ts=eligible,
            annotation_version="eligibility-rubric-v1", annotator_or_fixture_id="fixture",
        ),
        episode_end_ts=end, allowed_lateness_s=lateness,
    )


def test_drift_lane_uses_golden_eval_score():
    rec = _firing_run()  # a single FIRE at source_ts == 2.0
    episodes = [_episode("g1", 1.0, 1.5, 3.0), _episode("ep-missed", 4.5, 5.0, 5.5)]
    model = build_model(rec.manifest, list(rec.records), rec.summary(), golden=episodes)
    assert model.drift.matched_episode_ids == ("g1",)
    assert model.drift.missed_episode_ids == ("ep-missed",)
    assert model.drift.recall == pytest.approx(0.5)
    drift = next(lane for lane in model.lanes if lane.key == "drift")
    kinds = {i.kind for i in drift.items}
    assert kinds == {"match", "miss"}
    assert "recall 0.50" in drift.note


def test_drift_episodes_load_from_json(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps([{
        "episode_id": "g1", "event_start_ts": 1.0, "decision_eligible_ts": 1.5,
        "episode_end_ts": 3.0, "annotation_version": "eligibility-rubric-v1",
        "annotator_or_fixture_id": "fixture",
    }]), encoding="utf-8")
    rec = _firing_run()
    model = build_model(rec.manifest, list(rec.records), rec.summary(), golden=str(path))
    assert model.drift.matched_episode_ids == ("g1",)


def test_malformed_ground_truth_is_a_typed_error_not_a_silent_empty_lane():
    from percept.visualize import load_golden_episodes

    with pytest.raises(VisualizeError):
        load_golden_episodes([{"episode_id": "x"}])


# ---------------------------------------------------------------------------
# 7. Relations stay deferred to ADR-006
# ---------------------------------------------------------------------------


def test_context_lane_carries_only_what_the_trace_actually_has():
    rec = _firing_run()
    model = build_model(rec.manifest, list(rec.records), rec.summary())
    ctx = next(lane for lane in model.lanes if lane.key == "context")
    assert [i.label for i in ctx.items] == ["entity person-7"]
    assert "DEFERRED to ADR-006" in ctx.note


def test_no_lane_promises_relational_semantics_the_runtime_cannot_deliver():
    rec = _firing_run()
    model = build_model(rec.manifest, list(rec.records), rec.summary())
    lane_titles = " ".join(lane.title for lane in model.lanes).lower()
    for concept in DEFERRED_TO_ADR_006:
        assert concept not in lane_titles


# ---------------------------------------------------------------------------
# 8. Loading: a local trace JSON and a portable .ptrace bundle
# ---------------------------------------------------------------------------


def test_visualize_a_local_trace_json(tmp_path):
    rec = _firing_run()
    trace = tmp_path / "run.json"
    rec.write_json(trace)
    out = visualize_trace(trace, tmp_path / "viz.html")
    assert out.exists()
    assert "percept trace — where the decision died" in out.read_text(encoding="utf-8")


def test_visualize_a_ptrace_bundle(tmp_path):
    from percept.trace_bundle import TraceBundle

    rec = _firing_run()
    bundle = TraceBundle.from_recorder(rec)
    path = bundle.write(tmp_path / "run")
    manifest, records, summary, media = load_visualizable(path)
    assert manifest.run_id == rec.manifest.run_id
    assert media == []
    out = visualize_trace(path, tmp_path / "viz.html")
    assert f"${summary.cost_usd:.4f}" in out.read_text(encoding="utf-8")


def test_visualize_a_ptrace_archive(tmp_path):
    from percept.trace_bundle import TraceBundle

    rec = _firing_run()
    path = TraceBundle.from_recorder(rec).write_archive(tmp_path / "run")
    out = visualize_trace(path, tmp_path / "viz.html")
    assert out.exists()


def test_admission_lane_shows_the_motion_trace_and_the_threshold_rule():
    page = _render(_silent_run())
    assert "mean 0.0040 &lt; threshold 0.0200" in page


def test_loading_a_non_trace_is_a_typed_error(tmp_path):
    bad = tmp_path / "nope.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(VisualizeError):
        load_visualizable(bad)
    with pytest.raises(VisualizeError):
        load_visualizable(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# 9. The doctor wiring — one CLI, no second console-script family
# ---------------------------------------------------------------------------


def _quiet_frames(n: int = 8):
    from percept.harness.tier0 import Frame

    blank = Frame(w=48, h=36, pixels=tuple([0] * (48 * 36)))
    return [(float(i), blank) for i in range(n)]


def test_doctor_trace_carries_the_thresholds_the_counterfactual_needs():
    from percept.doctor import run_doctor

    report = run_doctor(frames=_quiet_frames(), goals=["someone walks in"])
    labels = report.manifest.labels
    assert set(labels["stat_thresholds"]) == {"mean", "block_max", "p95"}
    assert labels["salience_stat"] in ("mean", "block_max", "p95")


def test_doctor_writes_the_visualizer_and_it_agrees_with_the_terminal_funnel(tmp_path):
    from percept.doctor import run_doctor

    out = tmp_path / "viz.html"
    report = run_doctor(frames=_quiet_frames(), goals=["someone walks in"],
                        visualize_out=str(out))
    assert report.visualize_path == str(out)
    page = out.read_text(encoding="utf-8")
    # the SAME TraceSummary the terminal panel prints
    assert f"{report.summary.observed}" in page
    assert f"{report.summary.watch_duration_s:.2f}s" in page
    # ...and the counterfactual the doctor's own per-stat table reports
    for row in report.goals[0].stats:
        assert f"{row.effective_threshold:.4f}" in page


def test_doctor_cli_can_render_an_existing_trace_without_opening_a_source(tmp_path, capsys):
    from percept.doctor import EXIT_OK, main

    trace = tmp_path / "run.json"
    _firing_run().write_json(trace)
    out = tmp_path / "viz.html"
    code = main(["--visualize-from", str(trace), "--visualize-out", str(out)])
    assert code == EXIT_OK
    assert out.exists()
    assert "visualizer written" in capsys.readouterr().out


def test_doctor_cli_rejects_visualize_from_without_an_output(capsys):
    from percept.doctor import EXIT_INVALID_CONFIG, main

    assert main(["--visualize-from", "x.json"]) == EXIT_INVALID_CONFIG
    assert "requires --visualize-out" in capsys.readouterr().err


def test_doctor_cli_still_requires_a_source_for_a_dry_run(capsys):
    from percept.doctor import EXIT_INVALID_CONFIG, main

    assert main(["--goal", "someone walks in"]) == EXIT_INVALID_CONFIG
    assert "--source is required" in capsys.readouterr().err
