"""— the telemetry -> fixtures closed loop.

A synthetic ``.ptrace`` is built in-process (zero spend, via the real :class:`TraceRecorder` —
the exact same recorder production runs use) and round-tripped through
:mod:`percept.eval.telemetry_fixtures` into the derivation harness's EXISTING fixture format. The
bar: the derivation harness's own loader (:func:`percept.eval.differential.load_fixtures`, the same
function :mod:`percept.eval.derivation` scores a ladder against) must actually load what this
pipeline writes, and the numbers it produces via the harness's own matcher
(:func:`percept.harness.golden_eval.score`) must be the numbers a human labelling correct/false-fire/
miss would expect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from percept.eval.differential import (
    ArmSpec,
    _load_events,
    load_fixtures,
)
from percept.eval.telemetry_fixtures import (
    ANNOTATION_VERSION,
    DEFAULT_ARM_DIR,
    TelemetryFixtureError,
    extract_cases,
    ingest_trace,
    load_labels,
    write_fixture_cases,
)
from percept.harness.golden_eval import score
from percept.trace_bundle import BundlePolicy, TraceBundle
from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder

RUN_ID = "run-l2-abc123"


def _build_recorder(run_id: str = RUN_ID) -> TraceRecorder:
    """A realistic per-goal decision sequence: one correct fire, one wrong fire, one missed silent.

    Mirrors exactly what the runtime worker/agent emit (see percept/tracing.py's docstring) — same
    stage names, same default trace_id derivation (``f"{source}@{source_ts}"`` when no explicit
    trace_id is given), so a judgment.event/source.frame/incident.created triple at the same
    source_ts correlates by trace_id exactly as it does in production.
    """
    manifest = RunTraceManifest.create(
        run_id=run_id, source="camera", driver="perceive_judged",
        policy=TracePolicy(level="full"),
        goals=[], gate_config={"threshold": 0.6},
    )
    rec = TraceRecorder(manifest)

    # g1 @ ts=1.0 — a DELIVERED fire a human will label "correct".
    rec.record("source.frame", source_ts=1.0, goal_id="g1")
    rec.record("judge.started", source_ts=1.0, goal_id="g1")
    rec.record("judge.finished", source_ts=1.0, goal_id="g1", satisfied=True, confidence=0.92)
    rec.record("judgment.event", source_ts=1.0, goal_id="g1")  # recorder auto-stamps evidence_seq
    rec.record("gate.classified", source_ts=1.0, goal_id="g1", truth="yes", action="fire",
               reason="rising_edge", satisfied=True, confidence=0.92)
    rec.record("incident.created", source_ts=1.0, goal_id="g1", incident_id="inc-correct")

    # g1 @ ts=3.0 — a DELIVERED fire a human will label "false_fire" (wrong).
    rec.record("source.frame", source_ts=3.0, goal_id="g1")
    rec.record("judge.started", source_ts=3.0, goal_id="g1")
    rec.record("judge.finished", source_ts=3.0, goal_id="g1", satisfied=True, confidence=0.61)
    rec.record("judgment.event", source_ts=3.0, goal_id="g1")
    rec.record("gate.classified", source_ts=3.0, goal_id="g1", truth="yes", action="fire",
               reason="rising_edge", satisfied=True, confidence=0.61)
    rec.record("incident.created", source_ts=3.0, goal_id="g1", incident_id="inc-false")

    # g2 @ ts=5.0 — the Gate stayed SILENT; a human later notices this should have fired ("miss").
    rec.record("source.frame", source_ts=5.0, goal_id="g2")
    rec.record("judge.started", source_ts=5.0, goal_id="g2")
    rec.record("judge.finished", source_ts=5.0, goal_id="g2", satisfied=False, confidence=0.3)
    rec.record("gate.classified", source_ts=5.0, goal_id="g2", truth="unknown", action="silent",
               reason="low_confidence", satisfied=False, confidence=0.3)

    return rec


def _miss_span_id() -> str:
    """The default trace_id the g2/ts=5.0 silent decision above resolves to (no explicit trace_id
    was given to .record(), so it defaults to f"{source}@{source_ts}")."""
    return "camera@5.0"


def _labels_payload() -> dict[str, str]:
    return {
        "inc-correct": "correct",
        "inc-false": "false_fire",
        _miss_span_id(): "miss",
    }


def _write_bundle(tmp_path: Path, *, run_id: str = RUN_ID) -> Path:
    rec = _build_recorder(run_id)
    bundle = TraceBundle.from_recorder(rec, policy=BundlePolicy.local())
    return bundle.write(tmp_path / "run.ptrace")


def _write_labels(tmp_path: Path, payload: dict[str, str] | None = None) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload if payload is not None else _labels_payload()))
    return path


# ---------------------------------------------------------------------------
# 1. The round trip — ingest, then the DERIVATION HARNESS'S OWN loader reads it back
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_ingest_writes_one_fixture_per_goal(self, tmp_path):
        bundle_path = _write_bundle(tmp_path)
        labels_path = _write_labels(tmp_path)
        out_dir = tmp_path / "fixtures"

        written = ingest_trace(bundle_path, labels_path, out_dir)

        assert written == (f"telemetry-{RUN_ID}-g1", f"telemetry-{RUN_ID}-g2")
        for fixture_id in written:
            assert (out_dir / fixture_id / "ground_truth.json").exists()
            assert (out_dir / fixture_id / DEFAULT_ARM_DIR / "events.json").exists()
            assert (out_dir / fixture_id / DEFAULT_ARM_DIR / "trace.json").exists()

    def test_derivation_harness_loader_actually_loads_it(self, tmp_path):
        """The exact function percept.eval.derivation imports from differential.py to read a
        fixtures root — not a reimplementation of it."""
        bundle_path = _write_bundle(tmp_path)
        labels_path = _write_labels(tmp_path)
        out_dir = tmp_path / "fixtures"
        ingest_trace(bundle_path, labels_path, out_dir)

        fixtures = load_fixtures(out_dir)
        by_id = {f.fixture_id: f for f in fixtures}
        assert set(by_id) == {f"telemetry-{RUN_ID}-g1", f"telemetry-{RUN_ID}-g2"}

        g1 = by_id[f"telemetry-{RUN_ID}-g1"]
        assert g1.goal_shape == "point"
        assert g1.scope.scorable  # "point" is a SUPPORTED shape (ADR-006)
        assert {e.episode_id for e in g1.episodes} == {"inc-correct"}  # false_fire gets NO episode

        g2 = by_id[f"telemetry-{RUN_ID}-g2"]
        assert {e.episode_id for e in g2.episodes} == {_miss_span_id()}

    def test_scored_through_the_harness_matcher_matches_the_human_verdicts(self, tmp_path):
        """percept.harness.golden_eval.score — the SAME matcher percept.eval.derivation scores a
        ladder with — must turn "correct" into a match, "false_fire" into an unmatched false fire,
        and "miss" into a miss. Not asserted by this module; asserted BY the harness's own scorer."""
        bundle_path = _write_bundle(tmp_path)
        labels_path = _write_labels(tmp_path)
        out_dir = tmp_path / "fixtures"
        ingest_trace(bundle_path, labels_path, out_dir)

        fixtures = {f.fixture_id: f for f in load_fixtures(out_dir)}

        g1 = fixtures[f"telemetry-{RUN_ID}-g1"]
        g1_events = _load_events(Path(g1.path) / DEFAULT_ARM_DIR / "events.json")
        g1_result = score(g1.episodes, g1_events)
        assert g1_result.true_positives == 1
        assert g1_result.recall == pytest.approx(1.0)      # the correct fire matched
        assert g1_result.false_fires == 1                   # the false_fire matched no episode
        assert g1_result.precision == pytest.approx(0.5)    # 1 match / 2 predictions

        g2 = fixtures[f"telemetry-{RUN_ID}-g2"]
        g2_events = _load_events(Path(g2.path) / DEFAULT_ARM_DIR / "events.json")
        g2_result = score(g2.episodes, g2_events)
        assert g2_result.misses == 1
        assert g2_result.recall == pytest.approx(0.0)       # nothing was predicted for the miss
        assert g2_result.false_fires == 0

    def test_arm_spec_fixture_dir_matches_the_written_layout(self, tmp_path):
        """ArmSpec (what a ladder rung/differential arm names) defaults fixture_dir to its own
        name — confirms --arm's default lines up with that convention out of the box."""
        arm = ArmSpec(
            name=DEFAULT_ARM_DIR, plan_path="(n/a)",
            plan=__import__("percept.tracing", fromlist=["PlanIdentity"]).PlanIdentity(),
        )
        assert arm.fixture_dir == DEFAULT_ARM_DIR


# ---------------------------------------------------------------------------
# 2. Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_source_run_id_and_build_commit_are_stamped(self, tmp_path):
        bundle_path = _write_bundle(tmp_path)
        labels_path = _write_labels(tmp_path)
        out_dir = tmp_path / "fixtures"
        ingest_trace(bundle_path, labels_path, out_dir)

        gt = json.loads((out_dir / f"telemetry-{RUN_ID}-g1" / "ground_truth.json").read_text())
        prov = gt["provenance"]
        assert prov["source"] == "telemetry"
        assert prov["source_run_id"] == RUN_ID
        assert "build_commit" in prov          # present (value may be None off a dev checkout)
        assert prov["goal_id"] == "g1"
        assert set(prov["case_ids"]) == {"inc-correct", "inc-false"}

        for episode in gt["episodes"]:
            assert episode["annotation_version"] == ANNOTATION_VERSION
            assert episode["annotator_or_fixture_id"] == RUN_ID


# ---------------------------------------------------------------------------
# 3. Typed refusal — unlabeled cases are NEVER guessed
# ---------------------------------------------------------------------------


class TestRefusesUnlabeled:
    def test_delivered_incident_with_no_label_is_refused(self, tmp_path):
        bundle_path = _write_bundle(tmp_path)
        labels_path = _write_labels(tmp_path, {"inc-correct": "correct"})  # inc-false uncovered

        with pytest.raises(TelemetryFixtureError, match="inc-false"):
            ingest_trace(bundle_path, labels_path, tmp_path / "fixtures")

    def test_label_naming_an_id_the_trace_does_not_contain_is_refused(self, tmp_path):
        bundle_path = _write_bundle(tmp_path)
        payload = _labels_payload()
        payload["no-such-incident"] = "correct"
        labels_path = _write_labels(tmp_path, payload)

        with pytest.raises(TelemetryFixtureError, match="no-such-incident"):
            ingest_trace(bundle_path, labels_path, tmp_path / "fixtures")

    def test_fired_incident_cannot_be_labeled_miss(self, tmp_path):
        bundle_path = _write_bundle(tmp_path)
        payload = _labels_payload()
        payload["inc-correct"] = "miss"
        labels_path = _write_labels(tmp_path, payload)

        with pytest.raises(TelemetryFixtureError, match="inc-correct"):
            ingest_trace(bundle_path, labels_path, tmp_path / "fixtures")

    def test_silent_span_cannot_be_labeled_correct(self, tmp_path):
        bundle_path = _write_bundle(tmp_path)
        payload = _labels_payload()
        payload[_miss_span_id()] = "correct"
        labels_path = _write_labels(tmp_path, payload)

        with pytest.raises(TelemetryFixtureError, match=_miss_span_id()):
            ingest_trace(bundle_path, labels_path, tmp_path / "fixtures")

    def test_unknown_verdict_word_is_refused(self, tmp_path):
        labels_path = _write_labels(tmp_path, {"inc-correct": "probably_fine"})
        with pytest.raises(TelemetryFixtureError, match="probably_fine"):
            load_labels(labels_path)

    def test_empty_labels_file_is_refused(self, tmp_path):
        labels_path = _write_labels(tmp_path, {})
        with pytest.raises(TelemetryFixtureError):
            load_labels(labels_path)

    def test_missing_labels_file_is_refused(self, tmp_path):
        with pytest.raises(TelemetryFixtureError):
            load_labels(tmp_path / "does-not-exist.json")

    def test_extract_cases_without_labels_module_never_guesses(self, tmp_path):
        """A goal with delivered incidents and NO labels at all refuses outright."""
        rec = _build_recorder()
        with pytest.raises(TelemetryFixtureError):
            extract_cases(list(rec.records), {})


# ---------------------------------------------------------------------------
# 4. write_fixture_cases refuses a goal whose only cases are false_fire
# ---------------------------------------------------------------------------


def test_false_fire_only_goal_cannot_become_a_fixture(tmp_path):
    rec = _build_recorder()
    labels = {"inc-correct": "false_fire", "inc-false": "false_fire", _miss_span_id(): "miss"}
    cases = extract_cases(list(rec.records), labels)
    g1_only = [c for c in cases if c.goal_id == "g1"]
    with pytest.raises(TelemetryFixtureError, match="false_fire"):
        write_fixture_cases(rec.manifest, g1_only, list(rec.records), rec.summary(), tmp_path / "out")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def test_cli_main_writes_fixtures_and_exits_zero(tmp_path, capsys):
    from percept.eval.telemetry_fixtures import main as cli_main

    bundle_path = _write_bundle(tmp_path)
    labels_path = _write_labels(tmp_path)
    out_dir = tmp_path / "fixtures"

    rc = cli_main([str(bundle_path), "--labels", str(labels_path), "--out", str(out_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"telemetry-{RUN_ID}-g1" in out
    assert (out_dir / f"telemetry-{RUN_ID}-g1" / "ground_truth.json").exists()


def test_cli_main_refuses_with_exit_code_2(tmp_path, capsys):
    from percept.eval.telemetry_fixtures import main as cli_main

    bundle_path = _write_bundle(tmp_path)
    labels_path = _write_labels(tmp_path, {"inc-correct": "correct"})  # inc-false uncovered

    rc = cli_main([str(bundle_path), "--labels", str(labels_path), "--out", str(tmp_path / "fx")])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().err


def test_percept_cli_routes_ingest_trace(tmp_path, capsys):
    import percept.cli as cli

    bundle_path = _write_bundle(tmp_path)
    labels_path = _write_labels(tmp_path)
    out_dir = tmp_path / "fixtures"

    rc = cli.main(["eval", "ingest-trace", str(bundle_path),
                   "--labels", str(labels_path), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / f"telemetry-{RUN_ID}-g1" / "ground_truth.json").exists()
