"""Card D8b — COGNITION REPLAY is deterministic, and only claims what it can deliver.

The contract under test: recorded normalized verdicts + timestamps, fed back through the Gate and
the incident logic, produce BYTE-IDENTICAL decisions. Not "similar", not "the same actions modulo
timing" — the same digest, every time, including across a write/read round trip through a bundle.
"""

from __future__ import annotations

import unittest

from percept.gate import Gate
from percept.replay import (
    REPLAY_VERSION,
    ReplayGoalConfig,
    gate_config_of,
    replay_cognition,
    replay_inputs,
    replay_report_json,
)
from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder


def _recorder(**kw):
    manifest = RunTraceManifest.create(
        source="camera", driver="perceive_judged", run_id="run-123",
        policy=TracePolicy(level="full"), gate_config=gate_config_of(Gate()), **kw
    )
    return TraceRecorder(manifest)


def _gate_record(rec, *, t, goal="g1", entity=None, satisfied=True, confidence=0.9,
                 freshness=1.0, action="fire", truth="yes", hold_s=None, degraded=False):
    rec.record(
        "gate.classified", source_ts=t, goal_id=goal, entity_id=entity, truth=truth,
        action=action, satisfied=satisfied, confidence=confidence, freshness=freshness,
        hold_s=hold_s, degraded=degraded,
    )


class TestReplayIsDeterministic(unittest.TestCase):
    def _run(self):
        rec = _recorder()
        # true, true, false (re-arms), true again -> exactly TWO rising edges
        _gate_record(rec, t=1.0, satisfied=True, action="fire")
        _gate_record(rec, t=2.0, satisfied=True, action="silent")
        _gate_record(rec, t=3.0, satisfied=False, truth="no", action="silent")
        _gate_record(rec, t=4.0, satisfied=True, action="fire")
        _gate_record(rec, t=5.0, satisfied=True, confidence=0.2, truth="unknown", action="ask")
        return rec

    def test_same_trace_in_byte_identical_decisions_out(self):
        rec = self._run()
        a = replay_cognition(rec.manifest, list(rec.records))
        b = replay_cognition(rec.manifest, list(rec.records))
        self.assertEqual(a.digest, b.digest)
        self.assertEqual(
            [d.to_dict() for d in a.decisions], [d.to_dict() for d in b.decisions]
        )
        # and the canonical serialization is byte-identical too
        self.assertEqual(replay_report_json(a, indent=None), replay_report_json(b, indent=None))
        self.assertTrue(a.deterministic)
        self.assertEqual(a.replay_version, REPLAY_VERSION)

    def test_rising_edge_semantics_are_reproduced(self):
        rec = self._run()
        report = replay_cognition(rec.manifest, list(rec.records))
        actions = [d.action for d in report.decisions]
        self.assertEqual(actions, ["fire", "silent", "silent", "fire", "ask"])
        self.assertEqual([d.reason for d in report.decisions],
                         ["rising_edge", "already_fired", "not_satisfied", "rising_edge",
                          "low_confidence"])
        self.assertEqual(report.incidents, 2)          # only FIREs become incidents
        self.assertEqual(report.gate_outcomes["ask"], 1)
        self.assertEqual(report.divergences, ())       # matches what the run recorded

    def test_replay_uses_no_model_and_no_wall_clock(self):
        """A replay that consulted a live judge or time.time() would not be deterministic."""
        import time as _time

        rec = self._run()
        calls = []
        real = _time.time
        _time.time = lambda: (calls.append(1), real())[1]
        try:
            replay_cognition(rec.manifest, list(rec.records))
        finally:
            _time.time = real
        self.assertEqual(calls, [])

    def test_entity_keys_get_their_own_edge(self):
        rec = _recorder()
        _gate_record(rec, t=1.0, entity="cup-1", action="fire")
        _gate_record(rec, t=1.0, entity="cup-2", action="fire")
        _gate_record(rec, t=2.0, entity="cup-1", action="silent")
        report = replay_cognition(rec.manifest, list(rec.records))
        self.assertEqual([d.action for d in report.decisions], ["fire", "fire", "silent"])
        self.assertEqual(report.incidents, 2)


class TestReplayHonesty(unittest.TestCase):
    def test_gate_config_is_taken_from_the_manifest_not_todays_defaults(self):
        rec = _recorder()
        rec.manifest.gate_config.update({"threshold": 0.95})
        _gate_record(rec, t=1.0, confidence=0.9, action="fire")
        report = replay_cognition(rec.manifest, list(rec.records))
        self.assertTrue(report.gate_config_recorded)
        self.assertEqual(report.gate_config["threshold"], 0.95)
        # 0.9 < 0.95 -> unknown -> ASK, not a fire
        self.assertEqual([d.action for d in report.decisions], ["ask"])
        # and the disagreement with the recorded run is REPORTED, not smoothed away
        self.assertEqual(len(report.divergences), 1)
        self.assertEqual(report.divergences[0].recorded_action, "fire")
        self.assertEqual(report.divergences[0].replayed_action, "ask")

    def test_missing_gate_config_is_stated_not_assumed(self):
        manifest = RunTraceManifest.create(run_id="r", policy=TracePolicy(level="full"))
        rec = TraceRecorder(manifest)
        _gate_record(rec, t=1.0)
        report = replay_cognition(rec.manifest, list(rec.records))
        self.assertFalse(report.gate_config_recorded)

    def test_records_without_a_verdict_are_counted_unreplayable_never_guessed(self):
        manifest = RunTraceManifest.create(run_id="r", policy=TracePolicy(level="full"),
                                           gate_config=gate_config_of(Gate()))
        rec = TraceRecorder(manifest)
        rec.record("gate.classified", source_ts=1.0, goal_id="g1", truth="yes", action="fire")
        report = replay_cognition(rec.manifest, list(rec.records))
        self.assertEqual(report.inputs, 0)
        self.assertEqual(report.unreplayable, 1)
        self.assertEqual(report.decisions, ())

    def test_degraded_tick_never_fires_and_leaves_the_edge_untouched(self):
        rec = _recorder()
        _gate_record(rec, t=1.0, degraded=True, truth="unknown", action="ask")
        _gate_record(rec, t=2.0, satisfied=True, action="fire")
        report = replay_cognition(rec.manifest, list(rec.records))
        self.assertEqual([d.action for d in report.decisions], ["ask", "fire"])
        self.assertEqual(report.decisions[0].reason, "judge_degraded")
        self.assertEqual(report.incidents, 1)

    def test_hard_quadrant_escalation_is_declared_not_invented(self):
        rec = _recorder()
        _gate_record(rec, t=1.0, action="ask")  # the run delivered an escalated ASK
        plain = replay_cognition(rec.manifest, list(rec.records))
        self.assertEqual(plain.decisions[0].outward_action, "fire")   # undeclared -> honest mismatch
        self.assertEqual(len(plain.divergences), 1)
        declared = replay_cognition(
            rec.manifest, list(rec.records),
            goal_config={"g1": ReplayGoalConfig(escalates=True)},
        )
        self.assertEqual(declared.decisions[0].action, "fire")        # the GATE still fired
        self.assertEqual(declared.decisions[0].outward_action, "ask")  # outward it is an ask
        self.assertFalse(declared.decisions[0].incident)              # ...so it is not an incident
        self.assertEqual(declared.divergences, ())

    def test_refractory_from_the_recorded_config_is_honoured(self):
        rec = _recorder()
        rec.manifest.gate_config.update({"refractory_s": 30.0})
        _gate_record(rec, t=1.0, satisfied=True, action="fire")
        _gate_record(rec, t=2.0, satisfied=False, truth="no", action="silent")
        _gate_record(rec, t=3.0, satisfied=True, action="fire")
        report = replay_cognition(rec.manifest, list(rec.records))
        self.assertEqual([d.action for d in report.decisions], ["fire", "silent", "silent"])
        self.assertEqual(report.decisions[2].reason, "refractory")

    def test_replay_inputs_ignore_non_gate_stages(self):
        rec = _recorder()
        rec.record("source.frame", source_ts=1.0)
        rec.record("judge.finished", source_ts=1.0, goal_id="g1", satisfied=True, confidence=0.9)
        _gate_record(rec, t=1.0)
        self.assertEqual(len(replay_inputs(list(rec.records))), 1)


class TestReplayFromALiveRun(unittest.IsolatedAsyncioTestCase):
    async def test_a_real_perceive_judged_run_replays_to_the_same_gate_decisions(self):
        """End to end: record a real run, replay it offline, get the same outward actions."""
        from percept import Goal, Percept
        from percept.adapters import Verdict
        from percept.backends.fake import FakeVision

        vision = FakeVision(script={
            "f_yes": Verdict(satisfied=True, confidence=0.95, evidence="a spill"),
            "f_no": Verdict(satisfied=False, confidence=0.95, evidence="floor is clean"),
        })
        goal = Goal(id="spill", condition="a spill on the floor", say="spill!")
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(goal)
        manifest = RunTraceManifest.create(
            run_id="live", goals=[goal], policy=TracePolicy(level="full"),
            gate_config=gate_config_of(core.gate),
        )
        rec = TraceRecorder(manifest)
        core.on_trace = rec.tracer()
        recorded_actions = []
        for t, frame in enumerate(["f_yes", "f_yes", "f_no", "f_yes"], start=1):
            for ev in await core.perceive_judged(frame, source="camera", t=float(t)):
                recorded_actions.append(ev.action)

        report = replay_cognition(rec.manifest, list(rec.records))
        replayed = [d.outward_action for d in report.decisions if d.outward_action]
        self.assertEqual(replayed, recorded_actions)
        self.assertEqual(report.divergences, ())
        self.assertEqual(report.digest, replay_cognition(rec.manifest, list(rec.records)).digest)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
