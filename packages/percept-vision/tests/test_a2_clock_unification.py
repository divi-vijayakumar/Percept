"""A2 clock-unification acceptance (PR-A2, the SHIP-BLOCKER slice).

Every bug closed here shares one shape: a default or path that runs, looks correct, and silently does
nothing. Each test below fails on the pre-A2 tree for the stated reason:

  * A-1 — a live source ran on the MEDIA timebase, so a verification deadline stamped on the agent's
    (wall) clock could never be reached by a media-stamped frame ``t``: the verification concern never
    fired. Flipping live capture to ``timebase="wall"`` reconciles the two timelines.
  * A-2 — an authored deadline was a raw duration compared as an absolute stamp, so a goal armed at
    t=600 with a 300s deadline fired on the very next frame. Resolving the duration to an absolute
    ``deadline_at`` at ARM time fixes it.
  * A-5 — ``snapshot()`` omitted the gate's armed/dwell state, so a restart re-armed every edge and the
    first confident-yes re-fired everything already fired.
  * v1 back-compat — a restored v1 snapshot carries a legacy absolute ``deadline_s`` that cannot be
    rebased; accepting it blindly would drop the deadline silently. It must fail LOUDLY instead.
  * R6 — ``_assert_timebase`` is the CI net: it fires ONCE through ``on_health`` when a frame ``t`` and
    the core clock diverge past a per-path tolerance (a live source left on media, or a wedged pump).
"""
from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.behavior import coerce_authored_behavior
from percept.clock import FakeClock
from percept.consent import ConsentProfile
from percept.edge.tier0 import gray_frame
from percept.gate import FIRE

W, H = 8, 8
STILL = gray_frame(W, H, 0)
PERMIT = ConsentProfile(allow_computer_use=True, allowed_actions=frozenset({"press_stop"}))


# ==================================================================================================
# A-2 — a deadline armed mid-stream fires at its DURATION, not on the next frame.
# ==================================================================================================
class TestDeadlineArmTimeResolution(unittest.TestCase):
    def test_armed_at_600_with_300s_duration_fires_at_900_not_601(self):
        clk = FakeClock(600.0)  # the goal is armed mid-stream, at t=600
        a = Percept.create(vision=FakeVision(script={}), discover_plugins=False, clock=clk)
        beh = coerce_authored_behavior({"rearm": "deadline", "deadline_after_s": 300})
        self.assertEqual(beh, {"rearm": "deadline", "deadline_after_s": 300.0})  # a DURATION, coerced
        g = Goal(id="stove", condition="the stove is off", say="stove still on!")
        g.state.update(beh)
        a.add_goal(g)  # ARM: resolves deadline_at = clock()(=600) + 300 = 900
        self.assertEqual(g.state["deadline_at"], 900.0)
        self.assertEqual(a.check_deadlines(t=601.0), [])  # next frame -> NOT overdue (the A-2 bug)
        ev = a.check_deadlines(t=900.0)  # the true deadline
        self.assertEqual([e.action for e in ev], [FIRE])

    def test_legacy_deadline_s_in_state_refuses_to_arm_loudly(self):
        # A persisted/authored OLD absolute `deadline_s` has no recoverable duration base -> it must
        # NOT silently mean something new (fire on the wrong frame / never fire). Arm-time raises.
        a = Percept.create(vision=FakeVision(script={}), discover_plugins=False, clock=FakeClock(0.0))
        g = Goal(id="old", condition="c", say="s")
        g.state.update({"rearm": "deadline", "deadline_s": 300.0})
        with self.assertRaises(ValueError):
            a.add_goal(g)


# ==================================================================================================
# A-1 — a live source's verification concern FIRES end-to-end when the effect isn't confirmed in time.
# The fix is the timebase flip: run() must open a live (str) source on "wall" so frame t shares the
# agent's clock. The stand-in open_source below is faithful to the real one (source.py:137): media =>
# i/fps stamps, wall => the monotonic clock per kept frame.
# ==================================================================================================
class TestVerificationFiresThroughRun(unittest.TestCase):
    def _run_with_fake_capture(self):
        import percept.runtime.source as src_mod
        from percept.runtime import run

        clk = FakeClock(0.0)
        captured = {"timebase": None}

        async def fake_open_source(uri, *, fps=4.0, maxsize=64, reconnects=0,
                                   reconnect_backoff_s=0.5, rtsp_tcp=True, timebase="media",
                                   dropped=None, health=None):
            captured["timebase"] = timebase
            for i in range(6):
                clk.advance(1_000)  # +1s of real (wall) time per kept frame
                t = clk() if timebase == "wall" else i * 0.25  # faithful to source.py:137
                yield (t, STILL, "still")  # STILL -> no escalation -> the effect is never confirmed

        orig = src_mod.open_source
        src_mod.open_source = fake_open_source
        try:
            a = Percept.create(vision=FakeVision(script={}), discover_plugins=False, clock=clk)
            a.consent = PERMIT
            g = Goal(id="stove", condition="the stove is on", say="...", modality="act", source="camera")
            g.state["false_fire_cost"] = "irreversible"  # -> requires verification
            a.add_goal(g)
            a.actuate("stove", "press_stop", expect="the stove is off", verify_within=3.0)  # arm @ clk=0
            # deadline_at = 0 + 3 = 3.0, on the agent's (wall) clock.

            async def _collect():
                return [inc async for inc in run(a, source="webcam:0")]

            incidents = asyncio.run(_collect())
            return captured, incidents
        finally:
            src_mod.open_source = orig

    def test_live_source_opens_on_wall_and_verification_fires(self):
        captured, incidents = self._run_with_fake_capture()
        self.assertEqual(captured["timebase"], "wall")  # Step 1: live capture defaults to wall
        vid = "stove::verify::press_stop"
        self.assertTrue(
            any(getattr(i, "concern_id", None) == vid for i in incidents),
            f"verification concern {vid!r} should fire 'couldn't confirm'; got "
            f"{[getattr(i, 'concern_id', None) for i in incidents]}",
        )


# ==================================================================================================
# A-5 — the gate's armed edge survives a restart, so an already-fired goal does NOT re-fire.
# ==================================================================================================
class TestGateSnapshotSurvivesRestart(unittest.TestCase):
    def _agent(self):
        return Percept.create(vision=FakeVision(script={}), discover_plugins=False)

    def test_already_fired_goal_does_not_refire_after_restart(self):
        from percept.adapters import Verdict
        from percept.gate import SILENT

        a = self._agent()
        key = ("g", "cup")
        yes = Verdict(True, 0.95, "yes")
        self.assertEqual(a.gate.evaluate(key, yes), FIRE)     # rising edge fires, disarms the key
        self.assertEqual(a.gate.evaluate(key, yes), SILENT)   # already fired -> silent

        snap = a.snapshot()
        self.assertEqual(snap["version"], 2)
        self.assertIn("gate", snap)

        b = self._agent()  # a fresh process = a restart
        b.restore(snap)
        # The FIRST confident-yes after restart must stay SILENT — the armed=False edge was restored.
        # Without the gate snapshot the fresh gate defaults armed=True and re-fires (the A-5 bug).
        self.assertEqual(b.gate.evaluate(key, yes), SILENT)

    def test_snapshot_is_json_round_trippable(self):
        import json

        from percept.adapters import Verdict

        a = self._agent()
        a.gate.evaluate(("g", "cup"), Verdict(True, 0.95, "y"))  # disarm one key
        g = Goal(id="d", condition="c", say="s")
        g.state.update({"rearm": "deadline", "deadline_after_s": 60.0})
        a.add_goal(g)
        snap = a.snapshot()
        round_tripped = json.loads(json.dumps(snap))  # persistence backends serialize as JSON
        b = self._agent()
        b.restore(round_tripped)
        self.assertIsNotNone(b.goals.get("d").state.get("deadline_at"))  # rebased, present


# ==================================================================================================
# v1 back-compat trap — a restored v1 snapshot with a live deadline goal must NOT silently drop it.
# ==================================================================================================
class TestV1SnapshotDoesNotSilentlyDrop(unittest.TestCase):
    def test_v1_legacy_deadline_s_is_reresolved_loudly_and_still_fires(self):
        import dataclasses

        # A v1 snapshot: version 1, an absence goal carrying the OLD absolute `deadline_s`, no gate block.
        # Built from a real Goal so the serialized fields match exactly what a v1 snapshot() emitted.
        legacy_goal = Goal(id="salt", condition="salt has been added", say="waiting",
                           state={"rearm": "deadline", "deadline_s": 5.0})
        v1 = {
            "version": 1,
            "goals": [{"goal": dataclasses.asdict(legacy_goal), "status": "active"}],
            "action_log": [],
        }
        a = Percept.create(vision=FakeVision(script={}), discover_plugins=False, clock=FakeClock(1000.0))
        with self.assertWarns(UserWarning):  # LOUD — never a silent drop
            a.restore(v1)
        g = a.goals.get("salt")
        self.assertIsNotNone(g)
        # Re-resolved as a duration from restore-now (clock=1000) -> deadline_at = 1005, NOT dropped.
        self.assertEqual(g.state.get("deadline_at"), 1005.0)
        self.assertNotIn("deadline_s", g.state)  # the ambiguous legacy key is gone
        # And it actually fires when overdue (the whole point — it did NOT silently stop firing).
        self.assertEqual(a.check_deadlines(t=1004.0), [])
        self.assertEqual([e.action for e in a.check_deadlines(t=1005.0)], [FIRE])


# ==================================================================================================
# R6 — _assert_timebase is the CI net: fires ONCE on a real skew, silent on healthy queue backlog.
# ==================================================================================================
class TestAssertTimebase(unittest.TestCase):
    def _emit_n_times(self, t, clock, *, tol, path, n):
        from percept.runtime.worker import _assert_timebase

        events: list = []

        async def on_health(ev):
            events.append(ev)

        emitted: set = set()

        async def drive():
            for _ in range(n):
                await _assert_timebase(t, clock, tol=tol, path=path, emitted=emitted, on_health=on_health)

        asyncio.run(drive())
        return events

    def test_fires_exactly_once_on_million_second_skew(self):
        from percept.runtime.worker import _VIDEO_SKEW_TOL_S

        # A live source left on "media": frame t ~ 5s while the core clock is at ~1e6s (monotonic).
        events = self._emit_n_times(5.0, lambda: 1_000_000.0, tol=_VIDEO_SKEW_TOL_S, path="video", n=4)
        self.assertEqual(len(events), 1)  # ONCE, not per-frame
        self.assertEqual(events[0].kind, "timebase_skew")
        self.assertGreater(events[0].data["skew_s"], 900_000)

    def test_healthy_video_backlog_never_fires(self):
        from percept.runtime.worker import _VIDEO_SKEW_TOL_S

        # A 16s lag (video queue ~64 deep @ 4fps worst case) is UNDER tolerance -> silent.
        events = self._emit_n_times(984.0, lambda: 1_000.0, tol=_VIDEO_SKEW_TOL_S, path="video", n=4)
        self.assertEqual(events, [])

    def test_audio_backlog_under_wider_tolerance_never_fires(self):
        from percept.runtime.worker import _AUDIO_SKEW_TOL_S

        # A 64s lag (audio queue ~256 deep @ 4/s — a chunk behind a blocked pump, the SPEC-B condition)
        # is UNDER the wider audio tolerance -> silent. A single global (video) tolerance would false-fire.
        events = self._emit_n_times(936.0, lambda: 1_000.0, tol=_AUDIO_SKEW_TOL_S, path="audio", n=4)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
