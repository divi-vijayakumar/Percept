"""Regression test for the realtime voice-watch demo bug (2026-07-11).

The demo wired `percept.run(core, ...)` WITHOUT `regate_on_goal_change=True`, then armed goals at runtime
through the conductor (the `watch` voice tool → on_speech_endpoint → apply_goal_ops). Result: the goal
landed in the shared `core` but the running gate never re-armed for it → no rising edge → NO fire, for the
whole session. The proactive premise of the SDK was silently dead.

These tests pin the contract both ways:
  * WITHOUT regate — a conductor-authored goal does NOT fire (reproduces the bug).
  * WITH regate (what `session.start()` sets for you) — it DOES fire (the fix).

Mirrors the demo's composition exactly: ONE `core` shared between `run()` and the conductor that authors
goals mid-run. A scripted `FakeConductor` emits the same `goal_ops` shape as the live conductor.
"""
from __future__ import annotations

import asyncio
import unittest

import percept
from percept.adapters import Decision, Goal, Verdict
from percept.backends.fake import FakeConductor, FakeVision
from percept.edge.tier0 import gray_frame
from percept.goals import GoalOp
from percept.runtime import run

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)


def _core_with_authoring_conductor():
    """A core whose conductor AUTHORS a standing 'fuzzy' goal on any deferred speech turn — exactly what
    the real conductor does for 'let me know if the baby gets fuzzy' (route=respond_direct + goal_ops)."""
    vision = FakeVision(script={
        "baby": Verdict(satisfied=True, confidence=0.95, evidence="the baby is fuzzy"),
        "x": Verdict(satisfied=False, confidence=0.9, evidence="nothing"),
    })
    conductor = FakeConductor(Decision(
        route="respond_direct", say="On it — I'll watch for that.",
        goal_ops=[GoalOp(op="create", goal=Goal(
            id="fuzzy", condition="the baby is fuzzy", say="Heads up — the baby's getting fuzzy."))],
    ))
    core = percept.Percept.create(vision=vision, conductor=conductor, discover_plugins=False)
    core.router = None  # force the DEFER path (the router would else fast-path); deterministic
    return core


async def _run_arming_midflight(regate: bool):
    """Start a watch on an empty core, then arm the 'fuzzy' goal mid-run through the conductor (as the
    voice `watch` tool does), and collect incidents."""
    core = _core_with_authoring_conductor()

    async def source():
        yield (0.0, STILL, "x")                       # nothing armed yet
        await core.on_speech_endpoint("let me know if the baby gets fuzzy")  # the watch command
        for i in range(1, 9):
            frame = MOVED if i % 2 else STILL         # motion → gate escalates → judge reads "baby"
            yield (float(i), frame, "baby")

    return [inc async for inc in run(core, source=source(), regate_on_goal_change=regate)]


class RealtimeWatchWiringTests(unittest.TestCase):
    def test_conductor_authored_goal_does_NOT_fire_without_regate(self):
        # THE DEMO BUG. run() compiled its spec once at t0 (empty); the mid-run authored goal is invisible.
        incidents = asyncio.run(_run_arming_midflight(regate=False))
        self.assertFalse(
            any(i.concern_id == "fuzzy" for i in incidents),
            "without regate_on_goal_change the gate must NOT re-arm — this is the bug we regression-guard",
        )

    def test_conductor_authored_goal_fires_with_regate(self):
        # THE FIX (what session.start() passes for you). Revision bump → gate recompiles → goal fires.
        incidents = asyncio.run(_run_arming_midflight(regate=True))
        self.assertTrue(
            any(i.concern_id == "fuzzy" for i in incidents),
            "with regate_on_goal_change the runtime-authored goal must arm and fire on its rising edge",
        )


if __name__ == "__main__":
    unittest.main()
