"""AGENT-E1 — LOOP PARITY: the bench loop and the shipped loop are NOT the same loop.

`percept.harness.tier0.GatedWatchLoop.run` is what the BENCHMARK measures (``bench_gating`` prints its
"83% cost-reduction vs ungated", and ``eval/bench_matrix.py`` bakes that into ``EDGE_KEEP = 0.17``).
`percept.runtime.worker.run` is what the PRODUCT ships. They are DIFFERENT loops:

  * GatedWatchLoop.run does  gate -> detector banks -> perceive_judged  and stops there. It collects the
    RAW ``FireEvent`` list ``perceive_judged`` returns (fires AND asks, unfiltered) into a ``LoopReport``.
    It never calls ``make_incident`` / ``snippet_from_window`` — it produces NO ``Incident`` at all.
  * worker.run has FIVE extra escalation paths GatedWatchLoop lacks — coast-admit (brief-goal fovea),
    audio-wake (off-camera onset), check_episode_boundary, check_observability, check_deadlines — plus
    paused/muted gating, and it turns each FIRE into a durable ``Incident`` (ASKs route to ``on_ask``,
    never yielded). So a shared stream can make the shipped loop JUDGE MORE (coast) and it always
    produces a DIFFERENT OUTPUT TYPE (Incident, FIRE-only) than the bench loop (FireEvent, fire+ask).

That means every gating-economics number in the repo is measured on a strictly CHEAPER loop than ships.
This file is ``percept-edge/parity/compare.py`` pointed INWARD: one synthetic stream through BOTH loops,
enumerating the exact known divergences so any FURTHER drift (e.g. GatedWatchLoop silently growing a
coast, or worker dropping a path) breaks CI. The final ``test_loops_are_at_full_parity`` is the
aspirational identical-output assertion, ``@expectedFailure`` because the loops genuinely diverge TODAY.

Filed divergence: the two run-loops share a ``MotionGate`` core (judge counts match on a plain stream)
but NOT the escalation surface or the output type. Reconciling them is a real refactor (synthesizing a
``LoopReport`` from ``RuntimeStats``, or teaching the bench the five paths); until then this pins reality.
"""
from __future__ import annotations

import asyncio
import inspect
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.agent import FireEvent
from percept.backends.fake import FakeVision
from percept.contracts import Incident
from percept.harness.tier0 import Frame, GatedWatchLoop, gray_frame
from percept.runtime.worker import RuntimeStats
from percept.runtime.worker import run as worker_run

W = H = 8
# The SAME synthetic stream bench_gating replays: mostly-still with occasional motion bursts.
MOTION_FRAMES = frozenset({5, 6, 14, 23, 24, 38})
TOTAL_FRAMES = 48


def _still() -> Frame:
    return gray_frame(W, H, 0)


def _motion() -> Frame:
    px = [0] * (W * H)
    for i in range(40):
        px[i] = 255
    return Frame(w=W, h=H, pixels=tuple(px))


def _frames() -> list[Frame]:
    return [_motion() if i in MOTION_FRAMES else _still() for i in range(TOTAL_FRAMES)]


def _fire_core() -> Percept:
    """A core whose FakeVision returns a FIRE verdict on motion frames — so the shipped loop actually
    produces Incidents (and the bench loop produces FIRE-action FireEvents) on the shared stream.
    FakeVision keys off ``str(frame)`` (the Frame repr), identical in both loops (same judge_frame)."""
    vision = FakeVision(script={
        str(_motion()): Verdict(satisfied=True, confidence=0.95, evidence="something moved"),
        str(_still()): Verdict(satisfied=False, confidence=0.95, evidence="still scene"),
    })
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="g_person", condition="a person entered the frame", say="someone's here"))
    return core


def _run_bench_loop(core: Percept):
    """The BENCH route: GatedWatchLoop.run -> LoopReport (FireEvents, no incidents)."""
    loop = GatedWatchLoop(core, list(core.goals.active()))
    stream = [(float(i), f) for i, f in enumerate(_frames())]
    report = asyncio.run(loop.run(stream, clock_start=0.0, dt=1.0))
    return loop, report


def _run_shipped_loop(core: Percept):
    """The SHIPPED route: worker.run -> yielded Incidents + RuntimeStats (judge_calls, escalations)."""
    stats = RuntimeStats()
    src = [(float(i), f, f) for i, f in enumerate(_frames())]  # (t, gate_frame, judge_frame)

    async def _collect():
        return [inc async for inc in worker_run(core, source=src, stats=stats)]

    incidents = asyncio.run(_collect())
    return stats, incidents


# --- brief-action goal on an all-STILL stream: ONLY a coast can judge it (the shipped-only path) -------
def _brief_core() -> Percept:
    vision = FakeVision(script={
        "wave": Verdict(satisfied=True, confidence=0.95, evidence="a hand waves"),
        "clear": Verdict(satisfied=False, confidence=0.95, evidence="nothing"),
    })
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="wave", condition="someone briefly waves", say="waved",
                       state={"sense": {"brief_window_s": 0.5}}))
    return core


# all-STILL: the motion gate NEVER escalates; only the brief-action coast (shipped-only) can judge here.
_QUIET = [
    (0.0, _still(), "clear"),
    (0.125, _still(), "clear"),
    (0.25, _still(), "wave"),   # quiet, in-window: gate skips, coast judges -> FIRE
    (0.375, _still(), "clear"),
    (0.5, _still(), "clear"),
]


class LoopParityTests(unittest.TestCase):
    """One shared stream through both loops; each test pins one KNOWN divergence (compare.py, inward)."""

    # -- divergence 1: OUTPUT TYPE (FireEvent vs Incident) ------------------------------------------
    def test_bench_loop_produces_fireevents_never_incidents(self):
        """GatedWatchLoop collects RAW FireEvents (fires AND asks) and never builds an Incident."""
        _loop, report = _run_bench_loop(_fire_core())
        self.assertTrue(report.fire_events, "the shared motion stream should have fired the gate")
        self.assertTrue(all(isinstance(e, FireEvent) for e in report.fire_events))
        # The LoopReport has NO incident channel at all — the bench loop lacks make_incident entirely.
        self.assertFalse(hasattr(report, "incidents"))

    def test_shipped_loop_yields_incidents_never_fireevents(self):
        """worker.run yields durable Incident objects (FIRE-only; ASKs route to on_ask, not yielded)."""
        _stats, incidents = _run_shipped_loop(_fire_core())
        self.assertTrue(incidents, "the shared motion stream should have produced incidents")
        self.assertTrue(all(isinstance(i, Incident) for i in incidents))
        self.assertFalse(any(isinstance(i, FireEvent) for i in incidents))

    # -- divergence 2: the GATE CORE is shared -> judge counts match on a plain stream --------------
    def test_judge_counts_match_on_shared_stream_the_gate_core_is_one(self):
        """The reassuring half: both loops drive the SAME MotionGate, so on a stream with no coast/
        audio/absence paths engaged the JUDGE COUNT is identical (this is what makes the 83% number
        *nearly* transferable — see test_cost_reduction_number below)."""
        _loop, report = _run_bench_loop(_fire_core())
        stats, _incidents = _run_shipped_loop(_fire_core())
        self.assertEqual(report.vision_calls, 8)           # bench_gating's headline escalation count
        self.assertEqual(stats.judge_calls, report.vision_calls)  # shipped loop judges the same 8 frames
        self.assertEqual(stats.escalations, report.escalations)

    # -- divergence 3: fire-count vs incident-count gap on the SAME stream --------------------------
    def test_fire_count_and_incident_count_diverge_on_shared_stream(self):
        """Even at equal judge counts the OUTPUTS differ: the bench loop keeps every FireEvent the
        judge returned; the shipped loop yields only FIRE-action ones as Incidents. Here every judged
        frame FIRES, so counts happen to line up — but they are DIFFERENT OBJECTS on DIFFERENT channels
        (report.fire_events vs yielded Incidents). This pins that they are not interchangeable."""
        _loop, report = _run_bench_loop(_fire_core())
        stats, incidents = _run_shipped_loop(_fire_core())
        n_fires = sum(1 for e in report.fire_events if getattr(e, "action", None) == "fire")
        self.assertEqual(len(incidents), n_fires)          # same COUNT here (all judged frames fire)...
        self.assertEqual(len(incidents), stats.fires)
        # ...but the bench loop's fire_events are FireEvents, the shipped loop's are Incidents.
        self.assertNotEqual(type(report.fire_events[0]), type(incidents[0]))

    # -- divergence 4: COAST-ADMIT is shipped-only -> a CONCRETE judge-count gap --------------------
    def test_coast_admit_makes_shipped_loop_judge_a_stream_the_bench_loop_skips(self):
        """A brief-action goal on an all-STILL stream: the shipped loop's fovea coast forces judges the
        motion gate would skip (>=1 incident); the bench loop has NO coast, so its gate never escalates
        (0 vision calls). This is one of the FIVE escalation paths the bench loop lacks, made concrete
        as a non-zero judge-count gap on identical input."""
        # bench loop
        loop = GatedWatchLoop(_brief_core(), None)
        bench_report = asyncio.run(loop.run([(t, f) for t, f, _tok in _QUIET]))
        # shipped loop
        stats = RuntimeStats()

        async def _collect():
            return [inc async for inc in worker_run(_brief_core(), source=_QUIET, stats=stats)]

        incidents = asyncio.run(_collect())

        self.assertEqual(bench_report.vision_calls, 0, "bench loop has no coast -> never judges a still stream")
        self.assertGreaterEqual(stats.judge_calls, 1, "shipped loop's coast judges the quiet in-window frame")
        self.assertGreaterEqual(len(incidents), 1)
        self.assertGreater(stats.judge_calls, bench_report.vision_calls)  # the concrete gap

    # -- divergence 5: the FIVE extra escalation paths exist in worker.run, absent in GatedWatchLoop -
    def test_five_escalation_paths_are_shipped_only(self):
        """Structural tripwire (durable past this fixture): the five escalation paths + paused/muted
        appear in worker.run's source and NOT in GatedWatchLoop.run's. If the bench loop ever silently
        grows one (or worker drops one) this breaks, forcing a review of the parity claim."""
        shipped_src = inspect.getsource(worker_run)
        bench_src = inspect.getsource(GatedWatchLoop.run)
        shipped_only = [
            "coast_admit", "audio_admit",
            "check_episode_boundary", "check_observability", "check_deadlines",
            "paused", "muted",
        ]
        for token in shipped_only:
            self.assertIn(token, shipped_src, f"{token} should be a shipped-loop escalation path")
            self.assertNotIn(token, bench_src, f"{token} unexpectedly appeared in the bench loop")
        # And the output-type asymmetry, structurally: worker builds incidents; the bench loop does not.
        self.assertIn("make_incident", inspect.getsource(inspect.getmodule(worker_run)))
        self.assertNotIn("make_incident", inspect.getsource(inspect.getmodule(GatedWatchLoop)))


class LoopFullParityAspiration(unittest.TestCase):
    """The aspirational assertion: the two loops produce IDENTICAL output on a shared stream. This
    genuinely FAILS today (the coast path judges a stream the bench loop skips, and the output types
    differ), so it is @expectedFailure — a tripwire that will XPASS (breaking CI) the day the loops are
    reconciled, at which point delete this and keep the parity as a hard assertion."""

    @unittest.expectedFailure
    def test_loops_are_at_full_parity(self):
        # brief-goal quiet stream: coast makes the shipped loop judge where the bench loop cannot.
        loop = GatedWatchLoop(_brief_core(), None)
        bench_report = asyncio.run(loop.run([(t, f) for t, f, _tok in _QUIET]))
        stats = RuntimeStats()

        async def _collect():
            return [inc async for inc in worker_run(_brief_core(), source=_QUIET, stats=stats)]

        incidents = asyncio.run(_collect())
        # Full parity would require identical judge counts AND identical firings. Neither holds today.
        self.assertEqual(stats.judge_calls, bench_report.vision_calls)   # FAILS: coast (1 vs 0)
        self.assertEqual(len(incidents), len(bench_report.fire_events))  # FAILS: 1 vs 0


if __name__ == "__main__":
    unittest.main()
