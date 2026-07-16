"""PCPT-025 — off-loop judge dispatch: ``run(judge_dispatch="async")`` + ``_JudgeScheduler``.

The inline loop awaited ``perceive_judged`` at the admit site, so one slow VLM turn stalled capture,
the gate, and every absence check (the 47s-stall class of bugs). These tests pin the scheduler's whole
contract:

  * the capture loop NEVER blocks on a slow judge (Event-gated fake vision; frames keep flowing);
  * single-flight + one latest-pending — under overload the newest admitted frame REPLACES the queued
    candidate (``judge_superseded``), and the honest budget invariant is
    ``judge_calls + judge_superseded + judge_revision_dropped == admitted escalations`` once drained;
  * revision-abort — a goal disarmed/replaced while its judgment is IN FLIGHT returns ``[]`` and the
    Gate latch / action log are untouched (asserted on internal state, not just emitted fires);
  * pending-revision-drop — a candidate admitted under revision N is DISCARDED once goals move to N+1
    (old evidence is never reinterpreted under a newly authored plan); the next frame judges under N+1;
  * fail-loud — a judge exception is held and re-raised by ``drain()`` on the capture loop (the
    existing ``runtime_error`` path); nothing launches after a failure;
  * cancel race — cancelling the active task is NOT a fatal error, launches nothing, and
    ``close(drain=False)`` returns cleanly;
  * flush-to-quiescence — completing the active task LAUNCHES the pending one, so one await is not
    enough; both results are emitted;
  * absence-fires during an in-flight judge do not double-fire their gate key (the new interleaving
    window opened by off-loop dispatch).

Explicit-yield rule (plan step 1): every test source yields with ``await asyncio.sleep(0)`` — no
correctness is encoded around incidental event-loop scheduling of "synchronous" fakes, and a blocked
judge stays blocked until the TEST releases its event.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.goals import GoalOp
from percept.harness.tier0 import Frame, gray_frame
from percept.runtime.worker import RuntimeStats, _JudgeScheduler
from percept.runtime.worker import run as worker_run

W = H = 8


def _still() -> Frame:
    return gray_frame(W, H, 0)


def _motion() -> Frame:
    px = [0] * (W * H)
    for i in range(40):
        px[i] = 255
    return Frame(w=W, h=H, pixels=tuple(px))


_YES = Verdict(satisfied=True, confidence=0.95, evidence="seen")
_NO = Verdict(satisfied=False, confidence=0.95, evidence="nothing")


class GatedVision:
    """The slow-VLM double: ``judge`` blocks on an explicit ``asyncio.Event`` until the test releases
    it (then stays open — later judges return immediately). Records every judged frame token so tests
    can assert WHICH frames were actually judged (latest-wins, revision drops)."""

    def __init__(self, script: dict | None = None, default: Verdict = _YES) -> None:
        self.release = asyncio.Event()
        self.judged: list[str] = []
        self.in_flight = 0
        self._script = script or {}
        self._default = default

    async def detect(self, frame, goals):
        return []

    async def judge(self, condition, frame, **kwargs):
        self.judged.append(str(frame))
        self.in_flight += 1
        try:
            await self.release.wait()
        finally:
            self.in_flight -= 1
        return self._script.get(str(frame), self._default)


def _core(vision, *goals: Goal) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False)
    for g in goals:
        core.add_goal(g)
    return core


def _watch_goal(gid: str = "g") -> Goal:
    return Goal(id=gid, condition="a person entered the frame", say="someone's here")


class NonBlockingCaptureTests(unittest.TestCase):
    """The headline property: perception never awaits cognition."""

    def test_capture_loop_keeps_consuming_while_judge_is_blocked(self):
        vision = GatedVision()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        N = 12
        consumed: list[tuple[int, bool]] = []  # (frame index, was a judge in flight when consumed?)

        async def _source():
            for i in range(N):
                consumed.append((i, vision.in_flight > 0))
                # alternating still/motion: every frame after the first diffs -> the gate admits
                yield (float(i), _motion() if i % 2 else _still(), f"f{i}")
                await asyncio.sleep(0)  # explicit yield — the loop must advance on its own
            vision.release.set()  # unblock only AFTER the whole stream was consumed

        async def _collect():
            return [inc async for inc in worker_run(
                core, source=_source(), stats=stats, judge_dispatch="async")]

        incidents = asyncio.run(_collect())
        self.assertEqual(len(consumed), N, "the capture loop must consume every frame")
        blocked = [i for i, in_flight in consumed if in_flight]
        self.assertGreaterEqual(
            len(blocked), N - 3,
            f"nearly all frames should be consumed WHILE a judge is blocked in flight; got {blocked}",
        )
        self.assertGreaterEqual(len(incidents), 1, "the flushed judgment must still emit its incident")
        self.assertEqual(stats.judge_calls, 2, "single-flight: first admit + the one flushed pending")

    def test_latest_pending_replacement_and_budget_invariant_under_overload(self):
        vision = GatedVision()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        admitted_ts: list[float] = []
        N = 12

        def on_health(ev):
            if ev.kind == "escalation":
                admitted_ts.append(ev.ts)
                # live scheduler-shape proof: judge_active/judge_pending ride the payload, each 0/1
                self.assertIn(ev.data["judge_active"], (0, 1))
                self.assertIn(ev.data["judge_pending"], (0, 1))

        async def _source():
            for i in range(N):
                yield (float(i), _motion() if i % 2 else _still(), f"f{i}")
                await asyncio.sleep(0)
            vision.release.set()

        async def _collect():
            return [inc async for inc in worker_run(
                core, source=_source(), stats=stats, on_health=on_health, judge_dispatch="async")]

        asyncio.run(_collect())
        self.assertGreaterEqual(stats.escalations, 4, "sanity: the stream must actually overload")
        # single-flight while blocked: exactly the first admit launched, plus the ONE surviving pending
        self.assertEqual(stats.judge_calls, 2)
        self.assertEqual(stats.judge_superseded, stats.escalations - 2,
                         "every other admitted frame must have been replaced latest-wins")
        self.assertEqual(stats.judge_revision_dropped, 0)
        # the honest budget invariant (do NOT assert judge_calls == admits under async dispatch)
        self.assertEqual(
            stats.judge_calls + stats.judge_superseded + stats.judge_revision_dropped,
            stats.escalations,
        )
        # latest-wins: the surviving pending was the LAST admitted frame, not the second
        self.assertEqual(vision.judged[-1], f"f{int(admitted_ts[-1])}")


class RevisionAbortTests(unittest.TestCase):
    """The in-flight abort inside ``perceive_judged(expected_goals_revision=…)`` itself."""

    def _assert_untouched(self, core, fires):
        self.assertEqual(fires, [], "a stale in-flight judgment must return no fires")
        # internal latch state, not just emitted fires: the rising-edge bit was never written…
        self.assertNotIn(("g", None), core.gate._armed)
        # …and no disposition/action record was appended for the aborted judgment.
        self.assertEqual(len(core.action_log), 0)

    def test_disarm_mid_flight_aborts_before_gate_mutation(self):
        async def main():
            vision = GatedVision()
            core = _core(vision, _watch_goal("g"))
            rev = core.goals.revision
            task = asyncio.create_task(
                core.perceive_judged("frame-1", "camera", t=1.0, expected_goals_revision=rev))
            for _ in range(5):  # explicit hand-off: let the call reach the blocked judge
                await asyncio.sleep(0)
            self.assertEqual(vision.judged, ["frame-1"], "the judge must be in flight")
            core.apply_goal_ops([GoalOp(op="cancel", goal_id="g")])  # revision moves mid-flight
            vision.release.set()
            self._assert_untouched(core, await task)

        asyncio.run(main())

    def test_same_id_replacement_mid_flight_aborts_too(self):
        async def main():
            vision = GatedVision()
            core = _core(vision, _watch_goal("g"))
            rev = core.goals.revision
            task = asyncio.create_task(
                core.perceive_judged("frame-1", "camera", t=1.0, expected_goals_revision=rev))
            for _ in range(5):
                await asyncio.sleep(0)
            # same-ID replace: the id survives but MEANS something else now — the in-flight verdict
            # judged the OLD condition and must not latch/fire the new goal.
            core.add_goal(Goal(id="g", condition="the door was left open", say="door!"))
            vision.release.set()
            self._assert_untouched(core, await task)

        asyncio.run(main())

    def test_matching_revision_applies_normally(self):
        """Control: with the revision unchanged the pinned call is byte-for-byte the plain path."""
        async def main():
            vision = GatedVision()
            vision.release.set()  # judge returns immediately
            core = _core(vision, _watch_goal("g"))
            fires = await core.perceive_judged(
                "frame-1", "camera", t=1.0, expected_goals_revision=core.goals.revision)
            self.assertEqual(len(fires), 1)
            self.assertEqual(fires[0].action, "fire")
            self.assertIn(("g", None), core.gate._armed)  # the latch WAS written this time

        asyncio.run(main())


class PendingRevisionDropTests(unittest.TestCase):
    def test_pending_admitted_under_old_revision_is_dropped_next_frame_judges_fresh(self):
        vision = GatedVision(script={"f2": _NO})
        core = _core(vision, _watch_goal("g"))
        stats = RuntimeStats()

        async def _source():
            yield (0.0, _still(), "f0")   # baseline — no diff, no admit
            await asyncio.sleep(0)
            yield (1.0, _motion(), "f1")  # admit -> launch (blocked) under revision N
            await asyncio.sleep(0)
            yield (2.0, _still(), "f2")   # admit -> PENDING under revision N
            await asyncio.sleep(0)
            core.add_goal(Goal(id="g2", condition="a dog appears", say="dog"))  # N -> N+1
            vision.release.set()          # active completes (aborts to []) -> pending must be dropped
            for _ in range(6):            # explicit hand-off: completion + done-callback + drop
                await asyncio.sleep(0)
            yield (3.0, _motion(), "f3")  # next frame: admits + judges under N+1
            await asyncio.sleep(0)

        async def _collect():
            return [inc async for inc in worker_run(
                core, source=_source(), stats=stats, judge_dispatch="async")]

        asyncio.run(_collect())
        self.assertEqual(stats.judge_revision_dropped, 1, "the stale pending candidate must be dropped")
        self.assertNotIn("f2", vision.judged, "old evidence must not be judged under the new plan")
        self.assertIn("f3", vision.judged, "the NEXT frame judges under the new revision")
        self.assertEqual(stats.judge_calls, 2)  # f1 + f3; f2 never spent a call
        self.assertEqual(
            stats.judge_calls + stats.judge_superseded + stats.judge_revision_dropped,
            stats.escalations,
        )


class ExplodingVision(GatedVision):
    async def judge(self, condition, frame, **kwargs):
        await super().judge(condition, frame, **kwargs)
        raise RuntimeError("judge exploded")


class FatalJudgeTests(unittest.TestCase):
    def test_fatal_exception_drops_pending_and_drain_reraises(self):
        vision = ExplodingVision()
        core = _core(vision, _watch_goal("g"))
        stats = RuntimeStats()
        health_kinds: list[str] = []

        async def _source():
            yield (0.0, _still(), "f0")
            await asyncio.sleep(0)
            yield (1.0, _motion(), "f1")  # admit -> launch (blocked)
            await asyncio.sleep(0)
            yield (2.0, _still(), "f2")   # admit -> pending
            await asyncio.sleep(0)
            vision.release.set()          # the in-flight judge now raises
            for _ in range(6):
                await asyncio.sleep(0)    # let the failure land in the done-callback
            yield (3.0, _still(), "f3")   # next iteration's drain() must re-raise
            await asyncio.sleep(0)

        async def _collect():
            return [inc async for inc in worker_run(
                core, source=_source(), stats=stats,
                on_health=lambda ev: health_kinds.append(ev.kind), judge_dispatch="async")]

        with self.assertRaises(RuntimeError):
            asyncio.run(_collect())
        self.assertEqual(stats.judge_calls, 1, "pending must never launch after a failure")
        self.assertEqual(vision.judged, ["f1"])
        self.assertIn("runtime_error", health_kinds, "the existing fail-loud health path is preserved")


class _StubCore:
    """Minimal scheduler-facing core: a revision counter + a gated ``perceive_judged``."""

    class _Goals:
        def __init__(self):
            self.revision = 0

    def __init__(self):
        self.goals = self._Goals()
        self.block = asyncio.Event()
        self.judged: list = []

    async def perceive_judged(self, frame, source, *, t=None, expected_goals_revision=None):
        self.judged.append(frame)
        await self.block.wait()
        return [f"fires-{frame}"]


class SchedulerUnitTests(unittest.TestCase):
    """Direct ``_JudgeScheduler`` races that are awkward to provoke through the full worker."""

    def test_cancel_race_is_not_fatal_and_close_returns_cleanly(self):
        async def main():
            core = _StubCore()
            stats = RuntimeStats()
            sched = _JudgeScheduler(core, "camera", stats)
            sched.submit(1.0, "f1")
            await asyncio.sleep(0)  # explicit yield: let the in-flight judgment take its first step
            self.assertEqual(core.judged, ["f1"])
            sched.submit(2.0, "f2")  # queued behind the in-flight one
            self.assertEqual((sched.active, sched.pending), (1, 1))
            sched._active.cancel()
            for _ in range(3):  # deliver the cancellation + run the done-callback
                await asyncio.sleep(0)
            self.assertIsNone(sched._fatal, "cancellation must not be recorded as a fatal error")
            self.assertEqual(sched.active, 0)
            self.assertEqual(core.judged, ["f1"], "a cancelled completion must not launch the pending")
            self.assertEqual(sched.drain(), [], "nothing to re-raise, nothing to emit")
            await sched.close(drain=False)  # returns cleanly, discards the pending
            self.assertEqual(sched.pending, 0)
            self.assertEqual(stats.judge_calls, 1)

        asyncio.run(main())

    def test_flush_reaches_quiescence_through_the_pending_launch(self):
        async def main():
            core = _StubCore()
            stats = RuntimeStats()
            sched = _JudgeScheduler(core, "camera", stats)
            sched.submit(1.0, "f1")
            sched.submit(2.0, "f2")
            core.block.set()  # both judgments may now complete
            await sched.flush()  # ONE await is not enough: completing f1 LAUNCHES f2
            self.assertEqual((sched.active, sched.pending), (0, 0))
            self.assertEqual(
                sched.drain(),
                [(1.0, ["fires-f1"]), (2.0, ["fires-f2"])],
                "both results, in submission order, stamped with their ADMISSION t",
            )
            self.assertEqual(stats.judge_calls, 2)

        asyncio.run(main())

    def test_fatal_blocks_further_submits(self):
        async def main():
            core = _StubCore()
            stats = RuntimeStats()
            sched = _JudgeScheduler(core, "camera", stats)
            sched._fatal = RuntimeError("held failure")
            sched.submit(1.0, "f1")  # ignored — never launch after a failure
            self.assertEqual(core.judged, [])
            with self.assertRaises(RuntimeError):
                sched.drain()

        asyncio.run(main())


class AbsenceInterleavingTests(unittest.TestCase):
    def test_absence_fire_during_in_flight_judge_does_not_double_fire_its_key(self):
        """The new interleaving window: absence/deadline checks keep riding the frame cadence while a
        judgment is in flight. The __overdue__ gate key must fire ONCE across those frames (rising-edge
        dedup), and the flushed judgment must not re-fire it."""
        vision = GatedVision(script={"q1": _YES}, default=_NO)
        goal = Goal(
            id="dl", condition="the package arrives", say="package!",
            state={"rearm": "deadline", "deadline_at": 2.0},
        )
        core = _core(vision, goal)
        stats = RuntimeStats()

        async def _source():
            yield (0.0, _still(), "q0")    # baseline
            await asyncio.sleep(0)
            yield (1.0, _motion(), "q1")   # admit -> judge launches and BLOCKS
            await asyncio.sleep(0)
            yield (2.5, _still(), "q2")    # deadline passed -> overdue fires (absence path, on-loop)
            await asyncio.sleep(0)
            yield (3.5, _still(), "q3")    # still overdue: the latched key must NOT re-fire
            await asyncio.sleep(0)
            yield (4.5, _still(), "q4")
            await asyncio.sleep(0)
            vision.release.set()           # now let the in-flight judgment complete

        async def _collect():
            return [inc async for inc in worker_run(
                core, source=_source(), stats=stats, judge_dispatch="async")]

        incidents = asyncio.run(_collect())
        overdue = [i for i in incidents if i.verdict.evidence == "overdue"]
        judged = [i for i in incidents if i.verdict.evidence != "overdue"]
        self.assertEqual(len(overdue), 1, "the __overdue__ key fires once across the in-flight window")
        self.assertEqual(len(judged), 1, "the flushed q1 judgment still fires its own key")
        self.assertEqual(judged[0].ts, 1.0, "incident.ts stays the ADMISSION time, not completion")


if __name__ == "__main__":
    unittest.main()
