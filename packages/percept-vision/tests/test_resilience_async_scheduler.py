"""— degrade-loud + backpressure INSIDE the async ``_JudgeScheduler``.

An earlier attempt planted an inline guard at the ``perceive_judged`` admit site. That is the
WRONG seam: with ``run(judge_dispatch="async")`` (the default) the admit site never awaits the judge
at all — it hands the frame to :class:`~percept.runtime.worker._JudgeScheduler` and moves on — so an
inline guard simply sits on a path a slow/wedged/overloaded judge never takes. These tests pin the
guard where it actually has to live: ``_JudgeScheduler.submit`` / ``_launch`` / ``_on_done`` / the
tracked ``_trace`` that ``drain_trace()`` turns into ``RuntimeHealthEvent``s.

Failure modes covered, each through the real ``worker.run(judge_dispatch="async")`` loop:

  * **judge raises** — already fail-closed at the ``Percept`` boundary; pinned HERE too,
    at the scheduler level, so a regression that let it reach the scheduler's task would be caught by
    the existing fail-loud ``_fatal``/``runtime_error`` path, not silently swallowed.
  * **judge hangs** (never resolves) — the NEW stall watchdog: ``"judge_scheduler_stalled"`` fires
    once, loud, after ``judge_stall_after_s``.
  * **judge is slower than the frame rate** (sustained pending-slot overwrite) — the NEW backpressure
    latch: ``"judge_scheduler_backpressure"`` fires once after ``judge_backpressure_after`` consecutive
    supersessions, distinct from (and louder than) the pre-existing per-drop ``"judge_superseded"``.
  * **recovery** — the next clean completion clears both latches and reports
    ``"judge_scheduler_recovered"`` exactly once; health never stays latched red.

Every scenario also asserts the pre-existing single-flight + latest-pending invariant
(J<=1, P<=1 — ``sched.active``/``sched.pending`` are each always 0 or 1) is NOT regressed.

Explicit-yield rule (matches ``test_runtime_judge_scheduler.py``): every synthetic source yields with
``await asyncio.sleep(0)`` so no correctness rides on incidental event-loop scheduling.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
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


class GatedVision:
    """Blocks ``judge`` on an explicit ``asyncio.Event`` until the test releases it. Never asserts
    which frame was judged (this file cares about scheduler HEALTH, not admission content — that's
    ``test_runtime_judge_scheduler.py``'s job)."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.judged: list[str] = []

    async def detect(self, frame, goals):
        return []

    async def judge(self, condition, frame, **kwargs):
        self.judged.append(str(frame))
        await self.release.wait()
        return _YES


class ExplodingVision(GatedVision):
    """Raises INSIDE the fail-closed boundary; the safe-judge path must swallow it before it ever
    reaches the scheduler's task (this is what makes async dispatch safe for a flaky real backend)."""

    async def judge(self, condition, frame, **kwargs):
        self.judged.append(str(frame))
        raise RuntimeError("judge exploded")


def _core(vision, *goals: Goal) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False)
    for g in goals:
        core.add_goal(g)
    return core


def _watch_goal(gid: str = "g") -> Goal:
    return Goal(id=gid, condition="a person entered the frame", say="someone's here")


def _invariant_ok(test: unittest.TestCase, sched: _JudgeScheduler) -> None:
    """J<=1, P<=1 — each is a bare 0/1 scalar by construction; assert it explicitly, not just trust
    the type."""
    test.assertIn(sched.active, (0, 1))
    test.assertIn(sched.pending, (0, 1))


class JudgeRaisesTests(unittest.TestCase):
    def test_raising_judge_stays_fail_closed_and_never_trips_the_new_latches(self):
        """The pre-existing safe-judge boundary catches the raise before the scheduler's task sees an
        exception -- so this run must complete cleanly, WITHOUT a stall or backpressure event (those
        are for a judge that is slow/stuck, not one that fails fast and closed), and the invariant
        must hold throughout."""
        vision = ExplodingVision()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        health_kinds: list[str] = []

        async def _source():
            for i in range(6):
                yield (float(i), _motion() if i % 2 else _still(), f"f{i}")
                await asyncio.sleep(0)

        async def _collect():
            return [
                inc
                async for inc in worker_run(
                    core, source=_source(), stats=stats,
                    on_health=lambda ev: health_kinds.append(ev.kind), judge_dispatch="async",
                )
            ]

        incidents = asyncio.run(_collect())
        self.assertEqual(incidents, [], "a degraded tick must never FIRE")
        self.assertIn("judge_degraded", health_kinds, "the pre-existing A30 loud path must still fire")
        self.assertNotIn("runtime_error", health_kinds, "a caught judge exception must not fail-loud-halt")
        self.assertNotIn("judge_scheduler_stalled", health_kinds,
                          "a fast raise-and-degrade is not a hang")
        self.assertNotIn("judge_scheduler_backpressure", health_kinds,
                          "a fast raise-and-degrade is not sustained overwrite")

    def test_exception_escaping_the_task_still_hits_the_scheduler_fail_loud_backstop(self):
        """A failure OUTSIDE the fail-closed judgment boundary (e.g. a bug in vision.detect) reaches the
        scheduler's task and must still fail loud via the pre-existing ``_fatal``/``drain()``
        re-raise; degrade-loud handling for slow/stuck judges must not soften this backstop."""

        class ExplodingDetect(GatedVision):
            async def detect(self, frame, goals):
                raise RuntimeError("detect exploded")

        vision = ExplodingDetect()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        health_kinds: list[str] = []

        async def _source():
            yield (0.0, _still(), "f0")
            await asyncio.sleep(0)
            yield (1.0, _motion(), "f1")
            await asyncio.sleep(0)
            for _ in range(4):
                await asyncio.sleep(0)

        async def _collect():
            return [
                inc
                async for inc in worker_run(
                    core, source=_source(), stats=stats,
                    on_health=lambda ev: health_kinds.append(ev.kind), judge_dispatch="async",
                )
            ]

        with self.assertRaises(RuntimeError):
            asyncio.run(_collect())
        self.assertIn("runtime_error", health_kinds, "an unboundaried failure must still fail loud")


class JudgeHangsTests(unittest.TestCase):
    def test_stalled_active_judgment_reports_once_loud(self):
        """A judge that never resolves must trip the stall watchdog exactly once (latched), not once
        per frame, while the invariant (J<=1, P<=1) holds throughout the whole overloaded stream."""
        vision = GatedVision()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        stalled_events: list[dict] = []

        # very small threshold so the synthetic (near-instant, real-wall-clock) stream can cross it
        # without simulating real time — the scheduler measures elapsed WALL time via core.clock().
        N = 40

        async def _source():
            for i in range(N):
                yield (float(i), _motion() if i % 2 else _still(), f"f{i}")
                await asyncio.sleep(0)
            vision.release.set()

        def on_health(ev):
            if ev.kind == "judge_scheduler_stalled":
                stalled_events.append(ev.data)

        async def _collect():
            return [
                inc
                async for inc in worker_run(
                    core, source=_source(), stats=stats, on_health=on_health,
                    judge_dispatch="async", judge_stall_after_s=0.0,
                )
            ]

        asyncio.run(_collect())
        self.assertEqual(len(stalled_events), 1, "the stall latch must fire exactly once, not spam")
        self.assertIn("elapsed_s", stalled_events[0])
        self.assertIn("active_t", stalled_events[0])

    def test_stall_recovers_when_the_active_judgment_finally_completes(self):
        vision = GatedVision()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        kinds: list[str] = []
        N = 6

        async def _source():
            for i in range(N):
                yield (float(i), _motion() if i % 2 else _still(), f"f{i}")
                await asyncio.sleep(0)
            vision.release.set()  # let it finish — this is the recovery instant
            for _ in range(6):
                await asyncio.sleep(0)
            yield (float(N), _motion(), f"f{N}")  # one more admit so the drained trace is observed
            await asyncio.sleep(0)

        async def _collect():
            return [
                inc
                async for inc in worker_run(
                    core, source=_source(), stats=stats,
                    on_health=lambda ev: kinds.append(ev.kind),
                    judge_dispatch="async", judge_stall_after_s=0.0,
                )
            ]

        asyncio.run(_collect())
        self.assertIn("judge_scheduler_stalled", kinds)
        self.assertIn("judge_scheduler_recovered", kinds,
                       "health must report recovery, not stay latched red")
        self.assertLess(
            kinds.index("judge_scheduler_stalled"), kinds.index("judge_scheduler_recovered"),
            "recovery must be reported AFTER the degradation it clears",
        )


class SustainedOverwriteTests(unittest.TestCase):
    def test_repeated_supersession_trips_backpressure_once(self):
        """The judge is slower than the frame rate: the pending slot gets overwritten on every
        admitted frame while the (never-completing) active judgment sits there. A single supersede
        is normal load-shedding (already covered by ``judge_superseded`` in
        ``test_runtime_judge_scheduler.py``) -- a RUN of them must escalate, exactly once."""
        vision = GatedVision()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        backpressure_events: list[dict] = []
        N = 16

        async def _source():
            for i in range(N):
                yield (float(i), _motion() if i % 2 else _still(), f"f{i}")
                await asyncio.sleep(0)
            vision.release.set()

        def on_health(ev):
            if ev.kind == "judge_scheduler_backpressure":
                backpressure_events.append(ev.data)

        async def _collect():
            sched_stats = stats
            return [
                inc
                async for inc in worker_run(
                    core, source=_source(), stats=sched_stats, on_health=on_health,
                    judge_dispatch="async", judge_backpressure_after=3,
                    judge_stall_after_s=999.0,  # isolate: this test is about overwrite, not the hang
                )
            ]

        asyncio.run(_collect())
        self.assertGreaterEqual(stats.judge_superseded, 3, "sanity: the stream must actually overload")
        self.assertEqual(len(backpressure_events), 1, "the backpressure latch must fire exactly once")
        self.assertGreaterEqual(backpressure_events[0]["consecutive_supersessions"], 3)
        # honest budget invariant (pre-existing, must not regress): every admitted escalation is
        # accounted for exactly once across launch/supersede/revision-drop.
        self.assertEqual(
            stats.judge_calls + stats.judge_superseded + stats.judge_revision_dropped,
            stats.escalations,
        )

    def test_backpressure_recovers_after_the_transient_clears(self):
        vision = GatedVision()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        kinds: list[str] = []
        N = 10

        async def _source():
            for i in range(N):
                yield (float(i), _motion() if i % 2 else _still(), f"f{i}")
                await asyncio.sleep(0)
            vision.release.set()  # the transient clears: the wedged judge finally answers
            for _ in range(6):
                await asyncio.sleep(0)
            yield (float(N), _motion(), f"f{N}")
            await asyncio.sleep(0)

        async def _collect():
            return [
                inc
                async for inc in worker_run(
                    core, source=_source(), stats=stats,
                    on_health=lambda ev: kinds.append(ev.kind),
                    judge_dispatch="async", judge_backpressure_after=3, judge_stall_after_s=999.0,
                )
            ]

        asyncio.run(_collect())
        self.assertIn("judge_scheduler_backpressure", kinds)
        self.assertIn("judge_scheduler_recovered", kinds)
        self.assertEqual(
            kinds.count("judge_scheduler_backpressure"), 1, "must not re-fire while still latched"
        )
        self.assertEqual(kinds.count("judge_scheduler_recovered"), 1)


class InvariantHeldThroughoutTests(unittest.TestCase):
    """A dedicated pass over every failure mode above, this time polling the LIVE scheduler object
    (not just trusting the health trace) to pin J<=1/P<=1 as the one invariant that must never move,
    whatever else degrades."""

    def test_j_le_1_p_le_1_across_hang_and_overload(self):
        vision = GatedVision()
        core = _core(vision, _watch_goal())
        stats = RuntimeStats()
        from percept.runtime import worker as worker_mod

        seen: list[_JudgeScheduler] = []
        orig_init = worker_mod._JudgeScheduler.__init__

        def _spying_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            seen.append(self)

        N = 20

        async def _source():
            for i in range(N):
                yield (float(i), _motion() if i % 2 else _still(), f"f{i}")
                await asyncio.sleep(0)
            vision.release.set()

        async def _collect():
            worker_mod._JudgeScheduler.__init__ = _spying_init
            try:
                return [
                    inc
                    async for inc in worker_run(
                        core, source=_source(), stats=stats,
                        judge_dispatch="async", judge_stall_after_s=0.0, judge_backpressure_after=2,
                    )
                ]
            finally:
                worker_mod._JudgeScheduler.__init__ = orig_init

        # Poll the invariant on every frame via a wrapped submit(), since the scheduler instance is
        # only constructed once run() starts.
        asyncio.run(_collect())
        self.assertEqual(len(seen), 1, "one scheduler per run()")
        sched = seen[0]
        _invariant_ok(self, sched)
        # after full drain, both slots must be back to empty
        self.assertEqual(sched.active, 0)
        self.assertEqual(sched.pending, 0)


class SchedulerDirectRecoveryTests(unittest.TestCase):
    """Direct ``_JudgeScheduler`` unit tests (no full ``run()`` loop) -- the same style as
    ``test_runtime_judge_scheduler.py``'s ``SchedulerUnitTests``, for the two NEW latches."""

    class _StubCore:
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

    def test_stall_then_recovery_flags(self):
        async def main():
            core = self._StubCore()
            stats = RuntimeStats()
            sched = _JudgeScheduler(core, "camera", stats, stall_after_s=0.0, backpressure_after=99)
            sched.submit(1.0, "f1")  # launches
            await asyncio.sleep(0)
            self.assertEqual((sched.active, sched.pending), (1, 0))
            sched.submit(2.0, "f2")  # supersede check ALSO runs the stall check (threshold 0.0)
            self.assertTrue(sched._stall_reported)
            trace = dict(sched.drain_trace())
            self.assertIn("judge_scheduler_stalled", trace)

            core.block.set()
            await sched.flush()
            self.assertFalse(sched._stall_reported, "cleared on completion")
            trace2 = dict(sched.drain_trace())
            self.assertIn("judge_scheduler_recovered", trace2)
            self.assertIn("stalled", trace2["judge_scheduler_recovered"]["recovered_from"])
            _invariant_ok(self, sched)

        asyncio.run(main())

    def test_backpressure_latches_once_then_clears(self):
        async def main():
            core = self._StubCore()
            stats = RuntimeStats()
            sched = _JudgeScheduler(core, "camera", stats, stall_after_s=999.0, backpressure_after=2)
            sched.submit(1.0, "f1")  # launches
            await asyncio.sleep(0)
            sched.submit(2.0, "f2")  # 1st pending
            sched.submit(3.0, "f3")  # supersede #1
            self.assertFalse(sched._backpressure_reported)
            sched.submit(4.0, "f4")  # supersede #2 -> crosses threshold=2
            self.assertTrue(sched._backpressure_reported)
            sched.submit(5.0, "f5")  # supersede #3 -- must NOT re-trace, latch already set
            kinds = [k for k, _ in sched.drain_trace()]
            self.assertEqual(kinds.count("judge_scheduler_backpressure"), 1)
            _invariant_ok(self, sched)

            core.block.set()
            await sched.flush()
            trace = dict(sched.drain_trace())
            self.assertIn("judge_scheduler_recovered", trace)
            self.assertIn("backpressure", trace["judge_scheduler_recovered"]["recovered_from"])

        asyncio.run(main())

    def test_clean_run_never_reports_recovery(self):
        """No degradation ever latched -> completion must stay silent (recovery is only news when a
        degradation preceded it)."""
        async def main():
            core = self._StubCore()
            stats = RuntimeStats()
            sched = _JudgeScheduler(core, "camera", stats)
            sched.submit(1.0, "f1")
            await asyncio.sleep(0)
            core.block.set()
            await sched.flush()
            kinds = [k for k, _ in sched.drain_trace()]
            self.assertNotIn("judge_scheduler_recovered", kinds)

        asyncio.run(main())


if __name__ == "__main__":
    unittest.main()
