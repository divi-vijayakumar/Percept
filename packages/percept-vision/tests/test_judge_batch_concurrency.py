"""Multi-goal judge batching: makespan, boundedness, and fairness — measured, not asserted by hand.

WHAT THIS FILE IS. A deterministic offline harness for one question: when a frame admits N goals,
how long does the batch take? It answers it by MEASURING wall-clock against a fake judge that
sleeps a latency sampled from the real measured distribution, so the number it reports is a real
makespan and not a mock's opinion. No provider is called and no key is read.

THE DISTRIBUTION IS MEASURED, NOT INVENTED. ``_P50_S`` / ``_P95_S`` below are the UNCENSORED
single-call latency quantiles from the n=100 corpus run's deliberate route. The reflex figures from
that same run (p50 1.365s, p95 1.473s, max 1.516s) are deliberately NOT used: that route ran under a
1.5s deadline, so its "distribution" is the deadline's own shadow — every sample above the cut was
discarded before it could be measured. A harness calibrated on a censored sample would model a
provider that cannot be slow, which is precisely the case that makes batching hurt.

WHAT THE HARNESS CANNOT BUY. Fanning a batch out converts a SUM into a MAX, and the expected MAX of
N independent draws sits near the distribution's N/(N+1) quantile. A 73-goal batch therefore lands
near p99 no matter how many slots it gets: concurrency removes the WAVES, never the TAIL. The
``__main__`` block prints that curve at real speed so the residual is visible rather than argued.

RUNNING IT FOR REAL. In the suite every sleep is scaled by ``_SUITE_SCALE`` so the whole file costs
well under a second of (idle) wall-clock. ``python -m tests.test_judge_batch_concurrency`` — or
``python tests/test_judge_batch_concurrency.py --measure`` — re-runs the N x K grid at scale 1.0,
which takes minutes of sleeping and no CPU.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
import unittest
from typing import Any

from percept.adapters import Goal, Verdict
from percept.agent import _DEFAULT_JUDGE_CONCURRENCY, Percept
from percept.backends.judge_pool import JudgeCapability, JudgePool, JudgeRoute

# -- the measured single-call latency distribution -------------------------------------------------
# n=100 corpus run, deliberate route (not deadline-censored).
_P50_S = 1.653
_P95_S = 2.283
# A lognormal is the standard shape for provider round-trip latency and is fully determined by two
# quantiles: the median IS the scale, and sigma follows from the p95/p50 ratio (z(0.95) = 1.6449).
_SIGMA = math.log(_P95_S / _P50_S) / 1.6448536269514722
_MEAN_S = _P50_S * math.exp(_SIGMA**2 / 2)  # ~1.685s

#: Every sleep in the suite is multiplied by this. Sleeping is free of CPU but not of patience, and
#: this file has to co-exist with 4900 other tests. Ratios — which is all the assertions read — are
#: scale-invariant.
_SUITE_SCALE = 0.005


def _latency_s(index: int, *, seed: int = 20260727) -> float:
    """The index-th draw from the measured distribution. Deterministic, so the sequential baseline
    and the concurrent run judge the SAME N latencies and the comparison is apples-to-apples."""
    rng = random.Random(seed + index)
    return _P50_S * math.exp(_SIGMA * rng.gauss(0.0, 1.0))


class SleepingJudge:
    """A judge that costs exactly one sampled provider latency and nothing else.

    Also records concurrency: ``peak_inflight`` is what proves the bound is a real bound, and
    ``calls_by_goal`` is what proves fairness (every admitted goal judged, exactly once, per frame).
    """

    def __init__(self, *, scale: float = _SUITE_SCALE) -> None:
        self.scale = scale
        self.calls_by_goal: dict[str, int] = {}
        self.order: list[str] = []            # dispatch order (who got a slot, and when)
        self.completion_order: list[str] = []  # who answered first — reordered by the latencies
        self.inflight = 0
        self.peak_inflight = 0
        self._next = 0

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        goal_id = condition.rsplit(" ", 1)[-1]
        self.calls_by_goal[goal_id] = self.calls_by_goal.get(goal_id, 0) + 1
        self.order.append(goal_id)
        index = self._next
        self._next += 1
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            await asyncio.sleep(_latency_s(index) * self.scale)
        finally:
            self.inflight -= 1
        self.completion_order.append(goal_id)
        return Verdict(False, 0.9, f"judged {goal_id}")

    async def detect(self, frame: Any, goals: list) -> list:
        return []

    async def detect_events(self, frames: list, goals: list) -> list:
        return []


class _BoundedSleepingJudge(SleepingJudge):
    """A sleeping judge that DECLARES a concurrent-decision capacity, like ``JudgePool`` does."""

    def __init__(self, max_inflight: int, *, scale: float = _SUITE_SCALE) -> None:
        super().__init__(scale=scale)
        self.max_inflight = max_inflight


def _core(vision: Any, *, judge_concurrency: int | None = None) -> Percept:
    return Percept.create(
        vision=vision,
        discover_plugins=False,
        judge_concurrency=judge_concurrency,
    )


def _arm(core: Percept, n: int) -> None:
    for i in range(n):
        gid = f"g{i:03d}"
        core.add_goal(Goal(id=gid, condition=f"is it true for {gid}", say="x", source="camera"))


async def _one_batch(n: int, *, judge_concurrency: int | None, scale: float = _SUITE_SCALE):
    """Arm ``n`` watch goals, judge ONE frame, and return (wall_clock_s, the judge)."""
    vision = SleepingJudge(scale=scale)
    core = _core(vision, judge_concurrency=judge_concurrency)
    _arm(core, n)
    started = time.monotonic()
    await core.perceive_judged("frame", "camera", t=1.0)
    return time.monotonic() - started, vision


def _gini(counts: list[int]) -> float:
    """0.0 = perfectly equal. The fairness statistic the 82-minute run already scored 0.0 on."""
    if not counts or sum(counts) == 0:
        return 0.0
    ordered = sorted(counts)
    n = len(ordered)
    cumulative = sum((i + 1) * value for i, value in enumerate(ordered))
    return (2 * cumulative) / (n * sum(ordered)) - (n + 1) / n


class BatchMakespanTests(unittest.TestCase):
    """The SUM -> MAX conversion, measured."""

    def test_serial_batch_costs_the_sum_of_its_calls(self):
        # The defect, reproduced: a bound of 1 IS the pre-change behaviour, and its makespan is the
        # sum. This is the baseline every other number in this file is read against.
        wall, vision = asyncio.run(_one_batch(20, judge_concurrency=1))
        expected_sum = sum(_latency_s(i) for i in range(20)) * _SUITE_SCALE
        self.assertEqual(len(vision.calls_by_goal), 20)
        self.assertEqual(vision.peak_inflight, 1)
        self.assertGreaterEqual(wall, expected_sum * 0.9)

    def test_concurrent_batch_costs_the_max_of_its_calls(self):
        wall, vision = asyncio.run(_one_batch(20, judge_concurrency=20))
        slowest = max(_latency_s(i) for i in range(20)) * _SUITE_SCALE
        total = sum(_latency_s(i) for i in range(20)) * _SUITE_SCALE
        self.assertEqual(len(vision.calls_by_goal), 20)
        # A MAX, not a SUM: comfortably under a quarter of the serial cost, and within a small
        # constant of the slowest single call it had to wait for.
        self.assertLess(wall, total / 4)
        self.assertLess(wall, slowest * 3 + 0.05)

    def test_makespan_falls_as_the_bound_rises_and_then_stops_falling(self):
        # The honest shape of the win: waves, then the tail. Going 1 -> 8 buys a lot; 8 -> 73 buys
        # the last waves; nothing buys below the slowest single draw.
        walls = {
            slots: asyncio.run(_one_batch(40, judge_concurrency=slots))[0]
            for slots in (1, 8, 40)
        }
        self.assertLess(walls[8], walls[1] / 3)
        self.assertLess(walls[40], walls[8])
        slowest = max(_latency_s(i) for i in range(40)) * _SUITE_SCALE
        self.assertGreaterEqual(walls[40], slowest)  # the tail is a floor, not a target


class BatchBoundTests(unittest.TestCase):
    """The bound is real, and a DECLARED capacity always wins."""

    def test_inflight_never_exceeds_the_configured_bound(self):
        _, vision = asyncio.run(_one_batch(40, judge_concurrency=6))
        self.assertLessEqual(vision.peak_inflight, 6)
        self.assertEqual(vision.peak_inflight, 6)  # and it is actually used, not merely respected

    def test_default_bound_is_used_when_nothing_is_configured_or_declared(self):
        core = _core(SleepingJudge())
        self.assertEqual(core.effective_judge_concurrency(), _DEFAULT_JUDGE_CONCURRENCY)
        _, vision = asyncio.run(_one_batch(40, judge_concurrency=None))
        self.assertLessEqual(vision.peak_inflight, _DEFAULT_JUDGE_CONCURRENCY)

    def test_declared_adapter_capacity_floors_a_larger_configured_bound(self):
        core = _core(_BoundedSleepingJudge(3), judge_concurrency=16)
        self.assertEqual(core.effective_judge_concurrency(), 3)

    def test_configured_bound_floors_a_larger_declared_capacity(self):
        core = _core(_BoundedSleepingJudge(32), judge_concurrency=4)
        self.assertEqual(core.effective_judge_concurrency(), 4)

    def test_judge_pool_publishes_the_minimum_route_occupancy(self):
        # Every eligible route is scheduled per decision, so D concurrent decisions need D slots on
        # each of them: the pool can serve MIN(route occupancy) decisions, not the sum.
        def _route(name: str, inflight: int) -> JudgeRoute:
            return JudgeRoute(
                name,
                SleepingJudge(),
                JudgeCapability(
                    evidence_shapes=frozenset({"single_image"}),
                    latency_slo_s=2.5,
                    modes=frozenset({"deliberate"}),
                    max_inflight=inflight,
                ),
            )

        pool = JudgePool(
            [_route("wide", 9), _route("narrow", 2)],
            mode="deliberate",
            deadline_s=3.0,
        )
        self.assertEqual(pool.max_inflight, 2)
        self.assertEqual(_core(pool).effective_judge_concurrency(), 2)

    def test_a_pool_declaring_one_keeps_the_batch_strictly_serial(self):
        # A deployment that declared max_inflight=1 gets exactly what it declared. The pool has no
        # queue, so exceeding it would fail calls closed on "capacity" rather than slow them down —
        # honouring the declaration is the only safe reading, and raising it is the operator's lever.
        pool = JudgePool(
            [
                JudgeRoute(
                    "solo",
                    SleepingJudge(),
                    JudgeCapability(
                        evidence_shapes=frozenset({"single_image"}),
                        latency_slo_s=2.5,
                        modes=frozenset({"deliberate"}),
                        max_inflight=1,
                    ),
                )
            ],
            mode="deliberate",
            deadline_s=3.0,
        )
        self.assertEqual(_core(pool, judge_concurrency=64).effective_judge_concurrency(), 1)


class BatchFairnessTests(unittest.TestCase):
    """Gini 0.0 with nobody starved was the ONE good property of the serial batch. It survives."""

    def test_every_admitted_goal_is_judged_exactly_once_per_frame(self):
        for slots in (2, 8, 73):
            with self.subTest(slots=slots):
                _, vision = asyncio.run(_one_batch(73, judge_concurrency=slots))
                counts = [vision.calls_by_goal.get(f"g{i:03d}", 0) for i in range(73)]
                self.assertEqual(sorted(set(counts)), [1])   # nobody skipped, nobody double-spent
                self.assertEqual(_gini(counts), 0.0)

    def test_a_narrow_bound_queues_goals_it_does_not_drop_them(self):
        # The failure mode concurrency could have introduced: slots handed out as a lottery, with
        # goals late in the list starved. asyncio.Semaphore is FIFO, so the first wave is exactly
        # the first K goals in registry order and every later goal still gets served.
        _, vision = asyncio.run(_one_batch(30, judge_concurrency=4))
        self.assertEqual(vision.order[:4], ["g000", "g001", "g002", "g003"])
        self.assertEqual(sorted(vision.calls_by_goal), [f"g{i:03d}" for i in range(30)])

    def test_fires_stay_in_goal_order_regardless_of_completion_order(self):
        # Concurrency reorders COMPLETIONS; it must not reorder the batch's emitted events, which
        # the apply pass walks in registry order.
        class _FiringJudge(SleepingJudge):
            async def judge(self, condition, frame, **kwargs):
                await super().judge(condition, frame, **kwargs)
                return Verdict(True, 0.95, f"fired {condition.rsplit(' ', 1)[-1]}")

        async def main():
            vision = _FiringJudge()
            core = _core(vision, judge_concurrency=16)
            _arm(core, 16)
            return await core.perceive_judged("frame", "camera", t=1.0), vision

        events, vision = asyncio.run(main())
        self.assertEqual([e.goal_id for e in events], [f"g{i:03d}" for i in range(16)])
        # ...and the judges really did finish out of order, or this proves nothing.
        self.assertNotEqual(vision.completion_order, sorted(vision.completion_order))


class BatchDeadlineTests(unittest.TestCase):
    """A deadline sized for ONE goal must not cancel a whole batch."""

    def _scheduler(self, core, *, deadline_s: float):
        from percept.runtime.worker import _LegacyJudgeScheduler

        return _LegacyJudgeScheduler(core, "camera", None, decision_deadline_s=deadline_s)

    def test_batch_deadline_scales_with_the_waves_the_batch_actually_needs(self):
        core = _core(SleepingJudge(), judge_concurrency=8)
        _arm(core, 73)
        sched = self._scheduler(core, deadline_s=1.9)
        # 73 goals at 8 slots is 10 waves, so the batch's absolute budget is 10 per-goal budgets.
        self.assertEqual(sched._batch_deadline_s(None), 1.9 * 10)
        # ...and a batch that fits in one wave keeps the historical, unscaled value exactly.
        self.assertEqual(sched._batch_deadline_s(tuple(f"g{i:03d}" for i in range(8))), 1.9)
        self.assertEqual(sched._batch_deadline_s(("g000",)), 1.9)

    def test_a_bound_wide_enough_for_the_batch_leaves_the_shipped_deadline_unscaled(self):
        # The cross-component contract, pinned against the SHIPPED constant rather than a literal.
        # The wave multiplier is not a tax on batching: it is the wave count of a bounded batch, and
        # a bound wide enough to hold the batch in one wave returns the per-decision budget exactly.
        # Widening ``judge_concurrency`` — not loosening this formula — is how the backstop shrinks.
        from percept.backends.judge_pool import REFLEX_DECISION_DEADLINE_S

        core = _core(SleepingJudge(), judge_concurrency=73)
        _arm(core, 73)
        sched = self._scheduler(core, deadline_s=REFLEX_DECISION_DEADLINE_S)
        self.assertEqual(sched._batch_deadline_s(None), REFLEX_DECISION_DEADLINE_S)
        # ...and one slot short of that, exactly two waves. No rounding surprises at the boundary.
        narrower = _core(SleepingJudge(), judge_concurrency=72)
        _arm(narrower, 73)
        self.assertEqual(
            self._scheduler(narrower, deadline_s=REFLEX_DECISION_DEADLINE_S)._batch_deadline_s(None),
            REFLEX_DECISION_DEADLINE_S * 2,
        )

    def test_a_serial_core_still_gets_one_budget_per_goal(self):
        core = _core(SleepingJudge(), judge_concurrency=1)
        _arm(core, 5)
        sched = self._scheduler(core, deadline_s=1.9)
        self.assertEqual(sched._batch_deadline_s(None), 1.9 * 5)

    def test_no_deadline_configured_stays_no_deadline(self):
        from percept.runtime.worker import _LegacyJudgeScheduler

        core = _core(SleepingJudge())
        _arm(core, 12)
        sched = _LegacyJudgeScheduler(core, "camera", None)
        self.assertIsNone(sched._batch_deadline_s(None))

    def test_a_multi_goal_batch_survives_a_single_goal_deadline(self):
        # The defect end-to-end. The per-goal budget below comfortably covers ONE call and cannot
        # cover twenty; before the derivation, that cancelled the whole batch and delivered NOTHING
        # — including for the goals that had already answered.
        from percept.runtime.worker import _LEGACY_DEADLINE_EXPIRED

        per_goal_s = 3.0 * _SUITE_SCALE

        async def main():
            vision = SleepingJudge()
            core = _core(vision, judge_concurrency=4)
            _arm(core, 20)
            sched = self._scheduler(core, deadline_s=per_goal_s)
            sched.submit(1.0, "frame")
            for _ in range(20000):
                if sched._active is None:
                    break
                await asyncio.sleep(0)
            return sched.drain(), vision, sched._batch_deadline_s(None)

        drained, vision, batch_s = asyncio.run(main())
        self.assertEqual(len(vision.calls_by_goal), 20)         # every goal judged
        self.assertEqual(len(drained), 1)                       # one batch result delivered
        self.assertIsNot(drained[0], _LEGACY_DEADLINE_EXPIRED)  # not cancelled wholesale
        # And the batch genuinely needed more than one goal's budget: 20 goals / 4 slots = 5 waves.
        self.assertEqual(batch_s, per_goal_s * 5)
        self.assertGreater(sum(_latency_s(i) for i in range(4)) * _SUITE_SCALE, per_goal_s)


# -- real-speed measurement ------------------------------------------------------------------------


def _measure() -> None:  # pragma: no cover - operator tool, never run by the suite
    """Print the N x K makespan grid at scale 1.0. Minutes of sleeping, no CPU, no provider."""
    grid_n = (1, 5, 20, 73)
    grid_k = (1, 8, 20, 73)
    print(
        f"single-call model: lognormal(median={_P50_S}s, sigma={_SIGMA:.5f}) "
        f"-> p95 {_P95_S}s, mean {_MEAN_S:.3f}s"
    )
    header = "    N |" + "".join(f"    K={k:<3d}" for k in grid_k)
    print(header)
    print("-" * len(header))
    for n in grid_n:
        row = f"{n:5d} |"
        for k in grid_k:
            wall, _ = asyncio.run(_one_batch(n, judge_concurrency=k, scale=1.0))
            row += f"  {wall:7.2f}s"
        print(row)
    for n in grid_n:
        draws = [_latency_s(i) for i in range(n)]
        print(f"N={n:3d}  sum={sum(draws):8.2f}s  max={max(draws):6.3f}s")


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--measure" in sys.argv:
        _measure()
    else:
        unittest.main()
