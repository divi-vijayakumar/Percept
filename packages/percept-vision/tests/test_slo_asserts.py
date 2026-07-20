"""M4 -- SLO CI asserts: reflex p95, $/stream-hour, fovea call-reduction (percept/slo.py).

Offline/keyless: synthetic frames + a scripted :class:`~percept.contracts.Verdict`'s ``usage``
dict, no network, no API keys. Each assertion's threshold is a pre-registered constant in
``percept/slo.py`` with its own rationale -- see that module's docstring for the full "why this
number" (this file only exercises the assertions, it does not re-derive the thresholds).
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.contracts import Verdict
from percept.harness.tier0 import Frame, GatedWatchLoop, MotionGate, compile_watchspec, gray_frame
from percept.observability import judge_usage
from percept.slo import (
    DOLLARS_PER_STREAM_HOUR_CEILING_USD,
    FOVEA_CALL_REDUCTION_FLOOR,
    REFLEX_P95_THRESHOLD_S,
    JudgeCallSample,
    dollars_per_stream_hour,
    fovea_call_reduction,
    measure_reflex_latencies,
    price_usage,
    reflex_p95,
)

GRID_W, GRID_H = 64, 48


def _still() -> Frame:
    return gray_frame(GRID_W, GRID_H, fill=0)


def _motion() -> Frame:
    """A scattered, deterministic 'moving' pattern -- large enough of a change to clear a
    permissive gate threshold, same shape as bench_gating.py's fixture."""
    pixels = [0] * (GRID_W * GRID_H)
    for i in range(0, len(pixels), 7):
        pixels[i] = 255
    return Frame(w=GRID_W, h=GRID_H, pixels=tuple(pixels))


def _mostly_still_stream(n: int) -> list[Frame]:
    """A deterministic mostly-still sequence with motion roughly every 10th frame -- the same
    'still scene + occasional event' shape production tier-0 sees (percept/harness/bench_gating.py),
    sized up here for a stable p95 sample / a representative call-reduction fixture."""
    motion_idx = frozenset(i for i in range(n) if i % 10 == 0)
    return [_motion() if i in motion_idx else _still() for i in range(n)]


# ---------------------------------------------------------------------------
# SLO 1 -- REFLEX p95: the tier-0 gate's own decision latency
# ---------------------------------------------------------------------------


class TestReflexP95(unittest.TestCase):
    def test_reflex_p95_under_still_scene_budget(self):
        goals = [Goal(id="g1", condition="a person entered the frame", say="hi")]
        spec = compile_watchspec(goals, threshold=0.2)
        gate = MotionGate(spec)
        frames = _mostly_still_stream(300)

        # Warm-up: the first few calls pay one-time import/allocator costs unrelated to the
        # steady-state per-frame latency a long-running stream actually pays; exclude them from
        # the measured p95 sample (amortized once per process, not once per frame in production).
        warm, measured = frames[:10], frames[10:]
        for f in warm:
            gate.evaluate(f, ts=0.0)

        latencies = measure_reflex_latencies(gate, measured, ts_start=0.0)
        p95 = reflex_p95(latencies)

        self.assertLess(
            p95,
            REFLEX_P95_THRESHOLD_S,
            f"reflex p95 {p95 * 1000:.4f}ms exceeds the {REFLEX_P95_THRESHOLD_S * 1000:.1f}ms "
            f"still-scene budget (15% of the max_fps=15.0 frame period, percept.sampling."
            f"SensePolicy) -- the tier-0 gate is no longer 'effectively free'.",
        )


# ---------------------------------------------------------------------------
# SLO 2 -- $/stream-hour: judge_usage() token counts x PRICE_TABLE
# ---------------------------------------------------------------------------


class TestDollarsPerStreamHour(unittest.TestCase):
    def test_dollars_per_stream_hour_under_ceiling(self):
        # A representative mostly-still deployment: tier-0 suppresses most frames, so the judge
        # fires at a modest, sparse rate -- 30 judged calls/hour (about once every 2 minutes),
        # each a single-frame vision judgment (~2000 input / ~150 output tokens is a plausible
        # single-image-plus-prompt judge call) against the default vision_model (percept/config.py
        # Models.vision_model = "claude-opus-4-8").
        verdict = Verdict(
            satisfied=True,
            confidence=0.9,
            evidence="a person is visible",
            usage={"input_tokens": 2000.0, "output_tokens": 150.0},
        )
        usage = judge_usage(verdict)  # the already-landed hook (percept/observability.py)
        calls = [JudgeCallSample(usage=usage, model="claude-opus-4-8") for _ in range(30)]

        dph = dollars_per_stream_hour(calls, elapsed_s=3600.0, stream_hours=1.0)

        self.assertLess(
            dph,
            DOLLARS_PER_STREAM_HOUR_CEILING_USD,
            f"${dph:.4f}/stream-hour exceeds the ${DOLLARS_PER_STREAM_HOUR_CEILING_USD:.2f}/"
            f"stream-hour ceiling (tests/test_p_econ.py's Viability §7 placeholder, reused here).",
        )

    def test_price_usage_prefers_backend_reported_cost_over_token_estimate(self):
        """A backend that reports its own billed cost_usd wins over a token x price estimate."""
        usage = {"cost_usd": 0.5, "input_tokens": 999_999.0}
        self.assertEqual(price_usage(usage, model="claude-opus-4-8"), 0.5)

    def test_price_usage_falls_back_to_default_price_for_uncatalogued_model(self):
        """An unrecognized model prices at the most expensive known tier (fail conservative)."""
        usage = {"input_tokens": 1_000_000.0, "output_tokens": 0.0}
        self.assertEqual(price_usage(usage, model="some-future-model"), 15.00)

    def test_price_usage_is_zero_for_empty_usage(self):
        self.assertEqual(price_usage({}, model="claude-opus-4-8"), 0.0)


# ---------------------------------------------------------------------------
# SLO 3 -- fovea call-reduction: the fraction of frames tier-0 suppresses
# ---------------------------------------------------------------------------


class TestFoveaCallReduction(unittest.TestCase):
    def test_fovea_call_reduction_over_floor(self):
        core = Percept.create(mode="offline")  # offline fakes -- no keys, deterministic
        goals = [
            Goal(id="g_coffee", condition="a coffee cup is on the desk", say="that looks like coffee"),
            Goal(id="g_person", condition="a person entered the frame", say="someone's here"),
        ]
        for g in goals:
            core.add_goal(g)

        loop = GatedWatchLoop(core, goals)
        report = asyncio.run(loop.run(_mostly_still_stream(200), clock_start=0.0, dt=1.0))
        reduction = fovea_call_reduction(report)

        self.assertGreaterEqual(
            reduction,
            FOVEA_CALL_REDUCTION_FLOOR,
            f"fovea call-reduction {reduction:.1%} fell below the {FOVEA_CALL_REDUCTION_FLOOR:.0%} "
            f"floor -- tier-0 gating is no longer clearing its minimum economic bar (at least half "
            f"the naive per-frame VLM spend suppressed).",
        )


if __name__ == "__main__":
    unittest.main()
