"""Standard telemetry funnel (seen→gated→judged→refused→fired) plus judge latency/cost and OTel.

L1, fakes only, no keys, deterministic. Covers:
  * :class:`Funnel`        — the five stages bump the run total + per-concern + per-camera slices.
  * :func:`judge_usage`    — token/cost hook reads an annotated verdict, ignores a bare one.
  * judge latency/cost     — record_judge stores count+total; means fall out by division.
  * OTel export            — degrades to a no-op without the SDK; records to a fake meter when given one.
  * agent integration      — perceive_judged emits judged + (refused | fired) into the shared Metrics.
"""

import asyncio
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.observability import (
    FUNNEL_STAGES,
    Funnel,
    Metrics,
    OTelExporter,
    funnel_key,
    judge_usage,
    otel_exporter,
)


# ---------------------------------------------------------------------------
# Funnel — the five stages over the in-process Metrics
# ---------------------------------------------------------------------------


class FunnelStageTests(unittest.TestCase):
    def test_stages_in_order(self):
        self.assertEqual(FUNNEL_STAGES, ("seen", "gated", "judged", "refused", "fired"))

    def test_seen_gated_are_camera_scoped_run_total(self):
        m = Metrics()
        f = Funnel(m)
        f.seen(camera="cam")
        f.gated(camera="cam")
        snap = m.snapshot()
        self.assertEqual(snap[funnel_key("seen")], 1)
        self.assertEqual(snap[funnel_key("seen", camera="cam")], 1)
        self.assertEqual(snap[funnel_key("gated")], 1)
        # seen has no concern scope (a frame precedes any goal)
        self.assertNotIn(funnel_key("seen", concern="anything"), snap)

    def test_judged_refused_fired_slice_by_concern_and_camera(self):
        m = Metrics()
        f = Funnel(m)
        f.judged(concern="g1", camera="cam")
        f.refused(concern="g1", camera="cam")
        f.fired(concern="g1", camera="cam")
        snap = m.snapshot()
        for stage in ("judged", "refused", "fired"):
            self.assertEqual(snap[funnel_key(stage)], 1)
            self.assertEqual(snap[funnel_key(stage, concern="g1")], 1)
            self.assertEqual(snap[funnel_key(stage, camera="cam")], 1)

    def test_per_concern_slices_are_independent(self):
        m = Metrics()
        f = Funnel(m)
        f.judged(concern="g1")
        f.judged(concern="g2")
        f.judged(concern="g1")
        snap = m.snapshot()
        self.assertEqual(snap[funnel_key("judged")], 3)  # run total
        self.assertEqual(snap[funnel_key("judged", concern="g1")], 2)
        self.assertEqual(snap[funnel_key("judged", concern="g2")], 1)


# ---------------------------------------------------------------------------
# Judge latency + token/cost
# ---------------------------------------------------------------------------


class JudgeUsageTests(unittest.TestCase):
    def test_bare_verdict_has_no_usage(self):
        self.assertEqual(judge_usage(Verdict(True, 0.9, "x")), {})

    def test_usage_dict_attribute_is_read(self):
        v = Verdict(True, 0.9, "x")
        v.usage = {"input_tokens": 120, "output_tokens": 8, "cost_usd": 0.0021}  # type: ignore[attr-defined]
        self.assertEqual(
            judge_usage(v), {"input_tokens": 120.0, "output_tokens": 8.0, "cost_usd": 0.0021}
        )

    def test_loose_attributes_are_read(self):
        v = Verdict(True, 0.9, "x")
        v.input_tokens = 50  # type: ignore[attr-defined]
        self.assertEqual(judge_usage(v), {"input_tokens": 50.0})

    def test_record_judge_stores_count_and_totals(self):
        m = Metrics()
        f = Funnel(m)
        f.record_judge(latency_s=0.10, input_tokens=100, output_tokens=10, cost_usd=0.002)
        f.record_judge(latency_s=0.30, input_tokens=200, output_tokens=20, cost_usd=0.004)
        snap = m.snapshot()
        self.assertEqual(snap["judge.latency.count"], 2)
        self.assertAlmostEqual(snap["judge.latency.total_s"], 0.40)
        self.assertAlmostEqual(snap["judge.tokens.input"], 300.0)
        self.assertAlmostEqual(snap["judge.tokens.output"], 30.0)
        self.assertAlmostEqual(snap["judge.cost.usd"], 0.006)
        # mean latency = total / count
        self.assertAlmostEqual(snap["judge.latency.total_s"] / snap["judge.latency.count"], 0.20)

    def test_latency_only_when_no_usage(self):
        m = Metrics()
        Funnel(m).record_judge(latency_s=0.05)
        snap = m.snapshot()
        self.assertEqual(snap["judge.latency.count"], 1)
        self.assertNotIn("judge.tokens.input", snap)
        self.assertNotIn("judge.cost.usd", snap)


# ---------------------------------------------------------------------------
# OpenTelemetry export — no-op without the dep; records to a fake meter
# ---------------------------------------------------------------------------


class _FakeInstrument:
    def __init__(self):
        self.calls = []

    def add(self, amount, attributes=None):
        self.calls.append((amount, attributes))

    def record(self, amount, attributes=None):
        self.calls.append((amount, attributes))


class _FakeMeter:
    def __init__(self):
        self.created = {}

    def create_counter(self, name, **kw):
        return self.created.setdefault(name, _FakeInstrument())

    def create_histogram(self, name, **kw):
        return self.created.setdefault(name, _FakeInstrument())


class OTelExportTests(unittest.TestCase):
    def test_degrades_to_noop_without_meter(self):
        exp = OTelExporter(None)
        self.assertFalse(exp.enabled)
        # no-op methods don't raise
        exp.record_stage("seen", concern=None, camera="cam", n=1)
        exp.record_judge(latency_s=0.1, input_tokens=10)

    def test_otel_exporter_factory_no_sdk_is_noop(self):
        # opentelemetry is the [otel] extra and is NOT installed in the L1 lane -> graceful no-op.
        exp = otel_exporter()
        self.assertFalse(exp.enabled)

    def test_records_stages_and_judge_to_fake_meter(self):
        meter = _FakeMeter()
        exp = OTelExporter(meter)
        self.assertTrue(exp.enabled)
        exp.record_stage("fired", concern="g1", camera="cam", n=1)
        exp.record_judge(latency_s=0.2, input_tokens=100, output_tokens=10, cost_usd=0.003)
        self.assertEqual(meter.created["percept.funnel.fired"].calls,
                         [(1, {"concern": "g1", "camera": "cam"})])
        self.assertEqual(meter.created["percept.judge.latency"].calls, [(0.2, None)])
        self.assertIn((100.0, {"direction": "input"}), meter.created["percept.judge.tokens"].calls)
        self.assertIn((10.0, {"direction": "output"}), meter.created["percept.judge.tokens"].calls)
        self.assertEqual(meter.created["percept.judge.cost_usd"].calls, [(0.003, None)])

    def test_funnel_mirrors_to_otel(self):
        # A Funnel given an OTel exporter writes BOTH the in-process Metrics and the OTel meter.
        meter = _FakeMeter()
        m = Metrics()
        Funnel(m, otel=OTelExporter(meter)).judged(concern="g1", camera="cam")
        self.assertEqual(m.snapshot()[funnel_key("judged")], 1)  # in-process
        self.assertEqual(len(meter.created["percept.funnel.judged"].calls), 1)  # OTel mirror


# ---------------------------------------------------------------------------
# Agent integration — perceive_judged emits judged + (refused | fired)
# ---------------------------------------------------------------------------


def _agent(vision):
    s = Settings(gate=GateSettings(threshold=0.7))
    return Percept.create(vision=vision, settings=s, discover_plugins=False)


def _watch(a, gid="g1", cond="a cup is up"):
    a.add_goal(Goal(id=gid, condition=cond, say=f"saw {gid}"))


class AgentFunnelTests(unittest.TestCase):
    def test_judged_and_fired_on_confident_yes(self):
        vision = FakeVision(
            script={"yes": Verdict(True, 0.95, "yes")},
            detect_script={"yes": [Observation("cup", Box(0.4, 0.4, 0.2, 0.2), 0.95)]},
        )
        a = _agent(vision)
        _watch(a)
        asyncio.run(a.perceive_judged("yes", source="cam", t=1.0))
        snap = a.metrics.snapshot()
        self.assertEqual(snap[funnel_key("judged", concern="g1")], 1)
        self.assertEqual(snap[funnel_key("fired", concern="g1")], 1)
        self.assertEqual(snap[funnel_key("fired", camera="cam")], 1)
        self.assertNotIn(funnel_key("refused", concern="g1"), snap)  # confident -> not refused

    def test_refused_on_low_confidence(self):
        vision = FakeVision(script={"maybe": Verdict(True, 0.4, "unsure")})
        a = _agent(vision)
        _watch(a)
        asyncio.run(a.perceive_judged("maybe", source="cam", t=1.0))
        snap = a.metrics.snapshot()
        self.assertEqual(snap[funnel_key("judged", concern="g1")], 1)
        self.assertEqual(snap[funnel_key("refused", concern="g1")], 1)  # ASK -> refused to guess
        self.assertNotIn(funnel_key("fired", concern="g1"), snap)

    def test_judge_latency_recorded_per_call(self):
        # Two frames -> two judge calls -> latency count == 2; total >= 0 (the default clock is monotonic).
        a = _agent(FakeVision(script={"f": Verdict(False, 0.9, "no")}))
        _watch(a)
        asyncio.run(a.perceive_judged("f", source="cam", t=1.0))
        asyncio.run(a.perceive_judged("f", source="cam", t=2.0))
        snap = a.metrics.snapshot()
        self.assertEqual(snap["judge.latency.count"], 2)
        self.assertGreaterEqual(snap["judge.latency.total_s"], 0.0)

    def test_funnel_property_is_stable_and_shares_metrics(self):
        a = _agent(FakeVision(script={}))
        self.assertIs(a.funnel, a.funnel)  # lazily built once
        a.funnel.seen(camera="cam")
        self.assertEqual(a.metrics.snapshot()[funnel_key("seen")], 1)  # same registry as health()


if __name__ == "__main__":
    unittest.main()
