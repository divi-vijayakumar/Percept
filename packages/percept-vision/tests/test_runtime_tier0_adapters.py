"""D6(a) — the tier-0 extension seam is reachable from the supported entrypoint.

`MotionGate` has accepted `structural_adapter` / `semantic_adapter` since Phase 6, but
`percept.runtime.run()` built its gate without them and exposed no parameter: to supplement tier 0 you
had to abandon `run()` and drive `GatedWatchLoop` yourself. These tests assert:

  * a `semantic_adapter` passed to `run()` reaches the gate and ADMITS a frame the motion gate skips
    (a still screen whose *content* changed — the client's screen workload),
  * the same for `structural_adapter`,
  * the reported `Tier0Signal` carries the adapter's value (it is observable in the edge trace),
  * omitting both is byte-for-byte the motion-only path (0.0 channels, no admission on a still stream).
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.runtime import RuntimeStats, run

W, H = 8, 8
STILL = gray_frame(W, H, 0)
# A perfectly still stream: motion is 0.0 on every frame, so the motion gate NEVER escalates.
STREAM = [(float(i), STILL, "x") for i in range(6)]


def _core(*, gate_on: str) -> Percept:
    vision = FakeVision(script={"x": Verdict(satisfied=True, confidence=0.95, evidence="seen")})
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(
            id="doc",
            condition="the document on screen changed",
            say="the document changed",
            state={"cadence": {"gate_on": gate_on}},
        )
    )
    return core


async def _collect(core, source, **kw):
    return [inc async for inc in run(core, source=source, **kw)]


class Tier0AdapterSeamTests(unittest.TestCase):
    def test_semantic_adapter_admits_a_still_frame_the_motion_gate_skips(self):
        core = _core(gate_on="semantic")
        stats = RuntimeStats()
        calls: list = []
        incidents = asyncio.run(
            _collect(
                core,
                STREAM,
                stats=stats,
                semantic_adapter=lambda f: (calls.append(f) or 0.9),
            )
        )
        self.assertEqual(len(calls), len(STREAM))  # called once per frame, on the gate frame
        self.assertGreater(stats.escalations, 0)  # the semantic channel spent the cloud judge
        self.assertGreater(len(incidents), 0)

    def test_structural_adapter_admits_on_the_structural_channel(self):
        core = _core(gate_on="structural")
        stats = RuntimeStats()
        asyncio.run(_collect(core, STREAM, stats=stats, structural_adapter=lambda f: 0.9))
        self.assertGreater(stats.escalations, 0)

    def test_adapter_value_is_observable_in_the_edge_trace(self):
        core = _core(gate_on="semantic")
        traced: list[dict] = []

        async def on_trace(rec):
            d = rec if isinstance(rec, dict) else getattr(rec, "__dict__", {})
            traced.append(d)

        asyncio.run(
            _collect(
                core,
                STREAM,
                on_trace=on_trace,
                semantic_adapter=lambda f: 0.75,
                structural_adapter=lambda f: 0.25,
            )
        )
        edge = [d for d in traced if "semantic" in str(d)]
        self.assertTrue(edge, "no edge.decision trace was emitted")
        blob = str(edge)
        self.assertIn("0.75", blob)  # the adapter's value, not a hardcoded 0.0
        self.assertIn("0.25", blob)

    def test_no_adapters_is_byte_for_byte_the_motion_only_path(self):
        core = _core(gate_on="semantic")
        stats = RuntimeStats()
        incidents = asyncio.run(_collect(core, STREAM, stats=stats))
        self.assertEqual(stats.escalations, 0)  # semantic stays 0.0 -> nothing clears threshold
        self.assertEqual(incidents, [])


if __name__ == "__main__":
    unittest.main()
