"""L1: registry resolution — fake names resolve, bad names + unavailable vendors error clearly."""

import asyncio
import unittest

from percept import registry
from percept.adapters import Verdict
from percept.backends.fake import FakeConductor, FakeVision


class TestRegistry(unittest.TestCase):
    def test_resolves_fake_vision(self):
        v = registry.make_vision("fake")
        self.assertIsInstance(v, FakeVision)

    def test_resolves_all_seam_fakes(self):
        self.assertIsNotNone(registry.make_specialist("fake"))
        self.assertIsInstance(registry.make_conductor("fake"), FakeConductor)
        self.assertIsNotNone(registry.make_stt("fake"))
        self.assertIsNotNone(registry.make_tts("fake"))

    def test_unknown_backend_name_raises_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            registry.make_vision("nope")
        msg = str(ctx.exception)
        self.assertIn("unknown vision backend", msg)
        self.assertIn("fake", msg)  # lists what's available

    def test_unknown_seam_raises(self):
        with self.assertRaises(ValueError):
            registry.make("smell", "fake")

    def test_unavailable_vendor_raises_actionable_error(self):
        # 'claude' is registered but its module doesn't ship in Phase 0 -> clear runtime error.
        with self.assertRaises(RuntimeError) as ctx:
            registry.make_vision("claude")
        self.assertIn("not available", str(ctx.exception))

    def test_available_lists_fake(self):
        self.assertIn("fake", registry.available("vision"))

    def test_runtime_registration_overrides(self):
        sentinel = object()
        registry.register("vision", "unit-test-only", lambda s: sentinel)
        try:
            self.assertIs(registry.make_vision("unit-test-only"), sentinel)
        finally:
            registry._REGISTRY["vision"].pop("unit-test-only", None)

    def test_fake_vision_is_scripted_not_intelligent(self):
        v = registry.make_vision("fake")
        out = asyncio.run(v.judge("is this coffee?", "sip-coffee"))
        self.assertIsInstance(out, Verdict)
        self.assertTrue(out.satisfied)


if __name__ == "__main__":
    unittest.main()
