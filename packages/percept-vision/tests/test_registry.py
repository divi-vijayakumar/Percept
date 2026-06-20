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
        # A vendor whose adapter module doesn't ship yet -> clear runtime error at resolve time.
        with self.assertRaises(RuntimeError) as ctx:
            registry.make_vision("gemini")
        self.assertIn("not available", str(ctx.exception))

    def test_claude_vision_resolves_but_needs_key_or_sdk(self):
        # Phase 2: the Claude vision adapter now SHIPS and imports cleanly (the core never imports
        # a vendor SDK at module top level — invariant 7). Construction must succeed with no key;
        # the lazy ``anthropic`` import / key requirement only bites when a method is actually
        # called (L2 boundary). Without the SDK installed, calling raises the actionable error.
        import asyncio

        v = registry.make_vision("claude")
        self.assertEqual(type(v).__name__, "AnthropicVision")
        try:
            import anthropic  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(v.judge("is this coffee?", b"\x89PNG-bytes"))
            self.assertIn("anthropic", str(ctx.exception).lower())

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
