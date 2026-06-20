"""L1: voice-seam registry wiring + lazy vendor-import discipline.

These assert **wiring**, not model behaviour: the conductor/specialist 'claude' names and the
deepgram stt/tts names resolve to the right modules, and constructing a real backend never
imports a vendor SDK at module top level (invariant 7) — the import is deferred to first use,
and is reported as a clear, actionable error when the optional dependency is absent.
"""

import importlib
import sys
import unittest

from percept import registry
from percept.config import load_settings


class TestVoiceRegistryWiring(unittest.TestCase):
    def test_conductor_claude_resolves_to_anthropic_text(self):
        # Module ships in this phase; the class is importable even without the SDK installed.
        mod = importlib.import_module("percept.backends.anthropic_text")
        self.assertTrue(hasattr(mod, "AnthropicConductor"))
        self.assertTrue(hasattr(mod, "AnthropicSpecialist"))
        # The registry name resolves to a constructed adapter of that class.
        cond = registry.make_conductor("claude", load_settings())
        self.assertEqual(type(cond).__name__, "AnthropicConductor")
        spec = registry.make_specialist("claude", load_settings())
        self.assertEqual(type(spec).__name__, "AnthropicSpecialist")

    def test_deepgram_stt_tts_resolve(self):
        mod = importlib.import_module("percept.backends.deepgram")
        self.assertTrue(hasattr(mod, "DeepgramSTT"))
        self.assertTrue(hasattr(mod, "DeepgramTTS"))
        self.assertEqual(type(registry.make_stt("deepgram", load_settings())).__name__, "DeepgramSTT")
        self.assertEqual(type(registry.make_tts("deepgram", load_settings())).__name__, "DeepgramTTS")


class TestNoVendorImportAtModuleTop(unittest.TestCase):
    def test_importing_voice_backends_does_not_pull_a_vendor_sdk(self):
        # Importing the adapter modules must not import 'anthropic' or 'deepgram' (invariant 7).
        for vendor in ("anthropic", "deepgram"):
            sys.modules.pop(vendor, None)
        importlib.import_module("percept.backends.anthropic_text")
        importlib.import_module("percept.backends.deepgram")
        self.assertNotIn("anthropic", sys.modules)
        self.assertNotIn("deepgram", sys.modules)

    def test_constructing_conductor_does_not_pull_the_sdk(self):
        # The client is lazy; construction alone must not import the vendor SDK.
        sys.modules.pop("anthropic", None)
        registry.make_conductor("claude", load_settings())
        self.assertNotIn("anthropic", sys.modules)


class TestTTSBufferingIsOfflineSafe(unittest.TestCase):
    def test_push_without_flush_buffers_and_needs_no_key(self):
        # push() is pure buffering — assertable offline; flush() (which would synthesize) is L2.
        from percept.backends.deepgram import DeepgramTTS

        tts = DeepgramTTS(load_settings())
        tts.push("hello ")
        tts.push("world")
        self.assertEqual(tts._buffer, ["hello ", "world"])  # buffered, nothing synthesized


if __name__ == "__main__":
    unittest.main()
