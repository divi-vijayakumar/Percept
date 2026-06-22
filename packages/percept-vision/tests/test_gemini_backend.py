"""Gemini backend contract (L1) — resolves through the registry + conforms to the seam protocols,
import-safe with no key and no API calls (the client is lazy; construction never touches google.genai).
"""

import unittest

from percept import registry
from percept.backends.gemini import (
    GEMINI_DEFAULT,
    GeminiConductor,
    GeminiSpecialist,
    GeminiVision,
    _image_part,
    _model_for,
)


class TestRegistryResolves(unittest.TestCase):
    def test_make_resolves_gemini_backends(self):
        self.assertIsInstance(registry.make_vision("gemini"), GeminiVision)
        self.assertIsInstance(registry.make_conductor("gemini"), GeminiConductor)
        self.assertIsInstance(registry.make_specialist("gemini"), GeminiSpecialist)

    def test_gemini_listed_as_available(self):
        for seam in ("vision", "conductor", "specialist"):
            self.assertIn("gemini", registry.available(seam))


class TestProtocolConformance(unittest.TestCase):
    def test_vision_protocol_surface(self):
        v = GeminiVision()
        for m in ("judge", "detect", "detect_events"):
            self.assertTrue(callable(getattr(v, m)), m)

    def test_conductor_and_specialist_surface(self):
        self.assertTrue(callable(getattr(GeminiConductor(), "run")))
        self.assertTrue(callable(getattr(GeminiSpecialist(), "stream")))


class TestModelResolution(unittest.TestCase):
    def test_gemini_pin_kept_claude_pin_falls_back(self):
        class GeminiPinned:
            class models:
                vision_model = "gemini-3-flash"

        self.assertEqual(_model_for(GeminiPinned, "vision_model"), "gemini-3-flash")

        class ClaudePinned:  # a Claude-pinned Settings still works with vision="gemini"
            class models:
                vision_model = "claude-haiku-4-5"

        self.assertEqual(_model_for(ClaudePinned, "vision_model"), GEMINI_DEFAULT)


class TestImagePart(unittest.TestCase):
    def test_non_image_frame_rejected_before_touching_genai(self):
        # the non-image branch raises TypeError before any google.genai call, so `types` is unused
        with self.assertRaises(TypeError):
            _image_part(None, "not-an-image-token")


if __name__ == "__main__":
    unittest.main()
