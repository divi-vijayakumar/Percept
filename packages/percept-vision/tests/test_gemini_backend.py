"""Gemini backend contract (L1) — resolves through the registry + conforms to the seam protocols,
import-safe with no key and no API calls (the client is lazy; construction never touches google.genai).
"""

import asyncio
import types
import unittest
from unittest.mock import patch

from percept import registry
from percept.backends import gemini as gemini_backend
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


class TestEventChainPrompt(unittest.TestCase):
    def test_structured_phase_round_trip_and_reserved_context_filter(self):
        class Part:
            @staticmethod
            def from_bytes(**kwargs):
                return ("image", kwargs)

        class GenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        calls = []

        class Models:
            async def generate_content(self, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(
                    text=(
                        '{"satisfied": true, "confidence": 0.94, '
                        '"evidence": "visible temporal change", '
                        '"action_present": true, "phase": "during"}'
                    ),
                    usage_metadata=None,
                )

        fake_types = types.SimpleNamespace(
            Part=Part,
            GenerateContentConfig=GenerateContentConfig,
        )
        client = types.SimpleNamespace(
            aio=types.SimpleNamespace(models=Models())
        )
        with patch.object(
            gemini_backend,
            "_require_genai",
            return_value=(object(), fake_types),
        ):
            verdict = asyncio.run(
                GeminiVision(client=client).judge(
                    "one discrete putting occurrence",
                    b"current",
                    frames=[b"before"],
                    context={
                        "event_chain": True,
                        "_percept_frames_are_earlier": True,
                    },
                )
            )
        self.assertEqual((verdict.phase, verdict.action_present), ("during", True))
        prompt = calls[0]["contents"][-1]
        self.assertIn("pre (not begun)", prompt)
        self.assertIn('"action_present", "phase"', prompt)
        self.assertNotIn("_percept_frames_are_earlier", prompt)


if __name__ == "__main__":
    unittest.main()
