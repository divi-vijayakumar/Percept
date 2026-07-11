"""PR 2.3 — GeminiLiveVision no longer silently drops exemplar + corrections.

Before: `judge` did `del audio, roi, exemplar, corrections`, so a deictic reference image and
user-asserted authoritative facts vanished with no signal (confidently-wrong verdicts on
vision="gemini-live"). These assert both now reach the spoken-judge prompt/parts.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from percept.backends.gemini_live import GeminiLiveVision


class _FakePart:
    """Stand-in for google.genai types.Part — text parts and image parts, no SDK."""

    def __init__(self, *, text=None):
        self.text = text

    @staticmethod
    def from_bytes(*, data, mime_type):
        return ("IMG", mime_type)


FAKE_TYPES = SimpleNamespace(Part=_FakePart)


def _texts_and_images(parts):
    texts = [p.text for p in parts if isinstance(p, _FakePart) and p.text]
    images = [p for p in parts if isinstance(p, tuple) and p and p[0] == "IMG"]
    return " ".join(texts), images


class GeminiLiveGroundingTests(unittest.TestCase):
    def setUp(self):
        self.v = GeminiLiveVision()  # ctor loads settings; no client needed for _build_parts

    def test_exemplar_attached_and_deictic(self):
        parts = self.v._build_parts(
            "the mug is full", ("image/jpeg", b"frame"), FAKE_TYPES,
            exemplar=("image/jpeg", b"reference"),
        )
        text, images = _texts_and_images(parts)
        self.assertEqual(len(images), 2)  # frame + exemplar (was 1 — exemplar dropped)
        self.assertIn("THIS specific object", text)

    def test_corrections_reach_the_prompt(self):
        parts = self.v._build_parts(
            "the light is on", ("image/jpeg", b"frame"), FAKE_TYPES,
            corrections={"bulb": "LED", "state": "off"},
        )
        text, _ = _texts_and_images(parts)
        self.assertIn("authoritative facts", text)
        self.assertIn("bulb=LED", text)
        self.assertIn("state=off", text)

    def test_plain_judge_unchanged_without_grounding(self):
        parts = self.v._build_parts("a person is here", ("image/jpeg", b"frame"), FAKE_TYPES)
        text, images = _texts_and_images(parts)
        self.assertEqual(len(images), 1)
        self.assertNotIn("authoritative facts", text)
        self.assertNotIn("THIS specific object", text)


if __name__ == "__main__":
    unittest.main()
