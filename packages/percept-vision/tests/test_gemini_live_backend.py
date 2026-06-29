"""Gemini Live backend (L1) — spoken-verdict parsing + registry + bench backend selection. No network."""
import os
import unittest

from percept import registry
from percept.backends.gemini_live import parse_spoken_verdict


class SpokenVerdictParseTests(unittest.TestCase):
    def test_yes_with_confidence_and_reason(self):
        v = parse_spoken_verdict("YES confidence 0.9 because the square is red")
        self.assertTrue(v.satisfied)
        self.assertAlmostEqual(v.confidence, 0.9)
        self.assertIn("square is red", v.evidence)

    def test_no_is_not_satisfied(self):
        v = parse_spoken_verdict("NO confidence 0.7 because there is no circle")
        self.assertFalse(v.satisfied)
        self.assertAlmostEqual(v.confidence, 0.7)

    def test_first_yes_no_token_wins(self):
        self.assertTrue(parse_spoken_verdict("yes it is").satisfied)          # yes before any no
        self.assertFalse(parse_spoken_verdict("no, this is not a yes").satisfied)  # no before yes

    def test_empty_transcript_degrades_to_low_confidence_unknown(self):
        v = parse_spoken_verdict("")
        self.assertFalse(v.satisfied)
        self.assertEqual(v.confidence, 0.0)

    def test_only_in_range_confidence_is_taken(self):
        v = parse_spoken_verdict("YES 90 percent, confidence 0.85 because red")  # 90 rejected, 0.85 kept
        self.assertAlmostEqual(v.confidence, 0.85)

    def test_registry_resolves_gemini_live(self):
        self.assertIn("gemini-live", registry._REGISTRY["vision"])


class BenchVisionSelectionTests(unittest.TestCase):
    def test_bench_vision_reads_env_backend(self):
        from eval.bench.types import bench_vision

        old = os.environ.get("PERCEPT_VISION_BACKEND")
        try:
            os.environ["PERCEPT_VISION_BACKEND"] = "fake"  # avoids any key/SDK
            self.assertEqual(type(bench_vision(None)).__name__, "FakeVision")
        finally:
            if old is None:
                os.environ.pop("PERCEPT_VISION_BACKEND", None)
            else:
                os.environ["PERCEPT_VISION_BACKEND"] = old


if __name__ == "__main__":
    unittest.main()
