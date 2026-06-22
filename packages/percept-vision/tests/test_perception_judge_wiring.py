"""judge(frames=..., audio=...) wiring (L1) — the PerceptionSpec input actually reaches the eye.

Proves the additive ``frames``/``audio`` kwargs flow through ``Vision.judge`` and can change the call
(routed to a distinct scripted verdict), WITHOUT asserting any model intelligence (rule 2 — FakeVision
is scripted wiring). Also pins the gate-input invariant: every judge path returns a ``Verdict`` with the
same three fields, so nothing downstream of the eye changes shape.
"""

import asyncio
import unittest

from percept.adapters import Verdict
from percept.backends.fake import FakeVision

# "boil" alone reads NO; the strip/audio escalations are scripted to flip it — so a passing test means
# the extra modality genuinely reached judge (not that the fake "understood" anything).
_SCRIPT = {
    "boil": Verdict(satisfied=False, confidence=0.4, evidence="single: ambiguous"),
    "boil|strip": Verdict(satisfied=True, confidence=0.9, evidence="strip: rising"),
    "boil|audio": Verdict(satisfied=True, confidence=0.95, evidence="audio: hiss"),
}


def _judge(**kwargs) -> Verdict:
    v = FakeVision(script=_SCRIPT)
    return asyncio.run(v.judge("the milk is about to boil over", "boil", **kwargs))


class TestJudgePerceptionWiring(unittest.TestCase):
    def test_single_frame_is_unchanged(self):
        # byte-for-byte: no frames/audio -> the single-frame verdict
        self.assertFalse(_judge().satisfied)
        self.assertEqual(_judge().evidence, "single: ambiguous")

    def test_strip_reaches_judge(self):
        v = _judge(frames=["t0", "t1", "boil"])  # earlier context + current
        self.assertTrue(v.satisfied)
        self.assertEqual(v.evidence, "strip: rising")

    def test_audio_reaches_judge(self):
        v = _judge(audio=("audio/wav", b"\x00\x01"))
        self.assertTrue(v.satisfied)
        self.assertEqual(v.evidence, "audio: hiss")

    def test_empty_strip_does_not_route(self):
        # frames=[] is falsy -> still the single-frame path (no spurious escalation)
        self.assertEqual(_judge(frames=[]).evidence, "single: ambiguous")

    def test_audio_takes_precedence_over_strip(self):
        v = _judge(frames=["t0", "boil"], audio=b"\x00")
        self.assertEqual(v.evidence, "audio: hiss")  # the fused/escalated read wins

    def test_verdict_shape_is_invariant_across_paths(self):
        # the gate downstream only ever sees Verdict{satisfied, confidence, evidence}
        for v in (_judge(), _judge(frames=["a", "boil"]), _judge(audio=b"\x00")):
            self.assertIsInstance(v, Verdict)
            self.assertIsInstance(v.satisfied, bool)
            self.assertIsInstance(v.confidence, float)
            self.assertIsInstance(v.evidence, str)


if __name__ == "__main__":
    unittest.main()
