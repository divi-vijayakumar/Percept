"""L1: DTO shapes, Protocol conformance, and the Phase-0 import + create demo."""

import asyncio
import unittest

import percept
from percept.adapters import (
    Conductor,
    Decision,
    Goal,
    Vision,
    Verdict,
)
from percept.backends.fake import FakeConductor, FakeVision


class TestShapes(unittest.TestCase):
    def test_verdict_defaults(self):
        v = Verdict(satisfied=True, confidence=0.8)
        self.assertEqual(v.evidence, "")

    def test_goal_default_state_is_watch(self):
        g = Goal(id="g1", condition="user drinks coffee", say="careful")
        self.assertEqual(g.state["type"], "watch")
        self.assertEqual(g.modality, "speak")

    def test_decision_defaults(self):
        d = Decision(route="ignore")
        self.assertEqual(d.look, "none")
        self.assertEqual(d.goal_ops, [])

    def test_fakes_satisfy_protocols(self):
        # runtime_checkable structural conformance — the seam contract holds.
        self.assertIsInstance(FakeVision(), Vision)
        self.assertIsInstance(FakeConductor(), Conductor)


class TestPhase0Demo(unittest.TestCase):
    def test_import_and_create_shell(self):
        a = percept.Percept.create(vision="fake")
        h = a.health()
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["backends"]["vision"], "FakeVision")
        self.assertTrue(percept.__version__)

    def test_created_vision_judges_scripted_frame(self):
        a = percept.Percept.create(vision="fake")
        out = asyncio.run(a.vision.judge("is this coffee?", "sip-coffee"))
        self.assertTrue(out.satisfied)
        out2 = asyncio.run(a.vision.judge("is this coffee?", "sip-water"))
        self.assertFalse(out2.satisfied)


if __name__ == "__main__":
    unittest.main()
