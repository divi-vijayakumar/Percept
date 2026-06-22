"""PerceptionSpec — the VLM-side HOW (L1). The brain authors HOW to look; the firewall guards it.

Three things, all pure (no SDK / no keys):
  1. coerce_authored_perception — the firewall on the untrusted conductor "perception" block (sibling of
     coerce_authored_behavior): closed vocab, malformed dropped to the safe single-frame default, numerics clamped.
  2. program_of — overlays the perception axis last (sense-preset default + authored override); an un-authored
     goal stays byte-for-byte PerceptionSpec() (single, vision-only, focus).
  3. _coerce_decision — wires a validated perception block into goal.state["perception"].
"""

import unittest

from percept.adapters import Goal
from percept.backends.anthropic_text import _coerce_decision
from percept.behavior import (
    PerceptionSpec,
    coerce_authored_perception,
    program_of,
)


class TestPerceptionSpecDefault(unittest.TestCase):
    def test_default_is_todays_behavior(self):
        # the defaults MUST reproduce today: one vision-only, focus-cropped frame
        p = PerceptionSpec()
        self.assertEqual(
            (p.modality, p.temporal, p.n_frames, p.roi, p.sampling, p.audio_s),
            ("vision", "single", 1, "focus", "recent", None),
        )


class TestCoerceAuthoredPerception(unittest.TestCase):
    def test_full_valid_block(self):
        self.assertEqual(
            coerce_authored_perception(
                {"modality": "fused", "temporal": "strip", "n_frames": 4, "roi": "hint",
                 "sampling": "uniform", "audio_s": 2.0}
            ),
            {"perception": {"modality": "fused", "temporal": "strip", "n_frames": 4, "roi": "hint",
                            "sampling": "uniform", "audio_s": 2.0}},
        )

    def test_partial_block_keeps_only_authored_fields(self):
        self.assertEqual(
            coerce_authored_perception({"modality": "fused"}), {"perception": {"modality": "fused"}}
        )

    def test_unknown_vocab_is_dropped(self):
        self.assertEqual(coerce_authored_perception({"modality": "telepathy"}), {})
        self.assertEqual(coerce_authored_perception({"temporal": "forever"}), {})
        self.assertEqual(coerce_authored_perception({"roi": "everywhere"}), {})
        self.assertEqual(coerce_authored_perception({"sampling": "random"}), {})

    def test_n_frames_clamped_and_floored(self):
        # a token-bomb strip is clamped to the ceiling, never trusted
        self.assertEqual(coerce_authored_perception({"n_frames": 200}), {"perception": {"n_frames": 6}})
        # < 1 is meaningless -> dropped; a bool is not a number -> dropped
        self.assertEqual(coerce_authored_perception({"n_frames": 0}), {})
        self.assertEqual(coerce_authored_perception({"n_frames": True}), {})

    def test_audio_s_clamped_and_dropped(self):
        self.assertEqual(coerce_authored_perception({"audio_s": 99}), {"perception": {"audio_s": 5.0}})
        self.assertEqual(coerce_authored_perception({"audio_s": 0}), {})
        self.assertEqual(coerce_authored_perception({"audio_s": -1}), {})

    def test_non_dict_and_empty_drop_to_nothing(self):
        self.assertEqual(coerce_authored_perception(None), {})
        self.assertEqual(coerce_authored_perception("strip"), {})
        self.assertEqual(coerce_authored_perception({}), {})
        self.assertEqual(coerce_authored_perception({"junk": 1}), {})


class TestProgramOfPerceptionOverlay(unittest.TestCase):
    def test_unauthored_goal_is_single_frame(self):
        # byte-for-byte: no authored perception -> the safe default
        g = Goal(id="g", condition="c", say="s")
        self.assertEqual(program_of(g).perception, PerceptionSpec())

    def test_authored_perception_overlays(self):
        g = Goal(id="g", condition="c", say="s")
        g.state.update(coerce_authored_perception({"temporal": "strip", "n_frames": 3}))
        p = program_of(g).perception
        self.assertEqual((p.temporal, p.n_frames), ("strip", 3))
        # unauthored fields keep the default
        self.assertEqual((p.modality, p.roi), ("vision", "focus"))

    def test_overlay_does_not_change_sense_or_rearm(self):
        # perception is orthogonal to the behavior axis: authoring HOW must not alter WHAT fires
        g = Goal(id="g", condition="c", say="s")
        g.state["rearm"] = "never"
        g.state.update(coerce_authored_perception({"roi": "full"}))
        prog = program_of(g)
        self.assertEqual(prog.sense, "judge")
        self.assertTrue(prog.key.suppress_false)  # the never-contract is intact
        self.assertEqual(prog.perception.roi, "full")


class TestCoerceDecisionWiring(unittest.TestCase):
    def _goal(self, *, behavior=None, perception=None):
        goal = {"id": "g", "condition": "c", "say": "s"}
        if behavior is not None:
            goal["behavior"] = behavior
        if perception is not None:
            goal["perception"] = perception
        payload = {"route": "respond_direct", "goal_ops": [{"op": "create", "goal": goal}]}
        return _coerce_decision(payload).goal_ops[0].goal

    def test_authored_perception_reaches_goal_state(self):
        g = self._goal(perception={"modality": "fused", "temporal": "strip", "n_frames": 4, "audio_s": 2.0})
        self.assertEqual(
            g.state["perception"], {"modality": "fused", "temporal": "strip", "n_frames": 4, "audio_s": 2.0}
        )
        # and it compiles into the program
        self.assertEqual(program_of(g).perception.modality, "fused")

    def test_no_perception_leaves_default_state(self):
        g = self._goal()
        self.assertNotIn("perception", g.state)  # nothing added -> single-frame default
        self.assertEqual(program_of(g).perception, PerceptionSpec())

    def test_malformed_perception_leaves_default_state(self):
        g = self._goal(perception={"modality": "telepathy", "n_frames": -3})
        self.assertNotIn("perception", g.state)
        self.assertEqual(program_of(g).perception, PerceptionSpec())

    def test_perception_and_behavior_coexist(self):
        g = self._goal(behavior={"rearm": "never"}, perception={"roi": "hint"})
        self.assertEqual(g.state["rearm"], "never")
        self.assertEqual(g.state["perception"], {"roi": "hint"})


if __name__ == "__main__":
    unittest.main()
