"""B1 (L1): temporal `judge_clip` + a Concern declaring it needs a CLIP.

All offline against the scripted FakeVision — wiring, not intelligence (standing rule 2). We assert:
  - a Concern's `window_s`/`clip` declaration maps to the perception sub-dict the agent consumes;
  - that declaration assembles a TIME-BOUNDED temporal strip from the rolling window, so the fake's
    `"{token}|strip"` escalation key is reached (the strip actually flows into `judge`);
  - the public `judge_clip(condition, frames)` seam returns a Verdict over a multi-frame clip;
  - `Concern.wants_clip` reflects the declaration.
"""

import asyncio
import unittest

from percept.adapters import Concern, Goal, Subject, Verdict
from percept.agent import Percept
from percept.behavior import perception_for_concern, program_of
from percept.backends.fake import FakeVision


def _concern(**kw) -> Concern:
    base = dict(id="c", identity=Subject(kind="entity", ref="x"), condition="someone is loitering", say="heads up")
    base.update(kw)
    return Concern(**base)


class TestConcernClipDeclaration(unittest.TestCase):
    def test_wants_clip_property(self):
        self.assertFalse(_concern().wants_clip)  # default: single-frame, unchanged
        self.assertTrue(_concern(clip=True).wants_clip)
        self.assertTrue(_concern(window_s=4.0).wants_clip)
        self.assertFalse(_concern(window_s=0.0).wants_clip)  # non-positive window is not a clip

    def test_perception_for_concern_maps_window_to_time_bounded_strip(self):
        self.assertEqual(perception_for_concern(_concern()), {})  # default -> untouched single frame
        self.assertEqual(
            perception_for_concern(_concern(window_s=4.0)),
            {"perception": {"temporal": "window", "window_s": 4.0}},
        )
        self.assertEqual(
            perception_for_concern(_concern(clip=True)),
            {"perception": {"temporal": "strip"}},
        )

    def test_window_s_clamped(self):
        spec = perception_for_concern(_concern(window_s=10_000.0))["perception"]
        self.assertLessEqual(spec["window_s"], 30.0)  # _MAX_WINDOW_S ceiling — no minutes-long strip

    def test_program_of_honors_a_concern_clip_perception(self):
        # The concern's perception sub-dict drops straight into goal.state and program_of consumes it.
        state = {"type": "watch", "count": 0, **perception_for_concern(_concern(window_s=4.0))}
        prog = program_of(Goal(id="c", condition="someone is loitering", say="s", state=state))
        self.assertEqual(prog.perception.temporal, "window")
        self.assertEqual(prog.perception.window_s, 4.0)


class TestClipStripAssembly(unittest.TestCase):
    def test_window_s_concern_assembles_a_time_bounded_strip_reaching_judge(self):
        # The fake routes "loiter|strip" to a distinct verdict ONLY when a non-empty strip reaches judge,
        # proving the window_s declaration actually assembled a clip (wiring, not intelligence).
        vision = FakeVision(
            script={
                "loiter": Verdict(satisfied=False, confidence=0.9, evidence="single frame: inconclusive"),
                "loiter|strip": Verdict(satisfied=True, confidence=0.92, evidence="clip: dwelling > window"),
            }
        )
        a = Percept.create(vision=vision, discover_plugins=False)
        state = {"type": "watch", "count": 0, **perception_for_concern(_concern(window_s=4.0))}
        a.add_goal(Goal(id="loiter-goal", condition="someone is loitering", say="heads up", state=state))

        async def go():
            # Frames within the 4s window; once an earlier in-window frame exists, the strip is
            # assembled and routes to the |strip clip verdict (satisfied), firing on the rising edge.
            all_events = []
            for t in (1.0, 2.0, 3.0, 4.0):
                all_events += await a.perceive_judged("loiter", t=t)
            return all_events

        events = asyncio.run(go())
        # satisfied=True (clip verdict) on the rising edge -> exactly one fire over the run.
        fires = [e for e in events if e.action == "fire"]
        self.assertEqual(len(fires), 1)  # the strip reached judge and fired once (refractory holds the rest)

    def test_window_s_excludes_frames_outside_the_time_bound(self):
        # Earlier frames OLDER than window_s seconds back must be dropped from the strip.
        captured: list = []

        class Recording:
            async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                            frames=None, audio=None):
                captured.append(frames)
                return Verdict(satisfied=False, confidence=0.5, evidence="stub")

            async def detect(self, frame, goals):
                return []

            async def detect_events(self, frames, goals):
                return []

        a = Percept.create(vision=Recording(), discover_plugins=False)
        state = {"type": "watch", "count": 0, **perception_for_concern(_concern(window_s=2.0))}
        a.add_goal(Goal(id="g", condition="c", say="s", state=state))

        async def go():
            await a.perceive_judged("old", t=0.0)    # 4s before the anchor -> outside a 2s window
            await a.perceive_judged("mid", t=3.0)    # 1s before the anchor -> inside the window
            await a.perceive_judged("now", t=4.0)    # the anchor

        asyncio.run(go())
        # Last call's strip: only "mid" (within 2s), "old" dropped, "now" is the anchor (excluded).
        self.assertEqual(captured[-1], ["mid"])


class TestJudgeClip(unittest.TestCase):
    def test_judge_clip_returns_a_verdict_over_a_clip(self):
        vision = FakeVision(
            script={"loiter|strip": Verdict(satisfied=True, confidence=0.9, evidence="clip")}
        )
        a = Percept.create(vision=vision, discover_plugins=False)
        v = asyncio.run(a.judge_clip("someone is loitering", ["loiter", "loiter", "loiter"]))
        self.assertIsInstance(v, Verdict)
        self.assertTrue(v.satisfied)  # the multi-frame strip reached judge -> the |strip verdict

    def test_judge_clip_single_frame_degrades(self):
        vision = FakeVision(script={"x": Verdict(satisfied=True, confidence=0.9, evidence="one")})
        a = Percept.create(vision=vision, discover_plugins=False)
        v = asyncio.run(a.judge_clip("c", ["x"]))  # single-element clip -> plain single-frame judge
        self.assertTrue(v.satisfied)

    def test_judge_clip_rejects_empty_clip(self):
        a = Percept.create(vision=FakeVision(), discover_plugins=False)
        with self.assertRaises(ValueError):
            asyncio.run(a.judge_clip("c", []))


if __name__ == "__main__":
    unittest.main()
