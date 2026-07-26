"""— the resolved WatchProfile.plan_mode is LIVE: "window" makes the judge see an ORDERED
FRAME WINDOW (``judge(frames=[...])``), not one still frame.

The first profile resolver carried ``plan_mode`` onto the core and nothing read it. These tests pin the wiring end-to-end with a
recording vision stub (no model intelligence, no keys):

  1. ``_planned_perception`` — the pure overlay: a profile's plan_mode is a FLOOR (it raises a cheaper
     spec, never downgrades a richer authored one), ``None`` is byte-for-byte legacy behavior, an unknown mode is
     ignored, and the deprecated ``sprite`` normalizes to ``window`` with a DeprecationWarning.
  2. the agent path — ``Percept.create(plan_mode="window")`` sends the ordered earlier frames with the
     current frame as the anchor; ``plan_mode=None`` still sends ``frames=None``.
  3. ``watch(profile=...)`` — a real default profile whose plan_mode is "window" reaches the core.
  4. latest-wins — the window a judgment sees is anchored on ITS OWN admitted frame; a frame the async
     scheduler dropped (superseded/stale) never enters any window.
"""

from __future__ import annotations

import asyncio
import unittest
import warnings

from percept.adapters import Goal, Verdict
from percept.agent import Percept, _planned_perception
from percept.behavior import PerceptionSpec
from percept.profile_wiring import resolve_watch_profile
from percept.profiles_default import ALL_PROFILES


class RecordingVision:
    def __init__(self):
        self.calls: list[dict] = []

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None):
        self.calls.append({"frame": frame, "frames": frames})
        return Verdict(satisfied=False, confidence=0.5, evidence="stub")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _watch_goal(perception: dict | None = None) -> Goal:
    state: dict = {"type": "watch", "count": 0}
    if perception:
        state["perception"] = perception
    return Goal(id="g", condition="c", say="s", state=state)


def _run(plan_mode, perception=None, frames=("f0", "f1", "f2")):
    rec = RecordingVision()
    agent = Percept.create(vision=rec, discover_plugins=False, plan_mode=plan_mode)
    agent.add_goal(_watch_goal(perception))

    async def go():
        for i, f in enumerate(frames):
            await agent.perceive_judged(f, t=float(i))

    asyncio.run(go())
    return rec


class PlannedPerceptionOverlayTests(unittest.TestCase):
    """The pure overlay — a profile's plan_mode is a floor, never a downgrade."""

    def test_none_plan_mode_is_byte_for_byte_pre_j2(self):
        base = PerceptionSpec()
        self.assertIs(_planned_perception(base, None), base)
        self.assertIs(_planned_perception(base, ""), base)

    def test_window_raises_a_single_frame_spec(self):
        spec = _planned_perception(PerceptionSpec(), "window")
        self.assertEqual(spec.temporal, "window")
        self.assertGreaterEqual(spec.n_frames, 3)  # a widened plan is never left at n_frames=1

    def test_window_raises_a_strip_spec(self):
        spec = _planned_perception(PerceptionSpec(temporal="strip", n_frames=4), "window")
        self.assertEqual((spec.temporal, spec.n_frames), ("window", 4))  # authored width preserved

    def test_never_downgrades_a_richer_authored_spec(self):
        authored = PerceptionSpec(temporal="window", n_frames=5)
        self.assertIs(_planned_perception(authored, "single"), authored)
        self.assertIs(_planned_perception(authored, "strip"), authored)
        strip = PerceptionSpec(temporal="strip", n_frames=4)
        self.assertIs(_planned_perception(strip, "single"), strip)

    def test_unknown_plan_mode_is_ignored_not_trusted(self):
        base = PerceptionSpec()
        self.assertIs(_planned_perception(base, "telepathy"), base)

    def test_other_perception_axes_are_untouched(self):
        base = PerceptionSpec(modality="fused", roi="full", sampling="uniform", audio_s=2.0,
                              window_s=4.0)
        spec = _planned_perception(base, "window")
        self.assertEqual(
            (spec.modality, spec.roi, spec.sampling, spec.audio_s, spec.window_s),
            ("fused", "full", "uniform", 2.0, 4.0),
        )


class SpriteDeprecationShimTests(unittest.TestCase):
    def test_sprite_plan_mode_becomes_window_with_a_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = _planned_perception(PerceptionSpec(), "sprite")
        self.assertEqual(spec.temporal, "window")
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_authored_sprite_spec_becomes_window_with_a_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = _planned_perception(PerceptionSpec(temporal="sprite", n_frames=4), None)
        self.assertEqual((spec.temporal, spec.n_frames), ("window", 4))
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_live_modes_never_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for mode in ("single", "strip", "window", None):
                _planned_perception(PerceptionSpec(), mode)
        self.assertEqual([w for w in caught if issubclass(w.category, DeprecationWarning)], [])


class AgentAssemblyTests(unittest.TestCase):
    """The live path: plan_mode="window" -> judge(frames=[ordered earlier frames])."""

    def test_window_plan_sends_the_ordered_earlier_frames(self):
        rec = _run("window")
        last = rec.calls[-1]
        self.assertEqual(last["frame"], "f2")          # anchor passed separately, as always
        self.assertEqual(last["frames"], ["f0", "f1"])  # oldest→newest, current excluded

    def test_no_plan_mode_stays_single_frame(self):
        rec = _run(None)
        self.assertTrue(all(c["frames"] is None for c in rec.calls))

    def test_single_plan_mode_stays_single_frame(self):
        rec = _run("single")
        self.assertTrue(all(c["frames"] is None for c in rec.calls))

    def test_window_plan_does_not_truncate_to_n_frames_like_a_strip(self):
        # a strip is bounded by n_frames-1; a window takes every earlier frame it holds
        rec = _run("window", frames=("f0", "f1", "f2", "f3", "f4"))
        self.assertEqual(rec.calls[-1]["frames"], ["f0", "f1", "f2", "f3"])
        strip = _run("strip", frames=("f0", "f1", "f2", "f3", "f4"))
        self.assertEqual(len(strip.calls[-1]["frames"]), 2)  # n_frames floor 3 -> 2 earlier

    def test_authored_window_is_unchanged_by_a_single_plan_mode(self):
        rec = _run("single", perception={"temporal": "strip", "n_frames": 3})
        self.assertEqual(rec.calls[-1]["frames"], ["f0", "f1"])  # authored spec wins

    def test_first_frame_has_no_earlier_context(self):
        rec = _run("window", frames=("f0",))
        self.assertIsNone(rec.calls[0]["frames"])  # nothing earlier -> a plain single-frame judge


class DefaultProfilesShipWindowTests(unittest.TestCase):
    def test_dynamic_default_profiles_resolve_to_a_window_plan(self):
        window_profiles = [p for p in ALL_PROFILES if p.plan_mode == "window"]
        self.assertTrue(window_profiles, "the seeded defaults must carry the validated window plan")
        for profile in window_profiles:
            self.assertEqual(resolve_watch_profile(profile).plan_mode, "window")

    def test_resolved_plan_mode_reaches_the_core_and_the_judge(self):
        profile = next(p for p in ALL_PROFILES if p.plan_mode == "window")
        resolved = resolve_watch_profile(profile)
        rec = RecordingVision()
        agent = Percept.create(vision=rec, discover_plugins=False, plan_mode=resolved.plan_mode)
        agent.add_goal(_watch_goal())

        async def go():
            for i, f in enumerate(("f0", "f1", "f2")):
                await agent.perceive_judged(f, t=float(i))

        asyncio.run(go())
        self.assertEqual(rec.calls[-1]["frames"], ["f0", "f1"])


class LatestWinsTests(unittest.TestCase):
    """The async scheduler's latest-wins semantics: a judgment's window is anchored on its OWN admitted
    frame, and a frame that lost the race contributes to nothing."""

    def test_window_is_anchored_on_the_frame_being_judged(self):
        rec = _run("window", frames=("f0", "f1", "f2"))
        for i, call in enumerate(rec.calls):
            self.assertEqual(call["frame"], f"f{i}")
            expected = [f"f{j}" for j in range(i)] or None
            self.assertEqual(call["frames"], expected)

    def test_a_superseded_frame_never_enters_a_later_window(self):
        # A frame the scheduler drops never reaches perceive_judged, so it is never pushed into the
        # rolling window: the next judged frame's context contains only frames that were actually judged.
        rec = RecordingVision()
        agent = Percept.create(vision=rec, discover_plugins=False, plan_mode="window")
        agent.add_goal(_watch_goal())

        async def go():
            await agent.perceive_judged("f0", t=0.0)
            # "f1" is admitted but superseded by "f2" before launch -> perceive_judged is never called
            await agent.perceive_judged("f2", t=2.0)

        asyncio.run(go())
        self.assertEqual(rec.calls[-1]["frames"], ["f0"])
        self.assertNotIn("f1", rec.calls[-1]["frames"])


if __name__ == "__main__":
    unittest.main()
