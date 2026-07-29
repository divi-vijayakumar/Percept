"""A missing or broken pose provider must fail ONE rhythm goal, never the whole watch.

The defect: ``rhythm_drive.observe(t, judge_frame)`` (``percept/runtime/worker.py``, the frame loop)
had no try/except. A missing pose provider raises ``RuntimeError`` (``percept/harness/pose.py:89-93``,
``RTMOPoseProvider._ensure``) straight OUT of the frame loop, on the first frame, after arming already
succeeded silently — killing the watch for every OTHER goal on the source too.
``percept.harness.pose.ensure_pose_ready`` already existed to pre-flight this at arm time but had zero
callers.

Two containment layers, proven separately because a single test cannot cover both (pre-flight alone
would leave the frame-loop crash still reachable through a provider that only fails LATER):

  * arm-time refusal — a rhythm goal whose provider's ``warm()`` fails is refused at
    :class:`~percept.rhythm.RhythmDrive` construction (``RhythmDrive.arm_failures``); its ``detect()``
    is never reached at all.
  * mid-run containment — a provider whose ``warm()`` succeeds (passes pre-flight) and then raises
    INSIDE ``detect()`` (the exact shape of ``percept/harness/pose.py:97-100``,
    ``RTMOPoseProvider.detect``) is caught in :meth:`RhythmDrive.observe`, reported exactly once via
    :meth:`RhythmDrive.pop_mid_run_failures`, and the goal is dropped — loudly, never silently
    degraded to counting zero (the RAISE-over-silent-zero choice ``percept/rhythm.py`` documents).

Both provider doubles below implement the WHOLE ``PoseProvider`` protocol (``warm()`` AND ``detect()``)
and raise unconditionally rather than tolerating anything — a permissive double here would recreate
the exact blind spot this fix closes (see ``test_rhythm_route_runtime.py``'s ``_PixelAssertingPose`` for
the established convention this follows).
"""

from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.harness.detectors import DetectorManifest, PoseChannels, register
from percept.rhythm import RhythmDrive
from percept.runtime import run
from percept.runtime.worker import RuntimeHealthEvent

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is an extra, absent in the L1 stdlib lane
    _HAVE_NUMPY = False

FPS = 6.0
W, H = 8, 8


def _pixels() -> "np.ndarray":
    """An already-decoded RGB uint8 HxWx3 buffer — ``frame_pixels`` passes this through untouched, so
    it reaches a pose provider's ``detect()`` without needing a real JPEG encode/decode round trip."""
    return np.zeros((H, W, 3), dtype=np.uint8)


class _RefusingAtArmProvider:
    """The 'no pose provider available' shape: ``warm()`` fails exactly like
    ``RTMOPoseProvider._ensure()`` does when mmpose is not installed (percept/harness/pose.py:89-93).

    ``detect()`` must NEVER be reached if arm-time pre-flight is doing its job — it raises
    ``AssertionError`` rather than tolerating a call, so a regression that lets this goal arm anyway
    fails LOUDLY here instead of quietly returning a plausible pose.
    """

    def __init__(self) -> None:
        self.detect_calls = 0

    def warm(self) -> str:
        raise RuntimeError(
            "RTMO pose runtime is unresolved; missing Python module 'mmpose'. Install the full "
            "RTMO/MMPose runtime, including compiled MMCV ops, and set PERCEPT_RTMO_MODEL if needed."
        )

    def detect(self, frame):
        self.detect_calls += 1
        raise AssertionError(
            "detect() must never be called — arm-time pre-flight should already have refused this goal"
        )


class _RaisingAtDetectProvider:
    """The 'passes pre-flight, then raises inside detect' shape (percept/harness/pose.py:97-100,
    ``RTMOPoseProvider.detect``: ``_ensure()``/``warm()`` succeed, then the inference call itself
    errors). ``warm()`` succeeds unconditionally; every ``detect()`` call raises — a persistent
    backend failure, not a one-off bad frame.
    """

    def __init__(self) -> None:
        self.detect_calls = 0

    def warm(self) -> str:
        return "test-raising-at-detect"

    def detect(self, frame):
        self.detect_calls += 1
        raise RuntimeError("simulated inference backend crash: CUDA context lost")


def _rhythm_goal(goal_id: str, detector_name: str) -> Goal:
    return Goal(
        id=goal_id,
        condition="a completed push-up repetition",
        say="rep",
        state={"type": "rhythm", "count": 0, "detectors": [detector_name]},
    )


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed pose eye reads pixels)")
class ArmTimeRefusalTests(unittest.TestCase):
    """Acceptance #1 + #4: a rhythm goal with no pose provider is refused/failed AT ARM, naming the
    goal, the missing extra, and the frame-driven alternative — before any frame is ever measured."""

    def test_a_rhythm_goal_with_no_pose_provider_is_refused_at_arm_not_observed(self):
        provider = _RefusingAtArmProvider()
        register(
            "test-refuse-at-arm",
            lambda: PoseChannels(provider=provider),
            DetectorManifest(
                "test-refuse-at-arm", "pose channels behind a provider whose warm() fails",
                "arm-time refusal test", "per-frame channel measurements",
            ),
        )
        goal = _rhythm_goal("pushups", "test-refuse-at-arm")
        core = Percept.create(mode="offline", discover_plugins=False)
        core.add_goal(goal)

        drive = RhythmDrive(core, [goal])

        # Refused, not armed: the goal never enters the observed set...
        self.assertEqual(drive.goal_ids, ())
        # ...and observing frames afterward touches nothing for it (never armed, never silently counts).
        fires = drive.observe(0.0, _pixels())
        self.assertEqual(fires, [])
        self.assertEqual(drive.count("pushups"), 0)
        self.assertEqual(provider.detect_calls, 0)  # detect() truly never reached — pre-flight, not luck

        # The refusal is loud and specific: names the GOAL, the missing EXTRA, and the frame-driven
        # ALTERNATIVE that needs no pose model at all.
        self.assertIn("pushups", drive.arm_failures)
        message = drive.arm_failures["pushups"]
        self.assertIn("pushups", message)
        self.assertIn("percept-vision[pose]", message)
        self.assertIn("motion-periodicity", message)
        self.assertIn("mmpose", message)  # the underlying reason is preserved, not swallowed


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed pose eye reads pixels)")
class MidRunContainmentTests(unittest.TestCase):
    """Acceptance #2 + #4: a provider that PASSES pre-flight and then raises inside detect() (the
    exact shape of percept/harness/pose.py:97-100) must not escape RhythmDrive.observe — today it
    escapes worker.py:6579 uncontained. Pre-flight alone (ArmTimeRefusalTests above) cannot catch
    this: warm() already returned cleanly, so the goal is armed."""

    def test_a_later_detect_raise_is_caught_not_propagated_and_reported_exactly_once(self):
        provider = _RaisingAtDetectProvider()
        register(
            "test-raise-at-detect",
            lambda: PoseChannels(provider=provider),
            DetectorManifest(
                "test-raise-at-detect", "pose channels behind a provider whose detect() fails",
                "mid-run containment test", "per-frame channel measurements",
            ),
        )
        goal = _rhythm_goal("situps", "test-raise-at-detect")
        core = Percept.create(mode="offline", discover_plugins=False)
        core.add_goal(goal)

        drive = RhythmDrive(core, [goal])
        # Pre-flight PASSED: the goal is armed (proves this failure mode is invisible to arm-time
        # pre-flight alone, and so needs its OWN containment layer).
        self.assertEqual(drive.goal_ids, ("situps",))
        self.assertEqual(drive.arm_failures, {})

        frame = _pixels()
        # THE defect being pinned: this call must not raise.
        fires = drive.observe(0.0, frame)
        self.assertEqual(fires, [])
        self.assertEqual(provider.detect_calls, 1)

        # Reported loudly, exactly once — never silently downgraded to a phantom zero count.
        failures = drive.pop_mid_run_failures()
        self.assertIn("situps", failures)
        message = failures["situps"]
        self.assertIn("situps", message)
        self.assertIn("percept-vision[pose]", message)
        self.assertIn("motion-periodicity", message)
        self.assertIn("CUDA context lost", message)  # the underlying reason is preserved

        # Popping again is empty — a caller draining every frame must not re-report a goal already
        # surfaced. And the goal itself does not come back: it is dropped for the rest of the run,
        # even though the provider would happily keep raising forever.
        self.assertEqual(drive.pop_mid_run_failures(), {})
        fires_again = drive.observe(1.0 / FPS, frame)
        self.assertEqual(fires_again, [])
        self.assertEqual(provider.detect_calls, 1)  # never called again
        self.assertEqual(drive.goal_ids, ())


class _RaisingChannel:
    """A measuring eye that always raises — the SAME 'passes construction, fails on use' shape as
    ``_RaisingAtDetectProvider`` above, but detached from ``PoseChannels``/pixel decoding so this
    goal can share a source with a STRING-keyed :class:`~percept.backends.fake.FakeVision` script
    (which stringifies whatever frame it is handed) without also exercising the real pose decode
    path — that reachability is already proven directly, above."""

    name = "test-raising-channel"
    signal = "frame"

    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def measure(self, frame):
        self.calls += 1
        raise RuntimeError("simulated eye crash: no pose provider available")

    def push(self, t, frame):
        return []


class CoexistenceTests(unittest.TestCase):
    """Acceptance #3: a rhythm goal armed BESIDE a working judged Watch on the same source — the
    Watch still runs and fires; the rhythm goal is reported failed, not fatal."""

    async def _collect(self, core, source, **kw):
        return [incident async for incident in run(core, source=source, **kw)]

    def test_a_failing_rhythm_goal_does_not_end_the_watch_for_a_working_watch_goal(self):
        from percept.adapters import Verdict
        from percept.backends.fake import FakeVision
        from percept.harness.tier0 import gray_frame

        register(
            "test-raising-channel",
            _RaisingChannel,
            DetectorManifest(
                "test-raising-channel", "a measuring eye that always raises",
                "coexistence containment test", "no measurement — always fails",
            ),
        )
        vision = FakeVision(
            script={
                "hot": Verdict(satisfied=True, confidence=0.95, evidence="a person is present"),
                "cold": Verdict(satisfied=False, confidence=0.95, evidence="nobody"),
            }
        )
        core = Percept.create(mode="offline", vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="who", condition="a person is present", say="someone is here"))
        core.add_goal(_rhythm_goal("situps3", "test-raising-channel"))

        still = gray_frame(W, H, 0)
        moved = gray_frame(W, H, 255)
        source = [
            (i / FPS, moved if i % 2 else still, "hot" if i % 2 else "cold")
            for i in range(8)
        ]

        health_events: list[RuntimeHealthEvent] = []
        incidents = asyncio.run(
            self._collect(core, source, on_health=health_events.append)
        )

        # The judged Watch still ran and fired — one failing rhythm goal did not end the watch.
        by_goal = {incident.concern_id for incident in incidents}
        self.assertIn("who", by_goal)
        # The rhythm goal produced no incident of its own: it FAILED, it did not fire.
        self.assertNotIn("situps3", by_goal)

        # And the failure was reported, loudly, naming the goal — not fatal, not silent. Exactly
        # once across all 8 frames: the goal is dropped the instant it fails, never re-measured
        # (and so never re-reported) on any later frame of this same run.
        failed_events = [e for e in health_events if e.kind == "rhythm_goal_failed"]
        self.assertEqual(len(failed_events), 1, "expected exactly one rhythm_goal_failed health event")
        reasons = {e.data.get("goal_id"): e.data.get("reason") for e in failed_events}
        self.assertIn("situps3", reasons)
        self.assertIn("situps3", reasons["situps3"])
        self.assertIn("percept-vision[pose]", reasons["situps3"])
        self.assertIn("motion-periodicity", reasons["situps3"])


if __name__ == "__main__":
    unittest.main()
