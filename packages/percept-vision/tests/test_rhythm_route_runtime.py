"""The EXTERNAL_RHYTHM route, driven end to end (L1) — ``type="rhythm"`` reaches ``percept.run()``.

Before this, five of the eight authored behavior routes had no driver: ``worker_can_drive`` admitted
only the three FRAME routes, so a ``type="rhythm"`` goal was refused at ``percept.run`` /
``GatedWatchLoop`` construction and ``RhythmTracker`` appeared nowhere in the runtime. The counting
primitive existed and was exercised only through ``core.perceive_rhythm`` called by hand.

What is proven here, offline, with a scripted channel signal and no provider call:

  * a rhythm goal ARMS through ``apply_watch_plan`` and RUNS through both frame drivers;
  * the eye is asked to MEASURE once per frame (not once per admission, not once per judgment) and
    the frame it receives is the real frame off the source;
  * the count comes from the SIGNAL — it is not capped by judgment density, and it spends no judge;
  * what an INCIDENT means for a rhythm goal: a count MILESTONE (an authored ``fire_when``), never
    one per rep. The un-authored default is silent-with-a-count.
"""

from __future__ import annotations

import asyncio
import math
import unittest

from percept import Goal, Percept
from percept.behavior_routes import (
    BehaviorRoute,
    behavior_route,
    drives_rhythm,
    goal_behavior_route,
    needs_watch_condition,
    worker_can_drive,
)
from percept.contracts.watchplan import WatchPlan
from percept.harness.detectors import (
    DetectorManifest,
    PoseChannels,
    make_detector,
    register,
)
from percept.harness.pose import FrameNotDecodable, PoseDetection, PoseKeypoint
from percept.harness.tier0 import Frame, GatedWatchLoop, gray_frame
from percept.rhythm import RhythmDrive, rhythm_emits_incidents
from percept.runtime import run
from percept.watchplan_apply import apply_watch_plan

W, H = 8, 8
PERIOD = 12  # samples per rep cycle
FPS = 6.0

_INSTANCES: list["_ScriptedChannel"] = []


class _ScriptedChannel:
    """Test-only MEASURING eye: one clean sinusoid per ``PERIOD`` frames on ``limb-vertical``.

    Stands in for ``pose-channels`` (whose real provider needs a pose model). It records every frame
    it was handed, so the test can prove the drive calls it once per frame with the real frame.
    """

    signal = "frame"
    name = "test-scripted-channel"

    def __init__(self) -> None:
        self.i = 0
        self.seen: list = []
        _INSTANCES.append(self)

    def reset(self) -> None:
        self.i = 0

    def measure(self, frame) -> dict[str, float]:
        self.seen.append(frame)
        v = math.sin(2 * math.pi * self.i / PERIOD)
        self.i += 1
        return {"limb-vertical": v}

    def push(self, t, frame):  # a measuring eye proposes no boundaries
        return []


class _BoundaryOnly:
    """A detector that proposes occurrences but MEASURES nothing — the wrong primitive for rhythm."""

    signal = "motion"
    name = "test-boundary-only"

    def reset(self) -> None:
        pass

    def push(self, t, v):
        return []


def _register_test_detectors() -> None:
    """(Re)register the fixtures. Called per-test: the detector registry is a process-global that
    other suites legitimately ``clear_detectors()``, and an unregistered name silently falls back to
    the real pose provider instead of failing here."""
    register(
        "test-scripted-channel",
        _ScriptedChannel,
        DetectorManifest(
            "test-scripted-channel", "a scripted oscillating channel", "rhythm route wiring test",
            "per-frame channel measurements",
        ),
    )
    register(
        "test-boundary-only",
        _BoundaryOnly,
        DetectorManifest(
            "test-boundary-only", "nothing", "negative test: a non-measuring detector",
            "no boundaries",
        ),
    )


class _RhythmFixture(unittest.TestCase):
    def setUp(self) -> None:
        _register_test_detectors()
        _INSTANCES.clear()


def _still() -> Frame:
    return gray_frame(W, H, 0)


def _rhythm_goal(goal_id: str = "reps", **state) -> Goal:
    return Goal(
        id=goal_id,
        condition="a completed push-up repetition",
        say="rep",
        state={
            "type": "rhythm",
            "count": 0,
            "detectors": ["test-scripted-channel"],
            **state,
        },
    )


def _core(goal: Goal) -> Percept:
    core = Percept.create(mode="offline", discover_plugins=False)
    core.add_goal(goal)
    return core


def _source(n_cycles: int = 8):
    """(t, gate_frame, judge_frame) — a still gate frame (nothing to admit) and a tagged judge frame."""
    return [(i / FPS, _still(), f"f{i}") for i in range(n_cycles * PERIOD)]


async def _collect(core, source, **kw):
    return [incident async for incident in run(core, source=source, **kw)]


class RhythmRouteContractTests(_RhythmFixture):
    """The route table itself: drivable, but NOT a tier-0 watch condition."""

    def test_rhythm_is_worker_drivable_but_never_compiles_to_a_watch_condition(self):
        route = behavior_route({"type": "rhythm"})
        self.assertIs(route, BehaviorRoute.EXTERNAL_RHYTHM)
        self.assertTrue(worker_can_drive(route))  # the worker HAS a driver for it now
        self.assertTrue(drives_rhythm(route))
        # ...but it is measured, not admitted: no WatchCondition, so no judge is ever spent on it.
        self.assertFalse(needs_watch_condition(route))

    def test_the_other_external_routes_are_still_undrivable(self):
        for behavior_type in ("event", "count", "remind", "scheduler"):
            with self.subTest(type=behavior_type):
                self.assertFalse(worker_can_drive(behavior_route({"type": behavior_type})))

    def test_a_rhythm_plan_arms_through_apply_watch_plan(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="reps",
                condition="a completed push-up repetition",
                explicit={"Goal.state.type": "rhythm", "Goal.state.rhythm.channel": "limb-vertical"},
            ),
        )
        self.assertEqual(receipt.status, "applied")
        armed = core.goals.get("reps")
        self.assertIs(goal_behavior_route(armed), BehaviorRoute.EXTERNAL_RHYTHM)
        self.assertEqual(armed.state["rhythm"]["channel"], "limb-vertical")


class RhythmIncidentPolicyTests(_RhythmFixture):
    """What an Incident MEANS for a rhythm goal: a count milestone, never a rep."""

    def test_only_an_authored_milestone_may_mint_incidents(self):
        self.assertFalse(rhythm_emits_incidents({"type": "rhythm"}))
        self.assertFalse(rhythm_emits_incidents({"type": "rhythm", "fire_when": "  "}))
        self.assertFalse(rhythm_emits_incidents(None))
        self.assertTrue(rhythm_emits_incidents({"type": "rhythm", "fire_when": "n == 20"}))

    def test_a_non_measuring_detector_is_a_loud_shape_error_not_a_silent_no_op(self):
        goal = _rhythm_goal(detectors=["test-boundary-only"])
        with self.assertRaises(ValueError) as caught:
            RhythmDrive(Percept.create(mode="offline", discover_plugins=False), [goal])
        self.assertIn("measure(frame)", str(caught.exception))


class RhythmThroughPerceptRunTests(_RhythmFixture):
    """percept.run(): a rhythm goal arms, runs, and counts — with no judge and no incident per rep."""

    def test_a_rhythm_goal_counts_through_run_and_stays_silent_by_default(self):
        core = _core(_rhythm_goal())
        source = _source(n_cycles=8)
        incidents = asyncio.run(_collect(core, source))

        self.assertAlmostEqual(core.rhythm["reps"].count, 8, delta=1)  # counted from the SIGNAL
        self.assertEqual(core.goals.get("reps").state["count"], core.rhythm["reps"].count)
        # The pathology this route exists to break: a rep is not an Incident.
        self.assertEqual(incidents, [])

    def test_the_eye_measures_every_frame_and_receives_the_real_frame(self):
        core = _core(_rhythm_goal())
        source = _source(n_cycles=4)
        asyncio.run(_collect(core, source))

        self.assertEqual(len(_INSTANCES), 1)  # exactly one eye, built once for the goal
        eye = _INSTANCES[0]
        self.assertEqual(len(eye.seen), len(source))  # once per FRAME, not once per admission
        self.assertEqual(eye.seen, [judge_frame for _, _, judge_frame in source])

    def test_the_count_is_not_capped_by_judgment_density(self):
        # The whole point: the gate never escalated (a still frame, and a rhythm goal compiles to no
        # admission condition at all), so ZERO judgments were available -- yet the count is real.
        core = _core(_rhythm_goal())
        asyncio.run(_collect(core, _source(n_cycles=8)))
        self.assertEqual(core.metrics.snapshot().get("judge.calls", 0), 0)
        self.assertGreaterEqual(core.rhythm["reps"].count, 7)

    def test_an_authored_milestone_mints_exactly_one_incident(self):
        core = _core(_rhythm_goal(fire_when="n == 5"))
        incidents = asyncio.run(_collect(core, _source(n_cycles=8)))

        self.assertEqual(len(incidents), 1)  # ONE incident at the milestone, not one per rep
        self.assertEqual(incidents[0].concern_id, "reps")
        self.assertGreaterEqual(core.rhythm["reps"].count, 7)  # ...and counting continued past it

    def test_a_rhythm_goal_coexists_with_a_judged_watch_on_the_same_source(self):
        from percept.adapters import Verdict
        from percept.backends.fake import FakeVision

        vision = FakeVision(
            script={
                "hot": Verdict(satisfied=True, confidence=0.95, evidence="a person is present"),
                "cold": Verdict(satisfied=False, confidence=0.95, evidence="nobody"),
            }
        )
        core = Percept.create(mode="offline", vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="who", condition="a person is present", say="someone is here"))
        core.add_goal(_rhythm_goal(fire_when="n == 3"))

        moved = gray_frame(W, H, 255)
        source = [
            (i / FPS, _still() if i % 2 == 0 else moved, "hot" if i % 2 else "cold")
            for i in range(8 * PERIOD)
        ]
        incidents = asyncio.run(_collect(core, source))

        self.assertGreaterEqual(core.rhythm["reps"].count, 7)  # the rhythm goal counted
        by_goal = {incident.concern_id for incident in incidents}
        self.assertIn("who", by_goal)  # the judged Watch still fired
        self.assertEqual(sum(1 for i in incidents if i.concern_id == "reps"), 1)  # one milestone


class RhythmThroughGatedWatchLoopTests(_RhythmFixture):
    """The same drive on the harness loop (the cost-demo / bring-your-own-capture seam)."""

    def test_gated_loop_counts_reps_and_reports_the_count(self):
        goal = _rhythm_goal()
        core = _core(goal)
        loop = GatedWatchLoop(core, [goal], threshold=0.05)
        report = asyncio.run(loop.run([_still() for _ in range(8 * PERIOD)], dt=1 / FPS))

        self.assertEqual(list(loop.spec.conditions), [])  # measured, never admitted
        self.assertEqual(report.vision_calls, 0)  # no judge spent
        self.assertAlmostEqual(report.rhythm_counts["reps"], 8, delta=1)
        self.assertEqual(report.fire_events, [])  # no milestone authored -> silence

    def test_gated_loop_emits_the_authored_milestone_once(self):
        goal = _rhythm_goal(fire_when="n == 4")
        core = _core(goal)
        loop = GatedWatchLoop(core, [goal], threshold=0.05)
        report = asyncio.run(loop.run([_still() for _ in range(8 * PERIOD)], dt=1 / FPS))

        self.assertEqual(len(report.fire_events), 1)
        self.assertEqual(report.fire_events[0].goal_id, "reps")


class ConductorToCountTests(_RhythmFixture):
    """The whole chain, offline: plain-language goal -> conductor -> WatchPlan -> arm -> run -> count.

    Part 1 (a drivable route) and Part 2 (something that authors it) are only worth anything joined
    up. Nothing here is hand-armed with a primitive: the ``type="rhythm"`` in the running goal came
    from the conductor deciding it from the goal TEXT.
    """

    def test_a_rep_counting_goal_text_ends_up_counting_reps(self):
        from percept.conductor import PlanConductor

        async def scripted_haiku(system: str, user: str) -> str:
            self.assertIn("RHYTHM", system)  # the real prompt, not a stub
            self.assertIn("push-ups", user)
            return (
                '{"behavior": {"type": "rhythm", "accumulate": {"fire_when": "n == 5"}}, '
                '"rhythm": {"channel": "limb-vertical"}}'
            )

        goal_text = "count my push-ups and tell me when I hit 5"
        primitive = asyncio.run(PlanConductor(call=scripted_haiku).author(goal_text))
        self.assertEqual(primitive.behavior_type, "rhythm")
        self.assertEqual(primitive.explicit["Goal.state.fire_when"], "n == 5")

        core = Percept.create(mode="offline", discover_plugins=False)
        # The eye is a deployment choice, not an authoring one: the conductor picks the PRIMITIVE,
        # the plan/profile picks which detector measures for it.
        plan = primitive.to_watch_plan(id="reps", detectors=("test-scripted-channel",))
        self.assertEqual(apply_watch_plan(core, plan).status, "applied")
        self.assertIs(goal_behavior_route(core.goals.get("reps")), BehaviorRoute.EXTERNAL_RHYTHM)

        incidents = asyncio.run(_collect(core, _source(n_cycles=8)))
        self.assertGreaterEqual(core.rhythm["reps"].count, 7)  # counted from the signal
        self.assertEqual(len(incidents), 1)  # exactly the authored milestone
        self.assertEqual(incidents[0].concern_id, "reps")


# ---------------------------------------------------------------------------
# The PAYLOAD contract — what the frame loop publishes vs what an eye can read
# ---------------------------------------------------------------------------
#
# Everything above drives the route through ``_ScriptedChannel``, which accepts ANY object as its
# frame. That is exactly what let a payload-contract break ship intact: for a file/camera source the
# frame loop publishes ``("image/jpeg", bytes)`` (see ``percept.runtime.source``), the shipped eye
# reads pixels, and no offline test ever handed the shipped eye a real frame — so the whole route
# ran green while producing ``'tuple' object has no attribute 'ndim'`` and zero reps on real video.
#
# Below, the EYE is the shipped ``PoseChannels``. Only the pose MODEL is a double (neither venv ships
# mmpose/mediapipe), and it is a HOSTILE double: it raises unless it is handed the RGB uint8 HxWx3
# buffer the eye documents. A lenient double here would re-create the original blind spot.

try:
    import cv2
    import numpy as np
    _HAVE_RUNTIME = True
except ImportError:  # pragma: no cover - the [runtime] extra is absent in the L1 stdlib lane
    _HAVE_RUNTIME = False

JPEG_W, JPEG_H = 32, 24
#: The block colour written in BGR order, so an RGB/BGR mix-up is visible in the decoded pixels.
BLOCK_BGR = (32, 64, 192)


def _bgr(i: int = 0):
    """One captured frame as OpenCV hands it to the source loop: a BGR uint8 buffer with structure."""
    bgr = np.zeros((JPEG_H, JPEG_W, 3), dtype=np.uint8)
    bgr[:, (i * 3) % JPEG_W:] = BLOCK_BGR
    return bgr


def _judge_frame(i: int = 0) -> tuple:
    """EXACTLY the judge payload ``percept.runtime.source`` publishes: ``cv2.imencode('.jpg', bgr)``."""
    encoded, buf = cv2.imencode(".jpg", _bgr(i))
    assert encoded and buf is not None
    return ("image/jpeg", buf.tobytes())


class _PixelAssertingPose:
    """A pose-model double that REFUSES anything but the eye's documented input.

    The refusal is the assertion: handed the transport payload (or a gate ``Frame``, or ``None``) it
    raises instead of quietly returning a pose, so a regression in the decode seam fails the test
    rather than passing through a double that tolerates it. Oscillates the wrists once per ``period``
    frames so the brain still has a real signal to count."""

    def __init__(self, period: int = PERIOD, amp: float = 0.3) -> None:
        self.i, self.period, self.amp = 0, period, amp
        self.seen: list[tuple] = []   # (shape, dtype, top-right pixel) per accepted frame
        self.frames: list = []        # the buffer objects themselves (identity, for pass-through)

    def warm(self) -> str:
        """The rest of the ``PoseProvider`` protocol: RhythmDrive's arm-time pre-flight
        (percept.rhythm._pose_preflight_failure) calls this on construction, before any frame is
        measured. A double standing in for a real provider must implement the WHOLE protocol it
        claims — a fake missing ``warm()`` would make every goal using it refused at arm instead of
        exercising what these tests are actually for."""
        return "test-pixel-asserting-pose"

    def detect(self, frame):
        if not hasattr(frame, "shape") or not hasattr(frame, "dtype"):
            raise AssertionError(
                f"the pose model was handed {type(frame).__name__}, not decoded pixels"
            )
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise AssertionError(
                f"the pose model documents RGB uint8 HxWx3; got {frame.shape} {frame.dtype}"
            )
        self.frames.append(frame)
        self.seen.append((tuple(frame.shape), str(frame.dtype), tuple(int(c) for c in frame[0, -1])))
        wy = 0.5 - self.amp * math.sin(2 * math.pi * self.i / self.period)
        self.i += 1
        k = [PoseKeypoint(0.5, 0.5) for _ in range(17)]
        k[5], k[6] = PoseKeypoint(0.4, 0.5), PoseKeypoint(0.6, 0.5)        # shoulders
        k[9], k[10] = PoseKeypoint(0.4, wy), PoseKeypoint(0.6, wy)          # wrists (the moving DOF)
        k[11], k[12] = PoseKeypoint(0.45, 0.8), PoseKeypoint(0.55, 0.8)     # hips (torso length 0.3)
        k[15], k[16] = PoseKeypoint(0.45, 0.95), PoseKeypoint(0.55, 0.95)   # ankles (still)
        return [PoseDetection(tuple(k), confidence=0.9)]


def _register_pose_eye() -> _PixelAssertingPose:
    """Register the SHIPPED ``PoseChannels`` eye behind a fake pose model; return the model."""
    model = _PixelAssertingPose()
    register(
        "test-pose-eye",
        lambda: PoseChannels(provider=model),
        DetectorManifest(
            "test-pose-eye", "pose channels off decoded pixels",
            "payload-contract test: the real eye, a fake model",
            "per-frame channel measurements",
        ),
    )
    return model


@unittest.skipUnless(_HAVE_RUNTIME, "needs numpy + OpenCV (encode a real JPEG and decode it back)")
class RhythmFramePayloadTests(_RhythmFixture):
    """The real eye, driven by the real payload — the test the original break would have failed."""

    def test_the_fixture_payload_is_the_shape_the_frame_loop_actually_publishes(self):
        # Guard the guard: if this fixture were a hand-rolled approximation of the judge frame, every
        # test below would prove nothing. Validate it with the runtime's OWN envelope check.
        from percept.runtime.envelope import SourceEnvelope, _video_payload_valid
        from percept.runtime.source import _video_format

        envelope = SourceEnvelope(
            source_id="video:test", modality="video", epoch=0, sequence=0,
            observed_time=0.0, source_time=0.0, format=_video_format(_bgr(0)),
            payload=_judge_frame(0), signal=gray_frame(W, H, 0), legacy_timebase="source",
        )
        self.assertTrue(_video_payload_valid(envelope))
        self.assertEqual(envelope.format.media_type, "image/jpeg")

    def test_the_real_eye_measures_the_real_encoded_payload_through_percept_run(self):
        model = _register_pose_eye()
        core = _core(_rhythm_goal(detectors=["test-pose-eye"]))
        source = [(i / FPS, _still(), _judge_frame(i)) for i in range(8 * PERIOD)]

        incidents = asyncio.run(_collect(core, source))

        # The eye saw a decoded frame per published frame — the break returned zero channels forever.
        self.assertEqual(len(model.seen), len(source))
        self.assertEqual({shape for shape, _, _ in model.seen}, {(JPEG_H, JPEG_W, 3)})
        self.assertEqual({dtype for _, dtype, _ in model.seen}, {"uint8"})
        # ...and it is RGB, not BGR: the block was encoded B=32 G=64 R=192, so red must dominate.
        r, _g, b = model.seen[0][2]
        self.assertGreater(r, b + 100)
        # The point of all of it: reps counted off a real video payload, with no judge and no incident.
        self.assertAlmostEqual(core.rhythm["reps"].count, 8, delta=1)
        self.assertEqual(core.metrics.snapshot().get("judge.calls", 0), 0)
        self.assertEqual(incidents, [])

    def test_the_drive_dispatches_the_published_payload_and_the_eye_crosses_to_pixels(self):
        # RhythmDrive.observe is the seam that regressed. Driven directly (no percept.run) so a
        # regression here fails at the seam instead of somewhere downstream in the worker.
        model = _register_pose_eye()
        core = _core(_rhythm_goal(detectors=["test-pose-eye"]))
        drive = RhythmDrive(core, [core.goals.get("reps")])

        for i in range(4 * PERIOD):
            drive.observe(i / FPS, _judge_frame(i))

        self.assertEqual(len(model.seen), 4 * PERIOD)
        self.assertGreaterEqual(drive.count("reps"), 3)

    def test_an_undecodable_payload_raises_instead_of_silently_measuring_nothing(self):
        model = _register_pose_eye()
        core = _core(_rhythm_goal(detectors=["test-pose-eye"]))
        drive = RhythmDrive(core, [core.goals.get("reps")])

        for payload, why in (
            ("f0", "a payload that is not a frame at all"),
            (("image/jpeg", b"\xff\xd8not-a-jpeg\xff\xd9"), "image bytes that decode to nothing"),
            (("application/x-percept-opaque", b""), "an opaque payload carrying no bytes"),
            (object(), "an arbitrary object"),
        ):
            with self.subTest(why), self.assertRaises(FrameNotDecodable):
                drive.observe(0.0, payload)

        self.assertEqual(model.seen, [])          # the model was never reached...
        self.assertEqual(drive.count("reps"), 0)  # ...and nothing was silently counted

    def test_a_decoded_buffer_passes_through_untouched_and_the_other_payload_shapes_decode(self):
        model = _register_pose_eye()
        eye = make_detector("test-pose-eye")

        rgb = np.zeros((JPEG_H, JPEG_W, 3), dtype=np.uint8)
        eye.measure(rgb)
        # The in-process benchmark path (eval/bench/repcount) hands the eye its own array: identity,
        # not a copy and not a re-encode round trip.
        self.assertIs(model.frames[-1], rgb)

        eye.measure(_judge_frame(0)[1])  # bare JPEG bytes, no media tuple
        self.assertEqual(model.seen[-1][0], (JPEG_H, JPEG_W, 3))

        eye.measure(gray_frame(4, 4, 128))  # the harness loop's w/h/pixels grid frame
        self.assertEqual(model.seen[-1][0], (4, 4, 3))

    def test_the_boundary_bank_reading_the_same_loop_variable_decodes_too(self):
        # ``DetectorBank.push(frame=judge_frame)`` is the OTHER consumer of the worker's judge_frame
        # and carried the identical mismatch — a frame-backed BOUNDARY detector (pose-openness) was
        # handed the transport pair too. One decode seam, shared by both, is why this passes.
        from percept.harness.detectors import DetectorBank

        model = _register_pose_eye()
        bank = DetectorBank(["test-pose-eye"])

        out = bank.push(0.0, motion=0.0, frame=_judge_frame(0))

        self.assertTrue(out and all(b.kind == "sample" for b in out))
        self.assertEqual(model.seen[-1][0], (JPEG_H, JPEG_W, 3))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
