"""The eight ``BehaviorRoute`` values, driven end to end on the REAL published frame payload.

Every unit in this repo can pass while the seam BETWEEN two units is broken, because a unit test is
free to invent a convenient stand-in for the thing the other side actually sends. This file removes
that freedom for the route matrix:

* the source yields exactly what ``percept.runtime.source._capture`` publishes — a
  ``percept.edge.tier0.Frame`` gate signal plus an ``("image/jpeg", bytes)`` judge payload built by
  ``cv2.imencode`` — never a token string;
* :class:`StrictPayloadVision` and :class:`PixelPoseProvider` **raise** on anything that is not that
  payload (respectively: a decodable JPEG pair, and a decoded ``uint8 HxWx3`` buffer), and derive
  their scripted answers from real decoded pixels. A double that accepts any object would make every
  assertion below vacuous, which is the exact failure mode this file exists to catch;
* zero provider calls and no model weights: the judge is scripted and the pose provider is a
  pixel-driven stub, so the pose CHECKPOINT is never downloaded while the ``PoseChannels`` ->
  ``frame_pixels`` -> provider contract is still exercised for real.

The matrix each test pins (arms / receives / fires):

===========================  =====  ========  =====  =========================================
route                        arms   receives  fires  driver
===========================  =====  ========  =====  =========================================
FRAME_WATCH                  yes    yes       yes    ``percept.run`` -> ``perceive_judged``
FRAME_TRANSITION             yes    yes       yes    ``percept.run`` -> ``_perceive_transition``
FRAME_PROCESS                yes    yes       yes    ``percept.run`` -> ``check_episode_boundary``
EXTERNAL_EVENT_WINDOW        warns  yes       yes    ``push_frame`` + ``perceive_events`` (host)
EXTERNAL_BOUNDARY            warns  yes       yes    ``perceive_boundary`` (host)
EXTERNAL_RHYTHM              yes    yes       yes    ``percept.run`` -> ``RhythmDrive.observe``
EXTERNAL_SCHEDULER           warns  yes       yes*   ``schedule_every`` + ``tick`` (host)
UNSUPPORTED                  no     n/a       no     nothing — ``add_goal`` refuses
===========================  =====  ========  =====  =========================================

``yes*``: the scheduler emits a fire only when the conductor answers ``proactive_speak``; the
offline/scripted conductor answers ``ignore``, so the tick is consumed and nothing is emitted.
"""

from __future__ import annotations

import asyncio
import math
import unittest
import warnings
from unittest.mock import patch

try:  # the [runtime] extra
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without the runtime extra
    cv2 = None
    np = None

from percept import Goal, Percept
from percept.adapters import Decision, Verdict
from percept.behavior import program_of
from percept.behavior_routes import (
    BehaviorRoute,
    UndrivenBehaviorRouteWarning,
    UnsupportedBehaviorRouteError,
    goal_behavior_route,
)
from percept.edge.tier0 import Frame
from percept.events import Event
from percept.incidents import Incident, make_incident
from percept.runtime import run
from percept.runtime.source import GATE_H, GATE_W

W, H = 64, 48  # deliberately tiny: this file must stay cheap enough to run on a busy machine


# ---------------------------------------------------------------------------------------------
# The real payload — byte-for-byte the shape percept.runtime.source publishes
# ---------------------------------------------------------------------------------------------
class PayloadContractError(AssertionError):
    """A seam was handed something that is not the published frame payload."""


def _bgr(fill: int, *, blob: bool = False):
    img = np.full((H, W, 3), fill, dtype=np.uint8)
    if blob:
        img[12:36, 16:48] = 250
    return img


def _publish(img):
    """One decoded BGR image -> ``(gate_frame, judge_frame)`` exactly as ``_capture`` publishes it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (GATE_W, GATE_H))
    gate = Frame(w=GATE_W, h=GATE_H, pixels=tuple(int(p) for p in small.flatten()))
    encoded, buf = cv2.imencode(".jpg", img)
    if not encoded:
        raise RuntimeError("cv2.imencode failed")
    return gate, ("image/jpeg", buf.tobytes())


def _decode_or_raise(frame, where):
    """Refuse anything that is not a real, decodable ``("image/jpeg", bytes)`` payload."""
    if not (isinstance(frame, tuple) and len(frame) == 2):
        raise PayloadContractError(
            f"{where} expected ('image/jpeg', bytes), got {type(frame).__name__}"
        )
    media_type, payload = frame
    if media_type != "image/jpeg" or not isinstance(payload, (bytes, bytearray)):
        raise PayloadContractError(
            f"{where} expected ('image/jpeg', bytes), got ({media_type!r}, {type(payload).__name__})"
        )
    arr = cv2.imdecode(np.frombuffer(bytes(payload), dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None or arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise PayloadContractError(f"{where} payload did not decode to uint8 HxWx3")
    return arr


def _stream():
    """Dark / bright / dark / bright — real motion for the gate, real content for the judge."""
    dark_gate, dark_jpeg = _publish(_bgr(10))
    lit_gate, lit_jpeg = _publish(_bgr(10, blob=True))
    return [
        (0.0, dark_gate, dark_jpeg),
        (1.0, lit_gate, lit_jpeg),
        (2.0, dark_gate, dark_jpeg),
        (3.0, lit_gate, lit_jpeg),
    ]


def _oscillation(cycles=6, per_cycle=12, dt=0.08):
    """A real oscillating scene: a bright bar whose height is a sinusoid (one rep per cycle)."""
    frames, t = [], 0.0
    for _ in range(cycles):
        for k in range(per_cycle):
            y = int(20 + 14 * math.sin(2 * math.pi * k / per_cycle))
            img = np.full((H, W, 3), 10, dtype=np.uint8)
            img[max(0, y - 4): y + 4, 12:52] = 240
            gate, jpeg = _publish(img)
            frames.append((t, gate, jpeg))
            t += dt
    return frames


# ---------------------------------------------------------------------------------------------
# Scripted seams that refuse a stand-in
# ---------------------------------------------------------------------------------------------
class StrictPayloadVision:
    """Scripted verdicts derived from REAL decoded pixels; raises on any non-payload object.

    Wiring, not intelligence (standing rule 2): the answer is ``mean luminance > 40``, negated for a
    condition phrased about darkness. It understands nothing — but it cannot be satisfied by an
    object the frame loop would never emit, which is the property under test.
    """

    def __init__(self, *, events_for=None, step_evidence=None):
        self.judged = []          # (condition, mean, satisfied) per judge call
        self.detected = []        # every frame handed to detect()
        self.event_windows = []   # every window handed to detect_events()
        self.summarized = []      # every log handed to summarize_process()
        self._events_for = events_for
        self._step_evidence = step_evidence

    async def judge(self, condition, frame, **kwargs):
        strip = kwargs.get("frames")
        if frame is None and strip:
            newest = strip[-1]
            frame = newest[1] if isinstance(newest, tuple) and len(newest) == 2 else newest
        arr = _decode_or_raise(frame, "vision.judge")
        mean = float(arr.mean())
        lit = mean > 40.0
        satisfied = (not lit) if "dark" in condition else lit
        self.judged.append((condition, mean, satisfied))
        if (kwargs.get("context") or {}).get("observe_step") and self._step_evidence is not None:
            return Verdict(satisfied=False, confidence=0.9, evidence=self._step_evidence(mean))
        return Verdict(satisfied=satisfied, confidence=0.95, evidence=f"mean={mean:.1f}")

    async def detect(self, frame, goals):
        _decode_or_raise(frame, "vision.detect")
        self.detected.append(frame)
        return []

    async def detect_events(self, frames, goals):
        self.event_windows.append(list(frames))
        if not frames:
            return []
        t, frame = frames[-1]
        arr = _decode_or_raise(frame, "vision.detect_events")
        return self._events_for(float(t), arr) if self._events_for else []

    async def summarize_process(self, condition, log):
        self.summarized.append(list(log))
        ordered = []
        for step in log:
            if step and step not in ordered:
                ordered.append(step)
        return "; ".join(ordered)


class PixelPoseProvider:
    """A ``PoseProvider`` whose keypoints are read off the pixels it is handed.

    It stands in for the RTMO checkpoint so this file downloads no weights, WITHOUT weakening the
    seam it guards: it raises unless it receives a decoded ``uint8 HxWx3`` buffer (what a real
    provider needs, and what ``percept.harness.pose.frame_pixels`` is responsible for producing from
    the published ``("image/jpeg", bytes)`` payload), and the wrist keypoints track the bright bar's
    real y-position, so the channel signal is genuinely scene-derived.
    """

    name = "pixel-stub"

    def __init__(self):
        self.received = []

    def warm(self):
        return self.name

    def detect(self, frame):
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise PayloadContractError(
                "pose provider expected a decoded uint8 HxWx3 buffer, got "
                f"{getattr(arr, 'dtype', None)} {getattr(arr, 'shape', None)}"
            )
        self.received.append(arr)
        from percept.harness.pose import PoseDetection, PoseKeypoint

        rows = np.where(arr.mean(axis=2).max(axis=1) > 128)[0]
        if rows.size == 0:
            return []
        bar_y = float(rows.mean()) / arr.shape[0]
        kp = [PoseKeypoint(0.5, 0.1, 1.0)] * 17
        kp[5], kp[6] = PoseKeypoint(0.4, 0.35, 1.0), PoseKeypoint(0.6, 0.35, 1.0)  # shoulders
        kp[9], kp[10] = PoseKeypoint(0.35, bar_y, 1.0), PoseKeypoint(0.65, bar_y, 1.0)  # wrists
        kp[11], kp[12] = PoseKeypoint(0.45, 0.65, 1.0), PoseKeypoint(0.55, 0.65, 1.0)  # hips
        kp[15], kp[16] = PoseKeypoint(0.45, 0.95, 1.0), PoseKeypoint(0.55, 0.95, 1.0)  # ankles
        return [PoseDetection(keypoints=tuple(kp), confidence=0.9, provider=self.name)]


class SpeakingConductor:
    """Answers every scheduler tick with ``proactive_speak`` (the offline default answers ignore)."""

    def __init__(self):
        self.calls = []

    async def run(self, event, utterance, context):
        self.calls.append((event, utterance, dict(context)))
        return Decision(route="proactive_speak", say=utterance or "timer")


def _core(**kwargs) -> Percept:
    return Percept.create(discover_plugins=False, **kwargs)


def _arm(core: Percept, goal: Goal) -> list[warnings.WarningMessage]:
    """Arm ``goal`` and hand back the arm-time warnings (the host-driven routes warn by design)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        core.add_goal(goal)
    return [w for w in caught if issubclass(w.category, UndrivenBehaviorRouteWarning)]


async def _drive(core: Percept, source, **kwargs):
    return [incident async for incident in run(core, source=source, **kwargs)]


@unittest.skipIf(cv2 is None, "needs the [runtime] extra (cv2 + numpy) for real frame payloads")
class RouteMatrixTests(unittest.TestCase):
    """One test per route: arms, receives its driver's input, and delivers (or does not) an Incident."""

    # -- FRAME_WATCH ---------------------------------------------------------------------------
    def test_frame_watch_delivers_an_incident_on_the_real_payload(self):
        vision = StrictPayloadVision()
        core = _core(vision=vision)
        goal = Goal(id="w", condition="something bright is present", say="Alert: bright")
        self.assertEqual(_arm(core, goal), [])  # a frame route arms silently
        self.assertIs(goal_behavior_route(core.goals.get("w")), BehaviorRoute.FRAME_WATCH)

        incidents = asyncio.run(_drive(core, _stream()))

        self.assertTrue(vision.judged, "the judge never received a frame")
        self.assertEqual(len(incidents), 2)  # two rising edges in dark/lit/dark/lit
        self.assertTrue(all(isinstance(i, Incident) for i in incidents))
        self.assertEqual({i.message for i in incidents}, {"Alert: bright"})

    # -- FRAME_TRANSITION ----------------------------------------------------------------------
    def test_frame_transition_delivers_an_incident_on_the_real_payload(self):
        vision = StrictPayloadVision()
        core = _core(vision=vision)
        goal = Goal(
            id="tr", condition="the scene lights up", say="good morning",
            state={"type": "transition", "from": "the scene is dark",
                   "to": "something bright is present", "count": 0},
        )
        self.assertEqual(_arm(core, goal), [])
        self.assertIs(goal_behavior_route(core.goals.get("tr")), BehaviorRoute.FRAME_TRANSITION)

        incidents = asyncio.run(_drive(core, _stream()))

        judged_conditions = {condition for condition, _, _ in vision.judged}
        self.assertEqual(  # BOTH halves of the transition reach the judge, per frame
            judged_conditions, {"the scene is dark", "something bright is present"}
        )
        self.assertEqual(len(incidents), 1)  # A witnessed, then B -> one fire, then refractory
        self.assertEqual(incidents[0].message, "good morning")
        self.assertEqual(core.goals.get("tr").state["count"], 1)

    # -- FRAME_PROCESS -------------------------------------------------------------------------
    def test_frame_process_delivers_the_consolidated_answer_at_the_episode_boundary(self):
        vision = StrictPayloadVision(
            step_evidence=lambda mean: ("step: bright" if mean > 40 else "")
        )
        core = _core(vision=vision)
        goal = Goal(id="pr", condition="what steps happened", say="answer ready",
                    state={"type": "process"})
        self.assertEqual(_arm(core, goal), [])
        self.assertIs(goal_behavior_route(core.goals.get("pr")), BehaviorRoute.FRAME_PROCESS)

        dark_gate, dark_jpeg = _publish(_bgr(10))
        source = _stream() + [(10.0, dark_gate, dark_jpeg), (11.0, dark_gate, dark_jpeg)]
        incidents = asyncio.run(_drive(core, source, episode_inactivity_s=2.0))

        self.assertTrue(vision.judged, "the process observe-step never reached the judge")
        self.assertEqual(vision.summarized, [["step: bright", "step: bright"]])
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0].message, "step: bright")

    def test_frame_process_needs_quiet_longer_than_the_inactivity_window(self):
        # The boundary is an ABSENCE of new steps; the default window is
        # ``settings.gate.validity_window_s`` (90 s), so a short stream emits nothing at all.
        vision = StrictPayloadVision(
            step_evidence=lambda mean: ("step: bright" if mean > 40 else "")
        )
        core = _core(vision=vision)
        _arm(core, Goal(id="pr", condition="what steps happened", say="answer ready",
                        state={"type": "process"}))
        self.assertEqual(asyncio.run(_drive(core, _stream())), [])

    # -- EXTERNAL_EVENT_WINDOW -----------------------------------------------------------------
    def test_external_event_window_fires_from_push_frame_plus_perceive_events(self):
        def events_for(t, arr):
            return [Event(type="door_opened", t=t, confidence=0.9)] if arr.mean() > 40 else []

        vision = StrictPayloadVision(events_for=events_for)
        core = _core(vision=vision)
        goal = Goal(id="ev", condition="the door opens", say="the door opened",
                    state={"type": "event", "event_type": "door_opened"})
        warned = _arm(core, goal)

        self.assertEqual(len(warned), 1)  # host-driven: arms, but warns and names the call
        self.assertIn("Percept.push_frame", str(warned[0].message))
        self.assertIs(goal_behavior_route(core.goals.get("ev")), BehaviorRoute.EXTERNAL_EVENT_WINDOW)

        dark_gate, dark_jpeg = _publish(_bgr(10))
        lit_gate, lit_jpeg = _publish(_bgr(10, blob=True))

        async def go():
            core.push_frame(dark_jpeg, source="camera", t=0.0)
            quiet = await core.perceive_events(source="camera")
            core.push_frame(lit_jpeg, source="camera", t=1.0)
            return quiet, await core.perceive_events(source="camera")

        quiet, fired = asyncio.run(go())

        self.assertTrue(vision.event_windows, "detect_events never received the rolling window")
        self.assertEqual(vision.event_windows[-1][-1][1], lit_jpeg)  # the REAL payload, unwrapped
        self.assertEqual(quiet, [])
        self.assertEqual([(f.goal_id, f.action) for f in fired], [("ev", "fire")])
        self.assertEqual(core.goals.get("ev").state["count"], 1)
        # The host path hands back a FireEvent; the Incident is minted by the caller.
        self.assertIsInstance(make_incident(fired[0], ts=1.0), Incident)

    # -- EXTERNAL_BOUNDARY ---------------------------------------------------------------------
    def test_external_boundary_fires_from_perceive_boundary(self):
        core = _core(vision=StrictPayloadVision())
        goal = Goal(id="cnt", condition="a repetition happened", say="rep counted",
                    state={"type": "count"})
        warned = _arm(core, goal)

        self.assertEqual(len(warned), 1)
        self.assertIn("Percept.perceive_boundary", str(warned[0].message))
        self.assertIs(goal_behavior_route(core.goals.get("cnt")), BehaviorRoute.EXTERNAL_BOUNDARY)

        fired = [core.perceive_boundary("cnt", t=float(n)) for n in range(1, 4)]

        self.assertTrue(all(f is not None for f in fired))
        self.assertEqual(core.edge_counts["cnt"], 3)
        self.assertIsInstance(make_incident(fired[0], ts=1.0), Incident)

    def test_count_route_carries_its_authored_fire_when_into_the_program(self):
        """REGRESSION: ``program_of`` used to DROP ``state['fire_when']`` for ``type='count'``.

        The count branch built a bare ``Accumulator(kind="count_distinct")`` while the rhythm branch
        four lines below passed ``fire_when=s.get("fire_when")``, so an authored milestone was
        silently discarded — "tell me when I hit 10" fired twelve times starting at rep 1, which
        contradicted ``perceive_boundary``'s own docstring (``percept/agent.py``: "Fires per the
        goal's ``accumulate.fire_when`` (e.g. 'n == 10')"), the ``external_boundary`` route remedy,
        and the authoring firewall that deliberately writes ``fire_when`` into a count goal's state.
        """
        core = _core(vision=StrictPayloadVision())
        _arm(core, Goal(id="cnt", condition="a repetition happened", say="ten reps",
                        state={"type": "count", "fire_when": "n == 10"}))
        goal = core.goals.get("cnt")

        self.assertEqual(goal.state["fire_when"], "n == 10")  # authored, and it survives arming
        self.assertEqual(program_of(goal).accumulate.fire_when, "n == 10")  # ...and reaches the program
        self.assertEqual(program_of(goal).accumulate.kind, "count_distinct")

    def test_count_route_honours_its_authored_fire_when(self):
        """The milestone a client actually asked for: ONE fire, at the tenth boundary."""
        core = _core(vision=StrictPayloadVision())
        _arm(core, Goal(id="cnt", condition="a repetition happened", say="ten reps",
                        state={"type": "count", "fire_when": "n == 10"}))
        fired_at = [n for n in range(1, 13) if core.perceive_boundary("cnt", t=float(n)) is not None]
        self.assertEqual(fired_at, [10])
        self.assertEqual(core.edge_counts["cnt"], 12)  # the tally never stops; only the ALERT is gated

    def test_count_route_without_a_milestone_still_fires_every_boundary(self):
        """The documented default is unchanged: no ``fire_when`` -> every distinct occurrence fires."""
        core = _core(vision=StrictPayloadVision())
        _arm(core, Goal(id="cnt", condition="a repetition happened", say="rep",
                        state={"type": "count"}))
        self.assertIsNone(program_of(core.goals.get("cnt")).accumulate.fire_when)
        fired_at = [n for n in range(1, 6) if core.perceive_boundary("cnt", t=float(n)) is not None]
        self.assertEqual(fired_at, [1, 2, 3, 4, 5])

    # -- EXTERNAL_RHYTHM -----------------------------------------------------------------------
    def test_external_rhythm_counts_and_fires_its_milestone_from_the_frame_loop(self):
        vision = StrictPayloadVision()
        core = _core(vision=vision)
        goal = Goal(id="rh", condition="repetitions of the bar", say="three reps done",
                    state={"type": "rhythm", "fire_when": "n == 3"})
        self.assertEqual(_arm(core, goal), [])  # the frame loop drives it -> no arm warning
        self.assertIs(goal_behavior_route(core.goals.get("rh")), BehaviorRoute.EXTERNAL_RHYTHM)

        provider = PixelPoseProvider()
        source = _oscillation()
        with patch("percept.harness.pose.make_pose_provider", return_value=provider):
            incidents = asyncio.run(_drive(core, source))

        # The eye MEASURES every frame (ungated, unjudged) and is handed decoded pixels, not the
        # transport payload: ``PoseChannels.measure`` must call ``frame_pixels`` first.
        self.assertEqual(len(provider.received), len(source))
        self.assertEqual(provider.received[0].shape, (H, W, 3))
        self.assertEqual(vision.judged, [])  # a rhythm goal spends no judge
        self.assertGreaterEqual(core.rhythm["rh"].count, 3)
        self.assertEqual(core.goals.get("rh").state["count"], core.rhythm["rh"].count)
        self.assertEqual([i.message for i in incidents], ["three reps done"])

    def test_external_rhythm_without_a_milestone_counts_but_stays_silent(self):
        core = _core(vision=StrictPayloadVision())
        _arm(core, Goal(id="rh", condition="repetitions", say="x", state={"type": "rhythm"}))

        provider = PixelPoseProvider()
        with patch("percept.harness.pose.make_pose_provider", return_value=provider):
            incidents = asyncio.run(_drive(core, _oscillation()))

        self.assertGreaterEqual(core.goals.get("rh").state["count"], 3)
        self.assertEqual(incidents, [])  # documented default: the tally is state, not an event

    # -- EXTERNAL_SCHEDULER --------------------------------------------------------------------
    def test_external_scheduler_fires_from_schedule_every_plus_tick(self):
        conductor = SpeakingConductor()
        core = _core(vision=StrictPayloadVision(), conductor=conductor)
        goal = Goal(id="rem", condition="remind me to stretch", say="stretch time",
                    state={"type": "remind"})
        warned = _arm(core, goal)

        self.assertEqual(len(warned), 1)
        self.assertIn("Percept.schedule_every", str(warned[0].message))
        self.assertIs(goal_behavior_route(core.goals.get("rem")), BehaviorRoute.EXTERNAL_SCHEDULER)

        async def go():
            core.schedule_every(5.0, "rem", now=0.0)
            early = await core.tick(t=1.0)
            return early, await core.tick(t=6.0)

        early, fired = asyncio.run(go())

        self.assertEqual(early, [])
        self.assertEqual([(f.goal_id, f.action, f.text) for f in fired],
                         [("rem", "fire", "stretch time")])
        self.assertEqual(conductor.calls[0][0], "scheduler_tick")
        self.assertIsInstance(make_incident(fired[0], ts=6.0), Incident)

    def test_external_scheduler_is_silent_under_the_offline_scripted_conductor(self):
        # ``tick`` routes the due timer through the conductor; the scripted offline conductor
        # answers ``respond_direct``/``ignore``, never ``proactive_speak`` — so the timer is
        # consumed and nothing is emitted. A deployment must supply a conductor that speaks.
        core = _core(vision=StrictPayloadVision())
        _arm(core, Goal(id="rem", condition="remind me", say="stretch time",
                        state={"type": "remind"}))

        async def go():
            core.schedule_every(5.0, "rem", now=0.0)
            return await core.tick(t=6.0)

        self.assertEqual(asyncio.run(go()), [])

    # -- UNSUPPORTED ---------------------------------------------------------------------------
    def test_unsupported_routes_are_refused_at_arm_time(self):
        for label, state in (
            ("unknown behavior type", {"type": "wobble"}),
            ("transition missing 'to'", {"type": "transition", "from": "asleep"}),
            ("pre-apply nested behavior block", {"behavior": {"type": "watch"}}),
        ):
            with self.subTest(label):
                core = _core(vision=StrictPayloadVision())
                with self.assertRaises(UnsupportedBehaviorRouteError) as caught:
                    core.add_goal(Goal(id="u", condition="c", say="s", state=dict(state)))
                self.assertIn("unsupported", str(caught.exception))
                self.assertEqual(list(core.goals.active()), [])  # nothing armed


@unittest.skipIf(cv2 is None, "needs the [runtime] extra (cv2 + numpy) for real frame payloads")
class RouteCoexistenceTests(unittest.TestCase):
    """How the host-driven routes interact with a frame run — the part a per-route test cannot see.

    A route decision must be about ONE goal. These pin that a goal the frame loop cannot drive is
    skipped and reported, never allowed to decide the fate of the goals beside it.
    """

    def test_a_host_driven_goal_on_the_run_source_is_skipped_not_fatal(self):
        """REGRESSION: one ``count`` goal used to abort the run and kill the watch beside it.

        ``require_worker_routes(_source_goals(), …)`` checked EVERY active goal on the source key, so
        following our own advice — arm a count goal and drive it with ``Percept.perceive_boundary``
        — next to a watch goal on the same source produced a dead run with no obvious cause. The
        preflight's question is "can this goal ever fire?", not "is this run worthwhile?": a
        host-driven route CAN fire, just not from here, so it is reported and skipped.
        """
        vision = StrictPayloadVision()
        core = _core(vision=vision)
        _arm(core, Goal(id="w", condition="something bright is present", say="bright"))
        _arm(core, Goal(id="cnt", condition="a repetition", say="rep", state={"type": "count"}))
        health: list = []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            incidents = asyncio.run(_drive(core, _stream(), on_health=health.append))

        # 1. The unrelated watch goal RAN and FIRED — the whole point.
        self.assertEqual([i.message for i in incidents], ["bright", "bright"])
        self.assertEqual({condition for condition, _, _ in vision.judged},
                         {"something bright is present"})
        # 2. The count goal was skipped, not judged: no frame ever bought it a provider call.
        self.assertNotIn("a repetition", {condition for condition, _, _ in vision.judged})
        self.assertEqual(core.edge_counts.get("cnt", 0), 0)
        # 3. ...and not silently: the warning names the goal, its route, and the exact host call.
        skipped = [w for w in caught if issubclass(w.category, UndrivenBehaviorRouteWarning)]
        self.assertEqual(len(skipped), 1)
        message = str(skipped[0].message)
        self.assertIn("'cnt'", message)
        self.assertIn("external_boundary", message)
        self.assertIn("Percept.perceive_boundary('cnt', t=...)", message)
        # 4. The same fact reaches a dashboard as structured health, once.
        undriven = [e for e in health if e.kind == "undriven_behavior_route"]
        self.assertEqual([e.data["goal_id"] for e in undriven], ["cnt"])
        self.assertEqual(undriven[0].data["route"], "external_boundary")
        self.assertIn("Percept.perceive_boundary", undriven[0].data["remedy"])
        # 5. The goal the run skipped is still armed, and its documented driver still fires it.
        self.assertIsNotNone(core.perceive_boundary("cnt", t=9.0))

    def test_the_skipped_route_also_reaches_the_deployment_log(self):
        # A 24/7 deployment has nobody watching stderr — the same two channels as the arm seam.
        core = _core(vision=StrictPayloadVision())
        _arm(core, Goal(id="w", condition="something bright is present", say="bright"))
        _arm(core, Goal(id="cnt", condition="a repetition", say="rep", state={"type": "count"}))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertLogs("percept.runtime.worker", level="WARNING") as logs:
                asyncio.run(_drive(core, _stream()))
        self.assertTrue(
            any("undriven_behavior_route" in line and "'cnt'" in line for line in logs.output),
            logs.output,
        )

    def test_a_strict_deployment_can_still_make_an_undriven_route_fatal(self):
        # The escalation knob the arm seam documents keeps working at run time, so nobody who WANTED
        # the old abort loses it — they opt in instead of everyone paying for it.
        core = _core(vision=StrictPayloadVision())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            core.add_goal(Goal(id="w", condition="something bright is present", say="bright"))
            core.add_goal(Goal(id="cnt", condition="a repetition", say="rep",
                               state={"type": "count"}))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UndrivenBehaviorRouteWarning)
            with self.assertRaises(UndrivenBehaviorRouteWarning):
                asyncio.run(_drive(core, _stream()))

    def test_a_route_nothing_can_drive_still_aborts_before_the_source_is_pulled(self):
        # The other half of the split: UNSUPPORTED has no driver ANYWHERE, so a run carrying one is
        # incoherent and still refuses up front. Only host-driven routes became non-fatal.
        core = _core(vision=StrictPayloadVision())
        _arm(core, Goal(id="w", condition="something bright is present", say="bright"))
        core.goals.get("w").state["type"] = "wobble"  # corrupt it PAST the arm-time gate

        with self.assertRaises(UnsupportedBehaviorRouteError) as caught:
            asyncio.run(_drive(core, _stream()))
        self.assertIn("route=unsupported", str(caught.exception))
        self.assertIn("closed behavior types", str(caught.exception))

    def test_a_mid_run_host_driven_arm_does_not_kill_the_running_watch(self):
        # The regate preflight is the same decision as the startup one: a goal armed by voice while
        # the loop is running must not take the loop down with it.
        vision = StrictPayloadVision()
        core = _core(vision=vision)
        _arm(core, Goal(id="w", condition="something bright is present", say="bright"))

        def source():
            frames = _stream()
            yield frames[0]
            yield frames[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                core.add_goal(Goal(id="cnt", condition="a repetition", say="rep",
                                   state={"type": "count"}))
            yield frames[2]
            yield frames[3]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            incidents = asyncio.run(_drive(core, source(), regate_on_goal_change=True))

        # How many fires survive a regate is the REGATE's contract (arming anything mid-run costs
        # the same in-flight fire); what belongs here is that the watch goal is still alive and
        # still delivering after a goal the worker cannot drive was armed beside it.
        self.assertTrue(incidents, "the unrelated watch goal stopped firing after the mid-run arm")
        self.assertEqual({i.message for i in incidents}, {"bright"})
        self.assertNotIn("a repetition", {c for c, _, _ in vision.judged})  # no judge spent on it
        skipped = [w for w in caught if issubclass(w.category, UndrivenBehaviorRouteWarning)]
        self.assertEqual(len(skipped), 1)  # reported once per goal per run, not once per frame
        self.assertIn("Percept.perceive_boundary", str(skipped[0].message))

    def test_binding_the_host_driven_goal_to_another_source_key_lets_both_run(self):
        core = _core(vision=StrictPayloadVision())
        _arm(core, Goal(id="w", condition="something bright is present", say="bright"))
        _arm(core, Goal(id="cnt", condition="a repetition", say="rep",
                        state={"type": "count"}, source="external"))

        incidents = asyncio.run(_drive(core, _stream()))

        self.assertEqual(len(incidents), 2)                       # the frame route still fires
        self.assertIsNotNone(core.perceive_boundary("cnt", t=9.0))  # and the host route still fires

    def test_percept_run_never_drives_the_event_window_or_the_scheduler(self):
        # Both routes are host-fed by construction: the worker calls neither API. Bound to a
        # non-run source key so the preflight admits them, they still emit nothing from frames.
        conductor = SpeakingConductor()
        core = _core(vision=StrictPayloadVision(), conductor=conductor)
        _arm(core, Goal(id="ev", condition="the door opens", say="door",
                        state={"type": "event", "event_type": "door_opened"}, source="external"))
        _arm(core, Goal(id="rem", condition="remind me", say="stretch",
                        state={"type": "remind"}, source="external"))
        core.schedule_every(0.5, "rem", now=0.0)

        self.assertEqual(asyncio.run(_drive(core, _stream())), [])
        self.assertEqual(conductor.calls, [])  # no tick ever happened


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
