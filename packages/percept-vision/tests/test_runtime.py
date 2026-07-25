"""Runtime worker (L1) — run(core, source) -> incidents, offline via an iterable source + FakeVision.

The source yields (t, gate_frame, judge_frame): a gray Frame drives the motion gate; a token string is
what FakeVision judges. No cv2, no camera — the iterable-source path makes the whole worker testable.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.incidents import make_incident
from percept.runtime import RuntimeHealthEvent, RuntimeStats, incident_payload, run
from percept.runtime.source import _normalize_uri
from percept.runtime.worker import _capture_plan
from percept.edge.tier0 import gray_frame

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)

# (t, gate_frame, judge_frame): motion alternates to clear the gate; tokens drive FakeVision.
SOURCE = [
    (0.0, STILL, "clear"),     # prime — no motion
    (1.0, MOVED, "intruder"),  # still->moved: escalate -> "intruder" -> FIRE (rising edge) -> incident
    (2.0, STILL, "clear"),     # moved->still: escalate -> "clear" -> silent (re-arms the edge)
    (3.0, MOVED, "intruder"),  # still->moved: escalate -> "intruder" -> FIRE again -> incident
]


def _core() -> Percept:
    vision = FakeVision(
        script={
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="the zone is empty"),
        }
    )
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone"))
    return core


def _ask_core() -> Percept:
    vision = FakeVision(script={"maybe": Verdict(satisfied=True, confidence=0.2, evidence="unclear")})
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert"))
    return core


async def _collect(core, source, **kw):
    return [inc async for inc in run(core, source=source, **kw)]


class RuntimeWorkerTests(unittest.TestCase):
    def test_emits_an_incident_per_fire(self):
        incidents = asyncio.run(_collect(_core(), SOURCE))
        self.assertEqual(len(incidents), 2)  # two rising edges
        self.assertTrue(all(i.concern_id == "zone" for i in incidents))
        self.assertTrue(all(i.message == "Alert: person in zone" for i in incidents))
        self.assertTrue(all(i.verdict.satisfied for i in incidents))
        self.assertTrue(all(i.snippet is not None and i.snippet.kind == "frame" for i in incidents))
        # D4: a non-entity-scoped stream stays entity_id=None (the back-compatible default) and the
        # field is present on the wire payload rather than missing.
        self.assertTrue(all(i.entity_id is None for i in incidents))
        self.assertIn("entity_id", incident_payload(incidents[0]))

    def test_on_incident_sink_is_called(self):
        seen = []
        asyncio.run(_collect(_core(), SOURCE, on_incident=seen.append))
        self.assertEqual(len(seen), 2)

    def test_wake_policy_none_explicit_still_fires_on_a_wakeup_worthy_source(self):
        # A4 regression: run()'s `wake_policy` kwarg (default None) flows straight into the
        # MotionGate `_build_gate()` constructs (percept/runtime/worker.py). Passed explicitly here
        # to prove the real, wired path — not just MotionGate in isolation — never silently drops a
        # wake-worthy escalation when no WakePolicy is configured.
        incidents = asyncio.run(_collect(_core(), SOURCE, wake_policy=None))
        self.assertEqual(len(incidents), 2)  # identical to the implicit-default path above

    def test_wake_policy_none_a_motionless_source_fires_nothing(self):
        # The negative half, driven through the same real worker.run() path: no wake-worthy
        # evidence anywhere in the stream -> wake_policy=None must not spuriously escalate either.
        still_only = [(0.0, STILL, "clear"), (1.0, STILL, "clear"), (2.0, STILL, "clear")]
        incidents = asyncio.run(_collect(_core(), still_only, wake_policy=None))
        self.assertEqual(incidents, [])

    def test_on_frame_feed_hook_gets_every_frame(self):
        seen = []
        asyncio.run(_collect(_core(), SOURCE, on_frame=lambda t, f: seen.append(t)))
        self.assertEqual(seen, [0.0, 1.0, 2.0, 3.0])  # the live-feed hook sees every frame, gated or not

    def test_async_hooks_are_awaited(self):
        seen = []

        async def on_incident(incident):
            await asyncio.sleep(0)
            seen.append(incident.id)

        incidents = asyncio.run(_collect(_core(), SOURCE[:2], on_incident=on_incident))
        self.assertEqual(len(incidents), 1)
        self.assertEqual(seen, [incidents[0].id])

    def test_sink_errors_can_be_ignored(self):
        def broken(_incident):
            raise RuntimeError("sink down")

        stats = RuntimeStats()
        incidents = asyncio.run(
            _collect(_core(), SOURCE[:2], on_incident=broken, sink_errors="ignore", stats=stats)
        )
        self.assertEqual(len(incidents), 1)
        self.assertEqual(stats.sink_errors, 1)

    def test_runtime_stats_count_frames_escalations_fires_and_incidents(self):
        stats = RuntimeStats()
        incidents = asyncio.run(_collect(_core(), SOURCE, stats=stats))
        self.assertEqual(len(incidents), 2)
        self.assertEqual(stats.frames, 4)
        self.assertEqual(stats.escalations, 3)
        self.assertEqual(stats.judge_calls, 3)
        self.assertEqual(stats.fires, 2)
        self.assertEqual(stats.asks, 0)
        self.assertEqual(stats.incidents, 2)

    def test_runtime_health_events_report_lifecycle_and_incidents(self):
        events = []
        incidents = asyncio.run(_collect(_core(), SOURCE[:2], on_health=events.append))
        self.assertEqual(len(incidents), 1)
        kinds = [e.kind for e in events]
        self.assertEqual(kinds[0], "runtime_started")
        self.assertIn("escalation", kinds)
        self.assertIn("incident", kinds)
        self.assertEqual(kinds[-1], "runtime_ended")
        self.assertTrue(all(isinstance(e, RuntimeHealthEvent) for e in events))

    def test_runtime_ended_carries_this_run_s_coverage_counters(self):
        # X4: `runtime_ended` used to carry an EMPTY payload, so a consumer that had not kept its own
        # counters could not tell what the run actually did. It now states the run's coverage.
        events = []
        stats = RuntimeStats()
        asyncio.run(_collect(_core(), SOURCE, on_health=events.append, stats=stats))
        ended = [e for e in events if e.kind == "runtime_ended"]
        self.assertEqual(len(ended), 1)
        self.assertEqual(ended[0].data, {
            "frames_seen": 4, "admitted": 3,
            "signal_admits": 3, "fixed_admits": 0, "coast_admits": 0, "audio_wakes": 0,
            "fixed_opportunities": 0,
            "goal_admissions": 3,
            "admission_by_goal": {"zone": 3},
            "admission_by_cause": {"motion": 3},
            "perceive_dispatches": 3,
            "judge_calls": 3, "incidents": 2,
            "source_coverage": {
                "video": {
                    "observed": 4, "excluded": 0, "discontinuities": 0,
                    "missing_sequences": 0,
                },
                "audio": {
                    "observed": 0, "excluded": 0, "discontinuities": 0,
                    "missing_sequences": 0,
                },
            },
        })
        # the per-frame admission split partitions the total.
        self.assertEqual(
            ended[0].data["admitted"],
            ended[0].data["signal_admits"]
            + ended[0].data["fixed_admits"]
            + ended[0].data["coast_admits"]
            + ended[0].data["audio_wakes"],
        )
        # the payload is exactly what the caller's own stats object saw — one set of counters
        self.assertEqual(ended[0].data["frames_seen"], stats.frames)
        self.assertEqual(ended[0].data["admitted"], stats.escalations)
        self.assertEqual(ended[0].data["judge_calls"], stats.judge_calls)
        self.assertEqual(ended[0].data["incidents"], stats.incidents)

    def test_zero_admission_run_emits_zero_admission_coverage(self):
        # The client's original complaint: a whole session admitted nothing and the runtime was
        # SILENT about it. A motionless source never clears the gate -> the fact is now stated.
        still_only = [(float(i), STILL, "clear") for i in range(5)]
        events = []
        incidents = asyncio.run(_collect(_core(), still_only, on_health=events.append))
        self.assertEqual(incidents, [])
        zero = [e for e in events if e.kind == "zero_admission_coverage"]
        self.assertEqual(len(zero), 1)
        self.assertEqual(zero[0].data["frames_seen"], 5)
        self.assertEqual(zero[0].data["admitted"], 0)
        self.assertEqual(zero[0].data["judge_calls"], 0)
        self.assertEqual(zero[0].data["incidents"], 0)
        # it names the bar the frames never cleared, so the reader can tune the eye
        self.assertIsInstance(zero[0].data["threshold"], float)
        self.assertIn("stat", zero[0].data)
        kinds = [e.kind for e in events]
        # emitted PRECEDING runtime_ended, which still closes the stream
        self.assertEqual(kinds[-2:], ["zero_admission_coverage", "runtime_ended"])

    def test_a_run_that_admits_anything_does_not_report_zero_admission(self):
        events = []
        asyncio.run(_collect(_core(), SOURCE[:2], on_health=events.append))
        self.assertNotIn("zero_admission_coverage", [e.kind for e in events])

    def test_an_empty_source_does_not_report_zero_admission(self):
        # Zero frames is not zero-admission-over-N-frames: nothing was observed to admit FROM.
        events = []
        asyncio.run(_collect(_core(), [], on_health=events.append))
        self.assertNotIn("zero_admission_coverage", [e.kind for e in events])
        ended = [e for e in events if e.kind == "runtime_ended"]
        self.assertEqual(ended[0].data["frames_seen"], 0)

    def test_zero_admission_coverage_is_not_a_degradation_verdict(self):
        # CRITICAL: a still camera under a motion-gated watch CORRECTLY admits nothing and saves
        # every model call. This event states a FACT for a consumer to interpret — it must never
        # latch an unhealthy interval, or percept would be alarming on its own economic claim.
        from percept.tracing import DEGRADED_HEALTH_KINDS

        self.assertNotIn("zero_admission_coverage", DEGRADED_HEALTH_KINDS)

    def test_zero_admission_still_camera_stays_non_alarming_with_no_declared_floor(self):
        # Regression (card C2): still-camera, motion-gated, admitting 0 — with NO coverage_floor
        # passed to run() (the default) — must stay byte-for-byte the pre-existing diagnostic-only
        # reading: floor_unmet is False and coverage_floor is None in the payload.
        still_only = [(float(i), STILL, "clear") for i in range(5)]
        events = []
        asyncio.run(_collect(_core(), still_only, on_health=events.append))
        zero = [e for e in events if e.kind == "zero_admission_coverage"]
        self.assertEqual(len(zero), 1)
        self.assertIsNone(zero[0].data["coverage_floor"])
        self.assertFalse(zero[0].data["floor_unmet"])

    def test_zero_admission_with_declared_unmet_coverage_floor_is_flagged(self):
        # A DECLARED, UNMET coverage_floor (e.g. sourced from a SourceProfile via
        # percept.runtime_compiler) is what changes the reading — never the zero itself.
        still_only = [(float(i), STILL, "clear") for i in range(5)]
        events = []
        asyncio.run(_collect(_core(), still_only, on_health=events.append, coverage_floor=1.0))
        zero = [e for e in events if e.kind == "zero_admission_coverage"]
        self.assertEqual(len(zero), 1)
        self.assertEqual(zero[0].data["coverage_floor"], 1.0)
        self.assertTrue(zero[0].data["floor_unmet"])

    def test_zero_admission_with_declared_met_coverage_floor_of_zero_is_not_flagged(self):
        # A DECLARED floor of exactly 0 admissions ("this source is expected to be quiet") is MET
        # by an actual zero, so floor_unmet stays False even though a floor was declared.
        still_only = [(float(i), STILL, "clear") for i in range(5)]
        events = []
        asyncio.run(_collect(_core(), still_only, on_health=events.append, coverage_floor=0.0))
        zero = [e for e in events if e.kind == "zero_admission_coverage"]
        self.assertEqual(zero[0].data["coverage_floor"], 0.0)
        self.assertFalse(zero[0].data["floor_unmet"])

    def test_runtime_health_reports_sink_errors(self):
        events = []

        def broken(_incident):
            raise RuntimeError("sink down")

        asyncio.run(
            _collect(
                _core(),
                SOURCE[:2],
                on_incident=broken,
                on_health=events.append,
                sink_errors="ignore",
            )
        )
        sink_errors = [e for e in events if e.kind == "incident_sink_error"]
        self.assertEqual(len(sink_errors), 1)
        self.assertIn("sink down", sink_errors[0].data["error"])

    def test_ask_events_do_not_become_incidents(self):
        asks = []
        source = [(0.0, STILL, "maybe"), (1.0, MOVED, "maybe")]
        incidents = asyncio.run(_collect(_ask_core(), source, on_ask=asks.append))
        self.assertEqual(incidents, [])
        self.assertEqual(len(asks), 1)
        self.assertEqual(asks[0].action, "ask")

    def test_clip_snippet_is_encoded_as_json_safe_bundle(self):
        incidents = asyncio.run(_collect(_core(), SOURCE[:2], snippet_kind="clip"))
        self.assertEqual(len(incidents), 1)
        snip = incidents[0].snippet
        self.assertEqual(snip.kind, "clip")
        self.assertEqual(snip.mime, "application/x-percept-frame-bundle+json")
        payload = json.loads(snip.media.decode("utf-8"))
        self.assertEqual(payload["version"], 1)
        self.assertEqual(len(payload["frames"]), 2)

    def test_webhook_payload_redacts_media_by_default(self):
        incident = asyncio.run(_collect(_core(), SOURCE[:2], snippet_kind="clip"))[0]
        payload = incident_payload(incident)
        self.assertIsNone(payload["snippet"]["media"])
        self.assertIsNone(payload["snippet"]["frames"])
        self.assertEqual(payload["message"], "Alert: person in zone")

    def test_incident_ids_are_unique_for_same_goal_and_timestamp(self):
        fire = asyncio.run(_core().perceive_judged("intruder", t=1.0))[0]
        first = make_incident(fire, ts=1.0)
        second = make_incident(fire, ts=1.0)
        self.assertNotEqual(first.id, second.id)

    def test_source_uri_normalization(self):
        self.assertEqual(_normalize_uri("webcam"), 0)
        self.assertEqual(_normalize_uri("webcam:1"), 1)
        self.assertEqual(_normalize_uri("webcam://2"), 2)
        self.assertEqual(_normalize_uri("video.mp4"), "video.mp4")

    def test_uri_source_options_pass_through(self):
        seen = {}

        async def fake_open_source(uri, **kwargs):
            seen["uri"] = uri
            seen["kwargs"] = kwargs
            for item in SOURCE[:2]:
                yield item

        with patch("percept.runtime.source.open_source", fake_open_source):
            incidents = asyncio.run(
                _collect(
                    _core(),
                    "rtsp://camera/stream",
                    source_queue_size=7,
                    source_reconnects=3,
                    source_reconnect_backoff_s=0.1,
                    source_rtsp_tcp=False,
                )
            )
        self.assertEqual(len(incidents), 1)
        self.assertEqual(seen["uri"], "rtsp://camera/stream")
        self.assertEqual(seen["kwargs"]["maxsize"], 7)
        self.assertEqual(seen["kwargs"]["reconnects"], 3)
        self.assertEqual(seen["kwargs"]["reconnect_backoff_s"], 0.1)
        self.assertIs(seen["kwargs"]["rtsp_tcp"], False)

    def test_still_stream_emits_nothing(self):
        still = [(float(t), STILL, "clear") for t in range(5)]  # no motion -> no escalation -> no cloud call
        self.assertEqual(asyncio.run(_collect(_core(), still)), [])


# --- TODO 1 / Build 5: the runtime worker is cadence-aware for brief-action goals ----------------
def _brief_vision() -> FakeVision:
    return FakeVision(script={
        "wave": Verdict(satisfied=True, confidence=0.95, evidence="a hand waves"),
        "clear": Verdict(satisfied=False, confidence=0.95, evidence="nothing"),
    })


def _brief_core() -> Percept:
    core = Percept.create(vision=_brief_vision(), discover_plugins=False)
    core.add_goal(Goal(id="wave", condition="someone briefly waves", say="waved",
                       state={"sense": {"brief_window_s": 0.5}}))  # the brief-action prior
    return core


def _plain_wave_core() -> Percept:
    core = Percept.create(vision=_brief_vision(), discover_plugins=False)
    core.add_goal(Goal(id="wave", condition="someone waves", say="waved"))  # NO brief_window_s
    return core


# all-STILL stream: the motion gate NEVER escalates, so ONLY a brief-action coast can judge the quiet
# in-window "wave" frame. base_fps = 2/0.5s = 4 fps -> coast judges at t = 0.0, 0.25, 0.5.
QUIET_SRC = [
    (0.0, STILL, "clear"),
    (0.125, STILL, "clear"),
    (0.25, STILL, "wave"),    # quiet, in-window: gate skips it, coast judges it -> FIRE
    (0.375, STILL, "clear"),
    (0.5, STILL, "clear"),
]


class BriefActionCadenceTests(unittest.TestCase):
    def test_capture_plan_brief_goal_raises_capture_and_arms_coast(self):
        g = Goal(id="b", condition="someone briefly waves", say="hi",
                 state={"sense": {"brief_window_s": 0.5}})
        cap, coast, required = _capture_plan([g], 4.0)              # D1c: also returns the Nyquist floor
        self.assertEqual(cap, 8.0)                                  # raised to min(headroom, max_fps)
        self.assertIsNotNone(coast)                                 # fovea coast armed
        self.assertEqual(coast.policy("__coast__").base_fps, 4.0)   # 2 / 0.5s window Nyquist
        self.assertEqual(required, 4.0)                             # coverage floor the source must meet

    def test_capture_plan_ordinary_goal_is_byte_for_byte_default(self):
        g = Goal(id="n", condition="a person is in the zone", say="alert")
        cap, coast, required = _capture_plan([g], 4.0)
        self.assertEqual(cap, 4.0)                                  # unchanged capture rate
        self.assertIsNone(coast)                                    # no coast -> gate-only path
        self.assertEqual(required, 0.0)                             # nothing promised -> no tripwire

    def test_brief_goal_coast_judges_a_quiet_in_window_frame(self):
        # the win: the coast covers a sub-second window the motion gate would miss entirely.
        incidents = asyncio.run(_collect(_brief_core(), QUIET_SRC))
        self.assertEqual(len(incidents), 1)

    def test_ordinary_goal_on_same_quiet_stream_emits_nothing(self):
        # the A/B control: same quiet stream, a non-brief goal has no coast -> gate never escalates -> silent.
        incidents = asyncio.run(_collect(_plain_wave_core(), QUIET_SRC))
        self.assertEqual(len(incidents), 0)


if __name__ == "__main__":
    unittest.main()
