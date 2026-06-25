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

    def test_on_incident_sink_is_called(self):
        seen = []
        asyncio.run(_collect(_core(), SOURCE, on_incident=seen.append))
        self.assertEqual(len(seen), 2)

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


if __name__ == "__main__":
    unittest.main()
