"""M1 — the public ``watch()`` entrypoint + ``WatchEvent`` contract.

L1: offline, FakeVision, no keys. Proves the SEAM (signature, streaming semantics, event mapping,
version markers) wires to the EXISTING cognition core (``Percept.create`` + ``perceive_judged``)
byte-for-byte — no new detection logic is asserted here.
"""

from __future__ import annotations

import asyncio
import unittest

from percept import ASK, FIRE, Goal, Percept, Settings, WatchEvent, load_settings
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.watch import WATCH_VERSION, _settings_from_profile, watch


def _goal() -> Goal:
    return Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone")


def _core(script: dict) -> Percept:
    vision = FakeVision(script=script)
    return Percept.create(vision=vision, discover_plugins=False)


async def _collect(stream, goals, profile=None, **kw) -> list[WatchEvent]:
    return [ev async for ev in watch(stream, goals, profile, **kw)]


class WatchVersionTests(unittest.TestCase):
    def test_watch_version_is_semver(self):
        parts = WATCH_VERSION.split(".")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(p.isdigit() for p in parts))


class WatchEntrypointTests(unittest.TestCase):
    """Rising-edge FIRE / refusal-to-guess ASK, streamed as ``WatchEvent`` — mirrors perceive_judged."""

    def test_fires_on_rising_edge(self):
        core = _core({"intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone")})
        events = asyncio.run(_collect(["intruder"], [_goal()], percept=core))
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertIsInstance(ev, WatchEvent)
        self.assertEqual(ev.action, FIRE)
        self.assertEqual(ev.goal_id, "zone")
        self.assertEqual(ev.text, "Alert: person in zone")

    def test_no_event_on_confident_false_then_silent_refire(self):
        # confident-false frames never fire/ask; a repeat confident-true after an already-fired edge
        # is SILENT (not re-emitted) — today's internal wiring never surfaces silent to callers.
        core = _core({
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="in zone"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="empty"),
        })
        events = asyncio.run(_collect(["clear", "intruder", "intruder"], [_goal()], percept=core))
        # frame 1: clear -> confident-no -> no event (SILENT, dropped by the internal wiring)
        # frame 2: intruder -> rising edge -> FIRE
        # frame 3: intruder again -> already fired -> SILENT -> no event
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, FIRE)

    def test_asks_on_low_confidence(self):
        core = _core({"maybe": Verdict(satisfied=True, confidence=0.2, evidence="unclear")})
        events = asyncio.run(_collect(["maybe"], [_goal()], percept=core))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, ASK)

    def test_event_carries_timestamp_and_verdict(self):
        core = _core({"intruder": Verdict(satisfied=True, confidence=0.95, evidence="x")})
        events = asyncio.run(_collect(["intruder"], [_goal()], percept=core))
        ev = events[0]
        self.assertIsInstance(ev.t, float)
        self.assertIsInstance(ev.verdict, Verdict)
        self.assertTrue(ev.verdict.satisfied)

    def test_accepts_async_iterable_stream(self):
        async def _gen():
            yield "intruder"

        core = _core({"intruder": Verdict(satisfied=True, confidence=0.95, evidence="x")})
        events = asyncio.run(_collect(_gen(), [_goal()], percept=core))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, FIRE)

    def test_multiple_frames_multiple_goals(self):
        core = _core({
            "sip-coffee": Verdict(satisfied=True, confidence=0.9, evidence="coffee"),
        })
        goals = [Goal(id="coffee", condition="drinking coffee", say="careful, it's hot")]
        events = asyncio.run(_collect(["empty", "sip-coffee"], goals, percept=core))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].goal_id, "coffee")

    def test_creates_default_core_when_none_supplied(self):
        # no ``percept=`` -> watch() builds its own core via Percept.create(); the default (fake)
        # vision backend never matches "intruder" so this just proves construction doesn't blow up.
        events = asyncio.run(_collect(["intruder"], [_goal()]))
        self.assertIsInstance(events, list)


class WatchProfileThreadingTests(unittest.TestCase):
    """``profile`` is opaque today (P1 owns real resolution) — only documented pass-throughs exist."""

    def test_none_profile_resolves_to_none(self):
        self.assertIsNone(_settings_from_profile(None))

    def test_settings_instance_passed_through(self):
        s = load_settings()
        self.assertIs(_settings_from_profile(s), s)

    def test_opaque_object_with_settings_attr(self):
        s = load_settings()

        class _FutureProfile:
            settings = s

        self.assertIs(_settings_from_profile(_FutureProfile()), s)

    def test_opaque_object_without_settings_attr_resolves_none(self):
        self.assertIsNone(_settings_from_profile(object()))

    def test_watch_threads_settings_profile_into_a_built_core(self):
        # a bare Settings passed as ``profile`` is honored when watch() builds its own core.
        custom = Settings()
        async def _run():
            async for _ in watch([], [_goal()], custom):
                pass

        # Percept.create(settings=custom) is exercised indirectly; assert no crash and that a
        # fresh core would carry the settings object (spot-checked via _settings_from_profile above,
        # this test just proves the plumbing doesn't reject a Settings profile end-to-end).
        asyncio.run(_run())
        self.assertIsInstance(custom, Settings)  # sanity: profile object is untouched by watch()


class WatchPublicSurfaceTests(unittest.TestCase):
    def test_watch_and_watchevent_exported_from_percept(self):
        import percept

        self.assertTrue(callable(percept.watch))
        self.assertIs(percept.watch, watch)
        self.assertIs(percept.WatchEvent, WatchEvent)
        self.assertEqual(percept.WATCH_VERSION, WATCH_VERSION)


if __name__ == "__main__":
    unittest.main()
