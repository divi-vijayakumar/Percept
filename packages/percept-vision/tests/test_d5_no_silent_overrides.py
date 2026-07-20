"""Card D5 — three silent behaviours in ``run()``, each of which cost a client team runs.

  (a) ``run(core=…, goals=…, vision=…)`` silently DISCARDED goals/vision when a core was given: the
      watch ran byte-identically and simply never armed the caller's goals. It must RAISE — a warning
      still permits the silent failure.
  (b) ``fps`` was silently forced to 8.0 for ``str`` sources (an unrequested 5x cost override). The
      override may be correct; doing it INVISIBLY is not. The run now RECORDS the effective config
      (requested/effective/reason) instead of only warning.
  (c) ``timebase="wall"`` for every ``str`` source made ``incident.ts`` meaningless against media time
      (a 36s clip decoded in 0.72s stamped every incident with the decode instant). ``timebase="auto"``
      resolves to MEDIA for recorded sources, WALL for live devices, with an explicit override — and
      BOTH clocks are recorded regardless.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.runtime import RuntimeStats, run
from percept.runtime.worker import (
    FPS_REASONS,
    TIMEBASE_CHOICES,
    EffectiveConfig,
    _frame_clocks,
    _is_recorded_source,
    _resolve_timebase,
)

STILL = gray_frame(8, 8, 0)


def _core(goal_id="g"):
    vision = FakeVision(script={"tok": Verdict(satisfied=False, confidence=0.9, evidence="n/a")})
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id=goal_id, condition="the cup is up", say="!"))
    return core


def _frames(n=3):
    return [(float(i) / 4.0, STILL, "tok") for i in range(n)]


def _fake_open_source(captured):
    async def fake_open_source(uri, *, fps=4.0, timebase="media", dropped=None, health=None, **_kw):
        captured["uri"] = uri
        captured["fps"] = fps
        captured["timebase"] = timebase
        for i in range(4):
            # faithful to source.py: media => i/fps, wall => a monotonic-looking stamp
            t = (1_000_000.0 + i * 0.01) if timebase == "wall" else i / fps
            yield (t, STILL, "tok")
    return fake_open_source


# ==================================================================================================
# (a) core + goals/vision is a HARD error, not a silent discard.
# ==================================================================================================
class CoreWithGoalsRaisesTests(unittest.TestCase):
    def _drain(self, **kw):
        async def go():
            async for _ in run(**kw):
                pass
        asyncio.run(go())

    def test_core_plus_goals_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._drain(core=_core(), source=_frames(), goals=["the milk is boiling over"])
        self.assertEqual(
            str(ctx.exception),
            "goals= and vision= cannot be supplied when core= is provided; "
            "apply goals to the core or omit core.",
        )

    def test_core_plus_vision_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._drain(core=_core(), source=_frames(), vision="fake")

    def test_core_plus_both_raises_before_any_frame_is_consumed(self):
        consumed = []

        async def source():
            for item in _frames():
                consumed.append(item)
                yield item

        with self.assertRaises(ValueError):
            self._drain(core=_core(), source=source(), goals=["x"], vision="fake")
        self.assertEqual(consumed, [], "the run must refuse to START, not fail partway through")

    def test_core_alone_and_goals_alone_still_work(self):
        self._drain(core=_core(), source=_frames())  # canonical form untouched
        self._drain(source=_frames(), goals=["the milk is boiling over"], vision="fake")

    def test_empty_goals_alongside_core_is_not_an_error(self):
        # goals=None / [] is not a supplied override — the canonical call must not be made to raise.
        self._drain(core=_core(), source=_frames(), goals=None)
        self._drain(core=_core(), source=_frames(), goals=[])


# ==================================================================================================
# (b) the effective capture config is RECORDED, with a reason from a closed vocabulary.
# ==================================================================================================
class EffectiveFpsRecordedTests(unittest.TestCase):
    def _run(self, source, **kw):
        stats = RuntimeStats()
        events: list = []
        core = kw.pop("core", None) or _core()

        async def go():
            async for _ in run(core, source=source, stats=stats,
                               on_health=lambda e: events.append(e), **kw):
                pass

        asyncio.run(go())
        return core, stats, events

    def test_deterministic_source_records_the_requested_fps_unchanged(self):
        _core_, stats, events = self._run(_frames(), fps=4.0)
        cfg = stats.effective_config
        self.assertIsInstance(cfg, EffectiveConfig)
        self.assertEqual((cfg.fps.requested_fps, cfg.fps.effective_fps), (4.0, 4.0))
        self.assertEqual(cfg.fps.reason, "profile-default")
        self.assertFalse(cfg.fps.overridden)

    def test_str_source_records_the_8fps_override_instead_of_hiding_it(self):
        captured: dict = {}
        with patch("percept.runtime.source.open_source", _fake_open_source(captured)):
            _core_, stats, events = self._run("clip.mp4", fps=4.0)
        # Behaviour is unchanged (the source really is opened at the headroom cap) …
        self.assertEqual(captured["fps"], 8.0)
        # … but it is no longer INVISIBLE.
        cfg = stats.effective_config
        self.assertEqual((cfg.fps.requested_fps, cfg.fps.effective_fps), (4.0, 8.0))
        self.assertEqual(cfg.fps.reason, "source-limit")
        self.assertTrue(cfg.fps.overridden)

    def test_brief_goal_records_brief_window_coverage_as_the_reason(self):
        core = Percept.create(vision=FakeVision(script={}), discover_plugins=False)
        core.add_goal(Goal(id="brief", condition="the cat briefly appears", say="!",
                           state={"sense": {"brief_window_s": 0.5}}))
        _core_, stats, _events = self._run(_frames(), core=core, fps=1.0)
        cfg = stats.effective_config
        self.assertEqual(cfg.fps.requested_fps, 1.0)
        self.assertGreater(cfg.fps.effective_fps, 1.0)
        self.assertEqual(cfg.fps.reason, "brief-window-coverage")

    def test_reason_is_always_from_the_closed_vocabulary(self):
        for source in (_frames(), _frames(1)):
            _core_, stats, _ = self._run(source)
            self.assertIn(stats.effective_config.fps.reason, FPS_REASONS)

    def test_effective_config_is_emitted_once_on_health_and_mirrored_on_the_core(self):
        core, stats, events = self._run(_frames(), fps=4.0)
        cfgs = [e for e in events if e.kind == "effective_config"]
        self.assertEqual(len(cfgs), 1, "the record is announced exactly once, at start")
        self.assertEqual(cfgs[0].data, stats.effective_config.as_dict())
        self.assertEqual(cfgs[0].data["fps"]["requested_fps"], 4.0)
        self.assertEqual(getattr(core, "effective_config", None), stats.effective_config)

    def test_an_fps_override_alone_emits_no_warning_health_event(self):
        # "emit health only when a requirement is actually unmet": no brief goal is armed here, so
        # nothing is under-covered and no capture_plan_unmet may fire despite the 4->8 override.
        captured: dict = {}
        with patch("percept.runtime.source.open_source", _fake_open_source(captured)):
            _core_, _stats, events = self._run("clip.mp4", fps=4.0)
        self.assertNotIn("capture_plan_unmet", [e.kind for e in events])


# ==================================================================================================
# (c) timebase="auto" — media for recorded sources, wall for live ones, override honoured.
# ==================================================================================================
class TimebaseResolutionTests(unittest.TestCase):
    def test_recorded_sources_are_recognised(self):
        for uri in ("clip.mp4", "/data/CLIP.MOV", "file:///x/y.mkv", "session.wav", "a.webm?x=1"):
            self.assertTrue(_is_recorded_source(uri), uri)

    def test_live_sources_are_recognised(self):
        for uri in ("webcam", "webcam:0", "webcam://1", "mic:0", "rtsp://cam/stream",
                    "udp://1.2.3.4:5", "screen://0", "some-opaque-stream"):
            self.assertFalse(_is_recorded_source(uri), uri)

    def test_auto_resolves_media_for_a_file_and_wall_for_a_camera(self):
        rec = _resolve_timebase("clip.mp4", "auto")
        self.assertEqual((rec.resolved, rec.reason), ("media", "recorded-source"))
        live = _resolve_timebase("webcam:0", "auto")
        self.assertEqual((live.resolved, live.reason), ("wall", "live-source"))

    def test_explicit_override_wins_both_ways(self):
        self.assertEqual(_resolve_timebase("clip.mp4", "wall").resolved, "wall")
        self.assertEqual(_resolve_timebase("webcam:0", "media").resolved, "media")
        self.assertEqual(_resolve_timebase("clip.mp4", "wall").reason, "explicit-override")

    def test_non_str_source_is_never_restamped(self):
        rec = _resolve_timebase(_frames(), "auto")
        self.assertEqual(rec.reason, "deterministic-stream")

    def test_unknown_timebase_raises(self):
        with self.assertRaises(ValueError):
            _resolve_timebase("clip.mp4", "monotonic")
        self.assertEqual(TIMEBASE_CHOICES, ("auto", "media", "wall"))

    def test_frame_clocks_record_both_regardless_of_the_stamped_one(self):
        media, wall = _frame_clocks(3.5, timebase="media", t0=0.0, wall_now=1000.0)
        self.assertEqual((media, wall), (3.5, 1000.0))
        media, wall = _frame_clocks(1002.0, timebase="wall", t0=1000.0, wall_now=1002.0)
        self.assertEqual((media, wall), (2.0, 1002.0))


class TimebaseEndToEndTests(unittest.TestCase):
    def _run(self, source, **kw):
        captured: dict = {}
        stats = RuntimeStats()

        async def go():
            async for _ in run(_core(), source=source, stats=stats, **kw):
                pass

        with patch("percept.runtime.source.open_source", _fake_open_source(captured)):
            asyncio.run(go())
        return captured, stats

    def test_a_recorded_clip_now_runs_on_media_time(self):
        captured, stats = self._run("clip.mp4")
        self.assertEqual(captured["timebase"], "media")
        self.assertEqual(stats.effective_config.video_timebase.resolved, "media")
        # incident.ts would now be a media offset: the last frame's media t, not a decode instant.
        self.assertLess(stats.last_media_t, 10.0)

    def test_a_live_camera_still_runs_on_wall_time(self):
        captured, stats = self._run("webcam:0")
        self.assertEqual(captured["timebase"], "wall")
        self.assertEqual(stats.effective_config.video_timebase.resolved, "wall")
        self.assertGreater(stats.last_wall_t, 1000.0)

    def test_explicit_timebase_overrides_auto_end_to_end(self):
        captured, stats = self._run("clip.mp4", timebase="wall")
        self.assertEqual(captured["timebase"], "wall")
        self.assertEqual(stats.effective_config.video_timebase.reason, "explicit-override")

    def test_both_clocks_are_recorded_on_a_media_run(self):
        _captured, stats = self._run("clip.mp4")
        self.assertIsNotNone(stats.last_media_t)
        self.assertIsNotNone(stats.last_wall_t)
        self.assertNotEqual(stats.last_media_t, stats.last_wall_t)

    def test_both_clocks_are_recorded_on_a_wall_run(self):
        _captured, stats = self._run("webcam:0")
        self.assertIsNotNone(stats.last_media_t)
        self.assertIsNotNone(stats.last_wall_t)
        # media-equivalent on a live source is the elapsed stream offset, not the monotonic stamp
        self.assertLess(stats.last_media_t, 1.0)

    def test_effective_config_serializes_to_a_flat_record(self):
        _captured, stats = self._run("clip.mp4")
        d = stats.effective_config.as_dict()
        self.assertEqual(set(d), {"fps", "video_timebase"})
        self.assertEqual(set(d["fps"]),
                         {"requested_fps", "effective_fps", "reason", "overridden"})
        self.assertEqual(set(d["video_timebase"]), {"requested", "resolved", "reason"})


if __name__ == "__main__":
    unittest.main()
