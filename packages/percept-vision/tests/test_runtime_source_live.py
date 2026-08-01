"""HLS/DASH live sources, and the media-clock rule that separates them from progressive HTTP files.

Two things are locked here:

  * **The media-clock bug.** An ``https://….m3u8`` LIVE stream used to fall through the "no
    trustworthy media clock" scheme test, so ``CAP_PROP_POS_MSEC`` — a position inside whatever
    sliding playlist window the demuxer currently holds — was stamped on the envelope as source
    provenance and flowed into latency math. It is now classified live. A plain
    ``https://…/video.mp4`` is a FILE fetched over HTTP and MUST keep its real media clock.

  * **The live path.** Manifest/`ytdlp:` URIs open on the FFMPEG backend with live-edge options, and
    fold a failed open into the EXISTING ``reconnects``/``reconnect_backoff_s`` mechanism rather
    than a second one. ``rtsp://``/``file``/``webcam:`` behaviour is unchanged, and there are
    regression guards below that say so.

No network: every capture runs against an in-process fake cv2, and ``yt_dlp`` is a fake module
injected into ``sys.modules``. Nothing in this file resolves or opens a real stream.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import unittest
from unittest.mock import patch

from percept.runtime.source import (
    _FFMPEG_OPTIONS_ENV,
    SourceConfig,
    SourceConfigurationError,
    _attempts_remain,
    _capture,
    _capture_target,
    _has_media_clock,
    _is_live_stream,
    _live_ffmpeg_options,
    _open_capture,
    _reconnect_delay,
    _sleep_backoff,
    _source_time,
)


# --- fakes ---------------------------------------------------------------------------------------
class _FakeGray:
    def __init__(self, n):
        self.n = n

    def flatten(self):
        return [0] * self.n


class _FakeBuffer:
    def tobytes(self):
        return b"\xff\xd8jpeg\xff\xd9"


class _FakeCv2:
    CAP_FFMPEG = 1900
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 0
    COLOR_BGR2GRAY = 6

    def __init__(self, cap=None, caps=None):
        self.calls = []
        self.env_at_open = []
        self.cap = cap
        self.caps = list(caps) if caps is not None else None

    def VideoCapture(self, *args):
        self.calls.append(args)
        self.env_at_open.append(os.environ.get(_FFMPEG_OPTIONS_ENV))
        if self.caps is not None:
            return self.caps.pop(0) if self.caps else _FakeCap(opened=False)
        return self.cap if self.cap is not None else _FakeCap(reads=[(False, None)])

    def cvtColor(self, bgr, code):
        return bgr

    def resize(self, gray, size):
        return _FakeGray(size[0] * size[1])

    def imencode(self, ext, bgr):
        return True, _FakeBuffer()


class _FakeCap:
    """A decoder whose reported position ADVANCES — the shape that made the bug observable.

    A permanently-zero ``POS_MSEC`` is already rejected by the non-advancing guard, so a fake that
    never advances could not tell the two classifications apart.
    """

    def __init__(self, opened=True, reads=None, fps=30.0, step_ms=250.0, on_release=None):
        self.opened = opened
        self.reads = list(reads or [])
        self.fps = fps
        self.step_ms = step_ms
        self.position_ms = 0.0
        self.released = False
        self.sets = []
        self.on_release = on_release

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.position_ms if prop == _FakeCv2.CAP_PROP_POS_MSEC else self.fps

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def read(self):
        if not self.reads:
            return (False, None)
        item = self.reads.pop(0)
        if item[0]:
            self.position_ms += self.step_ms
        return item

    def release(self):
        self.released = True
        if self.on_release is not None:
            self.on_release()


class _UnsettableCap(_FakeCap):
    def set(self, prop, value):
        raise RuntimeError("this backend does not support CAP_PROP_BUFFERSIZE")


def _drain(q):
    """Every non-sentinel item on the queue.

    Drained to EMPTY rather than to the ``None`` end marker on purpose: ``_put_terminal`` declines to
    publish a sentinel once ``stop`` is set, which is exactly how the cancellation-driven tests below
    end their capture.
    """
    items = []
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return items
        if item is not None:
            items.append(item)


# --- 1. the media-clock rule ---------------------------------------------------------------------
class MediaClockClassificationTests(unittest.TestCase):
    def test_live_http_manifests_have_no_media_clock(self):
        for uri in (
            "https://cam.example/hls/traffic.m3u8",
            "http://cam.example/traffic.m3u8",
            "https://cam.example/stream.M3U8",  # scheme + suffix compared case-insensitively
            "https://cam.example/manifest.mpd",
            "https://cam.example/stream.m3u8?token=abc123",  # query stripped before the suffix test
            "https://cam.example/stream.m3u8#frag",  # fragment stripped too
            "https://cam.example/live/edge",  # no extension, but the route says live
            "https://cam.example/dash/edge",
            "https://cam.example/play?islive=true",  # query marker
        ):
            with self.subTest(uri=uri):
                self.assertFalse(_has_media_clock(uri))

    def test_progressive_http_files_keep_their_media_clock(self):
        for uri in (
            "https://cdn.example/video.mp4",
            "http://cdn.example/video.mp4",
            "https://cdn.example/clip.webm?sig=deadbeef",
            "https://cdn.example/clip.mkv#t=10",
            # A file that merely SITS under a live-looking route is still a file: the progressive
            # extension is checked first, so the route marker cannot demote it.
            "https://cdn.example/live/archive/clip.mp4",
            # ...and a live-looking HOST is never a marker at all, because the authority is stripped
            # before any substring test runs.
            "https://live.cdn.example/archive/clip.mov",
        ):
            with self.subTest(uri=uri):
                self.assertTrue(_has_media_clock(uri))

    def test_extensionless_http_endpoint_keeps_todays_behaviour(self):
        # Genuinely undecidable from the URI. Rather than silently strip provenance from what may be
        # a progressive fetch, the classification is left alone and the caller can override it.
        self.assertTrue(_has_media_clock("https://cdn.example/asset/9f2c"))
        self.assertFalse(_has_media_clock("https://cdn.example/asset/9f2c", SourceConfig(live=True)))
        self.assertTrue(
            _has_media_clock("https://cam.example/traffic.m3u8", SourceConfig(live=False))
        )

    def test_local_and_file_sources_are_untouched(self):
        for uri in ("clip.mp4", "file:///tmp/clip.mp4", "/var/media/clip.mkv"):
            with self.subTest(uri=uri):
                self.assertTrue(_has_media_clock(uri))
        # A webcam is not reached by this classification at all: it normalizes to a device INDEX,
        # and _source_time rejects a non-str source before any of the above runs.
        self.assertIsNone(_source_time(_FakeCap(), _FakeCv2(), "webcam:0"))

    def test_push_transports_have_no_media_clock(self):
        for uri in (
            "rtsp://cam/stream",
            "rtsps://cam/stream",
            "rtp://cam/stream",
            "udp://239.0.0.1:1234",
            "tcp://cam:5000",
            "rtmp://cam/live",
            "rtmps://cam/live",
            "srt://cam:9000",
        ):
            with self.subTest(uri=uri):
                self.assertFalse(_has_media_clock(uri))

    def test_ytdlp_sources_have_no_media_clock(self):
        self.assertFalse(_has_media_clock("ytdlp:https://www.example.com/watch?v=abc"))
        self.assertTrue(_is_live_stream("ytdlp://https://www.example.com/watch?v=abc"))

    def test_source_time_drops_an_advancing_position_on_a_live_manifest(self):
        cap, cv2 = _FakeCap(), _FakeCv2()
        cap.position_ms = 4200.0
        self.assertIsNone(_source_time(cap, cv2, "https://cam.example/hls/traffic.m3u8"))
        # ...and the same advancing position IS provenance for a progressive HTTP file.
        self.assertEqual(_source_time(cap, cv2, "https://cdn.example/video.mp4"), 4.2)

    def test_capture_does_not_stamp_source_time_for_a_live_manifest(self):
        uri = "https://cam.example/hls/traffic.m3u8"
        cap = _FakeCap(reads=[(True, object()), (True, object()), (False, None)], fps=4.0)
        cv2 = _FakeCv2(cap)
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture(uri, SourceConfig(fps=4.0), q, threading.Event())
        items = _drain(q)
        self.assertEqual(len(items), 2)
        for item in items:
            # Observation time is preserved INDEPENDENTLY; only source time is withheld.
            self.assertIsNone(item.source_time)
            self.assertIsNotNone(item.observed_time)
            self.assertEqual(item.source_id, f"video:{uri}")

    def test_capture_still_stamps_source_time_for_a_progressive_http_file(self):
        cap = _FakeCap(reads=[(True, object()), (True, object()), (False, None)], fps=4.0)
        cv2 = _FakeCv2(cap)
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("https://cdn.example/video.mp4", SourceConfig(fps=4.0), q, threading.Event())
        items = _drain(q)
        self.assertEqual([i.source_time for i in items], [0.25, 0.5])


# --- 2. opening the live path --------------------------------------------------------------------
class LiveOpenTests(unittest.TestCase):
    def test_existing_openers_are_unchanged(self):
        cv2 = _FakeCv2()
        _open_capture(cv2, "rtsp://camera/stream", SourceConfig(rtsp_tcp=True))
        _open_capture(cv2, "rtsp://camera/stream", SourceConfig(rtsp_tcp=False))
        _open_capture(cv2, "clip.mp4", SourceConfig())
        _open_capture(cv2, "webcam:1", SourceConfig())
        _open_capture(cv2, "https://cdn.example/video.mp4", SourceConfig())
        self.assertEqual(
            cv2.calls,
            [
                ("rtsp://camera/stream", _FakeCv2.CAP_FFMPEG),
                ("rtsp://camera/stream",),
                ("clip.mp4",),
                (1,),
                ("https://cdn.example/video.mp4",),
            ],
        )

    def test_manifest_uris_go_to_the_ffmpeg_backend_explicitly(self):
        for uri in ("https://cam.example/hls/traffic.m3u8", "https://cam.example/manifest.mpd"):
            with self.subTest(uri=uri):
                cv2 = _FakeCv2()
                _open_capture(cv2, uri, SourceConfig())
                self.assertEqual(cv2.calls, [(uri, _FakeCv2.CAP_FFMPEG)])

    def test_live_edge_options_are_set_during_construction_and_restored_after(self):
        self.assertIsNone(os.environ.get(_FFMPEG_OPTIONS_ENV))
        cv2 = _FakeCv2()
        _open_capture(cv2, "https://cam.example/hls/traffic.m3u8", SourceConfig())
        options = cv2.env_at_open[0]
        self.assertIsNotNone(options)
        self.assertIn("live_start_index;-1", options)  # start at the live edge, not the window head
        self.assertIn("reconnect;1", options)
        self.assertIn("rw_timeout;", options)
        # Process-global variable, so it must not survive the constructor.
        self.assertIsNone(os.environ.get(_FFMPEG_OPTIONS_ENV))

    def test_a_caller_set_ffmpeg_option_string_is_never_overwritten(self):
        cv2 = _FakeCv2()
        with patch.dict(os.environ, {_FFMPEG_OPTIONS_ENV: "rtsp_transport;tcp"}):
            _open_capture(cv2, "https://cam.example/hls/traffic.m3u8", SourceConfig())
            self.assertEqual(cv2.env_at_open[0], "rtsp_transport;tcp")
            self.assertEqual(os.environ[_FFMPEG_OPTIONS_ENV], "rtsp_transport;tcp")

    def test_config_can_replace_the_option_string(self):
        cv2 = _FakeCv2()
        _open_capture(
            cv2,
            "https://cam.example/hls/traffic.m3u8",
            SourceConfig(ffmpeg_options="live_start_index;0"),
        )
        self.assertEqual(cv2.env_at_open[0], "live_start_index;0")

    def test_live_start_index_is_hls_only(self):
        config = SourceConfig()
        self.assertIn("live_start_index", _live_ffmpeg_options("https://c/a.m3u8", config))
        # DASH has no equivalent knob, so claiming one would be noise in the option dict.
        self.assertNotIn("live_start_index", _live_ffmpeg_options("https://c/a.mpd", config))

    def test_decoder_buffer_is_trimmed_to_the_live_edge_best_effort(self):
        cap = _FakeCap()
        cv2 = _FakeCv2(cap)
        _open_capture(cv2, "https://cam.example/hls/traffic.m3u8", SourceConfig())
        self.assertIn((_FakeCv2.CAP_PROP_BUFFERSIZE, 1), cap.sets)

    def test_a_backend_that_rejects_the_buffer_property_still_opens(self):
        cap = _UnsettableCap()
        cv2 = _FakeCv2(cap)
        self.assertIs(_open_capture(cv2, "https://c/a.m3u8", SourceConfig()), cap)

    def test_live_classification_is_threaded_in_not_re_inferred(self):
        # A resolved platform CDN URL usually has no .m3u8 suffix. The caller's verdict must win.
        cv2 = _FakeCv2()
        _open_capture(cv2, "https://cdn.example/videoplayback/9f2c", SourceConfig(), live=True)
        self.assertEqual(cv2.calls, [("https://cdn.example/videoplayback/9f2c", _FakeCv2.CAP_FFMPEG)])


# --- 3. reconnect / backoff ----------------------------------------------------------------------
class ReconnectPolicyTests(unittest.TestCase):
    def test_attempts_remain_matches_the_historical_bound_and_adds_unbounded(self):
        self.assertFalse(_attempts_remain(0, SourceConfig(reconnects=0)))
        self.assertTrue(_attempts_remain(0, SourceConfig(reconnects=1)))
        self.assertFalse(_attempts_remain(1, SourceConfig(reconnects=1)))
        self.assertTrue(_attempts_remain(10_000, SourceConfig(reconnects=-1)))

    def test_backoff_is_constant_unless_a_ceiling_is_set(self):
        config = SourceConfig(reconnect_backoff_s=0.5)
        self.assertEqual([_reconnect_delay(n, config) for n in (1, 2, 3, 9)], [0.5] * 4)

    def test_backoff_doubles_up_to_the_ceiling_and_starts_at_the_base(self):
        config = SourceConfig(reconnect_backoff_s=0.5, reconnect_backoff_max_s=4.0)
        self.assertEqual(
            [_reconnect_delay(n, config) for n in (1, 2, 3, 4, 5, 6)],
            [0.5, 1.0, 2.0, 4.0, 4.0, 4.0],
        )

    def test_a_long_backoff_is_sliced_so_cancellation_is_still_prompt(self):
        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()
        started = time.monotonic()
        _sleep_backoff(30.0, stop)
        self.assertLess(time.monotonic() - started, 5.0)

    def test_a_default_length_backoff_is_a_single_sleep(self):
        stop = threading.Event()
        with patch("time.sleep") as sleep:
            _sleep_backoff(0.5, stop)
        sleep.assert_called_once_with(0.5)

    def test_unbounded_reconnects_keep_opening_new_epochs(self):
        stop = threading.Event()
        caps = [
            _FakeCap(reads=[(True, object()), (False, None)], fps=4.0) for _ in range(4)
        ]
        caps[-1].on_release = stop.set  # end the run deterministically after the 4th epoch
        cv2 = _FakeCv2(caps=caps)
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}), patch("time.sleep", return_value=None):
            _capture("https://cam.example/hls/traffic.m3u8", SourceConfig(fps=4.0, reconnects=-1),
                     q, stop)
        self.assertEqual([item.epoch for item in _drain(q)], [0, 1, 2, 3])

    def test_a_live_open_failure_is_retried_through_the_existing_reconnect_budget(self):
        stop = threading.Event()
        caps = [
            _FakeCap(opened=False),
            _FakeCap(opened=False),
            _FakeCap(reads=[(True, object()), (False, None)], fps=4.0, on_release=stop.set),
        ]
        cv2 = _FakeCv2(caps=caps)
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}), patch("time.sleep", return_value=None):
            _capture("https://cam.example/hls/traffic.m3u8", SourceConfig(fps=4.0, reconnects=5),
                     q, stop)
        items = _drain(q)
        self.assertEqual(len(items), 1)
        # Each failed open burns an epoch, so a consumer sees the reconnect discontinuity honestly.
        self.assertEqual(items[0].epoch, 2)
        self.assertTrue(all(cap.released for cap in caps))

    def test_a_live_open_failure_still_raises_once_the_budget_runs_out(self):
        cv2 = _FakeCv2(caps=[_FakeCap(opened=False), _FakeCap(opened=False)])
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}), patch("time.sleep", return_value=None):
            _capture("https://cam.example/hls/traffic.m3u8", SourceConfig(reconnects=1),
                     q, threading.Event())
        self.assertIsInstance(q.get_nowait(), RuntimeError)

    def test_a_nonlive_open_failure_is_still_fatal_immediately(self):
        # Regression guard: file/webcam/RTSP must not silently acquire retry-on-open.
        for uri in ("clip.mp4", "rtsp://cam/stream", "webcam:0"):
            with self.subTest(uri=uri):
                cv2 = _FakeCv2(caps=[_FakeCap(opened=False)])
                q: "queue.Queue" = queue.Queue()
                with patch.dict(sys.modules, {"cv2": cv2}), patch("time.sleep", return_value=None):
                    _capture(uri, SourceConfig(reconnects=3), q, threading.Event())
                self.assertIsInstance(q.get_nowait(), RuntimeError)
                self.assertEqual(len(cv2.calls), 1)  # exactly one open attempt, as before

    def test_retry_open_can_be_forced_on_a_nonlive_source(self):
        stop = threading.Event()
        caps = [
            _FakeCap(opened=False),
            _FakeCap(reads=[(True, object()), (False, None)], fps=4.0, on_release=stop.set),
        ]
        cv2 = _FakeCv2(caps=caps)
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}), patch("time.sleep", return_value=None):
            _capture("clip.mp4", SourceConfig(fps=4.0, reconnects=2, retry_open=True), q, stop)
        self.assertEqual(len(_drain(q)), 1)


# --- 4. optional yt-dlp resolution ---------------------------------------------------------------
class _FakeYoutubeDL:
    """Stands in for ``yt_dlp.YoutubeDL``; records the options it was built with."""

    instances: "list[_FakeYoutubeDL]" = []
    info: "dict | None" = None
    error: "Exception | None" = None

    def __init__(self, options):
        self.options = options
        _FakeYoutubeDL.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        self.url = url
        if _FakeYoutubeDL.error is not None:
            raise _FakeYoutubeDL.error
        return _FakeYoutubeDL.info


def _fake_ytdlp_module(info=None, error=None):
    import types

    _FakeYoutubeDL.instances = []
    _FakeYoutubeDL.info = info if info is not None else {"url": "https://cdn.example/resolved.m3u8"}
    _FakeYoutubeDL.error = error
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = _FakeYoutubeDL
    return module


class YtDlpResolutionTests(unittest.TestCase):
    def test_a_missing_yt_dlp_degrades_with_an_actionable_error(self):
        with patch.dict(sys.modules, {"yt_dlp": None}):
            with self.assertRaises(SourceConfigurationError) as caught:
                _capture_target("ytdlp:https://www.example.com/watch?v=abc", SourceConfig())
        message = str(caught.exception)
        self.assertIn("yt-dlp", message)
        self.assertIn("pip install yt-dlp", message)
        self.assertIn(".m3u8", message)  # names the workaround, not just the failure

    def test_a_missing_yt_dlp_reaches_the_consumer_instead_of_crashing_the_thread(self):
        cv2 = _FakeCv2()
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2, "yt_dlp": None}):
            _capture(
                "ytdlp:https://www.example.com/watch?v=abc",
                # Unbounded reconnects must NOT turn a permanent misconfiguration into a silent
                # infinite retry — this is the whole point of SourceConfigurationError.
                SourceConfig(reconnects=-1),
                q,
                threading.Event(),
            )
        first = q.get_nowait()
        self.assertIsInstance(first, SourceConfigurationError)
        self.assertIsNone(q.get_nowait())
        self.assertEqual(cv2.calls, [])

    def test_resolution_hands_the_decoder_the_resolved_url(self):
        module = _fake_ytdlp_module()
        with patch.dict(sys.modules, {"yt_dlp": module}):
            target = _capture_target("ytdlp:https://www.example.com/watch?v=abc", SourceConfig())
        self.assertEqual(target, "https://cdn.example/resolved.m3u8")
        built = _FakeYoutubeDL.instances[0]
        self.assertEqual(built.url, "https://www.example.com/watch?v=abc")
        self.assertEqual(built.options["format"], SourceConfig().ytdlp_format)

    def test_a_playlist_result_collapses_to_its_first_entry(self):
        module = _fake_ytdlp_module(
            info={"entries": [None, {"url": "https://cdn.example/entry.m3u8"}]}
        )
        with patch.dict(sys.modules, {"yt_dlp": module}):
            self.assertEqual(
                _capture_target("ytdlp:https://www.example.com/c/live", SourceConfig()),
                "https://cdn.example/entry.m3u8",
            )

    def test_a_result_with_no_url_is_a_clear_error(self):
        module = _fake_ytdlp_module(info={"title": "no media url here"})
        with patch.dict(sys.modules, {"yt_dlp": module}):
            with self.assertRaises(RuntimeError) as caught:
                _capture_target("ytdlp:https://www.example.com/watch?v=abc", SourceConfig())
        self.assertIn("no directly-openable URL", str(caught.exception))

    def test_a_resolved_stream_is_opened_live_and_keeps_the_original_source_id(self):
        uri = "ytdlp:https://www.example.com/watch?v=abc"
        module = _fake_ytdlp_module()
        cap = _FakeCap(reads=[(True, object()), (False, None)], fps=4.0)
        cv2 = _FakeCv2(cap)
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2, "yt_dlp": module}):
            _capture(uri, SourceConfig(fps=4.0), q, threading.Event())
        item = q.get_nowait()
        self.assertEqual(cv2.calls, [("https://cdn.example/resolved.m3u8", _FakeCv2.CAP_FFMPEG)])
        # The ORIGINAL uri identifies the source: a signed CDN URL churns on every re-resolution and
        # would otherwise look like a brand new source to SourceTracker on each reconnect.
        self.assertEqual(item.source_id, f"video:{uri}")
        self.assertIsNone(item.source_time)

    def test_the_url_is_re_resolved_on_every_reconnect(self):
        # A platform CDN URL is signed and expires in hours; a watch measured in days MUST refresh
        # it, so resolution belongs inside the reconnect loop and not before it.
        stop = threading.Event()
        module = _fake_ytdlp_module()
        caps = [_FakeCap(reads=[(True, object()), (False, None)], fps=4.0) for _ in range(3)]
        caps[-1].on_release = stop.set
        cv2 = _FakeCv2(caps=caps)
        q: "queue.Queue" = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2, "yt_dlp": module}), patch("time.sleep"):
            _capture(
                "ytdlp:https://www.example.com/watch?v=abc",
                SourceConfig(fps=4.0, reconnects=-1),
                q,
                stop,
            )
        self.assertEqual(len(_FakeYoutubeDL.instances), 3)
        self.assertEqual([item.epoch for item in _drain(q)], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
