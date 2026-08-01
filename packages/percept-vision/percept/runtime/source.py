"""Capture sources for the runtime worker.

``open_source(uri)`` opens a cv2-readable URI (file / RTSP / HLS-DASH live / ``webcam:0`` /
``webcam://0``) on a **background thread** and yields versioned source envelopes for programmable
Watches:
  * ``gate_frame`` — a small gray :class:`~percept.edge.tier0.Frame` for the motion gate (still-vs-moved
    doesn't need full resolution),
  * ``judge_frame`` — a ``("image/jpeg", bytes)`` pair the VLM judges.

cv2 (``opencv-python-headless``) is the ``[runtime]``/``[rtsp]`` extra — imported lazily, never in the core.
The hardened transport (rtsp-tcp, reconnect/backoff, avfoundation ``-video_size``, ONVIF) is the ``[rtsp]``
follow-up; v1 reuses the deterministic decode pattern from ``eval/count_video.py``.

``yt_dlp`` is likewise lazy and OPTIONAL: it is only imported for a ``ytdlp:`` URI, and its absence
raises a clear install message on the capture thread instead of failing the import of this module.
"""

from __future__ import annotations

import asyncio
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Literal

from percept.runtime.envelope import SourceEnvelope, SourceFormat

GATE_W, GATE_H = 48, 36  # tier-0 downsample (still vs moved; full res not needed for the gate)

# A-4 delivered-fps misreport heuristic. Measure the RAW read rate over a rolling wall-clock window
# and compare it to the source's REPORTED CAP_PROP_FPS. A camera that reports 30 but delivers 15 keeps
# every Nth frame at half the intended rate — the gate/fovea-coast Nyquist guarantee silently halves.
# Only evaluated once ``>= _FPS_MIN_EVAL_S`` of wall time has actually elapsed, so an offline/instant
# fixture (span ~0) can never false-fire.
_FPS_WINDOW_S = 2.0  # rolling window the delivered rate is measured over
_FPS_MIN_EVAL_S = 1.0  # minimum elapsed wall time before we trust a measurement
_FPS_TOL = 0.25  # relative gap (|measured-reported|/reported) that counts as a misreport
_POLL = object()


@dataclass(frozen=True)
class SourceConfig:
    """Runtime capture tuning for webcam/video/RTSP/HLS sources."""

    fps: float = 4.0
    queue_size: int = 64
    #: Reconnect attempts after the decoder ends a read loop.  ``0`` (the default) never reconnects.
    #: A NEGATIVE value means UNBOUNDED — what a multi-day live watch needs, since any finite budget
    #: is a countdown to a silent stop.  Positive values keep their existing "at most N" meaning.
    reconnects: int = 0
    reconnect_backoff_s: float = 0.5
    #: Ceiling for an EXPONENTIAL reconnect backoff.  ``None`` (the default) keeps the historical
    #: CONSTANT ``reconnect_backoff_s`` delay on every attempt.  When set above the base delay the
    #: delay doubles per attempt up to this ceiling, so a stream that is down for hours is not
    #: hammered at 2 Hz for the whole outage.
    reconnect_backoff_max_s: float | None = None
    rtsp_tcp: bool = True
    # Clock selected by the transitional tuple view.  Every envelope carries observed monotonic time
    # independently and carries decoder source time only when the decoder actually provides it.
    timebase: Literal["media", "wall"] = "media"
    # Live streams prefer fresh evidence over stale backlog. Deterministic file replays may instead
    # apply bounded producer backpressure so an evaluator observes every selected decoded frame.
    queue_policy: Literal["latest", "block"] = "latest"
    #: Tri-state override of the LIVE-STREAM classification inferred from the URI.  ``None`` infers
    #: (see :func:`_is_live_stream`); ``True``/``False`` force it.  Needed because an extension-less
    #: HTTP endpoint (``https://host/stream``) is genuinely undecidable from its URI alone.
    live: bool | None = None
    #: Whether a FAILED OPEN is retryable rather than fatal.  ``None`` infers: live streams retry,
    #: everything else keeps the historical "raise immediately" behaviour, so file/webcam/RTSP
    #: callers see no change.  A 72-hour watch must not die on the first DNS blip.
    retry_open: bool | None = None
    #: Extra ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` for the live path, ``key;value`` pairs joined by
    #: ``|``.  ``None`` uses :func:`_live_ffmpeg_options`.  Ignored for non-live sources, and never
    #: applied when the caller already exported the variable themselves.
    ffmpeg_options: str | None = None
    #: ``yt_dlp`` format selector for ``ytdlp:`` URIs.  The default prefers an HLS/m3u8 rendition,
    #: which is the one shape this capture path can actually hold open for days.
    ytdlp_format: str = "best[protocol^=m3u8]/best"


# =================================================================================================
# Source classification.
#
# Two INDEPENDENT questions get asked about a URI, and conflating them is the bug this section
# exists to prevent:
#
#   1. "Does the decoder have a trustworthy MEDIA CLOCK?"  -> _has_media_clock
#   2. "Is this a LIVE STREAM that needs the live-edge/reconnect treatment?" -> _is_live_stream
#
# They are not the same set.  ``rtsp://`` answers NO to (1) but is NOT routed through the new live
# path — its opener branch predates this and stays exactly as it was.  A ``.m3u8`` answers NO to (1)
# AND YES to (2).
#
# NOTE there is a THIRD, deliberately different scheme list in ``percept/sourceprofile.py``
# (``_UNAMBIGUOUS_CAMERA_PREFIXES``).  That one answers "is this so obviously a camera that
# recommending a source profile would be noise?" — and an HLS URL is NOT obviously a camera (it can
# just as easily carry a screen share or a broadcast), so it is correctly absent there.  Do not
# "sync" these three lists; they encode three different questions.
# =================================================================================================

#: Transports that PUSH frames with no seekable media timeline.  ``CAP_PROP_POS_MSEC`` on these is
#: at best an uptime counter, so it is never source-timestamp provenance.
_NO_MEDIA_CLOCK_SCHEMES: tuple[str, ...] = (
    "rtsp://", "rtsps://", "rtp://", "udp://", "tcp://",
    "rtmp://", "rtmps://", "srt://",
)

#: Manifest/playlist extensions.  A manifest is a WINDOW over a stream, not a file: for a live
#: playlist the window slides, segments age out, and the decoder's position resets to the start of
#: whatever window it currently holds.  HLS (``.m3u8``), DASH (``.mpd``), HDS (``.f4m``) and Smooth
#: Streaming (``.ism``/``.isml``) all behave this way.
_MANIFEST_SUFFIXES: tuple[str, ...] = (".m3u8", ".m3u", ".mpd", ".f4m", ".ism", ".isml")

#: Progressive container extensions — a whole file, fetched over HTTP.  These DO have a real media
#: clock: the byte range is the timeline, seeking works, ``POS_MSEC`` is genuine provenance.  Mirrors
#: ``percept.runtime.worker._RECORDED_SUFFIXES`` (not imported: that module imports this one).
_PROGRESSIVE_SUFFIXES: tuple[str, ...] = (
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".m2ts", ".ts", ".gif",
)

#: Route substrings that say "live" out loud when there is no extension to go on.  Checked against
#: the PATH ONLY (never the host), so ``https://live.example.com/archive/clip.mp4`` is not dragged
#: into the live bucket by its hostname.
_LIVE_ROUTE_MARKERS: tuple[str, ...] = ("/live/", "/hls/", "/dash/", "/manifest")

#: Query markers, checked against the query string only.
_LIVE_QUERY_MARKERS: tuple[str, ...] = ("live=1", "live=true", "islive=1", "islive=true")

#: Explicit opt-in prefixes for ``yt_dlp`` URL resolution.  Deliberately EXPLICIT — percept never
#: silently shells a platform URL out to a third-party extractor just because the host looked
#: familiar.
_YTDLP_PREFIXES: tuple[str, ...] = ("ytdlp:", "ytdlp://", "yt-dlp:", "yt-dlp://")

#: OpenCV's documented pass-through to the FFMPEG demuxer.  Process-global, so every write is taken
#: under ``_FFMPEG_OPTIONS_LOCK`` and restored immediately after the ``VideoCapture`` constructor.
_FFMPEG_OPTIONS_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_FFMPEG_OPTIONS_LOCK = threading.Lock()

#: Longest single ``time.sleep`` inside a reconnect backoff.  Delays at or below this sleep in one
#: call (byte-for-byte the historical behaviour for every default backoff); longer delays are sliced
#: so a multi-minute backoff still notices a cancelled consumer promptly.
_BACKOFF_SLICE_S = 1.0


def _uri_route_and_query(uri: str) -> "tuple[str, str]":
    """Split a lowercased URI into ``(path, query)`` with the scheme, host and fragment removed.

    Dropping the authority matters: a marker test that ran over the whole URI would classify
    ``https://live.cdn.example/archive/clip.mp4`` as live purely because of its hostname.
    """
    low = uri.strip().lower().split("#", 1)[0]
    path, _, query = low.partition("?")
    _, sep, after = path.partition("://")
    if sep:
        slash = after.find("/")
        path = after[slash:] if slash >= 0 else ""
    return path, query


def _is_http_live_stream(uri: str) -> bool:
    """Is this HTTP(S) URI a LIVE stream rather than a progressive file fetched over HTTP?

    The rule, in order, and why each step is the way round it is:

    1. **A progressive container extension wins outright.**  ``https://host/live/clip.mp4`` is a
       FILE.  Extension beats route marker, so a file that happens to sit under a ``/live/`` prefix
       keeps its media clock.
    2. **A manifest extension is live.**  ``.m3u8``/``.mpd``/``.f4m``/``.ism`` name a sliding window
       over a stream, not a file.  A VOD playlist is swept up here too — deliberately: the honest
       answer for a manifest is "no trustworthy media clock", and the cost of that is falling back
       to the observation clock, which is the same conservative direction the rest of this module
       already takes.
    3. **Then explicit live markers** in the route or query.
    4. **Everything else is treated as progressive** — today's behaviour, unchanged.  An
       extension-less endpoint is genuinely undecidable from its URI, so it is left alone rather
       than silently stripped of provenance; a caller who knows better passes ``live=True``.
    """
    path, query = _uri_route_and_query(uri)
    if path.endswith(_PROGRESSIVE_SUFFIXES):
        return False
    if path.endswith(_MANIFEST_SUFFIXES):
        return True
    if any(marker in path for marker in _LIVE_ROUTE_MARKERS):
        return True
    return any(marker in query for marker in _LIVE_QUERY_MARKERS)


def _is_ytdlp_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.strip().lower().startswith(_YTDLP_PREFIXES)


def _is_live_stream(uri: str, config: "SourceConfig | None" = None) -> bool:
    """Should ``uri`` take the LIVE path (FFMPEG backend, live-edge policy, retryable open)?

    ``config.live`` overrides outright.  Otherwise: ``ytdlp:`` URIs always, and HTTP(S) URIs that
    :func:`_is_http_live_stream` calls live.  ``rtsp://`` and friends are deliberately NOT here —
    they already have a working opener branch and changing it is out of scope.
    """
    if config is not None and config.live is not None:
        return bool(config.live)
    if not isinstance(uri, str):
        return False
    if _is_ytdlp_uri(uri):
        return True
    if uri.strip().lower().startswith(("http://", "https://")):
        return _is_http_live_stream(uri)
    return False


def _has_media_clock(uri: str, config: "SourceConfig | None" = None) -> bool:
    """Is the decoder's reported position real source-timestamp provenance for ``uri``?"""
    if not isinstance(uri, str):
        return False
    if uri.strip().lower().startswith(_NO_MEDIA_CLOCK_SCHEMES):
        return False
    return not _is_live_stream(uri, config)


def _normalize_uri(uri: str):
    if uri == "webcam":
        return 0
    if uri.startswith("webcam://"):
        return int(uri.removeprefix("webcam://") or 0)
    if uri.startswith("webcam:"):
        return int(uri.split(":", 1)[1] or 0)
    if uri.startswith("screen://"):
        raise NotImplementedError(
            "screen:// is not a capture source: percept 1.0 ships no built-in local screen capture, "
            "and there is no extra that adds one. The supported route is client-pushed frames — "
            "grab the display yourself and pass an (async) iterable of "
            "(t, gate_frame, judge_frame) as source=. See docs/watches/scenario-screenshare.md."
        )
    return uri


def _put_latest(q: "queue.Queue", item, dropped: "list[int] | None" = None) -> None:
    """Put without letting a slow consumer make capture block forever.

    On a full queue the oldest frame is discarded to make room for the latest; ``dropped`` (a 1-elem
    shared counter) is bumped so the worker can surface a ``frames_dropped`` telemetry count.
    """
    try:
        q.put(item, timeout=0.1)
    except queue.Full:
        try:
            q.get_nowait()
            if dropped is not None:
                dropped[0] += 1
        except queue.Empty:
            pass
        q.put(item, timeout=0.1)


def _put_blocking(q: "queue.Queue", item, stop: threading.Event) -> bool:
    """Put into a bounded queue without dropping evidence.

    This policy is for finite, deterministic fixture replay. It deliberately backpressures the
    decoder, but still polls ``stop`` so cancelling a consumer cannot strand the capture thread.
    """
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _put_frame(
    q: "queue.Queue",
    item,
    config: SourceConfig,
    stop: threading.Event,
    dropped: "list[int] | None" = None,
) -> bool:
    if config.queue_policy == "latest":
        _put_latest(q, item, dropped)
        return True
    if config.queue_policy == "block":
        return _put_blocking(q, item, stop)
    raise ValueError(f"unknown source queue policy: {config.queue_policy!r}")


def _put_terminal(q: "queue.Queue", item, stop: threading.Event) -> None:
    """Publish an error/end marker after queued evidence instead of evicting that evidence."""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return
        except queue.Full:
            continue


def _get_polling(q: "queue.Queue", timeout: float = 0.5):
    """Bound one executor-backed queue wait so cancellation cannot strand a worker thread."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return _POLL


class SourceConfigurationError(RuntimeError):
    """A source that is misconfigured, not merely unreachable — retrying can never fix it.

    A missing optional dependency or a malformed URI is permanent, so the reconnect loop re-raises
    this immediately instead of burning a multi-day retry budget on it.  Transient network failures
    raise plain :class:`RuntimeError` and stay retryable.
    """


def _ytdlp_target(uri: str, config: SourceConfig) -> str:
    """Resolve a ``ytdlp:<page url>`` source to a directly-openable media URL.

    ``yt_dlp`` is imported HERE and nowhere else, so it stays an optional extra: a missing install
    surfaces as a clear, actionable ``RuntimeError`` on the capture thread (which the existing
    terminal-marker path hands to the consumer) rather than an ``ImportError`` at module import.
    """
    page = uri.strip()
    for prefix in _YTDLP_PREFIXES:
        if page.lower().startswith(prefix):
            page = page[len(prefix):]
            break
    page = page.lstrip("/") if page.lower().startswith("//") else page
    if not page:
        raise SourceConfigurationError(f"ytdlp: source names no page URL: {uri!r}")
    try:
        import yt_dlp  # lazy + OPTIONAL: never a hard dependency of the runtime
    except ImportError as exc:
        raise SourceConfigurationError(
            f"{uri!r} needs yt-dlp to resolve a platform-hosted stream, and it is not installed. "
            "Install it (`pip install yt-dlp`) or pass the direct .m3u8/.mpd URL instead."
        ) from exc
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "format": config.ytdlp_format,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(page, download=False)
    except Exception as exc:  # noqa: BLE001 - third-party extractor, any failure is the same answer
        raise RuntimeError(f"yt-dlp could not resolve a stream URL for {page!r}: {exc}") from exc
    while isinstance(info, dict) and info.get("entries"):
        entries = [e for e in info["entries"] if e]
        if not entries:
            break
        info = entries[0]
    url = info.get("url") if isinstance(info, dict) else None
    if not url and isinstance(info, dict):
        for candidate in info.get("requested_formats") or ():
            if candidate.get("url"):
                url = candidate["url"]
                break
    if not url:
        raise RuntimeError(
            f"yt-dlp resolved {page!r} but returned no directly-openable URL "
            f"(format selector {config.ytdlp_format!r})"
        )
    return str(url)


def _capture_target(uri: str, config: SourceConfig) -> str:
    """The URI to actually hand the decoder — identity for everything except ``ytdlp:``.

    Called once per connection ATTEMPT, not once per capture, because a platform CDN URL is signed
    and expires (typically in hours).  Re-resolving per epoch is what lets a multi-day watch survive
    that expiry; the envelope's ``source_id`` keeps the ORIGINAL uri so the source identity does not
    churn underneath :class:`~percept.runtime.envelope.SourceTracker`.
    """
    if _is_ytdlp_uri(uri):
        return _ytdlp_target(uri, config)
    return uri


def _live_ffmpeg_options(uri: str, config: SourceConfig) -> str:
    """FFMPEG demuxer/protocol options for the live path, in OpenCV's ``key;value|key;value`` form.

    * ``live_start_index;-1`` is the LIVE-EDGE policy: start at the LAST segment of the playlist
      instead of FFmpeg's default of three segments back from the end, so a watch opens on now and
      not on a stale window.  HLS only — DASH has no equivalent knob.
    * ``reconnect*`` lets FFmpeg paper over brief HTTP hiccups without tearing the capture down.
    * ``rw_timeout``/``timeout`` (microseconds) are the reason a dead socket becomes an EOF this
      module can see: without them a vanished stream blocks inside ``read()`` indefinitely and the
      reconnect loop below never gets a turn.
    """
    if config.ffmpeg_options is not None:
        return config.ffmpeg_options
    timeout_us = int(max(1.0, config.reconnect_backoff_s * 20.0) * 1_000_000)
    options = [
        "reconnect;1",
        "reconnect_streamed;1",
        "reconnect_delay_max;5",
        f"rw_timeout;{timeout_us}",
        f"timeout;{timeout_us}",
    ]
    path, _ = _uri_route_and_query(uri)
    if path.endswith((".m3u8", ".m3u")):
        options.insert(0, "live_start_index;-1")
    return "|".join(options)


def _open_live_capture(cv2, source: str, config: SourceConfig):
    """Open a live stream on the FFMPEG backend, with live-edge options applied around construction.

    ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` is process-global and read by OpenCV inside the constructor,
    so it is set under a lock and restored immediately.  A value the CALLER already exported wins —
    percept does not overwrite an explicit environment choice.
    """
    backend = getattr(cv2, "CAP_FFMPEG", None)

    def _construct():
        return cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)

    if os.environ.get(_FFMPEG_OPTIONS_ENV):
        cap = _construct()
    else:
        with _FFMPEG_OPTIONS_LOCK:
            os.environ[_FFMPEG_OPTIONS_ENV] = _live_ffmpeg_options(source, config)
            try:
                cap = _construct()
            finally:
                os.environ.pop(_FFMPEG_OPTIONS_ENV, None)
    buffer_prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffer_prop is not None:
        try:
            # Best-effort second half of the live-edge policy: keep the decoder's own queue at one
            # frame so a slow consumer reads NOW, not a backlog.  Unsupported on several backends —
            # failure here is not an error, the "latest" queue policy already covers the common case.
            cap.set(buffer_prop, 1)
        except Exception:  # noqa: BLE001 - optional backend property
            pass
    return cap


def _open_capture(cv2, uri: str, config: SourceConfig, *, live: bool | None = None):
    source = _normalize_uri(uri)
    if isinstance(source, str) and source.startswith("rtsp://") and config.rtsp_tcp:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    # ``live`` is threaded in from the ORIGINAL uri: a resolved platform URL frequently carries no
    # ``.m3u8`` suffix at all, so re-inferring from ``source`` here would lose the classification.
    is_live = _is_live_stream(source, config) if live is None else live
    if isinstance(source, str) and is_live:
        return _open_live_capture(cv2, source, config)
    return cv2.VideoCapture(source)


def _source_time(
    cap,
    cv2,
    uri: str,
    *,
    previous: float | None = None,
    after_first: bool = False,
    has_media_clock: bool | None = None,
) -> float | None:
    """Read a decoder-provided media timestamp, or return ``None``.

    Device/network sources do not have a trustworthy media clock.  A reported FPS plus a local frame
    counter is a cadence estimate, not a source timestamp, so this adapter deliberately does not turn
    ``i / fps`` into provenance.  A LIVE HTTP stream is in the same boat as ``rtsp://``: the decoder
    reports a position within the playlist window it currently holds, which resets as segments age
    out — see :func:`_has_media_clock`.  ``has_media_clock`` lets the caller pass that verdict in
    (computed once, from the pre-resolution uri); ``None`` re-derives it from ``uri``.
    """
    source = _normalize_uri(uri)
    if not isinstance(source, str):
        return None
    if not (_has_media_clock(source) if has_media_clock is None else has_media_clock):
        return None
    prop = getattr(cv2, "CAP_PROP_POS_MSEC", None)
    if prop is None:
        return None
    try:
        value = float(cap.get(prop)) / 1000.0
    except Exception:  # noqa: BLE001 - optional backend property
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    # OpenCV backends commonly report a permanent 0.0 when POS_MSEC is unsupported.  Zero is valid
    # for the first decoded frame, but a non-advancing decoder clock is not timestamp provenance.
    if after_first and previous is not None and value <= previous:
        return None
    return value


def _attempts_remain(attempts: int, config: SourceConfig) -> bool:
    """Is another reconnect allowed?  Negative ``reconnects`` means unbounded."""
    return config.reconnects < 0 or attempts < config.reconnects


def _reconnect_delay(attempt: int, config: SourceConfig) -> float:
    """Delay before 1-based reconnect ``attempt``.

    Constant ``reconnect_backoff_s`` unless the caller set a higher ``reconnect_backoff_max_s``, in
    which case the delay doubles per attempt up to that ceiling.  The FIRST retry is always the base
    delay, so a caller who never sets the ceiling sees the historical schedule exactly.
    """
    base = max(0.0, float(config.reconnect_backoff_s))
    ceiling = config.reconnect_backoff_max_s
    if ceiling is None or float(ceiling) <= base:
        return base
    return min(float(ceiling), base * (2.0 ** max(0, attempt - 1)))


def _sleep_backoff(delay: float, stop: threading.Event) -> None:
    """Sleep ``delay``, in slices once it is long enough for cancellation latency to matter.

    Delays at or below ``_BACKOFF_SLICE_S`` — which includes every default — are a single
    ``time.sleep``, identical to what this loop always did.
    """
    if delay <= 0.0:
        return
    if delay <= _BACKOFF_SLICE_S:
        time.sleep(delay)
        return
    deadline = time.monotonic() + delay
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(_BACKOFF_SLICE_S, remaining))


def _video_format(bgr) -> SourceFormat:
    shape = getattr(bgr, "shape", ())
    height = int(shape[0]) if len(shape) >= 2 and int(shape[0]) > 0 else None
    width = int(shape[1]) if len(shape) >= 2 and int(shape[1]) > 0 else None
    return SourceFormat(media_type="image/jpeg", width=width, height=height)


def _capture(
    uri: str,
    config: SourceConfig,
    q: "queue.Queue",
    stop: threading.Event,
    dropped: "list[int] | None" = None,
    *,
    now: "Callable[[], float] | None" = None,
    health: "list | None" = None,
) -> None:
    """Blocking capture loop, run on a daemon thread; pushes frames into ``q`` until stopped/EOF.

    ``now`` is the observation clock stamped on every envelope and used by the delivered-fps measure
    (injectable so tests need no real sleeping; defaults to ``time.monotonic``). ``health`` mirrors the
    ``dropped`` shared-list telemetry pattern: the capture thread appends ``(kind, data)`` events (today
    only ``("fps_misreported", {"reported", "measured"})``) for the worker to surface via ``on_health``."""
    import cv2  # lazy: the [runtime]/[rtsp] extra

    from percept.edge.tier0 import Frame

    if now is None:
        now = time.monotonic
    # Both verdicts are computed ONCE, from the pre-resolution uri, and threaded down. A ytdlp:
    # source resolves to an opaque CDN URL whose shape says nothing about liveness, so re-deriving
    # either of these from the resolved target would silently lose the classification.
    live = _is_live_stream(uri, config)
    has_media_clock = _has_media_clock(uri, config)
    # A failed OPEN is fatal for file/webcam/RTSP exactly as before; only the live path (or an
    # explicit caller opt-in) folds it into the reconnect loop, because a multi-day watch must
    # survive the DNS/CDN blip that happens to land on a reconnect boundary.
    retry_open = live if config.retry_open is None else bool(config.retry_open)
    attempts = 0
    epoch = 0
    try:
        while not stop.is_set():
            cap = None
            try:
                target = _capture_target(uri, config)
                cap = _open_capture(cv2, target, config, live=live)
                opened = bool(cap.isOpened())
            except SourceConfigurationError:
                raise  # permanent: a missing extra or a malformed URI is not a network blip
            except Exception:
                if not (retry_open and _attempts_remain(attempts, config)):
                    raise
                opened = False
            if not opened:
                if cap is not None:
                    cap.release()
                if not (retry_open and _attempts_remain(attempts, config)):
                    raise RuntimeError(f"could not open runtime source: {uri!r}")
                attempts += 1
                epoch += 1
                _sleep_backoff(_reconnect_delay(attempts, config), stop)
                continue
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, round(src_fps / config.fps))
            i = 0
            sequence = 0
            previous_source_time: float | None = None
            read_ts: "list[float]" = []  # wall times of recent raw reads (A-4 delivered-fps window)
            fps_reported = False  # emit the misreport at most once per open
            try:
                while not stop.is_set():
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    # A-4: measure the RAW read rate over a rolling wall-clock window and flag a source
                    # whose delivered fps diverges from its reported CAP_PROP_FPS. Emit-only (does NOT
                    # touch decimation or either envelope clock). Guarded by `health`.
                    if health is not None and not fps_reported:
                        tw = now()
                        read_ts.append(tw)
                        cutoff = tw - _FPS_WINDOW_S
                        while read_ts and read_ts[0] < cutoff:
                            read_ts.pop(0)
                        span = read_ts[-1] - read_ts[0]
                        if span >= _FPS_MIN_EVAL_S and len(read_ts) >= 2:
                            measured = (len(read_ts) - 1) / span
                            if abs(measured - src_fps) > _FPS_TOL * src_fps:
                                health.append(
                                    ("fps_misreported", {"reported": src_fps, "measured": measured})
                                )
                                fps_reported = True
                    if i % step == 0:
                        observed_time = now()
                        source_time = _source_time(
                            cap,
                            cv2,
                            uri,
                            previous=previous_source_time,
                            after_first=sequence > 0,
                            has_media_clock=has_media_clock,
                        )
                        if source_time is not None:
                            previous_source_time = source_time
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        g = cv2.resize(gray, (GATE_W, GATE_H))
                        gate_frame = Frame(
                            w=GATE_W, h=GATE_H, pixels=tuple(int(p) for p in g.flatten())
                        )
                        encoded, buf = cv2.imencode(".jpg", bgr)
                        jpeg = buf.tobytes() if encoded and buf is not None else b""
                        published = _put_frame(
                            q,
                            SourceEnvelope(
                                source_id=f"video:{uri}",
                                modality="video",
                                epoch=epoch,
                                sequence=sequence,
                                observed_time=observed_time,
                                source_time=source_time,
                                format=_video_format(bgr),
                                payload=("image/jpeg", jpeg),
                                signal=gate_frame,
                                legacy_timebase="observed"
                                if config.timebase == "wall"
                                else "source",
                            ),
                            config,
                            stop,
                            dropped,
                        )
                        if not published:
                            break
                        sequence += 1
                    i += 1
            finally:
                cap.release()
            if not _attempts_remain(attempts, config):
                break
            attempts += 1
            epoch += 1
            _sleep_backoff(_reconnect_delay(attempts, config), stop)
    except Exception as exc:
        _put_terminal(q, exc, stop)
    finally:
        _put_terminal(q, None, stop)  # end sentinel never evicts a captured envelope


async def open_source(
    uri: str,
    *,
    fps: float = 4.0,
    maxsize: int = 64,
    reconnects: int = 0,
    reconnect_backoff_s: float = 0.5,
    reconnect_backoff_max_s: float | None = None,
    rtsp_tcp: bool = True,
    queue_policy: "Literal['latest', 'block']" = "latest",
    timebase: "Literal['media', 'wall']" = "media",
    live: bool | None = None,
    retry_open: bool | None = None,
    ffmpeg_options: str | None = None,
    ytdlp_format: str = "best[protocol^=m3u8]/best",
    dropped: "list[int] | None" = None,
    health: "list | None" = None,
) -> AsyncIterator[SourceEnvelope]:
    """Threaded capture: read frames off a daemon thread (so capture never blocks the consumer/event
    loop), yield :class:`~percept.runtime.envelope.SourceEnvelope` items asynchronously. Stops the
    thread on consumer exit.

    The envelope always preserves observation time and source time independently.  ``timebase`` only
    selects the transitional tuple view; source time remains ``None`` when the decoder supplied none.
    ``queue_policy="latest"`` keeps live capture fresh by evicting stale queued frames (the default).
    ``queue_policy="block"`` is a bounded, lossless mode for deterministic finite-file evaluation.
    ``health`` is an optional shared list the capture thread appends A-4 ``fps_misreported`` events to.

    **Long-horizon live streams.**  An ``https://….m3u8`` / ``….mpd`` URI, or a ``ytdlp:<page url>``
    one, takes the live path: the FFMPEG backend explicitly, live-edge start, and a retryable open.
    Pair it with ``reconnects=-1`` (unbounded) and ``reconnect_backoff_max_s`` for a watch measured
    in days.  ``live=`` forces the classification for an endpoint whose URI does not advertise it;
    ``ytdlp_format`` selects which rendition yt-dlp resolves to."""
    if queue_policy not in ("latest", "block"):
        raise ValueError(f"unknown source queue policy: {queue_policy!r}")
    if queue_policy == "block" and maxsize <= 0:
        raise ValueError("queue_policy='block' requires a positive bounded maxsize")
    config = SourceConfig(
        fps=fps,
        queue_size=maxsize,
        reconnects=reconnects,
        reconnect_backoff_s=reconnect_backoff_s,
        reconnect_backoff_max_s=reconnect_backoff_max_s,
        rtsp_tcp=rtsp_tcp,
        queue_policy=queue_policy,
        timebase=timebase,
        live=live,
        retry_open=retry_open,
        ffmpeg_options=ffmpeg_options,
        ytdlp_format=ytdlp_format,
    )
    q: "queue.Queue" = queue.Queue(maxsize=config.queue_size)
    stop = threading.Event()
    thread = threading.Thread(
        target=_capture,
        args=(uri, config, q, stop, dropped),
        kwargs={"health": health},
        daemon=True,
    )
    thread.start()
    try:
        while True:
            item = await asyncio.to_thread(_get_polling, q)
            if item is _POLL:
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
