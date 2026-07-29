"""Stage-0 INGEST validation — SV-0.1 … SV-0.6.

Stage 0 is "ingest + normalize -> timestamped A/V buffers". Everything downstream inherits its
errors, so these checks are deliberately about *accounting*, not about perception quality: a frame
that never arrives must be COUNTED, a clock the decoder never supplied must stay ``None``, and a
mismatched A/V pair must be typed rather than silently fused.

Every check drives the real runtime:

  * ``percept.runtime.source.open_source`` / ``_capture``  — the video capture adapter
  * ``percept.runtime.audio.open_audio_source``            — the audio capture adapter
  * ``percept.runtime.envelope.SourceTracker``             — the ingress continuity ledger
  * ``percept.events.AudioWindow``                         — the ordered audio timeline
  * ``percept.fused_sync._FusedSynchronizer``              — the fused event-time synchronizer
  * ``percept.run``                                        — the public front door

Nothing here re-implements runtime logic; the oracles are computed independently from the
*injected* schedule and compared against what the runtime reports.

Fully offline and deterministic: synthetic media, a scripted ``cv2`` double for the capture-thread
paths, and ``FakeVision`` for the judge. No provider or network call is made.

Run standalone::

    /private/tmp/percept-launch/oss-final-venv-bba572a/bin/python sv0_ingest.py

or import ``run_all()`` from a report generator.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import os
import queue
import struct
import sys
import tempfile
import threading
import time
import types
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
JSON_OUT = RESULTS / "sv0_ingest.json"

# --------------------------------------------------------------------------------------------
# Result plumbing
# --------------------------------------------------------------------------------------------


@dataclass
class Result:
    """One SV check: what it measured, what it observed, and whether it held."""

    id: str
    title: str
    measures: str
    bar: str
    measurements: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "ERROR"
        return "PASS" if not self.failures else "FAIL"

    def check(self, condition: bool, message: str) -> bool:
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "passed": self.status == "PASS",
            "measures": self.measures,
            "pass_bar": self.bar,
            "measurements": self.measurements,
            "failures": self.failures,
            "notes": self.notes,
            "error": self.error,
        }


# --------------------------------------------------------------------------------------------
# Synthetic media
# --------------------------------------------------------------------------------------------

_TMP: list[str] = []


def _tmpdir() -> str:
    d = tempfile.mkdtemp(prefix="sv0-ingest-")
    _TMP.append(d)
    return d


def synth_video(frames: int = 60, fps: int = 30, w: int = 64, h: int = 48) -> str:
    """Write a deterministic mp4 with a single moving block (real decoder, real container)."""
    import cv2
    import numpy as np

    path = os.path.join(_tmpdir(), "synth.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():  # pragma: no cover - environment guard
        raise RuntimeError("cv2.VideoWriter could not open an mp4v writer")
    for i in range(frames):
        img = np.zeros((h, w, 3), np.uint8)
        x = (i * 3) % (w - 8)
        img[10:20, x : x + 8] = 255
        writer.write(img)
    writer.release()
    return path


def synth_wav(samples: int, sample_rate: int = 16_000) -> tuple[str, bytes]:
    """Write a deterministic mono s16le wav; return its path and its exact PCM payload."""
    path = os.path.join(_tmpdir(), "synth.wav")
    pcm = bytearray()
    for n in range(samples):
        pcm += struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * n / sample_rate)))
    blob = bytes(pcm)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(blob)
    return path, blob


# --------------------------------------------------------------------------------------------
# Scripted cv2 double — drives the REAL percept.runtime.source._capture loop
# --------------------------------------------------------------------------------------------


class _Gray:
    def __init__(self, n: int) -> None:
        self.n = n

    def flatten(self):
        return [0] * self.n


class _Buf:
    def tobytes(self) -> bytes:
        return b"\xff\xd8scripted-jpeg\xff\xd9"


class _BGR:
    shape = (48, 64, 3)


class _FakeCv2(types.ModuleType):
    CAP_FFMPEG = 1900
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 0
    COLOR_BGR2GRAY = 6

    def __init__(self, cap: Any) -> None:
        super().__init__("cv2")
        self.cap = cap

    def VideoCapture(self, *args):  # noqa: N802 - mirrors the cv2 API
        return self.cap

    def cvtColor(self, bgr, code):  # noqa: N802
        return bgr

    def resize(self, gray, size):
        return _Gray(size[0] * size[1])

    def imencode(self, ext, bgr):  # noqa: N802
        return True, _Buf()


class _ScriptedCap:
    """A capture whose delivered frame count and POS_MSEC behaviour are fully scripted.

    ``pos_mode``:
      * ``advance`` — a real decoder clock that moves every frame
      * ``stuck``   — a backend that permanently reports 0.0 (POS_MSEC unsupported)
      * ``nan``     — a backend that reports a non-finite value
    """

    def __init__(self, n: int, *, fps: float = 30.0, pos_mode: str = "advance") -> None:
        self.n = n
        self.i = 0
        self.fps = fps
        self.pos_mode = pos_mode
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def get(self, prop):
        if prop == _FakeCv2.CAP_PROP_POS_MSEC:
            if self.pos_mode == "advance":
                return (self.i - 1) * (1000.0 / self.fps)
            if self.pos_mode == "stuck":
                return 0.0
            return float("nan")
        return self.fps

    def read(self):
        if self.i >= self.n:
            return (False, None)
        self.i += 1
        return (True, _BGR())

    def release(self) -> None:
        self.released = True


@contextlib.contextmanager
def scripted_cv2(cap: Any):
    """Install the scripted cv2 for the duration of a ``_capture`` call, then restore the real one."""
    previous = sys.modules.get("cv2")
    sys.modules["cv2"] = _FakeCv2(cap)
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = previous


def drive_capture(
    uri: str,
    *,
    frames: int,
    config: Any,
    pos_mode: str = "advance",
    reported_fps: float = 30.0,
    clock: Callable[[Any], float] | None = None,
    health: list | None = None,
    dropped: list | None = None,
) -> list:
    """Run the real ``_capture`` loop to completion against a scripted cv2 and drain its queue."""
    from percept.runtime.source import _capture

    cap = _ScriptedCap(frames, fps=reported_fps, pos_mode=pos_mode)
    q: queue.Queue = queue.Queue(maxsize=config.queue_size)
    stop = threading.Event()
    if clock is None:
        # A simulated wall clock driven by the number of RAW reads: one read == one frame period.
        def now() -> float:
            return cap.i / reported_fps
    else:
        def now() -> float:
            return clock(cap)

    with scripted_cv2(cap):
        _capture(uri, config, q, stop, dropped, now=now, health=health)
    out = []
    while True:
        item = q.get_nowait()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        out.append(item)
    return out


# --------------------------------------------------------------------------------------------
# Scripted percept.run driver
# --------------------------------------------------------------------------------------------


def _core():
    from percept import Goal, Percept
    from percept.adapters import Verdict
    from percept.backends.fake import FakeVision

    vision = FakeVision(
        script={
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="scripted: person"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="scripted: empty"),
        }
    )
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone")
    )
    return core


def drive_run(stream: list, **kwargs) -> dict[str, Any]:
    """Push a scripted SourceEnvelope stream through the public ``percept.run`` front door."""
    from percept.runtime import RuntimeStats, run

    stats = RuntimeStats()
    health: list = []
    frames: list = []

    async def go():
        return [
            inc
            async for inc in run(
                _core(),
                source=stream,
                stats=stats,
                on_health=health.append,
                on_frame=lambda t, f: frames.append(t),
                **kwargs,
            )
        ]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # the scripted-backend banner is not our signal
        incidents = asyncio.run(go())
    return {
        "incidents": incidents,
        "stats": stats,
        "health": health,
        "frame_times": frames,
    }


def video_envelope(seq: int, t: float, *, epoch: int = 0, moved: bool = False, token: str = "clear"):
    from percept.edge.tier0 import gray_frame
    from percept.runtime.envelope import SourceEnvelope, SourceFormat

    return SourceEnvelope(
        source_id="video:synthetic",
        modality="video",
        epoch=epoch,
        sequence=seq,
        observed_time=1000.0 + t,
        source_time=t,
        format=SourceFormat(media_type="application/x-percept-opaque"),
        payload=token,
        signal=gray_frame(8, 8, 255 if moved else 0),
        legacy_timebase="source",
    )


# --------------------------------------------------------------------------------------------
# SV-0.1 — Live video continuity
# --------------------------------------------------------------------------------------------


def sv0_1() -> Result:
    r = Result(
        id="SV-0.1",
        title="Live video continuity",
        measures=(
            "Drives a synthetic stream through the real ingest path (file capture + percept.run) "
            "and checks monotonic timestamps, zero reordering, and that every injected sequence "
            "gap is accounted for by a typed discontinuity with an exact missing count."
        ),
        bar="zero unaccounted gaps",
    )
    from percept.runtime.envelope import SourceTracker
    from percept.runtime.source import open_source

    # --- A. real decoder, lossless replay -----------------------------------------------------
    frames = 60
    path = synth_video(frames=frames, fps=30)

    async def capture():
        dropped = [0]
        out = []
        async for env in open_source(
            path, fps=30.0, maxsize=8, queue_policy="block", dropped=dropped
        ):
            out.append(env)
        return out, dropped[0]

    envs, dropped = asyncio.run(capture())
    seqs = [e.sequence for e in envs]
    obs = [e.observed_time for e in envs]
    src = [e.source_time for e in envs]

    r.measurements["captured_frames"] = len(envs)
    r.measurements["expected_frames"] = frames
    r.measurements["capture_dropped"] = dropped
    r.measurements["epochs"] = sorted({e.epoch for e in envs})
    r.measurements["distinct_formats"] = len({e.format for e in envs})

    r.check(len(envs) == frames, f"captured {len(envs)} of {frames} decoded frames")
    r.check(seqs == list(range(len(envs))), "capture sequence numbers are not contiguous from 0")
    non_monotonic_obs = sum(1 for a, b in zip(obs, obs[1:]) if b < a)
    r.measurements["observed_time_regressions"] = non_monotonic_obs
    r.check(non_monotonic_obs == 0, f"{non_monotonic_obs} observed_time regressions")
    src_known = [s for s in src if s is not None]
    r.measurements["source_time_known"] = len(src_known)
    non_increasing_src = sum(1 for a, b in zip(src_known, src_known[1:]) if b <= a)
    r.measurements["source_time_regressions"] = non_increasing_src
    r.check(
        len(src_known) == len(envs), "the decoder supplied media time for only some captured frames"
    )
    r.check(non_increasing_src == 0, f"{non_increasing_src} source_time regressions")
    r.check(len({e.format for e in envs}) == 1, "capture format changed mid-file")
    bad_payload = [
        e.sequence
        for e in envs
        if not (
            isinstance(e.payload, tuple)
            and e.payload[0] == "image/jpeg"
            and e.payload[1].startswith(b"\xff\xd8")
            and e.payload[1].endswith(b"\xff\xd9")
        )
    ]
    r.check(not bad_payload, f"{len(bad_payload)} captured frames carry a malformed JPEG payload")

    # --- B. the ingress ledger agrees the capture was continuous -----------------------------
    tracker = SourceTracker()
    for env in envs:
        tracker.inspect(env, require_source_time=True)
    cov = tracker.coverage.as_dict()
    r.measurements["tracker_coverage_clean_capture"] = cov
    r.check(cov["observed"] == frames, f"tracker observed {cov['observed']} of {frames}")
    r.check(cov["excluded"] == 0, f"tracker excluded {cov['excluded']} continuous frames")
    r.check(cov["discontinuities"] == 0, "tracker reported a discontinuity on a continuous file")
    r.check(cov["missing_sequences"] == 0, "tracker invented missing sequences on a clean capture")

    # --- C. injected gap + reconnect through the public run() path ----------------------------
    injected_missing = 3  # sequences 5, 6, 7 never arrive
    stream = []
    t = 0.0
    for s in list(range(5)) + [8, 9, 10]:
        stream.append(
            video_envelope(s, t, moved=(s % 2 == 1), token="intruder" if s % 2 else "clear")
        )
        t += 0.25
    for s in range(3):  # a reconnect: new epoch, sequence restarts at 0
        stream.append(
            video_envelope(
                s, t, epoch=1, moved=(s % 2 == 1), token="intruder" if s % 2 else "clear"
            )
        )
        t += 0.25

    out = drive_run(stream)
    stats = out["stats"]
    kinds = [h.kind for h in out["health"]]
    gap_events = [h for h in out["health"] if h.kind == "source_gap"]
    reconnect_events = [h for h in out["health"] if h.kind == "source_reconnect"]
    reported_missing = stats.source_missing_sequences
    r.measurements["injected_missing_sequences"] = injected_missing
    r.measurements["reported_missing_sequences"] = reported_missing
    r.measurements["unaccounted_gap_sequences"] = injected_missing - reported_missing
    r.measurements["source_observed"] = stats.source_observed
    r.measurements["source_excluded"] = stats.source_excluded
    r.measurements["source_discontinuities"] = stats.source_discontinuities
    r.measurements["typed_discontinuity_events"] = sorted(
        k for k in kinds if k.startswith("source_")
    )
    r.check(
        reported_missing == injected_missing,
        f"missing-sequence accounting is {reported_missing}, injected {injected_missing}",
    )
    r.check(
        stats.source_discontinuities == 2,
        f"{stats.source_discontinuities} typed discontinuities for 1 gap + 1 reconnect",
    )
    r.check(len(gap_events) == 1, "the injected gap did not emit exactly one source_gap event")
    r.check(
        bool(gap_events) and gap_events[0].data.get("missing_sequences") == injected_missing,
        "source_gap did not carry the exact missing-sequence count",
    )
    r.check(
        len(reconnect_events) == 1,
        "the injected reconnect did not emit exactly one source_reconnect event",
    )
    r.check(stats.source_excluded == 0, f"{stats.source_excluded} valid envelopes were excluded")
    r.check(
        stats.source_observed == len(stream),
        f"{stats.source_observed} of {len(stream)} delivered envelopes were observed",
    )
    ft = out["frame_times"]
    delivered_regressions = sum(1 for a, b in zip(ft, ft[1:]) if b <= a)
    r.measurements["delivered_frame_time_regressions"] = delivered_regressions
    r.check(delivered_regressions == 0, "frames were delivered downstream out of order")

    # --- D. an out-of-order arrival must be excluded, never silently accepted ------------------
    reorder_stream = [video_envelope(s, s * 0.25) for s in (0, 1, 2, 3)]
    reorder_stream.append(video_envelope(3, 0.75))  # an exact redelivery of the last sequence
    reorder_stream.append(video_envelope(2, 1.0))  # a stale sequence with a fresh timestamp
    reorder_stream.append(video_envelope(1, 1.1))  # an older stale sequence
    reorder_stream.append(video_envelope(4, 1.25))
    out2 = drive_run(reorder_stream)
    s2 = out2["stats"]
    kinds2 = sorted({h.kind for h in out2["health"] if h.kind.startswith("source_")})
    r.measurements["reorder_probe"] = {
        "delivered": len(reorder_stream),
        "observed": s2.source_observed,
        "excluded": s2.source_excluded,
        "missing_sequences": s2.source_missing_sequences,
        "events": kinds2,
        "frame_times": out2["frame_times"],
    }
    r.check(
        s2.source_excluded == 3,
        f"{s2.source_excluded} of the 3 duplicate/reordered arrivals were excluded",
    )
    r.check(
        s2.source_observed == 5,
        f"{s2.source_observed} of the 5 in-order arrivals were observed",
    )
    r.check(
        "source_duplicate" in kinds2 and "source_reorder" in kinds2,
        f"duplicate/reorder were not both typed: {kinds2}",
    )
    r.check(
        s2.source_missing_sequences == 0,
        "a rejected out-of-order arrival was mis-booked as a missing sequence",
    )
    ft2 = out2["frame_times"]
    r.check(
        all(b > a for a, b in zip(ft2, ft2[1:])),
        "an out-of-order envelope reached the downstream frame sink",
    )
    r.notes.append(
        "SourceTracker keeps only the LAST accepted (epoch, sequence) per stream, so only an exact "
        "redelivery of that sequence is typed source_duplicate; an earlier sequence arriving late "
        "is typed source_reorder (percept/runtime/envelope.py:459-462). Both are excluded without "
        "advancing state and neither is booked as a missing sequence, so the accounting is exact — "
        "but a consumer that wants a true duplicate/late split cannot get it from the event name."
    )
    return r


# --------------------------------------------------------------------------------------------
# SV-0.2 — Backpressure policy
# --------------------------------------------------------------------------------------------


def sv0_2() -> Result:
    r = Result(
        id="SV-0.2",
        title="Backpressure policy",
        measures=(
            "Stalls the consumer and checks both source queue policies: 'latest' must evict the "
            "OLDEST and count every eviction while memory stays bounded; 'block' on a finite "
            "fixture must backpressure losslessly."
        ),
        bar="fixture ('block') path drops exactly zero; live ('latest') path bounded with every drop counted",
    )
    from percept.runtime.source import SourceConfig, _capture, open_source

    # --- A. queue_policy="latest": drop-oldest, fully counted, bounded ------------------------
    produced = 40
    maxsize = 4
    cap = _ScriptedCap(produced, fps=30.0, pos_mode="advance")
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    stop = threading.Event()
    dropped = [0]
    cfg = SourceConfig(fps=30.0, queue_size=maxsize, queue_policy="latest")
    peak_qsize = [0]

    def now() -> float:
        return cap.i / 30.0

    with scripted_cv2(cap):
        started = time.perf_counter()
        thread = threading.Thread(
            target=lambda: _capture("file.mp4", cfg, q, stop, dropped, now=now), daemon=True
        )
        thread.start()
        # The consumer is deliberately stalled: nothing is read until the producer has finished
        # everything it can publish without blocking.
        deadline = time.time() + 30.0
        while dropped[0] < produced - maxsize and time.time() < deadline:
            peak_qsize[0] = max(peak_qsize[0], q.qsize())
            time.sleep(0.002)
        saturated_elapsed = time.perf_counter() - started
        peak_qsize[0] = max(peak_qsize[0], q.qsize())
        delivered = []
        while True:
            item = q.get(timeout=5)
            if item is None:
                break
            delivered.append(item)
        stop.set()
        thread.join(timeout=5)

    delivered_seqs = [e.sequence for e in delivered]
    per_eviction_ms = 1000.0 * saturated_elapsed / max(1, dropped[0])
    r.measurements["latest"] = {
        "produced": produced,
        "queue_maxsize": maxsize,
        "dropped_counter": dropped[0],
        "delivered": len(delivered),
        "delivered_sequences": delivered_seqs,
        "dropped_plus_delivered": dropped[0] + len(delivered),
        "peak_queue_size": peak_qsize[0],
        "saturated_wall_s": round(saturated_elapsed, 3),
        "ms_per_eviction": round(per_eviction_ms, 1),
        "effective_capture_fps_under_saturation": round(produced / saturated_elapsed, 1),
    }
    r.check(
        dropped[0] + len(delivered) == produced,
        f"latest: {dropped[0]} dropped + {len(delivered)} delivered != {produced} produced "
        "(a drop went uncounted)",
    )
    r.check(
        delivered_seqs == list(range(produced - maxsize, produced)),
        f"latest: retained {delivered_seqs}, expected the NEWEST {maxsize} "
        f"({list(range(produced - maxsize, produced))}) — oldest must be evicted",
    )
    r.check(
        peak_qsize[0] <= maxsize,
        f"latest: queue grew to {peak_qsize[0]} beyond its {maxsize} bound",
    )
    r.check(dropped[0] > 0, "latest: a stalled consumer produced no drops at all (test is inert)")

    # DEFECT probe (measured, not part of the stated bar): _put_latest evicts only after the put
    # has blocked for its full 0.1 s timeout, so eviction costs the capture thread ~100 ms/frame.
    r.notes.append(
        f"MEASURED: under a saturated consumer the 'latest' policy cost "
        f"{per_eviction_ms:.0f} ms of CAPTURE-THREAD stall per eviction "
        f"({produced} frames took {saturated_elapsed:.2f} s -> "
        f"{produced / saturated_elapsed:.1f} effective fps). percept/runtime/source.py:77 puts with "
        "timeout=0.1 and only evicts on the resulting queue.Full, so the drop trigger is 'full for "
        "100 ms', not 'full'. Memory stays bounded and every queue-level drop IS counted, so the "
        "SV-0.2 bar holds — but on a live 30 fps camera this throttles cap.read() to ~10 Hz under "
        "backpressure, and frames the driver discards while the capture thread is parked are "
        "invisible to the `dropped` counter."
    )

    # --- B. queue_policy="block" vs "latest" on the SAME fixture and the SAME stalled consumer -
    # The 'latest' arm is the anti-vacuity control: if it does not drop, the fixture path is not
    # actually under backpressure and 'block' losing nothing would prove nothing.
    frames = 20
    stall_s = 0.15  # deliberately longer than _put_latest's 0.1 s put timeout
    path = synth_video(frames=frames, fps=30)

    async def stalled_replay(policy: str):
        counter = [0]
        got = []
        async for env in open_source(
            path, fps=30.0, maxsize=1, queue_policy=policy, dropped=counter
        ):
            got.append(env)
            await asyncio.sleep(stall_s)
        return got, counter[0]

    got_latest, dropped_latest = asyncio.run(stalled_replay("latest"))
    got_block, dropped_block = asyncio.run(stalled_replay("block"))
    latest_seqs = [e.sequence for e in got_latest]
    block_seqs = [e.sequence for e in got_block]
    r.measurements["fixture_differential"] = {
        "decoded_frames": frames,
        "queue_maxsize": 1,
        "consumer_stall_s": stall_s,
        "latest": {
            "delivered": len(got_latest),
            "dropped_counter": dropped_latest,
            "sequences": latest_seqs,
            "accounted": len(got_latest) + dropped_latest,
        },
        "block": {
            "delivered": len(got_block),
            "dropped_counter": dropped_block,
            "contiguous": block_seqs == list(range(len(got_block))),
        },
    }
    r.check(
        dropped_latest > 0,
        "control: 'latest' lost nothing on the stalled fixture, so the 'block' result is vacuous",
    )
    r.check(
        len(got_latest) + dropped_latest == frames,
        f"latest fixture: {len(got_latest)} delivered + {dropped_latest} dropped != {frames} decoded "
        "(a drop went uncounted)",
    )
    r.check(
        dropped_block == 0, f"block: {dropped_block} frames dropped on a finite fixture (must be 0)"
    )
    r.check(
        len(got_block) == frames,
        f"block: delivered {len(got_block)} of {frames} decoded frames on a stalled consumer",
    )
    r.check(block_seqs == list(range(len(got_block))), "block: delivered sequences are not contiguous")

    # --- B2. the corollary of the 0.1 s put timeout: 'latest' is INERT below it ----------------
    # A consumer slower than the producer but faster than 100 ms/frame never triggers an eviction,
    # so the "prefer fresh evidence over stale backlog" policy delivers the whole stale backlog
    # with zero drops recorded. Measured, not asserted — the stated SV-0.2 bar does not cover it.
    sub_timeout_stall = 0.05

    async def sub_timeout_replay():
        counter = [0]
        got = []
        started = time.perf_counter()
        async for envx in open_source(
            path, fps=30.0, maxsize=2, queue_policy="latest", dropped=counter
        ):
            got.append(envx.sequence)
            await asyncio.sleep(sub_timeout_stall)
        return got, counter[0], time.perf_counter() - started

    sub_seqs, sub_dropped, sub_elapsed = asyncio.run(sub_timeout_replay())
    r.measurements["latest_inert_below_put_timeout"] = {
        "decoded_frames": frames,
        "queue_maxsize": 2,
        "consumer_stall_s": sub_timeout_stall,
        "put_timeout_s": 0.1,
        "delivered": len(sub_seqs),
        "dropped_counter": sub_dropped,
        "contiguous_backlog": sub_seqs == list(range(len(sub_seqs))),
        "wall_s": round(sub_elapsed, 2),
    }
    if sub_dropped == 0 and len(sub_seqs) == frames:
        r.notes.append(
            f"MEASURED (corollary): with a {sub_timeout_stall * 1000:.0f} ms/frame consumer — slower "
            f"than the producer but faster than the 100 ms put timeout — queue_policy='latest' "
            f"evicted NOTHING and delivered the entire {frames}-frame backlog in arrival order over "
            f"{sub_elapsed:.2f} s. Below the timeout the freshness policy is inert: the consumer "
            "receives the oldest queued frame, which is the opposite of the documented "
            "'live streams prefer fresh evidence over stale backlog' (percept/runtime/source.py:51)."
        )

    # --- C. the policy surface is validated, not silently defaulted ---------------------------
    policy_errors = {}
    for bad in ("drop", "", None):
        try:
            asyncio.run(_first(open_source(path, queue_policy=bad)))
        except ValueError as exc:
            policy_errors[repr(bad)] = str(exc)
        except Exception as exc:  # noqa: BLE001
            policy_errors[repr(bad)] = f"{type(exc).__name__}: {exc}"
    r.measurements["unknown_policy_rejected"] = policy_errors
    r.check(
        len(policy_errors) == 3,
        f"an unknown queue policy was accepted instead of rejected: {policy_errors}",
    )

    # --- D. API-surface observation (recorded, not a pass/fail bar) ----------------------------
    import inspect as _inspect

    from percept.runtime import worker as _worker

    run_params = set(_inspect.signature(_worker._run_worker).parameters)
    exposes_policy = any("queue_policy" in p for p in run_params)
    r.measurements["public_run_exposes_queue_policy"] = exposes_policy
    if not exposes_policy:
        r.notes.append(
            "percept.run()/_run_worker exposes source_queue_size but NOT queue_policy, so the "
            "lossless 'block' mode is unreachable from the public front door — a finite-file "
            "replay through percept.run() always uses drop-oldest 'latest'. "
            "(percept/runtime/worker.py:4301 builds source_kwargs without queue_policy; "
            "percept/runtime/source.py:302 defaults it to 'latest'.) Drops are still counted in "
            "RuntimeStats.frames_dropped, so this is lossy-but-honest, not silent."
        )
    return r


async def _first(agen):
    """Pull one item from an async generator (used only to trigger its argument validation)."""
    async for item in agen:
        return item
    return None


# --------------------------------------------------------------------------------------------
# SV-0.3 — Audio unit ordering
# --------------------------------------------------------------------------------------------


def sv0_3() -> Result:
    r = Result(
        id="SV-0.3",
        title="Audio unit ordering",
        measures=(
            "Streams a synthetic wav through the real audio capture adapter, then normalizes and "
            "pushes every unit into a real AudioWindow. Checks contiguity, ordering, stamping, "
            "byte-exact PCM normalization, and total sample conservation."
        ),
        bar="zero reordering, zero unaccounted sample loss",
    )
    from percept.audio_input import AudioInputCapability, canonicalize_audio_frame
    from percept.events import AudioWindow
    from percept.runtime.audio import (
        normalize_audio_envelope,
        open_audio_source,
        pcm_energy,
    )

    sample_rate = 16_000
    # Deliberately NOT a whole number of chunks, so a truncated tail would show up as loss.
    total_samples = sample_rate * 2 + 123
    path, source_pcm = synth_wav(total_samples, sample_rate=sample_rate)

    async def capture():
        out = []
        async for env in open_audio_source(path, chunk_s=0.25, sample_rate=sample_rate):
            out.append(env)
        return out

    envs = asyncio.run(capture())
    seqs = [e.sequence for e in envs]
    frames = [e.payload for e in envs]

    r.measurements["source_samples"] = total_samples
    r.measurements["units"] = len(envs)
    r.measurements["sequences_contiguous"] = seqs == list(range(len(envs)))
    r.check(bool(envs), "the audio adapter produced no units at all")
    r.check(seqs == list(range(len(envs))), f"audio sequences are not contiguous: {seqs}")

    # ordering + contiguity on the media clock
    reorderings = sum(1 for a, b in zip(frames, frames[1:]) if b.t_start < a.t_start)
    gaps = [
        (i, frames[i].t_start - frames[i - 1].t_end)
        for i in range(1, len(frames))
        if abs(frames[i].t_start - frames[i - 1].t_end) > 1e-9
    ]
    r.measurements["interval_reorderings"] = reorderings
    r.measurements["interval_discontinuities"] = gaps
    r.check(reorderings == 0, f"{reorderings} audio units arrived out of media order")
    r.check(not gaps, f"audio unit intervals are not contiguous: {gaps[:4]}")

    # stamping: source_time is the emitted sample position, never a device-rate estimate
    stamp_errors = [
        (e.sequence, e.source_time, e.payload.t_end)
        for e in envs
        if e.source_time is None or abs(e.source_time - e.payload.t_end) > 1e-12
    ]
    r.measurements["stamp_errors"] = stamp_errors
    r.check(not stamp_errors, f"{len(stamp_errors)} units are stamped off their PCM sample position")
    cumulative_ok = True
    cursor = 0
    for f in frames:
        n = len(f.pcm) // 2
        if abs(f.t_start - cursor / sample_rate) > 1e-12 or abs(
            f.t_end - (cursor + n) / sample_rate
        ) > 1e-12:
            cumulative_ok = False
            break
        cursor += n
    r.measurements["intervals_match_sample_positions"] = cumulative_ok
    r.check(cumulative_ok, "unit intervals do not match their cumulative PCM sample positions")

    # sample conservation + byte-exact normalization for the declared format
    delivered = b"".join(f.pcm for f in frames)
    r.measurements["delivered_samples"] = len(delivered) // 2
    r.measurements["unaccounted_samples"] = total_samples - len(delivered) // 2
    r.measurements["pcm_byte_identical"] = delivered == source_pcm
    r.check(
        len(delivered) // 2 == total_samples,
        f"{total_samples - len(delivered) // 2} samples unaccounted for across the unit stream",
    )
    r.check(delivered == source_pcm, "captured PCM is not byte-identical to the source wav payload")
    formats = {(f.sample_rate, f.channels, f.sample_format) for f in frames}
    r.measurements["declared_format"] = sorted(str(f) for f in formats)
    r.check(
        formats == {(sample_rate, 1, "s16le")},
        f"the declared PCM format was not preserved: {formats}",
    )

    # energy signal is the documented normalized mean-square, in range
    energies = [e.signal for e in envs]
    bad_energy = [
        (e.sequence, e.signal)
        for e in envs
        if e.signal is None
        or not (0.0 <= e.signal <= 1.0)
        or abs(e.signal - pcm_energy(e.payload.pcm)) > 1e-12
    ]
    r.measurements["energy_range"] = [min(energies), max(energies)]
    r.check(not bad_energy, f"{len(bad_energy)} units carry an energy outside the pcm_energy contract")

    # the compiled ingress boundary is lossless for the declared s16le format
    capability = AudioInputCapability(modes=frozenset({"fused"}))
    canon_mismatch = []
    for f in frames:
        out = canonicalize_audio_frame(f, t_start=f.t_start, t_end=f.t_end, capability=capability)
        if out.pcm != f.pcm or out.sample_format != "s16le" or out.sample_rate != f.sample_rate:
            canon_mismatch.append(f.t_start)
    r.measurements["canonicalization_mismatches"] = len(canon_mismatch)
    r.check(not canon_mismatch, f"{len(canon_mismatch)} frames were altered by canonicalization")

    # the ordered timeline accepts every unit with no reported discontinuity
    window = AudioWindow(seconds=600.0)
    pushes = []
    for env in envs:
        unit = normalize_audio_envelope(env, event_t=env.source_time)
        pushes.append(window.push_unit(unit))
    rejected = [p.as_dict() for p in pushes if not p.accepted]
    discontinuous = [p.as_dict() for p in pushes if p.discontinuous]
    r.measurements["window_pushes"] = len(pushes)
    r.measurements["window_rejected"] = len(rejected)
    r.measurements["window_discontinuous"] = len(discontinuous)
    r.check(not rejected, f"AudioWindow rejected {len(rejected)} contiguous units: {rejected[:2]}")
    r.check(
        not discontinuous,
        f"AudioWindow reported {len(discontinuous)} discontinuities on a contiguous stream: "
        f"{discontinuous[:2]}",
    )

    # control: the timeline must DETECT injected reorder/duplicate rather than flatten them
    probe = AudioWindow(seconds=600.0)
    units = [normalize_audio_envelope(e, event_t=e.source_time) for e in envs[:4]]
    probe.push_unit(units[0])
    probe.push_unit(units[2])
    late = probe.push_unit(units[1])
    dup = probe.push_unit(units[2])
    r.measurements["reorder_control"] = {"late": late.as_dict(), "duplicate": dup.as_dict()}
    r.check(late.reordered, "an out-of-order audio unit was accepted without a reorder report")
    r.check(dup.duplicate and not dup.accepted, "a duplicate audio unit was silently re-accepted")
    return r


# --------------------------------------------------------------------------------------------
# SV-0.4 — Fused A/V skew
# --------------------------------------------------------------------------------------------

_FUSED_SR = 16_000
_FUSED_APP = "sv0-fused"
_FUSED_REV = 0
_FUSED_AUTHORITY = "media:sv0-synthetic"
_FUSED_CONFIDENCE = 1.0


def _fused_video_unit(sync, seq: int, t: float):
    from percept.fused_sync import _AcceptedMediaUnit

    # Field-for-field the shape percept/runtime/worker.py:5648 builds for an accepted video frame.
    return _AcceptedMediaUnit(
        modality="video",
        payload=("image/jpeg", b"\xff\xd8sv0\xff\xd9"),
        source_id="video:sv0",
        source_epoch=0,
        source_sequence=seq,
        generation=sync.generation,
        watch_application=_FUSED_APP,
        watch_revision=_FUSED_REV,
        event_start=t,
        event_end=t,
        observed_start=t,
        observed_end=t,
        clock_domain="source",
        clock_authority=_FUSED_AUTHORITY,
        clock_confidence=_FUSED_CONFIDENCE,
        provenance={"adapter": "runtime.video"},
        source_watermark=t,
    )


def _fused_audio_unit(sync, seq: int, t0: float, t1: float):
    from percept.contracts import AudioFrame
    from percept.events import AudioUnit
    from percept.fused_sync import _AcceptedMediaUnit

    n = int(round((t1 - t0) * _FUSED_SR))
    pcm = b"".join(struct.pack("<h", (i * 37) % 20000 - 10000) for i in range(n))
    frame = AudioFrame(
        pcm=pcm, sample_rate=_FUSED_SR, channels=1, sample_format="s16le", t_start=t0, t_end=t1
    )
    unit = AudioUnit(
        seq=seq,
        t_start=t0,
        t_end=t1,
        chunk=frame,
        event_t=t1,
        energy=0.1,
        sample_rate=_FUSED_SR,
        channels=1,
        sample_format="s16le",
        source_id="audio:sv0",
        source_epoch=0,
        source_sequence=seq,
        observed_time=t1,
    )
    # Field-for-field the shape percept/runtime/worker.py:4649 builds for an accepted audio unit.
    return _AcceptedMediaUnit(
        modality="audio",
        payload=unit,
        source_id="audio:sv0",
        source_epoch=0,
        source_sequence=seq,
        generation=sync.generation,
        watch_application=_FUSED_APP,
        watch_revision=_FUSED_REV,
        event_start=t0,
        event_end=t1,
        observed_start=t1,
        observed_end=t1,
        clock_domain="source",
        clock_authority=_FUSED_AUTHORITY,
        clock_confidence=_FUSED_CONFIDENCE,
        provenance={"adapter": "runtime.audio"},
        source_watermark=t1,
    )


def _drive_fused(skew: float, *, max_skew: float, window_s: float, anchors: int = 8, cadence=0.25):
    """Feed a skewed A/V pair through the synchronizer exactly as the worker drains it."""
    from percept.fused_sync import (
        _FusedCoverageUncertainty,
        _FusedSyncPolicy,
        _FusedSynchronizer,
    )

    policy = _FusedSyncPolicy(
        window_s=window_s,
        allowed_lateness_s=0.0,
        max_skew_s=max_skew,
        min_clock_confidence=1.0,
    )
    sync = _FusedSynchronizer(policy, watch_application=_FUSED_APP, watch_revision=_FUSED_REV)
    emitted = []
    audio_intervals = []
    rejected_audio = 0
    for k in range(anchors):
        t = k * cadence
        v = _fused_video_unit(sync, k, t)
        if not sync.accept(v):
            raise AssertionError(f"synchronizer rejected video anchor {k}")
        if sync.generation_start_event_time is None:
            sync.set_generation_start(t)
        a0, a1 = k * cadence - skew, (k + 1) * cadence - skew
        if not sync.accept(_fused_audio_unit(sync, k, a0, a1)):
            rejected_audio += 1
        else:
            audio_intervals.append((a0, a1))
        # worker order: drain anything the new audio closed, then bundle this anchor.
        emitted.extend(sync.drain_ready())
        bundle = sync.bundle_for(v)
        if not any(
            reason is _FusedCoverageUncertainty.WATERMARK_PENDING
            for reason in bundle.coverage_reasons
        ):
            emitted.append(bundle)
    emitted.extend(sync.drain_ready())
    emitted.extend(sync.close())
    return sync, emitted, audio_intervals, rejected_audio


def _truth_skew(anchor_t: float, intervals: list[tuple[float, float]]) -> float | None:
    """Independent oracle for the runtime's `actual_skew_s` — distance to the nearest audio."""
    if not intervals:
        return None
    distances = []
    for a0, a1 in intervals:
        if a0 <= anchor_t <= a1:
            distances.append(0.0)
        elif a1 < anchor_t:
            distances.append(anchor_t - a1)
        else:
            distances.append(a0 - anchor_t)
    return min(distances)


def sv0_4() -> Result:
    r = Result(
        id="SV-0.4",
        title="Fused A/V skew",
        measures=(
            "Injects +/-50, 100, 250 and 500 ms of relative A/V skew plus a mid-stream "
            "discontinuity, and compares the synchronizer's reported actual_skew_s and typed "
            "coverage uncertainty against an independently computed oracle."
        ),
        bar="skew always classified; no silent pairing beyond declared tolerance",
    )
    from percept.fused_sync import _FusedCoverageUncertainty, _FusedDiscontinuity

    window_s = 1.0
    per_skew = {}
    # Two declared tolerances so the sweep straddles the boundary from both sides: at 250 ms the
    # +/-500 ms cases exceed it, at 100 ms the +/-250 and +/-500 ms cases do.
    sweep = [
        (max_skew, skew)
        for max_skew in (0.25, 0.10)
        for skew in (0.0, 0.05, -0.05, 0.10, -0.10, 0.25, -0.25, 0.50, -0.50)
    ]
    for max_skew, skew in sweep:
        sync, bundles, intervals, rejected = _drive_fused(
            skew, max_skew=max_skew, window_s=window_s
        )
        rows = []
        for b in bundles:
            anchor_t = b.video_event_interval[1] if b.video_event_interval else None
            truth = _truth_skew(anchor_t, intervals) if anchor_t is not None else None
            reasons = [x.value for x in b.coverage_reasons]
            rows.append(
                {
                    "anchor_t": anchor_t,
                    "video_interval": b.video_event_interval,
                    "audio_interval": b.audio_event_interval,
                    "reported_skew_s": b.actual_skew_s,
                    "oracle_skew_s": truth,
                    "coverage_reasons": reasons,
                    "complete": b.complete,
                }
            )
            # 1. the reported skew must equal the true distance to the nearest audio evidence
            if truth is not None:
                r.check(
                    b.actual_skew_s is not None
                    and abs(b.actual_skew_s - truth) <= 1e-9,
                    f"skew={skew:+.2f}: reported actual_skew_s={b.actual_skew_s} != oracle {truth} "
                    f"at anchor {anchor_t}",
                )
                # 2. an exceeded tolerance must be TYPED, never silently paired
                if truth > max_skew + 1e-9:
                    r.check(
                        _FusedCoverageUncertainty.SKEW_EXCEEDED in b.coverage_reasons,
                        f"skew={skew:+.2f}: true skew {truth} > tolerance {max_skew} at anchor "
                        f"{anchor_t} but SKEW_EXCEEDED was not raised (reasons={reasons})",
                    )
            # 3. a bundle claiming completeness must actually be inside tolerance
            if b.complete:
                r.check(
                    b.actual_skew_s is not None and b.actual_skew_s <= max_skew + 1e-9,
                    f"skew={skew:+.2f}: a COMPLETE bundle paired evidence at skew "
                    f"{b.actual_skew_s} beyond the declared {max_skew}",
                )
                r.check(
                    bool(b.video_units) and bool(b.audio_units),
                    f"skew={skew:+.2f}: a COMPLETE bundle is missing a modality",
                )
        per_skew[f"tol={max_skew:.2f} skew={skew:+.2f}"] = {
            "max_skew_s": max_skew,
            "skew_s": skew,
            "bundles": len(bundles),
            "audio_rejected": rejected,
            "classified": sum(1 for row in rows if row["coverage_reasons"] or row["complete"]),
            "typed_uncertainty": sorted(
                {reason for row in rows for reason in row["coverage_reasons"]}
            ),
            "max_reported_skew_s": max(
                (row["reported_skew_s"] for row in rows if row["reported_skew_s"] is not None),
                default=None,
            ),
            "max_oracle_skew_s": max(
                (row["oracle_skew_s"] for row in rows if row["oracle_skew_s"] is not None),
                default=None,
            ),
            "rows": rows,
        }
        r.check(bool(bundles), f"skew={skew:+.2f}: the synchronizer emitted no bundle at all")
    r.measurements["policy"] = {
        "window_s": window_s,
        "max_skew_s_tested": sorted({tol for tol, _ in sweep}),
        "injected_skews_s": sorted({s for _, s in sweep}),
        "cadence_s": 0.25,
    }
    r.measurements["per_skew"] = per_skew

    # Anti-vacuity: the sweep must actually have produced at least one over-tolerance pairing and
    # at least one under-tolerance one, otherwise the classification checks never ran.
    all_rows = [
        (entry["max_skew_s"], row) for entry in per_skew.values() for row in entry["rows"]
    ]
    over = [
        row
        for tol, row in all_rows
        if row["oracle_skew_s"] is not None and row["oracle_skew_s"] > tol + 1e-9
    ]
    under = [
        row
        for tol, row in all_rows
        if row["oracle_skew_s"] is not None and row["oracle_skew_s"] <= tol + 1e-9
    ]
    unclassified = [
        row
        for _tol, row in all_rows
        if row["reported_skew_s"] is None and not row["coverage_reasons"]
    ]
    r.measurements["sweep_totals"] = {
        "bundles": len(all_rows),
        "over_tolerance_pairings": len(over),
        "within_tolerance_pairings": len(under),
        "typed_skew_exceeded": sum(
            1 for _tol, row in all_rows if "skew_exceeded" in row["coverage_reasons"]
        ),
        "complete_bundles": sum(1 for _tol, row in all_rows if row["complete"]),
        "unclassified_bundles": len(unclassified),
    }
    r.check(
        len(over) > 0,
        "the skew sweep never produced an over-tolerance pairing — the SKEW_EXCEEDED branch is untested",
    )
    r.check(
        len(under) > 0,
        "the skew sweep never produced a within-tolerance pairing — the complete branch is untested",
    )
    r.check(
        not unclassified,
        f"{len(unclassified)} bundles carried neither a measured skew nor a typed uncertainty",
    )

    # --- mid-stream discontinuity -------------------------------------------------------------
    disc_max_skew = 0.25
    sync, _, _, _ = _drive_fused(0.0, max_skew=disc_max_skew, window_s=window_s, anchors=4)
    stale_video = _fused_video_unit(sync, 4, 1.0)
    generation_before = sync.generation
    sync.advance_generation(_FusedDiscontinuity.GAP)
    boundary = 5.0  # the stream resumes 4 s later
    stale_accepted = sync.accept(stale_video)
    post = []
    for k in range(4):
        t = boundary + k * 0.25
        v = _fused_video_unit(sync, k, t)
        accepted = sync.accept(v)
        if sync.generation_start_event_time is None:
            sync.set_generation_start(t)
        a = _fused_audio_unit(sync, k, t, t + 0.25)
        sync.accept(a)
        post.extend(sync.drain_ready())
        bundle = sync.bundle_for(v)
        if not any(
            reason is _FusedCoverageUncertainty.WATERMARK_PENDING
            for reason in bundle.coverage_reasons
        ):
            post.append(bundle)
        if not accepted:
            r.failures.append(f"post-discontinuity video anchor {k} was rejected")
    post.extend(sync.drain_ready())
    post.extend(sync.close())

    pre_boundary_leak = [
        b.video_event_interval
        for b in post
        if b.video_event_interval is not None and b.video_event_interval[0] < boundary - 1e-9
    ] + [
        b.audio_event_interval
        for b in post
        if b.audio_event_interval is not None and b.audio_event_interval[0] < boundary - 1e-9
    ]
    r.measurements["discontinuity"] = {
        "generation_before": generation_before,
        "generation_after": sync.generation,
        "stale_generation_unit_accepted": stale_accepted,
        "generation_start_event_time": sync.generation_start_event_time,
        "post_bundles": len(post),
        "pre_boundary_media_leaked": pre_boundary_leak,
        "post_reasons": sorted({x.value for b in post for x in b.coverage_reasons}),
    }
    r.check(
        sync.generation == generation_before + 1,
        "advance_generation did not advance the synchronizer generation",
    )
    r.check(
        stale_accepted is False,
        "a stale pre-discontinuity unit was accepted into the new generation",
    )
    r.check(
        not pre_boundary_leak,
        f"post-discontinuity bundles borrowed pre-boundary media: {pre_boundary_leak}",
    )
    r.check(bool(post), "no bundle was emitted after the mid-stream discontinuity")

    r.notes.append(
        "actual_skew_s is the distance from the video anchor to the NEAREST audio unit, not a "
        "systematic lip-sync offset: a uniformly shifted but still-overlapping audio lane reports "
        "0.0 by design (percept/fused_sync.py:611 _skew). Coverage, not offset, is what the "
        "synchronizer polices — audio that fails to cover [anchor-window, anchor-max_skew] is "
        "typed MISSING_AUDIO (percept/fused_sync.py:788)."
    )
    return r


# --------------------------------------------------------------------------------------------
# SV-0.5 — Delivered-fps honesty
# --------------------------------------------------------------------------------------------


def sv0_5() -> Result:
    r = Result(
        id="SV-0.5",
        title="Delivered-fps honesty",
        measures=(
            "Runs the real capture loop against a source that REPORTS 30 fps while delivering 15, "
            "and measures whether/when the rolling-window heuristic emits fps_misreported."
        ),
        bar="a 30-reported / 15-delivered source is flagged within the 2.0 s window",
    )
    from percept.runtime.source import _FPS_MIN_EVAL_S, _FPS_TOL, _FPS_WINDOW_S, SourceConfig

    r.measurements["heuristic"] = {
        "window_s": _FPS_WINDOW_S,
        "min_eval_s": _FPS_MIN_EVAL_S,
        "relative_tolerance": _FPS_TOL,
    }

    def probe(delivered_fps: float, reported_fps: float, frames: int = 200) -> dict[str, Any]:
        health: list = []
        fired_at: list = [None]
        config = SourceConfig(
            fps=reported_fps, queue_size=frames + 8, queue_policy="latest", timebase="media"
        )

        def clock(cap) -> float:
            # A simulated wall clock: elapsed time is (raw reads so far) / delivered rate.
            t = cap.i / delivered_fps
            if health and fired_at[0] is None:
                fired_at[0] = t
            return t

        drive_capture(
            "file.mp4",
            frames=frames,
            config=config,
            reported_fps=reported_fps,
            clock=clock,
            health=health,
        )
        return {
            "reported_fps": reported_fps,
            "delivered_fps": delivered_fps,
            "events": [(kind, dict(data)) for kind, data in health],
            "fired_at_elapsed_s": fired_at[0],
        }

    half = probe(15.0, 30.0)
    r.measurements["half_rate_source"] = half
    events = half["events"]
    r.check(
        any(kind == "fps_misreported" for kind, _ in events),
        "a 30-reported / 15-delivered source was NOT flagged (silent half-rate decimation)",
    )
    if events:
        kind, data = events[0]
        r.check(
            abs(float(data.get("reported", 0.0)) - 30.0) < 1e-9,
            f"fps_misreported reported the wrong nominal rate: {data}",
        )
        r.check(
            abs(float(data.get("measured", 0.0)) - 15.0) < 0.25,
            f"fps_misreported measured {data.get('measured')} for a true 15 fps delivery",
        )
    fired = half["fired_at_elapsed_s"]
    r.check(fired is not None, "the misreport never fired")
    if fired is not None:
        r.check(
            fired <= _FPS_WINDOW_S + 1e-9,
            f"the misreport fired at {fired:.3f}s, outside the {_FPS_WINDOW_S}s window",
        )
    r.check(
        sum(1 for kind, _ in events if kind == "fps_misreported") == 1,
        "fps_misreported was emitted more than once per open (should be emit-once)",
    )

    # negative controls: an honest source and a source inside tolerance must stay quiet
    honest = probe(30.0, 30.0)
    near = probe(29.0, 30.0)  # 3.3% off — inside the 25% tolerance
    quarter = probe(7.0, 30.0)  # far below — must also fire
    r.measurements["honest_source"] = honest
    r.measurements["within_tolerance_source"] = near
    r.measurements["quarter_rate_source"] = quarter
    r.check(not honest["events"], f"an honest 30/30 source was falsely flagged: {honest['events']}")
    r.check(
        not near["events"],
        f"a 30/29 source inside the {_FPS_TOL:.0%} tolerance was falsely flagged: {near['events']}",
    )
    r.check(
        any(kind == "fps_misreported" for kind, _ in quarter["events"]),
        "a 30-reported / 7-delivered source was not flagged",
    )

    # the heuristic must be inert on an instant offline fixture (span ~0 -> never evaluated)
    instant = {"events": []}
    health: list = []
    config = SourceConfig(fps=30.0, queue_size=128, queue_policy="latest")
    drive_capture(
        "file.mp4",
        frames=100,
        config=config,
        reported_fps=30.0,
        clock=lambda cap: 0.0,  # decoded instantly: no wall time elapses
        health=health,
    )
    instant["events"] = [(k, dict(d)) for k, d in health]
    r.measurements["instant_fixture"] = instant
    r.check(
        not instant["events"],
        f"an instantly-decoded offline fixture false-fired the fps heuristic: {instant['events']}",
    )

    # the delivered-fps signal reaches the public run() surface as a health event
    import inspect as _inspect

    from percept.runtime import worker as _worker

    src = _inspect.getsource(_worker._run_worker)
    wired = '"health": source_health' in src and "source_health.pop(0)" in src
    r.measurements["surfaced_through_on_health"] = wired
    r.check(
        wired,
        "the capture-thread health list is not drained into on_health by the worker",
    )
    return r


# --------------------------------------------------------------------------------------------
# SV-0.6 — Clock authority
# --------------------------------------------------------------------------------------------


def sv0_6() -> Result:
    r = Result(
        id="SV-0.6",
        title="Clock authority",
        measures=(
            "Checks that every envelope carries observation time independently of media time, that "
            "exactly one clock is authoritative for the selected timebase, and that decoder source "
            "time appears ONLY when the decoder truly provides it."
        ),
        bar="exactly one authoritative clock per envelope; source_time never manufactured",
    )
    from percept.runtime.envelope import SourceTracker
    from percept.runtime.source import SourceConfig

    def capture(uri: str, *, pos_mode: str, timebase: str = "media", frames: int = 6) -> list:
        config = SourceConfig(
            fps=30.0, queue_size=frames + 8, queue_policy="latest", timebase=timebase
        )
        return drive_capture(uri, frames=frames, config=config, pos_mode=pos_mode)

    # --- A. a real decoder clock is preserved, never replaced by observation time --------------
    advancing = capture("clip.mp4", pos_mode="advance")
    r.measurements["file_advancing_pos_msec"] = {
        "source_time": [e.source_time for e in advancing],
        "observed_time": [round(e.observed_time, 6) for e in advancing],
        "legacy_timebase": sorted({e.legacy_timebase for e in advancing}),
    }
    r.check(
        all(e.source_time is not None for e in advancing),
        "an advancing decoder clock was discarded",
    )
    r.check(
        all(e.observed_time is not None for e in advancing),
        "observation time is missing from a captured envelope",
    )
    r.check(
        all(e.source_time != e.observed_time for e in advancing),
        "source_time and observed_time are aliased — the two clocks are not independent",
    )

    # --- B. exactly one authoritative clock, selected by timebase ------------------------------
    media = capture("clip.mp4", pos_mode="advance", timebase="media")[0]
    wall = capture("clip.mp4", pos_mode="advance", timebase="wall")[0]
    r.measurements["timebase_selection"] = {
        "media": {
            "legacy_timebase": media.legacy_timebase,
            "event_time": media.event_time,
            "source_time": media.source_time,
            "observed_time": media.observed_time,
        },
        "wall": {
            "legacy_timebase": wall.legacy_timebase,
            "event_time": wall.event_time,
            "source_time": wall.source_time,
            "observed_time": wall.observed_time,
        },
    }
    r.check(media.legacy_timebase == "source", "timebase='media' did not select the source clock")
    r.check(media.event_time == media.source_time, "media event_time is not the decoder clock")
    r.check(wall.legacy_timebase == "observed", "timebase='wall' did not select the observation clock")
    r.check(wall.event_time == wall.observed_time, "wall event_time is not the observation clock")
    r.check(
        wall.source_time is not None,
        "selecting the wall timebase destroyed the independently-known media clock",
    )

    # --- C. source time is NEVER manufactured -------------------------------------------------
    cases = {
        "file_pos_msec_stuck_at_zero": capture("clip.mp4", pos_mode="stuck"),
        "file_pos_msec_non_finite": capture("clip.mp4", pos_mode="nan"),
        "rtsp_stream": capture("rtsp://camera/stream", pos_mode="advance"),
        "rtsps_stream": capture("rtsps://camera/stream", pos_mode="advance"),
        "udp_stream": capture("udp://239.0.0.1:1234", pos_mode="advance"),
        "webcam_device": capture("webcam:0", pos_mode="advance"),
    }
    manufactured = {}
    for name, envs in cases.items():
        times = [e.source_time for e in envs]
        manufactured[name] = times
        if name == "file_pos_msec_stuck_at_zero":
            # 0.0 is a legitimate FIRST frame; a non-advancing clock afterwards is not provenance.
            r.check(
                times[0] == 0.0 and all(t is None for t in times[1:]),
                f"{name}: a stuck decoder clock was treated as timestamp provenance: {times}",
            )
        else:
            r.check(
                all(t is None for t in times),
                f"{name}: source_time was manufactured where the decoder supplied none: {times}",
            )
    r.measurements["source_time_by_case"] = manufactured
    cadence_leak = [
        (name, times)
        for name, times in manufactured.items()
        if name != "file_pos_msec_stuck_at_zero"
        and any(t is not None and abs(t - i / 30.0) < 1e-9 for i, t in enumerate(times))
    ]
    r.measurements["cadence_derived_timestamps"] = cadence_leak
    r.check(not cadence_leak, f"a reported-fps cadence leaked in as source time: {cadence_leak}")

    # --- D. a missing media clock fails closed rather than silently substituting ---------------
    rtsp = cases["rtsp_stream"][0]
    raised = None
    try:
        rtsp.event_time
    except ValueError as exc:
        raised = str(exc)
    r.measurements["event_time_fails_closed"] = raised
    r.check(
        raised is not None,
        "event_time silently substituted observation time for an unknown media clock",
    )

    tracker = SourceTracker()
    lenient = tracker.inspect(rtsp)
    strict = SourceTracker().inspect(rtsp, require_source_time=True)
    r.measurements["tracker_unknown_clock"] = {
        "lenient": {"accepted": lenient.accepted, "events": list(lenient.events)},
        "strict": {
            "accepted": strict.accepted,
            "events": list(strict.events),
            "coverage": strict.coverage.kind,
        },
    }
    r.check(
        "source_time_unknown" in lenient.events,
        "an unknown media clock was not surfaced as a typed ingress event",
    )
    r.check(
        strict.accepted is False and "source_time_unknown" in strict.events,
        "require_source_time=True admitted an envelope with no media clock",
    )

    # --- E. audio: source time is the intrinsic PCM position, not a device rate ----------------
    from percept.runtime.audio import open_audio_source

    sample_rate = 16_000
    path, _ = synth_wav(sample_rate + 500, sample_rate=sample_rate)

    async def audio():
        return [env async for env in open_audio_source(path, chunk_s=0.25, sample_rate=sample_rate)]

    aenvs = asyncio.run(audio())
    cursor = 0
    audio_errors = []
    for env in aenvs:
        n = len(env.payload.pcm) // 2
        expected_end = (cursor + n) / sample_rate
        if env.source_time is None or abs(env.source_time - expected_end) > 1e-12:
            audio_errors.append((env.sequence, env.source_time, expected_end))
        cursor += n
    r.measurements["audio_source_time_is_sample_position"] = {
        "units": len(aenvs),
        "errors": audio_errors,
        "observed_time_present": all(e.observed_time is not None for e in aenvs),
        "observed_differs_from_source": all(
            e.observed_time != e.source_time for e in aenvs
        ),
    }
    r.check(not audio_errors, f"audio source_time is not the PCM sample position: {audio_errors[:3]}")
    r.check(
        all(e.observed_time is not None for e in aenvs),
        "audio envelopes lost their independent observation clock",
    )

    # --- F. a legacy tuple never gains a clock it did not carry -------------------------------
    from percept.runtime.envelope import legacy_video_envelope
    from percept.edge.tier0 import gray_frame

    legacy_observed = legacy_video_envelope(
        (3.0, gray_frame(4, 4, 0), "token"),
        source_id="video:legacy",
        sequence=0,
        observed_time=99.0,
        timebase="observed",
    )
    legacy_source = legacy_video_envelope(
        (3.0, gray_frame(4, 4, 0), "token"),
        source_id="video:legacy",
        sequence=0,
        observed_time=99.0,
        timebase="source",
    )
    r.measurements["legacy_tuple"] = {
        "observed_timebase": {
            "source_time": legacy_observed.source_time,
            "observed_time": legacy_observed.observed_time,
        },
        "source_timebase": {
            "source_time": legacy_source.source_time,
            "observed_time": legacy_source.observed_time,
        },
    }
    r.check(
        legacy_observed.source_time is None,
        "a legacy observation-time tuple was given a fabricated media clock",
    )
    r.check(
        legacy_source.source_time == 3.0 and legacy_source.observed_time == 99.0,
        "a legacy media-time tuple did not keep its two clocks separate",
    )
    return r


# --------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------

TESTS: tuple[Callable[[], Result], ...] = (sv0_1, sv0_2, sv0_3, sv0_4, sv0_5, sv0_6)


def runtime_provenance() -> dict[str, Any]:
    import percept

    return {
        "percept_version": getattr(percept, "__version__", None),
        "percept_path": getattr(percept, "__file__", None),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


def run_all() -> list[Result]:
    results: list[Result] = []
    for test in TESTS:
        try:
            results.append(test())
        except Exception as exc:  # noqa: BLE001 - a crashed check is a reportable outcome
            import traceback

            name = test.__name__.upper().replace("SV0_", "SV-0.")
            results.append(
                Result(
                    id=name,
                    title=name,
                    measures="(the check raised before it could complete)",
                    bar="n/a",
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
    return results


def payload(results: list[Result]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "suite": "sv0_ingest",
        "stage": "STAGE 0",
        "title": "Ingest + normalize",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime": runtime_provenance(),
        "counts": counts,
        "status": "PASS" if counts.get("PASS", 0) == len(results) else "FAIL",
        "tests": [r.as_dict() for r in results],
    }


def _print_table(results: list[Result]) -> None:
    width = max(len(r.title) for r in results) + 2
    print()
    print(f"{'ID':<8} {'TEST':<{width}} {'STATUS':<7} DETAIL")
    print("-" * (8 + width + 8 + 60))
    for r in results:
        detail = ""
        if r.status == "FAIL":
            detail = r.failures[0]
            if len(r.failures) > 1:
                detail += f"  (+{len(r.failures) - 1} more)"
        elif r.status == "ERROR":
            detail = (r.error or "").splitlines()[0]
        print(f"{r.id:<8} {r.title:<{width}} {r.status:<7} {detail}")
    print("-" * (8 + width + 8 + 60))
    passed = sum(1 for r in results if r.status == "PASS")
    print(f"{passed}/{len(results)} PASS")
    for r in results:
        if r.failures:
            print(f"\n{r.id} failures:")
            for f in r.failures:
                print(f"  - {f}")
        if r.error:
            print(f"\n{r.id} ERROR:\n{r.error}")
        for note in r.notes:
            print(f"\n{r.id} note: {note}")
    print()


def main() -> int:
    prov = runtime_provenance()
    repo_pkg = str(Path(__file__).resolve().parents[2] / "packages" / "percept-vision")
    if prov["percept_path"] and prov["percept_path"].startswith(repo_pkg):
        print(
            f"REFUSING TO RUN: percept resolved to the repo checkout ({prov['percept_path']}), "
            "not the installed launch wheel."
        )
        return 2
    print(f"runtime under test: percept {prov['percept_version']} @ {prov['percept_path']}")
    results = run_all()
    _print_table(results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload(results), indent=2, default=str) + "\n")
    print(f"wrote {JSON_OUT}")
    import shutil

    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
