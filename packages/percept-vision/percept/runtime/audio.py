"""Audio capture sources for the runtime worker — the twin of ``source.py`` for sound.

``open_audio_source(uri)`` opens a mic (``mic`` = default input, ``mic:0`` = an explicit index) or a
``.wav`` file on a **background thread** and yields ``(t, chunk, energy)``:
  * ``chunk``  — a typed :class:`~percept.contracts.AudioFrame` (pcm s16le mono), the payload the
    fused/acoustic judge sees via ``Percept.push_audio`` → ``AudioWindow`` → ``Vision.judge(audio=)``,
  * ``energy`` — normalized short-time RMS in ``[0, 1]``, the tier-0 audio salience signal (the audio
    analog of ``source.py``'s ``gate_frame``), fed to ``MotionGate.evaluate(energy=)`` / an onset gate.

``sounddevice`` (the ``[mic]`` extra) is imported lazily inside the capture thread and only for
``mic:`` URIs; a ``.wav`` source uses stdlib ``wave`` (zero-dep, so tests/replay need no extra). This
mirrors ``source.py``'s "worker is stdlib, capture pulls the optional dep" split.
"""
from __future__ import annotations

import array
import math
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Literal

from percept.contracts import AudioFrame

_DEFAULT_SR = 16_000  # matches AudioFrame.sample_rate and the STT stream rate


@dataclass(frozen=True)
class AudioSourceConfig:
    """Runtime audio-capture tuning for mic / wav sources."""

    sample_rate: int = _DEFAULT_SR
    channels: int = 1
    chunk_s: float = 0.25  # gate cadence: ~4 energy readings/sec
    queue_size: int = 256
    # Timestamp origin for the yielded event ``t`` (A-clock switch, introduced NO-OP). ``"media"`` =
    # today's ``(i+step)/sr`` byte-for-byte; ``"wall"`` = ``time.monotonic()`` per emitted chunk. The
    # ``AudioFrame.t_start/t_end`` PCM-slice offsets stay MEDIA under BOTH modes (two jobs, two fields).
    timebase: Literal["media", "wall"] = "media"


def _normalize_uri(uri: str):
    """``mic`` / ``mic:default`` → the system DEFAULT input device (``None``); ``mic:0`` / ``mic://1``
    → an explicit device index (int); anything else → a filesystem path (str). Bare ``mic`` maps to the
    default input, NOT device 0 — device 0 is frequently an OUTPUT-only device, so a hardcoded 0 opens a
    stream with no input channels and silently captures nothing."""
    if uri in ("mic", "mic:default", "mic://default"):
        return None
    if uri.startswith("mic://"):
        return int(uri.removeprefix("mic://") or 0)
    if uri.startswith("mic:"):
        return int(uri.split(":", 1)[1] or 0)
    return uri  # a .wav path


def _rms_energy(pcm: bytes) -> float:
    """Short-time RMS of s16le PCM, normalized to [0, 1] (pure stdlib — no numpy)."""
    n = len(pcm) - (len(pcm) % 2)
    if n <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[:n])
    if not samples:
        return 0.0
    mean_sq = sum(s * s for s in samples) / len(samples)
    return min(1.0, math.sqrt(mean_sq) / 32768.0)


_POLL = object()  # sentinel: "queue was empty this poll, try again" (distinct from the None end-sentinel)


def _get_polling(q: "queue.Queue", timeout: float = 0.5):
    """Bounded ``q.get()`` — returns ``_POLL`` on a timeout instead of blocking forever.

    AUDIO_PUMP_DEADLOCK (M0/A2): the consumer used a bare ``await asyncio.to_thread(q.get)`` with no
    timeout. ``asyncio.to_thread`` runs the call on a real OS thread; once that thread has *started*
    running, cancelling/closing the awaiting coroutine (task cancellation, ``aclose()`` on this async
    generator, or the process shutting down) does NOT stop it — ``concurrent.futures.Future.cancel()``
    is a no-op on a work item that already started. If the queue then never receives another item (the
    capture thread stalled — e.g. a mic device wedged in a blocking read — or simply hasn't produced one
    yet when the caller stops consuming), that worker thread is orphaned inside ``q.get()`` forever. At
    interpreter exit, ``concurrent.futures.thread`` registers an ``atexit`` hook that joins every live
    worker thread before letting the process die — so the ORPHANED thread hangs the entire process
    indefinitely. Polling with a bounded timeout caps how long any single thread-pool call can block, so
    a torn-down consumer stops re-issuing gets and the last one still in flight resolves within
    ``timeout`` instead of never."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return _POLL


def _put_latest(q: "queue.Queue", item, dropped: "list[int] | None" = None) -> None:
    """Drop-oldest put so a slow consumer never blocks capture (mirrors source.py)."""
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


def _emit(q, t_event: float, t0: float, t1: float, pcm: bytes, sr: int, ch: int, dropped) -> None:
    # ``t_event`` is the event timestamp on the yielded tuple (media or wall per timebase); ``t0``/``t1``
    # are the MEDIA PCM-slice offsets on the AudioFrame — always media, regardless of timebase.
    frame = AudioFrame(pcm=pcm, sample_rate=sr, channels=ch, sample_format="s16le",
                       t_start=t0, t_end=t1)
    _put_latest(q, (t_event, frame, _rms_energy(pcm)), dropped)


def _capture_wav(path: str, config: AudioSourceConfig, q: "queue.Queue",
                 stop: threading.Event, dropped: "list[int] | None",
                 *, now: "Callable[[], float] | None" = None) -> None:
    """Stream a wav file in chunk_s slices (stdlib; deterministic — used by tests/replay)."""
    if now is None:
        now = time.monotonic
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            step = max(1, int(sr * config.chunk_s))
            i = 0
            while not stop.is_set():
                pcm = w.readframes(step)
                if not pcm:
                    break
                t0, t1 = i / sr, (i + step) / sr
                t_event = now() if config.timebase == "wall" else t1
                _emit(q, t_event, t0, t1, pcm, sr, ch, dropped)
                i += step
    except Exception as exc:  # surface to the consumer, like source.py
        _put_latest(q, exc)
    finally:
        _put_latest(q, None)  # end sentinel


def _capture_mic(device: "int | None", config: AudioSourceConfig, q: "queue.Queue",
                 stop: threading.Event, dropped: "list[int] | None",
                 *, now: "Callable[[], float] | None" = None) -> None:
    """Blocking mic capture on a daemon thread; lazy sounddevice import (the [mic] extra). ``device``
    is an explicit input index, or ``None`` to use sounddevice's default input device."""
    if now is None:
        now = time.monotonic
    try:
        import sounddevice as sd  # lazy: the [mic] extra
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        _put_latest(q, RuntimeError(
            "mic: source needs the [mic] extra — pip install 'percept-vision[mic]' (sounddevice). "
            f"({exc})"))
        _put_latest(q, None)
        return
    sr, ch = config.sample_rate, config.channels
    step = max(1, int(sr * config.chunk_s))
    i = 0
    try:
        with sd.RawInputStream(samplerate=sr, channels=ch, dtype="int16",
                               blocksize=step, device=device) as stream:
            while not stop.is_set():
                data, _overflowed = stream.read(step)
                pcm = bytes(data)
                if not pcm:
                    continue
                t0, t1 = i / sr, (i + step) / sr
                t_event = now() if config.timebase == "wall" else t1
                _emit(q, t_event, t0, t1, pcm, sr, ch, dropped)
                i += step
    except Exception as exc:
        _put_latest(q, exc)
    finally:
        _put_latest(q, None)


async def open_audio_source(
    uri: str,
    *,
    sample_rate: int = _DEFAULT_SR,
    channels: int = 1,
    chunk_s: float = 0.25,
    maxsize: int = 256,
    timebase: "Literal['media', 'wall']" = "media",
    dropped: "list[int] | None" = None,
) -> AsyncIterator[tuple]:
    """Threaded audio capture → async ``(t, AudioFrame, energy)`` iterator. Stops the thread on exit.

    ``timebase`` selects the event ``t`` origin (``"media"`` = today's ``(i+step)/sr``; ``"wall"`` =
    monotonic per chunk). ``AudioFrame.t_start/t_end`` remain MEDIA offsets under both modes."""
    import asyncio

    config = AudioSourceConfig(sample_rate=sample_rate, channels=channels,
                               chunk_s=chunk_s, queue_size=maxsize, timebase=timebase)
    target = _normalize_uri(uri)
    q: "queue.Queue" = queue.Queue(maxsize=config.queue_size)
    stop = threading.Event()
    # None (default input device) OR an int index -> mic; a str -> a .wav path.
    fn = _capture_wav if isinstance(target, str) else _capture_mic
    thread = threading.Thread(target=fn, args=(target, config, q, stop, dropped), daemon=True)
    thread.start()
    try:
        while True:
            # AUDIO_PUMP_DEADLOCK (M0/A2): poll with a bounded timeout rather than a bare blocking
            # ``q.get()`` — see ``_get_polling`` for why an unbounded get here can hang the whole
            # process at exit once the consumer is torn down mid-wait.
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


__all__ = ["AudioSourceConfig", "open_audio_source"]
