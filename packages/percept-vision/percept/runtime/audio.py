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
import wave
from dataclasses import dataclass
from typing import AsyncIterator

from percept.contracts import AudioFrame

_DEFAULT_SR = 16_000  # matches AudioFrame.sample_rate and the STT stream rate


@dataclass(frozen=True)
class AudioSourceConfig:
    """Runtime audio-capture tuning for mic / wav sources."""

    sample_rate: int = _DEFAULT_SR
    channels: int = 1
    chunk_s: float = 0.25  # gate cadence: ~4 energy readings/sec
    queue_size: int = 256


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


def _emit(q, t0: float, t1: float, pcm: bytes, sr: int, ch: int, dropped) -> None:
    frame = AudioFrame(pcm=pcm, sample_rate=sr, channels=ch, sample_format="s16le",
                       t_start=t0, t_end=t1)
    _put_latest(q, (t1, frame, _rms_energy(pcm)), dropped)


def _capture_wav(path: str, config: AudioSourceConfig, q: "queue.Queue",
                 stop: threading.Event, dropped: "list[int] | None") -> None:
    """Stream a wav file in chunk_s slices (stdlib; deterministic — used by tests/replay)."""
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
                _emit(q, t0, t1, pcm, sr, ch, dropped)
                i += step
    except Exception as exc:  # surface to the consumer, like source.py
        _put_latest(q, exc)
    finally:
        _put_latest(q, None)  # end sentinel


def _capture_mic(device: "int | None", config: AudioSourceConfig, q: "queue.Queue",
                 stop: threading.Event, dropped: "list[int] | None") -> None:
    """Blocking mic capture on a daemon thread; lazy sounddevice import (the [mic] extra). ``device``
    is an explicit input index, or ``None`` to use sounddevice's default input device."""
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
                _emit(q, t0, t1, pcm, sr, ch, dropped)
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
    dropped: "list[int] | None" = None,
) -> AsyncIterator[tuple]:
    """Threaded audio capture → async ``(t, AudioFrame, energy)`` iterator. Stops the thread on exit."""
    import asyncio

    config = AudioSourceConfig(sample_rate=sample_rate, channels=channels,
                               chunk_s=chunk_s, queue_size=maxsize)
    target = _normalize_uri(uri)
    q: "queue.Queue" = queue.Queue(maxsize=config.queue_size)
    stop = threading.Event()
    # None (default input device) OR an int index -> mic; a str -> a .wav path.
    fn = _capture_wav if isinstance(target, str) else _capture_mic
    thread = threading.Thread(target=fn, args=(target, config, q, stop, dropped), daemon=True)
    thread.start()
    try:
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()


__all__ = ["AudioSourceConfig", "open_audio_source"]
