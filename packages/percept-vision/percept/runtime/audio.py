"""Audio capture sources for the runtime worker — the twin of ``source.py`` for sound.

``open_audio_source(uri)`` opens a mic (``mic`` = default input, ``mic:0`` = an explicit index), a
``.wav`` file, or a video container (``.mp4``/``.mov``/``.mkv``/``.webm``) on a **background thread**
and yields versioned source envelopes for programmable Watches:
  * ``chunk``  — a typed :class:`~percept.contracts.AudioFrame` (pcm s16le mono), the payload the
    fused/acoustic judge sees via ``Percept.push_audio`` → ``AudioWindow`` → ``Vision.judge(audio=)``,
  * ``energy`` — normalized mean-square PCM energy in ``[0, 1]``, the tier-0 audio salience signal
    (the audio analog of ``source.py``'s ``gate_frame``), fed to ``MotionGate.evaluate(energy=)`` /
    an onset gate.

``sounddevice`` (the ``[mic]`` extra) is imported lazily inside the capture thread and only for
``mic:`` URIs; a ``.wav`` source uses stdlib ``wave`` (zero-dep, so tests/replay need no extra). A
container source shells out to an ``ffmpeg`` subprocess (see ``_capture_container``) — the same demux
dependency ``source.py``/``eval/count_video.py`` already assume is on PATH for video decode, so this adds
no NEW external dependency, only a second consumer of it. This mirrors ``source.py``'s "worker is
stdlib, capture pulls the optional dep" split.
"""
from __future__ import annotations

import array
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Literal

from percept.contracts import AudioFrame
from percept.events import AudioDirection, AudioUnit
from percept.runtime.envelope import SourceEnvelope, SourceFormat

_DEFAULT_SR = 16_000  # matches AudioFrame.sample_rate and the STT stream rate

# Video-container extensions whose audio track this module can demux via ffmpeg. Kept in sync with
# ``runtime.worker``'s auto-audio-source trigger (a container path + an armed fused/audio goal).
CONTAINER_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")


class FFmpegNotFoundError(RuntimeError):
    """A video-container ``audio_source`` needs ``ffmpeg`` on PATH to demux its audio track and it
    wasn't found. Typed so a caller can distinguish "no audio channel available" from a generic capture
    failure — a missing dependency must never silently degrade a fused watch to vision-only."""


@dataclass(frozen=True)
class AudioSourceConfig:
    """Runtime audio-capture tuning for mic / wav sources."""

    sample_rate: int = _DEFAULT_SR
    channels: int = 1
    chunk_s: float = 0.25  # gate cadence: ~4 energy readings/sec
    queue_size: int = 256
    # Clock selected by the transitional tuple view.  Every envelope independently carries local
    # observation time and the PCM sample position intrinsic to its AudioFrame.
    timebase: Literal["media", "wall"] = "media"
    # Retained when known. A mic defaults to ``input``; a file/container has no inherent direction.
    direction: AudioDirection | None = None
    participant_id: str | None = None


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
    return uri  # a .wav path or a video-container path (see ``_is_container``)


def _is_container(path: str) -> bool:
    """True when ``path`` names a video container (mp4/mov/mkv/webm) rather than a bare ``.wav`` —
    the dispatch key ``open_audio_source`` uses to route to ``_capture_container`` (ffmpeg demux)
    instead of ``_capture_wav`` (stdlib ``wave``)."""
    low = path.strip().lower().split("?", 1)[0].split("#", 1)[0]
    return low.endswith(CONTAINER_SUFFIXES)


def pcm_energy(pcm: bytes) -> float:
    """Mean-square energy of s16le PCM normalized to ``[0, 1]``.

    This is the same definition as :func:`percept.runtime.frame_energy` and JavaScript
    ``percept-edge/frameEnergy``. It deliberately is not RMS: the VAD, onset gate, runtime capture,
    and edge must compare one quantity against their thresholds.
    """
    if len(pcm) % 2:
        raise ValueError("s16le PCM length must be even")
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":  # PCM is explicitly s16le; array follows host byte order.
        samples.byteswap()
    if not samples:
        return 0.0
    denom = float(32768 * 32768 * len(samples))
    return min(1.0, sum(s * s for s in samples) / denom)


# Private compatibility alias for downstream code that imported the old helper. Its value now follows
# the public mean-square energy contract; the name is retained only to avoid an import break.
_rms_energy = pcm_energy


def _pcm_frame_energy(pcm: bytes, sample_format: str) -> float:
    """Mean-square energy for the raw PCM formats accepted by :class:`AudioFrame`."""
    if sample_format == "s16le":
        return pcm_energy(pcm)
    if sample_format == "f32le":
        if len(pcm) % 4:
            raise ValueError("f32le PCM length must be divisible by four")
        if not pcm:
            return 0.0
        samples = array.array("f")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        if any(not math.isfinite(sample) or not -1.0 <= sample <= 1.0 for sample in samples):
            raise ValueError("f32le PCM samples must be finite and within [-1, 1]")
        return min(1.0, sum(sample * sample for sample in samples) / len(samples))
    if sample_format == "u8":
        if not pcm:
            return 0.0
        return min(1.0, sum(((sample - 128) / 128) ** 2 for sample in pcm) / len(pcm))
    raise ValueError("unsupported PCM sample format")


def normalize_audio_unit(
    item,
    *,
    seq: int,
) -> AudioUnit:
    """Normalize a pushed/replayed legacy row to the same unit capture produces.

    Legacy ``(t, chunk, energy)`` rows expose only one event timestamp. For typed raw PCM, derive the
    source interval from the payload's true duration and anchor its end at ``t``. Never stretch a row
    back to the previous event timestamp: doing so would hide an actual dropped-audio gap.
    """
    if isinstance(item, AudioUnit):
        return item
    try:
        at, chunk, energy = item
    except (TypeError, ValueError) as exc:
        raise TypeError("audio source items must be AudioUnit or (t, chunk, energy)") from exc
    at = float(at)
    sample_rate = getattr(chunk, "sample_rate", None)
    channels = getattr(chunk, "channels", None)
    sample_format = getattr(chunk, "sample_format", None)
    t_start = getattr(chunk, "t_start", None)
    t_end = getattr(chunk, "t_end", None)
    if t_start is None or t_end is None:
        duration = None
        pcm = getattr(chunk, "pcm", None)
        if (
            isinstance(pcm, (bytes, bytearray, memoryview))
            and isinstance(sample_rate, int)
            and sample_rate > 0
            and isinstance(channels, int)
            and channels > 0
            and sample_format in ("s16le", "f32le", "u8")
        ):
            bytes_per_sample = {"s16le": 2, "f32le": 4, "u8": 1}[sample_format]
            bytes_per_frame = bytes_per_sample * channels
            if len(pcm) % bytes_per_frame:
                raise ValueError("audio frame PCM is not aligned to its format/channels")
            duration = len(pcm) / (bytes_per_frame * sample_rate)
        t_end = at
        t_start = at - duration if duration is not None else at
    if energy is None:
        pcm = getattr(chunk, "pcm", None)
        if (
            isinstance(pcm, (bytes, bytearray, memoryview))
            and sample_format in ("s16le", "f32le", "u8")
        ):
            energy = _pcm_frame_energy(bytes(pcm), sample_format)
    return AudioUnit(
        seq=seq,
        t_start=float(t_start),
        t_end=float(t_end),
        chunk=chunk,
        event_t=at,
        energy=None if energy is None else float(energy),
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
    )


_POLL = object()  # sentinel: "queue was empty this poll, try again" (distinct from the None end-sentinel)


def _get_polling(q: "queue.Queue", timeout: float = 0.5):
    """Bounded ``q.get()`` — returns ``_POLL`` on a timeout instead of blocking forever.

    AUDIO_PUMP_DEADLOCK: the consumer used a bare ``await asyncio.to_thread(q.get)`` with no
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


def _put_terminal(q: "queue.Queue", item, stop: threading.Event) -> None:
    """Publish an error/end marker after queued evidence instead of evicting that evidence."""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return
        except queue.Full:
            continue


def _emit(
    q,
    observed_time: float,
    source_time: float,
    t0: float,
    t1: float,
    pcm: bytes,
    sr: int,
    ch: int,
    sequence: int,
    timebase: str,
    dropped,
    *,
    source_id: str,
    epoch: int = 0,
    sample_format: str = "s16le",
    discontinuity: "Literal['capture_overflow'] | None" = None,
) -> None:
    frame = AudioFrame(pcm=pcm, sample_rate=sr, channels=ch, sample_format=sample_format,
                       t_start=t0, t_end=t1)
    _put_latest(
        q,
        SourceEnvelope(
            source_id=source_id,
            modality="audio",
            epoch=epoch,
            sequence=sequence,
            observed_time=observed_time,
            # PCM sample position is intrinsic to the captured payload.  It is not inferred from a
            # device's claimed delivery rate.
            source_time=source_time,
            format=SourceFormat(
                media_type="audio/pcm",
                sample_rate=sr,
                channels=ch,
                sample_format=sample_format,
            ),
            payload=frame,
            signal=_pcm_frame_energy(pcm, sample_format),
            discontinuity=discontinuity,
            legacy_timebase="observed" if timebase == "wall" else "source",
        ),
        dropped,
    )


def _wav_pcm(pcm: bytes, sample_width: int) -> tuple[bytes, str]:
    """Return truthful AudioFrame PCM for a stdlib ``wave`` sample width.

    WAV 8-bit PCM is unsigned and is preserved as ``u8``.  Signed 24/32-bit PCM is narrowed to the
    high 16 bits sample-by-sample; declaring those original bytes ``s16le`` would corrupt channel
    alignment and energy.  Widths outside PCM 8/16/24/32 fail loudly.
    """
    if sample_width == 1:
        return bytes(pcm), "u8"
    if sample_width == 2:
        return bytes(pcm), "s16le"
    if sample_width not in (3, 4):
        raise ValueError(f"unsupported WAV sample width: {sample_width * 8} bits")
    shift = 8 * (sample_width - 2)
    out = bytearray()
    for offset in range(0, len(pcm), sample_width):
        raw = pcm[offset:offset + sample_width]
        if len(raw) != sample_width:
            raise ValueError("WAV payload ended inside a PCM sample")
        narrowed = int.from_bytes(raw, "little", signed=True) >> shift
        out.extend(narrowed.to_bytes(2, "little", signed=True))
    return bytes(out), "s16le"


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
            sample_width = w.getsampwidth()
            step = max(1, int(sr * config.chunk_s))
            i = 0
            while not stop.is_set():
                pcm = w.readframes(step)
                if not pcm:
                    break
                samples = len(pcm) // max(1, ch * sample_width)
                t0, t1 = i / sr, (i + samples) / sr
                converted, sample_format = _wav_pcm(pcm, sample_width)
                _emit(
                    q, now(), t1, t0, t1, converted, sr, ch, i // step,
                    config.timebase, dropped, source_id=f"audio:{path}",
                    sample_format=sample_format,
                )
                i += samples
    except Exception as exc:  # surface to the consumer, like source.py
        _put_terminal(q, exc, stop)
    finally:
        _put_terminal(q, None, stop)


def _read_exact(f, n: int) -> bytes:
    """Read exactly ``n`` bytes from a pipe (or fewer at EOF) — a ``BufferedReader.read(n)`` may return
    a short read even mid-stream, so a single call isn't enough to fill one PCM chunk."""
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _capture_container(path: str, config: AudioSourceConfig, q: "queue.Queue",
                       stop: threading.Event, dropped: "list[int] | None",
                       *, now: "Callable[[], float] | None" = None) -> None:
    """Demux the audio track of a video container (mp4/mov/mkv/webm) via an ``ffmpeg`` subprocess,
    streaming raw PCM s16le chunks with source time measured from emitted sample positions. Video
    capture independently preserves decoder-provided media positions; neither adapter derives source
    time from a reported delivery rate.

    Requires ``ffmpeg`` on PATH. Its absence raises :class:`FFmpegNotFoundError` (surfaced to the
    consumer exactly like a wav/mic capture error) rather than silently yielding zero chunks — a fused
    goal degrading to vision-only must be visible, never a quiet default."""
    if now is None:
        now = time.monotonic
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        _put_terminal(q, FFmpegNotFoundError(
            f"{path!r} is a video container (mp4/mov/mkv/webm) — demuxing its audio track needs "
            "ffmpeg on PATH. Install it (e.g. `brew install ffmpeg` on macOS, `apt-get install "
            "ffmpeg` on Debian/Ubuntu) or pass a pre-demuxed .wav as audio_source instead."), stop)
        _put_terminal(q, None, stop)
        return
    sr, ch = config.sample_rate, config.channels
    step = max(1, int(sr * config.chunk_s))
    nbytes = step * ch * 2  # s16le = 2 bytes/sample
    cmd = [ffmpeg, "-v", "error", "-i", path, "-vn", "-map", "0:a:0",
           "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", str(ch), "-"]
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        i = 0
        while not stop.is_set():
            pcm = _read_exact(proc.stdout, nbytes)
            if not pcm:
                break
            samples = len(pcm) // max(1, ch * 2)
            t0, t1 = i / sr, (i + samples) / sr
            _emit(
                q, now(), t1, t0, t1, pcm, sr, ch, i // step, config.timebase, dropped,
                source_id=f"audio:{path}",
            )
            i += samples
        proc.stdout.close()
        rc = proc.wait(timeout=10)
        if rc != 0 and i == 0:
            # Zero chunks EVER produced + a nonzero exit is a real demux failure (no audio stream,
            # unreadable container, ...) — not a source we can silently treat as "no audio".
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed to demux audio from {path!r}: {stderr.strip()[:500]}")
    except FileNotFoundError as exc:  # pragma: no cover - shutil.which already guards this path
        _put_terminal(q, FFmpegNotFoundError(f"ffmpeg not found while demuxing {path!r}: {exc}"), stop)
    except Exception as exc:
        _put_terminal(q, exc, stop)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        _put_terminal(q, None, stop)


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
        _put_terminal(q, RuntimeError(
            "mic: source needs the [mic] extra — pip install 'percept-vision[mic]' (sounddevice). "
            f"({exc})"), stop)
        _put_terminal(q, None, stop)
        return
    sr, ch = config.sample_rate, config.channels
    step = max(1, int(sr * config.chunk_s))
    epoch = 0
    epoch_i = 0
    sequence = 0
    try:
        with sd.RawInputStream(samplerate=sr, channels=ch, dtype="int16",
                               blocksize=step, device=device) as stream:
            while not stop.is_set():
                data, overflowed = stream.read(step)
                pcm = bytes(data)
                if not pcm:
                    continue
                discontinuity = None
                if overflowed:
                    # sounddevice reports that continuity was lost but not how many samples.  Start a
                    # new epoch rather than claim contiguous positions or fabricate a gap count.
                    epoch += 1
                    epoch_i = 0
                    sequence = 0
                    discontinuity = "capture_overflow"
                samples = len(pcm) // max(1, ch * 2)
                t0, t1 = epoch_i / sr, (epoch_i + samples) / sr
                device_id = "default" if device is None else str(device)
                _emit(
                    q, now(), t1, t0, t1, pcm, sr, ch, sequence,
                    config.timebase, dropped, source_id=f"audio:mic:{device_id}",
                    epoch=epoch, discontinuity=discontinuity,
                )
                epoch_i += samples
                sequence += 1
    except Exception as exc:
        _put_terminal(q, exc, stop)
    finally:
        _put_terminal(q, None, stop)


async def open_audio_source(
    uri: str,
    *,
    sample_rate: int = _DEFAULT_SR,
    channels: int = 1,
    chunk_s: float = 0.25,
    maxsize: int = 256,
    timebase: "Literal['media', 'wall']" = "media",
    direction: "AudioDirection | None" = None,
    participant_id: str | None = None,
    dropped: "list[int] | None" = None,
) -> AsyncIterator[SourceEnvelope]:
    """Threaded audio capture → async source-envelope iterator. Stops the thread on exit.

    Every item preserves the PCM source position and local observation time independently. ``timebase``
    selects only the transitional tuple view. ``AudioFrame.t_start/t_end`` remain media offsets."""
    import asyncio

    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        raise ValueError("channels must be a positive integer")
    if (
        not isinstance(chunk_s, (int, float))
        or isinstance(chunk_s, bool)
        or not math.isfinite(chunk_s)
        or chunk_s <= 0
    ):
        raise ValueError("chunk_s must be finite and positive")
    if not isinstance(maxsize, int) or isinstance(maxsize, bool) or maxsize <= 0:
        raise ValueError("maxsize must be positive; zero would create an unbounded queue")
    if timebase not in ("media", "wall"):
        raise ValueError("timebase must be 'media' or 'wall'")

    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        raise ValueError("channels must be a positive integer")
    if (
        not isinstance(chunk_s, (int, float))
        or isinstance(chunk_s, bool)
        or not math.isfinite(chunk_s)
        or chunk_s <= 0
    ):
        raise ValueError("chunk_s must be finite and positive")
    if not isinstance(maxsize, int) or isinstance(maxsize, bool) or maxsize <= 0:
        raise ValueError("maxsize must be positive; zero would create an unbounded queue")
    if timebase not in ("media", "wall"):
        raise ValueError("timebase must be 'media' or 'wall'")

    config = AudioSourceConfig(sample_rate=sample_rate, channels=channels,
                               chunk_s=chunk_s, queue_size=maxsize, timebase=timebase,
                               direction=direction, participant_id=participant_id)
    target = _normalize_uri(uri)
    q: "queue.Queue" = queue.Queue(maxsize=config.queue_size)
    stop = threading.Event()
    # None (default input device) OR an int index -> mic; a str -> a .wav path OR a video-container
    # path (mp4/mov/mkv/webm) -> ffmpeg demux.
    if not isinstance(target, str):
        fn = _capture_mic
    elif _is_container(target):
        fn = _capture_container
    else:
        fn = _capture_wav
    thread = threading.Thread(target=fn, args=(target, config, q, stop, dropped), daemon=True)
    thread.start()
    try:
        while True:
            # AUDIO_PUMP_DEADLOCK: poll with a bounded timeout rather than a bare blocking
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


__all__ = [
    "AudioSourceConfig",
    "AudioUnit",
    "open_audio_source",
    "normalize_audio_unit",
    "pcm_energy",
    "FFmpegNotFoundError",
    "CONTAINER_SUFFIXES",
]
