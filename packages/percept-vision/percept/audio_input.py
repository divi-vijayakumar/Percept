"""Trusted audio input and dispatch for compiled programmable Watches.

Manual backend calls remain flexible for compatibility.  The compiled runtime is deliberately
narrower: it accepts one typed :class:`AudioFrame`, validates and bounds it before allocating, and
dispatches only through a callable captured from an exact SDK-owned adapter at Watch preparation.
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
import threading
import types
import weakref
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from .backends import gemini as _gemini_backend
from .backends.fake import FakeVision as _SdkFakeVision
from .backends.gemini import GeminiVision as _SdkGeminiVision
from .contracts import AudioFrame, IncidentSnippet, Verdict

AudioJudgeMode = Literal["audio_only", "fused"]

MAX_AUDIO_INPUT_BYTES = 16 * 1024 * 1024
MAX_AUDIO_PROVIDER_BYTES = 16 * 1024 * 1024
_WAV_HEADER_BYTES = 44
_PCM_WIDTH = {"s16le": 2, "u8": 1, "f32le": 4}
_WAV_MIMES = frozenset({"audio/wav", "audio/x-wav"})


@dataclass(frozen=True)
class AudioInputCapability:
    """The closed, SDK-enforced audio surface of one trusted adapter."""

    modes: frozenset[AudioJudgeMode]
    pcm_formats: frozenset[str] = frozenset(_PCM_WIDTH)
    encoded_mimes: frozenset[str] = _WAV_MIMES
    max_input_bytes: int = MAX_AUDIO_INPUT_BYTES
    max_provider_bytes: int = MAX_AUDIO_PROVIDER_BYTES

    def __post_init__(self) -> None:
        if type(self.modes) is not frozenset:
            raise ValueError("audio modes must be an exact immutable frozenset")
        if type(self.pcm_formats) is not frozenset:
            raise ValueError("audio PCM formats must be an exact immutable frozenset")
        if type(self.encoded_mimes) is not frozenset:
            raise ValueError("audio MIME formats must be an exact immutable frozenset")
        if not self.modes or not self.modes <= {"audio_only", "fused"}:
            raise ValueError("audio modes must be a non-empty subset of {'audio_only', 'fused'}")
        if not self.pcm_formats or not self.pcm_formats <= set(_PCM_WIDTH):
            raise ValueError("audio PCM formats are invalid")
        if self.encoded_mimes != _WAV_MIMES:
            raise ValueError("compiled audio currently supports PCM WAV containers only")
        if (
            type(self.max_input_bytes) is not int
            or type(self.max_provider_bytes) is not int
            or self.max_input_bytes <= 0
            or self.max_provider_bytes <= _WAV_HEADER_BYTES
        ):
            raise ValueError("audio byte caps must be positive and leave room for a WAV header")


@dataclass(frozen=True)
class AudioJudgeBinding:
    """Immutable dispatch selected while a compiled Watch is prepared.

    ``dispatch`` is a bound method built directly from the accepted class's original function
    object.  Runtime dispatch never looks up ``core.vision`` or ``adapter.judge`` again.
    """

    adapter: Any = field(repr=False, compare=False)
    dispatch: Callable[..., Awaitable[Any]] = field(repr=False, compare=False)
    adapter_identity: str
    mode: AudioJudgeMode
    capability: AudioInputCapability
    application_key: str
    binding_token: object = field(default_factory=object, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.mode not in ("audio_only", "fused"):
            raise ValueError("audio judge binding mode is invalid")
        if type(self.adapter_identity) is not str or not self.adapter_identity:
            raise ValueError("audio judge binding requires an adapter identity")
        if type(self.application_key) is not str or not self.application_key:
            raise ValueError("audio judge binding requires an application key")
        if not callable(self.dispatch):
            raise ValueError("audio judge binding requires a callable dispatch")
        if type(self.capability) is not AudioInputCapability:
            raise ValueError("audio judge binding requires an exact AudioInputCapability")
        if self.mode not in self.capability.modes:
            raise ValueError("audio judge binding mode is outside its capability")
        if type(self.binding_token) is not object:
            raise ValueError("audio judge binding token must be an opaque exact object")


@dataclass(frozen=True)
class _WavInfo:
    sample_rate: int
    channels: int
    sample_format: str
    data: memoryview = field(repr=False, compare=False)
    input_bytes: int
    predicted_pcm16_bytes: int


def _buffer(value: Any, *, field_name: str) -> memoryview:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{field_name} must be bytes-like")
    view = memoryview(value)
    if not view.c_contiguous:
        raise ValueError(f"{field_name} must be a contiguous byte buffer")
    if view.nbytes > MAX_AUDIO_INPUT_BYTES:
        raise ValueError(f"{field_name} exceeds the compiled audio input cap")
    try:
        return view.cast("B")
    except TypeError as exc:
        raise ValueError(f"{field_name} must be byte-addressable") from exc


def _normalize_mime(value: Any) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("encoded audio requires a MIME type")
    mime = value.strip().lower()
    if mime not in _WAV_MIMES:
        raise ValueError(
            "compiled audio supports only raw s16le/u8/f32le PCM or PCM WAV"
        )
    return "audio/wav"


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"audio {name} must be a positive integer")
    return value


def _finite_time(value: Any, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"audio {name} must be finite")
    return float(value)


def _predict_pcm16_size(input_size: int, sample_format: str) -> int:
    if sample_format == "s16le":
        return input_size
    if sample_format == "u8":
        return input_size * 2
    if sample_format == "f32le":
        return input_size // 2
    raise ValueError(f"unsupported audio sample format {sample_format!r}")


def _require_output_cap(output_size: int, *, capability: AudioInputCapability) -> None:
    if output_size + _WAV_HEADER_BYTES > capability.max_provider_bytes:
        raise ValueError("audio provider payload would exceed the compiled output cap")


def _parse_pcm_wav(
    encoded: Any,
    *,
    capability: AudioInputCapability,
) -> _WavInfo:
    """Strict RIFF/PCM inspection with no payload copy or sample expansion."""
    view = _buffer(encoded, field_name="AudioFrame.encoded")
    total = view.nbytes
    if total > capability.max_input_bytes:
        raise ValueError("WAV input exceeds the compiled audio input cap")
    if total < 12 or bytes(view[0:4]) != b"RIFF" or bytes(view[8:12]) != b"WAVE":
        raise ValueError("encoded audio must be a RIFF/WAVE container")
    if int.from_bytes(view[4:8], "little") + 8 != total:
        raise ValueError("WAV RIFF declared length must equal the exact payload length")

    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    data: memoryview | None = None
    while offset < total:
        if offset + 8 > total:
            raise ValueError("WAV ended inside a chunk header")
        chunk_id = bytes(view[offset : offset + 4])
        size = int.from_bytes(view[offset + 4 : offset + 8], "little")
        start = offset + 8
        end = start + size
        padded_end = end + (size & 1)
        if end > total or padded_end > total:
            raise ValueError("WAV chunk exceeds the declared RIFF boundary")
        if chunk_id == b"fmt ":
            if fmt is not None:
                raise ValueError("WAV contains duplicate fmt chunks")
            if size != 16:
                raise ValueError("compiled audio requires one canonical 16-byte PCM fmt chunk")
            fmt = struct.unpack_from("<HHIIHH", view, start)
        elif chunk_id == b"data":
            if data is not None:
                raise ValueError("WAV contains duplicate data chunks")
            if size == 0:
                raise ValueError("WAV data chunk must be non-empty")
            data = view[start:end]
        offset = padded_end
    if offset != total:
        raise ValueError("WAV contains trailing bytes outside its chunk structure")
    if fmt is None or data is None:
        raise ValueError("WAV requires exactly one complete fmt chunk and one data chunk")

    encoding, channels, sample_rate, byte_rate, block_align, bits = fmt
    if encoding != 1:
        raise ValueError("compiled audio accepts uncompressed integer PCM WAV only")
    channels = _positive_int(channels, name="channels")
    sample_rate = _positive_int(sample_rate, name="sample_rate")
    if bits == 8:
        sample_format = "u8"
    elif bits == 16:
        sample_format = "s16le"
    else:
        raise ValueError("compiled PCM WAV supports only 8-bit unsigned or 16-bit signed samples")
    width = _PCM_WIDTH[sample_format]
    expected_align = channels * width
    if block_align != expected_align or byte_rate != sample_rate * expected_align:
        raise ValueError("WAV fmt rate/alignment metadata is inconsistent")
    if data.nbytes % expected_align:
        raise ValueError("WAV data ends inside an interleaved sample frame")
    predicted = _predict_pcm16_size(data.nbytes, sample_format)
    _require_output_cap(predicted, capability=capability)
    return _WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        data=data,
        input_bytes=total,
        predicted_pcm16_bytes=predicted,
    )


def _pcm16_bytes(view: memoryview, sample_format: str) -> bytes:
    """Convert an already-sized and validated buffer after every cap check has passed."""
    if sample_format == "s16le":
        return bytes(view)
    if sample_format == "u8":
        out = bytearray(view.nbytes * 2)
        for index, sample in enumerate(view):
            struct.pack_into("<h", out, index * 2, (sample - 128) * 256)
        return bytes(out)
    # Validate the complete source before allocating the conversion buffer. A hostile final NaN
    # therefore cannot make a malformed payload consume the predicted output allocation.
    for (sample,) in struct.iter_unpack("<f", view):
        if not math.isfinite(sample) or not -1.0 <= sample <= 1.0:
            raise ValueError("f32le samples must be finite and within [-1, 1]")
    out = bytearray((view.nbytes // 4) * 2)
    for index, (sample,) in enumerate(struct.iter_unpack("<f", view)):
        value = (
            -32768
            if sample == -1.0
            else 32767
            if sample == 1.0
            else int(round(sample * 32767.0))
        )
        struct.pack_into("<h", out, index * 2, value)
    return bytes(out)


def canonicalize_audio_frame(
    frame: Any,
    *,
    t_start: float,
    t_end: float,
    capability: AudioInputCapability,
) -> AudioFrame:
    """Return immutable canonical s16le PCM after pure inspection and size prediction.

    This is the sole compiled-runtime media boundary.  It performs no provider call and mutates no
    source, DSP, window, gate, or spend state.
    """
    if type(frame) is not AudioFrame:
        raise ValueError("compiled audio ingress requires an exact AudioFrame")
    start = _finite_time(t_start, name="t_start")
    end = _finite_time(t_end, name="t_end")
    if end < start:
        raise ValueError("audio t_end must be >= t_start")

    pcm_size = (
        memoryview(frame.pcm).nbytes
        if isinstance(frame.pcm, (bytes, bytearray, memoryview))
        else -1
    )
    encoded_size = (
        memoryview(frame.encoded).nbytes
        if isinstance(frame.encoded, (bytes, bytearray, memoryview))
        else 0 if frame.encoded is None else -1
    )
    if pcm_size < 0 or encoded_size < 0:
        raise ValueError("AudioFrame carriers must be bytes-like")
    has_pcm = pcm_size > 0
    has_encoded = encoded_size > 0
    if has_pcm == has_encoded:
        raise ValueError("AudioFrame requires exactly one non-empty carrier: PCM XOR encoded")
    if pcm_size > capability.max_input_bytes or encoded_size > capability.max_input_bytes:
        raise ValueError("AudioFrame exceeds the compiled audio input cap")

    declared_start = frame.t_start
    declared_end = frame.t_end
    if declared_start is not None and not math.isclose(
        _finite_time(declared_start, name="frame.t_start"),
        start,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("AudioFrame t_start disagrees with its AudioUnit")
    if declared_end is not None and not math.isclose(
        _finite_time(declared_end, name="frame.t_end"),
        end,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("AudioFrame t_end disagrees with its AudioUnit")

    if has_encoded:
        _normalize_mime(frame.mime)
        wav = _parse_pcm_wav(frame.encoded, capability=capability)
        declared_rate = _positive_int(frame.sample_rate, name="sample_rate")
        declared_channels = _positive_int(frame.channels, name="channels")
        if declared_rate != wav.sample_rate:
            raise ValueError("AudioFrame sample_rate disagrees with WAV fmt metadata")
        if declared_channels != wav.channels:
            raise ValueError("AudioFrame channels disagrees with WAV fmt metadata")
        if frame.sample_format != wav.sample_format:
            raise ValueError("AudioFrame sample_format disagrees with WAV fmt metadata")
        sample_rate, channels, source_format = (
            wav.sample_rate,
            wav.channels,
            wav.sample_format,
        )
        raw = wav.data
    else:
        sample_rate = _positive_int(frame.sample_rate, name="sample_rate")
        channels = _positive_int(frame.channels, name="channels")
        source_format = frame.sample_format
        if source_format not in capability.pcm_formats:
            raise ValueError(f"unsupported compiled PCM format {source_format!r}")
        raw = _buffer(frame.pcm, field_name="AudioFrame.pcm")
        width = _PCM_WIDTH[source_format]
        if raw.nbytes == 0 or raw.nbytes % (width * channels):
            raise ValueError("AudioFrame PCM must contain complete interleaved sample frames")
        predicted = _predict_pcm16_size(raw.nbytes, source_format)
        _require_output_cap(predicted, capability=capability)

    width = _PCM_WIDTH[source_format]
    samples_per_channel = raw.nbytes / (width * channels)
    duration = samples_per_channel / sample_rate
    if not math.isclose(end - start, duration, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("AudioFrame interval disagrees with its sample count and format")
    pcm16 = _pcm16_bytes(raw, source_format)
    return AudioFrame(
        pcm=pcm16,
        sample_rate=sample_rate,
        channels=channels,
        sample_format="s16le",
        encoded=None,
        mime=None,
        t_start=start,
        t_end=end,
    )


def canonicalize_audio_unit(
    unit: Any,
    *,
    capability: AudioInputCapability,
    source_id: str,
    source_epoch: int,
    source_sequence: int,
    observed_time: float,
    source_time: float | None,
) -> Any:
    """Canonicalize one AudioUnit and bind its carrier provenance without mutating callers."""
    from .events import AudioUnit

    if type(unit) is not AudioUnit:
        raise ValueError("compiled audio ingress requires an exact AudioUnit")
    if not isinstance(unit.seq, int) or isinstance(unit.seq, bool) or unit.seq < 0:
        raise ValueError("audio seq must be a non-negative exact integer")
    start = _finite_time(unit.t_start, name="t_start")
    end = _finite_time(unit.t_end, name="t_end")
    if end < start:
        raise ValueError("audio t_end must be >= t_start")
    if type(source_id) is not str or not source_id:
        raise ValueError("audio source_id must be a non-empty exact string")
    for value, name in (
        (source_epoch, "source_epoch"),
        (source_sequence, "source_sequence"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"audio {name} must be a non-negative exact integer")
    observed = _finite_time(observed_time, name="observed_time")
    source_clock = (
        None if source_time is None else _finite_time(source_time, name="source_time")
    )
    if unit.direction not in ("input", "output", "mixed", "unknown"):
        raise ValueError("audio direction is invalid")
    if unit.participant_id is not None and (
        type(unit.participant_id) is not str or not unit.participant_id
    ):
        raise ValueError("audio participant_id must be a non-empty exact string when present")
    from .events import AudioBoundary

    if type(unit.boundaries) is not tuple:
        raise ValueError("audio boundaries must be an immutable tuple")
    prior = -math.inf
    for boundary in unit.boundaries:
        if type(boundary) is not AudioBoundary:
            raise ValueError("audio boundaries require exact AudioBoundary values")
        if boundary.kind not in ("onset", "vad_start", "endpoint"):
            raise ValueError("audio boundary kind is invalid")
        point = _finite_time(boundary.t, name="boundary")
        if point < prior:
            raise ValueError("audio boundaries must be ordered on their event clock")
        prior = point
    original_frame = unit.chunk
    if type(original_frame) is not AudioFrame:
        raise ValueError("compiled audio ingress requires an exact AudioFrame chunk")
    for outer, inner, name in (
        (unit.sample_rate, original_frame.sample_rate, "sample_rate"),
        (unit.channels, original_frame.channels, "channels"),
        (unit.sample_format, original_frame.sample_format, "sample_format"),
    ):
        if outer is not None and outer != inner:
            raise ValueError(f"AudioUnit {name} contradicts its AudioFrame")
    frame = canonicalize_audio_frame(
        unit.chunk,
        t_start=start,
        t_end=end,
        capability=capability,
    )
    event_t = unit.event_t
    if event_t is not None:
        event_t = _finite_time(event_t, name="event_t")
    energy = unit.energy
    if energy is not None and (
        not isinstance(energy, (int, float))
        or isinstance(energy, bool)
        or not math.isfinite(energy)
        or not 0.0 <= float(energy) <= 1.0
    ):
        raise ValueError("audio energy must be finite and within [0, 1]")
    if energy is None:
        # Mean-square normalized s16 energy: deterministic, bounded, and computed only after the
        # strict input/output caps and complete sample validation have succeeded.
        samples = len(frame.pcm) // 2
        if samples <= 0:
            raise ValueError("canonical audio requires at least one sample")
        squared = 0.0
        for (sample,) in struct.iter_unpack("<h", frame.pcm):
            normalized = sample / 32768.0
            squared += normalized * normalized
        energy = min(1.0, squared / samples)
    return AudioUnit(
        seq=unit.seq,
        t_start=start,
        t_end=end,
        chunk=frame,
        event_t=event_t,
        energy=float(energy),
        sample_rate=frame.sample_rate,
        channels=frame.channels,
        sample_format=frame.sample_format,
        direction=unit.direction,
        participant_id=unit.participant_id,
        boundaries=unit.boundaries,
        source_id=source_id,
        source_epoch=source_epoch,
        source_sequence=source_sequence,
        observed_time=observed,
        source_time=source_clock,
    )


def canonicalize_audio_ingress_frame(
    frame: Any,
    *,
    event_t: float,
    seq: int,
    energy: float | None,
    capability: AudioInputCapability,
    source_id: str,
    source_epoch: int,
    source_sequence: int,
    observed_time: float,
    source_time: float | None,
    direction: str = "unknown",
    participant_id: str | None = None,
) -> Any:
    """Canonicalize an exact AudioFrame before constructing a carrier envelope.

    When the frame omits its media interval, its byte-count duration is anchored at the source
    timestamp when available, otherwise at the explicitly supplied legacy event timestamp. All
    carrier inspection and cap prediction happens before an AudioUnit/SourceEnvelope can copy it.
    """
    from .events import AudioUnit

    if type(frame) is not AudioFrame:
        raise ValueError("compiled audio ingress requires an exact AudioFrame")
    event = _finite_time(event_t, name="event_t")
    declared_start, declared_end = frame.t_start, frame.t_end
    if (declared_start is None) != (declared_end is None):
        raise ValueError("AudioFrame must declare both t_start/t_end or neither")
    if declared_start is not None:
        start = _finite_time(declared_start, name="frame.t_start")
        end = _finite_time(declared_end, name="frame.t_end")
    else:
        pcm_size = (
            memoryview(frame.pcm).nbytes
            if isinstance(frame.pcm, (bytes, bytearray, memoryview))
            else -1
        )
        encoded_size = (
            memoryview(frame.encoded).nbytes
            if isinstance(frame.encoded, (bytes, bytearray, memoryview))
            else 0 if frame.encoded is None else -1
        )
        if pcm_size < 0 or encoded_size < 0 or (pcm_size > 0) == (encoded_size > 0):
            raise ValueError("AudioFrame requires exactly one non-empty bytes-like carrier")
        if pcm_size > 0:
            rate = _positive_int(frame.sample_rate, name="sample_rate")
            channels = _positive_int(frame.channels, name="channels")
            if frame.sample_format not in capability.pcm_formats:
                raise ValueError("unsupported compiled PCM format")
            view = _buffer(frame.pcm, field_name="AudioFrame.pcm")
            width = _PCM_WIDTH[frame.sample_format]
            if not view.nbytes or view.nbytes % (width * channels):
                raise ValueError("AudioFrame PCM must contain complete interleaved sample frames")
            duration = view.nbytes / (width * channels * rate)
        else:
            _normalize_mime(frame.mime)
            wav = _parse_pcm_wav(frame.encoded, capability=capability)
            duration = wav.data.nbytes / (
                _PCM_WIDTH[wav.sample_format] * wav.channels * wav.sample_rate
            )
        end = _finite_time(
            source_time if source_time is not None else event,
            name="derived_t_end",
        )
        start = end - duration
    candidate = AudioUnit(
        seq=seq,
        t_start=start,
        t_end=end,
        event_t=event,
        chunk=frame,
        energy=energy,
        direction=direction,
        participant_id=participant_id,
        boundaries=(),
    )
    return canonicalize_audio_unit(
        candidate,
        capability=capability,
        source_id=source_id,
        source_epoch=source_epoch,
        source_sequence=source_sequence,
        observed_time=observed_time,
        source_time=source_time,
    )


def _wav_bytes(pcm16: bytes, sample_rate: int, channels: int) -> bytes:
    block_align = channels * 2
    byte_rate = sample_rate * block_align
    data_size = len(pcm16)
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", 36 + data_size),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, 16),
            b"data",
            struct.pack("<I", data_size),
            pcm16,
        )
    )


def validate_audio_units_for_provider(
    units: tuple[Any, ...],
    *,
    capability: AudioInputCapability,
) -> int:
    """Validate a retained tuple and return its predicted WAV byte size without joining it."""
    from .events import AudioUnit

    if not units:
        raise ValueError("audio judgment requires at least one retained unit")
    first = units[0]
    sample_rate = first.sample_rate
    channels = first.channels
    if sample_rate is None or channels is None:
        raise ValueError("canonical audio units require rate and channel metadata")
    total = 0
    previous = None
    for unit in units:
        if (
            type(unit) is not AudioUnit
            or unit.sample_rate != sample_rate
            or unit.channels != channels
            or unit.sample_format != "s16le"
            or type(unit.chunk) is not AudioFrame
            or unit.chunk.encoded is not None
            or not isinstance(unit.chunk.pcm, bytes)
            or not unit.chunk.pcm
            or type(unit.source_id) is not str
            or not unit.source_id
            or type(unit.source_epoch) is not int
            or isinstance(unit.source_epoch, bool)
            or type(unit.source_sequence) is not int
            or isinstance(unit.source_sequence, bool)
            or unit.observed_time is None
        ):
            raise ValueError("audio judgment units must be homogeneous canonical s16le PCM")
        if previous is not None:
            if (
                previous.source_id != unit.source_id
                or previous.source_epoch != unit.source_epoch
                or previous.source_sequence + 1 != unit.source_sequence
                or previous.seq + 1 != unit.seq
                or not math.isclose(
                    previous.t_end, unit.t_start, rel_tol=0.0, abs_tol=1e-9
                )
            ):
                raise ValueError("audio judgment evidence may not span a discontinuity")
        total += len(unit.chunk.pcm)
        _require_output_cap(total, capability=capability)
        previous = unit
    return total + _WAV_HEADER_BYTES


def audio_frame_from_units(
    units: tuple[Any, ...],
    *,
    capability: AudioInputCapability,
) -> AudioFrame:
    """Freeze one provider payload from one exact, contiguous retained unit tuple."""
    validate_audio_units_for_provider(units, capability=capability)
    first = units[0]
    sample_rate = first.sample_rate
    channels = first.channels
    pcm16 = b"".join(unit.chunk.pcm for unit in units)
    wav = _wav_bytes(pcm16, sample_rate, channels)
    return AudioFrame(
        pcm=b"",
        sample_rate=sample_rate,
        channels=channels,
        sample_format="s16le",
        encoded=wav,
        mime="audio/wav",
        t_start=float(first.t_start),
        t_end=float(units[-1].t_end),
    )


def audio_script_token(frame: AudioFrame) -> str:
    """Stable FakeVision key for the exact canonical audio payload."""
    if type(frame) is not AudioFrame:
        raise TypeError("audio script token requires an exact AudioFrame")
    mime, raw = frame.as_pair()
    digest = hashlib.sha256(mime.lower().encode("ascii") + b"\0" + raw).hexdigest()
    return f"audio:sha256:{digest}"


def audio_snippet_from_units(
    units: tuple[Any, ...],
    *,
    capability: AudioInputCapability,
) -> IncidentSnippet:
    """Build the delivered audio evidence from the same immutable tuple used by the judge."""
    payload = audio_frame_from_units(units, capability=capability)
    return IncidentSnippet(
        kind="audio",
        media=payload.encoded,
        mime=payload.mime,
        frames=list(units),
        t_start=payload.t_start,
        t_end=payload.t_end,
    )


_TRUSTED_AUDIO_JUDGES = {
    _SdkFakeVision: _SdkFakeVision.__dict__["judge"],
    _SdkGeminiVision: _SdkGeminiVision.__dict__["judge"],
}

# ``GeminiVision.judge`` intentionally lazy-loads its provider and uses module helpers. Capturing
# only the outer function would therefore leave several dynamic lookup paths that code mutating the
# live adapter or module after Watch application could redirect. Keep the accepted identities here,
# then check them immediately before every compiled-audio dispatch.
_GEMINI_BASE = _SdkGeminiVision.__mro__[1]
_TRUSTED_GEMINI_ENSURE_CLIENT = _GEMINI_BASE.__dict__["_ensure_client"]
_TRUSTED_GEMINI_METHODS = {
    "judge": _SdkGeminiVision.__dict__["judge"],
    "_judge_process_qa": _SdkGeminiVision.__dict__["_judge_process_qa"],
    "_observe_step": _SdkGeminiVision.__dict__["_observe_step"],
    "_budget_usage": _SdkGeminiVision.__dict__["_budget_usage"],
    "_metered_completion": _SdkGeminiVision.__dict__["_metered_completion"],
}
_TRUSTED_GEMINI_HELPERS = {
    name: _gemini_backend.__dict__[name]
    for name in (
        "_require_genai",
        "_gemini_key",
        "_provider_http_options",
        "_image_part",
        "_wrap_wav",
        "_audio_part",
        "_usage_of",
        "with_retry",
        "_parse_verdict",
        "_extract_json",
    )
}


@dataclass
class _SharedGeminiDispatchState:
    """One trusted lazy-client transition shared by every binding on an adapter."""

    settings: Any
    model: Any
    breaker: Any
    request_budget: Any
    expected_client: Any
    captured_provider: Any = None
    lock: Any = field(default_factory=threading.RLock, repr=False)
    binding_refs: list[Any] = field(default_factory=list, repr=False)

    def register(self, binding: AudioJudgeBinding) -> None:
        with self.lock:
            self.binding_refs = [ref for ref in self.binding_refs if ref() is not None]
            self.binding_refs.append(weakref.ref(binding))

    def has_live_bindings(self) -> bool:
        with self.lock:
            self.binding_refs = [ref for ref in self.binding_refs if ref() is not None]
            return bool(self.binding_refs)


_GEMINI_STATE_LOCK = threading.RLock()
_GEMINI_STATES: "weakref.WeakKeyDictionary[Any, _SharedGeminiDispatchState]" = (
    weakref.WeakKeyDictionary()
)


def _capture_gemini_client(client: Any) -> Any:
    """Expose only the provider callable captured from the accepted client."""
    generate = client.aio.models.generate_content
    if not callable(generate):
        raise RuntimeError("compiled Gemini audio client has no callable generate_content")

    class _CapturedModels:
        async def generate_content(self, *args: Any, **kwargs: Any) -> Any:
            return await generate(*args, **kwargs)

    return types.SimpleNamespace(
        aio=types.SimpleNamespace(models=_CapturedModels())
    )


def _shared_gemini_state(adapter: Any) -> _SharedGeminiDispatchState:
    settings = getattr(adapter, "_settings", None)
    model = getattr(adapter, "_model", None)
    breaker = getattr(adapter, "_breaker", None)
    request_budget = getattr(adapter, "_request_budget", None)
    client = getattr(adapter, "_client", None)
    with _GEMINI_STATE_LOCK:
        state = _GEMINI_STATES.get(adapter)
        if state is None:
            state = _SharedGeminiDispatchState(
                settings=settings,
                model=model,
                breaker=breaker,
                request_budget=request_budget,
                expected_client=client,
            )
            if client is not None:
                try:
                    state.captured_provider = _capture_gemini_client(client)
                except (AttributeError, RuntimeError):
                    # Inert placeholders remain apply-compatible and fail closed on dispatch.
                    pass
            _GEMINI_STATES[adapter] = state
        elif (
            state.settings is not settings
            or state.model != model
            or state.breaker is not breaker
            or state.request_budget is not request_budget
            or state.expected_client is not client
        ):
            if state.has_live_bindings():
                raise ValueError(
                    "compiled Gemini audio adapter state changed between Watch bindings"
                )
            # A prior plan may have prepared and rolled back without landing a binding. Such a
            # reservation must not poison a later valid application after ordinary adapter use.
            state = _SharedGeminiDispatchState(
                settings=settings,
                model=model,
                breaker=breaker,
                request_budget=request_budget,
                expected_client=client,
            )
            if client is not None:
                try:
                    state.captured_provider = _capture_gemini_client(client)
                except (AttributeError, RuntimeError):
                    pass
            _GEMINI_STATES[adapter] = state
        return state


def _assert_trusted_gemini_dispatch(
    adapter: Any,
    *,
    settings: Any,
    model: Any,
    breaker: Any,
    request_budget: Any,
    client: Any,
) -> None:
    """Fail closed if any dependency that can steer a captured Gemini call changed."""
    instance_dict = getattr(adapter, "__dict__", {})
    if any(
        name in instance_dict
        for name in ("judge", "_ensure_client", "_budget_usage", "_metered_completion")
    ):
        raise RuntimeError("compiled Gemini audio dispatch dependency was mutated")
    if _GEMINI_BASE.__dict__.get("_ensure_client") is not _TRUSTED_GEMINI_ENSURE_CLIENT:
        raise RuntimeError("compiled Gemini audio dispatch dependency was mutated")
    for name, implementation in _TRUSTED_GEMINI_METHODS.items():
        if _SdkGeminiVision.__dict__.get(name) is not implementation:
            raise RuntimeError("compiled Gemini audio dispatch dependency was mutated")
    for name, helper in _TRUSTED_GEMINI_HELPERS.items():
        if _gemini_backend.__dict__.get(name) is not helper:
            raise RuntimeError("compiled Gemini audio dispatch dependency was mutated")
    if (
        getattr(adapter, "_settings", None) is not settings
        or getattr(adapter, "_model", None) != model
        or getattr(adapter, "_breaker", None) is not breaker
        or getattr(adapter, "_request_budget", None) is not request_budget
        or getattr(adapter, "_client", None) is not client
    ):
        raise RuntimeError("compiled Gemini audio adapter state was mutated")


def _prepare_gemini_dispatch(
    adapter: Any,
) -> tuple[Callable[..., Awaitable[Any]], _SharedGeminiDispatchState]:
    """Capture Gemini state while permitting its one trusted lazy-client transition."""
    instance_dict = getattr(adapter, "__dict__", {})
    if "judge" in instance_dict or "_ensure_client" in instance_dict:
        raise ValueError("compiled audio rejects an instance-level Gemini dispatch shadow")
    state = _shared_gemini_state(adapter)
    settings = state.settings
    model = state.model
    breaker = state.breaker
    request_budget = state.request_budget

    # Validate before returning a binding as well as at call time. The sentinel client argument is
    # the current value here; after a trusted lazy initialization the closure advances it exactly
    # once to the SDK-created client.
    _assert_trusted_gemini_dispatch(
        adapter,
        settings=settings,
        model=model,
        breaker=breaker,
        request_budget=request_budget,
        client=state.expected_client,
    )

    def freeze_function(implementation: Any) -> Any:
        # An ordinary captured function still reads its module's live globals. Give the trusted
        # implementation a private globals table so a helper reassignment during an in-flight
        # provider await cannot change parsing or media construction when the coroutine resumes.
        frozen = types.FunctionType(
            implementation.__code__,
            dict(implementation.__globals__),
            name=implementation.__name__,
            argdefs=implementation.__defaults__,
            closure=implementation.__closure__,
        )
        frozen.__kwdefaults__ = implementation.__kwdefaults__
        return frozen

    snapshot = object.__new__(_SdkGeminiVision)
    snapshot._settings = settings
    snapshot._model = model
    snapshot._breaker = copy.copy(breaker)
    snapshot._request_budget = request_budget
    snapshot._client = None
    frozen_ensure = freeze_function(_TRUSTED_GEMINI_ENSURE_CLIENT)
    snapshot._ensure_client = types.MethodType(frozen_ensure, snapshot)
    for name in (
        "_judge_process_qa",
        "_observe_step",
        "_budget_usage",
        "_metered_completion",
    ):
        setattr(
            snapshot,
            name,
            types.MethodType(freeze_function(_TRUSTED_GEMINI_METHODS[name]), snapshot),
        )
    frozen_judge = types.MethodType(
        freeze_function(_TRUSTED_GEMINI_METHODS["judge"]),
        snapshot,
    )

    async def dispatch(*args: Any, **kwargs: Any) -> Any:
        # Lazy initialization is synchronous and serialized across every Watch binding sharing this
        # exact adapter. The provider await happens after releasing the lock.
        with state.lock:
            _assert_trusted_gemini_dispatch(
                adapter,
                settings=settings,
                model=model,
                breaker=breaker,
                request_budget=request_budget,
                client=state.expected_client,
            )
            if state.expected_client is None:
                initialized = _TRUSTED_GEMINI_ENSURE_CLIENT(adapter)
                if initialized is None or getattr(adapter, "_client", None) is not initialized:
                    raise RuntimeError(
                        "compiled Gemini audio lazy client initialization was invalid"
                    )
                state.expected_client = initialized
                state.captured_provider = _capture_gemini_client(initialized)
            elif state.captured_provider is None:
                state.captured_provider = _capture_gemini_client(
                    state.expected_client
                )
            provider = state.captured_provider
        snapshot._client = provider
        result = await frozen_judge(*args, **kwargs)
        with state.lock:
            _assert_trusted_gemini_dispatch(
                adapter,
                settings=settings,
                model=model,
                breaker=breaker,
                request_budget=request_budget,
                client=state.expected_client,
            )
        return result

    return dispatch, state


def _prepare_fake_dispatch(adapter: Any) -> Callable[..., Awaitable[Any]]:
    """Freeze the scripted table and audio-token algorithm for compiled Watches."""
    live_script = getattr(adapter, "_script", None)
    if type(live_script) is not dict or any(
        type(key) is not str or type(verdict) is not Verdict
        for key, verdict in live_script.items()
    ):
        raise ValueError("compiled FakeVision requires an exact scripted verdict table")
    script = copy.deepcopy(live_script)
    unknown = copy.deepcopy(
        _SdkFakeVision.__dict__["judge"].__globals__["_UNKNOWN"]
    )
    clone = copy.deepcopy
    sha256 = hashlib.sha256
    audio_frame_type = AudioFrame

    def frozen_audio_token(audio: Any) -> str:
        if type(audio) is not audio_frame_type:
            raise TypeError("compiled FakeVision audio requires an exact AudioFrame")
        if audio.encoded is not None:
            if not isinstance(audio.encoded, bytes):
                raise TypeError("compiled FakeVision encoded audio must be exact bytes")
            mime, raw = audio.mime or "audio/wav", audio.encoded
        else:
            if not isinstance(audio.pcm, bytes):
                raise TypeError("compiled FakeVision PCM audio must be exact bytes")
            base = "audio/L16" if audio.sample_format == "s16le" else "audio/pcm"
            mime = f"{base};rate={audio.sample_rate};channels={audio.channels}"
            raw = audio.pcm
        digest = sha256(mime.lower().encode("ascii") + b"\0" + raw).hexdigest()
        return f"audio:sha256:{digest}"

    async def dispatch(
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Any:
        token = (
            frozen_audio_token(audio)
            if frame is None and audio is not None
            else str(frame)
        )
        if context and context.get("observe_step"):
            verdict = script.get(
                f"{token}|step",
                type(unknown)(satisfied=False, confidence=0.5, evidence=""),
            )
        elif context and f"{token}|enriched" in script:
            verdict = script[f"{token}|enriched"]
        elif corrections and f"{token}|corrected" in script:
            verdict = script[f"{token}|corrected"]
        elif audio is not None and f"{token}|audio" in script:
            verdict = script[f"{token}|audio"]
        elif frames and f"{token}|strip" in script:
            verdict = script[f"{token}|strip"]
        else:
            verdict = script.get(token, unknown)
        return clone(verdict)

    return dispatch


def prepare_audio_judge_binding(
    adapter: Any,
    *,
    mode: AudioJudgeMode,
    application_key: str,
) -> AudioJudgeBinding:
    """Capture an exact SDK class implementation and bind it to the exact accepted instance."""
    adapter_type = type(adapter)
    if adapter_type not in _TRUSTED_AUDIO_JUDGES:
        raise ValueError(
            "compiled audio requires an exact SDK-owned FakeVision or GeminiVision adapter"
        )
    instance_dict = getattr(adapter, "__dict__", {})
    if "judge" in instance_dict:
        raise ValueError("compiled audio rejects an instance-level judge shadow")
    implementation = adapter_type.__dict__.get("judge")
    if implementation is not _TRUSTED_AUDIO_JUDGES[adapter_type]:
        raise ValueError("compiled audio rejects a class-level judge mutation")
    gemini_state = None
    if adapter_type is _SdkGeminiVision:
        dispatch, gemini_state = _prepare_gemini_dispatch(adapter)
    else:
        dispatch = _prepare_fake_dispatch(adapter)
    capability = AudioInputCapability(modes=frozenset({"audio_only", "fused"}))
    binding = AudioJudgeBinding(
        adapter=adapter,
        dispatch=dispatch,
        adapter_identity=f"{adapter_type.__module__}.{adapter_type.__qualname__}",
        mode=mode,
        capability=capability,
        application_key=application_key,
    )
    if gemini_state is not None:
        gemini_state.register(binding)
    return binding


__all__ = [
    "AudioInputCapability",
    "AudioJudgeBinding",
    "MAX_AUDIO_INPUT_BYTES",
    "MAX_AUDIO_PROVIDER_BYTES",
    "audio_frame_from_units",
    "audio_script_token",
    "audio_snippet_from_units",
    "canonicalize_audio_frame",
    "canonicalize_audio_ingress_frame",
    "canonicalize_audio_unit",
    "prepare_audio_judge_binding",
    "validate_audio_units_for_provider",
]
