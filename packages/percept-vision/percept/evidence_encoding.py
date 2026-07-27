"""Faithful evidence bytes — the ONE encoder every path that ships evidence off the machine uses.

An evidence artifact must carry the bytes it claims to carry. The failure this module exists to make
structurally impossible is SUBSTITUTION: a frame that could not be encoded silently replaced by a
*description* of itself (``repr(frame)``) while the artifact keeps advertising an evidence media
type. That substitution is not merely lossy — numpy elides the middle of any array over 1000
elements, so a 64x64x3 frame carrying 12,288 pixel bytes becomes a 1,034-character string with an
``...`` in the middle, and no consumer can tell it from real evidence.

There are exactly two outcomes here, and no third:

* :func:`encode_evidence_frame` returns ``(media_type, bytes)`` carrying the frame's REAL bytes —
  passthrough for an already-encoded payload, lossless PNG for a decoded pixel buffer; or
* it raises :class:`EvidenceNotEncodable`, which the caller records as an explicit, typed
  "unavailable" marker (:func:`unavailable_marker` / :func:`unavailable_artifact`). A marker carries
  no bytes and no rendering of the frame's contents at all — only WHAT was not carried and WHY — so a
  consumer cannot mistake it for evidence, and a payload that is missing evidence says so.

Pixel buffers are encoded LOSSLESSLY (PNG via stdlib ``zlib``): the encoded bytes decode back to the
exact buffer that was captured, so an audit can hash retained evidence against the pixels it claims
to be. The encoder is deterministic and needs no optional extra, so the guarantee holds in the
minimal install exactly as it does under ``percept-vision[runtime]``. Channel bytes are written in
the order the buffer supplies them: the encoder never reinterprets a colour convention it was not
told about, which is what makes the round-trip an identity.

Pure stdlib.
"""
from __future__ import annotations

import base64
import binascii
import json
import struct
import zlib
from typing import Any, NoReturn

from .events import frame_nbytes

#: Media type of the explicit "this frame could not be carried" artifact — deliberately NOT an image
#: or bundle type, so a store/consumer reading media types alone already knows this is a gap record.
UNAVAILABLE_MEDIA_TYPE = "application/x-percept-evidence-unavailable+json"

#: channels -> PNG colour type (1 grey, 2 grey+alpha, 3 RGB, 4 RGBA). Anything else is not an image
#: this encoder can write faithfully, so it becomes an explicit gap rather than a reshaped guess.
_PNG_COLOR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}

#: Magic-number sniff for a bare byte payload that arrived without a declared media type. Only used
#: to LABEL bytes that are carried verbatim either way; an unrecognised payload is carried as
#: ``application/octet-stream``, never dropped and never relabelled to something it is not.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
)


class EvidenceNotEncodable(TypeError):
    """A frame cannot be carried as evidence without substituting something else for it.

    Typed so a caller can distinguish "there is no faithful encoding of this object" from an ordinary
    bug, and record the former as an explicit gap. ``reason`` is a stable, machine-readable token;
    ``detail`` is optional metadata ABOUT the frame (a dtype, a shape, a byte count) — never a
    rendering of its contents."""

    def __init__(
        self,
        reason: str,
        *,
        frame_type: str,
        detail: str | None = None,
        source_nbytes: int = 0,
    ) -> None:
        super().__init__(
            f"{reason} [{frame_type}]" if detail is None else f"{reason} [{frame_type}: {detail}]"
        )
        self.reason = reason
        self.frame_type = frame_type
        self.detail = detail
        self.source_nbytes = int(source_nbytes)

    def marker(self) -> dict[str, Any]:
        """JSON-safe record of what was NOT carried and why. Contains no frame contents."""
        return {
            "reason": self.reason,
            "frame_type": self.frame_type,
            "detail": self.detail,
            "source_nbytes": self.source_nbytes,
        }


def unavailable_marker(exc: EvidenceNotEncodable) -> dict[str, Any]:
    """The bundle ENTRY for a frame that could not be carried: ``{"unavailable": {...}}``.

    Structurally distinct from an evidence entry (which carries ``mime`` + ``data``), so a consumer
    that reads entries by key can never read a gap as evidence."""
    return {"unavailable": exc.marker()}


def unavailable_artifact(exc: EvidenceNotEncodable) -> tuple[str, bytes]:
    """The stored ARTIFACT for a frame that could not be carried — ``(media_type, bytes)``.

    The same gap record as :func:`unavailable_marker`, in the shape a byte store takes, under
    :data:`UNAVAILABLE_MEDIA_TYPE` so the gap is visible from the media type alone."""
    return UNAVAILABLE_MEDIA_TYPE, json.dumps({"unavailable": exc.marker()}).encode("utf-8")


def _fail(frame: Any, reason: str, detail: str | None = None) -> NoReturn:
    raise EvidenceNotEncodable(
        reason,
        frame_type=type(frame).__name__,
        detail=detail,
        source_nbytes=frame_nbytes(frame),
    )


def _sniff(payload: bytes) -> str:
    for magic, media_type in _MAGIC:
        if payload.startswith(magic):
            return media_type
    return "application/octet-stream"


def png_bytes(width: int, height: int, channels: int, raw: bytes) -> bytes:
    """Encode a raw 8-bit pixel buffer as a LOSSLESS PNG. ``raw`` is row-major, ``width*channels``
    bytes per row, channel bytes in the order given (no colour-space reinterpretation)."""
    color_type = _PNG_COLOR_TYPE[channels]
    stride = width * channels
    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)  # per-row filter 0 (None) — deterministic, no predictor state
        scanlines += raw[row * stride:(row + 1) * stride]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(scanlines), 6))
        + chunk(b"IEND", b"")
    )


def _encode_array(frame: Any) -> tuple[str, bytes]:
    """Lossless PNG for a decoded 8-bit pixel buffer (numpy ndarray or any buffer exposing
    ``shape``/``dtype``/``tobytes``). Raises :class:`EvidenceNotEncodable` for anything whose exact
    contents this encoder cannot round-trip."""
    if str(getattr(frame, "dtype", "")) != "uint8":
        _fail(frame, "unsupported-array-dtype", str(getattr(frame, "dtype", "")))
    try:
        shape = tuple(int(dim) for dim in frame.shape)
    except (TypeError, ValueError):
        _fail(frame, "unsupported-array-shape", str(getattr(frame, "shape", None)))
    if len(shape) == 2:
        height, width, channels = shape[0], shape[1], 1
    elif len(shape) == 3:
        height, width, channels = shape
    else:
        _fail(frame, "unsupported-array-shape", str(shape))
    if height <= 0 or width <= 0 or channels not in _PNG_COLOR_TYPE:
        _fail(frame, "unsupported-array-shape", str(shape))
    raw = frame.tobytes()
    if len(raw) != height * width * channels:
        _fail(frame, "array-buffer-size-mismatch", f"{len(raw)}!={height * width * channels}")
    return "image/png", png_bytes(width, height, channels, raw)


def _encode_grid_frame(frame: Any) -> tuple[str, bytes]:
    """Lossless grey PNG for the ``w``/``h``/``pixels`` frame shape (``percept.harness.tier0.Frame``,
    the gate signal frame). Out-of-range or short pixel data is a gap, never a clamped guess."""
    width, height = int(frame.w), int(frame.h)
    pixels = frame.pixels
    if width <= 0 or height <= 0:
        _fail(frame, "unsupported-frame-size", f"{width}x{height}")
    try:
        raw = bytes(pixels)
    except (TypeError, ValueError):
        _fail(frame, "unsupported-frame-pixels")
    if len(raw) != width * height:
        _fail(frame, "frame-pixel-count-mismatch", f"{len(raw)}!={width * height}")
    return "image/png", png_bytes(width, height, 1, raw)


def encode_evidence_frame(frame: Any) -> tuple[str, bytes]:
    """Return ``(media_type, bytes)`` carrying ``frame``'s REAL bytes, or raise.

    Accepted, in order: an already-encoded ``(media_type, bytes)`` pair (passed through byte-for-byte,
    the shape the runtime's own capture path publishes); bare ``bytes``/``bytearray``/``memoryview``
    (carried verbatim, media type sniffed); the provider image-block dict
    ``{"type": "image", "source": {"media_type", "data"}}`` (base64 decoded); a decoded 8-bit pixel
    buffer (lossless PNG); the ``w``/``h``/``pixels`` grid frame (lossless grey PNG); a PIL-style
    image (lossless PNG through its own writer).

    Anything else raises :class:`EvidenceNotEncodable`. This function NEVER returns a description,
    summary, thumbnail or other stand-in for a frame it could not encode."""
    if (
        isinstance(frame, tuple)
        and len(frame) == 2
        and isinstance(frame[0], str)
        and isinstance(frame[1], (bytes, bytearray))
    ):
        return frame[0], bytes(frame[1])
    if isinstance(frame, (bytes, bytearray, memoryview)):
        payload = bytes(frame)
        return _sniff(payload), payload
    if isinstance(frame, dict):
        if frame.get("type") != "image" or not isinstance(frame.get("source"), dict):
            _fail(frame, "unencodable-frame-type")
        source = frame["source"]
        data = source.get("data")
        if isinstance(data, (bytes, bytearray)):
            return str(source.get("media_type") or "application/octet-stream"), bytes(data)
        if isinstance(data, str):
            try:
                payload = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                _fail(frame, "undecodable-image-block")
            return str(source.get("media_type") or "application/octet-stream"), payload
        _fail(frame, "unencodable-frame-type")
    if hasattr(frame, "shape") and hasattr(frame, "dtype") and callable(getattr(frame, "tobytes", None)):
        return _encode_array(frame)
    if (
        isinstance(getattr(frame, "w", None), int)
        and isinstance(getattr(frame, "h", None), int)
        and getattr(frame, "pixels", None) is not None
    ):
        return _encode_grid_frame(frame)
    if callable(getattr(frame, "save", None)) and getattr(frame, "mode", None) is not None:
        import io  # noqa: PLC0415 — only needed on the PIL branch

        buffer = io.BytesIO()
        try:
            frame.save(buffer, format="PNG")
        except Exception:  # noqa: BLE001 — a writer failure is a gap, never a substitution
            _fail(frame, "image-writer-failed")
        payload = buffer.getvalue()
        if not payload:
            _fail(frame, "image-writer-produced-no-bytes")
        return "image/png", payload
    _fail(frame, "unencodable-frame-type")


__all__ = [
    "UNAVAILABLE_MEDIA_TYPE",
    "EvidenceNotEncodable",
    "encode_evidence_frame",
    "png_bytes",
    "unavailable_artifact",
    "unavailable_marker",
]
