"""OCR — the first REAL :class:`~percept.extraction.EvidenceExtractor`.

Reads structured text facts out of a frame (optionally cropped to a document ``roi``) and hands
them to the judge as ``context["extractor"]`` — see :mod:`percept.extraction` for the seam this
plugs into. Two backends, same :class:`~percept.extraction.EvidenceExtractor` shape:

- :class:`ScriptedOCR` — deterministic, offline, zero dependencies. Looks a frame up in an
  injected script by exact token (the same convention ``percept.backends.fake.ScriptedExtractor``
  and ``FakeVision`` use), falling back to a stable content-hash key for non-string frames so a
  caller can drive it off real frame content without hand-maintaining a token map. This is what
  CI and every offline test in this SDK use — it never spends an API call and never needs the
  ``tesseract`` binary or a vision dependency installed.
- :class:`TesseractOCR` — a real backend over ``pytesseract`` + ``Pillow``, both OPTIONAL
  (``percept-vision[ocr]``). Neither is imported at module top level (invariant: the core never
  imports a vendor/optional dependency at import time — mirrors
  :func:`percept.backends.anthropic._require_anthropic`); calling :meth:`TesseractOCR.extract`
  without them installed raises a clear, typed ``RuntimeError`` naming the extra to install and
  the offline fallback to use instead of crashing on an opaque ``ModuleNotFoundError``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from ..contracts import Box
from ..extraction import Enrichment
from ..tracing import content_hash


def _frame_key(frame: Any) -> str:
    """A deterministic lookup key for ``frame``: itself when already a string token (the
    ``FakeVision``/``ScriptedExtractor`` token convention), otherwise a stable content hash of its
    ``repr`` — so a caller can script :class:`ScriptedOCR` against real (non-string) frame content
    without maintaining a hand-written token map. NEVER used for the real ``TesseractOCR`` backend
    (which reads the frame's actual pixels)."""
    if isinstance(frame, str):
        return frame
    return content_hash(repr(frame))


class ScriptedOCR:
    """Deterministic, offline OCR double — no ``tesseract``, no model, no randomness.

    ``script`` maps a frame token (or, for a non-string frame, its :func:`_frame_key` content
    hash) to either a ready-made :class:`~percept.extraction.Enrichment` or a plain dict of
    ``{"text": ..., "regions": [...], "confidence": ...}``. A frame with no script entry returns
    an empty ``Enrichment`` (the standard no-raise-on-a-miss contract every
    :class:`~percept.extraction.EvidenceExtractor` follows) — this proves the extraction seam's
    WIRING for OCR-shaped facts, never OCR intelligence.
    """

    def __init__(self, script: "Mapping[str, Any] | None" = None) -> None:
        self._script: dict[str, Any] = dict(script or {})

    def extract(self, condition: str, frame: Any, roi: Box | None = None) -> Enrichment:
        hit = self._script.get(_frame_key(frame))
        if hit is None:
            return Enrichment()
        if isinstance(hit, Enrichment):
            return hit
        facts = {"text": hit.get("text", ""), "regions": list(hit.get("regions", ()))}
        return Enrichment(
            facts=facts,
            evidence_refs=tuple(hit.get("evidence_refs", ())),
            ts_range=hit.get("ts_range"),
            confidence=hit.get("confidence"),
        )


def _require_ocr_backend():
    """Lazy-import ``pytesseract`` + ``PIL.Image`` with a clear, typed error if either is absent.

    Uses ``importlib.import_module`` (rather than a bare ``import`` statement) so the miss path is
    deterministic and easy to exercise from a test without needing the real packages absent from
    the environment.
    """
    try:
        pytesseract = import_module("pytesseract")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TesseractOCR needs the 'pytesseract' package (and the 'tesseract' binary on PATH) — "
            "install percept-vision[ocr]. Use the 'ocr-scripted' extractor for offline/CI runs."
        ) from exc
    try:
        image_mod = import_module("PIL.Image")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TesseractOCR needs the 'Pillow' package — install percept-vision[ocr]. Use the "
            "'ocr-scripted' extractor for offline/CI runs."
        ) from exc
    return pytesseract, image_mod


def _to_pil_image(frame: Any, image_mod: Any) -> Any:
    """Coerce ``frame`` to a ``PIL.Image`` — accepts an already-decoded image, a filesystem path,
    or anything exposing the numpy array protocol (``__array_interface__``/``__array__``, e.g. an
    ``opencv``/``numpy`` frame)."""
    if isinstance(frame, image_mod.Image):
        return frame
    if isinstance(frame, str):
        return image_mod.open(frame)
    if hasattr(frame, "__array_interface__") or hasattr(frame, "__array__"):
        return image_mod.fromarray(frame)
    raise TypeError(
        f"TesseractOCR.extract cannot read frame of type {type(frame).__name__!r} — expected a "
        "PIL.Image, a filesystem path, or an array-like (e.g. a numpy frame)."
    )


def _crop_roi(image: Any, roi: Box) -> Any:
    """Crop ``image`` to ``roi`` (normalized 0..1 coordinates, see ``percept.contracts.Box``)."""
    width, height = image.size
    left = max(0, int(roi.x * width))
    top = max(0, int(roi.y * height))
    right = min(width, int((roi.x + roi.w) * width))
    bottom = min(height, int((roi.y + roi.h) * height))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


class TesseractOCR:
    """Real OCR extractor over document ROIs — ``pytesseract`` + ``Pillow`` (optional extra
    ``percept-vision[ocr]``). Reads ``context["extractor"]`` facts ``{"text": <joined line text>,
    "regions": [{"text", "box", "confidence"}, ...]}``, with a self-reported overall
    ``Enrichment.confidence`` averaged over every word ``tesseract`` scored (``0..1``, ``None``
    when it found nothing).

    Constructing this class never imports ``pytesseract``/``Pillow`` — only
    :meth:`extract` does, so ``registry.make_extractor("ocr")`` always succeeds; the typed error
    surfaces at first actual use, exactly like every other optional-dependency backend in this
    package.
    """

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings

    def extract(self, condition: str, frame: Any, roi: Box | None = None) -> Enrichment:
        pytesseract, image_mod = _require_ocr_backend()
        image = _to_pil_image(frame, image_mod)
        if roi is not None:
            image = _crop_roi(image, roi)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        text_parts: list[str] = []
        regions: list[dict[str, Any]] = []
        confidences: list[float] = []
        for i, raw_text in enumerate(data.get("text", [])):
            text = raw_text.strip()
            if not text:
                continue
            conf: float | None
            try:
                raw_conf = float(data["conf"][i])
                conf = raw_conf / 100.0 if raw_conf >= 0 else None
            except (KeyError, IndexError, TypeError, ValueError):
                conf = None
            text_parts.append(text)
            regions.append({
                "text": text,
                "box": {
                    "x": data["left"][i], "y": data["top"][i],
                    "w": data["width"][i], "h": data["height"][i],
                },
                "confidence": conf,
            })
            if conf is not None:
                confidences.append(conf)

        if not text_parts:
            return Enrichment()
        overall_confidence = (sum(confidences) / len(confidences)) if confidences else None
        return Enrichment(
            facts={"text": " ".join(text_parts), "regions": regions},
            confidence=overall_confidence,
        )


__all__ = ["ScriptedOCR", "TesseractOCR"]
