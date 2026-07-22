"""Qualification runner — score an :class:`~percept.extraction.EvidenceExtractor` against a
labeled fixture set, producing a fixture-hashed :class:`~percept.catalog.QualificationRecord`.

:mod:`percept.catalog` defines ``QualificationRecord`` as "a MEASURED claim about one component,
stamped to the fixture that produced it" and says explicitly that qualifying a component against
a fixture is a later card's concern — this module is that later card, scoped to
:class:`~percept.extractors.ocr`'s extractors. It never touches
:class:`~percept.catalog.ComponentDescriptor`: the catalog's identity-only invariant (measured
performance is never a descriptor field) means a qualification run produces a record a caller
stores/reports separately, never something attached back onto the descriptor.

A fixture set is a plain sequence of :class:`OCRFixture` — a frame plus its ground-truth text (the
same frames a :class:`~percept.extractors.ocr.ScriptedOCR` script would use, or real rendered-text
images for :class:`~percept.extractors.ocr.TesseractOCR`). ``fixture_hash`` content-addresses the
EXACT fixture set used (:func:`percept.tracing.content_hash` over every fixture's plain-JSON
form) — change one fixture's expected text, or add/remove a fixture, and the hash changes; the
component identity (``component_name``) never does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..catalog import ROLE_BACKEND, QualificationRecord
from ..contracts import Box
from ..extraction import EvidenceExtractor
from ..tracing import content_hash


@dataclass(frozen=True)
class OCRFixture:
    """One labeled fixture: a frame + the ground-truth text an extractor should read from it.

    - ``frame`` — whatever the extractor under test's ``.extract()`` accepts (a
      :class:`~percept.extractors.ocr.ScriptedOCR` token string, or a real image path/array for
      :class:`~percept.extractors.ocr.TesseractOCR`).
    - ``expected_text`` — checked as a case-insensitive substring of the extractor's
      ``facts["text"]`` (not exact match — real OCR output spacing/casing varies even on a
      correct read).
    - ``condition`` / ``roi`` — passed through to ``.extract()`` unchanged.
    """

    frame: Any
    expected_text: str
    condition: str = "document text visible"
    roi: Box | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "expected_text": self.expected_text,
            "condition": self.condition,
            "roi": None if self.roi is None else
                {"x": self.roi.x, "y": self.roi.y, "w": self.roi.w, "h": self.roi.h},
        }


def qualify_extractor(
    extractor: EvidenceExtractor,
    fixtures: Sequence[OCRFixture],
    *,
    component_name: str = "ocr",
    notes: str = "",
    measured_at: float | None = None,
) -> QualificationRecord:
    """Run ``extractor`` over every ``fixture`` and return a fixture-hashed
    :class:`~percept.catalog.QualificationRecord` for it.

    Metrics (plain fractions, ``0..1``):

    - ``recall`` — fraction of fixtures whose ``expected_text`` was found (case-insensitive
      substring) in the extractor's read.
    - ``miss_rate`` — fraction of fixtures the extractor returned an empty
      :class:`~percept.extraction.Enrichment` for (found nothing at all).
    - ``mean_confidence`` — average of every non-``None`` self-reported
      ``Enrichment.confidence`` across the run (``0.0`` when the extractor never reports one).
    """
    if not fixtures:
        raise ValueError("qualify_extractor needs at least one fixture")

    hits = 0
    misses = 0
    confidences: list[float] = []
    for fx in fixtures:
        enrichment = extractor.extract(fx.condition, fx.frame, fx.roi)
        if enrichment.is_empty:
            misses += 1
            continue
        text = str(enrichment.facts.get("text", ""))
        if fx.expected_text.lower() in text.lower():
            hits += 1
        if enrichment.confidence is not None:
            confidences.append(enrichment.confidence)

    total = len(fixtures)
    metrics = {
        "recall": hits / total,
        "miss_rate": misses / total,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
    }
    fixture_hash = content_hash([fx.to_dict() for fx in fixtures])
    return QualificationRecord(
        component_role=ROLE_BACKEND,
        component_name=component_name,
        fixture_hash=fixture_hash,
        metrics=metrics,
        measured_at=measured_at,
        notes=notes,
    )


__all__ = ["OCRFixture", "qualify_extractor"]
