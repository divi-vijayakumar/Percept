"""EvidenceExtractor — the extraction seam admission was never allowed to be.

``percept.contracts.Enricher`` was removed as a REGISTRY: a detector decides WHEN to look
closely (admission); nothing in the runtime ever ran a registered enricher to turn that moment
into structured facts, so the seam was dead weight. That left ``Vision.judge(context={...})`` as
the only live door for structured facts, and it required the caller to remember to open it by
hand on every call.

This module is a real, greenfield extraction seam that closes the loop honestly: a client
registers an :class:`EvidenceExtractor` for a source, the brain calls it at the SAME point it
already calls the judge, and its :class:`Enrichment` is merged into ``context=`` under a
RESERVED sub-dict key (``context["extractor"]``) — visible to the judge and recorded to the trace
as its own stage, never silently folded into free-text evidence. "Admission is not extraction"
now has a mechanism, not just a doc section: a detector's boundary answers WHEN; an extractor's
:meth:`EvidenceExtractor.extract` answers WHAT the frame/roi structurally says, before the judge
is asked to reason over it.

What this module is NOT: it does not decide whether a frame is worth judging (that is still
admission — a detector or a tier-0 adapter) and it does not replace the judge's own reasoning —
:class:`Enrichment.facts` are handed to the judge as CONTEXT, exactly like a caller's own
``context={...}`` today; a backend that ignores ``context`` degrades gracefully, same as always.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import Box

#: The reserved ``context`` sub-dict key an :class:`Enrichment` is merged under when it reaches
#: the judge (``context={"extractor": enrichment.to_dict()}``) — mirrors the existing
#: ``context["process_qa"]`` / ``context["observe_step"]`` reserved-key convention in
#: ``percept.agent``. A caller MUST NOT put unrelated data under this key.
CONTEXT_KEY = "extractor"


@dataclass(frozen=True)
class Enrichment:
    """Structured facts one :class:`EvidenceExtractor` call produced for one frame/roi.

    - ``facts`` — the JSON-shaped dict handed to the judge as context (e.g. an OCR read's
      ``{"invoice_number": "INV-2291"}``). Read as authoritative by convention, distinct from a
      user-asserted ``corrections`` value; a judge that ignores it degrades gracefully.
    - ``evidence_refs`` — opaque handles (see ``percept.evidence.EvidenceArtifact.ref``) into
      whatever artifact grounded these facts, so a fact is traceable back to the pixels/audio it
      came from. Empty when the extractor stored nothing separately.
    - ``ts_range`` — the ``(start, end)`` production-domain window these facts describe, or
      ``None`` for an instantaneous single-frame read.
    - ``confidence`` — optional, ``0..1``; ``None`` when the extractor does not self-report one
      (never invented here).
    """

    facts: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    ts_range: tuple[float, float] | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", dict(self.facts))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        if self.confidence is not None and not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"Enrichment.confidence must be in 0..1 or None, got {self.confidence!r}")
        if self.ts_range is not None:
            start, end = self.ts_range
            if end < start:
                raise ValueError(f"Enrichment.ts_range end must be >= start, got {self.ts_range!r}")

    @property
    def is_empty(self) -> bool:
        """``True`` when this extractor call found nothing worth telling the judge — the common
        no-op case (e.g. an OCR extractor over a frame with no text)."""
        return not self.facts

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": dict(self.facts),
            "evidence_refs": list(self.evidence_refs),
            "ts_range": list(self.ts_range) if self.ts_range is not None else None,
            "confidence": self.confidence,
        }


@runtime_checkable
class EvidenceExtractor(Protocol):
    def extract(self, condition: str, frame: Any, roi: Box | None = None) -> Enrichment:
        """Read structured facts out of ``frame`` (optionally cropped to ``roi``) for ``condition``.

        MUST NOT raise on a normal miss — return ``Enrichment()`` (empty facts) instead, the same
        no-raise-on-a-miss contract the removed ``Enricher.enrich`` documented. ``condition`` is
        the concern's plain-language anchor, handed through unchanged so an extractor may key its
        behaviour off it (e.g. only OCR when the condition mentions text)."""
        ...


__all__ = ["CONTEXT_KEY", "Enrichment", "EvidenceExtractor"]
