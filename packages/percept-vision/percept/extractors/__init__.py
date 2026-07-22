"""Real ``percept.extraction.EvidenceExtractor`` implementations.

``percept.extraction`` defines the seam (the ``Protocol``, the ``Enrichment`` DTO, the reserved
``context["extractor"]`` key); ``percept.backends.fake.ScriptedExtractor`` proves the seam's
WIRING with an offline double. This package holds the first REAL extractor — OCR
(:mod:`percept.extractors.ocr`) — plus a fixture-based qualification runner
(:mod:`percept.extractors.qualification`) that scores one against a labeled fixture set and
produces a fixture-hashed ``percept.catalog.QualificationRecord``.

Nothing here is imported at package top level beyond stdlib — the optional OCR backend
(``pytesseract`` + ``Pillow``) is lazy-imported inside :meth:`percept.extractors.ocr.TesseractOCR.extract`
only, so ``import percept`` (and every offline/CI test) never needs it installed (the same
invariant :mod:`percept.backends.anthropic` documents for the vision seam: the core never imports
a vendor SDK at module top level).
"""

from __future__ import annotations

__all__: list[str] = []
