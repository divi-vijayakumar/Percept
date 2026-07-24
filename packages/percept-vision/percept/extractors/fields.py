"""Fields — typed field-comparison extraction: numeric/currency/date/frequency mentions in
OCR-style text, normalized to a CANONICAL string so a fused judge compares VALUES, never raw
surface text.

Motivating incident: a client's fused compare ("spoken threshold vs visible threshold") false-
fired a mismatch at confidence 1.0 on IDENTICAL values — the spoken side read ``"25,000 dollars"``,
the visible side read ``"threshold: 25,000 dollars"``, and an unstructured text compare treated the
differing SURFACE FORM (label present vs absent, no currency-code normalization) as a semantic
mismatch. This module closes that gap the same way :mod:`percept.extractors.ocr` closed "admission
is not extraction": a typed :class:`~percept.extraction.EvidenceExtractor` that turns a text
mention into ``{"field", "raw", "normalized", "kind"}`` BEFORE the judge ever sees it, so a
correctly-instructed condition compares two ``normalized`` strings instead of two ``raw`` ones.

Two backends, same shape as :mod:`percept.extractors.ocr`:

- :class:`FieldNormalizer` — the REAL backend. Unlike ``TesseractOCR`` it needs no vendor
  dependency: turning already-recognized text into typed field mentions is pure regex/format
  logic, not an OCR/ASR problem, so it is deterministic and always available. It reads ``frame`` as
  TEXT — a plain ``str``, or a ``Mapping`` exposing a ``"text"`` key (the composition point with an
  OCR extractor's own ``Enrichment.facts["text"]`` when chained, see :func:`chain_ocr_then_fields`).
- :class:`ScriptedFields` — deterministic, offline double for CI, scripted by frame token exactly
  like :class:`~percept.extractors.ocr.ScriptedOCR` (same :func:`~percept.extractors.ocr._frame_key`
  convention: a string frame is its own key; anything else keys off a stable content hash).

``Enrichment.facts`` shape (see :mod:`percept.extraction`): ``{"fields": [{"field": <label or
None>, "raw": <as-seen>, "normalized": <canonical str or None>, "kind": "currency"|"number"|
"date"|"frequency"|"unknown"}, ...]}``.

**NEVER GUESS.** An unparseable or genuinely ambiguous mention (a currency symbol/word this module
does not resolve unambiguously, e.g. ``"kr"`` — Swedish vs Norwegian vs Danish krona/krone; a
numeric date written ``MM/DD`` vs ``DD/MM`` with no way to tell which) keeps its ``kind`` (when the
SHAPE was recognized, e.g. "this was clearly currency-shaped") but sets ``normalized=None`` rather
than inventing a canonical form. A mention this module cannot classify at all gets
``kind="unknown"``, ``normalized=None``. A judge condition MUST treat ``normalized=None`` on either
side of a compare as "cannot confirm equal or different" (ASK), never as grounds for a confident
mismatch.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Mapping

from ..contracts import Box
from ..extraction import Enrichment
from .ocr import _frame_key

# ---- normalization vocabularies ---------------------------------------------------------------

_ISO_CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY"}
_CURRENCY_SYMBOL_MAP = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
_CURRENCY_WORD_MAP = {
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
    "yen": "JPY", "jpy": "JPY",
}
# Genuinely ambiguous currency words this module refuses to resolve on its own (multiple live
# ISO currencies share the same word/abbreviation) -- proves the "keep raw, normalized=None" path.
_AMBIGUOUS_CURRENCY_WORDS = {"kr", "kr.", "krona", "krone", "krones"}

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_FREQUENCY_CANON = {
    "daily": "1/day", "everyday": "1/day", "every day": "1/day",
    "per day": "1/day", "a day": "1/day", "/day": "1/day",
    "weekly": "1/week", "per week": "1/week", "a week": "1/week", "/week": "1/week",
    "monthly": "1/month", "per month": "1/month", "a month": "1/month", "/month": "1/month",
    "quarterly": "1/quarter", "per quarter": "1/quarter", "a quarter": "1/quarter",
    "yearly": "1/year", "annually": "1/year", "per year": "1/year", "a year": "1/year",
    "/year": "1/year",
}

# ---- shared regex primitives --------------------------------------------------------------------

_NUM = r"[\d][\d,]*(?:\.\d+)?"
_CURRENCY_ISO_RE = re.compile(rf"^(?P<code>[A-Za-z]{{3}})\s+(?P<num>{_NUM})$")
_CURRENCY_SYMBOL_RE = re.compile(rf"^(?P<sym>[$€£¥])\s*(?P<num>{_NUM})$")
_CURRENCY_WORD_RE = re.compile(rf"^(?P<num>{_NUM})\s+(?P<word>[A-Za-z.]+)$")
_PLAIN_NUMBER_RE = re.compile(rf"^{_NUM}$")

_ISO_DATE_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})$")
_MONTH_DAY_YEAR_RE = re.compile(r"^(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})$")
_DAY_MONTH_YEAR_RE = re.compile(r"^(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\s+(?P<year>\d{4})$")
# "MM/DD/YYYY" vs "DD/MM/YYYY" -- genuinely ambiguous without a locale hint; never guessed.
_AMBIGUOUS_NUMERIC_DATE_RE = re.compile(r"^(?P<a>\d{1,2})[/.](?P<b>\d{1,2})[/.](?P<c>\d{2,4})$")

_LABEL_RE = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9 _/\-]{0,60}?)\s*:\s*(?P<value>.+)$")


def _format_amount(num: str) -> str:
    return f"{float(num.replace(',', '')):.2f}"


def _try_currency(value: str) -> "tuple[str, str | None] | None":
    v = value.strip()
    m = _CURRENCY_ISO_RE.match(v)
    if m and m.group("code").upper() in _ISO_CURRENCY_CODES:
        return "currency", f"{m.group('code').upper()} {_format_amount(m.group('num'))}"
    m = _CURRENCY_SYMBOL_RE.match(v)
    if m:
        code = _CURRENCY_SYMBOL_MAP[m.group("sym")]
        return "currency", f"{code} {_format_amount(m.group('num'))}"
    m = _CURRENCY_WORD_RE.match(v)
    if m:
        word = m.group("word").lower()
        if word in _CURRENCY_WORD_MAP:
            return "currency", f"{_CURRENCY_WORD_MAP[word]} {_format_amount(m.group('num'))}"
        if word in _AMBIGUOUS_CURRENCY_WORDS:
            return "currency", None  # currency-shaped, but the word doesn't resolve unambiguously
    return None


def _try_date(value: str) -> "tuple[str, str | None] | None":
    v = value.strip()
    m = _ISO_DATE_RE.match(v)
    if m:
        try:
            date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        except ValueError:
            return "date", None
        return "date", v
    m = _MONTH_DAY_YEAR_RE.match(v)
    if m and m.group("month").lower() in _MONTHS:
        try:
            d = date(int(m.group("year")), _MONTHS[m.group("month").lower()], int(m.group("day")))
        except ValueError:
            return "date", None
        return "date", d.isoformat()
    m = _DAY_MONTH_YEAR_RE.match(v)
    if m and m.group("month").lower() in _MONTHS:
        try:
            d = date(int(m.group("year")), _MONTHS[m.group("month").lower()], int(m.group("day")))
        except ValueError:
            return "date", None
        return "date", d.isoformat()
    if _AMBIGUOUS_NUMERIC_DATE_RE.match(v):
        return "date", None  # date-shaped, MM/DD vs DD/MM ambiguity -> never guessed
    return None


def _try_frequency(value: str) -> "tuple[str, str | None] | None":
    v = value.strip().lower()
    if v in _FREQUENCY_CANON:
        return "frequency", _FREQUENCY_CANON[v]
    return None


def _try_number(value: str) -> "tuple[str, str | None] | None":
    v = value.strip()
    if _PLAIN_NUMBER_RE.match(v):
        f = float(v.replace(",", ""))
        return "number", (str(int(f)) if f.is_integer() else str(f))
    return None


def classify_value(value: str) -> "tuple[str, str | None]":
    """Classify one already-isolated value mention: ``(kind, normalized)``. Tries currency, then
    date, then frequency, then plain number, first shape match wins; ``("unknown", None)`` when
    nothing recognizes it. NEVER raises."""
    for classifier in (_try_currency, _try_date, _try_frequency, _try_number):
        result = classifier(value)
        if result is not None:
            return result
    return "unknown", None


def _segments(text: str) -> list[str]:
    """Split OCR-style multi-field text into per-mention segments on newline/``;`` boundaries —
    deliberately NOT on commas (a currency value like ``"25,000"`` needs its comma intact). A
    single-line/bare mention (e.g. a spoken transcript with no label) is its own one-segment text."""
    return [p.strip() for p in re.split(r"[\n;]+", text) if p.strip()]


def extract_fields(text: str) -> list[dict[str, Any]]:
    """Parse ``text`` into typed field dicts: ``{"field", "raw", "normalized", "kind"}``.

    A ``"<label>: <value>"`` segment splits into ``field=<label lowercased>`` / ``raw=<value>`` and
    ALWAYS emits an entry (a label is an explicit assertion of a field, so even an unclassifiable
    value is reported with ``kind="unknown"`` rather than dropped). A bare segment (no ``:``) is
    ``field=None`` with the whole segment as ``raw`` — the shape a spoken/transcript mention
    naturally has (no label is ever spoken) — but is only emitted when :func:`classify_value`
    recognizes a numeric/currency/date/frequency SHAPE in it (``kind != "unknown"``); unlabeled
    prose with no recognizable value shape at all is not a field mention and is silently skipped,
    never reported as a fabricated ``"unknown"`` field. NEVER raises and NEVER guesses: an
    unparseable/ambiguous value keeps its (possibly ``"unknown"``) ``kind`` with
    ``normalized=None`` rather than inventing a canonical form.
    """
    fields: list[dict[str, Any]] = []
    for segment in _segments(text):
        m = _LABEL_RE.match(segment)
        if m:
            label = m.group("label").strip().lower()
            raw = m.group("value").strip()
            kind, normalized = classify_value(raw)
        else:
            label = None
            raw = segment
            kind, normalized = classify_value(raw)
            if kind == "unknown":
                continue  # unlabeled prose with no recognizable value shape is not a field mention
        fields.append({"field": label, "raw": raw, "normalized": normalized, "kind": kind})
    return fields


def _frame_text(frame: Any) -> str:
    """Read ``frame`` as text: a plain ``str`` frame directly, a ``Mapping`` frame via its
    ``"text"`` key (the composition point with an OCR extractor's ``Enrichment.facts``), anything
    else as no text at all (never a crash, never a guess)."""
    if isinstance(frame, str):
        return frame
    if isinstance(frame, Mapping):
        text = frame.get("text", "")
        return text if isinstance(text, str) else ""
    return ""


class FieldNormalizer:
    """The REAL :class:`~percept.extraction.EvidenceExtractor` — pure-Python, no vendor
    dependency, no lazy import (parsing already-recognized text needs no OCR/ASR of its own).
    Reads ``frame`` as text (see :func:`_frame_text`), runs :func:`extract_fields`, and returns
    ``Enrichment(facts={"fields": [...]})`` — empty ``Enrichment()`` when the text has no
    field-shaped mentions at all (the standard no-raise-on-a-miss contract)."""

    def extract(self, condition: str, frame: Any, roi: "Box | None" = None) -> Enrichment:
        text = _frame_text(frame)
        if not text:
            return Enrichment()
        fields = extract_fields(text)
        if not fields:
            return Enrichment()
        return Enrichment(facts={"fields": fields})


class ScriptedFields:
    """Deterministic, offline field-normalization double — same convention as
    :class:`percept.extractors.ocr.ScriptedOCR`. ``script`` maps a frame token (or, for a
    non-string frame, its :func:`~percept.extractors.ocr._frame_key` content hash) to one of:

    - a ready-made :class:`~percept.extraction.Enrichment`, used as-is;
    - a dict with ``"fields"`` — already-typed field dicts, used as-is (script an exact fixture
      without depending on the regex parser);
    - a dict with ``"text"`` — run through the SAME :func:`extract_fields` normalization rules
      :class:`FieldNormalizer` uses (script raw text, get the real parser's output).

    A frame with no script entry returns an empty ``Enrichment`` — proves the extraction seam's
    WIRING for field-shaped facts, never the parser's intelligence (that is `FieldNormalizer`'s,
    or the hand-authored ``"fields"`` entry's, job).
    """

    def __init__(self, script: "Mapping[str, Any] | None" = None) -> None:
        self._script: dict[str, Any] = dict(script or {})

    def extract(self, condition: str, frame: Any, roi: "Box | None" = None) -> Enrichment:
        hit = self._script.get(_frame_key(frame))
        if hit is None:
            return Enrichment()
        if isinstance(hit, Enrichment):
            return hit
        if "fields" in hit:
            facts = {"fields": list(hit["fields"])}
        else:
            facts = {"fields": extract_fields(hit.get("text", ""))}
        return Enrichment(
            facts=facts,
            evidence_refs=tuple(hit.get("evidence_refs", ())),
            ts_range=hit.get("ts_range"),
            confidence=hit.get("confidence"),
        )


def chain_ocr_then_fields(ocr_extractor: Any, fields_extractor: Any = None) -> Any:
    """Compose an OCR extractor with :class:`FieldNormalizer` (or a caller-supplied fields
    extractor) into ONE :class:`~percept.extraction.EvidenceExtractor`: runs ``ocr_extractor``
    first, feeds its ``facts["text"]`` into the fields extractor, and merges both into a single
    ``Enrichment`` (``{"text": ..., "regions": [...], "fields": [...]}``) — the shape a fused
    compare condition reads directly, without a caller hand-wiring two separate extractor calls
    (and two ``extractor.enriched`` trace stages) into their own composition glue.

    Optional convenience, not required to use either extractor directly — ``Percept.create``
    still only ever takes ONE ``extractor=``, so chaining at the call site (this) or writing a
    custom composed extractor are the only two ways to run more than one per tick.
    """
    fields_extractor = fields_extractor if fields_extractor is not None else FieldNormalizer()

    class _ChainedOCRFields:
        def extract(self, condition: str, frame: Any, roi: "Box | None" = None) -> Enrichment:
            ocr_enrichment = ocr_extractor.extract(condition, frame, roi)
            if ocr_enrichment.is_empty:
                return ocr_enrichment
            text = ocr_enrichment.facts.get("text", "")
            fields_enrichment = fields_extractor.extract(condition, {"text": text}, roi)
            facts = dict(ocr_enrichment.facts)
            if not fields_enrichment.is_empty:
                facts["fields"] = fields_enrichment.facts.get("fields", [])
            evidence_refs = tuple(ocr_enrichment.evidence_refs) + tuple(fields_enrichment.evidence_refs)
            return Enrichment(
                facts=facts,
                evidence_refs=evidence_refs,
                ts_range=ocr_enrichment.ts_range,
                confidence=ocr_enrichment.confidence,
            )

    return _ChainedOCRFields()


__all__ = [
    "FieldNormalizer",
    "ScriptedFields",
    "classify_value",
    "extract_fields",
    "chain_ocr_then_fields",
]
