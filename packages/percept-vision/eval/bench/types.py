"""The track contract — every track scorer returns a TrackResult; the card emitter consumes a list of them.

A track scorer is a module exposing::

    def score(*, live: bool, n: int) -> TrackResult

- ``live=False``  -> a fast OFFLINE result (fakes / cached / stub cells) so the card renders with no API.
- ``live=True``   -> the real numbers (Gemini), reusing the existing per-track tool.

Cells are keyed ``(lane, metric)`` with ``lane in CONFIGS``. Lanes match the benchmark methodology:

``dense-modeled``
    A non-executed baseline estimate, usually call-count/cost only. No dense live VLM loops.
``edge``
    The conductor/edge layer only: wake decisions, recall, precision proxy, and call economics.
``e2e``
    Product path: goal -> edge candidates -> backend judges candidates -> core emits fire/ask/silent.
``core-replay``
    The cognition core replayed over already-collected observations/verdicts.

Numbers that are *modeled* (not executed) must be listed in ``.modeled`` so the card flags them with a
``~``. Non-dense modeled result cells refuse a RELEASE stamp; ``dense-modeled`` cost references are
allowed because they are explicitly part of the methodology. Tracks that are useful diagnostics but not
publishable product benchmarks must set ``diagnostic=True`` and explain why in ``excluded``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONFIGS = ("dense-modeled", "edge", "e2e", "core-replay")


@dataclass
class Coverage:
    """Per-track coverage/provenance gate for a publishable card."""

    dataset: str
    expected: int
    observed: int
    failed: int = 0
    skipped: int = 0
    notes: str = ""

    @property
    def passed(self) -> bool:
        return self.expected > 0 and self.observed >= self.expected and self.failed == 0 and self.skipped == 0


@dataclass
class TrackResult:
    track: str  # "relational" | "counting" | "acoustic" | ...
    cells: dict = field(default_factory=dict)  # {(lane, metric): value}  e.g. ("e2e","precision"): 0.83
    fmts: dict = field(default_factory=dict)  # {metric: "{:.2f}"} optional per-metric format
    headline: dict = field(default_factory=dict)  # {label: str} -> the card's headline vector
    measured: list = field(default_factory=list)  # provenance: measured cell labels
    modeled: list = field(default_factory=list)  # provenance: modeled cells (rendered with ~)
    excluded: list = field(default_factory=list)  # invalid items + reason
    reference: dict = field(default_factory=dict)  # {metric: {"value": v, "by": "TransRAC CVPR'22"}} SOTA ref
    calls: int = 0  # live vision calls spent (cost)
    notes: str = ""
    coverage: list[Coverage | dict] = field(default_factory=list)
    diagnostic: bool = False

    def metrics(self) -> list[str]:
        """Distinct metric names in cell-insertion order."""
        seen: list[str] = []
        for _config, metric in self.cells:
            if metric not in seen:
                seen.append(metric)
        return seen

    def fmt(self, metric: str) -> str:
        return self.fmts.get(metric, "{}")

    def normalized_coverage(self) -> list[Coverage]:
        out: list[Coverage] = []
        for item in self.coverage:
            out.append(item if isinstance(item, Coverage) else Coverage(**item))
        return out

    def invalid_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.diagnostic:
            reasons.append(f"{self.track}: diagnostic-only")
        reasons.extend(str(e) for e in self.excluded)
        reasons.extend(f"{self.track}: modeled {m}" for m in self.modeled if not str(m).startswith("dense-modeled:"))
        for item in self.normalized_coverage():
            if not item.passed:
                reasons.append(
                    f"{self.track}: coverage failed for {item.dataset} "
                    f"(expected={item.expected}, observed={item.observed}, failed={item.failed}, skipped={item.skipped})"
                )
        for lane, metric in self.cells:
            if lane not in CONFIGS:
                reasons.append(f"{self.track}: unsupported lane {lane!r} for metric {metric!r}")
        return reasons
