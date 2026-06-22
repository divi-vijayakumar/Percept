"""The track contract — every track scorer returns a TrackResult; the card emitter consumes a list of them.

A track scorer is a module exposing::

    def score(*, live: bool, n: int) -> TrackResult

- ``live=False``  -> a fast OFFLINE result (fakes / cached / stub cells) so the card renders with no API.
- ``live=True``   -> the real numbers (Gemini), reusing the existing per-track tool.

Cells are keyed ``(config, metric)`` with ``config in CONFIGS``; ``None`` = n/a for that config. Numbers
that are *modeled* (not executed) must be listed in ``.modeled`` so the card flags them with a ``~`` and
refuses a RELEASE stamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONFIGS = ("raw", "core", "e2e")


@dataclass
class TrackResult:
    track: str  # "relational" | "counting" | "acoustic" | ...
    cells: dict = field(default_factory=dict)  # {(config, metric): value}  e.g. ("core","precision"): 0.83
    fmts: dict = field(default_factory=dict)  # {metric: "{:.2f}"} optional per-metric format
    headline: dict = field(default_factory=dict)  # {label: str} -> the card's headline vector
    measured: list = field(default_factory=list)  # provenance: measured cell labels
    modeled: list = field(default_factory=list)  # provenance: modeled cells (rendered with ~)
    excluded: list = field(default_factory=list)  # invalid items + reason
    reference: dict = field(default_factory=dict)  # {metric: {"value": v, "by": "TransRAC CVPR'22"}} SOTA ref
    calls: int = 0  # live vision calls spent (cost)
    notes: str = ""

    def metrics(self) -> list[str]:
        """Distinct metric names in cell-insertion order."""
        seen: list[str] = []
        for _config, metric in self.cells:
            if metric not in seen:
                seen.append(metric)
        return seen

    def fmt(self, metric: str) -> str:
        return self.fmts.get(metric, "{}")
