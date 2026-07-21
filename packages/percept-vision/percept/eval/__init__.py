"""Golden-set loader + split discipline.

The golden corpus is versioned + PR-reviewed. This loader reads the label layer
(``splits.json``, ``*.jsonl``) and enforces the one rule that protects the trust metric: the
**train / dev / test splits must be disjoint**, and ``test`` is held out (never tuned on).

It deliberately reads only the label files, not the heavy pixel data — pixels are loaded by the
replay scorer (Phase 2), not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SPLIT_NAMES = ("train", "dev", "test")


class SplitError(ValueError):
    """Raised when splits overlap or reference unknown scenes."""


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a ``.jsonl`` label file into a list of dicts (blank lines skipped)."""
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """Load ``splits.json`` and validate the splits are disjoint."""
    data = json.loads(Path(path).read_text())
    splits = {name: list(data.get(name, [])) for name in SPLIT_NAMES}
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, list[str]]) -> None:
    """Raise ``SplitError`` if any scene appears in more than one split."""
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for scene in splits.get(name, []):
            if scene in seen:
                raise SplitError(
                    f"scene {scene!r} is in both {seen[scene]!r} and {name!r} splits"
                )
            seen[scene] = name


@dataclass
class GoldenSet:
    """The label layer of a golden corpus, indexed by scene id."""

    root: Path
    splits: dict[str, list[str]] = field(default_factory=dict)
    labels: list[dict] = field(default_factory=list)
    gate_cases: list[dict] = field(default_factory=list)
    utterances: list[dict] = field(default_factory=list)

    def split_of(self, scene_id: str) -> str | None:
        for name in SPLIT_NAMES:
            if scene_id in self.splits.get(name, []):
                return name
        return None

    def scenes(self, split: str) -> list[str]:
        return list(self.splits.get(split, []))


def load_golden(root: str | Path) -> GoldenSet:
    """Load the label layer of a golden set rooted at ``root``."""
    root = Path(root)
    return GoldenSet(
        root=root,
        splits=load_splits(root / "splits.json"),
        labels=load_jsonl(root / "labels.jsonl"),
        gate_cases=load_jsonl(root / "gate_cases.jsonl"),
        utterances=load_jsonl(root / "utterances.jsonl"),
    )


def find_golden_root(start: str | Path | None = None) -> Path | None:
    """Walk up from ``start`` looking for ``eval/fixtures-golden/splits.json``."""
    here = Path(start or __file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "eval" / "fixtures-golden"
        if (candidate / "splits.json").exists():
            return candidate
    return None


__all__ = [
    "SPLIT_NAMES",
    "SplitError",
    "GoldenSet",
    "load_jsonl",
    "load_splits",
    "assert_disjoint",
    "load_golden",
    "find_golden_root",
]
