"""Labels: ground truth for your own clips, in one plain-data document.

Experimental: the format may change before it is stable.

The format (loaded from JSON or YAML by the caller)::

    {"conditions": [{"label": "person falls", "text": "a person falls",
                     "kind": "event", "detector": "person"}],
     "clips": [{"file": "a.mp4", "group": "lobby-2026-03-01",
                "occurrences": [{"label": "person falls", "start": 12.0, "end": 14.5}]}]}

Times are seconds on the clip's timeline. ``kind`` is ``event`` (the default) or ``state``;
``detector`` optionally names what a cheap detector can express the condition as. ``group`` names
the recording a clip belongs to (clips recorded at the same time by different cameras share one)
and defaults to the file's stem; :func:`split` keeps a group together. Overlapping occurrences of
the same label are merged into one (an onset can't be matched twice).

:func:`load_labels` checks a document and returns :class:`Labels`, raising :class:`LabelError`
with the path of the offending field. :attr:`Labels.labels_hash` covers the content with file
names reduced to their base names, whatever the order of keys, conditions or clips, so it is the
same in every checkout.

Pure: no I/O, no clock. Durations for :func:`split` are passed in by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

FORMAT = "percept.labels/v1alpha1"
KINDS = ("event", "state")
SPLITS = ("dev", "validation", "test")


class LabelError(ValueError):
    """A malformed labels document. ``path`` names the field (``clips[2].occurrences[0].end``)."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path


@dataclass(frozen=True)
class LabeledCondition:
    label: str
    text: str
    kind: str = "event"
    detector: Optional[str] = None


@dataclass(frozen=True)
class Occurrence:
    label: str
    start: float
    end: float


@dataclass(frozen=True)
class Clip:
    file: str
    group: str
    occurrences: tuple[Occurrence, ...] = ()

    def occurrences_of(self, label: str) -> tuple[Occurrence, ...]:
        return tuple(o for o in self.occurrences if o.label == label)


@dataclass(frozen=True)
class Labels:
    conditions: tuple[LabeledCondition, ...]
    clips: tuple[Clip, ...]
    labels_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels_hash", _sha(_canonical(self)))

    def clip(self, file: str) -> Clip:
        """The clip with this file name (or base name); KeyError if none."""
        base = _basename(file)
        for c in self.clips:
            if c.file == file or _basename(c.file) == base:
                return c
        raise KeyError(file)


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _stem(path: str) -> str:
    base = _basename(path)
    return base.rsplit(".", 1)[0] if "." in base.lstrip(".") else base


def _sha(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(labels: Labels) -> dict:
    return {"format": FORMAT,
            "conditions": sorted(([c.label, c.text, c.kind, c.detector] for c in labels.conditions),
                                 key=lambda r: r[0]),
            "clips": sorted(({"file": _basename(c.file), "group": c.group,
                              "occurrences": [[o.label, o.start, o.end] for o in c.occurrences]}
                             for c in labels.clips), key=lambda r: r["file"])}


def merge_overlapping(occs: list[Occurrence]) -> list[Occurrence]:
    """Same-label occurrences that overlap become one (an onset can't be matched twice)."""
    out: list[Occurrence] = []
    for o in sorted(occs, key=lambda o: (o.label, o.start)):
        if out and out[-1].label == o.label and o.start <= out[-1].end:
            out[-1] = Occurrence(o.label, out[-1].start, max(out[-1].end, o.end))
        else:
            out.append(o)
    return sorted(out, key=lambda o: o.start)


# --------------------------------------------------------------------------- loading


def _mapping(value: Any, path: str, allowed: tuple[str, ...], required: tuple[str, ...]) -> Mapping:
    if not isinstance(value, Mapping):
        raise LabelError(path, f"expected a mapping, got {type(value).__name__}")
    for key in value:
        if key not in allowed:
            raise LabelError(f"{path}.{key}" if path else str(key), f"unknown key (expected one of {', '.join(allowed)})")
    for key in required:
        if key not in value:
            raise LabelError(f"{path}.{key}" if path else key, "is required")
    return value


def _list(value: Any, path: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise LabelError(path, f"expected a list, got {type(value).__name__}")
    return list(value)


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LabelError(path, "expected a non-empty string")
    return value


def _seconds(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelError(path, f"expected a number of seconds, got {type(value).__name__}")
    try:
        out = float(value)
    except OverflowError:
        raise LabelError(path, "is too large") from None
    if not math.isfinite(out) or out < 0:
        raise LabelError(path, "expected a finite, non-negative number of seconds")
    return out


def load_labels(doc: Any) -> Labels:
    """Check a labels document (a mapping) and return :class:`Labels`; any malformed field raises
    :class:`LabelError` naming it."""
    doc = _mapping(doc, "", ("conditions", "clips"), ("conditions", "clips"))
    conds: list[LabeledCondition] = []
    for i, raw in enumerate(_list(doc["conditions"], "conditions")):
        path = f"conditions[{i}]"
        raw = _mapping(raw, path, ("label", "text", "kind", "detector"), ("label", "text"))
        label = _text(raw["label"], f"{path}.label")
        if any(c.label == label for c in conds):
            raise LabelError(f"{path}.label", f"duplicate label {label!r}")
        kind = raw.get("kind", "event")
        if kind not in KINDS:
            raise LabelError(f"{path}.kind", f"expected one of {', '.join(KINDS)}")
        detector = raw.get("detector")
        if detector is not None:
            detector = _text(detector, f"{path}.detector")
        conds.append(LabeledCondition(label, _text(raw["text"], f"{path}.text"), kind, detector))
    known = {c.label for c in conds}
    clips: list[Clip] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(_list(doc["clips"], "clips")):
        path = f"clips[{i}]"
        raw = _mapping(raw, path, ("file", "group", "occurrences"), ("file",))
        file = _text(raw["file"], f"{path}.file")
        base = _basename(file)
        if not base:
            raise LabelError(f"{path}.file", "expected a file name")
        if base.casefold() in seen:     # case-folded: a.mp4 and A.MP4 are one file on many disks
            raise LabelError(f"{path}.file", f"duplicate file {base!r} (also clips[{seen[base.casefold()]}])")
        seen[base.casefold()] = i
        group = _text(raw["group"], f"{path}.group") if "group" in raw else _stem(file)
        occs = []
        for j, o in enumerate(_list(raw.get("occurrences", []), f"{path}.occurrences")):
            opath = f"{path}.occurrences[{j}]"
            o = _mapping(o, opath, ("label", "start", "end"), ("label", "start", "end"))
            label = _text(o["label"], f"{opath}.label")
            if label not in known:
                raise LabelError(f"{opath}.label", f"unknown label {label!r}")
            start, end = _seconds(o["start"], f"{opath}.start"), _seconds(o["end"], f"{opath}.end")
            if end < start:
                raise LabelError(f"{opath}.end", f"ends ({end:g}) before it starts ({start:g})")
            occs.append(Occurrence(label, start, end))
        clips.append(Clip(file, group, tuple(merge_overlapping(occs))))
    return Labels(tuple(conds), tuple(clips))


# --------------------------------------------------------------------------- frozen splits


def bucket(group: str) -> int:
    """A recording group's split bucket, 0-99: ``sha256(group) mod 100``."""
    return int.from_bytes(hashlib.sha256(group.encode("utf-8")).digest(), "big") % 100


@dataclass(frozen=True)
class SplitManifest:
    """Which clips are in each split. ``clips``: split -> file names; ``occurrences``: split ->
    label -> occurrences; ``hours``: split -> hours of footage (None without durations);
    ``manifest_hash`` covers all of it, the cutoffs and the labels' hash."""

    cutoffs: tuple[int, int]
    labels_hash: str
    clips: Mapping[str, tuple[str, ...]]
    occurrences: Mapping[str, Mapping[str, int]]
    hours: Mapping[str, Optional[float]]
    manifest_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_hash", _sha(
            {"format": FORMAT, "cutoffs": list(self.cutoffs), "labels_hash": self.labels_hash,
             "clips": {k: list(v) for k, v in self.clips.items()},
             "occurrences": {k: dict(v) for k, v in self.occurrences.items()}, "hours": dict(self.hours)}))


def split_of(group: str, cutoffs: tuple[int, int] = (40, 70)) -> str:
    """The split of a recording group: bucket below the first cutoff ``dev``, below the second
    ``validation``, else ``test``."""
    b = bucket(group)
    return SPLITS[0] if b < cutoffs[0] else SPLITS[1] if b < cutoffs[1] else SPLITS[2]


def split(labels: Labels, durations: Optional[Mapping[str, float]] = None, *,
          cutoffs: tuple[int, int] = (40, 70)) -> SplitManifest:
    """Freeze ``labels`` into dev, validation and test by recording group, so adding clips never
    moves an existing one and a group never straddles two splits. ``durations`` maps each clip's
    file (or base name) to its length in seconds; without it, ``hours`` are None."""
    c1, c2 = cutoffs
    if any(isinstance(c, bool) or not isinstance(c, int) for c in cutoffs) or not 0 <= c1 <= c2 <= 100:
        raise ValueError(f"cutoffs must be two integers with 0 <= c1 <= c2 <= 100, got {cutoffs!r}")
    secs: dict[str, float] = {}
    if durations is not None:
        by_base: dict[str, Any] = {}
        for k, v in durations.items():
            if _basename(k) in by_base:
                raise LabelError(f"durations.{k}", f"duplicate file {_basename(k)!r}")
            by_base[_basename(k)] = v
        for c in labels.clips:
            if _basename(c.file) not in by_base:
                raise LabelError(f"durations.{c.file}", "missing")
            secs[c.file] = _seconds(by_base[_basename(c.file)], f"durations.{c.file}")
    clips: dict[str, list[str]] = {s: [] for s in SPLITS}
    counts = {s: {c.label: 0 for c in labels.conditions} for s in SPLITS}
    total = dict.fromkeys(SPLITS, 0.0)
    for c in labels.clips:
        s = split_of(c.group, (c1, c2))
        clips[s].append(_basename(c.file))
        for o in c.occurrences:
            counts[s][o.label] += 1
        total[s] += secs.get(c.file, 0.0)
    return SplitManifest((c1, c2), labels.labels_hash, {s: tuple(sorted(v)) for s, v in clips.items()}, counts,
                         {s: (round(total[s] / 3600.0, 6) if durations is not None else None) for s in SPLITS})
