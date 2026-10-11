"""The one schema every dataset adapter produces, and the frame feed every arm reads.

A ``VideoCase`` is one stream to watch: its media (one file, or a declared-synthetic
concatenation of segments), the plain-text conditions watched on it, and the ground-truth
occurrences. A condition with no occurrence in the case is an absent-condition row: any
decision about it is a false event.

Every arm reads frames through :class:`Feed`, which decodes the media the same way for each
arm and keeps a digest of what it produced, so the report can show the arms saw identical
evidence.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np


def slug(label: str) -> str:
    """A condition id from a label: lower case, words joined by underscores."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:60] or "condition"


@dataclass(frozen=True)
class Occurrence:
    label: str
    start: float            # seconds on the case timeline
    end: float


@dataclass(frozen=True)
class CondSpec:
    """A watched condition. ``detector`` names what a cheap detector can express it as
    (``"person"`` or ``"motion"``), or None when no detector can."""

    label: str
    text: str
    kind: str = "event"
    detector: Optional[str] = None

    @property
    def id(self) -> str:
        return slug(self.label)


@dataclass(frozen=True)
class Segment:
    """Part of a case timeline: ``path`` from ``start`` to ``end`` seconds, or, with ``hold``
    set, the frame at ``start`` held still for ``hold`` seconds (an idle gap)."""

    path: str
    start: float
    end: float
    hold: float = 0.0

    @property
    def duration(self) -> float:
        return self.hold if self.hold else self.end - self.start


@dataclass(frozen=True)
class VideoCase:
    case_id: str
    dataset: str
    segments: tuple[Segment, ...]
    conditions: tuple[CondSpec, ...]
    occurrences: tuple[Occurrence, ...]
    synthetic: bool = False                  # a constructed timeline, never natural footage
    notes: str = ""
    media_sha256: dict = field(default_factory=dict, compare=False)   # path -> sha256, filled lazily

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def media(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.path for s in self.segments))

    def occurrences_of(self, label: str) -> list[Occurrence]:
        return [o for o in self.occurrences if o.label == label]

    def hashes(self) -> dict:
        for path in self.media:
            if path not in self.media_sha256:
                self.media_sha256[path] = sha256_file(path)
        return {os.path.basename(p): self.media_sha256[p] for p in self.media}

    def summary(self) -> dict:
        return {"case_id": self.case_id, "dataset": self.dataset, "duration_s": round(self.duration, 3),
                "media": self.hashes(), "synthetic": self.synthetic, "notes": self.notes,
                "conditions": [{"id": c.id, "text": c.text, "kind": c.kind, "detector": c.detector,
                                "present": bool(self.occurrences_of(c.label))} for c in self.conditions],
                "occurrences": len(self.occurrences)}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merge_overlapping(occs: list[Occurrence]) -> list[Occurrence]:
    """Same-label occurrences that overlap become one (an onset can't be matched twice)."""
    out: list[Occurrence] = []
    for o in sorted(occs, key=lambda o: (o.label, o.start)):
        if out and out[-1].label == o.label and o.start <= out[-1].end:
            out[-1] = Occurrence(o.label, out[-1].start, max(out[-1].end, o.end))
        else:
            out.append(o)
    return sorted(out, key=lambda o: o.start)


class Feed:
    """Frames of a case as ``(t, image)``: ``t`` on the case timeline, ``image`` uint8 BGR with
    its longest side at most ``max_side``, sampled to ``fps``. ``digest`` covers every frame
    yielded so far.

    Frame accounting: ``sampled_out`` counts decoded source frames the ``fps`` sampling skipped
    (intentional); ``decode_gaps`` counts sample slots lost because the decoder returned nothing
    for more than ``GAP_S`` (a hole in the stream, a failed read, or a file shorter than its
    segment): unintended loss."""

    GAP_S = 1.0

    def __init__(self, case: VideoCase, fps: float = 5.0, max_side: int = 640, note=None) -> None:
        self.case, self.fps, self.max_side, self.note = case, fps, max_side, note   # note(t, image) per frame
        self._h = hashlib.sha256()
        self.frames = 0
        self.sampled_out = 0
        self.decode_gaps = 0

    @property
    def digest(self) -> str:
        return f"{self.frames}:{self._h.hexdigest()[:16]}"

    def _fit(self, image: np.ndarray) -> np.ndarray:
        import cv2
        h, w = image.shape[:2]
        if max(h, w) > self.max_side:
            s = self.max_side / max(h, w)
            image = cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        return image

    def _segment(self, seg: Segment) -> Iterator[tuple[float, np.ndarray]]:
        """(time within the segment, frame), sampled to ``fps``."""
        import cv2
        cap = cv2.VideoCapture(seg.path)
        if not cap.isOpened():          # unreadable media is a failed case, never quiet footage
            cap.release()
            raise OSError(f"cannot open {seg.path}")
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, seg.start * 1000.0)
            step, due, last = 1.0 / self.fps, 0.0, 0.0
            while cap.grab():                       # decode every frame, convert only the sampled ones
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - seg.start
                if not seg.hold and 0.0 <= t < seg.end - seg.start:
                    self._gap(t - last)
                    last = max(last, t)
                if seg.hold or t + 1e-6 >= due:
                    ok, image = cap.retrieve()
                    if not ok:
                        break
                if seg.hold:
                    still = self._fit(image)
                    rng = np.random.default_rng(int(seg.start * 1000))   # sensor noise, deterministic
                    for i in range(int(seg.hold * self.fps)):
                        noise = rng.integers(-2, 3, still.shape, dtype=np.int16)
                        yield i * step, np.clip(still.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    return
                if t >= seg.end - seg.start:
                    return
                if t + 1e-6 >= due:
                    due += step
                    yield max(t, 0.0), self._fit(image)
                elif t >= 0.0:
                    self.sampled_out += 1
            if not seg.hold:                        # the stream ended (or failed) before the segment did
                self._gap(seg.end - seg.start - last)
        finally:
            cap.release()

    def _gap(self, dt: float) -> None:
        if dt > self.GAP_S:
            self.decode_gaps += int(dt * self.fps)

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        offset = 0.0
        for seg in self.case.segments:
            for t, image in self._segment(seg):
                t = round(offset + t, 4)
                self._h.update(f"{t:.3f}".encode())
                self._h.update(np.ascontiguousarray(image).tobytes())
                self.frames += 1
                if self.note is not None:
                    self.note(t, image)
                yield t, image
            offset += seg.duration
