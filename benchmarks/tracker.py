"""A small multi-person tracker for the benchmark's detector baselines.

Written for this harness from the association method described in the paper Zhang et al.,
"ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022,
arXiv:2110.06864), Algorithm 1: every detection box is used, high-score boxes are associated
first, and low-score boxes then only extend tracks the first stage left unmatched. No code from
that paper's repository, from SORT or from DeepSORT was used or consulted. Motion is this file's
own: a track predicts its next box by adding a smoothed per-frame velocity of its corners, in
place of the paper's Kalman filter. Matching uses scipy.optimize.linear_sum_assignment (BSD-3).

A new track is confirmed on its second matched frame (or at once on the very first frame of the
footage); an unconfirmed track that misses a frame is dropped; a confirmed track unmatched for more
than ``track_buffer`` frames is dropped, so a person who returns later gets a new id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HIGH_MIN_IOU = 0.2       # first stage: a high-score box joins a track it overlaps at least this much
LOW_MIN_IOU = 0.5        # second stage: a low-score box must overlap a track more to extend it
SMOOTH = 0.3             # weight of the newest per-frame corner motion in a track's velocity


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in ``a`` (n, 4) with every box in ``b`` (m, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])      # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def pair(score: np.ndarray, floor: float) -> tuple[list, list, list]:
    """Best one-to-one pairs of rows and columns by ``score`` (higher is better), keeping pairs at
    ``floor`` or more: (pairs, unpaired rows, unpaired columns)."""
    rows, cols = score.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-score)
    pairs = [(i, j) for i, j in zip(r, c) if score[i, j] >= floor]
    used_r, used_c = {i for i, _ in pairs}, {j for _, j in pairs}
    return pairs, [i for i in range(rows) if i not in used_r], [j for j in range(cols) if j not in used_c]


@dataclass
class Track:
    track_id: int
    box: np.ndarray                 # (x0, y0, x1, y1) as last matched
    score: float
    last_frame: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    confirmed: bool = False
    history: list = field(default_factory=list)     # (t, box) of every matched frame

    def predicted(self, frame: int) -> np.ndarray:
        return self.box + self.velocity * (frame - self.last_frame)


class Tracker:
    """Tracks over (x0, y0, x1, y1, score) detections, one ``update`` per frame.

    Boxes scoring ``high`` or more are associated first and may start a track (from ``new``);
    boxes between ``low`` and ``high`` only extend tracks seen on the previous frame that the
    first stage left unmatched."""

    def __init__(self, high: float = 0.5, low: float = 0.1, new: float = 0.5, track_buffer: int = 30) -> None:
        self.high, self.low, self.new, self.buffer = high, low, new, track_buffer
        self.tracks: list[Track] = []
        self.frame, self.next_id = 0, 1

    def update(self, dets: np.ndarray, t: float = 0.0) -> list[Track]:
        """Associate this frame's detections; returns the confirmed tracks matched on it."""
        self.frame += 1
        dets = np.asarray(dets, dtype=float).reshape(-1, 5)
        high = dets[dets[:, 4] >= self.high]
        low = dets[(dets[:, 4] >= self.low) & (dets[:, 4] < self.high)]
        guess = np.array([tr.predicted(self.frame) for tr in self.tracks]).reshape(-1, 4)
        matched = set()

        def extend(tr: Track, det: np.ndarray) -> None:
            gap = self.frame - tr.last_frame
            step = (det[:4] - tr.box) / gap
            tr.velocity = (1 - SMOOTH) * tr.velocity + SMOOTH * step if tr.hits > 1 else step
            tr.box, tr.score, tr.last_frame, tr.hits = det[:4].copy(), float(det[4]), self.frame, tr.hits + 1
            tr.confirmed = True
            tr.history.append((t, det[:4].copy()))
            matched.add(id(tr))

        pairs, left_tracks, left_high = pair(overlap(guess, high[:, :4]), HIGH_MIN_IOU)
        for i, j in pairs:
            extend(self.tracks[i], high[j])
        recent = [i for i in left_tracks if self.tracks[i].last_frame == self.frame - 1]
        pairs2, _, _ = pair(overlap(guess[recent], low[:, :4]), LOW_MIN_IOU)
        for i, j in pairs2:
            extend(self.tracks[recent[i]], low[j])
        self.tracks = [tr for tr in self.tracks if id(tr) in matched
                       or (tr.confirmed and self.frame - tr.last_frame <= self.buffer)]
        for j in left_high:
            det = high[j]
            if det[4] >= self.new:
                self.tracks.append(Track(self.next_id, det[:4].copy(), float(det[4]), self.frame,
                                         confirmed=self.frame == 1, history=[(t, det[:4].copy())]))
                self.next_id += 1
        return [tr for tr in self.tracks if tr.confirmed and tr.last_frame == self.frame]
