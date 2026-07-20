"""``FrameHashLayer`` — the perception cache (a cost/latency optimisation only).

A ``CacheLayer`` that skips the VLM when a goal's ROI hasn't visibly changed. Per-goal it
stores an **ROI-cropped dHash** of the last-perceived frame plus the verdict it produced. On
``lookup`` it near-matches the incoming frame's dHash by **Hamming distance** (``<= tau_bits``);
a near-match within the **TTL** is a HIT carrying a *decayed* confidence:

    confidence = min(verdict.confidence, sim * age_decay)

where ``sim`` rises toward 1 as the Hamming distance falls to 0, and ``age_decay`` falls toward
0 as the entry ages toward the TTL. Past the TTL the entry is a hard miss (always DEFER) — the
ceiling that bounds the slow drift a hash cannot see (spec §6.2).

Invalidation order (spec §6.2): **goal-change -> assertion-override -> TTL -> L0 frame-diff**.
``invalidate(goal_id=...)`` drops a goal's entries on goal-change; ``invalidate_entity(...)``
drops the entries a user correction touched (assertion-override); TTL is enforced on every
lookup; the dHash near-match *is* the L0 frame-diff backstop.

This module is **pure stdlib**. ``numpy`` is an OPTIONAL fast path for frame conversion only —
never required; the L1 tests run on synthetic frames (a small 2D grid / bytes) with no numpy.

The cache **returns a cached** ``Verdict`` — it never compiles a goal to ``attr=value`` and
exact-matches (the firing contract; one path, one gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..adapters import Box, Verdict
from ..config import CacheSettings
from .cascade import CacheResult

# dHash works on a small WxH luminance grid. 8x8 cells -> a 64-bit hash from horizontal
# gradients over a 9-wide row (a literal lives here only as the hash geometry, not as a
# tunable of the cognition logic; tau_bits / ttl_s / etc. all come from CacheSettings).
_HASH_W = 8
_HASH_H = 8


# ---------------------------------------------------------------------------
# Frame -> grayscale grid (pure stdlib; numpy optional, never required)
# ---------------------------------------------------------------------------


def _grid_from_frame(frame: Any) -> list[list[float]] | None:
    """Best-effort, deterministic luminance grid from an opaque frame.

    Recognized shapes (no third-party import on the L1 path):

    * a 2D grid: ``list[list[number]]`` / tuple-of-tuples (synthetic-frame test path);
    * raw ``bytes`` / ``bytearray``: reshaped row-major into a near-square grid;
    * anything exposing ``tolist()`` (e.g. a numpy array) -> handled via its nested lists.

    Returns ``None`` when the frame is not griddable (e.g. a bare string token) — the layer
    then has no visual signal and safely DEFERs.
    """
    if frame is None:
        return None
    # numpy-like / array-like: convert without importing numpy ourselves.
    tolist = getattr(frame, "tolist", None)
    if callable(tolist):
        try:
            frame = tolist()
        except Exception:  # noqa: BLE001 — fall through to the generic handling
            pass
    if isinstance(frame, (bytes, bytearray)):
        data = list(frame)
        if not data:
            return None
        # reshape row-major into a near-square grid (deterministic).
        import math

        side = max(1, int(math.isqrt(len(data))))
        rows = [data[i : i + side] for i in range(0, side * side, side)]
        return [[float(v) for v in row] for row in rows if row]
    if isinstance(frame, (list, tuple)) and frame:
        first = frame[0]
        if isinstance(first, (list, tuple)):
            grid: list[list[float]] = []
            for row in frame:
                if not isinstance(row, (list, tuple)) or not row:
                    return None
                grid.append([float(_lum(v)) for v in row])
            return grid
    return None


def _lum(value: Any) -> float:
    """Coerce a cell to a luminance scalar. (r, g, b)[+a] -> Rec.601 luma; scalar -> float."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        r, g, b = float(value[0]), float(value[1]), float(value[2])
        return 0.299 * r + 0.587 * g + 0.114 * b
    return float(value)


def _crop_grid(grid: list[list[float]], roi: Box | None) -> list[list[float]]:
    """Crop a luminance grid to ``roi`` (normalized 0..1). Degenerate crops fall back to full."""
    if roi is None:
        return grid
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return grid
    x0 = max(0, min(w - 1, int(roi.x * w)))
    y0 = max(0, min(h - 1, int(roi.y * h)))
    x1 = max(x0 + 1, min(w, int((roi.x + roi.w) * w) + 1 if roi.w > 0 else x0 + 1))
    y1 = max(y0 + 1, min(h, int((roi.y + roi.h) * h) + 1 if roi.h > 0 else y0 + 1))
    cropped = [row[x0:x1] for row in grid[y0:y1]]
    if not cropped or not cropped[0]:
        return grid
    return cropped


def _resample(grid: list[list[float]], out_w: int, out_h: int) -> list[list[float]]:
    """Nearest-neighbour resample to ``out_h x out_w`` (deterministic, stdlib-only)."""
    in_h = len(grid)
    in_w = len(grid[0]) if in_h else 0
    if in_h == 0 or in_w == 0:
        return [[0.0] * out_w for _ in range(out_h)]
    out: list[list[float]] = []
    for oy in range(out_h):
        iy = min(in_h - 1, (oy * in_h) // out_h)
        row = grid[iy]
        out.append([row[min(in_w - 1, (ox * in_w) // out_w)] for ox in range(out_w)])
    return out


def dhash(frame: Any, roi: Box | None = None) -> int | None:
    """Difference-hash a frame's ROI to a ``_HASH_W * _HASH_H``-bit integer (or ``None``).

    The classic dHash: resample the ROI to ``(W+1) x H`` luminance, then emit one bit per
    horizontal adjacency (``left < right``). Robust to brightness/scale, sensitive to structure
    — exactly the "did the ROI's content change?" signal the cache wants.
    """
    grid = _grid_from_frame(frame)
    if grid is None:
        return None
    grid = _crop_grid(grid, roi)
    small = _resample(grid, _HASH_W + 1, _HASH_H)
    bits = 0
    pos = 0
    for row in small:
        for x in range(_HASH_W):
            bits |= (1 if row[x] < row[x + 1] else 0) << pos
            pos += 1
    return bits


def hamming(a: int, b: int) -> int:
    """Hamming distance between two hashes (popcount of the XOR)."""
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    goal_id: str
    entity_id: str | None
    hash_bits: int
    verdict: Verdict
    confidence: float  # the confidence stored at perceive time
    stored_at: float


class FrameHashLayer:
    """Per-goal ROI-dHash near-match cache. Implements the ``CacheLayer`` protocol.

    ``key`` is a ``CacheKey`` (goal_id, frame, roi, entity_id) — or, for cascade-composition
    compatibility, any object the caller adapts via ``key_fn``. The layer holds at most one live
    entry per ``(goal_id, entity_id)``; a fresh ``store`` for that pair overwrites the old one
    (the ROI moved, the verdict changed — keep the newest).
    """

    name = "framehash"

    def __init__(
        self,
        settings: CacheSettings,
        *,
        clock: Callable[[], float],
        key_fn: Callable[[Any], "CacheKey"] | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._key_fn = key_fn or _default_key_fn
        # (goal_id, entity_id, condition) -> _Entry. Condition is part of the slot so a goal re-armed
        # under a new condition can't be served its stale verdict (D3a); invalidate*() still key off
        # slot[0]/slot[1] (goal_id/entity_id), so their scoping is unchanged.
        self._entries: dict[tuple[str, str | None, str | None], _Entry] = {}

    # -- the CacheLayer protocol ------------------------------------------
    def lookup(self, key: Any) -> CacheResult:
        ck = self._key_fn(key)
        slot = (ck.goal_id, ck.entity_id, ck.condition)
        entry = self._entries.get(slot)
        if entry is None:
            return CacheResult.defer()

        now = self._clock()
        age = now - entry.stored_at
        ttl = self._settings.ttl_s
        # TTL is a hard ceiling: past it, drop and always DEFER (bounds slow drift).
        if ttl <= 0.0 or age >= ttl:
            del self._entries[slot]
            return CacheResult.defer()

        bits = dhash(ck.frame, ck.roi)
        if bits is None:
            # no visual signal to compare -> cannot prove the ROI is unchanged -> DEFER.
            return CacheResult.defer()

        dist = hamming(bits, entry.hash_bits)
        if dist > self._settings.tau_bits:
            # structural change beyond the near-match radius -> L0 frame-diff says re-perceive.
            return CacheResult.defer()

        sim = self._similarity(dist)
        decay = self._age_decay(age, ttl)
        confidence = min(entry.confidence, sim * decay)
        return CacheResult(
            status="HIT",
            verdict=entry.verdict,
            confidence=confidence,
            provenance="cached",
            source_layer=self.name,
        )

    def store(self, key: Any, verdict: Verdict, confidence: float) -> None:
        ck = self._key_fn(key)
        bits = dhash(ck.frame, ck.roi)
        if bits is None:
            return  # un-hashable frame: nothing to cache against -> leave the slot empty.
        slot = (ck.goal_id, ck.entity_id, ck.condition)
        self._entries[slot] = _Entry(
            goal_id=ck.goal_id,
            entity_id=ck.entity_id,
            hash_bits=bits,
            verdict=verdict,
            confidence=confidence,
            stored_at=self._clock(),
        )

    # -- invalidation (order: goal-change -> assertion-override -> TTL -> L0) ----
    def invalidate(self, goal_id: str | None = None) -> int:
        """Drop entries for ``goal_id`` (goal-change); ``None`` clears every entry.

        Returns the number of entries dropped (handy for metrics / tests).
        """
        if goal_id is None:
            n = len(self._entries)
            self._entries.clear()
            return n
        victims = [slot for slot in self._entries if slot[0] == goal_id]
        for slot in victims:
            del self._entries[slot]
        return len(victims)

    def invalidate_entity(
        self, entity_id: str | None, *, goal_id: str | None = None
    ) -> int:
        """Drop entries touching ``entity_id`` (assertion-override).

        A user correction binds to an entity; every cached verdict resting on that entity is now
        suspect and must be re-perceived. Optionally scope to a single ``goal_id``.
        """
        victims = [
            slot
            for slot in self._entries
            if slot[1] == entity_id and (goal_id is None or slot[0] == goal_id)
        ]
        for slot in victims:
            del self._entries[slot]
        return len(victims)

    # -- internals ---------------------------------------------------------
    def _similarity(self, dist: int) -> float:
        """Map Hamming distance -> [0, 1] similarity. 0 bits -> 1.0; tau bits -> ~0.

        Linear over ``[0, tau_bits]``; ``tau_bits == 0`` means only an exact hash counts.
        """
        tau = self._settings.tau_bits
        if tau <= 0:
            return 1.0 if dist == 0 else 0.0
        return max(0.0, 1.0 - dist / float(tau))

    def _age_decay(self, age: float, ttl: float) -> float:
        """Map age -> [0, 1] decay. age 0 -> 1.0; age -> ttl -> 0.0 (linear)."""
        if ttl <= 0.0:
            return 0.0
        return max(0.0, 1.0 - age / ttl)

    def __len__(self) -> int:  # introspection / tests
        return len(self._entries)


# ---------------------------------------------------------------------------
# Cache key (what the layer hashes against)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheKey:
    """The information the frame-hash layer needs to key + hash an ROI lookup.

    ``condition`` scopes the entry to the goal's CURRENT plain-language meaning: a goal id re-armed with
    a new condition (voice re-arm) must not be served a verdict cached under its OLD condition. It is
    part of the entry slot, so a condition change is a clean miss (D3a). ``None`` (the default) keeps the
    pre-D3a keying — every existing caller that omits it shares one condition-agnostic slot, unchanged."""

    goal_id: str
    frame: Any
    roi: Box | None = None
    entity_id: str | None = None
    condition: str | None = None


def _default_key_fn(key: Any) -> CacheKey:
    """Adapt a lookup ``key`` to a ``CacheKey``.

    Accepts a ``CacheKey`` directly, or a ``(goal_id, frame[, roi[, entity_id]])`` tuple — the
    minimal shape ``agent._judge`` already builds. A bare value is treated as ``(value, value)``
    so the layer degrades gracefully rather than raising.
    """
    if isinstance(key, CacheKey):
        return key
    if isinstance(key, tuple):
        goal_id = key[0]
        frame = key[1] if len(key) > 1 else None
        roi = key[2] if len(key) > 2 else None
        entity_id = key[3] if len(key) > 3 else None
        condition = key[4] if len(key) > 4 else None
        return CacheKey(
            goal_id=str(goal_id), frame=frame, roi=roi, entity_id=entity_id, condition=condition
        )
    return CacheKey(goal_id=str(key), frame=key)


__all__ = [
    "FrameHashLayer",
    "CacheKey",
    "dhash",
    "hamming",
]
