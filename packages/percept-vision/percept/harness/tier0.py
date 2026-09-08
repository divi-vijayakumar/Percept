"""Tier-0 salience gate — the server-side reference for the edge wire contract.

The perception analog of voice VAD: a coarse pass that decides *when* to spend a
cloud vision call.

- **down:** the brain compiles active goals to a :class:`~percept.adapters.WatchSpec`
  (:func:`compile_watchspec`) — anchors + thresholds, one ``WatchCondition`` per watch goal.
- **up:** the gate reports a :class:`~percept.adapters.Tier0Signal` (motion / structural / semantic
  salience). This reference computes ``motion`` only (structural/semantic = 0.0); the real
  ``percept-edge`` adds CLIP / pose.
- the expensive ``Percept.perceive_judged`` runs **only when salience clears threshold**.

Frame-diff is arithmetic — pure stdlib, deterministic, zero user data. ``numpy`` is an
OPTIONAL fast path for :func:`motion_score`; it is **never required** for the L1 lane and the
stdlib and numpy paths return the same number.

**The wired loop** (:class:`GatedWatchLoop` / :func:`run_gated_watch`) is the proof point:
``compile_watchspec``'s output is actually consumed by the gate, and a ``Tier0Signal`` is reported
back up — even motion-only.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Literal, Sequence

from percept._numeric import clamp01 as _clamp01
from percept._numeric import finite as _finite
from percept.adapters import Box, Goal, Schedule, Tier0Signal, WatchCondition, WatchSpec
from percept.behavior_routes import (
    UndrivenBehaviorRouteWarning,
    drives_rhythm,
    goal_behavior_route,
    needs_watch_condition,
    require_drivable_routes,
    route_remedy,
    skipped_route_message,
    worker_can_drive,
)
from percept._streamtime import StreamTimeGuard, StreamTimeRewindWarning
from percept.behavior import _VALID_GATE_ON
from percept.behavior import coerce_authored_cadence as _coerce_cadence
from percept.config import Settings, load_settings
from percept.duration_routes import (
    require_duration_routes,
    require_in_process_duration_topology,
)
from percept.interest import InterestIntervalEvent, Tier0MotionInterest
from percept.observability import get_logger
from percept.rhythm import RhythmDrive

from .detectors import (
    Boundary,
    DetectorBank,
    WakePolicy,
    detectors_for,
    manifest_of,
    partition_by_capability,
    wake_score,
)


def _accepts_keyword(callable_obj: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )

try:  # OPTIONAL fast path only — the stdlib path below is the source of truth for L1.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy is not a dependency of the L1 lane.
    _np = None


# ---------------------------------------------------------------------------
# Frame representation (SIMPLE, stdlib, deterministic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """A coarse grayscale grid — the simplest thing a motion gate can diff.

    ``pixels`` is a flat, row-major tuple of luma ints in ``0..255`` (length ``w * h``). This is a
    deliberately tiny representation: the tier-0 gate never needs the full frame, only enough to
    tell "still" from "something moved". The real edge diffs a downscaled luma plane; the shape is
    the same.
    """

    w: int
    h: int
    pixels: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"frame dims must be positive, got {self.w}x{self.h}")
        if len(self.pixels) != self.w * self.h:
            raise ValueError(
                f"pixel count {len(self.pixels)} != {self.w}*{self.h}={self.w * self.h}"
            )


def gray_frame(w: int, h: int, fill: int = 0) -> Frame:
    """A uniform grayscale frame (handy for fixtures / a 'still scene' baseline)."""
    return Frame(w=w, h=h, pixels=tuple([_clamp_u8(fill)] * (w * h)))


def _clamp_u8(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else int(v)


def _as_frame(obj: Any) -> Frame:
    """Coerce a frame-ish object to a :class:`Frame`.

    Accepts a :class:`Frame` directly, or a ``(w, h, pixels)`` tuple. Anything else is a wire /
    contract mismatch and raises — the gate must diff a *known* representation, not guess.
    """
    if isinstance(obj, Frame):
        return obj
    if isinstance(obj, tuple) and len(obj) == 3:
        w, h, pixels = obj
        return Frame(w=int(w), h=int(h), pixels=tuple(int(p) for p in pixels))
    raise TypeError(f"unsupported frame type {type(obj).__name__!r}; expected Frame or (w,h,pixels)")


# ---------------------------------------------------------------------------
# motion_score — deterministic pixel-diff (still -> ~0, big change -> high)
# ---------------------------------------------------------------------------


def motion_score(
    prev_frame: Any,
    frame: Any,
    *,
    region: Box | None = None,
    stat: str = "mean",
    block_grid: tuple[int, int] = (8, 6),
) -> float:
    """Absolute-luma-difference salience between two frames, normalized to ``0..1``.

    Still scene -> ``~0``; a large change -> high (toward ``1.0``). Deterministic and pure stdlib;
    ``numpy`` is used only as a speed path and yields the identical value.

    ``prev_frame is None`` (the first frame, no predecessor) -> ``0.0``: there is nothing to diff,
    so the gate treats the very first frame as "no motion yet". Mismatched dimensions are a
    contract error and raise.

    ``region`` (a normalized :class:`~percept.adapters.Box`, the "watch-out zone") restricts the
    diff to that rectangle: motion outside the box is ignored, so only movement *inside* the zone
    can escalate.

    ``stat`` selects the localizing statistic over a coarse ``block_grid`` (``Gx x Gy``) partition of
    the active area (full frame, or ``region``):

    - ``"mean"`` (default) — whole-area mean abs-diff. **Byte-for-byte unchanged** from the pre-block
      path; the block branches are strictly opt-in and never touch this computation.
    - ``"block_max"`` — the single most-active block. A localized event (a hand, a cup) lights one
      block and clears threshold even when its whole-frame mean is ~0 (recall on a static camera).
    - ``"p95"`` — the 95th-percentile block (nearest-rank), a noise-robust cousin of ``block_max``.

    The block partition uses floor-of-product boundaries and integer per-block denominators so the
    Python and JS (``percept-edge``) results are bit-identical (a release-gated parity invariant).
    ``block_grid`` dims must be positive (a non-positive grid raises, exactly as an unknown ``stat``
    does) and are clamped down to the active area when finer than it, so the percentile always ranks
    a population of real blocks — see :func:`_partition`.
    """
    if prev_frame is None:
        return 0.0
    a = _as_frame(prev_frame)
    b = _as_frame(frame)
    if a.w != b.w or a.h != b.h:
        raise ValueError(f"frame size mismatch: {a.w}x{a.h} vs {b.w}x{b.h}")
    if stat != "mean":
        return _motion_score_blocks(a, b, region, block_grid, stat)
    if region is not None:
        return _motion_score_region(a, b, region)
    n = a.w * a.h
    if n == 0:  # pragma: no cover - Frame.__post_init__ forbids empty frames.
        return 0.0
    if _np is not None:
        pa = _np.frombuffer(bytes(a.pixels), dtype=_np.uint8).astype(_np.int16)
        pb = _np.frombuffer(bytes(b.pixels), dtype=_np.uint8).astype(_np.int16)
        total = int(_np.abs(pa - pb).sum())
    else:
        total = 0
        for pa, pb in zip(a.pixels, b.pixels):
            total += pa - pb if pa >= pb else pb - pa
    # 255 is the max per-pixel abs diff -> divide to land in 0..1.
    return total / (n * 255.0)


def _region_bounds(region: Box, w: int, h: int) -> tuple[int, int, int, int]:
    """Normalized :class:`Box` -> integer pixel bounds ``(x0, y0, x1, y1)`` clamped to the frame.

    Half-open in pixels (``x0..x1`` exclusive): a box covering the whole frame yields the full grid.
    A degenerate / off-frame box collapses to an empty span (``x0 == x1`` or ``y0 == y1``).
    """
    x0 = max(0, min(w, int(region.x * w)))
    y0 = max(0, min(h, int(region.y * h)))
    x1 = max(x0, min(w, int(round((region.x + region.w) * w))))
    y1 = max(y0, min(h, int(round((region.y + region.h) * h))))
    return x0, y0, x1, y1


def _motion_score_region(a: Frame, b: Frame, region: Box) -> float:
    """Mean absolute luma diff within ``region`` only (the watch-out zone), normalized to ``0..1``.

    Pixels outside the rectangle do not contribute. An empty / degenerate region scores ``0.0``
    (nothing to diff in-zone). Like the full-frame path, ``numpy`` is an optional speed path that
    returns the identical number.
    """
    w, h = a.w, a.h
    x0, y0, x1, y1 = _region_bounds(region, w, h)
    rw, rh = x1 - x0, y1 - y0
    n = rw * rh
    if n == 0:
        return 0.0
    if _np is not None:
        pa = _np.frombuffer(bytes(a.pixels), dtype=_np.uint8).astype(_np.int16).reshape(h, w)
        pb = _np.frombuffer(bytes(b.pixels), dtype=_np.uint8).astype(_np.int16).reshape(h, w)
        total = int(_np.abs(pa[y0:y1, x0:x1] - pb[y0:y1, x0:x1]).sum())
    else:
        total = 0
        for row in range(y0, y1):
            base = row * w
            for col in range(x0, x1):
                pav, pbv = a.pixels[base + col], b.pixels[base + col]
                total += pav - pbv if pav >= pbv else pbv - pav
    return total / (n * 255.0)


#: ADR-002 §2 reason code for the block partition the wired area cannot carry. A ``region_hint``
#: narrower than the grid is literally the ADR's own example of the shape ("a ROI outside frame
#: bounds") — a *configuration* failure, not a runtime one — and ADR-002 maps it to ``state``
#: directly rather than to a 0..1 coverage axis, because "this grid cannot be laid on this area" is
#: binary. Deliberately the same string as :data:`UNSUPPORTED_DETECTOR_REASON`: one closed
#: vocabulary, two occasions.
DEGENERATE_PARTITION_REASON = "PLAN_UNOBSERVABLE"


class DegenerateBlockPartitionWarning(UserWarning):
    """The requested block grid does not fit the active area, so a coarser one was used.

    Raised as a *warning*, not an error, for the same reason
    :class:`UnsupportedEdgeCapabilityWarning` is: the Watch still admits frames and still buys judge
    calls, so refusing the frame would be a worse answer than a degraded one. It is not silence
    because the localizing statistic really did change meaning — a ``p95`` over 24 real blocks is a
    different number from a ``p95`` over 48 blocks, half of which were empty padding, and an
    operator reading a suspiciously quiet ``p95`` has no other way to learn that their 10%-square
    ``region_hint`` is 6x4 px at the shipped 48x36 gate raster.

    Deliberately no companion log line: unlike the compile-time capability drop, this is reached on
    the per-frame hot path, and a ``logger.warning`` there is a 30-per-second log flood. Python's
    default warning filter shows one line per call site per configuration, which is the right
    cardinality for a fact that is a property of the plan, not of the frame.
    """


def _partition(
    x0: int, y0: int, x1: int, y1: int, gx: int, gy: int
) -> tuple[int, int, bool]:
    """The block grid actually laid on the active area ``[x0,x1) x [y0,y1)``, and whether it bit.

    ONE guard point, TWO deliberately distinct rules — the inputs look alike and are not:

    * **An impossible grid raises.** A non-positive ``Gx``/``Gy`` describes no partition at all;
      there is no salience it could plausibly be. It used to answer ``0.0``, which is a *reading* —
      "the scene is still" — for a frame that may have inverted end to end. That matches
      :func:`_motion_score_blocks`'s own refusal of an unknown ``stat`` name and
      :meth:`Frame.__post_init__`'s refusal of a non-positive frame dimension: an input outside the
      domain is a contract error, not a quiet zero.
    * **A grid finer than the area is CLAMPED, not padded.** ``Gx x Gy`` blocks over an ``aw x ah``
      area with ``aw < Gx`` used to emit ``Gx - aw`` columns of 0-px blocks, each contributing a
      literal ``0.0`` to the percentile population. That is a valid, authored input — a 10%-square
      ``region_hint`` is 6x4 px at the shipped 48x36 gate raster — so it must not raise; but the
      padding made the population fictional. Clamping to ``min(G, a)`` means every block holds at
      least one real pixel and the percentile ranks a real sample.

    Returns ``(effective_gx, effective_gy, degenerate)``. ``degenerate`` is True exactly when the
    clamp bit, i.e. when the caller does not get the partition it asked for — the caller surfaces
    it as ADR-002's :data:`DEGENERATE_PARTITION_REASON`. An empty active area clamps to ``0``, which
    is also degenerate: a zero-extent ``region_hint`` is a permanent never-fire.
    """
    if gx <= 0 or gy <= 0:
        raise ValueError(f"block_grid dims must be positive, got {gx}x{gy}")
    aw, ah = x1 - x0, y1 - y0
    effective_gx = gx if gx < aw else aw
    effective_gy = gy if gy < ah else ah
    return effective_gx, effective_gy, (effective_gx != gx or effective_gy != gy)


def _block_scores(
    a: Frame, b: Frame, region: Box | None, block_grid: tuple[int, int]
) -> tuple[list[float], int, int, bool]:
    """The per-block salience list for the active area, plus ``(total_abs_diff, area_px, degenerate)``.

    ONE pass over the pixels; every salience statistic in this module is a pure selector over what
    this returns (``block_max`` = ``max``, ``p95`` = nearest-rank, ``mean`` =
    ``total / (area_px * 255)``, because the blocks tile the area exactly with no gaps or overlaps).
    That is what lets :func:`salience_stats` price all three statistics for the cost of one — and
    what guarantees ``percept doctor``'s counterfactual table cannot disagree with the live gate.

    Block boundaries are floor-of-product (``bx(i) = x0 + (i * aw) // Gx``): blocks differ by <=1 px
    and are bit-identical to the JS edge (``i * aw`` stays far under 2**53). Each block's score is
    ``sum_abs_diff / (block_px * 255)`` — an integer sum over an integer denominator, so numpy and
    stdlib return the same float.

    The grid comes from :func:`_partition`, which clamps it to the active area, so **no block is
    ever empty**: with ``Gx <= aw`` every column band spans at least ``aw // Gx >= 1`` px, and the
    same for rows. The 0-px branch this used to carry is not defensive, it is unreachable.
    """
    w, h = a.w, a.h
    if region is not None:
        x0, y0, x1, y1 = _region_bounds(region, w, h)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    aw, ah = x1 - x0, y1 - y0
    gx, gy, degenerate = _partition(x0, y0, x1, y1, int(block_grid[0]), int(block_grid[1]))
    if gx <= 0 or gy <= 0:  # an empty / off-frame active area — nothing in-zone to diff.
        return [], 0, 0, degenerate
    diff = None
    if _np is not None:
        pa = _np.frombuffer(bytes(a.pixels), dtype=_np.uint8).astype(_np.int16).reshape(h, w)
        pb = _np.frombuffer(bytes(b.pixels), dtype=_np.uint8).astype(_np.int16).reshape(h, w)
        diff = _np.abs(pa - pb)
    scores: list[float] = []
    grand_total = 0
    for j in range(gy):
        by0 = y0 + (j * ah) // gy
        by1 = y0 + ((j + 1) * ah) // gy
        for i in range(gx):
            bx0 = x0 + (i * aw) // gx
            bx1 = x0 + ((i + 1) * aw) // gx
            block_px = (bx1 - bx0) * (by1 - by0)
            if diff is not None:
                total = int(diff[by0:by1, bx0:bx1].sum())
            else:
                total = 0
                for row in range(by0, by1):
                    base = row * w
                    for col in range(bx0, bx1):
                        pav, pbv = a.pixels[base + col], b.pixels[base + col]
                        total += pav - pbv if pav >= pbv else pbv - pav
            grand_total += total
            scores.append(total / (block_px * 255.0))
    return scores, grand_total, aw * ah, degenerate


def _warn_degenerate_partition(
    region: Box | None, block_grid: tuple[int, int], caller: str
) -> None:
    """Report a clamped partition once per call site (see :class:`DegenerateBlockPartitionWarning`)."""
    warnings.warn(
        f"{caller} asked for a {int(block_grid[0])}x{int(block_grid[1])} block grid over an active "
        "area too small to carry it"
        + (f" (region_hint {region!r})" if region is not None else "")
        + "; the grid was clamped to the area, so the statistic is computed over a coarser "
        f"partition than the one configured ({DEGENERATE_PARTITION_REASON})",
        DegenerateBlockPartitionWarning,
        stacklevel=3,
    )


def _motion_score_blocks(
    a: Frame, b: Frame, region: Box | None, block_grid: tuple[int, int], stat: str
) -> float:
    """Localized salience: partition the active area into blocks (:func:`_block_scores`), select
    per ``stat``. ``block_max`` = the top block; ``p95`` = the nearest-rank 95th-percentile block
    (integer rank, no float percentile). A clamped partition still scores — it warns."""
    scores, _total, _px, degenerate = _block_scores(a, b, region, block_grid)
    if degenerate:
        _warn_degenerate_partition(region, block_grid, f"motion_score(stat={stat!r})")
    if not scores:
        return 0.0
    if stat == "block_max":
        return max(scores)
    if stat == "p95":
        return _nearest_rank_p95(scores)
    raise ValueError(f"unknown salience stat {stat!r}; expected 'mean', 'block_max', or 'p95'")


#: The salience statistics :func:`salience_stats` reports, in the order a diagnostic renders them.
#: They live on DIFFERENT VALUE SCALES and each carries its own calibrated cutoff
#: (``EdgeSettings.motion_gate`` / ``motion_gate_block_max`` / ``motion_gate_p95``) — comparing them
#: to one threshold is a category error, so nothing here ever returns a single "the" salience.
SALIENCE_STATS: tuple[str, ...] = ("mean", "block_max", "p95")


def salience_stats(
    prev_frame: Any,
    frame: Any,
    *,
    region: Box | None = None,
    block_grid: tuple[int, int] = (8, 6),
) -> dict[str, float]:
    """ALL three salience statistics for one frame pair, from ONE pass over the pixels.

    Returns ``{"mean": ..., "block_max": ..., "p95": ...}``. Each value is bit-identical to
    :func:`motion_score` called with that ``stat`` (a release-gated parity invariant — see
    ``tests/test_doctor.py``), so a diagnostic that prints all three and a gate that escalates on one
    can never disagree about what the frame looked like.

    ``prev_frame is None`` -> all zeros (no predecessor to diff against).
    """
    if prev_frame is None:
        return {stat: 0.0 for stat in SALIENCE_STATS}
    a = _as_frame(prev_frame)
    b = _as_frame(frame)
    if a.w != b.w or a.h != b.h:
        raise ValueError(f"frame size mismatch: {a.w}x{a.h} vs {b.w}x{b.h}")
    scores, total, area_px, degenerate = _block_scores(a, b, region, block_grid)
    if degenerate:
        _warn_degenerate_partition(region, block_grid, "salience_stats")
    if not scores or area_px == 0:
        return {stat: 0.0 for stat in SALIENCE_STATS}
    return {
        # The blocks tile the active area exactly, so summing their integer numerators over the
        # integer area denominator IS the whole-area mean motion_score computes on its own path.
        "mean": total / (area_px * 255.0),
        "block_max": max(scores),
        "p95": _nearest_rank_p95(scores),
    }


def _nearest_rank_p95(scores: list[float]) -> float:
    """The 95th-percentile value by the nearest-rank method, with an integer-only rank.

    ``idx = ceil(0.95 * n) - 1`` computed as ``(95 * n + 99) // 100 - 1`` so it is identical in
    Python and JS (no float percentile / interpolation, which would drift across languages).
    """
    if not scores:
        return 0.0
    s = sorted(scores)
    n = len(s)
    idx = (95 * n + 99) // 100 - 1
    if idx < 0:
        idx = 0
    elif idx >= n:
        idx = n - 1
    return s[idx]


def ego_residual_score(
    prev_frame: Any,
    frame: Any,
    *,
    k: int = 4,
    region: Box | None = None,
    stat: str = "mean",
    block_grid: tuple[int, int] = (8, 6),
) -> float:
    """Ego-motion-compensated salience after a mean-estimated integer global shift.

    Estimates and removes whole-frame translation (camera pan / head turn) by searching integer
    shifts ``(dx, dy)`` in ``[-k, k]`` and choosing the shift with the smallest whole-frame MAD.
    That mean-based estimate is independent of the goal and therefore remains a statement about
    camera motion rather than about whichever local block happens to be active. The already-aligned
    residual is then summarized with ``stat`` (``mean`` / ``block_max`` / ``p95``), on the same value
    scale as :func:`motion_score`. A pure camera pan is explained by some shift, so every residual
    statistic is ``~0``; localized subject motion remains localized after alignment and can therefore
    clear a block-stat threshold. Deterministic and O(k**2 * pixels).

    The default ``stat="mean"`` path is byte-for-byte the legacy result. Its ``(0, 0)`` candidate
    equals the uncompensated mean, so that result is always ``<=`` the full-frame mean
    :func:`motion_score`. Block statistics intentionally make no such cross-stat comparison: they
    live on their own calibrated scales and must be compared to their own cutoffs.

    ``region`` (the watch-out zone) restricts **what is measured**, never **what is estimated**. The
    shift search still runs over the whole frame, because a pan is a whole-frame property: a zone
    can be too small, too flat, or too busy to identify a translation, and letting it pick the shift
    would mean each zone reporting its own private opinion of how the camera moved. Once the global
    shift is chosen, the residual sum *and its denominator* cover only ``region ∩ overlap``, so
    movement outside the zone can no longer escalate the goal. ``region=None`` (the default) is the
    whole-frame path, byte-for-byte unchanged. A zone that misses the winning shift's overlap
    entirely -> ``0.0`` (nothing in-zone to measure), matching :func:`_motion_score_region`.
    ``block_grid`` has the same meaning and validation as on :func:`motion_score`.

    Without ``region`` this silently discarded the zone: the gate's ego branch returns before its
    localized branch, so a goal carrying BOTH ego compensation and a watch-out zone escalated on
    motion entirely outside the zone, while still reporting the zone to the caller.

    ``prev_frame is None`` -> ``0.0`` (no predecessor). Mirrors the JS ``egoResidualScore`` bit-for-bit
    (mean-estimated shift, then identical aligned-residual partitioning).
    """
    if prev_frame is None:
        return 0.0
    a = _as_frame(prev_frame)
    b = _as_frame(frame)
    if a.w != b.w or a.h != b.h:
        raise ValueError(f"frame size mismatch: {a.w}x{a.h} vs {b.w}x{b.h}")
    w, h = a.w, a.h
    kk = int(k)
    if kk < 0:
        kk = 0
    grid_a = grid_b = None
    if _np is not None:
        grid_a = _np.frombuffer(bytes(a.pixels), dtype=_np.uint8).astype(_np.int16).reshape(h, w)
        grid_b = _np.frombuffer(bytes(b.pixels), dtype=_np.uint8).astype(_np.int16).reshape(h, w)
    best: float | None = None
    best_shift: tuple[int, int] | None = None
    for dy in range(-kk, kk + 1):
        oh = h - (dy if dy >= 0 else -dy)  # overlap height = h - |dy|
        if oh <= 0:
            continue
        fy0 = dy if dy > 0 else 0
        for dx in range(-kk, kk + 1):
            ow = w - (dx if dx >= 0 else -dx)  # overlap width = w - |dx|
            if ow <= 0:
                continue
            fx0 = dx if dx > 0 else 0
            npx = ow * oh
            if grid_a is not None:
                fb = grid_b[fy0:fy0 + oh, fx0:fx0 + ow]
                pa = grid_a[fy0 - dy:fy0 - dy + oh, fx0 - dx:fx0 - dx + ow]
                total = int(_np.abs(pa - fb).sum())
            else:
                total = 0
                for fy in range(fy0, fy0 + oh):
                    fb_base = fy * w
                    pa_base = (fy - dy) * w
                    for fx in range(fx0, fx0 + ow):
                        pav = a.pixels[pa_base + (fx - dx)]
                        pbv = b.pixels[fb_base + fx]
                        total += pav - pbv if pav >= pbv else pbv - pav
            score = total / (npx * 255.0)
            if best is None or score < best:
                best = score
                best_shift = (dx, dy)
    if best is None or best_shift is None:
        return 0.0
    if stat == "mean" and region is None:
        return best
    return _ego_aligned_residual(
        a,
        b,
        region=region,
        stat=stat,
        block_grid=block_grid,
        dx=best_shift[0],
        dy=best_shift[1],
    )


def _ego_aligned_residual(
    a: Frame,
    b: Frame,
    *,
    region: Box | None,
    stat: str,
    block_grid: tuple[int, int],
    dx: int,
    dy: int,
) -> float:
    """Summarize an ALREADY-ALIGNED residual over ``region ∩ overlap`` using ``stat``.

    Two rectangles in ``frame`` (``b``) coordinates: the overlap the shift leaves behind, and the
    watch-out zone. Their intersection is the only place the zone's promise ("movement elsewhere is
    ignored") and the shift's validity ("both frames have a pixel here") both hold. Cropping that
    intersection into two same-size frames lets :func:`motion_score` be the single implementation of
    mean / block_max / p95 partition semantics. The shift was chosen globally before this function;
    a local block can affect the residual statistic but can never choose the camera transform.
    """
    w, h = a.w, a.h
    if region is None:
        rx0, ry0, rx1, ry1 = 0, 0, w, h
    else:
        rx0, ry0, rx1, ry1 = _region_bounds(region, w, h)
    # The overlap rectangle in frame coordinates: x in [fx0, fx0+ow), y in [fy0, fy0+oh).
    ox0 = dx if dx > 0 else 0
    oy0 = dy if dy > 0 else 0
    ox1 = ox0 + w - (dx if dx >= 0 else -dx)
    oy1 = oy0 + h - (dy if dy >= 0 else -dy)
    mx0, my0 = (rx0 if rx0 > ox0 else ox0), (ry0 if ry0 > oy0 else oy0)
    mx1, my1 = (rx1 if rx1 < ox1 else ox1), (ry1 if ry1 < oy1 else oy1)
    mw, mh = mx1 - mx0, my1 - my0
    # Per DIMENSION, not on the product: a zone sitting outside the overlap on BOTH axes makes mw
    # and mh negative, and their product positive again. The loops below would still be empty (so
    # the answer is 0.0 either way), but a positive `npx` from two empty spans is not something to
    # leave standing in a denominator.
    if mw <= 0 or mh <= 0:
        return 0.0
    aligned_prev: list[int] = []
    aligned_cur: list[int] = []
    for fy in range(my0, my1):
        fb_base = fy * w
        pa_base = (fy - dy) * w
        for fx in range(mx0, mx1):
            aligned_prev.append(a.pixels[pa_base + (fx - dx)])
            aligned_cur.append(b.pixels[fb_base + fx])
    prev_crop = Frame(w=mw, h=mh, pixels=tuple(aligned_prev))
    cur_crop = Frame(w=mw, h=mh, pixels=tuple(aligned_cur))
    return motion_score(prev_crop, cur_crop, stat=stat, block_grid=block_grid)


# ---------------------------------------------------------------------------
# Civil time — the injected clock the schedule gate consults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CivilTime:
    """The civil moment a :class:`~percept.adapters.Schedule` is checked against.

    Deliberately just ``(weekday, minute_of_day)`` so the gate's schedule decision is *pure*: no
    timezone, no wall-clock read inside the gate. The wall-clock -> civil conversion lives at the
    boundary (:func:`civil_now` / :func:`civil_from_epoch`), injected as a callable, so pure logic
    stays deterministic and testable. ``weekday`` is ISO (Monday=1 … Sunday=7).
    """

    weekday: int
    minute_of_day: int


def civil_from_epoch(epoch_s: float, *, tz: timezone = timezone.utc) -> CivilTime:
    """Convert epoch seconds to a :class:`CivilTime` in ``tz`` (default UTC). Boundary helper."""
    dt = datetime.fromtimestamp(epoch_s, tz=tz)
    return CivilTime(weekday=dt.isoweekday(), minute_of_day=dt.hour * 60 + dt.minute)


def civil_now(*, tz: timezone = timezone.utc) -> CivilTime:
    """Read the wall clock NOW as a :class:`CivilTime` — the one place wall-clock is read (boundary)."""
    return civil_from_epoch(datetime.now(tz=tz).timestamp(), tz=tz)


def schedule_active(schedule: Schedule | None, now: CivilTime | None) -> bool:
    """Is a condition armed at ``now``? Opt-in: a ``None`` schedule OR a ``None`` clock -> always armed.

    The two ``None`` short-circuits keep the default path (no schedule, or no injected clock) byte-for-byte
    identical to the schedule-less gate — schedule enforcement engages **only** when both a schedule and a clock are present.
    """
    if schedule is None or now is None:
        return True
    return schedule.is_active(weekday=now.weekday, minute_of_day=now.minute_of_day)


# ---------------------------------------------------------------------------
# compile_watchspec — goals DOWN to a WatchSpec (the contract the edge receives)
# ---------------------------------------------------------------------------

# Per-goal threshold bias lives on the goal's mutable state under this key (optional). The
# conductor (or a tuning sweep) can bias a noisy goal up / a critical goal down; absent -> 0.0.
MOTION_BIAS_KEY = "motion_bias"
# Optional per-goal region-of-interest hint (a normalized Box dict or Box) under this key.
REGION_HINT_KEY = "region_hint"
# Optional per-goal arming Schedule (a Schedule, or its windows/days dict) under this key.
SCHEDULE_KEY = "schedule"
# Optional per-goal temporal cadence: {"min_judge_interval": float, "gate_on": motion|structural|
# semantic|fixed|fused} under this key. Controls how OFTEN to look (distinct from PerceptionSpec =
# how DEEP).
CADENCE_KEY = "cadence"
# Optional per-goal salience MODE overrides (tracks the camera mount). Fall back to the deployment
# default passed to compile_watchspec (from settings.edge) when absent / malformed.
SALIENCE_STAT_KEY = "salience_stat"
EGO_COMPENSATE_KEY = "ego_compensate"
#: The closed ``gate_on`` vocabulary, ONE copy. It is :data:`percept.behavior._VALID_GATE_ON` itself,
#: not a duplicate of it — this module used to carry a byte-identical second copy beside a
#: byte-identical second cadence coercer, held equal only by a drift test.
_GATE_ON_VOCAB = _VALID_GATE_ON
_SALIENCE_STAT_VOCAB = {"mean", "block_max", "p95"}
#: Which channel a Watch was admitted ON. ``"fused"`` is a first-class channel, not a mode: a
#: ``wake_policy`` used to override every other branch and then report the admission as the goal's own
#: ``gate_on``, so a receipt could name ``semantic`` for a frame whose measured semantic value was
#: 0.0. A goal now OPTS IN to fusion by authoring ``gate_on="fused"``, and the cause it reports is the
#: channel the number actually came from.
AdmissionCause = Literal["motion", "structural", "semantic", "fixed", "fused"]
ADMISSION_CAUSES: tuple[AdmissionCause, ...] = (
    "motion", "structural", "semantic", "fixed", "fused",
)
NonAdmissionReason = Literal[
    "signal_below_threshold",
    "schedule_inactive",
    "cadence_limited",
    "fixed_cadence_unconfigured",
]
NON_ADMISSION_REASONS: tuple[NonAdmissionReason, ...] = (
    "signal_below_threshold",
    "schedule_inactive",
    "cadence_limited",
    "fixed_cadence_unconfigured",
)


class UnsupportedEdgeCapabilityWarning(UserWarning):
    """The connected edge cannot run a detector the brain selected, so a boundary can never arrive.

    Raised as a *warning*, not an error: the Watch still admits frames and still buys judge calls, so
    aborting the compile would be a worse answer than a degraded one. But the silence it replaces was
    the defect — ``capabilities=None`` ("run the whole registry") and ``capabilities=[]`` ("run
    nothing") are opposite instructions, and the second one used to compile a detector-less condition
    without a word. A deployment that wants the drop to be fatal filters this to ``"error"``.
    """


#: ADR-002 §2 reason code for a plan the wired configuration cannot answer in principle. A dropped
#: detector is exactly that shape — not a runtime failure, a configuration one — and ADR-002 maps it
#: to ``state`` directly rather than to a 0..1 coverage axis, because "this boundary can never
#: arrive" is binary.
UNSUPPORTED_DETECTOR_REASON = "PLAN_UNOBSERVABLE"

#: The health-event kind :class:`GatedWatchLoop` emits alongside the warning and the log line — the
#: same three-channel report ``_preflight_routes`` makes for an undriven behavior route.
UNSUPPORTED_DETECTOR_HEALTH_KIND = "unsupported_edge_detectors"


def watchspec_kwargs(settings: Settings) -> dict[str, Any]:
    """The deployment's salience configuration, as :func:`compile_watchspec` keyword arguments.

    ``compile_watchspec``'s ``salience_stat`` / ``ego_compensate`` / ``stat_thresholds`` defaults
    (``'mean'`` / ``False`` / ``None``) are **signature defaults, not the product's defaults** — the
    shipped deployment is ``block_max`` at ``motion_gate_block_max``. Any caller that compiles
    without passing them silently gates on a different statistic, at a different cutoff, than the one
    ``settings.edge`` configured; every number it then measures describes a configuration nobody
    deployed. There is one right answer, so it lives in one place and every compile site reads it
    from here.

    A caller with a per-Watch override (a bound WatchPlan) **overlays** onto this mapping rather than
    replacing it — a plan that names one field must not silently revert the other three to the
    signature defaults.
    """
    edge = settings.edge
    return {
        "threshold": edge.motion_gate,
        "salience_stat": edge.salience_stat,
        "ego_compensate": edge.ego_compensate,
        "stat_thresholds": {
            "mean": edge.motion_gate,
            "block_max": edge.motion_gate_block_max,
            "p95": edge.motion_gate_p95,
        },
    }


def _capability_traversal(
    goals: Iterable[Goal], edge_capabilities: Iterable[str] | None
) -> list[tuple[Goal, list[str], list[str]]]:
    """THE per-goal traversal: which goals become conditions, and how the capability filter split them.

    Returns ``(goal, supported, dropped)`` for every goal that needs a WatchCondition. One
    implementation of one rule: :func:`compile_watchspec` (the compiling half) and
    :func:`watchspec_unsupported` (the reporting half) both read *this*. They used to be two
    independent walks each recomputing ``needs_watch_condition`` + ``detectors_for`` +
    ``partition_by_capability``, i.e. two implementations of one rule, free to drift apart on which
    goals even count as watch goals — and the compiling one threw its ``dropped`` half away.
    """
    return [
        (goal, *partition_by_capability(detectors_for(goal), edge_capabilities))
        for goal in goals
        if needs_watch_condition(goal_behavior_route(goal))
    ]


def compile_watchspec(
    goals: Iterable[Goal], *, threshold: float, version: int = 1,
    edge_capabilities: Iterable[str] | None = None,
    salience_stat: str = "mean", ego_compensate: bool = False,
    stat_thresholds: dict[str, float] | None = None,
    on_unsupported: Callable[[list[tuple[str, list[str]]]], None] | None = None,
) -> WatchSpec:
    """Compile active *watch* goals DOWN to a :class:`~percept.adapters.WatchSpec`.

    One :class:`~percept.adapters.WatchCondition` per watch goal:

    - ``goal_id`` <- ``goal.id``
    - ``anchor_text`` <- ``goal.condition`` (the plain-language anchor the edge salience-matches)
    - ``threshold`` <- a per-stat base cutoff (``stat_thresholds[salience_stat]``, else ``threshold``)
      + optional per-goal bias (``goal.state['motion_bias']``), clamped to ``0..1``. block_max/p95 use
      a higher base than the whole-frame mean (they sit on a higher value scale).
    - ``region_hint`` <- optional ``goal.state['region_hint']`` (a :class:`~percept.adapters.Box`)
    - ``schedule`` <- optional ``goal.state['schedule']`` (a :class:`~percept.adapters.Schedule`); absent
      -> ``None`` -> always armed (the gate skips the goal when the current time is out of schedule)
    - ``detectors`` <- the goal's selected edge skills (sense-preset + override), **filtered to**
      ``edge_capabilities`` when given (the capability handshake)

    ``edge_capabilities`` (optional): the skill set the connected edge advertised. When given, each goal's
    detectors are intersected with it so the brain never sends a skill the edge can't run; ``None``
    (default, in-process topology A — the edge runs the whole registry) sends the full selection.

    **A drop is reported, never swallowed.** A goal whose detectors were all filtered out compiles a
    condition that admits frames but can never emit a boundary, so the same traversal that filters
    also warns (:class:`UnsupportedEdgeCapabilityWarning`), logs, and — when the caller wired one —
    hands the whole ``[(goal_id, dropped)]`` list to ``on_unsupported`` so a structured health event
    can carry it (ADR-002 ``PLAN_UNOBSERVABLE``). ``on_unsupported`` is a callback rather than a new
    ``WatchSpec`` field on purpose: the WatchSpec shape is the wire contract the JS ``contract.js``
    mirrors, and a diagnostic does not belong on the wire. :func:`watchspec_unsupported` reports the
    same list without compiling, for a caller choosing a fallback before it arms anything.

    Every frame-worker route compiles: ordinary Watches, transitions, and process observation.
    External event, boundary, rhythm, and scheduler primitives never become frame Watches.
    """
    goals = list(goals)
    require_duration_routes(
        goals,
        driver="compile_watchspec",
        require_worker_cadence=True,
    )
    require_in_process_duration_topology(
        goals,
        driver="compile_watchspec",
        remote=edge_capabilities is not None,
    )
    conditions: list[WatchCondition] = []
    unsupported: list[tuple[str, list[str]]] = []
    for goal, supported, dropped in _capability_traversal(goals, edge_capabilities):
        if dropped:
            unsupported.append((goal.id, dropped))
        bias = _coerce_bias(goal.state.get(MOTION_BIAS_KEY), goal_id=goal.id)
        cadence_s, gate_on = _coerce_cadence(goal.state.get(CADENCE_KEY))
        # Salience MODE = deployment default (settings.edge) overlaid with a per-goal override. The
        # resolved value rides the wire so the edge honors it without reading Python settings.
        stat = _coerce_salience_stat(goal.state.get(SALIENCE_STAT_KEY), salience_stat)
        ego = _coerce_ego(goal.state.get(EGO_COMPENSATE_KEY), ego_compensate)
        # block_max/p95 live on a higher value scale than the mean, so they use their own base cutoff
        # (from ``stat_thresholds``); ``mean`` and any un-mapped stat fall back to ``threshold``.
        base = threshold if not stat_thresholds else stat_thresholds.get(stat, threshold)
        conditions.append(
            WatchCondition(
                goal_id=goal.id,
                anchor_text=goal.condition,
                threshold=_clamp01(base + bias),
                region_hint=_coerce_box(goal.state.get(REGION_HINT_KEY), goal_id=goal.id),
                detectors=supported,  # cheap edge skills the brain selected, filtered to what the edge can run
                schedule=_coerce_schedule(goal.state.get(SCHEDULE_KEY)),
                cadence_s=cadence_s,  # how OFTEN to look (distinct from PerceptionSpec = how DEEP)
                gate_on=gate_on,
                salience_stat=stat,
                ego_compensate=ego,
            )
        )
    if unsupported:
        for goal_id, dropped in unsupported:
            message = (
                f"goal {goal_id!r} selected edge detector(s) {dropped!r} the connected edge does "
                "not advertise; they were dropped, so this Watch admits frames but can never emit "
                f"a boundary from them ({UNSUPPORTED_DETECTOR_REASON})"
            )
            warnings.warn(message, UnsupportedEdgeCapabilityWarning, stacklevel=2)
            get_logger("percept.harness.tier0").warning("%s", message)
        if on_unsupported is not None:
            on_unsupported(unsupported)
    return WatchSpec(conditions=conditions, version=version)


def watchspec_unsupported(
    goals: Iterable[Goal], edge_capabilities: Iterable[str]
) -> list[tuple[str, list[str]]]:
    """``(goal_id, dropped_detectors)`` the brain selected but the advertised edge can't run — so the
    caller can pick a fallback (server-side detection / a coarser skill / skip). Empty when the edge
    supports everything selected. The reporting half of :func:`compile_watchspec`'s capability handshake.

    A thin read of :func:`_capability_traversal` — the SAME traversal :func:`compile_watchspec` runs
    — so the two halves cannot disagree about which goals are watch goals or which detectors a goal
    selected. Quiet by construction: this is the *asking* path, and a caller who asked has already
    been told.
    """
    return [
        (goal.id, dropped)
        for goal, _supported, dropped in _capability_traversal(goals, edge_capabilities)
        if dropped
    ]


class UnsupportedMotionBiasError(ValueError):
    """A goal's ``state['motion_bias']`` is not a real, finite number, so no cutoff can be computed.

    Typed and attributed — it names the goal and the field — because the untyped ``float()`` it
    replaces raised a bare stdlib ``ValueError``/``TypeError`` naming neither, from inside a compile
    of a whole batch of goals. That aborts every Watch in the batch (the module's own precedent for
    an unusable authored value: :func:`~percept.duration_routes.require_duration_routes` refuses the
    same way), and an operator staring at ``could not convert string to float: 'high'`` has no way to
    tell WHICH goal carried it.
    """

    def __init__(self, goal_id: str, value: Any, reason: str) -> None:
        self.goal_id = goal_id
        self.value = value
        super().__init__(
            f"compile_watchspec rejected goal {goal_id!r}: "
            f"state[{MOTION_BIAS_KEY!r}]={value!r} — {reason}"
        )


def _coerce_bias(obj: Any, *, goal_id: str) -> float:
    """Per-goal threshold bias -> float; absent -> ``0.0``; anything unusable -> a typed refusal.

    The last untrusted goal-state hint with no coercer. It is deliberately the *strict* member of the
    ``_coerce_*`` family — the others widen a malformed hint to a harmless default (no region, no
    schedule, no cadence), because for them "malformed" and "absent" are the same watch. A bias is
    not: it moves the admission cutoff, so silently reading it as ``0.0`` would ship a Watch tuned to
    a number nobody authored.

    ``bool`` is refused rather than accepted, which is the second real defect here: ``True`` passes
    ``float()`` as ``1.0``, saturating the cutoff to the clamp ceiling, and a Watch with a threshold
    of ``1.0`` admits **nothing, ever** — the loudest possible failure delivered in total silence.
    """
    if obj is None:
        return 0.0
    if isinstance(obj, bool):
        raise UnsupportedMotionBiasError(
            goal_id, obj,
            "a bool is not a bias (True reads as a full point of threshold, so the Watch would be "
            "compiled at a cutoff of 1.0 and admit nothing, ever)",
        )
    try:
        return _finite(f"goal {goal_id!r} state[{MOTION_BIAS_KEY!r}]", obj)
    except ValueError as exc:
        raise UnsupportedMotionBiasError(
            goal_id, obj, "the bias must be a finite int or float",
        ) from exc


class UnsupportedRegionHintError(ValueError):
    """An authored ``region_hint`` the gate cannot answer, named by goal and by field.

    Untyped and unattributed refusals were the defect in the sibling coercers (BUG-A-01-06); this one
    carries ``goal_id`` and the offending field so the author is told which goal to edit.
    """

    def __init__(self, message: str, *, goal_id: str | None = None, field: str | None = None) -> None:
        super().__init__(message)
        self.goal_id = goal_id
        self.field = field


def _coerce_box(obj: Any, *, goal_id: str | None = None) -> Box | None:
    """The AUTHORING boundary for a watch-out zone — the full Box contract, refused loudly.

    This mirrors the JS ``validateBox`` behind ``makeBox``/``parseBox`` term for term, because the two
    runtimes must answer "is this zone admissible?" identically or the same authored goal admits in the
    harness and not on device (FIND-PARITY-01). The rules are:

    * every field finite (inherited from :class:`~percept.contracts.Box`, which enforces it for
      derived boxes too),
    * every field within ``[0, 1]`` — these are normalized coordinates and nothing outside that range
      names a position in the frame,
    * ``w > 0 and h > 0`` — a zero-area zone selects no pixels, so the goal can never admit. Left
      unchecked it was a **permanent, undiagnosable never-fire** reachable from a valid WatchPlan.

    Deliberately NOT checked: ``x + w <= 1``. An overhang is clamped by ``_region_bounds`` to a defined
    answer, and a float-exact sum check would reject a legitimate full-width zone (``0.7 + 0.3`` is not
    ``1.0`` in binary floating point). The JS side makes the same exclusion for the same reason.
    """
    if obj is None:
        return obj
    if isinstance(obj, Box):
        box = obj
    elif isinstance(obj, dict):
        try:
            box = Box(x=obj["x"], y=obj["y"], w=obj["w"], h=obj["h"])
        except KeyError as exc:
            raise UnsupportedRegionHintError(
                f"goal {goal_id!r} region_hint is missing {exc.args[0]!r}; a zone needs x, y, w and h",
                goal_id=goal_id, field=str(exc.args[0]),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise UnsupportedRegionHintError(
                f"goal {goal_id!r} region_hint is not a usable zone: {exc}", goal_id=goal_id
            ) from exc
    else:
        return None  # pragma: no cover - defensive: unknown hint shape -> no hint.

    for name in ("x", "y", "w", "h"):
        value = float(getattr(box, name))
        if not 0.0 <= value <= 1.0:
            raise UnsupportedRegionHintError(
                f"goal {goal_id!r} region_hint.{name}={value!r} is outside [0, 1]; a watch-out zone is "
                "in normalized image coordinates",
                goal_id=goal_id, field=name,
            )
    if box.is_empty:
        raise UnsupportedRegionHintError(
            f"goal {goal_id!r} region_hint has zero area (w={box.w!r}, h={box.h!r}); it selects no "
            "pixels, so the goal could never admit a frame",
            goal_id=goal_id, field="w" if not box.w > 0.0 else "h",
        )
    return box


def _coerce_schedule(obj: Any) -> Schedule | None:
    """Goal-state schedule hint -> :class:`~percept.adapters.Schedule`; ``None`` -> always armed.

    Accepts a :class:`Schedule`, or a ``{'windows': [[s,e],...], 'days': [...]}`` dict (the wire form).
    An unknown shape -> ``None`` (always armed) so a malformed hint never silently restricts a watch.
    """
    if obj is None or isinstance(obj, Schedule):
        return obj
    if isinstance(obj, dict):
        windows = tuple((int(s), int(e)) for s, e in obj.get("windows", ()))
        days = obj.get("days")
        if days is None:
            return Schedule(windows=windows)
        return Schedule(windows=windows, days=tuple(int(d) for d in days))
    return None  # pragma: no cover - defensive: unknown schedule shape -> always armed.


# The cadence firewall used to live here TWICE — once in this module and once in ``percept.behavior``
# (so the pure-stdlib plan compiler could run it without importing the harness), held byte-identical
# only by a drift test. There is now one implementation: ``_coerce_cadence`` above is
# :func:`percept.behavior.coerce_authored_cadence`, imported. Its vocabulary is
# :data:`_GATE_ON_VOCAB`, which is that module's set object, so ``fused`` cannot be added to one
# authoring surface and not the other.


def _coerce_salience_stat(obj: Any, default: str) -> str:
    """Per-goal ``salience_stat`` override -> a valid stat; unknown / out-of-vocab -> ``default``.

    The firewall for the untrusted goal-state hint: only ``mean``/``block_max``/``p95`` pass; anything
    else falls back to the deployment default so a malformed hint never picks an invalid mode.
    """
    if obj in _SALIENCE_STAT_VOCAB:
        return obj  # type: ignore[return-value]
    return default if default in _SALIENCE_STAT_VOCAB else "mean"


def _coerce_ego(obj: Any, default: bool) -> bool:
    """Per-goal ``ego_compensate`` override -> bool; a non-bool hint -> the deployment ``default``."""
    if isinstance(obj, bool):
        return obj
    return bool(default)


def _condition_motion(
    prev: Frame | None, f: Frame, c: WatchCondition, full_motion: float
) -> float:
    """The escalate-comparison motion for one condition, honoring its salience MODE.

    The default (``salience_stat == "mean"``, no ``region_hint``, no ``ego_compensate``) returns the
    precomputed full-frame ``full_motion`` unchanged — byte-for-byte the pre-mode path. Otherwise ego
    compensation aligns the frames before applying that same statistic (optionally masked to
    ``region_hint``); the non-ego branch applies it directly.

    Ego alignment and the salience statistic are deliberately separate operations: the global shift
    is estimated with whole-frame mean residual, then the aligned residual is summarized with the
    condition's selected statistic. That keeps ``block_max@0.20`` and ``p95@0.15`` on the scales their
    compiled thresholds name instead of silently comparing a mean residual to a block cutoff.
    """
    if c.ego_compensate:
        return ego_residual_score(prev, f, region=c.region_hint, stat=c.salience_stat)
    if c.salience_stat != "mean" or c.region_hint is not None:
        return motion_score(prev, f, region=c.region_hint, stat=c.salience_stat)
    return full_motion


def cheap_signals(
    *,
    motion: float,
    structural: float,
    semantic: float,
    energy: float | None = None,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    """The gate's cheap-channel vector — the named producer the wake fusion consumes.

    THE INVARIANT: **one physical measurement populates exactly one channel.** Four measurements, four
    channels, and no aliases.

    ==============  ============================================================
    channel         measurement
    ==============  ============================================================
    ``motion``      the frame-diff gauge (always available)
    ``structural``  the structural adapter, ``0.0`` when none is wired
    ``semantic``    the semantic adapter, ``0.0`` when none is wired
    ``audio``       ``Tier0Signal.energy``; ABSENT when no audio source is open
    ==============  ============================================================

    This used to build SEVEN keys from those four measurements — ``change`` aliased ``motion``,
    ``semantic_relevance`` aliased ``semantic``, and ``audio_event`` aliased ``audio``. The noisy-OR
    in :func:`~percept.harness.detectors.wake_score` assumes its channels are INDEPENDENT cues, so an
    alias fuses one measurement with itself: a policy weighting ``motion`` and ``change`` at 0.5 each
    scored 0.51 on one reading of 0.6, against 0.30 for the same reading on the single channel. The
    duplicate was worth more than the measurement.

    ``audio_event`` is absent rather than aliased for a second reason: it names an ACOUSTIC-EVENT
    class, and ``detectors_in_category(CATEGORY_AUDIO_EVENT)`` is empty — nothing in this build
    classifies acoustic events. Mapping raw energy onto it would have claimed a sense that does not
    exist. A weight on an absent channel now contributes nothing, which is the honest answer.

    ``audio`` is OMITTED, not zeroed, when ``energy is None``: "no microphone is open" and "the room
    is silent" are different facts, and the fusion reads an absent channel as "no vote" rather than as
    evidence of quiet.

    ``extra`` is the caller's overlay (the gate's ``signals=`` argument) — the supported way to add a
    channel a real detector measures (``presence``, ``pose``, an acoustic-event classifier). It is
    applied last, so a caller measuring a channel itself overrides the gate's value for it.
    """
    out: dict[str, float] = {
        "motion": motion,
        "structural": structural,
        "semantic": semantic,
    }
    if energy is not None:
        out["audio"] = energy
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# MotionGate — consumes a WatchSpec + a frame, emits a Tier0Signal + a decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchAdmissionOutcome:
    """The gate's complete policy outcome for one Watch on one accepted unit.

    Live lifecycle consumers use this typed result instead of re-running schedule, cadence, signal,
    or fixed-cadence policy outside :class:`MotionGate`.
    """

    goal_id: str
    admitted: bool
    cause: AdmissionCause | None
    non_admission_reason: NonAdmissionReason | None
    score: float
    threshold: float
    admission_id: str
    detector_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.goal_id, str) or not self.goal_id:
            raise ValueError("WatchAdmissionOutcome goal_id must be a non-empty string")
        if not isinstance(self.admitted, bool):
            raise TypeError("WatchAdmissionOutcome admitted must be a bool")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("WatchAdmissionOutcome score must be in [0, 1]")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("WatchAdmissionOutcome threshold must be in [0, 1]")
        if self.admitted:
            if self.cause not in ADMISSION_CAUSES or self.non_admission_reason is not None:
                raise ValueError("an admitted outcome needs one cause and no non-admission reason")
        elif self.cause is not None or self.non_admission_reason not in NON_ADMISSION_REASONS:
            raise ValueError("a non-admitted outcome needs one reason and no admission cause")
        for name in ("admission_id", "detector_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"WatchAdmissionOutcome {name} must be a non-empty string")


def _watch_admission_outcome(
    condition: WatchCondition,
    *,
    admitted: bool,
    cause: AdmissionCause | None,
    non_admission_reason: NonAdmissionReason | None,
    score: float,
) -> WatchAdmissionOutcome:
    """Build the typed lifecycle-only view of one already-computed gate outcome.

    ``detector_id`` names WHAT produced the score. A ``fused`` admission names the wake policy rather
    than a tier-0 channel, because the number is the noisy-OR over several channels and no single
    ``tier0.signal.*`` produced it.
    """
    detector_id = (
        f"motion_score.{condition.salience_stat}"
        if condition.gate_on == "motion"
        else (
            "cadence.fixed"
            if condition.gate_on == "fixed"
            else (
                "wake_policy.noisy_or"
                if condition.gate_on == "fused"
                else f"tier0.signal.{condition.gate_on}"
            )
        )
    )
    return WatchAdmissionOutcome(
        goal_id=condition.goal_id,
        admitted=admitted,
        cause=cause,
        non_admission_reason=non_admission_reason,
        score=score,
        threshold=condition.threshold,
        admission_id=f"tier0.signal.{condition.gate_on}",
        detector_id=detector_id,
    )


@dataclass(frozen=True)
class GateDecision:
    """One gate decision: the reported :class:`Tier0Signal` + whether to escalate.

    ``escalate`` True -> spend the cloud vision call; False -> skip (the still-scene economics).
    ``threshold`` is the effective cutoff actually applied (the lowest per-condition threshold in
    the spec, so the most-sensitive goal wins — any goal that wants to look gets to look).
    """

    signal: Tier0Signal
    escalate: bool
    threshold: float
    # Exact Watches whose own condition admitted this frame.  Empty when nothing admitted. Appended
    # and defaulted so existing positional construction stays compatible.
    goal_ids: tuple[str, ...] = ()
    # Ordered 1:1 with ``goal_ids``. A fixed cadence is reported as ``"fixed"`` and never relabelled
    # as motion salience.
    admission_causes: tuple[AdmissionCause, ...] = ()
    # One typed outcome for every condition in the gate, admitted or not. Empty unless a lifecycle
    # consumer explicitly requests it, and on legacy manually constructed decisions.
    outcomes: tuple[WatchAdmissionOutcome, ...] = ()
    # Content identity of the exact causal frame pair, computed only when explicitly requested.
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        if self.goal_ids and not self.admission_causes:
            # Additive compatibility: before typed causes existed, a non-empty ``goal_ids`` tuple
            # could only describe the historical motion/salience admission path.
            object.__setattr__(
                self,
                "admission_causes",
                tuple("motion" for _ in self.goal_ids),
            )
        if len(self.goal_ids) != len(self.admission_causes):
            raise ValueError("GateDecision goal_ids and admission_causes must be aligned 1:1")
        if len(set(self.goal_ids)) != len(self.goal_ids):
            raise ValueError("GateDecision must carry each admitted goal id exactly once")
        invalid = [cause for cause in self.admission_causes if cause not in ADMISSION_CAUSES]
        if invalid:
            raise ValueError(f"GateDecision admission cause(s) outside closed vocabulary: {invalid!r}")
        if self.outcomes:
            outcome_ids = tuple(outcome.goal_id for outcome in self.outcomes)
            if len(set(outcome_ids)) != len(outcome_ids):
                raise ValueError("GateDecision must carry one outcome per Watch")
            admitted = tuple(
                (outcome.goal_id, outcome.cause)
                for outcome in self.outcomes
                if outcome.admitted
            )
            if admitted != tuple(zip(self.goal_ids, self.admission_causes)):
                raise ValueError("GateDecision outcomes must align with admitted goal ids and causes")
        if self.evidence_id is not None and (
            not isinstance(self.evidence_id, str) or not self.evidence_id
        ):
            raise ValueError("GateDecision evidence_id must be a non-empty string or None")


# ``StreamTimeRewindWarning`` is defined in :mod:`percept._streamtime` beside the guard that raises
# it, and re-exported here (it is in this module's ``__all__``) so the existing
# ``from percept.harness.tier0 import StreamTimeRewindWarning`` callers keep working. It is shared
# rather than gate-local because the fovea (:mod:`percept.sampling`) chose the same RESET policy on
# the same untrusted timeline: one warning class means a caller catches every stream discontinuity,
# whichever seam noticed it first.


def _adapter_reading(value: Any) -> float:
    """One pluggable structural/semantic adapter's answer, as a usable salience in ``[0, 1]``.

    ``float()`` runs FIRST and its exceptions propagate: an adapter that answers ``None`` (or any
    junk ``float()`` refuses) takes the frame down loudly, which is documented intent — a broken
    adapter stops the watch rather than quietly scoring 0.0 forever.

    The finiteness firewall runs SECOND, and it absorbs: a NaN or an infinity is a *reading*, not a
    broken contract, and a NaN reading must be worth what "saw nothing" is worth. Left alone it was
    worth strictly less than nothing — every comparison against NaN is False, so the goal could not
    clear even a ``0.0`` cutoff, for the rest of the run. Matches the edge's JS ``clamp01``, which
    has always mapped a non-finite channel to 0; before this the two runtimes forked on exactly this
    input.
    """
    return _clamp01(float(value))


class MotionGate:
    """The coarse tier-0 gate: diff frames, report salience, decide escalate-vs-skip.

    Consumes a :class:`~percept.adapters.WatchSpec` (the anchors + thresholds the brain compiled
    down) and a frame stream. For each frame it computes ``motion`` against the previous frame,
    packs a :class:`~percept.adapters.Tier0Signal` (``structural``/``semantic`` = 0.0 in this
    motion-only reference), and escalates when a watched goal's motion clears its threshold.

    Effective threshold = the **minimum** per-condition threshold in the spec (the most sensitive
    goal wins), falling back to the gate default (``PERCEPT_MOTION_GATE``) when the spec is empty.
    An empty spec means no active watch goals -> nothing to look for -> never escalate.

    **Opt-ins (both strictly off by default — the no-region/no-schedule path is byte-for-byte
    the original gate):**

    - *watch-out zone* — a condition's ``region_hint`` masks the motion diff to that rectangle, so
      only movement *inside* the zone can escalate that goal (movement elsewhere is ignored).
    - *schedule* — a condition's ``Schedule`` arms it only at certain times; when the injected civil
      ``now`` is out of schedule the condition is skipped (it cannot escalate). Schedule enforcement
      engages only when a ``clock`` (or an explicit ``now``) is supplied — otherwise every goal is
      treated as always armed, exactly as before.
    """

    def __init__(
        self,
        spec: WatchSpec,
        *,
        external_conditions: Sequence[WatchCondition] = (),
        settings: Settings | None = None,
        clock: Callable[[], CivilTime] | None = None,
        wake_policy: WakePolicy | dict[str, Any] | None = None,
        structural_adapter: Callable[[Frame], float] | None = None,
        semantic_adapter: Callable[[Frame], float] | None = None,
    ) -> None:
        self._spec = spec
        self._external_conditions = tuple(external_conditions)
        cfg = settings or load_settings()
        self._default = cfg.edge.motion_gate
        self._prev: Frame | None = None
        # Per-goal last-escalation time, for the cadence cap / fixed-cadence floor. Empty when no
        # condition sets cadence_s -> the default motion path is byte-for-byte unchanged.
        self._last_escalate: dict[str, float] = {}
        # Injected civil clock for the schedule gate. None -> schedules are not enforced (default
        # path) unless a `now` is passed to evaluate(); we never read the wall clock in pure logic.
        self._clock = clock
        self._wake_policy = wake_policy
        # Optional channels that fill the structural/semantic Tier0 slots (CLIP/pose/privacy on the real
        # edge). None -> 0.0, so the reported signal and escalation are byte-for-byte the motion-only path.
        self._structural_adapter = structural_adapter
        self._semantic_adapter = semantic_adapter
        # The caller's stream timeline is untrusted (see StreamTimeGuard). Shared by evaluate() and
        # admit_external(), which share ``_last_escalate`` and therefore share one timeline.
        self._ts_guard = StreamTimeGuard()

    @property
    def spec(self) -> WatchSpec:
        return self._spec

    def threshold(self) -> float:
        """The effective escalation cutoff for the current spec (min over all conditions)."""
        if not self._spec.conditions:
            return self._default
        return min(c.threshold for c in self._spec.conditions)

    def reset(self) -> None:
        """Forget the previous frame (e.g. a stream discontinuity / new source).

        Also forgets the timeline baseline, so a *declared* restart (a new replay epoch, a new
        source) is not additionally reported as a rewind by :meth:`_gate_ts` — the caller already
        said the stream changed. Cadence policy (``_last_escalate``) is deliberately untouched, as
        it always has been.
        """
        self._prev = None
        self._ts_guard.reset()

    def adopt_state(self, prev: "MotionGate") -> None:
        """Carry a PRIOR gate's per-goal throttle + prev frame into this freshly-recompiled gate.

        A regate (a goal armed/cancelled mid-run) rebuilds the gate from scratch; without this, the new
        gate starts with an EMPTY ``_last_escalate`` — so arming ONE goal by voice released the per-goal
        cadence brake on ALL the others (they could re-escalate before their ``cadence_s`` elapsed). We
        copy the throttle ONLY for goal_ids still present in THIS gate's spec: a cancelled goal's brake is
        correctly forgotten, and a re-armed id keeps its cadence. The ``prev`` frame is carried too, so
        the first frame after a regate diffs against the real predecessor instead of a spurious 0-motion
        cold start. Deliberately touches ONLY ``_last_escalate`` + ``_prev`` — NOT any gate-key/dwell
        state (that is SPEC-C's territory)."""
        live_ids = {
            c.goal_id
            for c in (*self._spec.conditions, *self._external_conditions)
        }
        self._last_escalate = {
            gid: t for gid, t in prev._last_escalate.items() if gid in live_ids
        }
        self._prev = prev._prev

    @property
    def time_discontinuities(self) -> int:
        """How many backwards timestamp steps this gate absorbed — the traceable half of the reset.

        A silent reset is only half a fix: the cadence brake really was dropped, and an operator
        reading "this goal spent more than its cadence allows" needs to be able to see why. Non-zero
        means the source's timeline restarted at least once (a reconnect, a seek, a corrected host
        clock); each one also warned and logged when it happened.
        """
        return self._ts_guard.rewinds

    def _gate_ts(self, ts: Any) -> float:
        """Validate one caller-supplied stream timestamp before ANY temporal policy reads it.

        Two failures, two answers (see :class:`~percept.clock.StreamTimeGuard`):

        * **not finite** — refused loudly. A NaN does not tighten the cadence brake, it *bypasses*
          it: ``ts - last < cadence_s`` is False against a NaN, so the goal spends inside a window it
          was told not to, and then NaN is written into ``_last_escalate`` and poisons every later
          comparison for that goal. There is no sane admission decision to make on a NaN, and one
          bad timestamp must not be able to buy a judge call the deployment forbade.
        * **backwards** — a stream discontinuity, and treated as one: forget the per-goal cadence
          brake, exactly as :meth:`reset` forgets ``_prev``, and carry on. This is deliberately NOT
          :mod:`percept.interest`'s raise-on-rewind policy. A rewind here is a reconnect or a file
          seek — recoverable source noise — and raising would convert it into a dead watch. Leaving
          it alone is worse still: ``ts - last`` stays negative forever, so ``< cadence_s`` is always
          true and the goal is **permanently, silently muted** while frames keep arriving.

        The ``-inf`` seed is not a rewind: an unseen goal has no prior timeline, so a loop joined
        mid-stream still spends on its first qualifying frame whatever the wall time says.
        """
        verdict = self._ts_guard.check("ts", ts)
        value = float(ts)
        if verdict == "rewind":
            message = (
                f"tier-0 gate saw the stream timeline step backwards to ts={value!r}; treating it as "
                "a source discontinuity and clearing the per-goal cadence brake "
                f"({len(self._last_escalate)} goal(s)) so the affected Watches are not muted for the "
                "rest of the run"
            )
            warnings.warn(message, StreamTimeRewindWarning, stacklevel=3)
            get_logger("percept.harness.tier0").warning("%s", message)
            self._last_escalate.clear()
        return value

    def admit_external(
        self,
        goal_ids: Sequence[str],
        *,
        ts: float,
        now: CivilTime | None = None,
    ) -> tuple[str, ...]:
        """Admit an external wake (for example audio onset) through each Watch's schedule/cadence.

        The external signal decides *why* the worker asks, but it does not bypass the same temporal
        policy that guards visual admission. Accepted ids share ``_last_escalate`` with
        :meth:`evaluate`, so an audio wake cannot double-spend inside a goal's cadence window — and
        share :meth:`_gate_ts` with it too, so an external wake carrying a hostile ``ts`` cannot mute
        or over-spend a goal by a route the visual path is guarded against.
        """
        ts = self._gate_ts(ts)
        requested = frozenset(goal_ids)
        when = now if now is not None else (self._clock() if self._clock is not None else None)
        admitted: list[str] = []
        for condition in (*self._spec.conditions, *self._external_conditions):
            if (
                condition.goal_id not in requested
                or not schedule_active(condition.schedule, when)
            ):
                continue
            if (
                condition.cadence_s is not None
                and ts - self._last_escalate.get(condition.goal_id, float("-inf"))
                < condition.cadence_s
            ):
                continue
            admitted.append(condition.goal_id)
            self._last_escalate[condition.goal_id] = ts
        return tuple(admitted)

    def evaluate(
        self,
        frame: Any,
        *,
        ts: float,
        now: CivilTime | None = None,
        energy: float | None = None,
        signals: dict[str, float] | None = None,
        include_evidence_id: bool = False,
        include_outcomes: bool = False,
    ) -> GateDecision:
        """Score one frame against the prior, report a ``Tier0Signal``, decide escalate-vs-skip.

        An empty WatchSpec (no active watch goals) never escalates — there is nothing to look for.
        Otherwise escalate when an **armed** condition's motion (full-frame, or masked to its
        ``region_hint``) clears that condition's threshold. ``now`` (or the gate's injected ``clock``)
        is the civil time the per-condition :class:`~percept.adapters.Schedule` is checked against; a
        condition whose schedule is out of window at ``now`` is skipped. With no schedules and no
        ``region_hint`` this reduces to the original ``full_motion >= min(threshold)``. Advances the
        prev-frame cursor. ``include_outcomes`` and ``include_evidence_id`` are lifecycle-only opt-ins;
        the normal gate path allocates neither outcome DTOs nor causal hashes.
        """
        ts = self._gate_ts(ts)
        f = _as_frame(frame)
        prev = self._prev
        # Full-frame mean motion is always the REPORTED gauge (the Tier0Signal is a region-agnostic,
        # config-stable salience). Per-goal escalation may use a localized/ego-compensated statistic
        # instead (below), but the up-channel gauge stays the plain mean so it's comparable across configs.
        motion = motion_score(prev, f)
        # structural/semantic default to 0.0 (motion-only reference); optional adapters fill them
        # (CLIP/pose/privacy on the real edge). None -> 0.0 keeps the default path byte-for-byte.
        #
        # ``_clamp01`` is the SHARED firewall (percept._numeric), matching the edge's JS ``clamp01``:
        # a non-finite reading is 0.0, not a NaN. It must not be ``float()`` alone — a NaN compares
        # False against EVERY cutoff, including 0.0, so an adapter that hiccups once made its goal
        # permanently unable to admit and the two runtimes forked on what a broken channel is worth.
        # An adapter returning ``None`` still raises out of the gate: that is documented intent (a
        # broken adapter stops the watch loudly), and ``float(None)`` is where it raises.
        structural = (
            0.0 if self._structural_adapter is None
            else _adapter_reading(self._structural_adapter(f))
        )
        semantic = (
            0.0 if self._semantic_adapter is None
            else _adapter_reading(self._semantic_adapter(f))
        )
        channels = cheap_signals(
            motion=motion, structural=structural, semantic=semantic, energy=energy, extra=signals
        )
        fused = wake_score(channels, self._wake_policy) if self._wake_policy is not None else None
        when = now if now is not None else (self._clock() if self._clock is not None else None)
        # Armed = in-schedule conditions only; an out-of-schedule goal cannot escalate or set the cutoff.
        escalate = False
        admitted_goal_ids: list[str] = []
        admission_causes: list[AdmissionCause] = []
        outcomes: list[WatchAdmissionOutcome] | None = [] if include_outcomes else None
        armed: list[WatchCondition] = []
        for c in self._spec.conditions:
            if not schedule_active(c.schedule, when):
                if outcomes is not None:
                    outcomes.append(
                        _watch_admission_outcome(
                            c,
                            admitted=False,
                            cause=None,
                            non_admission_reason="schedule_inactive",
                            score=0.0,
                        )
                    )
                continue
            armed.append(c)
            # Cadence: a goal can't escalate more than once per cadence_s. For gate_on="fixed" this
            # cadence IS the escalation trigger (a floor — look every cadence_s even with no pixel motion,
            # so content events like a screenshare aren't silently under-sampled by the motion gate).
            if c.cadence_s is not None and ts - self._last_escalate.get(c.goal_id, float("-inf")) < c.cadence_s:
                if outcomes is not None:
                    outcomes.append(
                        _watch_admission_outcome(
                            c,
                            admitted=False,
                            cause=None,
                            non_admission_reason="cadence_limited",
                            score=0.0,
                        )
                    )
                continue
            # ONE branch per channel, each reached only by the goal that asked for it. The fused
            # branch used to sit ABOVE structural/semantic and fire on `fused is not None`, so a
            # configured wake_policy replaced every goal's channel — and the admission was then
            # recorded as `c.gate_on`, naming a channel whose measured value could be 0.0. A goal now
            # opts in with `gate_on="fused"` and is reported as cause `"fused"`.
            if c.gate_on == "fixed":
                score = 1.0 if c.cadence_s is not None else 0.0
                fired = c.cadence_s is not None  # cadence elapsed (checked above) -> escalate regardless of motion
            elif c.gate_on == "fused":
                # No wake_policy wired -> `wake_score` was never called, and a goal that asked to be
                # fused has nothing to be fused from. It scores 0.0 rather than silently falling back
                # to motion, the same way `gate_on="fixed"` with no cadence scores 0.0.
                score = fused if fused is not None else 0.0
                fired = score >= c.threshold
            elif c.gate_on == "structural":
                score = structural
                fired = structural >= c.threshold
            elif c.gate_on == "semantic":
                score = semantic
                fired = semantic >= c.threshold
            else:  # "motion" (default) — full-frame, or the goal's localized / ego-compensated statistic
                score = _condition_motion(prev, f, c, motion)
                fired = score >= c.threshold
            if fired:
                escalate = True
                admitted_goal_ids.append(c.goal_id)
                admission_causes.append(c.gate_on)
                self._last_escalate[c.goal_id] = ts
                if outcomes is not None:
                    outcomes.append(
                        _watch_admission_outcome(
                            c,
                            admitted=True,
                            cause=c.gate_on,
                            non_admission_reason=None,
                            score=score,
                        )
                    )
            else:
                if outcomes is not None:
                    outcomes.append(
                        _watch_admission_outcome(
                            c,
                            admitted=False,
                            cause=None,
                            non_admission_reason=(
                                "fixed_cadence_unconfigured"
                                if c.gate_on == "fixed"
                                else "signal_below_threshold"
                            ),
                            score=score,
                        )
                    )
        self._prev = f
        # Reported cutoff = the most-sensitive ARMED goal's threshold (default when none is armed).
        thr = min((c.threshold for c in armed), default=self._default)
        signal = Tier0Signal(
            motion=motion, structural=structural, semantic=semantic, ts=ts, energy=energy
        )
        return GateDecision(
            signal=signal,
            escalate=escalate,
            threshold=thr,
            goal_ids=tuple(admitted_goal_ids),
            admission_causes=tuple(admission_causes),
            outcomes=tuple(outcomes) if outcomes is not None else (),
            evidence_id=_motion_evidence_id(prev, f) if include_evidence_id else None,
        )


# ---------------------------------------------------------------------------
# The wired loop — the proof point (compile -> gate -> perceive_judged -> report up)
# ---------------------------------------------------------------------------


@dataclass
class LoopReport:
    """The end-of-run accounting the Phase-6 cost demo prints.

    ``signals`` is the upward stream of :class:`~percept.adapters.Tier0Signal` (one per frame).
    ``fire_events`` collects whatever ``perceive_judged`` returned on escalated frames (flattened).
    """

    frames: int = 0
    vision_calls: int = 0  # escalated frames -> real perceive_judged calls actually spent
    escalations: int = 0
    fire_events: list = field(default_factory=list)
    signals: list[Tier0Signal] = field(default_factory=list)
    # edge boundaries the detectors emit (the count comes from HERE, not from a per-frame VLM call):
    # (goal_id, Boundary) pairs across the whole run.
    boundaries: list[tuple[str, Boundary]] = field(default_factory=list)
    # Opt-in VID lifecycle events over the REPORTED mean-motion gauge. Empty unless
    # ``GatedWatchLoop(interest=...)`` is supplied; these are admission-interest intervals, never
    # semantic Watch verdicts.
    interest_events: list[InterestIntervalEvent] = field(default_factory=list)
    # UNITS — (frame x goal), NOT frames, exactly like the runtime's RuntimeStats twin: one admitted
    # frame admits every goal that cleared its own bar, so both maps sum to the same number and that
    # number is >= ``escalations`` (the FRAME count), equal only on a single-goal run. ``frames`` and
    # ``escalations`` are the frame-unit terms; these two are the goal-unit ones.
    fixed_opportunities: int = 0
    admission_by_goal: dict[str, int] = field(default_factory=dict)
    admission_by_cause: dict[str, int] = field(default_factory=dict)
    # goal_id -> confirmed-rep tally for every ``type="rhythm"`` goal driven this run. This is the
    # rhythm sense's real output — the COUNT, which is state, not an event. ``fire_events`` stays
    # reserved for the sparse count MILESTONES an authored ``fire_when`` admitted (see
    # :mod:`percept.rhythm`'s "what an Incident means for a rhythm goal").
    rhythm_counts: dict[str, int] = field(default_factory=dict)

    def boundary_count(self, goal_id: str, *, kind: str | None = None) -> int:
        """How many boundaries a goal's detectors proposed (e.g. peaks = reps), optionally by kind."""
        return sum(
            1 for gid, b in self.boundaries if gid == goal_id and (kind is None or b.kind == kind)
        )

    @property
    def ungated_calls(self) -> int:
        """What a naive 'perceive every frame' loop would have spent (the baseline)."""
        return self.frames

    @property
    def reduction(self) -> float:
        """Fractional reduction in vision calls vs the ungated baseline (0..1)."""
        if self.frames == 0:
            return 0.0
        return 1.0 - (self.vision_calls / self.frames)


#: The driver name this loop reports itself as — one string, so the warning, the log line and the
#: health event can never name it three different ways (``percept.run``'s twin is the literal
#: ``"percept.run"``).
_GATED_LOOP_DRIVER = "GatedWatchLoop"


@dataclass(frozen=True)
class _FrameItem:
    frame: Any
    ts: float


def _normalize_stream(stream: Iterable[Any], *, clock_start: float, dt: float) -> list[_FrameItem]:
    """Accept ``frame`` or ``(ts, frame)`` items; synthesize timestamps where absent.

    A bare frame is a :class:`Frame` or a ``(w, h, pixels)`` len-3 tuple. A timestamped item is a
    ``(ts, frame)`` len-2 tuple whose first element is a number — the only len-2 form, so it is
    unambiguous against a (len-3) frame tuple.
    """
    items: list[_FrameItem] = []
    t = clock_start
    for entry in stream:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], (int, float)):
            ts, frame = entry
            items.append(_FrameItem(frame=frame, ts=float(ts)))
        else:
            items.append(_FrameItem(frame=entry, ts=t))
            t += dt
    return items


def _motion_evidence_id(previous: Frame | None, current: Frame) -> str:
    """Content identity for the exact ordered frame pair that caused one motion sample."""

    def _payload(frame: Frame) -> dict[str, Any]:
        return {"w": frame.w, "h": frame.h, "pixels": frame.pixels}

    payload = json.dumps(
        {
            "version": 2,
            "previous": None if previous is None else _payload(previous),
            "current": _payload(current),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "frame_" + hashlib.sha256(payload).hexdigest()


class GatedWatchLoop:
    """The wired WatchSpec↓ / Tier0Signal↑ loop against a live :class:`~percept.agent.Percept`.

    Construction compiles either the caller's explicitly-selected ``goals`` or the core's
    **active goals for this source** down to a :class:`WatchSpec` and arms a
    :class:`MotionGate` with it. :meth:`run` replays a frame stream: per frame it gates, reports a
    :class:`Tier0Signal` up, and calls ``core.perceive_judged`` **only on escalation** — the
    cloud call is spent exactly when salience clears threshold. Returns a :class:`LoopReport` with
    gated-vs-ungated call counts (the cost graph) and the upward signal stream.

    This is the proof point: ``compile_watchspec``'s output is consumed by the gate, and a
    ``Tier0Signal`` is reported on every frame.

    A Goal on this source whose route the frame loop does not drive is decided by
    :meth:`_preflight_routes`, on exactly ``percept.run``'s terms: an ``UNSUPPORTED`` route still
    refuses at construction, a host-driven one (``count``/``event``/``remind``) is reported and
    skipped so it cannot take the Watches beside it down. ``on_health`` (optional) receives the
    same ``undriven_behavior_route`` event that runtime emits; the warning and the log line are
    emitted whether or not it is wired.
    """

    def __init__(
        self,
        core: Any,
        goals: Sequence[Goal] | None = None,
        *,
        threshold: float | None = None,
        source: str = "camera",
        settings: Settings | None = None,
        edge_capabilities: Iterable[str] | None = None,
        clock: Callable[[], CivilTime] | None = None,
        wake_policy: WakePolicy | dict[str, Any] | None = None,
        interest: Tier0MotionInterest | None = None,
        on_health: Callable[[Any], None] | None = None,
    ) -> None:
        self._core = core
        cfg = settings or getattr(core, "settings", None) or load_settings()
        self._settings = cfg
        self._source = source
        self._on_health = on_health  # kept for run()'s mid-run rhythm-failure reporting, below
        self._skipped_routes: list[dict] = []  # host-driven Goals this loop reports and skips
        # An explicit list is already a caller selection (and historically supports logical source
        # aliases such as a default-"camera" Goal on a "camera:front" loop). Only the implicit
        # registry-discovery path owns source selection and must filter before route preflight.
        active = (
            list(goals)
            if goals is not None
            else [
                goal
                for goal in core.goals.active()
                if getattr(goal, "source", "camera") == source
            ]
        )
        # Refuse the UNDRIVABLE; report — never abort on — the host-driven. Same split, same table
        # and same three client channels as ``percept.run``'s preflight (see _preflight_routes).
        driven = self._preflight_routes(active, on_health=on_health)
        # A rhythm goal rides the SAME frame loop but never becomes a WatchCondition: it is measured
        # and counted, not admitted and judged. Split it out here so compile_watchspec keeps its
        # "one condition per watch goal" contract and the counting path is explicit.
        self._rhythm = RhythmDrive(core, [g for g in driven if drives_rhythm(goal_behavior_route(g))])
        # Arm-time pre-flight: RhythmDrive warmed each goal's pose provider above (construction), so a
        # goal whose provider could not be prepared (missing extra, missing model) is ALREADY refused
        # — never armed, never observed by run() below. Report it the same way _preflight_routes
        # reports an undriven route: logged, and a structured health event naming the goal and the
        # missing extra, never raised — one goal's missing pose extra must not abort this loop for
        # every unrelated Watch beside it.
        for _rhythm_goal_id, _rhythm_reason in self._rhythm.arm_failures.items():
            get_logger("percept.harness.tier0").warning("%s", _rhythm_reason)
            if on_health is not None:
                from percept.runtime.worker import RuntimeHealthEvent  # noqa: PLC0415 — import cycle

                on_health(RuntimeHealthEvent(
                    "rhythm_goal_refused_at_arm",
                    data={"goal_id": _rhythm_goal_id, "reason": _rhythm_reason},
                ))
        # The deployment's salience configuration, in one place (see watchspec_kwargs): the stat, the
        # ego mode and the per-stat cutoffs, not just a bare threshold. Passing only `threshold` here
        # took compile_watchspec's SIGNATURE defaults (mean @ threshold) for the other three, so this
        # loop gated on the whole-frame mean while the deployment — and the runtime worker, which
        # reads the same settings — was configured for the localized statistic at its own higher
        # cutoff. Every recall/economics number ever measured through this loop described a
        # configuration nobody ran.
        kwargs = watchspec_kwargs(cfg)
        if threshold is not None:
            # An explicit threshold is a cutoff, not a mean-scale cutoff: it overrides the whole
            # per-stat table (which would otherwise shadow it for every non-mean stat) while leaving
            # the configured statistic itself alone.
            kwargs["threshold"] = threshold
            kwargs["stat_thresholds"] = None
        self._unsupported_detectors: list[tuple[str, list[str]]] = []
        # edge_capabilities=None -> in-process edge runs the whole registry; a remote edge advertises a
        # subset and the brain selects within it (the capability handshake).
        self._spec = compile_watchspec(
            driven,
            edge_capabilities=edge_capabilities,
            on_unsupported=self._report_unsupported_detectors,
            **kwargs,
        )
        # clock=None -> schedules are not enforced (default path, byte-for-byte unchanged). A caller that
        # arms scheduled goals injects a civil clock (e.g. civil_now) here; pure logic never reads wall-clock.
        self._gate = MotionGate(self._spec, settings=cfg, clock=clock, wake_policy=wake_policy)
        # The standard telemetry funnel — this loop owns the seen/gated stages (frame seen; gate
        # escalate). Reuse the core's funnel (so judged/refused/fired land in the SAME Metrics) when the
        # core exposes one; else build one off the core's Metrics; else None (no telemetry sink wired).
        self._funnel = self._resolve_funnel(core)
        # One detector bank per goal that selected detectors — the edge proposes timed boundaries from
        # the cheap motion signal at native fps (no extra vision calls). Default: no detectors -> no banks.
        self._banks = {
            c.goal_id: DetectorBank(c.detectors) for c in self._spec.conditions if c.detectors
        }
        if interest is not None and interest.source != source:
            raise ValueError(
                "tier-0 interest source must match the GatedWatchLoop source "
                f"({interest.source!r} != {source!r})"
            )
        self._interest = interest

    def _preflight_routes(
        self, active: Sequence[Goal], *, on_health: Callable[[Any], None] | None
    ) -> list:
        """Refuse the UNDRIVABLE, report (never abort on) the host-driven; return what THIS loop drives.

        The identical decision ``percept.run`` makes, from the identical table
        (:func:`~percept.behavior_routes.require_drivable_routes`) — because it is the identical
        question. Two frame drivers must not answer "can this goal ever fire?" two different ways:

        * ``UNSUPPORTED`` — no driver exists anywhere and no host API can feed it. **Raises here**,
          in ``__init__``, so nothing is armed into the gate, no frame is pulled and no judge is
          spent. That guarantee is unchanged; only its scope narrowed to the routes it was ever true
          of.
        * a **host-driven** route (``count``/``event``/``remind``) — a complete, shipped capability
          whose driver is the HOST rather than this frame loop. It is EXCLUDED from the compiled
          spec and the rhythm drive (so it can never buy a judge call here) and reported, not
          raised: this loop aborting for it would take every unrelated Watch beside it down, which
          is exactly the defect ``percept.run`` fixed. A deployment that wants the old abort filters
          :class:`~percept.behavior_routes.UndrivenBehaviorRouteWarning` to ``"error"``.

        Reported on the same three channels, with the same text, differing only in the driver name:
        the warning an interactive/test caller sees, the log a 24/7 deployment reads, and — when the
        caller wired one — an ``undriven_behavior_route`` health event carrying the same payload.
        """
        skipped = require_drivable_routes(active, driver=_GATED_LOOP_DRIVER)
        for item in skipped:
            message = skipped_route_message(
                item, driver=_GATED_LOOP_DRIVER, source=self._source
            )
            warnings.warn(message, UndrivenBehaviorRouteWarning, stacklevel=3)
            get_logger("percept.harness.tier0").warning("%s", message)
            record = {
                "goal_id": item.goal_id,
                "route": item.route.value,
                "behavior_type": item.authored_type_display,
                "source": self._source,
                "driver": _GATED_LOOP_DRIVER,
                "remedy": route_remedy(item.route, goal_id=item.goal_id),
            }
            self._skipped_routes.append(record)
            if on_health is not None:
                from percept.runtime.worker import RuntimeHealthEvent  # noqa: PLC0415 — import cycle

                on_health(RuntimeHealthEvent("undriven_behavior_route", data=record))
        return [goal for goal in active if worker_can_drive(goal_behavior_route(goal))]

    def _report_unsupported_detectors(
        self, unsupported: list[tuple[str, list[str]]]
    ) -> None:
        """The third channel for a dropped detector — the structured health event.

        ``compile_watchspec`` already warned and logged (channels one and two); this adds the one a
        24/7 deployment can route, on exactly ``_preflight_routes``' terms. ADR-002 §2 names
        ``PLAN_UNOBSERVABLE`` for precisely this shape — "the wired source/goal combination cannot in
        principle answer what the plan asks" — and maps it to ``state`` directly rather than to a
        coverage float, because a boundary that can never arrive is binary, not a rate.
        """
        self._unsupported_detectors.extend(unsupported)
        if self._on_health is None:
            return
        from percept.runtime.worker import RuntimeHealthEvent  # noqa: PLC0415 — import cycle

        for goal_id, dropped in unsupported:
            self._on_health(RuntimeHealthEvent(
                UNSUPPORTED_DETECTOR_HEALTH_KIND,
                data={
                    "goal_id": goal_id,
                    "dropped_detectors": list(dropped),
                    "source": self._source,
                    "driver": _GATED_LOOP_DRIVER,
                    "reason": UNSUPPORTED_DETECTOR_REASON,
                },
            ))

    @property
    def unsupported_detectors(self) -> list[tuple[str, list[str]]]:
        """``(goal_id, dropped_detectors)`` the connected edge could not run — as data.

        The accessor half of the capability handshake, so a caller can pick a fallback without
        re-deriving the partition. A separate accessor rather than a ``WatchSpec`` field on purpose:
        ``WatchSpec`` is the wire contract ``contract.js`` mirrors, and adding a field to it would be
        a wire MINOR for what is purely a local diagnostic.
        """
        return [(goal_id, list(dropped)) for goal_id, dropped in self._unsupported_detectors]

    @property
    def skipped_routes(self) -> list[dict]:
        """The host-driven Goals this loop is NOT driving — the run-time report, as data."""
        return [dict(record) for record in self._skipped_routes]

    @staticmethod
    def _resolve_funnel(core: Any):
        """The core's :class:`~percept.observability.Funnel`, or one built off its ``Metrics``, or None.

        Reusing ``core.funnel`` keeps the seen/gated stages this loop emits in the same registry as the
        judged/refused/fired stages the agent emits — one funnel per watch loop. A core without a funnel
        but with a ``metrics`` registry gets a fresh funnel over it; a core with neither yields None (the
        loop simply skips funnel emission)."""
        funnel = getattr(core, "funnel", None)
        if funnel is not None:
            return funnel
        metrics = getattr(core, "metrics", None)
        if metrics is None:
            return None
        from percept.observability import Funnel

        return Funnel(metrics)

    @property
    def spec(self) -> WatchSpec:
        """The compiled WatchSpec the gate is consuming (the downward contract)."""
        return self._spec

    @property
    def gate(self) -> MotionGate:
        return self._gate

    async def run(
        self, stream: Iterable[Any], *, clock_start: float = 0.0, dt: float = 1.0,
        energies: Sequence[float] | None = None,
        cheap_signals: Sequence[dict[str, float]] | None = None,
        interest_stream_epoch: str | None = None,
        interest_observed_times: Sequence[float] | None = None,
    ) -> LoopReport:
        """Drive the loop over a frame stream; gate, report up, perceive only on escalation.

        ``stream`` yields either bare frames (timestamps synthesized from ``clock_start`` stepping
        by ``dt``) or ``(ts, frame)`` pairs. ``energies`` (optional) is a per-frame audio-energy signal
        (one float per frame, same order) feeding energy-backed detectors like acoustic-onset — the audio
        analog of the motion channel, the same energy already computed off the mic/STT stream. Returns
        the :class:`LoopReport`.

        When ``interest=`` was configured, this is explicitly a FINITE REPLAY epoch:
        ``interest_stream_epoch`` and one independent observer-clock timestamp per frame in
        ``interest_observed_times`` are required. Successful completion closes the interval as
        ``end_of_stream`` and leaves the adapter ready for another run/epoch. Any exception aborts
        and discards the unreturned lifecycle report; it emits no exit that could later be exposed
        without its enter.
        """
        report = LoopReport()
        items = _normalize_stream(stream, clock_start=clock_start, dt=dt)
        epoch_started = False
        if self._interest is not None:
            if interest_stream_epoch is None:
                raise ValueError("interest_stream_epoch is required for an interest replay")
            if interest_observed_times is None:
                raise ValueError("interest_observed_times is required for an interest replay")
            if len(interest_observed_times) != len(items):
                raise ValueError(
                    "interest_observed_times must contain exactly one timestamp per frame"
                )
        previous_interest_frame: Frame | None = None
        try:
            if self._interest is not None:
                self._interest.begin_epoch(interest_stream_epoch)
                epoch_started = True
                # Each declared replay epoch is an independent stream. The tier-0 eye must cold-start
                # too: carrying the prior epoch's final frame would manufacture cross-epoch motion.
                self._gate.reset()
                self._rhythm.reset()  # same reason: no cross-epoch channel continuity -> no phantom rep
            for i, item in enumerate(items):
                energy = energies[i] if energies is not None and i < len(energies) else None
                extra = cheap_signals[i] if cheap_signals is not None and i < len(cheap_signals) else None
                current_interest_frame = (
                    _as_frame(item.frame) if self._interest is not None else None
                )
                decision = self._gate.evaluate(
                    item.frame, ts=item.ts, energy=energy, signals=extra,
                    # The interest adapter scores intervals from `decision.outcomes` — the per-Watch
                    # typed result — and falls back to the whole-frame `mean` gauge when the tuple is
                    # empty. Without this opt-in the fallback was not a fallback, it was the only
                    # path: a localized-salience Watch admits on `block_max` while its mean gauge sits
                    # near zero, so the replay loop handed the adapter a decision that could not
                    # describe why it admitted and the timeline came back empty. Gated on the adapter
                    # being wired, so a run with no interest recorder does not pay per-Watch outcome
                    # construction on every frame. The live path (worker.py) already opts in.
                    include_outcomes=self._interest is not None,
                )
                for goal_id, cause in zip(decision.goal_ids, decision.admission_causes):
                    report.admission_by_goal[goal_id] = report.admission_by_goal.get(goal_id, 0) + 1
                    report.admission_by_cause[cause] = report.admission_by_cause.get(cause, 0) + 1
                    if cause == "fixed":
                        report.fixed_opportunities += 1
                report.frames += 1
                report.signals.append(decision.signal)  # report Tier0Signal UP every frame
                if self._interest is not None:
                    assert current_interest_frame is not None
                    report.interest_events.extend(
                        self._interest.observe(
                            decision,
                            observed_at=interest_observed_times[i],
                            evidence_id=_motion_evidence_id(
                                previous_interest_frame,
                                current_interest_frame,
                            ),
                        )
                    )
                    previous_interest_frame = current_interest_frame
                if self._funnel is not None:
                    self._funnel.seen(camera=self._source)  # the ungated baseline: a frame was observed
                # Edge detectors run on the cheap motion/energy signals at native fps — the count/boundaries
                # come from here, decoupled from the (sparse, motion-gated) vision calls below. Each boundary
                # is fed to the brain as a distinct occurrence (edge proposes · brain counts), firing per the
                # goal's accumulate.fire_when without spending a vision call.
                count_boundary = getattr(self._core, "perceive_boundary", None)
                candidate_proposals: list[tuple[str, Boundary]] = []
                for goal_id, bank in self._banks.items():
                    # feed every channel the loop has: motion (always) + the raw frame (for frame-backed
                    # detectors like pose) + audio energy (for acoustic-onset).
                    for b in bank.push(
                        item.ts,
                        motion=decision.signal.motion,
                        frame=item.frame,
                        energy=energy,
                    ):
                        report.boundaries.append((goal_id, b))
                        if manifest_of(b.detector).route == "admission":
                            if b.kind == "candidate":
                                candidate_proposals.append((goal_id, b))
                        elif count_boundary is not None:
                            fired = count_boundary(goal_id, t=b.t)
                            if fired is not None:
                                report.fire_events.append(fired)
                candidate_goal_ids = self._gate.admit_external(
                    [goal_id for goal_id, _ in candidate_proposals], ts=item.ts
                ) if candidate_proposals else ()
                for goal_id in candidate_goal_ids:
                    report.admission_by_goal[goal_id] = report.admission_by_goal.get(goal_id, 0) + 1
                    report.admission_by_cause["candidate"] = (
                        report.admission_by_cause.get("candidate", 0) + 1
                    )
                # The rhythm sense: MEASURE this frame's channels and let the brain count reps. Runs
                # on every frame at native rate (never gated, never judged) — the count comes from
                # the signal, so it is bounded by the frame rate, not by judgment density.
                if self._rhythm:
                    # RhythmDrive.observe already isolates a per-goal eye failure (e.g. a pose
                    # provider that passed arm-time pre-flight and then raised inside detect(),
                    # percept/harness/pose.py:97-100) so it never reaches here — wrapped anyway so
                    # this call is never the one uncontained hop between a goal's eye and this whole
                    # loop: one rhythm goal's failure must fail only that goal, never this report.
                    try:
                        report.fire_events.extend(self._rhythm.observe(item.ts, item.frame))
                    except Exception as exc:  # noqa: BLE001 - a rhythm goal's eye must never end the loop for every other goal
                        get_logger("percept.harness.tier0").warning(
                            "rhythm_drive_observe_failed: %s", exc
                        )
                        if self._on_health is not None:
                            from percept.runtime.worker import RuntimeHealthEvent  # noqa: PLC0415 — import cycle

                            self._on_health(RuntimeHealthEvent(
                                "rhythm_drive_error", ts=item.ts, data={"error": repr(exc)}
                            ))
                    for _rhythm_goal_id, _rhythm_reason in self._rhythm.pop_mid_run_failures().items():
                        get_logger("percept.harness.tier0").warning("%s", _rhythm_reason)
                        if self._on_health is not None:
                            from percept.runtime.worker import RuntimeHealthEvent  # noqa: PLC0415 — import cycle

                            self._on_health(RuntimeHealthEvent(
                                "rhythm_goal_failed",
                                ts=item.ts,
                                data={"goal_id": _rhythm_goal_id, "reason": _rhythm_reason},
                            ))
                    for goal_id in self._rhythm.goal_ids:
                        report.rhythm_counts[goal_id] = self._rhythm.count(goal_id)
                if decision.escalate or candidate_goal_ids:
                    report.escalations += 1
                    report.vision_calls += 1
                    if self._funnel is not None:
                        self._funnel.gated(camera=self._source)  # the gate spent a cloud call this frame
                    kwargs = {"t": item.ts}
                    accepts_goal_ids = _accepts_keyword(
                        self._core.perceive_judged, "goal_ids"
                    )
                    admitted_goal_ids = tuple(dict.fromkeys((*decision.goal_ids, *candidate_goal_ids)))
                    if accepts_goal_ids:
                        kwargs["goal_ids"] = admitted_goal_ids
                    fired = await self._core.perceive_judged(
                        item.frame, self._source, **kwargs
                    )
                    if not accepts_goal_ids:
                        admitted = frozenset(admitted_goal_ids)
                        fired = [
                            event
                            for event in (fired or ())
                            if getattr(event, "goal_id", None) in admitted
                        ]
                    report.fire_events.extend(fired or [])
            if self._interest is not None:
                eos_at = interest_observed_times[-1] if interest_observed_times else None
                report.interest_events.extend(
                    self._interest.end_of_stream(observed_at=eos_at)
                )
        except BaseException:
            if self._interest is not None and epoch_started:
                self._interest.abort_epoch()
            raise
        finally:
            if epoch_started:
                # Success and failure both end the independent epoch with a cold eye. Reset is
                # idempotent and only drops the perceptual predecessor; cadence policy remains.
                self._gate.reset()
        return report


async def run_gated_watch(
    core: Any,
    stream: Iterable[Any],
    goals: Sequence[Goal] | None = None,
    *,
    threshold: float | None = None,
    source: str = "camera",
    clock_start: float = 0.0,
    dt: float = 1.0,
    energies: Sequence[float] | None = None,
    cheap_signals: Sequence[dict[str, float]] | None = None,
    settings: Settings | None = None,
    wake_policy: WakePolicy | dict[str, Any] | None = None,
    interest: Tier0MotionInterest | None = None,
    interest_stream_epoch: str | None = None,
    interest_observed_times: Sequence[float] | None = None,
    on_health: Callable[[Any], None] | None = None,
) -> LoopReport:
    """One-call entrypoint for the wired loop (the cross-package e2e drives this).

    Builds a :class:`GatedWatchLoop` (compiling ``goals`` — or the core's active goals — down to a
    WatchSpec) and runs it over ``stream`` (optionally with a per-frame ``energies`` audio signal).
    Returns the :class:`LoopReport`. ``on_health`` is the same optional structured channel the loop
    reports an undriven route on.
    """
    loop = GatedWatchLoop(
        core,
        goals,
        threshold=threshold,
        source=source,
        settings=settings,
        wake_policy=wake_policy,
        interest=interest,
        on_health=on_health,
    )
    return await loop.run(
        stream,
        clock_start=clock_start,
        dt=dt,
        energies=energies,
        cheap_signals=cheap_signals,
        interest_stream_epoch=interest_stream_epoch,
        interest_observed_times=interest_observed_times,
    )


# ---------------------------------------------------------------------------
# RemoteEdgeReceiver — the TOPOLOGY-B server shim (consume a remote on-device edge)
# ---------------------------------------------------------------------------


class RemoteEdgeReceiver:
    """Server-side consumer of a REMOTE on-device edge's up-channel (topology B).

    Unlike :class:`GatedWatchLoop` (topology A — gates frames in-process), the remote edge already gated
    and ran its skills on the device, and sends only messages up. This shim never gates or re-detects; it:

    1. takes the edge's advertised capabilities (an ``EdgeCapabilities`` message) so the brain selects
       skills within them — :meth:`watchspec` compiles the capability-filtered ``WatchSpec`` to send down;
    2. maps each incoming boundary (a ``BoundaryReport``) 1:1 to ``core.perceive_boundary(goal_id, t)`` —
       exactly what :class:`GatedWatchLoop` does inline (the ``count_boundary`` call), but driven by the
       wire instead of a local :class:`~.detectors.DetectorBank`.

    The frame/escalation path (the edge sends a frame up on salience -> ``core.perceive_judged``) is the
    product's media transport; this shim covers the capability + boundary halves of the contract.
    """

    def __init__(self, core: Any) -> None:
        self._core = core
        self._capabilities: list[str] | None = None

    def on_capabilities(self, capabilities: Iterable[str]) -> None:
        """Record the edge's advertised skill set (from an ``EdgeCapabilities`` message)."""
        self._capabilities = list(capabilities)

    @property
    def capabilities(self) -> list[str] | None:
        """The advertised set, or ``None`` if the edge hasn't advertised yet (brain then sends all)."""
        return None if self._capabilities is None else list(self._capabilities)

    def watchspec(
        self, *, threshold: float, goals: Sequence[Goal] | None = None, version: int = 1
    ) -> WatchSpec:
        """Compile the ``WatchSpec`` to send DOWN — detectors filtered to the edge's advertised skills."""
        active = list(goals) if goals is not None else list(self._core.goals.active())
        require_in_process_duration_topology(
            active,
            driver="RemoteEdgeReceiver.watchspec",
            remote=True,
        )
        return compile_watchspec(
            active, threshold=threshold, version=version, edge_capabilities=self._capabilities
        )

    def on_boundaries(self, report: Any) -> list:
        """Consume a ``BoundaryReport`` (``{'boundaries': [{goal_id, ..., t[, channel]}]}`` or a bare list):

        * a peak/onset/threshold event -> ``core.perceive_boundary(goal_id, t)`` (the legacy blind tally);
        * a ``"sample"`` event carrying a ``channel`` -> grouped per frame into a channel vector and routed
          to ``core.perceive_rhythm(goal_id, t, channels)`` (the rhythm sense over the wire).

        Both are additive and order-preserving; an edge that only sends peaks behaves exactly as before.
        Returns the fire events. No cloud call here — the brain counts the edge's measurements.
        """
        events = report.get("boundaries", []) if isinstance(report, dict) else list(report)
        perceive_boundary = getattr(self._core, "perceive_boundary", None)
        perceive_rhythm = getattr(self._core, "perceive_rhythm", None)
        fires: list = []
        pending: tuple | None = None  # (goal_id, t, {channel: value}) buffer for one frame of samples

        def flush() -> None:
            nonlocal pending
            if pending is not None and perceive_rhythm is not None:
                gid, t, chans = pending
                fired = perceive_rhythm(gid, t=t, channels=chans)
                if fired is not None:
                    fires.append(fired)
            pending = None

        for ev in events:
            channel = ev.get("channel")
            if channel is not None:  # a rhythm channel sample
                key = (ev["goal_id"], ev["t"])
                if pending is not None and (pending[0], pending[1]) != key:
                    flush()
                if pending is None:
                    pending = (ev["goal_id"], ev["t"], {})
                pending[2][channel] = ev.get("value", 0.0)
                continue
            flush()  # a non-sample boundary ends any open sample frame
            if perceive_boundary is not None:
                fired = perceive_boundary(ev["goal_id"], t=ev["t"])
                if fired is not None:
                    fires.append(fired)
        flush()
        return fires


__all__ = [
    "Frame",
    "gray_frame",
    "motion_score",
    "salience_stats",
    "SALIENCE_STATS",
    "ego_residual_score",
    "CivilTime",
    "civil_from_epoch",
    "civil_now",
    "schedule_active",
    "compile_watchspec",
    "watchspec_kwargs",
    "watchspec_unsupported",
    "UnsupportedEdgeCapabilityWarning",
    "UnsupportedMotionBiasError",
    "StreamTimeRewindWarning",
    "DegenerateBlockPartitionWarning",
    "UNSUPPORTED_DETECTOR_REASON",
    "UNSUPPORTED_DETECTOR_HEALTH_KIND",
    "DEGENERATE_PARTITION_REASON",
    "MOTION_BIAS_KEY",
    "REGION_HINT_KEY",
    "SCHEDULE_KEY",
    "SALIENCE_STAT_KEY",
    "EGO_COMPENSATE_KEY",
    "MotionGate",
    "GateDecision",
    "WatchAdmissionOutcome",
    "NonAdmissionReason",
    "NON_ADMISSION_REASONS",
    "GatedWatchLoop",
    "LoopReport",
    "run_gated_watch",
    "RemoteEdgeReceiver",
]
