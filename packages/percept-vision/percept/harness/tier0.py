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

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from percept.adapters import Box, Goal, Schedule, Tier0Signal, WatchCondition, WatchSpec
from percept.config import Settings, load_settings
from percept.interest import InterestIntervalEvent, Tier0MotionInterest

from .detectors import (
    Boundary,
    DetectorBank,
    WakePolicy,
    detectors_for,
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


def _block_scores(
    a: Frame, b: Frame, region: Box | None, block_grid: tuple[int, int]
) -> tuple[list[float], int, int]:
    """The per-block salience list for the active area, plus ``(total_abs_diff, area_px)``.

    ONE pass over the pixels; every salience statistic in this module is a pure selector over what
    this returns (``block_max`` = ``max``, ``p95`` = nearest-rank, ``mean`` =
    ``total / (area_px * 255)``, because the blocks tile the area exactly with no gaps or overlaps).
    That is what lets :func:`salience_stats` price all three statistics for the cost of one — and
    what guarantees ``percept doctor``'s counterfactual table cannot disagree with the live gate.

    Block boundaries are floor-of-product (``bx(i) = x0 + (i * aw) // Gx``): blocks differ by <=1 px
    and are bit-identical to the JS edge (``i * aw`` stays far under 2**53). Each block's score is
    ``sum_abs_diff / (block_px * 255)`` — an integer sum over an integer denominator, so numpy and
    stdlib return the same float. A degenerate block (0 px, only possible when the area is smaller
    than the grid) contributes ``0.0``.
    """
    w, h = a.w, a.h
    if region is not None:
        x0, y0, x1, y1 = _region_bounds(region, w, h)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    aw, ah = x1 - x0, y1 - y0
    if aw <= 0 or ah <= 0:
        return [], 0, 0
    gx, gy = int(block_grid[0]), int(block_grid[1])
    if gx <= 0 or gy <= 0:
        return [], 0, 0
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
            if block_px == 0:
                scores.append(0.0)
                continue
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
    return scores, grand_total, aw * ah


def _motion_score_blocks(
    a: Frame, b: Frame, region: Box | None, block_grid: tuple[int, int], stat: str
) -> float:
    """Localized salience: partition the active area into blocks (:func:`_block_scores`), select
    per ``stat``. ``block_max`` = the top block; ``p95`` = the nearest-rank 95th-percentile block
    (integer rank, no float percentile)."""
    scores, _total, _px = _block_scores(a, b, region, block_grid)
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
    scores, total, area_px = _block_scores(a, b, region, block_grid)
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


def ego_residual_score(prev_frame: Any, frame: Any, *, k: int = 4) -> float:
    """Ego-motion-compensated salience: the minimum mean-abs-diff over an integer-pixel global shift.

    Estimates and removes whole-frame translation (camera pan / head turn) by searching integer
    shifts ``(dx, dy)`` in ``[-k, k]`` and taking the *residual* — the smallest MAD between ``prev``
    shifted by ``(dx, dy)`` and ``frame`` over their overlap. A pure camera pan is explained by some
    shift, so its residual is ``~0`` (no false escalate); a subject entering the frame cannot be
    shifted away, so its residual stays high. Deterministic and O(k**2 * pixels); the ``(0, 0)`` shift
    equals the uncompensated mean, so the residual is always ``<=`` the full-frame :func:`motion_score`.

    ``prev_frame is None`` -> ``0.0`` (no predecessor). Mirrors the JS ``egoResidualScore`` bit-for-bit
    (integer overlap sums, integer overlap denominator).
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
    return 0.0 if best is None else best


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
# semantic|fixed} under this key. Controls how OFTEN to look (distinct from PerceptionSpec = how DEEP).
CADENCE_KEY = "cadence"
# Optional per-goal salience MODE overrides (tracks the camera mount). Fall back to the deployment
# default passed to compile_watchspec (from settings.edge) when absent / malformed.
SALIENCE_STAT_KEY = "salience_stat"
EGO_COMPENSATE_KEY = "ego_compensate"
_GATE_ON_VOCAB = {"motion", "structural", "semantic", "fixed"}
_SALIENCE_STAT_VOCAB = {"mean", "block_max", "p95"}


def compile_watchspec(
    goals: Iterable[Goal], *, threshold: float, version: int = 1,
    edge_capabilities: Iterable[str] | None = None,
    salience_stat: str = "mean", ego_compensate: bool = False,
    stat_thresholds: dict[str, float] | None = None,
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
    (default, in-process topology A — the edge runs the whole registry) sends the full selection. Use
    :func:`watchspec_unsupported` to see what was dropped (to pick a fallback).

    Only goals whose ``state['type']`` is ``'watch'`` (the default) or ``'process'`` compile — event /
    count / remind goals run their own tracks and never gate the watch-frame stream. A ``process`` goal
    (Item 4 log→summarize) gates the same motion stream so the edge wakes it to observe a step delta.
    """
    conditions: list[WatchCondition] = []
    for goal in goals:
        if goal.state.get("type", "watch") not in ("watch", "process"):
            continue
        bias = float(goal.state.get(MOTION_BIAS_KEY, 0.0))
        supported, _dropped = partition_by_capability(detectors_for(goal), edge_capabilities)
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
                region_hint=_coerce_box(goal.state.get(REGION_HINT_KEY)),
                detectors=supported,  # cheap edge skills the brain selected, filtered to what the edge can run
                schedule=_coerce_schedule(goal.state.get(SCHEDULE_KEY)),
                cadence_s=cadence_s,  # how OFTEN to look (distinct from PerceptionSpec = how DEEP)
                gate_on=gate_on,
                salience_stat=stat,
                ego_compensate=ego,
            )
        )
    return WatchSpec(conditions=conditions, version=version)


def watchspec_unsupported(
    goals: Iterable[Goal], edge_capabilities: Iterable[str]
) -> list[tuple[str, list[str]]]:
    """``(goal_id, dropped_detectors)`` the brain selected but the advertised edge can't run — so the
    caller can pick a fallback (server-side detection / a coarser skill / skip). Empty when the edge
    supports everything selected. The reporting half of :func:`compile_watchspec`'s capability handshake.
    """
    out: list[tuple[str, list[str]]] = []
    for goal in goals:
        if goal.state.get("type", "watch") not in ("watch", "process"):
            continue
        _, dropped = partition_by_capability(detectors_for(goal), edge_capabilities)
        if dropped:
            out.append((goal.id, dropped))
    return out


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)


def _coerce_box(obj: Any) -> Box | None:
    if obj is None or isinstance(obj, Box):
        return obj
    if isinstance(obj, dict):
        return Box(x=obj["x"], y=obj["y"], w=obj["w"], h=obj["h"])
    return None  # pragma: no cover - defensive: unknown hint shape -> no hint.


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


def _coerce_cadence(obj: Any) -> tuple[float | None, str]:
    """Goal-state cadence hint -> ``(cadence_s, gate_on)``; the firewall for the temporal cadence.

    Accepts ``{"min_judge_interval": float, "gate_on": "motion"|"structural"|"semantic"|"fixed"}``.
    Unknown shape / out-of-vocab -> ``(None, "motion")`` (the default, no cadence cap). A non-positive
    interval is dropped; ``gate_on="fixed"`` without an interval falls back to ``"motion"`` (a fixed
    cadence with no period is meaningless), so a malformed hint can never silently break a watch.
    """
    if not isinstance(obj, dict):
        return None, "motion"
    raw = obj.get("min_judge_interval")
    cadence_s = float(raw) if isinstance(raw, (int, float)) and raw > 0 else None
    gate_on = obj.get("gate_on", "motion")
    if gate_on not in _GATE_ON_VOCAB:
        gate_on = "motion"
    if gate_on == "fixed" and cadence_s is None:
        gate_on = "motion"
    return cadence_s, gate_on


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
    compensation (whole-frame residual) takes precedence; else the localized block statistic (optionally
    masked to ``region_hint``). Ego + block compose is deferred: an ego goal uses the whole-frame residual.
    """
    if c.ego_compensate:
        return ego_residual_score(prev, f)
    if c.salience_stat != "mean" or c.region_hint is not None:
        return motion_score(prev, f, region=c.region_hint, stat=c.salience_stat)
    return full_motion


# ---------------------------------------------------------------------------
# MotionGate — consumes a WatchSpec + a frame, emits a Tier0Signal + a decision
# ---------------------------------------------------------------------------


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
        settings: Settings | None = None,
        clock: Callable[[], CivilTime] | None = None,
        wake_policy: WakePolicy | dict[str, Any] | None = None,
        structural_adapter: Callable[[Frame], float] | None = None,
        semantic_adapter: Callable[[Frame], float] | None = None,
    ) -> None:
        self._spec = spec
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

    @property
    def spec(self) -> WatchSpec:
        return self._spec

    def threshold(self) -> float:
        """The effective escalation cutoff for the current spec (min over all conditions)."""
        if not self._spec.conditions:
            return self._default
        return min(c.threshold for c in self._spec.conditions)

    def reset(self) -> None:
        """Forget the previous frame (e.g. a stream discontinuity / new source)."""
        self._prev = None

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
        live_ids = {c.goal_id for c in self._spec.conditions}
        self._last_escalate = {
            gid: t for gid, t in prev._last_escalate.items() if gid in live_ids
        }
        self._prev = prev._prev

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
        :meth:`evaluate`, so an audio wake cannot double-spend inside a goal's cadence window.
        """
        requested = frozenset(goal_ids)
        when = now if now is not None else (self._clock() if self._clock is not None else None)
        admitted: list[str] = []
        for condition in self._spec.conditions:
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
    ) -> GateDecision:
        """Score one frame against the prior, report a ``Tier0Signal``, decide escalate-vs-skip.

        An empty WatchSpec (no active watch goals) never escalates — there is nothing to look for.
        Otherwise escalate when an **armed** condition's motion (full-frame, or masked to its
        ``region_hint``) clears that condition's threshold. ``now`` (or the gate's injected ``clock``)
        is the civil time the per-condition :class:`~percept.adapters.Schedule` is checked against; a
        condition whose schedule is out of window at ``now`` is skipped. With no schedules and no
        ``region_hint`` this reduces to the original ``full_motion >= min(threshold)``. Advances the
        prev-frame cursor.
        """
        f = _as_frame(frame)
        prev = self._prev
        # Full-frame mean motion is always the REPORTED gauge (the Tier0Signal is a region-agnostic,
        # config-stable salience). Per-goal escalation may use a localized/ego-compensated statistic
        # instead (below), but the up-channel gauge stays the plain mean so it's comparable across configs.
        motion = motion_score(prev, f)
        # structural/semantic default to 0.0 (motion-only reference); optional adapters fill them
        # (CLIP/pose/privacy on the real edge). None -> 0.0 keeps the default path byte-for-byte.
        structural = 0.0 if self._structural_adapter is None else float(self._structural_adapter(f))
        semantic = 0.0 if self._semantic_adapter is None else float(self._semantic_adapter(f))
        cheap_signals = {
            "motion": motion,
            "change": motion,
            "structural": structural,
            "semantic": semantic,
            "semantic_relevance": semantic,
            "audio": 0.0 if energy is None else energy,
            "audio_event": 0.0 if energy is None else energy,
        }
        if signals:
            cheap_signals.update(signals)
        fused = wake_score(cheap_signals, self._wake_policy) if self._wake_policy is not None else None
        when = now if now is not None else (self._clock() if self._clock is not None else None)
        # Armed = in-schedule conditions only; an out-of-schedule goal cannot escalate or set the cutoff.
        armed = [c for c in self._spec.conditions if schedule_active(c.schedule, when)]
        escalate = False
        admitted_goal_ids: list[str] = []
        for c in armed:
            # Cadence: a goal can't escalate more than once per cadence_s. For gate_on="fixed" this
            # cadence IS the escalation trigger (a floor — look every cadence_s even with no pixel motion,
            # so content events like a screenshare aren't silently under-sampled by the motion gate).
            if c.cadence_s is not None and ts - self._last_escalate.get(c.goal_id, float("-inf")) < c.cadence_s:
                continue
            if c.gate_on == "fixed":
                fired = c.cadence_s is not None  # cadence elapsed (checked above) -> escalate regardless of motion
            elif fused is not None:
                fired = fused >= c.threshold
            elif c.gate_on == "structural":
                fired = structural >= c.threshold
            elif c.gate_on == "semantic":
                fired = semantic >= c.threshold
            else:  # "motion" (default) — full-frame, or the goal's localized / ego-compensated statistic
                m = _condition_motion(prev, f, c, motion)
                fired = m >= c.threshold
            if fired:
                escalate = True
                admitted_goal_ids.append(c.goal_id)
                self._last_escalate[c.goal_id] = ts
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


class GatedWatchLoop:
    """The wired WatchSpec↓ / Tier0Signal↑ loop against a live :class:`~percept.agent.Percept`.

    Construction compiles the core's **active goals** down to a :class:`WatchSpec` and arms a
    :class:`MotionGate` with it. :meth:`run` replays a frame stream: per frame it gates, reports a
    :class:`Tier0Signal` up, and calls ``core.perceive_judged`` **only on escalation** — the
    cloud call is spent exactly when salience clears threshold. Returns a :class:`LoopReport` with
    gated-vs-ungated call counts (the cost graph) and the upward signal stream.

    This is the proof point: ``compile_watchspec``'s output is consumed by the gate, and a
    ``Tier0Signal`` is reported on every frame.
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
    ) -> None:
        self._core = core
        cfg = settings or getattr(core, "settings", None) or load_settings()
        self._settings = cfg
        self._source = source
        active = list(goals) if goals is not None else list(core.goals.active())
        thr = cfg.edge.motion_gate if threshold is None else threshold
        # edge_capabilities=None -> in-process edge runs the whole registry; a remote edge advertises a
        # subset and the brain selects within it (the capability handshake).
        self._spec = compile_watchspec(active, threshold=thr, edge_capabilities=edge_capabilities)
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
        if interest is not None and interest.engine.source != source:
            raise ValueError(
                "tier-0 interest source must match the GatedWatchLoop source "
                f"({interest.engine.source!r} != {source!r})"
            )
        self._interest = interest

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
    ) -> LoopReport:
        """Drive the loop over a frame stream; gate, report up, perceive only on escalation.

        ``stream`` yields either bare frames (timestamps synthesized from ``clock_start`` stepping
        by ``dt``) or ``(ts, frame)`` pairs. ``energies`` (optional) is a per-frame audio-energy signal
        (one float per frame, same order) feeding energy-backed detectors like acoustic-onset — the audio
        analog of the motion channel, the same energy already computed off the mic/STT stream. Returns
        the :class:`LoopReport`.
        """
        report = LoopReport()
        items = _normalize_stream(stream, clock_start=clock_start, dt=dt)
        for i, item in enumerate(items):
            energy = energies[i] if energies is not None and i < len(energies) else None
            extra = cheap_signals[i] if cheap_signals is not None and i < len(cheap_signals) else None
            decision = self._gate.evaluate(item.frame, ts=item.ts, energy=energy, signals=extra)
            report.frames += 1
            report.signals.append(decision.signal)  # report Tier0Signal UP every frame
            if self._interest is not None:
                report.interest_events.extend(self._interest.observe(decision))
            if self._funnel is not None:
                self._funnel.seen(camera=self._source)  # the ungated baseline: a frame was observed
            # Edge detectors run on the cheap motion/energy signals at native fps — the count/boundaries
            # come from here, decoupled from the (sparse, motion-gated) vision calls below. Each boundary
            # is fed to the brain as a distinct occurrence (edge proposes · brain counts), firing per the
            # goal's accumulate.fire_when without spending a vision call.
            count_boundary = getattr(self._core, "perceive_boundary", None)
            for goal_id, bank in self._banks.items():
                # feed every channel the loop has: motion (always) + the raw frame (for frame-backed
                # detectors like pose) + audio energy (for acoustic-onset).
                for b in bank.push(item.ts, motion=decision.signal.motion, frame=item.frame, energy=energy):
                    report.boundaries.append((goal_id, b))
                    if count_boundary is not None:
                        fired = count_boundary(goal_id, t=b.t)
                        if fired is not None:
                            report.fire_events.append(fired)
            if decision.escalate:
                report.escalations += 1
                report.vision_calls += 1
                if self._funnel is not None:
                    self._funnel.gated(camera=self._source)  # the gate spent a cloud call this frame
                kwargs = {"t": item.ts}
                accepts_goal_ids = _accepts_keyword(
                    self._core.perceive_judged, "goal_ids"
                )
                if accepts_goal_ids:
                    kwargs["goal_ids"] = decision.goal_ids
                fired = await self._core.perceive_judged(
                    item.frame, self._source, **kwargs
                )
                if not accepts_goal_ids:
                    admitted = frozenset(decision.goal_ids)
                    fired = [
                        event
                        for event in (fired or ())
                        if getattr(event, "goal_id", None) in admitted
                    ]
                report.fire_events.extend(fired or [])
        if self._interest is not None:
            # ``run`` consumes a finite iterable, so reaching its end has unambiguous EOS semantics.
            # The adapter preserves the last positive evidence bound and stamps EOS separately.
            report.interest_events.extend(self._interest.end_of_stream())
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
) -> LoopReport:
    """One-call entrypoint for the wired loop (the cross-package e2e drives this).

    Builds a :class:`GatedWatchLoop` (compiling ``goals`` — or the core's active goals — down to a
    WatchSpec) and runs it over ``stream`` (optionally with a per-frame ``energies`` audio signal).
    Returns the :class:`LoopReport`.
    """
    loop = GatedWatchLoop(
        core,
        goals,
        threshold=threshold,
        source=source,
        settings=settings,
        wake_policy=wake_policy,
        interest=interest,
    )
    return await loop.run(
        stream, clock_start=clock_start, dt=dt, energies=energies, cheap_signals=cheap_signals
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
    "watchspec_unsupported",
    "MOTION_BIAS_KEY",
    "REGION_HINT_KEY",
    "SCHEDULE_KEY",
    "SALIENCE_STAT_KEY",
    "EGO_COMPENSATE_KEY",
    "MotionGate",
    "GateDecision",
    "GatedWatchLoop",
    "LoopReport",
    "run_gated_watch",
    "RemoteEdgeReceiver",
]
