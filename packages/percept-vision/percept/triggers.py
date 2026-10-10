"""Triggers: when a frame is worth judging, and when it can't be trusted at all.

Everything here is pure: numpy and the standard library, no I/O, no clock. ``now`` is passed
in (seconds; any clock, as long as one stream uses the same one), and every piece of state is
plain data (dicts, lists, floats and numpy arrays) returned to the caller, so the same inputs
always give the same outputs and state can be persisted and restored.

- :func:`on_change` - fires a condition when its zone changes against a slow moving average
  of the scene (the ``block_max`` statistic, with hysteresis).
- :func:`every` - a heartbeat: due every ``seconds`` per condition.
- :class:`QualityGate` - vetoes corrupt or blurry frames and tracks whether the feed is degraded.
- :class:`SceneMonitor` - detects a scene reset (camera bumped, PTZ move, IR day/night switch)
  and reports a short suppression window while baselines rebuild.

All statistics are computed on a 48x36 grayscale "gate frame" (:func:`gate_frame`), so the
cost per frame is the same at any resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from percept.contracts import Condition, Due, Frame
from percept.zones import zone_mask

GATE_W, GATE_H = 48, 36
GRID = (8, 6)  # block_max partition: 8 columns x 6 rows, about 6x6 px per block on the gate frame

# --------------------------------------------------------------------------- gate frame


def _gray(image: np.ndarray) -> np.ndarray:
    """``uint8`` HxW or HxWx3 BGR -> float32 luma (ITU-R BT.601 weights)."""
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        b, g, r = (image[..., i].astype(np.float32) for i in range(3))
        return np.float32(0.114) * b + np.float32(0.587) * g + np.float32(0.299) * r
    raise ValueError(f"expected an HxW or HxWx3 image, got shape {image.shape}")


@lru_cache(maxsize=32)
def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """``n_out x n_in`` matrix: each output cell's average over the input pixels it covers."""
    edges = np.arange(n_out + 1, dtype=np.float64) * n_in / n_out
    i = np.arange(n_in, dtype=np.float64)
    lo = np.maximum(i[None, :], edges[:-1, None])
    hi = np.minimum(i[None, :] + 1, edges[1:, None])
    weights = np.clip(hi - lo, 0.0, None) / (n_in / n_out)
    weights.flags.writeable = False
    return weights


def gate_frame(frame: Frame | np.ndarray) -> np.ndarray:
    """The 48x36 (``36 x 48`` array) float32 grayscale frame every trigger statistic reads.

    Area averaging: each output pixel is the mean of the input pixels it covers, with
    fractional overlap at the edges, so any input size works (including ones smaller than the
    gate frame). Colour input is BGR, as decoded by OpenCV.
    """
    image = frame.image if isinstance(frame, Frame) else np.asarray(frame)
    gray = _gray(image).astype(np.float64)
    h, w = gray.shape
    out = _area_weights(h, GATE_H) @ gray @ _area_weights(w, GATE_W).T
    return out.astype(np.float32)


# --------------------------------------------------------------------------- block statistics


def block_scores(
    reference: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray | None = None,
    grid: tuple[int, int] = GRID,
) -> np.ndarray:
    """Mean absolute difference per block, normalised to ``0..1`` (255 = 1.0).

    The active area is the whole frame, or the bounding box of ``mask``. It is cut into
    ``grid`` (columns, rows) blocks with floor-of-product boundaries; a grid finer than the area
    is clamped to it so no block is empty. With a mask, only masked pixels count, and blocks
    with no masked pixel are left out. Returns the scores of the remaining blocks, row-major.
    """
    diff = np.abs(np.asarray(current, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    h, w = diff.shape
    if mask is None:
        y0, y1, x0, x1 = 0, h, 0, w
        weight = None
    else:
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        if rows.size == 0:
            return np.zeros(0)
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        weight = mask[y0:y1, x0:x1].astype(np.float64)
    gx, gy = int(grid[0]), int(grid[1])
    if gx <= 0 or gy <= 0:
        raise ValueError(f"grid dims must be positive, got {grid}")
    aw, ah = x1 - x0, y1 - y0
    gx, gy = min(gx, aw), min(gy, ah)
    xs = (np.arange(gx) * aw) // gx
    ys = (np.arange(gy) * ah) // gy
    area = diff[y0:y1, x0:x1]
    if weight is None:
        weight = np.ones_like(area)
    else:
        area = area * weight

    def per_block(a: np.ndarray) -> np.ndarray:
        return np.add.reduceat(np.add.reduceat(a, ys, axis=0), xs, axis=1)

    totals, counts = per_block(area), per_block(weight)
    keep = counts > 0
    return totals[keep] / (counts[keep] * 255.0)


def block_max(
    reference: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray | None = None,
    grid: tuple[int, int] = GRID,
) -> float:
    """The most-changed block's score. A localized change (a hand, a cup) lights one block and
    scores high even when the whole-frame mean barely moves. An empty mask scores 0."""
    scores = block_scores(reference, current, mask, grid)
    return float(scores.max()) if scores.size else 0.0


def _ema(baseline: np.ndarray, gate: np.ndarray, dt: float, baseline_s: float) -> np.ndarray:
    """Move ``baseline`` toward ``gate`` with time constant ``baseline_s`` (0: replace)."""
    alpha = 1.0 if baseline_s <= 0 else 1.0 - math.exp(-max(0.0, dt) / baseline_s)
    return (baseline + np.float32(alpha) * (gate - baseline)).astype(np.float32)


def _contrast(baseline: np.ndarray, mask: np.ndarray) -> float:
    """The zone's robust grey-level range in the baseline, as a threshold scale in [0.25, 1]."""
    values = baseline[mask]
    if values.size == 0:
        return 1.0
    p5, p95 = np.percentile(values, [5, 95])
    return float(np.clip((p95 - p5) / 255.0, 0.25, 1.0))


# --------------------------------------------------------------------------- on_change


class OnChange:
    """Due when a condition's zone differs from the scene's moving average.

    Per step, the gate frame is computed once and compared with an exponential moving average
    of past gate frames (time constant ``baseline_s``); each condition reads the per-block
    scores of ``block_max`` inside its own zone mask. Hysteresis is per block: a block turns
    active when its score reaches the threshold and inactive when it falls to the exit
    threshold, and the condition is due whenever any block turns active. So a lasting change
    (a door left open) fires once on entry and rearms after the average absorbs it, while a
    second change in another block (someone walking through the door) still fires. Slow
    lighting drift is absorbed by the average and never fires.

    Both thresholds are scaled by the zone's contrast in the baseline,
    ``clip((p95 - p5) / 255, 0.25, 1)``, so a dark, low-contrast scene still fires on a subject
    that is large relative to it.

    Limit: the average follows the scene, so a block that changes slowly (ramping over roughly
    10 s or more at the default ``baseline_s``) may never reach the threshold. Pair
    slow-approach conditions with a heartbeat (``every``).

    The first step has no baseline: it records one and nothing is due. Passing ``state=None``
    later rebuilds the baseline the same way (the runtime does this after a scene reset).

    ``baseline_s=0`` compares each frame with the previous one instead of an average.
    """

    def __init__(
        self,
        threshold: float = 0.20,
        exit_threshold: float | None = None,
        baseline_s: float = 5.0,
        grid: tuple[int, int] = GRID,
    ) -> None:
        exit_threshold = threshold / 2 if exit_threshold is None else exit_threshold
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold is in (0, 1]")
        if not 0.0 <= exit_threshold < threshold:
            raise ValueError("exit_threshold is in [0, threshold)")
        if not (math.isfinite(baseline_s) and baseline_s >= 0):
            raise ValueError("baseline_s is >= 0")
        self.threshold = float(threshold)
        self.exit_threshold = float(exit_threshold)
        self.baseline_s = float(baseline_s)
        self.grid = (int(grid[0]), int(grid[1]))

    def __repr__(self) -> str:
        return (f"on_change(threshold={self.threshold}, exit_threshold={self.exit_threshold}, "
                f"baseline_s={self.baseline_s})")

    def step(self, frame: Frame, conditions: tuple[Condition, ...], state: Any, now: float) -> tuple[Due, Any]:
        gate = gate_frame(frame)
        if not state or state.get("baseline") is None:
            fresh = {c.id: {"active": [], "score": 0.0} for c in conditions}
            return {}, {"baseline": gate, "t": float(now), "conditions": fresh}
        baseline = np.asarray(state["baseline"], dtype=np.float32)
        previous = state.get("conditions", {})
        due: dict[str, str] = {}
        per_condition: dict[str, dict[str, Any]] = {}
        for c in conditions:
            mask = zone_mask(c.zones, c.exclude_zones, GATE_W, GATE_H)
            scores = block_scores(baseline, gate, mask, self.grid)
            scale = _contrast(baseline, mask)
            was = np.asarray(previous.get(c.id, {}).get("active", ()), dtype=bool)
            if was.shape != scores.shape:  # first sight of this condition, or its zone changed
                was = np.zeros(scores.shape, dtype=bool)
            entering = ~was & (scores >= self.threshold * scale)
            active = entering | (was & (scores > self.exit_threshold * scale))
            score = float(scores.max()) if scores.size else 0.0
            if entering.any():
                due[c.id] = f"change {score:.2f}"
            per_condition[c.id] = {"active": active.tolist(), "score": score}
        new_baseline = _ema(baseline, gate, float(now) - float(state["t"]), self.baseline_s)
        return due, {"baseline": new_baseline, "t": float(now), "conditions": per_condition}


# --------------------------------------------------------------------------- every


class Every:
    """A heartbeat: each condition is due once every ``seconds``.

    A condition is due on the first step that sees it, so state conditions get an initial
    look without waiting a full period. After that it is due when ``now - last >= seconds``,
    and ``last`` becomes ``now`` (not ``last + seconds``), so a gap never causes a burst of
    catch-up looks. If ``now`` goes backwards (a restored state on a different clock) the
    condition is due again immediately.
    """

    def __init__(self, seconds: float) -> None:
        if not (isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0):
            raise ValueError("every(seconds) needs a positive number of seconds")
        self.seconds = float(seconds)

    def __repr__(self) -> str:
        return f"every({self.seconds:g})"

    def step(self, frame: Frame, conditions: tuple[Condition, ...], state: Any, now: float) -> tuple[Due, Any]:
        last: dict[str, float] = dict(state or {})
        due: dict[str, str] = {}
        new: dict[str, float] = {}
        for c in conditions:
            t = last.get(c.id)
            if t is None or now - t >= self.seconds or now < t:
                due[c.id] = "heartbeat" if t is None else f"every {self.seconds:g}s"
                new[c.id] = float(now)
            else:
                new[c.id] = t
        return due, new


def on_change(
    threshold: float = 0.20,
    exit_threshold: float | None = None,
    baseline_s: float = 5.0,
) -> OnChange:
    """A change trigger: judge a condition when something in its zone changes.

    ``threshold`` is the block score (0..1, a block's mean absolute grey-level change over 255)
    that fires, before scaling by the zone's contrast; ``exit_threshold`` (default half of it)
    is where a block rearms; ``baseline_s`` is how quickly lasting changes and lighting drift
    are absorbed. A change that ramps in over roughly 10 s or more may never fire: use a
    heartbeat for slow approaches. The defaults are starting points, not calibrated values.
    """
    return OnChange(threshold=threshold, exit_threshold=exit_threshold, baseline_s=baseline_s)


def every(seconds: float) -> Every:
    """A heartbeat trigger: judge a condition every ``seconds``, and once at the start."""
    return Every(seconds)


# --------------------------------------------------------------------------- frame quality


@dataclass(frozen=True)
class QualityVerdict:
    """Whether a frame may be judged. ``reason`` says why not ("corrupt" or "blurry").

    ``degraded`` is the feed-level state: too many recent frames were bad.
    """

    ok: bool
    reason: str | None
    degraded: bool


def smear_map(gray: np.ndarray, block: int = 8) -> np.ndarray:
    """Per 8x8 block: does it look like concealed packet loss?

    A decoder hides missing macroblocks by smearing the row above downward: rows inside the
    block are identical while its columns still vary. Vertical structure (blinds, fences) looks
    the same in one frame, which is why :class:`QualityGate` only counts blocks that newly
    became smeared.
    """
    h, w = gray.shape
    h, w = h - h % block, w - w % block
    if h < block or w < block:
        return np.zeros((0, 0), dtype=bool)
    b = gray[:h, :w].astype(np.float32).reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)
    down = np.abs(np.diff(b, axis=2)).mean(axis=(2, 3))
    across = np.abs(np.diff(b, axis=3)).mean(axis=(2, 3))
    return (down < 0.6) & (across > 2.0)


def smear_fraction(gray: np.ndarray, block: int = 8) -> float:
    """Fraction of 8x8 blocks that are smear-shaped in this one frame."""
    m = smear_map(gray, block)
    return float(m.mean()) if m.size else 0.0


def sharpness(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian over the interior, divided by the frame's variance.

    Dividing by contrast keeps a dimmer but focused scene from reading as blurry.
    """
    g = gray.astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var()) / max(float(g.var()), 1e-6)


class QualityGate:
    """Vetoes frames that are corrupt (smeared blocks) or blurry, and tracks a degraded feed.

    Corruption is temporal: a frame is corrupt when at least ``smear_bad`` of its blocks are
    smear-shaped now and were not in the previous frame. Structure that persists never counts.

    Blur is relative to this stream: a frame is blurry when its contrast-normalised sharpness
    is below ``blur_ratio`` times the median of recent good frames, judged only once
    ``blur_min_samples`` good frames are known. The veto is for transient defocus: a blur streak
    longer than ``blur_min_samples`` frames is taken as the new normal and the history restarts
    from it, so a lasting change of view never blinds the stream.

    The feed is degraded once at least ``degraded_on`` of the last ``window`` frames were bad
    (with at least 10 seen), and recovers at ``degraded_off`` or below.

    Checks run on the frame as given; the runtime passes the reduced (at most 640 px wide)
    frame it keeps for evidence, and passes ``state=None`` after a scene reset.
    """

    def __init__(
        self,
        smear_bad: float = 0.05,
        blur_ratio: float = 0.25,
        blur_min_samples: int = 30,
        sharp_history: int = 300,
        window: int = 60,
        degraded_on: float = 0.30,
        degraded_off: float = 0.10,
    ) -> None:
        if not 0.0 <= degraded_off < degraded_on <= 1.0:
            raise ValueError("need 0 <= degraded_off < degraded_on <= 1")
        self.smear_bad = smear_bad
        self.blur_ratio = blur_ratio
        self.blur_min_samples = blur_min_samples
        self.sharp_history = sharp_history
        self.window = window
        self.degraded_on = degraded_on
        self.degraded_off = degraded_off

    def check(self, frame: Frame, state: Any, now: float) -> tuple[QualityVerdict, Any]:
        state = state or {}
        sharp = list(state.get("sharp", ()))
        recent = list(state.get("recent", ()))
        degraded = bool(state.get("degraded", False))
        streak = int(state.get("blur_streak", 0))
        gray = _gray(frame.image)
        smeared = smear_map(gray)
        before = np.asarray(state.get("smear", ()), dtype=bool)
        if before.shape != smeared.shape:  # first frame, or the size changed: nothing to compare
            before = smeared
        reason: str | None = None
        if smeared.size and float((smeared & ~before).mean()) >= self.smear_bad:
            reason = "corrupt"
        else:
            s = sharpness(gray)
            if len(sharp) >= self.blur_min_samples and s < self.blur_ratio * float(np.median(sharp)):
                streak += 1
                if streak > self.blur_min_samples:  # lasting, not transient: the new normal
                    sharp, streak = [s], 0
                else:
                    reason = "blurry"
            else:
                sharp, streak = (sharp + [s])[-self.sharp_history:], 0  # good frames define "normal"
        recent = (recent + [reason])[-self.window:]
        bad = sum(1 for r in recent if r is not None) / len(recent)
        if not degraded and len(recent) >= 10 and bad >= self.degraded_on:
            degraded = True
        elif degraded and bad <= self.degraded_off:
            degraded = False
        verdict = QualityVerdict(ok=reason is None, reason=reason, degraded=degraded)
        return verdict, {"sharp": sharp, "recent": recent, "degraded": degraded,
                         "blur_streak": streak, "smear": smeared.tolist()}


# --------------------------------------------------------------------------- scene reset


@dataclass(frozen=True)
class SceneSignal:
    """What :class:`SceneMonitor` saw on one frame.

    ``reset``: a scene reset starts on this frame (open a ``scene_reset`` blind interval).
    ``suppressed``: triggers must not run on this frame.
    ``ended``: the suppression ended on this frame. Close the interval, restart triggers from
    a fresh baseline, and make **every condition due once** on this frame: anything that
    happened during the reset was absorbed into the new baseline, so the settled scene gets
    one look (the runtime batches it, keeping the reset frame as evidence).
    ``changed`` is the fraction of blocks that changed.
    """

    reset: bool = False
    suppressed: bool = False
    ended: bool = False
    changed: float = 0.0


class SceneMonitor:
    """Detects a scene reset: most of the frame changing at once.

    A reset is declared when more than ``fraction`` of the blocks differ from the scene's
    moving average by more than ``block_threshold`` and the change also covers at least
    ``border_fraction`` of the blocks on the frame's border: a bumped camera, a PTZ move, an IR
    day/night switch. A large subject close to the camera rarely reaches every edge, so it is
    an ordinary change, not a reset. Triggers are then suppressed for ``suppress_s``; while
    more than ``fraction / 2`` of the blocks keep changing frame to frame (a PTZ or slow pan
    still moving) the window is extended. Afterwards change triggers restart from a fresh
    baseline and every condition gets one look (see :class:`SceneSignal`).
    """

    def __init__(
        self,
        fraction: float = 0.6,
        block_threshold: float = 0.08,
        suppress_s: float = 2.0,
        baseline_s: float = 5.0,
        border_fraction: float = 0.75,
        grid: tuple[int, int] = GRID,
    ) -> None:
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction is in (0, 1)")
        self.fraction = fraction
        self.border_fraction = border_fraction
        self.block_threshold = block_threshold
        self.suppress_s = suppress_s
        self.baseline_s = baseline_s
        self.grid = grid

    def step(self, frame: Frame, state: Any, now: float) -> tuple[SceneSignal, Any]:
        gate = gate_frame(frame)
        now = float(now)
        if not state or state.get("baseline") is None:
            return SceneSignal(), {"baseline": gate, "t": now, "until": None}
        baseline = np.asarray(state["baseline"], dtype=np.float32)
        until = state.get("until")
        gx, gy = min(self.grid[0], GATE_W), min(self.grid[1], GATE_H)
        hit = (block_scores(baseline, gate, None, self.grid) > self.block_threshold).reshape(gy, gx)
        changed = float(hit.mean())
        border = np.concatenate([hit[0], hit[-1], hit[1:-1, 0], hit[1:-1, -1]])
        if until is None:
            if changed > self.fraction and float(border.mean()) >= self.border_fraction:
                signal = SceneSignal(reset=True, suppressed=True, changed=changed)
                return signal, {"baseline": gate, "t": now, "until": now + self.suppress_s}
            baseline = _ema(baseline, gate, now - float(state["t"]), self.baseline_s)
            return SceneSignal(changed=changed), {"baseline": baseline, "t": now, "until": None}
        # Suppressing: follow the picture frame to frame until it holds still.
        until = now + self.suppress_s if changed > self.fraction / 2 else float(until)
        if now >= until:
            return SceneSignal(ended=True, changed=changed), {"baseline": gate, "t": now, "until": None}
        return SceneSignal(suppressed=True, changed=changed), {"baseline": gate, "t": now, "until": until}
