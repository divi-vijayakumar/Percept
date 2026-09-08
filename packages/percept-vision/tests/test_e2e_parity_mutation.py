"""Cross-runtime agreement, and proof that the parity gate can fail (the E2E capability test).

The edge ships twice: once in JavaScript (``packages/percept-edge``) and once in Python
(``percept.harness.tier0``). ``make parity`` is the release gate that says those two routes agree,
and three shipped docstrings cite it as the proof of bit-identity (``harness/tier0.py`` twice,
``src/motion.js`` once). This file asks the two questions that gate cannot ask about itself.

**1. Does the same media produce the same ADMISSION DECISION?** The parity script compares
*functions* — a salience number, an energy number, a detector boundary. A product does not admit a
number, it admits a frame: the same clip, driven through the JS ``Edge`` and the Python
``MotionGate`` with the same authored watch, must escalate on exactly the same frames. That is
asserted here over ten scored watch shapes (mean / block_max / p95 / watch-out zone / ego / two
goals / cadence / fixed cadence / structural / the fused noisy-OR with an audio channel), and the
comparison is made through :mod:`percept.eval.stage_attribution` rather than by eyeballing two
lists, so a disagreement is reported as a funnel that moved.

**2. Can the frozen vectors actually kill a mutant?** ``make parity`` passing told us nothing until
mutation testing showed its checks could not die: six of eight forks of the localizing statistic
scored ``max|delta| = 0.0`` across every vector the file had. The vector set was then widened from
33 checks to 81. This file re-runs that experiment as a test. It carries a second, independent
implementation of the block statistic and of the ego residual, each with a switch for one specific
arithmetic fork, and asserts:

* the unforked model reproduces the shipped function EXACTLY on every frozen vector (otherwise the
  mutation results describe some other program), and
* every fork DIES on the widened vector set, under the gate's own ``bit_identical`` comparator, and
* every fork SURVIVES on the historical prefix the widening replaced — which is what makes the
  widening evidence rather than decoration.

**Cross-cutting, per the stage spec.** Two runtimes must agree on what a frame IS (the luma-byte
constructors are compared byte-for-byte, including out-of-range and fractional luma) and on ONE
definition of audio energy (mean-square, not RMS, compared with ``==`` rather than a tolerance).

**This test needs node.** Parity is a release gate and ``make parity`` itself exits rather than
skipping when node is missing; so does this file. A silent skip here would restore exactly the
condition the gate exists to prevent. It downloads nothing: every frame, every PCM sample and every
watch is synthesized in-process or read from the tracked parity vectors.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path

from percept.adapters import Box, Schedule, WatchCondition, WatchSpec
from percept.config import load_settings
from percept.eval.stage_attribution import stage_funnel
from percept.harness.tier0 import (
    CivilTime,
    DegenerateBlockPartitionWarning,
    Frame,
    MotionGate,
    ego_residual_score,
    gray_frame,
    motion_score,
)
from percept.runtime.audio import pcm_energy
from percept.tracing import (
    ADMISSION_STAGE,
    OBSERVATION_STAGE,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
)

_TESTS_DIR = Path(__file__).resolve().parent
_EDGE_ROOT = _TESTS_DIR.parents[1] / "percept-edge"
_PARITY_DIR = _EDGE_ROOT / "parity"
_EDGE_RUNTIME_JS = _PARITY_DIR / "edge_runtime.mjs"
_COMPARE_PY = _PARITY_DIR / "compare.py"

#: One seed for every fuzz population in this file. Fixed, so a failure is reproducible by rerunning
#: the file rather than by hoping the same case comes up again.
FUZZ_SEED = 20260731


# ---------------------------------------------------------------------------
# 0. Reaching the other runtime
# ---------------------------------------------------------------------------


def _node_binary() -> str | None:
    """The node executable, discovered exactly as ``parity/compare.py`` discovers it.

    Same three places in the same order (``PERCEPT_NODE``, ``PATH``, the newest ``~/.nvm`` build), so
    this file and ``make parity`` can never disagree about which runtime was measured.
    """
    node = os.environ.get("PERCEPT_NODE") or shutil.which("node")
    if node:
        return node
    nvm = Path.home() / ".nvm/versions/node"
    candidates = sorted(nvm.glob("*/bin/node")) if nvm.exists() else []
    return str(candidates[-1]) if candidates else None


NODE = _node_binary()

_NO_NODE = (
    "node was not found (PERCEPT_NODE, PATH, ~/.nvm all empty). The JS edge is half of this "
    "contract, so there is nothing to compare against and this result is NOT a pass. "
    "`make parity` exits for the same reason; install node or set PERCEPT_NODE."
)


def _run_node(script: Path, cases: dict) -> dict:
    """Feed ``cases`` to a node script as JSON on stdin and read its JSON answer back."""
    if NODE is None:  # pragma: no cover - the tests assert this before calling
        raise AssertionError(_NO_NODE)
    proc = subprocess.run(
        [NODE, str(script)],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"the JS edge failed to run {script.name}:\n{proc.stderr}")
    return json.loads(proc.stdout)


# The admission driver. The frozen-vector runner (`parity/edge_runtime.mjs`) exercises the edge's
# ARITHMETIC; this one exercises the edge's DECISION — the same `Edge.processFrame` loop a device
# runs, one Tier0Signal-or-null per frame. It is written to a temporary directory rather than into
# the tree: it is a test fixture for one assertion, not a second parity runner to keep in sync.
_ADMISSION_DRIVER = """
import {{ Edge }} from "{edge_js}";
import {{ makeFrame }} from "{motion_js}";

function readStdin() {{
  return new Promise((resolve) => {{
    let s = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (d) => (s += d));
    process.stdin.on("end", () => resolve(s));
  }});
}}

const cases = JSON.parse(await readStdin());
const out = {{}};

// One run of the wired edge loop per case: a frame sequence, a watch, an optional per-frame audio
// energy, an optional wake policy. Reports whether each frame escalated and the motion gauge the
// up-channel Tier0Signal carried when it did.
out.admission = (cases.admission || []).map((c) => {{
  let i = 0;
  const edge = new Edge({{ clock: () => c.ts[i], wakePolicy: c.wakePolicy ?? null }});
  if (c.spec !== null && c.spec !== undefined) edge.applyWatchSpec(c.spec);
  const escalate = [];
  const motion = [];
  for (i = 0; i < c.frames.length; i++) {{
    const signal = edge.processFrame(
      {{ width: c.w, height: c.h, data: c.frames[i] }},
      {{ energy: c.energies ? c.energies[i] : null }}
    );
    escalate.push(signal !== null);
    motion.push(signal === null ? null : signal.motion);
  }}
  return {{ escalate, motion }};
}});

// What a frame IS: the luma bytes the JS frame constructor produces for a scalar fill or a plain
// array, including values outside 0..255 and fractional ones. JSON cannot carry a NaN, so the
// caller sends the sentinel below and it becomes a real NaN here, before the constructor sees it.
const luma = (v) => (v === "{nan_sentinel}" ? Number.NaN : v);
out.frameBytes = (cases.frameBytes || []).map(({{ w, h, fill }}) =>
  Array.from(makeFrame(w, h, Array.isArray(fill) ? fill.map(luma) : luma(fill)).data)
);

process.stdout.write(JSON.stringify(out));
"""

#: JSON has no NaN literal (``json.dumps`` emits a bare ``NaN`` that ``JSON.parse`` rejects), so the
#: one luma value that is not a number travels as this string and is rehydrated in the driver.
NAN_SENTINEL = "not-a-number"

_DRIVER_DIR: tempfile.TemporaryDirectory | None = None


def _admission_driver_path() -> Path:
    """Materialize the admission driver once per process and return its path."""
    global _DRIVER_DIR
    if _DRIVER_DIR is None:
        _DRIVER_DIR = tempfile.TemporaryDirectory(prefix="percept-parity-")
        # Explicit, so the interpreter does not have to clean it implicitly at shutdown and warn.
        atexit.register(_DRIVER_DIR.cleanup)
    path = Path(_DRIVER_DIR.name) / "admission_runtime.mjs"
    if not path.exists():
        path.write_text(
            _ADMISSION_DRIVER.format(
                edge_js=(_EDGE_ROOT / "src" / "edge.js").as_uri(),
                motion_js=(_EDGE_ROOT / "src" / "motion.js").as_uri(),
                nan_sentinel=NAN_SENTINEL,
            ),
            encoding="utf-8",
        )
    return path


# ---------------------------------------------------------------------------
# 1. The frozen vectors, loaded from the gate that owns them
# ---------------------------------------------------------------------------


def _load_parity_vectors():
    """Import ``parity/compare.py`` for its vectors and its comparator.

    Loaded by path because it is a script beside the JS package, not an importable module. Taking the
    vectors from the gate itself is the point: a copy here would drift, and a mutation result about a
    copy says nothing about what ``make parity`` runs.
    """
    spec = importlib.util.spec_from_file_location("percept_parity_compare", _COMPARE_PY)
    if spec is None or spec.loader is None:  # pragma: no cover - the file is tracked
        raise AssertionError(f"cannot load the parity vectors from {_COMPARE_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARITY = _load_parity_vectors()

#: The salience vectors that predate the widening. The comment above ``SALIENCE_PAIRS`` calls them
#: "the six shape/statistic vectors" and states they are structurally blind to the arithmetic the
#: gate claims to cover; the mutation tests below re-measure that claim, so their identity is pinned
#: here by shape rather than trusted by index alone.
HISTORICAL_SALIENCE_PREFIX = 6
HISTORICAL_SALIENCE_SHAPES = (
    (48, 36, "mean"),
    (48, 36, "block_max"),
    (48, 36, "p95"),
    (13, 11, "block_max"),
    (13, 11, "p95"),
    (48, 36, "block_max"),
)

#: The ego vectors that predate the widening — "the first three vectors ... all horizontal-only
#: motion on generous frames", per the comment above ``EGO_PAIRS``.
HISTORICAL_EGO_PREFIX = 3
HISTORICAL_EGO_SHAPES = ((48, 36, 4), (48, 36, 4), (13, 11, 3))


def _case_grid(case) -> tuple[int, int]:
    return tuple(case.get("blockGrid") or (8, 6))  # type: ignore[return-value]


def _case_region(case) -> Box | None:
    return PARITY._box(case.get("region"))


def _shipped_salience(case) -> float:
    """The shipped ``motion_score`` answer for one frozen vector.

    ``DegenerateBlockPartitionWarning`` is muted for the same reason ``compare.py`` mutes it: three
    vectors drive an area smaller than the grid ON PURPOSE, and their expected values ARE the clamped
    answers.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateBlockPartitionWarning)
        return motion_score(
            (case["w"], case["h"], tuple(case["a"])),
            (case["w"], case["h"], tuple(case["b"])),
            region=_case_region(case),
            stat=case.get("stat", "mean"),
            block_grid=_case_grid(case),
        )


def _shipped_ego(case) -> float:
    return ego_residual_score(
        (case["w"], case["h"], tuple(case["a"])),
        (case["w"], case["h"], tuple(case["b"])),
        k=case["k"],
        region=_case_region(case),
    )


# ---------------------------------------------------------------------------
# 2. A forkable second implementation of the localizing statistic
# ---------------------------------------------------------------------------

#: No fork — the faithful model. Spelled as a name so a mutant is never selected by passing ``None``
#: and a typo in a fork name cannot silently mean "unforked".
FAITHFUL = "faithful"

#: Every fork this file mutates, with the arithmetic it breaks and the value ``compare.py`` says the
#: forked runtime returns. The three the brief names — block boundaries, region rounding, the
#: nearest-rank index — plus the three ways to get the grid clamp wrong, because the clamp is what
#: decides WHICH blocks the rank ranks.
SALIENCE_FORKS: dict[str, str] = {
    "block_boundaries_rounded": "block boundaries rounded instead of floor-of-product",
    "region_far_half_up": "the zone's far bound rounded half-UP instead of half-to-even",
    "region_origin_rounded": "the zone's origin rounded instead of truncated",
    "p95_rank_floor": "nearest rank as floor(0.95n) instead of ceil(0.95n)-1",
    "p95_rank_positional": "nearest rank as a positional round(0.95*(n-1))",
    "grid_unclamped": "the grid padded out to the area instead of clamped down to it",
    "grid_clamp_short": "the grid clamped one block short of the area",
    "grid_clamp_long": "the grid clamped one block long of the area",
}


def _half_to_even(x: float) -> int:
    """Python's ``round`` — the rule ``_region_bounds`` uses and ``pyRound`` mirrors in JS."""
    return int(round(x))


def _half_up(x: float) -> int:
    """JavaScript's ``Math.round`` — identical except at an exact .5, which is the whole point."""
    return math.floor(x + 0.5)


def _region_bounds(region: Box, w: int, h: int, fork: str) -> tuple[int, int, int, int]:
    origin_x, origin_y = region.x * w, region.y * h
    far_x, far_y = (region.x + region.w) * w, (region.y + region.h) * h
    if fork == "region_origin_rounded":
        x0, y0 = _half_to_even(origin_x), _half_to_even(origin_y)
    else:
        x0, y0 = int(origin_x), int(origin_y)
    if fork == "region_far_half_up":
        x1, y1 = _half_up(far_x), _half_up(far_y)
    else:
        x1, y1 = _half_to_even(far_x), _half_to_even(far_y)
    x0, y0 = max(0, min(w, x0)), max(0, min(h, y0))
    return x0, y0, max(x0, min(w, x1)), max(y0, min(h, y1))


def _partition(area_w: int, area_h: int, gx: int, gy: int, fork: str) -> tuple[int, int]:
    if fork == "grid_unclamped":
        return gx, gy
    if fork == "grid_clamp_short":
        return max(1, min(gx, area_w - 1)), max(1, min(gy, area_h - 1))
    if fork == "grid_clamp_long":
        return min(gx, area_w + 1), min(gy, area_h + 1)
    return (gx if gx < area_w else area_w), (gy if gy < area_h else area_h)


def _block_bound(index: int, span: int, blocks: int, fork: str) -> int:
    if fork == "block_boundaries_rounded":
        return _half_to_even(index * span / blocks)
    return (index * span) // blocks


def _nearest_rank_p95(scores: list[float], fork: str) -> float:
    ordered = sorted(scores)
    n = len(ordered)
    if fork == "p95_rank_floor":
        index = int(math.floor(0.95 * n))
    elif fork == "p95_rank_positional":
        index = _half_to_even(0.95 * (n - 1))
    else:
        index = (95 * n + 99) // 100 - 1
    return ordered[max(0, min(n - 1, index))]


def model_salience(
    w: int,
    h: int,
    a,
    b,
    *,
    stat: str = "mean",
    region: Box | None = None,
    grid: tuple[int, int] = (8, 6),
    fork: str = FAITHFUL,
) -> float:
    """A second implementation of ``motion_score``, with one arithmetic fork selectable.

    Deliberately written from the CONTRACT (floor-of-product boundaries, integer per-block
    denominators, half-to-even far bounds, an integer nearest rank, a grid clamped to the active
    area) rather than by copying the shipped code, so agreeing with it is evidence. An empty block
    contributes a literal ``0.0``, which is unreachable while the clamp holds and is exactly what the
    ``grid_unclamped`` fork brings back.
    """
    if region is not None:
        x0, y0, x1, y1 = _region_bounds(region, w, h, fork)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    area_w, area_h = x1 - x0, y1 - y0

    if stat == "mean":
        pixels = area_w * area_h
        if pixels == 0:
            return 0.0
        total = 0
        for row in range(y0, y1):
            base = row * w
            for col in range(x0, x1):
                total += abs(a[base + col] - b[base + col])
        return total / (pixels * 255.0)

    gx, gy = _partition(area_w, area_h, int(grid[0]), int(grid[1]), fork)
    if gx <= 0 or gy <= 0:
        return 0.0
    scores: list[float] = []
    for j in range(gy):
        by0 = y0 + _block_bound(j, area_h, gy, fork)
        by1 = y0 + _block_bound(j + 1, area_h, gy, fork)
        for i in range(gx):
            bx0 = x0 + _block_bound(i, area_w, gx, fork)
            bx1 = x0 + _block_bound(i + 1, area_w, gx, fork)
            block_px = (bx1 - bx0) * (by1 - by0)
            if block_px <= 0:
                scores.append(0.0)
                continue
            total = 0
            for row in range(by0, by1):
                base = row * w
                for col in range(bx0, bx1):
                    total += abs(a[base + col] - b[base + col])
            scores.append(total / (block_px * 255.0))
    if not scores:
        return 0.0
    if stat == "block_max":
        return max(scores)
    if stat == "p95":
        return _nearest_rank_p95(scores, fork)
    raise ValueError(f"unknown salience stat {stat!r}")


# ---------------------------------------------------------------------------
# 3. A forkable second implementation of the ego residual
# ---------------------------------------------------------------------------

#: The ego forks. One of five survived every vector before the widening; these re-measure that.
EGO_FORKS: dict[str, str] = {
    "search_range_short": "the shift search shortened to [-(k-1), k-1]",
    "search_dx_only": "the shift search restricted to horizontal shifts",
    "search_drops_zero_shift": "the (0,0) shift dropped from the search",
    "region_ignored": "the watch-out zone dropped on the floor",
    "region_local_search": "the shift ESTIMATED inside the zone instead of whole-frame",
    "region_denominator": "divided by the zone instead of by zone-overlap intersection",
    "whole_frame_denominator": "divided by the whole frame instead of by the overlap",
}


def _mad(a, b, w: int, dx: int, dy: int, x0: int, y0: int, x1: int, y1: int, denom: int) -> float:
    total = 0
    for fy in range(y0, y1):
        frame_base = fy * w
        prev_base = (fy - dy) * w
        for fx in range(x0, x1):
            total += abs(a[prev_base + (fx - dx)] - b[frame_base + fx])
    return total / (denom * 255.0)


def model_ego(
    w: int, h: int, a, b, *, k: int = 4, region: Box | None = None, fork: str = FAITHFUL
) -> float:
    """A second implementation of ``ego_residual_score``, with one fork selectable.

    The contract: search every integer shift in ``[-k, k]``, score each over ITS overlap, take the
    minimum, and — when a zone is present — report that already-chosen shift's residual over
    ``zone ∩ overlap``. What is MEASURED is masked; what is ESTIMATED never is.
    """
    kk = max(0, int(k))
    lo, hi = (-(kk - 1), kk - 1) if fork == "search_range_short" else (-kk, kk)
    best: float | None = None
    best_shift: tuple[int, int] = (0, 0)
    for dy in range(lo, hi + 1):
        if fork == "search_dx_only" and dy != 0:
            continue
        overlap_h = h - abs(dy)
        if overlap_h <= 0:
            continue
        fy0 = dy if dy > 0 else 0
        for dx in range(lo, hi + 1):
            if fork == "search_drops_zero_shift" and dx == 0 and dy == 0:
                continue
            overlap_w = w - abs(dx)
            if overlap_w <= 0:
                continue
            fx0 = dx if dx > 0 else 0
            x0, y0, x1, y1 = fx0, fy0, fx0 + overlap_w, fy0 + overlap_h
            denom = overlap_w * overlap_h
            if fork == "region_local_search" and region is not None:
                rx0, ry0, rx1, ry1 = _region_bounds(region, w, h, FAITHFUL)
                x0, y0, x1, y1 = max(x0, rx0), max(y0, ry0), min(x1, rx1), min(y1, ry1)
                if x1 - x0 <= 0 or y1 - y0 <= 0:
                    continue
                denom = (x1 - x0) * (y1 - y0)
            if fork == "whole_frame_denominator":
                denom = w * h
            score = _mad(a, b, w, dx, dy, x0, y0, x1, y1, denom)
            if best is None or score < best:
                best, best_shift = score, (dx, dy)
    if best is None:
        return 0.0
    if region is None or fork == "region_ignored":
        return best

    dx, dy = best_shift
    rx0, ry0, rx1, ry1 = _region_bounds(region, w, h, FAITHFUL)
    ox0, oy0 = (dx if dx > 0 else 0), (dy if dy > 0 else 0)
    ox1, oy1 = ox0 + w - abs(dx), oy0 + h - abs(dy)
    mx0, my0 = max(rx0, ox0), max(ry0, oy0)
    mx1, my1 = min(rx1, ox1), min(ry1, oy1)
    if mx1 - mx0 <= 0 or my1 - my0 <= 0:
        return 0.0
    denom = (mx1 - mx0) * (my1 - my0)
    if fork == "region_denominator":
        denom = (rx1 - rx0) * (ry1 - ry0)
    return _mad(a, b, w, dx, dy, mx0, my0, mx1, my1, denom)


# ---------------------------------------------------------------------------
# 4. The synthetic clip and the watches driven over it
# ---------------------------------------------------------------------------

CLIP_W, CLIP_H = 24, 18


def _still(value: int = 20) -> list[int]:
    return [value] * (CLIP_W * CLIP_H)


def _with_patch(base, px: int, py: int, pw: int, ph: int, value: int) -> list[int]:
    out = list(base)
    for row in range(py, py + ph):
        for col in range(px, px + pw):
            out[row * CLIP_W + col] = value
    return out


def _panned(base, dx: int) -> list[int]:
    """``base`` sampled at ``x + dx`` with the edge column clamped — a camera pan, no new content."""
    out = [0] * (CLIP_W * CLIP_H)
    for y in range(CLIP_H):
        for x in range(CLIP_W):
            sx = min(CLIP_W - 1, max(0, x + dx))
            out[y * CLIP_W + x] = base[y * CLIP_W + sx]
    return out


_GRADIENT = [(x * 5 + y * 3) % 256 for y in range(CLIP_H) for x in range(CLIP_W)]

#: A ten-frame clip with four regimes a tier-0 gate must tell apart: a still prefix, a small
#: localized event in the top-left corner (visible to block_max / a zone, invisible to the mean), a
#: whole-scene change, a pure pan (which ego compensation must explain away), and a subject appearing
#: inside a panned scene (which it must not).
CLIP: tuple[list[int], ...] = (
    _still(),
    _still(),
    _with_patch(_still(), 2, 2, 3, 3, 220),
    _still(),
    _GRADIENT,
    _panned(_GRADIENT, 2),
    _panned(_GRADIENT, 2),
    _with_patch(_panned(_GRADIENT, 2), 10, 8, 4, 4, 250),
    _still(),
    _still(),
)
CLIP_TS: tuple[float, ...] = tuple(float(i) for i in range(len(CLIP)))

#: Per-frame audio energy for the fused-channel watch. Deliberately NOT aligned with the visual
#: events: an audio-only wake is the case where the two runtimes could agree about every pixel and
#: still disagree about admission.
CLIP_ENERGY: tuple[float, ...] = (0.0, 0.5, 0.0, 0.6, 0.0, 0.0, 0.7, 0.0, 0.2, 0.9)

FUSED_POLICY = {"threshold": 0.45, "weights": {"motion": 0.7, "audio": 0.7}}

#: The watch shapes driven over the clip. Each is one authored WatchSpec; the two runtimes must
#: escalate on exactly the same frames for every one of them.
WATCHES: dict[str, list[WatchCondition]] = {
    "whole-frame mean": [WatchCondition("g", "movement", 0.05)],
    "localized block_max": [WatchCondition("g", "a cup is up", 0.30, salience_stat="block_max")],
    "localized p95": [WatchCondition("g", "a cup is up", 0.10, salience_stat="p95")],
    "watch-out zone + block": [
        WatchCondition(
            "g", "the corner", 0.20, salience_stat="block_max",
            region_hint=Box(0.0, 0.0, 0.25, 0.25),
        )
    ],
    "ego compensated": [WatchCondition("g", "someone entered", 0.02, ego_compensate=True)],
    "ego + zone": [
        WatchCondition(
            "g", "someone entered", 0.02, ego_compensate=True,
            region_hint=Box(0.375, 0.4, 0.3, 0.3),
        )
    ],
    "two goals, one sensitive": [
        WatchCondition("insensitive", "a truck", 0.90),
        WatchCondition("sensitive", "a hand", 0.02),
    ],
    "one goal on a cadence": [WatchCondition("g", "a hand", 0.02, cadence_s=3.0)],
    "fixed cadence": [WatchCondition("g", "the dashboard", 0.5, gate_on="fixed", cadence_s=2.0)],
    "structural, nothing wired": [WatchCondition("g", "a pose", 0.5, gate_on="structural")],
}

#: The watch shapes that need the wake policy wired on both sides.
FUSED_WATCH = [WatchCondition("g", "someone is here", 0.45, gate_on="fused")]


def _condition_json(condition: WatchCondition) -> dict:
    """One ``WatchCondition`` on the wire, in the shape ``contract.js`` parses.

    ``schedule`` is written only when the caller asked for it: ``parseWatchCondition`` tolerantly
    drops the field, which is a known drift the schedule test below pins rather than hides.
    """
    out = {
        "goal_id": condition.goal_id,
        "anchor_text": condition.anchor_text,
        "threshold": condition.threshold,
        "detectors": list(condition.detectors),
        "cadence_s": condition.cadence_s,
        "gate_on": condition.gate_on,
        "salience_stat": condition.salience_stat,
        "ego_compensate": condition.ego_compensate,
    }
    if condition.region_hint is not None:
        zone = condition.region_hint
        out["region_hint"] = {"x": zone.x, "y": zone.y, "w": zone.w, "h": zone.h}
    if condition.schedule is not None:
        out["schedule"] = {
            "windows": [list(window) for window in condition.schedule.windows],
            "days": list(condition.schedule.days),
        }
    return out


def _spec_json(conditions) -> dict:
    return {"version": 1, "conditions": [_condition_json(c) for c in conditions]}


def _python_admissions(
    conditions,
    *,
    frames=CLIP,
    ts=CLIP_TS,
    energies=None,
    wake_policy=None,
    now: CivilTime | None = None,
) -> tuple[list[bool], list[float]]:
    """Drive the Python gate over a clip: one escalate decision and one motion gauge per frame."""
    gate = MotionGate(
        WatchSpec(conditions=list(conditions)),
        settings=load_settings(),
        wake_policy=wake_policy,
    )
    decisions: list[bool] = []
    gauges: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateBlockPartitionWarning)
        for index, (pixels, when) in enumerate(zip(frames, ts)):
            decision = gate.evaluate(
                Frame(w=CLIP_W, h=CLIP_H, pixels=tuple(pixels)),
                ts=when,
                energy=None if energies is None else energies[index],
                now=now,
            )
            decisions.append(decision.escalate)
            gauges.append(decision.signal.motion)
    return decisions, gauges


def _funnel_of(decisions, *, run_id: str):
    """The admission funnel for one runtime's decisions, derived by the shared stage helper.

    Every frame that reached the gate is one ``edge.decision``; every frame it admitted is one
    ``source.frame``. Nothing downstream is recorded, because this capability test traverses the
    measurement stage only — and the helper reports an unmeasured step as "no ratio" rather than as
    zero, which is what keeps that honest.
    """
    manifest = RunTraceManifest.create(
        wall_clock=lambda: 1000.0, run_id=run_id, policy=TracePolicy(level="full")
    )
    recorder = TraceRecorder(manifest, wall_clock=lambda: 1000.0)
    for index, admitted in enumerate(decisions):
        trace_id = f"clip@{index}"
        recorder.record(
            OBSERVATION_STAGE,
            trace_id=trace_id,
            source_ts=float(index),
            observed_wall_ts=1000.0,
            escalate=bool(admitted),
        )
        if admitted:
            recorder.record(
                ADMISSION_STAGE,
                trace_id=trace_id,
                source_ts=float(index),
                observed_wall_ts=1000.0,
            )
    return stage_funnel(recorder.records, manifest=manifest)


def _as_marks(decisions) -> str:
    """``.1..1`` — a per-frame decision string, so a failure message shows WHERE they diverged."""
    return "".join("1" if d else "." for d in decisions)


# ---------------------------------------------------------------------------
# 5. The node payloads, built once and shared
# ---------------------------------------------------------------------------

_JS_CACHE: dict[str, dict] = {}


def _seeded_pairs():
    """Seeded fuzz frame pairs for the salience and ego paths.

    Small frames on purpose: the forks this file cares about live at boundaries — a grid finer than
    the area, a zone bound landing on an exact .5, a block one pixel wide — and a 200x200 frame hides
    every one of them under an average.
    """
    rng = random.Random(FUZZ_SEED)
    zones = (
        None,
        {"x": 0.0, "y": 0.0, "w": 0.25, "h": 1.0},
        {"x": 0.26, "y": 0.0, "w": 0.5, "h": 1.0},
        {"x": 0.125, "y": 0.125, "w": 0.75, "h": 0.75},
        {"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5},
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"x": 0.9, "y": 0.9, "w": 0.5, "h": 0.5},
        # A zero-extent zone. It stays in the population deliberately even though the two runtimes
        # are known to fork on it: the fuzz assertion routes this input class to the pin in
        # `ADegenerateWatchOutZoneForksTests` instead of deleting the input, so the day the fork
        # closes, the pin says so rather than the population quietly never having tested it.
        {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.5},
        {"x": 0.3333, "y": 0.16667, "w": 0.3333, "h": 0.5},
    )
    salience, ego = [], []
    for _ in range(120):
        w, h = rng.randint(1, 14), rng.randint(1, 14)
        n = w * h
        if rng.random() < 0.6:
            # A still background with a few hot pixels — the localized-event shape.
            base = rng.randint(0, 255)
            a = [base] * n
            b = list(a)
            for _ in range(rng.randint(0, 6)):
                b[rng.randrange(n)] = rng.randint(0, 255)
        else:
            a = [rng.randint(0, 255) for _ in range(n)]
            b = [rng.randint(0, 255) for _ in range(n)]
        salience.append({
            "w": w, "h": h, "a": a, "b": b,
            "stat": rng.choice(("mean", "block_max", "p95")),
            "region": rng.choice(zones),
            "blockGrid": [rng.randint(1, 10), rng.randint(1, 10)],
        })
        ego.append({
            "w": w, "h": h, "a": a, "b": b,
            "k": rng.randint(0, 4),
            "region": rng.choice(zones),
        })
    return salience, ego


def _seeded_pcm():
    """Seeded s16 PCM windows, including the saturating extremes, for the energy comparison."""
    rng = random.Random(FUZZ_SEED + 1)
    vectors = [
        {"name": "fuzz-extremes", "samples": [-32768, 32767, -32768, 32767]},
        {"name": "fuzz-single", "samples": [-1]},
    ]
    for i in range(40):
        n = rng.randint(1, 64)
        vectors.append(
            {"name": f"fuzz-{i}", "samples": [rng.randint(-32768, 32767) for _ in range(n)]}
        )
    return vectors


FUZZ_SALIENCE, FUZZ_EGO = _seeded_pairs()
FUZZ_PCM = _seeded_pcm()


def _zero_extent_zone(case) -> bool:
    """Whether this case's watch-out zone encloses no area — the known-forked input class."""
    zone = case.get("region")
    return zone is not None and (zone["w"] <= 0.0 or zone["h"] <= 0.0)


#: The minimal reproduction of that fork, spelled out rather than left to a fuzz draw. A 7x3 frame,
#: a zone of ZERO width at x=0.1: the origin truncates (0.7 -> 0) and the far bound rounds
#: (0.7 -> 1), so the "empty" zone is one real pixel column wide. Python measures it; the JS mean
#: path never looks. The block statistics on the identical zone agree, which is what localizes the
#: divergence to the one short-circuit.
_ZERO_ZONE_A = [0] * 21
_ZERO_ZONE_B = [255] * 21
DEGENERATE_ZONE_CASES = [
    {"w": 7, "h": 3, "a": _ZERO_ZONE_A, "b": _ZERO_ZONE_B, "stat": "mean",
     "region": {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.5}},
    {"w": 7, "h": 3, "a": _ZERO_ZONE_A, "b": _ZERO_ZONE_B, "stat": "block_max",
     "region": {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.5}},
    {"w": 7, "h": 3, "a": _ZERO_ZONE_A, "b": _ZERO_ZONE_B, "stat": "p95",
     "region": {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.5}},
    # The same shape one pixel-fraction lower: 0.3*3 = 0.9 truncates AND rounds to 0, so this one
    # really does collapse to an empty span and the two runtimes agree. It is here to prove the
    # divergence above is about the ROUNDING, not about zero-extent zones in general.
    {"w": 7, "h": 3, "a": _ZERO_ZONE_A, "b": _ZERO_ZONE_B, "stat": "mean",
     "region": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.0}},
]


def js_vectors() -> dict:
    """One call into ``parity/edge_runtime.mjs`` carrying the frozen vectors and the fuzz."""
    if "vectors" not in _JS_CACHE:
        _JS_CACHE["vectors"] = _run_node(
            _EDGE_RUNTIME_JS,
            {
                "saliencePairs": (
                    list(PARITY.SALIENCE_PAIRS) + FUZZ_SALIENCE + DEGENERATE_ZONE_CASES
                ),
                "egoPairs": list(PARITY.EGO_PAIRS) + FUZZ_EGO,
                "audioEnergyVectors": list(PARITY.AUDIO_FIXTURE["energy_vectors"]) + FUZZ_PCM,
            },
        )
    return _JS_CACHE["vectors"]


#: Every admission scenario, in one node call. The key is the scenario name the assertions read back.
def _admission_payload() -> list[dict]:
    cases = [
        {
            "name": name,
            "w": CLIP_W, "h": CLIP_H,
            "frames": [list(f) for f in CLIP],
            "ts": list(CLIP_TS),
            "spec": _spec_json(conditions),
        }
        for name, conditions in WATCHES.items()
    ]
    cases.append({
        "name": "fused with audio",
        "w": CLIP_W, "h": CLIP_H,
        "frames": [list(f) for f in CLIP],
        "ts": list(CLIP_TS),
        "energies": list(CLIP_ENERGY),
        "wakePolicy": FUSED_POLICY,
        "spec": _spec_json(FUSED_WATCH),
    })
    # A schedule that is closed at the civil time the Python gate is given.
    closed = [
        WatchCondition(
            "g", "a hand", 0.02,
            schedule=Schedule(windows=((540, 1020),), days=(1, 2, 3, 4, 5, 6, 7)),
        )
    ]
    cases.append({
        "name": "out-of-schedule",
        "w": CLIP_W, "h": CLIP_H,
        "frames": [list(f) for f in CLIP],
        "ts": list(CLIP_TS),
        "spec": _spec_json(closed),
    })
    # Two goals sharing one cadence window — the case where an admission is decided by WHICH goal
    # got its brake stamped, not by the pixels.
    alternating = [_still(0) if i % 2 == 0 else _still(255) for i in range(6)]
    cases.append({
        "name": "two goals, shared cadence",
        "w": CLIP_W, "h": CLIP_H,
        "frames": alternating,
        "ts": [float(i) for i in range(6)],
        "spec": _spec_json([
            WatchCondition("first", "a hand", 0.2, cadence_s=2.0),
            WatchCondition("second", "a cup", 0.2, cadence_s=2.0),
        ]),
    })
    # A spec whose goals have all been cancelled.
    cases.append({
        "name": "no conditions",
        "w": CLIP_W, "h": CLIP_H,
        "frames": [list(f) for f in CLIP],
        "ts": list(CLIP_TS),
        "spec": {"version": 1, "conditions": []},
    })
    # ...and the state it is NOT: an edge that was never handed a spec at all. It has no instruction,
    # so it keeps the documented motion-only fallback (`defaultThreshold`). The Python gate cannot be
    # in this state — `MotionGate` always holds a spec — so it is asserted JS-side only, as the
    # control that says the empty-spec change is about a CANCELLED watch and not about the fallback.
    cases.append({
        "name": "no spec ever applied",
        "w": CLIP_W, "h": CLIP_H,
        "frames": [list(f) for f in CLIP],
        "ts": list(CLIP_TS),
        "spec": None,
    })
    return cases


#: Luma values a frame constructor must map to the same byte in both runtimes. Fractional, negative,
#: over-range and exactly-boundary — every value Python's ``_clamp_u8`` has a defined answer for.
FRAME_FILLS = (0, 1, 127, 254, 255, 256, 300, 1e9, -1, -300, 12.7, 254.999, 0.5, -0.5)


def js_admission() -> dict:
    """One call into the admission driver, carrying every scenario and every frame-byte case."""
    if "admission" not in _JS_CACHE:
        cases = _admission_payload()
        frame_bytes = [{"w": 2, "h": 1, "fill": fill} for fill in FRAME_FILLS]
        frame_bytes.append({"w": 3, "h": 1, "fill": [300, -5, 12.7]})
        frame_bytes.append({"w": 2, "h": 1, "fill": [NAN_SENTINEL, 40]})
        result = _run_node(_admission_driver_path(), {
            "admission": cases, "frameBytes": frame_bytes,
        })
        result["byName"] = {
            case["name"]: answer for case, answer in zip(cases, result["admission"])
        }
        result["frameFills"] = dict(zip(FRAME_FILLS, result["frameBytes"]))
        _JS_CACHE["admission"] = result
    return _JS_CACHE["admission"]


# ---------------------------------------------------------------------------
# 6. The vectors describe the shipped arithmetic
# ---------------------------------------------------------------------------


class TheModelIsFaithfulTests(unittest.TestCase):
    """Before a mutant means anything, the unforked model must BE the shipped function.

    A mutation result is a statement about the program that was mutated. If the model and
    ``motion_score`` disagree anywhere, every kill below describes some other program's arithmetic
    and none of it transfers. This is the cheapest way to be wrong, so it is asserted first.
    """

    def test_the_historical_vector_prefix_is_still_the_historical_prefix(self):
        # The mutation results below are stated relative to "the six vectors that predate the
        # widening". Pin them by shape: a reorder or an insert would silently redefine the claim.
        shapes = tuple(
            (case["w"], case["h"], case.get("stat", "mean"))
            for case in PARITY.SALIENCE_PAIRS[:HISTORICAL_SALIENCE_PREFIX]
        )
        self.assertEqual(shapes, HISTORICAL_SALIENCE_SHAPES)
        ego_shapes = tuple(
            (case["w"], case["h"], case["k"])
            for case in PARITY.EGO_PAIRS[:HISTORICAL_EGO_PREFIX]
        )
        self.assertEqual(ego_shapes, HISTORICAL_EGO_SHAPES)
        self.assertGreater(len(PARITY.SALIENCE_PAIRS), HISTORICAL_SALIENCE_PREFIX)
        self.assertGreater(len(PARITY.EGO_PAIRS), HISTORICAL_EGO_PREFIX)

    def test_the_unforked_salience_model_is_bit_identical_to_the_shipped_one(self):
        for index, case in enumerate(PARITY.SALIENCE_PAIRS):
            with self.subTest(vector=index, stat=case.get("stat", "mean")):
                mine = model_salience(
                    case["w"], case["h"], case["a"], case["b"],
                    stat=case.get("stat", "mean"),
                    region=_case_region(case),
                    grid=_case_grid(case),
                )
                # The gate's own comparator, not `assertAlmostEqual`: a tolerance here would let the
                # model drift from the function it is supposed to be standing in for.
                label, ok = PARITY.bit_identical(f"model salience[{index}]", _shipped_salience(case), mine)
                self.assertTrue(ok, label)

    def test_the_unforked_ego_model_is_bit_identical_to_the_shipped_one(self):
        for index, case in enumerate(PARITY.EGO_PAIRS):
            with self.subTest(vector=index, k=case["k"]):
                mine = model_ego(
                    case["w"], case["h"], case["a"], case["b"],
                    k=case["k"], region=_case_region(case),
                )
                label, ok = PARITY.bit_identical(f"model ego[{index}]", _shipped_ego(case), mine)
                self.assertTrue(ok, label)


# ---------------------------------------------------------------------------
# 7. Mutation-testing the vector set itself
# ---------------------------------------------------------------------------


def _salience_kills(fork: str, cases) -> list[int]:
    """Indices in ``cases`` where ``fork`` changes the answer — the vectors that kill this mutant.

    A kill is decided by ``compare.bit_identical``, the comparator ``make parity`` actually applies,
    so "dies" here means exactly "the release gate goes red".
    """
    kills = []
    for index, case in enumerate(cases):
        reference = _shipped_salience(case)
        mutant = model_salience(
            case["w"], case["h"], case["a"], case["b"],
            stat=case.get("stat", "mean"),
            region=_case_region(case),
            grid=_case_grid(case),
            fork=fork,
        )
        if not PARITY.bit_identical("m", reference, mutant)[1]:
            kills.append(index)
    return kills


def _ego_kills(fork: str, cases) -> list[int]:
    kills = []
    for index, case in enumerate(cases):
        reference = _shipped_ego(case)
        mutant = model_ego(
            case["w"], case["h"], case["a"], case["b"],
            k=case["k"], region=_case_region(case), fork=fork,
        )
        if not PARITY.bit_identical("m", reference, mutant)[1]:
            kills.append(index)
    return kills


class EveryForkMustDieTests(unittest.TestCase):
    """The gate must be shown to FAIL. Eight salience forks, seven ego forks, no survivors.

    ``make parity`` passing 33 checks told us nothing until mutation testing showed those checks
    could not die; it now runs 81. This is that experiment as an assertion, so the next person who
    "simplifies" a shape, a grid or a zone bound in ``SALIENCE_PAIRS`` finds out here instead of in
    a release.
    """

    def test_every_salience_fork_dies_on_the_current_vectors(self):
        survivors = []
        for fork, what in SALIENCE_FORKS.items():
            kills = _salience_kills(fork, PARITY.SALIENCE_PAIRS)
            with self.subTest(fork=fork):
                if not kills:
                    survivors.append(f"{fork} ({what})")
                self.assertTrue(
                    kills,
                    f"HOLLOW GATE: {what} passes every one of the "
                    f"{len(PARITY.SALIENCE_PAIRS)} frozen salience vectors at zero delta. The gate "
                    f"cannot see this fork, so it is not evidence about it.",
                )
        self.assertEqual(survivors, [])

    def test_every_ego_fork_dies_on_the_current_vectors(self):
        for fork, what in EGO_FORKS.items():
            with self.subTest(fork=fork):
                kills = _ego_kills(fork, PARITY.EGO_PAIRS)
                self.assertTrue(
                    kills,
                    f"HOLLOW GATE: {what} passes every one of the {len(PARITY.EGO_PAIRS)} frozen "
                    f"ego vectors at zero delta.",
                )

    def test_the_widening_is_what_closed_the_salience_forks(self):
        # The claim the widening rests on: every one of these forks was INVISIBLE to the six vectors
        # that came before it. If a fork turns out to have been detectable all along, then the
        # widening is not what closed it and the measurement that justified it was wrong.
        historical = PARITY.SALIENCE_PAIRS[:HISTORICAL_SALIENCE_PREFIX]
        for fork, what in SALIENCE_FORKS.items():
            with self.subTest(fork=fork):
                self.assertEqual(
                    _salience_kills(fork, historical), [],
                    f"{what} was already detectable on the historical six vectors",
                )
                self.assertTrue(_salience_kills(fork, PARITY.SALIENCE_PAIRS))

    def test_the_widening_is_what_closed_the_ego_search_fork(self):
        # The one the ego comment names: three horizontal-only pans on generous frames cannot see a
        # search range short by one, because no vector needs the outermost shift.
        historical = PARITY.EGO_PAIRS[:HISTORICAL_EGO_PREFIX]
        self.assertEqual(_ego_kills("search_range_short", historical), [])
        self.assertTrue(_ego_kills("search_range_short", PARITY.EGO_PAIRS))

    def test_each_fork_names_the_vectors_that_are_load_bearing_for_it(self):
        # Not an outcome assertion — a REPORT, pinned so a change is visible. Most forks are held by
        # exactly one vector, which is the honest state of a minimal counterexample set: delete that
        # vector and the fork walks straight back in.
        salience_map = {fork: _salience_kills(fork, PARITY.SALIENCE_PAIRS) for fork in SALIENCE_FORKS}
        ego_map = {fork: _ego_kills(fork, PARITY.EGO_PAIRS) for fork in EGO_FORKS}
        for fork, kills in salience_map.items():
            with self.subTest(fork=fork):
                self.assertTrue(kills)
                # Every killing vector must be one the widening added; a historical vector killing a
                # fork would contradict the test above.
                self.assertTrue(min(kills) >= HISTORICAL_SALIENCE_PREFIX)
        self.assertTrue(all(ego_map.values()))

    def test_the_documented_fork_values_are_true(self):
        # `compare.py` states, beside several vectors, the exact number a forked runtime returns
        # ("-> 1.000000, forks to 0.250000"). Those numbers are the only record of what each vector
        # is FOR. Recomputing them from an independent implementation is what keeps the comment from
        # becoming folklore.
        documented = (
            # (vector index, fork, the shipped value, the forked value)
            (6, "block_boundaries_rounded", 1.000000, 0.250000),
            (7, "p95_rank_floor", 0.827451, 0.866667),
            (8, "p95_rank_positional", 0.878431, 0.837473),
            (9, "region_far_half_up", 0.000000, 0.333333),
            (10, "region_origin_rounded", 0.166667, 0.000000),
            (11, "grid_unclamped", 1.000000, 0.000000),
            (11, "grid_clamp_short", 1.000000, 0.500000),
            (12, "grid_unclamped", 0.133333, 0.125490),
            (12, "grid_clamp_short", 0.133333, 0.111765),
            (12, "grid_clamp_long", 0.133333, 0.125490),
            (13, "grid_clamp_short", 0.176471, 0.162745),
        )
        for index, fork, shipped, forked in documented:
            case = PARITY.SALIENCE_PAIRS[index]
            with self.subTest(vector=index, fork=fork):
                self.assertAlmostEqual(_shipped_salience(case), shipped, places=6)
                mutant = model_salience(
                    case["w"], case["h"], case["a"], case["b"],
                    stat=case.get("stat", "mean"),
                    region=_case_region(case),
                    grid=_case_grid(case),
                    fork=fork,
                )
                self.assertAlmostEqual(mutant, forked, places=6)

    def test_the_documented_ego_fork_values_are_true(self):
        documented = (
            (6, "region_ignored", 0.000000, 0.012663),
            (7, "region_ignored", 0.312605, 0.012663),
            (7, "region_local_search", 0.312605, 0.309020),
            (8, "region_ignored", 0.288739, 0.290498),
            (8, "region_local_search", 0.288739, 0.272371),
            (8, "region_denominator", 0.288739, 0.262490),
        )
        for index, fork, shipped, forked in documented:
            case = PARITY.EGO_PAIRS[index]
            with self.subTest(vector=index, fork=fork):
                self.assertAlmostEqual(_shipped_ego(case), shipped, places=6)
                mutant = model_ego(
                    case["w"], case["h"], case["a"], case["b"],
                    k=case["k"], region=_case_region(case), fork=fork,
                )
                self.assertAlmostEqual(mutant, forked, places=6)


# ---------------------------------------------------------------------------
# 8. The two runtimes, on the frozen vectors and on seeded fuzz
# ---------------------------------------------------------------------------


class TheTwoRuntimesComputeTheSameNumberTests(unittest.TestCase):
    """``make parity``'s bar, re-asserted inside the suite — and then pushed past its vector set.

    The frozen vectors are counterexamples someone chose. Seeded fuzz is the population they were
    chosen from: small frames, every statistic, grids finer and coarser than the area, zones whose
    bounds land on an exact .5, and a zero-extent zone. Bit-identical means ``==``; the tolerance
    this comparison used to carry was ~6.4e6 ULP wide and hid a fork seven orders of magnitude
    larger than a rounding difference.
    """

    def setUp(self):
        self.assertIsNotNone(NODE, _NO_NODE)

    def test_the_frozen_salience_vectors_agree_across_runtimes(self):
        js = js_vectors()["saliences"]
        for index, case in enumerate(PARITY.SALIENCE_PAIRS):
            with self.subTest(vector=index, stat=case.get("stat", "mean")):
                label, ok = PARITY.bit_identical(
                    f"salience[{index}]", _shipped_salience(case), js[index]
                )
                self.assertTrue(ok, label)

    def test_the_frozen_ego_vectors_agree_across_runtimes(self):
        js = js_vectors()["egoScores"]
        for index, case in enumerate(PARITY.EGO_PAIRS):
            with self.subTest(vector=index, k=case["k"]):
                label, ok = PARITY.bit_identical(f"ego[{index}]", _shipped_ego(case), js[index])
                self.assertTrue(ok, label)

    def test_seeded_fuzz_salience_pairs_agree_across_runtimes(self):
        start = len(PARITY.SALIENCE_PAIRS)
        js = js_vectors()["saliences"][start:start + len(FUZZ_SALIENCE)]
        self.assertEqual(len(js), len(FUZZ_SALIENCE))
        checked = 0
        for index, case in enumerate(FUZZ_SALIENCE):
            if _zero_extent_zone(case):
                # Known-forked input class, pinned in `ADegenerateWatchOutZoneForksTests`. Routed,
                # not deleted: when the fork closes this population starts covering it again.
                continue
            checked += 1
            with self.subTest(case=index, shape=(case["w"], case["h"]), stat=case["stat"]):
                label, ok = PARITY.bit_identical(
                    f"fuzz salience[{index}] {case['w']}x{case['h']} {case['stat']} "
                    f"grid={case['blockGrid']} zone={case['region']}",
                    _shipped_salience(case),
                    js[index],
                )
                self.assertTrue(ok, label)
        # The routing above must not be able to empty the population.
        self.assertGreater(checked, len(FUZZ_SALIENCE) // 2)

    def test_seeded_fuzz_ego_pairs_agree_across_runtimes(self):
        js = js_vectors()["egoScores"][len(PARITY.EGO_PAIRS):]
        self.assertEqual(len(js), len(FUZZ_EGO))
        for index, case in enumerate(FUZZ_EGO):
            with self.subTest(case=index, shape=(case["w"], case["h"]), k=case["k"]):
                label, ok = PARITY.bit_identical(
                    f"fuzz ego[{index}] {case['w']}x{case['h']} k={case['k']} "
                    f"zone={case['region']}",
                    _shipped_ego(case),
                    js[index],
                )
                self.assertTrue(ok, label)

    def test_the_cross_runtime_comparison_can_fail(self):
        # The comparison above is only worth its runtime if a fork would break it. Feed the JS answer
        # for one vector against the SHIPPED answer for a different one and confirm the comparator
        # says no — the same mutation discipline applied to this file's own assertion.
        js = js_vectors()["saliences"]
        _, ok = PARITY.bit_identical("mutated", _shipped_salience(PARITY.SALIENCE_PAIRS[6]), js[7])
        self.assertFalse(ok)
        # ...and one ULP is enough. A tolerance would swallow this and call it a match.
        one_ulp_off = math.nextafter(js[3], math.inf)
        _, ok = PARITY.bit_identical("one ulp", one_ulp_off, js[3])
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# 9. Cross-cutting: what a frame IS, and one definition of audio energy
# ---------------------------------------------------------------------------


class ADegenerateWatchOutZoneScoresTheSameTests(unittest.TestCase):
    """A zero-extent watch-out zone scores the same in both runtimes (was ``...ForksTests``).

    FIND-PARITY-01, fixed edge-side. Found by the seeded fuzz, not by a frozen vector. The bounds are
    not symmetric — the ORIGIN truncates and the FAR bound rounds — so a zero-EXTENT ``Box`` is not a
    zero-PIXEL box: at ``x=0.1`` on a 7px frame that is ``trunc(0.7)=0`` and ``round(0.7)=1``, one
    real column, and ``_region_bounds`` returns exactly that. Python measured it and scored 1.0; the
    JS mean path short-circuited on ``!nonEmptyBox(region)`` and answered 0, so the same authored
    goal admitted EVERY frame in the harness and NO frame on device — the widest divergence this
    comparison can express. The block statistics on the identical zone always agreed, which is what
    localized it to the short-circuit rather than to the bounds arithmetic.

    The fix is two-level, and only the edge half of it is landed:

    * the VALUE type (``motionScore``) no longer short-circuits — a Box it is handed is measured over
      the span its bounds give, exactly as Python does, because a derived zone may legitimately
      collapse and must degrade rather than fork;
    * the AUTHORING/WIRE boundary (``contract.makeBox``/``parseBox``) now refuses a box that is not
      finite, not inside ``[0,1]``, or has no area — a zone that can never fire is an authoring
      mistake, not an input to interpret.

    The Python authoring layer has NOT been changed to match (``percept.contracts.Box`` validates
    nothing and ``harness.tier0._coerce_box`` builds one straight from ``goal.state``), so the
    degenerate zone is still authorable server-side. That remaining half is pinned by
    ``test_the_python_authoring_layer_still_accepts_what_the_edge_refuses`` below.
    """

    def setUp(self):
        self.assertIsNotNone(NODE, _NO_NODE)
        start = len(PARITY.SALIENCE_PAIRS) + len(FUZZ_SALIENCE)
        self.js = js_vectors()["saliences"][start:]
        self.assertEqual(len(self.js), len(DEGENERATE_ZONE_CASES))

    def test_a_zero_extent_zone_scores_the_same_in_both_runtimes(self):
        for index, case in enumerate(DEGENERATE_ZONE_CASES):
            with self.subTest(case=index, stat=case["stat"]):
                label, ok = PARITY.bit_identical(
                    f"zero-extent zone[{index}] {case['stat']}",
                    _shipped_salience(case),
                    self.js[index],
                )
                self.assertTrue(ok, label)

    def test_the_mean_path_measures_the_same_column_the_block_paths_do(self):
        # Same frames, same zone, three statistics. All three now read the one real pixel column, on
        # both sides. The two numbers that moved are `self.js[0]` (was 0) and nothing else: the block
        # paths never had the short-circuit, so pinning them is what keeps a "fix" that collapses all
        # three to 0 — agreement by going blind — from passing this class.
        mean_case, block_case, p95_case, zero_height = DEGENERATE_ZONE_CASES
        self.assertEqual(_shipped_salience(mean_case), 1.0)
        self.assertEqual(_shipped_salience(block_case), 1.0)
        self.assertEqual(_shipped_salience(p95_case), 1.0)
        self.assertEqual(self.js[0], 1)
        self.assertEqual(self.js[1], 1)
        self.assertEqual(self.js[2], 1)
        # ...and a zero-extent zone that really does collapse (0.1*3 = 0.3 truncates and rounds to
        # the same 0) reads 0 on both sides, so this is about the rounding, not about zero extent.
        self.assertEqual(_shipped_salience(zero_height), 0.0)
        self.assertEqual(self.js[3], 0)

    def test_the_python_authoring_layer_still_accepts_what_the_edge_refuses(self):
        # The REMAINING half of FIND-PARITY-01, pinned so closing it is visible rather than silent.
        # x=0.03 on a 24px-wide raster is 0.72 — trunc 0, round 1 — so this zero-width zone is one
        # real column and the Python gate admits every frame after the first. The edge now refuses
        # the same zone at `applyWatchSpec` (packages/percept-edge/test/contract.test.js and
        # edge.test.js pin that), so until `percept.contracts.Box` / `harness.tier0._coerce_box`
        # enforce the same three rules, a degenerate zone authored server-side is a spec the edge
        # will reject. When Python starts refusing it, this test fails and says so.
        zone = Box(x=0.03, y=0.1, w=0.0, h=0.5)
        alternating = [
            [0] * (CLIP_W * CLIP_H) if i % 2 == 0 else [255] * (CLIP_W * CLIP_H)
            for i in range(4)
        ]
        decisions, _ = _python_admissions(
            [WatchCondition("g", "a hand", 0.2, region_hint=zone)],
            frames=alternating,
            ts=(0.0, 1.0, 2.0, 3.0),
        )
        self.assertEqual(_as_marks(decisions), ".111")


class OneDefinitionOfAFrameTests(unittest.TestCase):
    """A luma value must become the same byte in both runtimes, or the vectors compare two scenes.

    This is the failure mode ``clampU8``'s own docstring names: ``Uint8Array``'s implicit conversion
    takes the low byte, so a 300-luma fill became 44 and -1 became 255 — the exact inversions of what
    the Python helper produces. A parity vector authored through both frame constructors would then
    hand the two runtimes DIFFERENT frames and compare their scores for bit-identity.
    """

    def setUp(self):
        self.assertIsNotNone(NODE, _NO_NODE)

    def test_a_scalar_luma_fill_saturates_identically(self):
        js = js_admission()["frameFills"]
        for fill in FRAME_FILLS:
            with self.subTest(luma=fill):
                expected = list(gray_frame(2, 1, fill).pixels)
                self.assertEqual(js[fill], expected)

    def test_an_authored_luma_array_saturates_identically(self):
        # Python has no array constructor that clamps, so the per-element answer is the public
        # single-pixel one. Each element must land on the byte `gray_frame` would give it.
        js = js_admission()["frameBytes"][len(FRAME_FILLS)]
        self.assertEqual(js, [gray_frame(1, 1, v).pixels[0] for v in (300, -5, 12.7)])

    def test_a_non_numeric_luma_is_the_one_documented_divergence(self):
        # `int(nan)` RAISES in Python — there is no saturating answer — so the JS twin maps NaN to 0,
        # "the same no-signal byte the default fill uses". Stated in `clampU8`'s docstring, and
        # pinned here so the divergence stays deliberate rather than becoming a surprise.
        js = js_admission()["frameBytes"][len(FRAME_FILLS) + 1]
        self.assertEqual(js, [0, 40])
        with self.assertRaises(ValueError):
            gray_frame(1, 1, float("nan"))


class OneDefinitionOfAudioEnergyTests(unittest.TestCase):
    """Mean-square, normalized, on both sides — never RMS, never one of each.

    The cross-cutting rule this stage carries: amplitude and mean-square must never be crossed
    unsquared. Both runtimes divide by ``32768`` (not 32767) and by the SAMPLE COUNT, and the
    comparison is exact: the two implementations sum in different orders and still land on the same
    double, so there is no drift for a tolerance to be protecting.
    """

    def setUp(self):
        self.assertIsNotNone(NODE, _NO_NODE)

    @staticmethod
    def _python_energy(samples) -> float:
        return pcm_energy(struct.pack(f"<{len(samples)}h", *samples))

    def test_the_shipped_energy_vectors_are_bit_identical_not_merely_close(self):
        js = js_vectors()["audioEnergies"]
        for index, case in enumerate(PARITY.AUDIO_FIXTURE["energy_vectors"]):
            with self.subTest(vector=case["name"]):
                mine = self._python_energy(case["samples"])
                self.assertEqual(
                    mine, js[index],
                    f"{case['name']}: python={mine!r} js={js[index]!r} — the parity gate compares "
                    f"these at 1e-12 (~4.5e3 ULP at this magnitude); they are in fact equal, so the "
                    f"tolerance is protecting nothing and hiding everything below it",
                )

    def test_seeded_fuzz_energy_windows_are_bit_identical(self):
        js = js_vectors()["audioEnergies"][len(PARITY.AUDIO_FIXTURE["energy_vectors"]):]
        self.assertEqual(len(js), len(FUZZ_PCM))
        for index, case in enumerate(FUZZ_PCM):
            with self.subTest(vector=case["name"], samples=len(case["samples"])):
                self.assertEqual(self._python_energy(case["samples"]), js[index])

    def test_an_rms_or_wrong_denominator_energy_would_die(self):
        # Mutation: the two ways this definition has actually been got wrong. Both must be visible
        # even under the parity gate's LOOSE 1e-12 comparison, or the energy check is decorative.
        samples = [8192, -8192, 8192, -8192]
        shipped = self._python_energy(samples)
        as_rms = math.sqrt(shipped)
        wrong_denominator = sum((s / 32767) ** 2 for s in samples) / len(samples)
        self.assertGreater(abs(shipped - as_rms), 1e-12)
        self.assertGreater(abs(shipped - wrong_denominator), 1e-12)
        # ...and a drift one ULP wide survives that tolerance while `==` kills it.
        one_ulp_off = math.nextafter(shipped, math.inf)
        self.assertLess(abs(shipped - one_ulp_off), 1e-12)
        self.assertNotEqual(shipped, one_ulp_off)


# ---------------------------------------------------------------------------
# 10. The product question: same media, same admission decision
# ---------------------------------------------------------------------------


class SameMediaSameAdmissionDecisionTests(unittest.TestCase):
    """Ten authored watches over one clip: the two runtimes must escalate on the same frames.

    A salience number agreeing is necessary and not sufficient. What a deployment buys is a judge
    call, and the decision to buy one runs through thresholds, per-goal cadence, the channel a goal
    asked for, and the fused noisy-OR. Those are compared here as FUNNELS — ``percept.eval.
    stage_attribution`` derives ``observed``/``admitted`` from the trace's own fold, so "the two
    runtimes admitted the same frames" and "the two runtimes buy the same number of calls per hour of
    source" are one assertion instead of two.
    """

    def setUp(self):
        self.assertIsNotNone(NODE, _NO_NODE)
        self.js = js_admission()["byName"]

    def _assert_agrees(self, name: str, conditions, **kw):
        python_decisions, python_gauges = _python_admissions(conditions, **kw)
        js_case = self.js[name]
        self.assertEqual(
            _as_marks(python_decisions), _as_marks(js_case["escalate"]),
            f"{name}: the two runtimes admitted different frames of the same clip "
            f"(python={_as_marks(python_decisions)} js={_as_marks(js_case['escalate'])})",
        )
        # The up-channel gauge on every admitted frame, bit-identical. A runtime could agree about
        # WHETHER to escalate and still report a different salience to the brain.
        for index, (admitted, js_motion) in enumerate(zip(js_case["escalate"], js_case["motion"])):
            if admitted:
                with self.subTest(watch=name, frame=index):
                    label, ok = PARITY.bit_identical(
                        f"{name} frame {index} motion", python_gauges[index], js_motion
                    )
                    self.assertTrue(ok, label)
        return python_decisions, js_case["escalate"]

    def test_every_authored_watch_admits_the_same_frames(self):
        for name, conditions in WATCHES.items():
            with self.subTest(watch=name):
                python_decisions, js_decisions = self._assert_agrees(name, conditions)
                # The funnel is the comparison of record: same observed, same admitted, same economy.
                python_funnel = _funnel_of(python_decisions, run_id="python-edge")
                js_funnel = _funnel_of(js_decisions, run_id="js-edge")
                self.assertEqual(dict(python_funnel.counts), dict(js_funnel.counts))
                self.assertEqual(
                    python_funnel.admission_economy, js_funnel.admission_economy
                )
                self.assertEqual(python_funnel.counts["observed"], len(CLIP))

    def test_the_fused_channel_admits_the_same_frames_with_audio_wired(self):
        self._assert_agrees(
            "fused with audio",
            FUSED_WATCH,
            energies=CLIP_ENERGY,
            wake_policy=FUSED_POLICY,
        )

    def test_a_measurement_only_run_reports_no_ratio_downstream(self):
        # Honesty, through the shared helper: this capability test traverses the measurement stage
        # only. Nothing was judged, so the judge step has NO ratio — not 0.0, which is a number a bar
        # would happily pass.
        decisions, _ = _python_admissions(WATCHES["localized block_max"])
        funnel = _funnel_of(decisions, run_id="meas-only")
        self.assertEqual(funnel.counts["judge_started"], 0)
        self.assertIsNone(funnel.ratio("judge"))
        self.assertIsNone(
            funnel.ratio("dispatch"),
            "measurement-only admissions declare no per-goal dispatch population",
        )
        self.assertGreater(funnel.counts["admitted"], 0)

    def test_the_funnel_comparison_can_fail(self):
        # Mutation of this file's own bar: flip ONE frame's decision and the funnel must move. A
        # comparison that reports the same table for two different decision streams would pass this
        # whole class while the runtimes disagreed on every frame.
        decisions, _ = _python_admissions(WATCHES["localized block_max"])
        mutated = list(decisions)
        first_admitted = mutated.index(True)
        mutated[first_admitted] = False
        self.assertNotEqual(
            dict(_funnel_of(decisions, run_id="a").counts),
            dict(_funnel_of(mutated, run_id="b").counts),
        )
        self.assertNotEqual(_as_marks(decisions), _as_marks(mutated))

    def test_a_still_prefix_costs_no_admissions_in_either_runtime(self):
        # The economy claim the gate exists for, asserted on both sides of the wire at once.
        for name in ("whole-frame mean", "localized block_max", "localized p95"):
            with self.subTest(watch=name):
                python_decisions, _ = _python_admissions(WATCHES[name])
                js_decisions = self.js[name]["escalate"]
                self.assertEqual(python_decisions[:2], [False, False])
                self.assertEqual(js_decisions[:2], [False, False])

    def test_a_localized_event_is_seen_by_the_block_statistic_and_not_by_the_mean(self):
        # The vectors' claim, restated as a product outcome on a real clip: frame 2 is a 3x3 patch on
        # a still 24x18 background. Both runtimes must agree about which watch can see it.
        mean_python, _ = _python_admissions(WATCHES["whole-frame mean"])
        block_python, _ = _python_admissions(WATCHES["localized block_max"])
        self.assertFalse(mean_python[2])
        self.assertTrue(block_python[2])
        self.assertFalse(self.js["whole-frame mean"]["escalate"][2])
        self.assertTrue(self.js["localized block_max"]["escalate"][2])


# ---------------------------------------------------------------------------
# 11. The divergences this comparison actually found
# ---------------------------------------------------------------------------


class EveryFiringGoalGetsItsCadenceBrakeTests(unittest.TestCase):
    """Two goals, one cadence window: the runtimes admit the same frames of the same clip.

    FIND-PARITY-02, fixed (was ``TheEdgeStampsOnlyTheFirstFiringGoalTests``).
    ``Edge._shouldEscalate`` used to RETURN on the first condition that fired, so only that goal's
    ``_lastEscalate`` was stamped, while ``MotionGate.evaluate`` evaluates every armed condition and
    stamps every one that fired. With two goals sharing a cadence the second goal's brake was never
    set on the JS side, so it admitted again on the very next frame — inside the window the
    deployment authored. The JS branch this sits on is commented "Twin of Python
    ``MotionGate.evaluate``", which is the contract it was violating.

    Measured on a six-frame alternating-luma clip with two goals at ``cadence_s=2``: python
    ``.1.1.1`` against js ``.11111`` — three extra judge calls the harness refuses. The edge loop now
    accumulates over every condition and stamps each one that fired, so both read ``.1.1.1``. The
    same comparison is frozen into the release gate as ``parity/compare.py::GATE_CASES``.
    """

    def setUp(self):
        self.assertIsNotNone(NODE, _NO_NODE)

    def test_two_goals_sharing_a_cadence_admit_the_same_frames(self):
        frames = [_still(0) if i % 2 == 0 else _still(255) for i in range(6)]
        ts = tuple(float(i) for i in range(6))
        conditions = [
            WatchCondition("first", "a hand", 0.2, cadence_s=2.0),
            WatchCondition("second", "a cup", 0.2, cadence_s=2.0),
        ]
        python_decisions, _ = _python_admissions(conditions, frames=frames, ts=ts)
        js_decisions = js_admission()["byName"]["two goals, shared cadence"]["escalate"]
        self.assertEqual(_as_marks(python_decisions), _as_marks(js_decisions))

    def test_a_second_goal_on_the_same_cadence_costs_nothing_extra(self):
        # The economy statement of the same fact, and the control that rules out a threshold or
        # salience explanation: ONE goal on that cadence admits the frames TWO goals do, on both
        # runtimes. The assertion that moved is the last one — it read `assertGreater(sum(js),
        # sum(two_goals))`, the edge buying calls the harness refused.
        frames = [_still(0) if i % 2 == 0 else _still(255) for i in range(6)]
        ts = tuple(float(i) for i in range(6))
        one_goal, _ = _python_admissions(
            [WatchCondition("first", "a hand", 0.2, cadence_s=2.0)], frames=frames, ts=ts
        )
        two_goals, _ = _python_admissions(
            [
                WatchCondition("first", "a hand", 0.2, cadence_s=2.0),
                WatchCondition("second", "a cup", 0.2, cadence_s=2.0),
            ],
            frames=frames,
            ts=ts,
        )
        self.assertEqual(_as_marks(one_goal), _as_marks(two_goals))
        self.assertEqual(_as_marks(two_goals), ".1.1.1")
        js_decisions = js_admission()["byName"]["two goals, shared cadence"]["escalate"]
        self.assertEqual(_as_marks(js_decisions), _as_marks(two_goals))
        self.assertEqual(sum(js_decisions), sum(two_goals))


class ACancelledWatchStopsBothRuntimesTests(unittest.TestCase):
    """A spec with no conditions: both runtimes stop looking.

    FIND-PARITY-03, fixed (was ``ACancelledWatchStillEscalatesOnTheEdgeTests``). ``MotionGate``
    documents "an empty spec means no active watch goals -> nothing to look for -> never escalate".
    ``Edge._shouldEscalate`` documented the opposite for the same input: "with no spec/conditions we
    fall back to defaultThreshold so motion-only still gates". Each rationale was written down; the
    DIVERGENCE was not, and it was reachable — every goal being cancelled compiles to exactly this
    spec. On device it meant frames kept leaving the camera after the last watch was cancelled, which
    is the privacy promise of the on-device topology failing in the one state where nothing can
    possibly matter.

    The edge now separates the two states the one branch used to serve. An APPLIED spec with no
    conditions never escalates (this class); an edge that has NEVER been handed a spec keeps the
    motion-only fallback, which is what makes a bare ``new Edge()`` useful and is not a state the
    Python gate can be in at all.
    """

    def setUp(self):
        self.assertIsNotNone(NODE, _NO_NODE)

    def test_a_spec_with_no_conditions_admits_nothing_in_either_runtime(self):
        python_decisions, _ = _python_admissions([])
        js_decisions = js_admission()["byName"]["no conditions"]["escalate"]
        self.assertEqual(_as_marks(python_decisions), _as_marks(js_decisions))

    def test_both_gates_stop_looking_when_the_last_goal_is_cancelled(self):
        python_decisions, _ = _python_admissions([])
        self.assertEqual(sum(python_decisions), 0)
        # ...and the edge stops too. This assertion read `assertGreater(sum(js), 0)` — the edge
        # publishing frames for a watch nobody has.
        self.assertEqual(sum(js_admission()["byName"]["no conditions"]["escalate"]), 0)

    def test_an_edge_that_was_never_configured_keeps_its_motion_only_fallback(self):
        # The control that says the fix is about a CANCELLED watch, not about the specless fallback:
        # the same clip through an edge that never received a spec still admits. Without this, "the
        # empty spec admits nothing" would also be satisfied by an edge that never admits anything.
        never_configured = js_admission()["byName"]["no spec ever applied"]["escalate"]
        self.assertGreater(sum(never_configured), 0)
        self.assertNotEqual(
            _as_marks(never_configured),
            _as_marks(js_admission()["byName"]["no conditions"]["escalate"]),
        )


class TheEdgeDropsTheScheduleTests(unittest.TestCase):
    """An out-of-window schedule suppresses admission in Python and is ignored by the edge.

    ``contract.js``'s ``parseWatchCondition`` states it: "``schedule`` (present on the Python side) is
    not mirrored here yet — it is tolerantly dropped (a known parity drift, tracked separately)."
    Documented intent with a rationale, so this is not filed as a defect — but it IS a cross-runtime
    admission divergence with a product consequence (a 09:00-17:00 watch admits at 02:00 on device),
    and the wire silently accepting a field it will not honour is what makes it invisible. This test
    pins the CURRENT behavior so closing the drift shows up here as an unexpected pass.
    """

    def setUp(self):
        self.assertIsNotNone(NODE, _NO_NODE)

    def test_the_schedule_is_honoured_in_python_and_dropped_on_the_edge(self):
        closed = [
            WatchCondition(
                "g", "a hand", 0.02,
                schedule=Schedule(windows=((540, 1020),), days=(1, 2, 3, 4, 5, 6, 7)),
            )
        ]
        # 02:00 on a Monday — outside a 09:00-17:00 window.
        python_decisions, _ = _python_admissions(
            closed, now=CivilTime(minute_of_day=120, weekday=1)
        )
        js_decisions = js_admission()["byName"]["out-of-schedule"]["escalate"]
        self.assertEqual(sum(python_decisions), 0)
        self.assertGreater(sum(js_decisions), 0)
        self.assertNotEqual(_as_marks(python_decisions), _as_marks(js_decisions))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
