"""Edge↔harness selected-contract parity + edge latency.

Compares the explicitly listed detector boundaries, motion scores, gate admissions, audio energy,
and ordered-audio result fields between the JS percept-edge route and Python percept-harness route
on identical inputs.
Passing means those selected fields conform for these frozen vectors; it is not a byte-for-byte
serialization proof or a claim about untested downstream precision/recall. Also measures edge frame
latency (JS Edge.processFrame throughput).

The motion, salience, and ego fields are compared with EXACT float equality (:func:`bit_identical`),
because that is the invariant their sources advertise; the gate admissions are compared as a
per-frame decision STRING (two runtimes can buy the same NUMBER of judge calls on different frames);
the detector-boundary and audio fields keep a small tolerance, since those paths are not
integer-exact. The vector lists are not decoration — each one was chosen against a measured fork of
the implementation, so read the notes above them before editing.

Run (node auto-discovered via PATH or ~/.nvm; or set PERCEPT_NODE):
    PYTHONPATH=packages/percept-vision \
        python3.12 packages/percept-edge/parity/compare.py
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import warnings
from pathlib import Path

from percept.adapters import AudioFrame, Box, WatchCondition, WatchSpec
from percept.events import AudioBoundary, AudioUnit, AudioWindow
from percept.harness.detectors import (
    AcousticOnset,
    MotionPeriodicity,
    should_wake,
    wake_score,
)
from percept.harness.tier0 import (
    DegenerateBlockPartitionWarning,
    MotionGate,
    ego_residual_score,
    motion_score,
)
from percept.runtime.audio import pcm_energy

HERE = Path(__file__).resolve().parent
AUDIO_FIXTURE = json.loads(
    (HERE.parent / "test" / "fixtures" / "audio_stream.golden.json").read_text()
)

# --- deterministic test cases (must produce identical output in both languages) ---
MOTION_SIGNALS = [
    [0, 1, 0, 1, 0, 1, 0],
    [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0],
    [0, 0.01, 0, 0.015, 0, 0.005, 0],                       # jitter below min_height -> no peaks
    [0, 0.3, 0.9, 0.4, 0.1, 0.7, 0.2, 0.05, 0.6, 0.1],      # noisy / irregular
]
ENERGY_SIGNALS = [
    [0.0, 0.5, 0.5, 0.0, 0.5],
    [0.0, 0.5, 0.2, 0.5, 0.2],                              # hysteresis: no re-fire while above low
    [0.0, 0.9, 0.9, 0.9, 0.9],                              # sustained -> single onset
    [0.0, 0.4, 0.0, 0.31, 0.0, 0.29, 0.0],                  # straddles the high threshold
    # THE SHIPPED BARS. The four vectors above are all far above BOTH the current default
    # (high=0.001) and the retired one (high=0.09), so they are blind to the number itself: the JS
    # edge shipped 0.09/0.0225 against Python's 0.001/0.00025 for a whole release and this file
    # matched anyway. 0.0277 is the loudest chunk in the entire 987 s annotated audio corpus — the
    # measurement that retired 0.09 — so it fires under the shipped bar and is inaudible under the
    # old one. -> 2 onsets; a runtime still on 0.09 forks to 0.
    [0.0, 0.0277, 0.0, 0.0277, 0.0],
    # ...and the RE-ARM bar from the same side: 0.0005 is above low=0.00025 (stays disarmed) and
    # below the retired low=0.0225 (would re-arm). -> 1 onset, forks to 2.
    [0.0, 0.01, 0.0005, 0.01],
]

# --- the wake fusion: the noisy-OR, its population, and what it names --------------------------
# Every case here was a measured fork of the implementation before the wake-fusion contract landed.
# The `signals` and `weights` populations are DIFFERENT sets on purpose in most of them — iterating
# the wrong one is the defect this section exists to catch, and a case whose two populations agree
# cannot see it. Compared with EXACT float equality on the score, plus the decision's `top_channel`
# and `wake` flag, because score and top channel are derived from the same contribution terms and a
# runtime could agree on one while forking on the other.
WAKE_CASES = [
    {"signals": {}, "policy": None},                                    # nothing measured
    {"signals": {"motion": 0.2, "semantic": 0.7}, "policy": None},      # legacy max, no policy
    {"signals": {"motion": 0.5, "presence": 0.5}, "policy": None},      # tie -> first channel
    # a measured channel the policy does NOT price: weight-population iteration deletes it (0.8),
    # signal-population iteration prices it at the default 1.0 (0.98).
    {"signals": {"presence": 0.9, "motion": 0.8}, "policy": {"weights": {"motion": 1.0}}},
    # a typo'd CHANNEL name: total deafness (0.0) under the old population, a no-op under the new.
    {"signals": {"presence": 0.9, "motion": 0.8}, "policy": {"weights": {"presenc": 1.0}}},
    # a weight on a channel that was never measured: it must cast no vote, not vote 0.0.
    {"signals": {"motion": 0.5}, "policy": {"weights": {"presence": 1.0, "motion": 1.0}}},
    # an explicit 0.0 prior IS how a channel is silenced — and the top channel must skip it.
    {"signals": {"presence": 0.9, "motion": 0.4}, "policy": {"weights": {"presence": 0.0, "motion": 1.0}}},
    # every contribution zero with signals present -> score 0.0 and NO top channel.
    {"signals": {"presence": 0.9}, "policy": {"weights": {"presence": 0.0}}},
    # a weight above 1 saturates its channel rather than overflowing.
    {"signals": {"semantic": 0.6}, "policy": {"threshold": 0.9, "weights": {"semantic": 2.0}}},
    # the floor: it lifts the score without naming a channel that produced it.
    {"signals": {}, "policy": {"threshold": 0.5, "floor": 1.0}},
    {"signals": {"motion": 0.1}, "policy": {"threshold": 0.5, "floor": 0.25}},
    # the default threshold on both sides (0.1), exactly at the bar.
    {"signals": {"motion": 0.1}, "policy": None},
    # several weak cues combining, the case the policy exists for.
    {"signals": {"presence": 0.5, "audio": 0.5},
     "policy": {"threshold": 0.45, "weights": {"presence": 0.7, "audio": 0.7}}},
    # an out-of-range reading clamps identically on both sides before it is priced.
    {"signals": {"motion": 9.0, "audio": -3.0}, "policy": {"weights": {"motion": 0.5}}},
    # the per-term clamp, which the outer clamp HIDES on a single channel (one overflowing term
    # saturates the OR either way). TWO overflowing terms make the misses multiply to a positive
    # number instead of zero: -> 1.000000, forks to 0.960000 if the term is left unclamped.
    {"signals": {"motion": 0.6, "audio": 0.6}, "policy": {"weights": {"motion": 2.0, "audio": 2.0}}},
    # the same clamp seen through the TOP CHANNEL: clamped, both terms are 1.0 and the tie takes the
    # first; unclamped, the second term is 1.2 and wins. -> 'motion', forks to 'audio'.
    {"signals": {"motion": 1.0, "audio": 0.6}, "policy": {"weights": {"motion": 1.0, "audio": 2.0}}},
]
W = H = 8
FRAME_PAIRS = [
    (W, H, [0] * 64, [0] * 64),                             # identical -> 0
    (W, H, [0] * 64, [255] * 64),                           # inverted -> 1
    (W, H, [0] * 64, [128] * 64),                           # half
    (W, H, list(range(64)), [(i * 3) % 256 for i in range(64)]),  # arbitrary
]


def _patch(base, w, px, py, pw, ph, val):
    """A copy of `base` (flat row-major) with a `pw x ph` block at `(px, py)` set to `val`."""
    out = list(base)
    for r in range(py, py + ph):
        for c in range(px, px + pw):
            out[r * w + c] = val
    return out


def _grad(w, h):
    return [(x * 5 + y * 3) % 256 for y in range(h) for x in range(w)]


def _box(region):
    """A case's normalized watch-out zone dict -> :class:`Box`; a case without one -> ``None``."""
    if not region:
        return None
    return Box(x=region["x"], y=region["y"], w=region["w"], h=region["h"])


def _pan(base, w, h, dx, dy=0):
    """`base` sampled at ``(x+dx, y+dy)`` with the edge row/column clamped — a camera pan.

    ``dy=0`` (the default) is the pure horizontal pan the original ego vectors were frozen on.
    """
    out = [0] * (w * h)
    for y in range(h):
        for x in range(w):
            sx = x + dx
            sx = 0 if sx < 0 else (w - 1 if sx >= w else sx)
            sy = y + dy
            sy = 0 if sy < 0 else (h - 1 if sy >= h else sy)
            out[y * w + x] = base[sy * w + sx]
    return out


# Mode A (localized statistic): a 6x6 event on a still 48x36 background — block_max/p95 catch it,
# whole-frame mean does not. Plus a non-divisible 13x11 case to lock the block-boundary math, and a
# region+block compose. Every case must be byte-for-byte identical Python<->JS.
_BG = [20] * (48 * 36)
_LOC = _patch(_BG, 48, 0, 0, 6, 6, 200)
_NDA = [(i * 7) % 256 for i in range(13 * 11)]
_NDB = [(i * 5 + 9) % 256 for i in range(13 * 11)]

# --- discriminating vectors: each one FAILS on exactly one arithmetic fork -------------------
# The six shape/statistic vectors above are structurally blind to the arithmetic this file claims to
# gate. 48x36 divides exactly by the default [8,6] grid, so floor- and round-boundaries coincide;
# 13x11 carries near-uniform |diff|, so block_max and p95 return the same block however the blocks
# are cut; the ROI bounds (0.25*48, 0.25*36) are integral, so the half-to-even rounding never runs;
# and no case has an active area smaller than the grid, so the clamping rule is never exercised.
# Mutation testing confirmed it: six separate forks of `motionScore` each scored max|delta| = 0.0
# across all six vectors. Every vector below is a measured counterexample to exactly one of those
# forks — do not "simplify" a shape, a grid, or a region bound here without re-running mutation
# testing, or the gate quietly goes back to gating nothing.
_HOT_A = [0] * (13 * 11)
_HOT_B = _patch(_HOT_A, 13, 0, 0, 1, 1, 255)          # one hot pixel in the (1x1) top-left block
_GRAD40 = _grad(40, 40)
_GRAD36 = _grad(36, 36)
_COL_A = [0] * (10 * 10)
_COL_B = _patch(_COL_A, 10, 2, 0, 1, 10, 255)         # all the diff in column 2 alone
_TINY_A = [0] * (3 * 2)
_TINY_B = _patch(_TINY_A, 3, 0, 0, 1, 1, 255)         # 3x2 area, smaller than the default grid
_GRAD65 = _grad(6, 5)                                 # under the grid in BOTH dims
_GRAD86 = _grad(8, 6)                                 # exactly the grid — the clamp must not bite

SALIENCE_PAIRS = [
    {"w": 48, "h": 36, "a": _BG, "b": _LOC, "stat": "mean"},        # same input, mean stays low...
    {"w": 48, "h": 36, "a": _BG, "b": _LOC, "stat": "block_max"},   # ...block_max lights the event
    {"w": 48, "h": 36, "a": _BG, "b": _LOC, "stat": "p95"},
    {"w": 13, "h": 11, "a": _NDA, "b": _NDB, "stat": "block_max"},  # non-divisible boundary lock
    {"w": 13, "h": 11, "a": _NDA, "b": _NDB, "stat": "p95"},
    {"w": 48, "h": 36, "a": _BG, "b": _LOC, "stat": "block_max",    # ROI + block compose
     "region": {"x": 0.0, "y": 0.0, "w": 0.25, "h": 0.25}},
    # floor-of-product block boundaries. 13 // 8 makes the top-left block 1px wide; rounding the
    # boundaries instead makes it 2x2, diluting the hot pixel. -> 1.000000, forks to 0.250000.
    {"w": 13, "h": 11, "a": _HOT_A, "b": _HOT_B, "stat": "block_max", "blockGrid": [8, 6]},
    # nearest-rank p95. n = 5*4 = 20 blocks, so 0.95n is an INTEGER and ceil(0.95n)-1 = 18 parts
    # company with floor(0.95n) = 19 (the max block). -> 0.827451, forks to 0.866667.
    {"w": 40, "h": 40, "a": [0] * (40 * 40), "b": _GRAD40, "stat": "p95", "blockGrid": [5, 4]},
    # the same rank from the other side. n = 36, so ceil(0.95n)-1 = 34 but a positional
    # round(0.95*(n-1)) = 33 — a different block. -> 0.878431, forks to 0.837473.
    {"w": 36, "h": 36, "a": [0] * (36 * 36), "b": _GRAD36, "stat": "p95", "blockGrid": [6, 6]},
    # the region FAR bound at an exact .5: 0.25*10 = 2.5, and Python rounds half-to-even -> 2, so
    # column 2 is outside the zone. Math.round would take it -> 3. -> 0.000000, forks to 0.333333.
    {"w": 10, "h": 10, "a": _COL_A, "b": _COL_B, "stat": "mean",
     "region": {"x": 0.0, "y": 0.0, "w": 0.25, "h": 1.0}},
    # the region ORIGIN, which truncates rather than rounds: 0.26*10 = 2.6 -> x0 = 2, so column 2 is
    # inside the zone; rounding the origin would drop it. -> 0.166667, forks to 0.000000.
    {"w": 10, "h": 10, "a": _COL_A, "b": _COL_B, "stat": "mean",
     "region": {"x": 0.26, "y": 0.0, "w": 0.5, "h": 1.0}},
    # --- the clamping rule (the grid is fitted to the active area, never padded out to it) ------
    # These three used to gate "an empty block contributes 0.0 rather than being skipped". That fork
    # is now unreachable — the grid is clamped to `min(G, a)`, so no block is ever empty and
    # mutating the 0-px branch is a no-op mutation. What replaced it is a stricter rule with three
    # ways to get it wrong (drop the clamp, clamp one short, clamp one long), and all three are
    # measured counterexamples below. Re-derive these baselines only from a run of the reference
    # implementation, never by hand — a p95 rank is not a number anyone should be predicting.
    #
    # a 3x2 area under the default [8,6] grid. Clamped it is a 3x2 grid of 6 real 1-px blocks and
    # the rank lands on the hot one. -> 1.000000; drop the clamp (48 blocks, 42 of them phantom
    # 0.0 padding) and it forks to 0.000000; clamp one short and it forks to 0.500000.
    {"w": 3, "h": 2, "a": _TINY_A, "b": _TINY_B, "stat": "p95"},
    # a 6x5 gradient: under the grid in BOTH dims, and every block carries a DIFFERENT value, so the
    # rank moves under all three mis-clamps instead of saturating. -> 0.133333; no clamp forks to
    # 0.125490, one short to 0.111765, one long to 0.125490.
    {"w": 6, "h": 5, "a": [0] * (6 * 5), "b": _GRAD65, "stat": "p95"},
    # the boundary from the other side: an 8x6 area is EXACTLY the [8,6] grid, so the clamp must not
    # bite at all. -> 0.176471, forks to 0.162745 if it fires one pixel early.
    {"w": 8, "h": 6, "a": [0] * (8 * 6), "b": _GRAD86, "stat": "p95"},
    # --- a ZERO-EXTENT zone, which the bounds do NOT collapse (FIND-PARITY-01) -------------------
    # Every region vector above encloses area, so none of them can see what the two runtimes do with
    # a zone that does not. The bounds are asymmetric — the origin TRUNCATES and the far bound ROUNDS
    # — so a zero-extent Box is only a zero-PIXEL box when that shared product rounds down: at x=0.1
    # on a 7px raster it is trunc(0.7)=0 .. round(0.7)=1, one real column. Python measured it and the
    # JS mean path short-circuited to 0 on `!nonEmptyBox(region)`, which is the widest admission fork
    # this gate can express: the same authored goal admitted EVERY frame server-side and NO frame on
    # device. -> 1.000000, forks to 0.000000. The zone is refused at the authoring/wire boundary now,
    # but `motionScore` still takes a Box as a value (a derived zone may collapse), so the two
    # runtimes must still answer it identically.
    {"w": 7, "h": 3, "a": [0] * 21, "b": [255] * 21, "stat": "mean",
     "region": {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.5}},
    # the same zone on a BLOCK stat, which never had the short-circuit: it agreed with Python
    # throughout, and pinning it is what keeps a "fix" that collapses all three to 0 from passing.
    # -> 1.000000.
    {"w": 7, "h": 3, "a": [0] * 21, "b": [255] * 21, "stat": "block_max",
     "region": {"x": 0.1, "y": 0.1, "w": 0.0, "h": 0.5}},
    # the control from the other side: 0.1*3 = 0.3 truncates AND rounds to 0, so THIS zone really
    # does collapse to an empty span and 0 is the right answer on both sides. Together with the two
    # above it says the divergence was about the rounding, not about zero-extent zones in general.
    # -> 0.000000.
    {"w": 7, "h": 3, "a": [0] * 21, "b": [255] * 21, "stat": "mean",
     "region": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.0}},
]

# Mode B (ego compensation): a pure pan residual ~0 while its raw mean is high; a localized subject the
# shift cannot explain away (residual preserved); a non-divisible noise pair.
_GRAD = _grad(48, 36)
# The first three vectors are all horizontal-only motion on generous frames (k << min(w,h)), which
# leaves the shift SEARCH under-covered: mutation testing showed a fork that shrinks the range to
# [-(k-1), k-1] scores max|delta| = 0.0 across them. The last three pin the search extent itself.
_GRAD20 = _grad(20, 16)
EGO_PAIRS = [
    {"w": 48, "h": 36, "a": _GRAD, "b": _pan(_GRAD, 48, 36, 2), "k": 4},       # pan -> ~0
    {"w": 48, "h": 36, "a": _GRAD, "b": _patch(_GRAD, 48, 10, 10, 6, 6, 255), "k": 4},  # subject preserved
    {"w": 13, "h": 11, "a": _NDA, "b": _NDB, "k": 3},                          # noise, non-divisible
    # a VERTICAL pan of exactly k: only a dy = +/-k shift explains it, so a dx-only search or a
    # range short by one cannot reach the residual. -> 0.000000.
    {"w": 20, "h": 16, "a": _GRAD20, "b": _pan(_GRAD20, 20, 16, 0, 4), "k": 4},
    # k LARGER than the frame (5x3, k=4): exercises the `overlap <= 0 -> skip this shift` guard and
    # makes the overlap-vs-whole-frame denominator visible. -> 0.003922.
    {"w": 5, "h": 3, "a": [(i * 37) % 256 for i in range(15)],
     "b": [(i * 53 + 7) % 256 for i in range(15)], "k": 4},
    # identical frames: the minimum sits exactly at the (0,0) shift, so dropping it is visible. -> 0.
    {"w": 11, "h": 7, "a": _NDA[: 11 * 7], "b": _NDA[: 11 * 7], "k": 2},
    # --- the watch-out zone: what the residual MEASURES, versus what it ESTIMATES ---------------
    # `region` masks the residual sum and its denominator to `region ∩ overlap`, while the shift
    # search stays whole-frame. Three vectors, because there are three separable ways to get it
    # wrong: ignore the region, search within the region, or divide by the region instead of by the
    # intersection.
    #
    # the bug's own signature: a 6x6 subject at (10,10), a zone that excludes it. -> 0.000000, and
    # 0.012663 (the whole-frame residual) if the region is dropped on the floor as it used to be.
    {"w": 48, "h": 36, "a": _GRAD, "b": _patch(_GRAD, 48, 10, 10, 6, 6, 255), "k": 4,
     "region": {"x": 0.0, "y": 0.0, "w": 0.125, "h": 0.1}},
    # the same subject with a zone AROUND it: the mask must not zero what it contains. -> 0.312605;
    # a region-local shift search forks to 0.309020, dropping the region to 0.012663.
    {"w": 48, "h": 36, "a": _GRAD, "b": _patch(_GRAD, 48, 10, 10, 6, 6, 255), "k": 4,
     "region": {"x": 0.1875, "y": 0.25, "w": 0.2, "h": 0.2}},
    # non-divisible noise, right half, k=3: no shift explains it, so the residual is non-zero
    # everywhere and every part of the arithmetic is visible. -> 0.288739; drop the region ->
    # 0.290498, search locally -> 0.272371, divide by the region instead of the intersection ->
    # 0.262490. This is the vector that pins the DENOMINATOR.
    {"w": 13, "h": 11, "a": _NDA, "b": _NDB, "k": 3,
     "region": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}},
]
# --- the GATE: same media + same authored WatchSpec -> the same ADMITTED FRAMES ------------------
# Everything above compares a FUNCTION. A deployment does not buy a number, it buys a judge call, and
# the decision to buy one runs through per-goal cadence and through what an APPLIED-but-empty spec
# means — neither of which any salience vector can reach. Both were measured forks:
#
#   * the edge RETURNED on the first firing condition, so only that goal's cadence brake was stamped
#     and every other goal sharing the frame admitted again on the very next one (FIND-PARITY-02);
#   * an applied spec with no conditions fell through to the specless `defaultThreshold` fallback, so
#     a device kept publishing frames after the last watch was cancelled (FIND-PARITY-03).
#
# Compared as a decision STRING (`.1.1.1`), not a count: two runtimes can buy the same number of
# calls on different frames. The clip is alternating luma, so motion is 1.0 on every frame after the
# first and the ONLY thing that can suppress an admission is the policy under test.
_GATE_W, _GATE_H = 24, 18
_GATE_FRAMES = [
    ([0] * (_GATE_W * _GATE_H) if i % 2 == 0 else [255] * (_GATE_W * _GATE_H)) for i in range(6)
]
_GATE_TS = [float(i) for i in range(6)]

# A SECOND clip, for the over-correction of the same bug. Every goal above is whole-frame, so they
# all fire on the same frames and a runtime that stamped a goal which never fired would score the
# same marks. Here the first change misses the left quarter entirely: a zone goal does NOT fire on
# frame 1 and DOES on frame 2, so a brake it never earned suppresses a real admission.
_ZONE_LEFT = {"x": 0.0, "y": 0.0, "w": 0.25, "h": 1.0}
_ZONE_FRAMES = [
    [0] * (_GATE_W * _GATE_H),
    # everything RIGHT of the zone flips (whole-frame mean 0.75, in-zone 0.0)
    [0 if (i % _GATE_W) < 6 else 255 for i in range(_GATE_W * _GATE_H)],
    [255] * (_GATE_W * _GATE_H),   # ...now the zone flips too (in-zone 1.0)
    [0] * (_GATE_W * _GATE_H),
    [255] * (_GATE_W * _GATE_H),
    [0] * (_GATE_W * _GATE_H),
]


def _goal(goal_id, **overrides):
    """One condition of the shared shape: threshold 0.2, cadence 2 s, motion channel."""
    condition = {
        "goal_id": goal_id, "anchor_text": "a hand", "threshold": 0.2, "cadence_s": 2.0,
    }
    condition.update(overrides)
    return condition


GATE_CASES = [
    # two goals whose thresholds are cleared by the SAME frame, sharing one cadence window.
    # -> '.1.1.1'; an edge that stamps only the first goal forks to '.11111'.
    {"name": "two goals, one cadence window", "conditions": [_goal("first"), _goal("second")]},
    # the control that rules out threshold/salience explanations: ONE goal, same cadence, same clip.
    # Two goals must not cost more than one. -> '.1.1.1' on both sides before AND after the fix.
    {"name": "one goal, same cadence (control)", "conditions": [_goal("only")]},
    # three, because "stamp the first" and "stamp the first two" are different bugs.  -> '.1.1.1'.
    {"name": "three goals, one cadence window",
     "conditions": [_goal("a"), _goal("b"), _goal("c")]},
    # the accumulation must not become "brake everything once anything fires": `deaf` can never clear
    # 1.5, so its brake must never be stamped and its presence must not move the marks. -> '.1.1.1'.
    {"name": "a goal that never clears its bar",
     "conditions": [_goal("loud"), _goal("deaf", threshold=1.5)]},
    # every goal cancelled — the spec `compile_watchspec` emits for an empty goal list. Nothing to
    # look for. -> '......'; an edge falling back to its specless default forks to '.11111'.
    {"name": "every goal cancelled (empty spec)", "conditions": []},
    # ...and the control that says the empty case is about the SPEC, not about the clip: one armed
    # goal with no cadence admits every frame after the first. -> '.11111'.
    {"name": "one armed goal, no cadence (control)",
     "conditions": [_goal("armed", cadence_s=None)]},
    # THE OVER-CORRECTION of FIND-PARITY-02, on the zone clip: only a goal that FIRED may be braked.
    # The zone goal cannot fire on frame 1 (nothing moved in its quarter) and fires on frame 2; a
    # runtime that stamps every armed condition instead of every firing one brakes it at frame 1 and
    # eats the real admission at frame 2. -> '.11111', forks to '.1.1.1'.
    {"name": "a brake only a goal that fired may earn",
     "frames": _ZONE_FRAMES,
     "conditions": [_goal("whole"), _goal("zone", region_hint=_ZONE_LEFT)]},
]

LATENCY = {"width": 32, "height": 32, "frames": 5000}


def _gate_condition(condition: dict) -> WatchCondition:
    """One GATE_CASES condition dict -> the Python :class:`WatchCondition` the harness gates on.

    The SAME dict is sent to the JS side as wire JSON, so both runtimes are driven from one authored
    description of the watch rather than from two hand-kept copies of it.
    """
    return WatchCondition(
        goal_id=condition["goal_id"],
        anchor_text=condition["anchor_text"],
        threshold=condition["threshold"],
        region_hint=_box(condition.get("region_hint")),
        cadence_s=condition.get("cadence_s"),
        gate_on=condition.get("gate_on", "motion"),
        salience_stat=condition.get("salience_stat", "mean"),
        ego_compensate=condition.get("ego_compensate", False),
    )


def py_gate_admissions(case: dict) -> list[bool]:
    """Drive the Python :class:`MotionGate` over the gate clip: one escalate decision per frame."""
    gate = MotionGate(WatchSpec(conditions=[_gate_condition(c) for c in case["conditions"]]))
    return [
        gate.evaluate((_GATE_W, _GATE_H, tuple(pixels)), ts=ts).escalate
        for pixels, ts in zip(case.get("frames", _GATE_FRAMES), _GATE_TS)
    ]


def _marks(decisions) -> str:
    """``.1.1.1`` — a per-frame decision string, so a mismatch shows WHICH frame moved."""
    return "".join("1" if d else "." for d in decisions)


def py_detector(cls, signal):
    det = cls()
    out = []
    for t, v in enumerate(signal):
        for b in det.push(float(t), v):
            out.append({"detector": b.detector, "kind": b.kind, "t": b.t, "value": b.value})
    return out


def run_js(cases):
    node = os.environ.get("PERCEPT_NODE") or shutil.which("node")
    if node is None:
        nvm = Path.home() / ".nvm/versions/node"
        cands = sorted(nvm.glob("*/bin/node")) if nvm.exists() else []
        node = str(cands[-1]) if cands else None
    if node is None:
        sys.exit("node not found (set PERCEPT_NODE or add node to PATH)")
    p = subprocess.run(
        [node, str(HERE / "edge_runtime.mjs")], input=json.dumps(cases),
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        sys.exit(f"node failed:\n{p.stderr}")
    return json.loads(p.stdout)


def bit_identical(name, ps, jsc):
    """A ``(label, ok)`` check demanding EXACT float equality between the Python and JS value.

    The salience and ego paths accumulate integer absolute differences and divide ONCE by an integer
    denominator, so both languages land on the same IEEE-754 double on any conforming platform.
    That is the invariant `motion.js` and `harness/tier0.py` advertise ("bit-identical", "bit-for-
    bit"), and only ``==`` actually tests it: the previous ``abs(delta) <= 1e-9`` was ~6.4e6 ULP at
    a typical magnitude of 0.7, so a drift seven orders of magnitude larger than a rounding
    difference still passed. A mismatch prints both values, since the point of the gate is to say
    which side moved.
    """
    ok = ps == jsc
    label = f"{name} = {ps:.6f}"
    if not ok:
        label = f"{name}  py={ps!r} != js={jsc!r}  (delta {abs(ps - jsc):.3e})"
    return label, ok


def boundaries_equal(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x["detector"] != y["detector"] or x["kind"] != y["kind"]:
            return False
        if abs(x["t"] - y["t"]) > tol or abs(x["value"] - y["value"]) > tol:
            return False
    return True


def main():
    cases = {
        "motionSignals": MOTION_SIGNALS,
        "energySignals": ENERGY_SIGNALS,
        "framePairs": [{"w": w, "h": h, "a": a, "b": b} for (w, h, a, b) in FRAME_PAIRS],
        "saliencePairs": SALIENCE_PAIRS,
        "egoPairs": EGO_PAIRS,
        "wakeCases": WAKE_CASES,
        "gateCases": [
            {
                "name": case["name"],
                "w": _GATE_W, "h": _GATE_H,
                "frames": case.get("frames", _GATE_FRAMES), "ts": _GATE_TS,
                "spec": {"version": 1, "conditions": case["conditions"]},
            }
            for case in GATE_CASES
        ],
        "audioEnergyVectors": AUDIO_FIXTURE["energy_vectors"],
        "audioTimelineVectors": AUDIO_FIXTURE["timeline_vectors"],
        "latency": LATENCY,
    }
    js = run_js(cases)

    py_motion = [py_detector(MotionPeriodicity, s) for s in MOTION_SIGNALS]
    py_onset = [py_detector(AcousticOnset, s) for s in ENERGY_SIGNALS]
    py_scores = [motion_score((w, h, tuple(a)), (w, h, tuple(b))) for (w, h, a, b) in FRAME_PAIRS]
    with warnings.catch_warnings():
        # Three vectors deliberately drive an area smaller than the grid, which is exactly the
        # condition `DegenerateBlockPartitionWarning` exists to report. Reporting it here would be
        # noise on top of the gate's own output — the vectors' comments say why they are shaped that
        # way, and their baselines ARE the clamped answers.
        warnings.simplefilter("ignore", DegenerateBlockPartitionWarning)
        py_salience = [
            motion_score(
                (c["w"], c["h"], tuple(c["a"])), (c["w"], c["h"], tuple(c["b"])),
                region=_box(c.get("region")),
                stat=c.get("stat", "mean"),
                # `blockGrid` is optional per case and defaults to [8, 6] on BOTH sides — the JS
                # runtime reads the same key with the same default, so a case that omits it is
                # unaffected.
                block_grid=tuple(c.get("blockGrid") or (8, 6)),
            )
            for c in SALIENCE_PAIRS
        ]
    py_ego = [
        ego_residual_score(
            (c["w"], c["h"], tuple(c["a"])), (c["w"], c["h"], tuple(c["b"])),
            k=c["k"],
            # `region` is optional per case and defaults to no zone on BOTH sides — the JS runtime
            # reads the same key with the same default, so a case that omits it is unaffected.
            region=_box(c.get("region")),
        )
        for c in EGO_PAIRS
    ]
    py_wake = [wake_score(c["signals"], c.get("policy")) for c in WAKE_CASES]
    py_gate = [py_gate_admissions(case) for case in GATE_CASES]
    py_wake_decisions = [should_wake(c["signals"], c.get("policy")) for c in WAKE_CASES]
    py_audio_energy = [
        pcm_energy(struct.pack(f"<{len(c['samples'])}h", *c["samples"]))
        for c in AUDIO_FIXTURE["energy_vectors"]
    ]
    py_audio_timelines = []
    for case in AUDIO_FIXTURE["timeline_vectors"]:
        timeline = AudioWindow(seconds=10)
        results = []
        for row in case["units"]:
            boundaries = tuple(
                AudioBoundary(boundary["kind"], boundary["t"])
                for boundary in row.get("boundaries", ())
            )
            unit = AudioUnit(
                seq=row["seq"],
                t_start=row["t_start"],
                t_end=row["t_end"],
                chunk=AudioFrame(
                    pcm=b"\x00\x00",
                    sample_rate=row.get("sample_rate", 16_000),
                    channels=row.get("channels", 1),
                ),
                sample_rate=row.get("sample_rate"),
                channels=row.get("channels"),
                direction=row.get("direction", "unknown"),
                participant_id=row.get("participant_id"),
                boundaries=boundaries,
            )
            results.append(timeline.push_unit(unit).as_dict())
        py_audio_timelines.append({
            "results": results,
            "order": [unit.seq for unit in timeline.units()],
            "directions": [unit.direction for unit in timeline.units()],
            "participants": [unit.participant_id for unit in timeline.units()],
            "boundaryKinds": [boundary.kind for boundary in timeline.boundaries()],
        })

    checks = []
    for i, (pj, jj) in enumerate(zip(py_motion, js["motionBoundaries"])):
        checks.append((f"motion-periodicity[{i}]  ({len(pj)} peaks)", boundaries_equal(pj, jj)))
    for i, (pj, jj) in enumerate(zip(py_onset, js["onsetBoundaries"])):
        checks.append((f"acoustic-onset[{i}]     ({len(pj)} onsets)", boundaries_equal(pj, jj)))
    for i, (ps, jsc) in enumerate(zip(py_scores, js["motionScores"])):
        checks.append(bit_identical(f"motion-gate score[{i}]", ps, jsc))
    for i, (ps, jsc) in enumerate(zip(py_salience, js["saliences"])):
        stat = SALIENCE_PAIRS[i].get("stat", "mean")
        roi = "+roi" if SALIENCE_PAIRS[i].get("region") else ""
        grid = SALIENCE_PAIRS[i].get("blockGrid")
        grid = f" {grid[0]}x{grid[1]}" if grid else ""
        checks.append(bit_identical(f"salience[{i}] {stat}{roi}{grid}", ps, jsc))
    for i, (ps, jsc) in enumerate(zip(py_ego, js["egoScores"])):
        roi = "+roi" if EGO_PAIRS[i].get("region") else ""
        checks.append(bit_identical(f"ego-residual[{i}]{roi} (k={EGO_PAIRS[i]['k']})", ps, jsc))
    for i, (ps, jsc) in enumerate(zip(py_wake, js["wakeScores"])):
        weights = sorted((WAKE_CASES[i].get("policy") or {}).get("weights", {}))
        label = f"wake-score[{i}] {sorted(WAKE_CASES[i]['signals'])} x {weights}"
        checks.append(bit_identical(label, ps, jsc))
    for i, (pd, jt, jw) in enumerate(
        zip(py_wake_decisions, js["wakeTopChannels"], js["wakeFlags"])
    ):
        same = pd.top_channel == jt and pd.wake == jw
        label = f"wake-decision[{i}] top={pd.top_channel!r} wake={pd.wake}"
        if not same:
            label = (
                f"wake-decision[{i}]  py=({pd.top_channel!r}, {pd.wake}) != "
                f"js=({jt!r}, {jw})"
            )
        checks.append((label, same))
    for i, (pd, jd) in enumerate(zip(py_gate, js["gateAdmissions"])):
        name = GATE_CASES[i]["name"]
        py_marks, js_marks = _marks(pd), _marks(jd)
        same = py_marks == js_marks
        label = f"gate-admission[{i}] {name} = {py_marks}"
        if not same:
            label = f"gate-admission[{i}] {name}  py={py_marks} != js={js_marks}"
        checks.append((label, same))
    for i, (ps, jsc) in enumerate(zip(py_audio_energy, js["audioEnergies"])):
        name = AUDIO_FIXTURE["energy_vectors"][i]["name"]
        checks.append((f"audio-energy[{i}] {name} = {ps:.6f}", abs(ps - jsc) <= 1e-12))
    for i, (py_timeline, js_timeline) in enumerate(
        zip(py_audio_timelines, js["audioTimelines"])
    ):
        name = AUDIO_FIXTURE["timeline_vectors"][i]["name"]
        same = (
            py_timeline["order"] == js_timeline["order"]
            and py_timeline["directions"] == js_timeline["directions"]
            and py_timeline["participants"] == js_timeline["participants"]
            and py_timeline["boundaryKinds"] == js_timeline["boundaryKinds"]
        )
        if same:
            keys = (
                "accepted", "duplicate", "reordered", "missing", "gap_s", "overlap_s",
                "format_changed", "expired", "expired_bytes", "evicted", "evicted_bytes",
            )
            for py_result, js_result in zip(py_timeline["results"], js_timeline["results"]):
                for key in keys:
                    left, right = py_result[key], js_result[key]
                    if isinstance(left, float):
                        same = same and abs(left - right) <= 1e-12
                    else:
                        same = same and left == right
        checks.append((f"audio-timeline[{i}] {name}", same))

    print("=" * 72)
    print("  EDGE <-> HARNESS SELECTED-CONTRACT PARITY   (JS vs Python)")
    print("=" * 72)
    allok = True
    for name, ok in checks:
        allok = allok and ok
        print(f"   [{'MATCH' if ok else 'DIFF!'}]  {name}")
    print("-" * 72)
    print(
        f"   {len(checks)} checks - "
        f"{'ALL SELECTED FIELDS MATCH ON FROZEN VECTORS' if allok else 'MISMATCH FOUND'}"
    )
    lat = js.get("latency")
    if lat:
        print("-" * 72)
        print(f"  EDGE LATENCY  (JS Edge.processFrame, {lat['width']}x{lat['height']} grayscale, "
              f"motion-periodicity + gate active):")
        print(f"   {lat['frames']:,} frames in {lat['totalMs']:.1f} ms  ->  "
              f"{lat['perFrameUs']:.2f} us/frame  ~  {lat['fps']:,} fps")
    print("=" * 72)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
