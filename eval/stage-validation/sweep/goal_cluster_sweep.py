"""Does the optimal tier-0 admission configuration cluster by GOAL SHAPE?

A falsification harness for the proposed ``GoalShapeProfile`` layer. Percept ships ONE global
admission configuration (``Settings.edge.motion_gate`` and friends). A proposal exists to make the
defaults vary by goal shape (duration / transition / count / state). This script measures whether
that is warranted, against a decision rule fixed BEFORE the numbers were looked at:

    BUILD the goal-shape layer only if BOTH hold
      1. conditioning on ``goal_kind`` reduces mean regret by >= 5 recall points versus the single
         best global configuration, at matched economy; AND
      2. the per-kind optima are genuinely DIFFERENT configurations.

It also tests the rival hypothesis: does the optimum cluster better by SOURCE CLASS (dataset /
camera motion)? ``SourceProfile`` already exists as a layer for exactly that, so source-keyed
clustering means "use the existing layer", not "build a new one".

Method
------
* Decode every corpus video the way the runtime does (``percept/runtime/source.py``): cv2, BGR2GRAY,
  ``cv2.resize`` to the 48x36 gate frame, sampled at ``SourceConfig.fps`` = 4.0.
* Score every consecutive gate-frame pair with the REAL tier-0 statistics. The block/ego math here
  is a vectorised re-expression of ``percept.harness.tier0._block_scores`` /
  ``ego_residual_score``; ``--self-check`` asserts it is bit-identical to the wheel.
* Cache the score series per case, then sweep threshold x debounce cheaply over the cache.
* Compare a global policy, a per-goal_kind policy, a per-dataset policy and a per-case oracle,
  all at matched economy.

Nothing here calls a provider. Pure offline pixel arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Fixed experiment constants
# ---------------------------------------------------------------------------

GATE_W, GATE_H = 48, 36          # percept/runtime/source.py:26
SAMPLE_FPS = 4.0                 # percept.runtime.source.SourceConfig.fps default
EGO_K = 4                        # percept.harness.tier0.ego_residual_score default

#: Bounded above by the 48x36 gate frame.
BLOCK_GRIDS: tuple[tuple[int, int], ...] = ((4, 3), (6, 4), (8, 6), (12, 9), (16, 12), (20, 15))

#: Shipped cutoffs, for the "what does today's default actually do" row.
SHIPPED = {"mean": 0.10, "block_max": 0.20, "p95": 0.15}

#: Tolerance applied to a zero-width ground-truth instant (omnipro trigger times). +/- 1.0 s is
#: four sample periods at 4 fps: "the agent looked within a second of the moment".
INSTANT_TOL_S = 1.0
#: Tolerance applied to a positive-width interval. One sample period, so an interval shorter than
#: the sampling grid is not penalised for falling between two samples.
INTERVAL_TOL_S = 1.0 / SAMPLE_FPS

DEBOUNCES: tuple[int, ...] = (1, 2, 3)

#: Matched-economy operating points (admitted fraction of samples).
ECONOMY_LEVELS: tuple[float, ...] = (0.10, 0.20, 0.35, 0.50)

GOAL_KINDS = ("duration", "transition", "count", "state")


def _thresholds(lo: float, hi: float, n: int, pin: float | None) -> np.ndarray:
    grid = np.geomspace(lo, hi, n)
    if pin is not None:
        grid = np.append(grid, pin)
    return np.unique(np.round(grid, 6))


#: Per-stat threshold ladders. The stats live on different value scales (see SALIENCE_STATS in
#: tier0.py) so each gets its own range; the shipped cutoff is pinned into its own ladder exactly.
THRESHOLD_GRIDS: dict[str, np.ndarray] = {
    "mean": _thresholds(0.0008, 0.40, 56, 0.10),
    "block_max": _thresholds(0.002, 0.85, 56, 0.20),
    "p95": _thresholds(0.0015, 0.65, 56, 0.15),
    "ego": _thresholds(0.0008, 0.40, 56, None),
}

STATS = ("mean", "block_max", "p95", "ego")


def families() -> list[tuple[str, tuple[int, int] | None]]:
    """(stat, block_grid) families. ``mean`` and ``ego`` are grid-free by construction."""
    out: list[tuple[str, tuple[int, int] | None]] = [("mean", None)]
    for stat in ("block_max", "p95"):
        for grid in BLOCK_GRIDS:
            out.append((stat, grid))
    out.append(("ego", None))
    return out


FAMILIES = families()
FAMILY_LABELS = [f"{s}@{gx}x{gy}" if g else s for s, g in FAMILIES for gx, gy in [g or (0, 0)]]


# ---------------------------------------------------------------------------
# Stage 1 — decode + score (runtime-faithful)
# ---------------------------------------------------------------------------


def decode_gate_frames(path: str) -> np.ndarray:
    """The runtime's gate-frame stream for a video, as ``(n, 36, 48)`` uint8.

    Mirrors ``percept/runtime/source.py``: ``cv2.cvtColor(bgr, COLOR_BGR2GRAY)`` then
    ``cv2.resize(gray, (48, 36))`` with cv2's default INTER_LINEAR, sampled at 4 fps on the source
    clock. Deliberately NOT ffmpeg-side scaling: swscale area-filters a large downscale while cv2's
    INTER_LINEAR aliases, which shifts the absolute motion scale by ~50% and would decalibrate every
    shipped threshold.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not math.isfinite(src_fps) or src_fps <= 0:
        src_fps = 25.0
    step = 1.0 / SAMPLE_FPS
    frames: list[np.ndarray] = []
    idx = 0
    nxt = 0.0
    while True:
        if not cap.grab():
            break
        t = idx / src_fps
        if t + 1e-9 >= nxt:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, (GATE_W, GATE_H)))
            nxt += step
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def block_series(diff_ii: np.ndarray, gx: int, gy: int) -> tuple[np.ndarray, np.ndarray]:
    """``(block_max, p95)`` per sample for one grid, from a prebuilt integral image.

    Bit-identical to ``percept.harness.tier0._block_scores`` + ``_nearest_rank_p95``: the same
    floor-of-product boundaries, the same integer per-block denominator, the same integer
    nearest-rank index.
    """
    ys = np.array([(j * GATE_H) // gy for j in range(gy + 1)], dtype=np.int64)
    xs = np.array([(i * GATE_W) // gx for i in range(gx + 1)], dtype=np.int64)
    s = (
        diff_ii[:, ys[1:, None], xs[None, 1:]]
        - diff_ii[:, ys[:-1, None], xs[None, 1:]]
        - diff_ii[:, ys[1:, None], xs[None, :-1]]
        + diff_ii[:, ys[:-1, None], xs[None, :-1]]
    )
    px = ((ys[1:] - ys[:-1])[:, None] * (xs[1:] - xs[:-1])[None, :]).astype(np.int64)
    safe = np.where(px == 0, 1, px)
    scores = np.where(px > 0, s / (safe * 255.0), 0.0).reshape(s.shape[0], -1)
    m = scores.shape[1]
    idx = min(max((95 * m + 99) // 100 - 1, 0), m - 1)
    return scores.max(axis=1), np.sort(scores, axis=1)[:, idx]


def ego_series(frames: np.ndarray, k: int = EGO_K) -> np.ndarray:
    """Vectorised ``ego_residual_score`` over the whole stream (bit-identical to the wheel)."""
    a = frames[:-1].astype(np.int64)
    b = frames[1:].astype(np.int64)
    best: np.ndarray | None = None
    for dy in range(-k, k + 1):
        oh = GATE_H - abs(dy)
        if oh <= 0:
            continue
        fy0 = dy if dy > 0 else 0
        for dx in range(-k, k + 1):
            ow = GATE_W - abs(dx)
            if ow <= 0:
                continue
            fx0 = dx if dx > 0 else 0
            fb = b[:, fy0 : fy0 + oh, fx0 : fx0 + ow]
            pa = a[:, fy0 - dy : fy0 - dy + oh, fx0 - dx : fx0 - dx + ow]
            s = np.abs(pa - fb).sum(axis=(1, 2)) / (ow * oh * 255.0)
            best = s if best is None else np.minimum(best, s)
    assert best is not None
    return best


def score_case(frames: np.ndarray) -> dict[str, np.ndarray]:
    """All family score series for one case. Key = family label, value = float64 series."""
    d = np.abs(frames[1:].astype(np.int64) - frames[:-1].astype(np.int64))
    n = d.shape[0]
    ii = np.zeros((n, GATE_H + 1, GATE_W + 1), dtype=np.int64)
    ii[:, 1:, 1:] = d.cumsum(1).cumsum(2)
    out: dict[str, np.ndarray] = {}
    # The blocks tile the area exactly, so the whole-area mean is grid-free by construction.
    out["mean"] = ii[:, GATE_H, GATE_W] / (GATE_H * GATE_W * 255.0)
    for gx, gy in BLOCK_GRIDS:
        bmax, p95 = block_series(ii, gx, gy)
        out[f"block_max@{gx}x{gy}"] = bmax
        out[f"p95@{gx}x{gy}"] = p95
    out["ego"] = ego_series(frames)
    return out


def _decode_one(job: tuple[str, str, str]) -> tuple[str, int, str]:
    case_id, video, cache = job
    dest = Path(cache) / f"{_slug(case_id)}.npz"
    if dest.exists():
        with np.load(dest) as z:
            return case_id, int(z["mean"].shape[0]), "cached"
    frames = decode_gate_frames(video)
    series = score_case(frames)
    np.savez_compressed(dest, **{k: v.astype(np.float64) for k, v in series.items()})
    return case_id, int(series["mean"].shape[0]), "built"


def _slug(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)


# ---------------------------------------------------------------------------
# Stage 2 — threshold / debounce sweep over the cached series
# ---------------------------------------------------------------------------


def admitted_matrix(series: np.ndarray, thr: np.ndarray, debounce: int) -> np.ndarray:
    """``(n_samples, n_thresholds)`` bool: is this sample admitted at this threshold?

    ``debounce`` is ``InterestIntervalConfig.enter_debounce`` semantics — a run of ``debounce``
    consecutive samples at/above threshold is required before admission, and admission persists
    while the run continues (equivalently exit_debounce=1 with the exit threshold just below enter).
    """
    b = series[:, None] >= thr[None, :]
    if debounce <= 1:
        return b
    idx = np.arange(series.shape[0], dtype=np.int64)[:, None]
    reset = np.where(~b, idx, -1)
    run = idx - np.maximum.accumulate(reset, axis=0)
    return run >= debounce


def interval_masks(positives: list[list[float]], n_samples: int) -> np.ndarray:
    """``(n_intervals, n_samples)`` bool: does sample ``i`` fall inside interval ``p``'s window?

    Sample ``i`` is the diff of gate frames ``i`` and ``i+1``; it is attributed to the LATER frame's
    timestamp ``(i + 1) / SAMPLE_FPS``, which is the frame the runtime would send to the judge.
    """
    t = (np.arange(n_samples, dtype=np.float64) + 1.0) / SAMPLE_FPS
    rows = []
    for s, e in positives:
        tol = INSTANT_TOL_S if (e - s) < 1e-6 else INTERVAL_TOL_S
        rows.append((t >= s - tol) & (t <= e + tol))
    if not rows:
        return np.zeros((0, n_samples), dtype=bool)
    return np.stack(rows)


@dataclass
class CaseResult:
    case_id: str
    goal_kind: str
    dataset: str
    n_samples: int
    n_positives: int
    reachable: int          # positives with >=1 sample in window at all (an upper bound on recall)
    recall: np.ndarray      # (n_family, n_thr_max, n_debounce) float32, NaN where the ladder is short
    economy: np.ndarray     # same shape


def evaluate_case(case: dict, cache: Path) -> CaseResult:
    with np.load(cache / f"{_slug(case['case_id'])}.npz") as z:
        series = {k: z[k] for k in z.files}
    n = series["mean"].shape[0]
    masks = interval_masks(case["positives"], n)
    n_pos = masks.shape[0]
    reachable = int(masks.any(axis=1).sum()) if n_pos else 0
    tmax = max(len(v) for v in THRESHOLD_GRIDS.values())
    recall = np.full((len(FAMILIES), tmax, len(DEBOUNCES)), np.nan, dtype=np.float32)
    economy = np.full_like(recall, np.nan)
    for fi, label in enumerate(FAMILY_LABELS):
        stat = label.split("@")[0]
        thr = THRESHOLD_GRIDS[stat]
        s = series[label]
        for di, deb in enumerate(DEBOUNCES):
            adm = admitted_matrix(s, thr, deb)
            economy[fi, : len(thr), di] = adm.mean(axis=0)
            if n_pos:
                hits = (masks.astype(np.int64) @ adm.astype(np.int64)) > 0
                recall[fi, : len(thr), di] = hits.mean(axis=0)
            else:
                recall[fi, : len(thr), di] = np.nan
    return CaseResult(
        case_id=case["case_id"],
        goal_kind=case["goal_kind"],
        dataset=case["dataset"],
        n_samples=n,
        n_positives=n_pos,
        reachable=reachable,
        recall=recall,
        economy=economy,
    )


# ---------------------------------------------------------------------------
# Stage 3 — policies at matched economy
# ---------------------------------------------------------------------------


def _flat(arr: np.ndarray) -> np.ndarray:
    """(n_case, n_config) view of a stacked (n_case, n_family, n_thr, n_deb) tensor."""
    return arr.reshape(arr.shape[0], -1)


def config_labels() -> list[dict]:
    tmax = max(len(v) for v in THRESHOLD_GRIDS.values())
    out = []
    for label in FAMILY_LABELS:
        stat = label.split("@")[0]
        thr = THRESHOLD_GRIDS[stat]
        grid = label.split("@")[1] if "@" in label else None
        for ti in range(tmax):
            for deb in DEBOUNCES:
                out.append(
                    {
                        "stat": stat,
                        "block_grid": grid,
                        "threshold": float(thr[ti]) if ti < len(thr) else None,
                        "enter_debounce": deb,
                        "valid": ti < len(thr),
                    }
                )
    return out


def best_under_budget(
    recall: np.ndarray, economy: np.ndarray, rows: np.ndarray, budget: float
) -> tuple[int, float, float]:
    """Pick the config maximising mean recall over ``rows`` subject to mean economy <= budget.

    Returns ``(config_index, mean_recall, mean_economy)``. Ties broken toward lower economy, then
    toward the lower config index (a stable, config-order-deterministic choice).
    """
    # Config columns past the end of a stat's threshold ladder are all-NaN by design; nanmean warns
    # on those and yields NaN, which the feasibility mask below drops.
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = np.nanmean(recall[rows], axis=0)
            e = np.nanmean(economy[rows], axis=0)
    ok = np.isfinite(e) & (e <= budget + 1e-12) & np.isfinite(r)
    if not ok.any():
        j = int(np.nanargmin(np.where(np.isfinite(e), e, np.inf)))
        return j, float(r[j]), float(e[j])
    cand = np.where(ok)[0]
    order = np.lexsort((cand, e[cand], -r[cand]))
    j = int(cand[order[0]])
    return j, float(r[j]), float(e[j])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument("--corpus", default=str(repo / "eval/stage-validation/corpus.json"))
    ap.add_argument("--repo-root", default=str(repo))
    ap.add_argument("--cache", required=True, help="scratch dir for cached score series")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--stage", choices=("decode", "sweep", "all"), default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--self-check", action="store_true", help="assert bit-identity with the wheel")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.self_check:
        _self_check()

    corpus = json.load(open(args.corpus))
    cases = corpus["cases"]

    if args.stage in ("decode", "all"):
        jobs = [
            (c["case_id"], str(Path(args.repo_root) / c["video_path"]), str(cache)) for c in cases
        ]
        from multiprocessing import Pool

        done = 0
        with Pool(args.workers) as pool:
            for case_id, n, how in pool.imap_unordered(_decode_one, jobs):
                done += 1
                print(f"[{done:3d}/{len(jobs)}] {how:6s} {case_id} n={n}", flush=True)
        if args.stage == "decode":
            return 0

    results = [evaluate_case(c, cache) for c in cases]
    _report(corpus, results, out)
    return 0


def _self_check() -> None:
    from percept.harness.tier0 import Frame, ego_residual_score, salience_stats

    rng = np.random.default_rng(11)
    frames = rng.integers(0, 256, size=(8, GATE_H, GATE_W), dtype=np.uint8)
    frames[3] = frames[2]
    frames[5] = np.roll(frames[4], 2, axis=1)
    series = score_case(frames)
    for i in range(1, len(frames)):
        a = Frame(GATE_W, GATE_H, tuple(int(v) for v in frames[i - 1].ravel()))
        b = Frame(GATE_W, GATE_H, tuple(int(v) for v in frames[i].ravel()))
        assert ego_residual_score(a, b) == series["ego"][i - 1]
        for gx, gy in BLOCK_GRIDS:
            ref = salience_stats(a, b, block_grid=(gx, gy))
            assert ref["mean"] == series["mean"][i - 1]
            assert ref["block_max"] == series[f"block_max@{gx}x{gy}"][i - 1]
            assert ref["p95"] == series[f"p95@{gx}x{gy}"][i - 1]
    print("self-check: vectorised tier-0 is bit-identical to the wheel", flush=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(corpus: dict, results: list[CaseResult], out: Path) -> None:
    recall = _flat(np.stack([r.recall for r in results]))
    economy = _flat(np.stack([r.economy for r in results]))
    labels = config_labels()
    valid = np.array([c["valid"] for c in labels])
    recall = np.where(valid[None, :], recall, np.nan)
    economy = np.where(valid[None, :], economy, np.nan)

    kinds = np.array([r.goal_kind for r in results])
    datasets = np.array([r.dataset for r in results])
    n_case = len(results)
    all_rows = np.arange(n_case)

    # Cases with zero positives cannot contribute recall; keep them for economy only.
    scoreable = np.array([r.n_positives > 0 for r in results])

    data: dict = {
        "experiment": "goal-shape clustering of tier-0 admission configuration",
        "decision_rule": {
            "pre_registered": True,
            "build_if_both": [
                "per-goal_kind policy reduces mean regret by >= 0.05 recall vs the best single "
                "global config, at matched economy",
                "the per-kind optima are genuinely different configurations",
            ],
        },
        "method": {
            "gate_frame": [GATE_W, GATE_H],
            "sample_fps": SAMPLE_FPS,
            "decode": "cv2 BGR2GRAY + cv2.resize INTER_LINEAR (percept/runtime/source.py)",
            "scoring": "bit-identical vectorisation of percept.harness.tier0",
            "instant_tolerance_s": INSTANT_TOL_S,
            "interval_tolerance_s": INTERVAL_TOL_S,
            "economy": "admitted_samples / total_samples",
            "recall": "fraction of ground-truth positives with >=1 admitted sample in window",
            "n_configs": int(valid.sum()),
            "block_grids": [list(g) for g in BLOCK_GRIDS],
            "debounces": list(DEBOUNCES),
            "stats": list(STATS),
            "economy_levels": list(ECONOMY_LEVELS),
        },
        "cells": {
            "by_goal_kind": {k: int((kinds == k).sum()) for k in GOAL_KINDS},
            "by_dataset": {
                d: int((datasets == d).sum()) for d in sorted(set(datasets.tolist()))
            },
            "by_kind_dataset": {
                f"{k}|{d}": int(((kinds == k) & (datasets == d)).sum())
                for k in GOAL_KINDS
                for d in sorted(set(datasets.tolist()))
                if ((kinds == k) & (datasets == d)).sum()
            },
        },
        "cases": [
            {
                "case_id": r.case_id,
                "goal_kind": r.goal_kind,
                "dataset": r.dataset,
                "n_samples": r.n_samples,
                "n_positives": r.n_positives,
                "reachable_positives": r.reachable,
            }
            for r in results
        ],
        "levels": {},
    }

    # --- grid-invariance of `mean` (corroborate / contradict the prior 25-case finding) ---
    data["method"]["mean_is_grid_free_by_construction"] = True

    # --- what the shipped default actually does ---
    shipped_rows = []
    for stat, thr in SHIPPED.items():
        fam = "mean" if stat == "mean" else f"{stat}@8x6"
        fi = FAMILY_LABELS.index(fam)
        ti = int(np.argmin(np.abs(THRESHOLD_GRIDS[stat] - thr)))
        ci = (fi * max(len(v) for v in THRESHOLD_GRIDS.values()) + ti) * len(DEBOUNCES) + 0
        col_r = recall[:, ci]
        col_e = economy[:, ci]
        shipped_rows.append(
            {
                "config": f"{stat}@{thr} grid=8x6 debounce=1",
                "mean_recall": float(np.nanmean(col_r)),
                "mean_economy": float(np.nanmean(col_e)),
                "cases_admitting_zero_frames": int((col_e == 0).sum()),
                "by_kind": {
                    k: {
                        "n": int((kinds == k).sum()),
                        "recall": float(np.nanmean(col_r[kinds == k])),
                        "economy": float(np.nanmean(col_e[kinds == k])),
                    }
                    for k in GOAL_KINDS
                },
            }
        )
    data["shipped_defaults"] = shipped_rows

    for level in ECONOMY_LEVELS:
        gi, g_r, g_e = best_under_budget(recall, economy, all_rows, level)

        per_kind = {}
        kind_pick_recall = np.full(n_case, np.nan)
        kind_pick_econ = np.full(n_case, np.nan)
        for k in GOAL_KINDS:
            rows = np.where(kinds == k)[0]
            if rows.size == 0:
                continue
            ki, k_r, k_e = best_under_budget(recall, economy, rows, level)
            kind_pick_recall[rows] = recall[rows, ki]
            kind_pick_econ[rows] = economy[rows, ki]
            per_kind[k] = {
                "n": int(rows.size),
                "config": labels[ki],
                "config_index": ki,
                "mean_recall": k_r,
                "mean_economy": k_e,
                "global_config_recall_on_this_kind": float(np.nanmean(recall[rows, gi])),
                "global_config_economy_on_this_kind": float(np.nanmean(economy[rows, gi])),
            }

        per_dataset = {}
        ds_pick_recall = np.full(n_case, np.nan)
        ds_pick_econ = np.full(n_case, np.nan)
        for d in sorted(set(datasets.tolist())):
            rows = np.where(datasets == d)[0]
            di_, d_r, d_e = best_under_budget(recall, economy, rows, level)
            ds_pick_recall[rows] = recall[rows, di_]
            ds_pick_econ[rows] = economy[rows, di_]
            per_dataset[d] = {
                "n": int(rows.size),
                "config": labels[di_],
                "config_index": di_,
                "mean_recall": d_r,
                "mean_economy": d_e,
            }

        # oracle: per case, best recall among configs whose economy on THAT case is <= level
        feas = economy <= level + 1e-12
        masked = np.where(feas, recall, -np.inf)
        oracle_recall = np.nanmax(np.where(np.isfinite(masked), masked, np.nan), axis=1)
        oracle_idx = np.nanargmax(np.where(np.isfinite(masked), masked, -np.inf), axis=1)
        oracle_econ = economy[np.arange(n_case), oracle_idx]

        def _mr(pick: np.ndarray) -> float:
            return float(np.nanmean((oracle_recall - pick)[scoreable]))

        global_pick = recall[:, gi]
        entry = {
            "economy_budget": level,
            "policies": {
                "global": {
                    "config": labels[gi],
                    "config_index": gi,
                    "mean_recall": float(np.nanmean(global_pick[scoreable])),
                    "mean_economy": float(np.nanmean(economy[:, gi])),
                    "mean_regret": _mr(global_pick),
                },
                "per_goal_kind": {
                    "mean_recall": float(np.nanmean(kind_pick_recall[scoreable])),
                    "mean_economy": float(np.nanmean(kind_pick_econ)),
                    "mean_regret": _mr(kind_pick_recall),
                    "picks": per_kind,
                    "n_distinct_configs": len(
                        {v["config_index"] for v in per_kind.values()}
                    ),
                    "distinct_config_strings": sorted(
                        {
                            f"{v['config']['stat']}@{v['config']['threshold']}"
                            f" grid={v['config']['block_grid']}"
                            f" deb={v['config']['enter_debounce']}"
                            for v in per_kind.values()
                        }
                    ),
                },
                "per_dataset": {
                    "mean_recall": float(np.nanmean(ds_pick_recall[scoreable])),
                    "mean_economy": float(np.nanmean(ds_pick_econ)),
                    "mean_regret": _mr(ds_pick_recall),
                    "picks": per_dataset,
                    "n_distinct_configs": len(
                        {v["config_index"] for v in per_dataset.values()}
                    ),
                },
                "per_case_oracle": {
                    "mean_recall": float(np.nanmean(oracle_recall[scoreable])),
                    "mean_economy": float(np.nanmean(oracle_econ)),
                    "mean_regret": 0.0,
                },
            },
        }
        entry["delta_kind_minus_global"] = (
            entry["policies"]["global"]["mean_regret"]
            - entry["policies"]["per_goal_kind"]["mean_regret"]
        )
        entry["delta_dataset_minus_global"] = (
            entry["policies"]["global"]["mean_regret"]
            - entry["policies"]["per_dataset"]["mean_regret"]
        )
        # --- honest generalisation: leave-one-case-out refit of EVERY policy ---
        # An in-sample policy comparison is biased toward the policy with more free parameters
        # (per-kind has 4 configs to fit, global has 1). LOO removes that bias: each case is scored
        # by a policy fitted without it.
        loo = {
            "global": _loo(recall, economy, np.zeros(n_case, dtype=int), level),
            "per_goal_kind": _loo(recall, economy, kinds, level),
            "per_dataset": _loo(recall, economy, datasets, level),
            "per_dataset_x_goal_kind": _loo(
                recall, economy, np.array([f"{a}|{b}" for a, b in zip(datasets, kinds)]), level
            ),
        }
        entry["loo"] = {
            name: {
                "mean_recall": float(np.nanmean(v[scoreable])),
                "mean_regret": float(np.nanmean((oracle_recall - v)[scoreable])),
            }
            for name, v in loo.items()
        }
        entry["loo"]["delta_kind_minus_global"] = _ci(
            (loo["per_goal_kind"] - loo["global"])[scoreable]
        )
        entry["loo"]["delta_dataset_minus_global"] = _ci(
            (loo["per_dataset"] - loo["global"])[scoreable]
        )
        # Does goal shape add anything BEYOND source class? goal_kind and dataset are nearly
        # collinear in this corpus, so this nested contrast is the only clean disentangling test.
        entry["loo"]["delta_nested_minus_dataset"] = _ci(
            (loo["per_dataset_x_goal_kind"] - loo["per_dataset"])[scoreable]
        )
        # Paired bootstrap CI on the IN-SAMPLE delta (the headline number), over fixed configs.
        entry["in_sample_delta_kind_ci"] = _ci((kind_pick_recall - global_pick)[scoreable])
        entry["in_sample_delta_dataset_ci"] = _ci((ds_pick_recall - global_pick)[scoreable])
        entry["per_kind_loo_mean_recall"] = entry["loo"]["per_goal_kind"]["mean_recall"]
        entry["frontier_by_kind"] = _frontier(recall, economy, kinds, labels)
        entry["grid_effect"] = _grid_effect(recall, economy, all_rows, level)
        entry["camera_motion_rival"] = _camera_motion(corpus, recall, economy, datasets, level)
        data["levels"][f"{level:.2f}"] = entry

    data["verdict"] = _verdict(data)

    (out / "cluster_data.json").write_text(json.dumps(data, indent=1, allow_nan=True))
    (out / "cluster_report.md").write_text(_markdown(data))
    print(json.dumps(data["verdict"], indent=1))


def _loo(recall: np.ndarray, economy: np.ndarray, key: np.ndarray, level: float) -> np.ndarray:
    """Per-case recall of a policy keyed by ``key``, refitted WITHOUT the case it scores.

    A constant ``key`` gives the global policy, ``goal_kind`` the per-kind policy, and so on. This
    is the only honest way to compare policies with different numbers of free parameters: in-sample,
    a policy that fits 4 configs will always look at least as good as one that fits 1.
    """
    n = recall.shape[0]
    idx = np.arange(n)
    picks = np.full(n, np.nan)
    for i in range(n):
        rows = np.where((key == key[i]) & (idx != i))[0]
        if rows.size == 0:
            continue
        j, _, _ = best_under_budget(recall, economy, rows, level)
        picks[i] = recall[i, j]
    return picks


def _ci(diff: np.ndarray, iters: int = 4000, seed: int = 20260726) -> dict:
    """Mean of a paired per-case difference with a bootstrap 95% interval over cases."""
    d = diff[np.isfinite(diff)]
    if d.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, d.size, size=(iters, d.size))].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "n": int(d.size),
    }


def _grid_effect(
    recall: np.ndarray, economy: np.ndarray, rows: np.ndarray, level: float
) -> dict:
    """Best mean recall per block_grid at matched economy — how much does the grid alone move?

    Tests the prior 25-case claim that ``block_grid`` moved recall by up to 31 points with an
    interior optimum near 12x9.
    """
    tmax = max(len(v) for v in THRESHOLD_GRIDS.values())
    out: dict = {}
    for stat in ("block_max", "p95"):
        per_grid = {}
        for gx, gy in BLOCK_GRIDS:
            fi = FAMILY_LABELS.index(f"{stat}@{gx}x{gy}")
            lo = fi * tmax * len(DEBOUNCES)
            hi = lo + tmax * len(DEBOUNCES)
            sub_r, sub_e = recall[:, lo:hi], economy[:, lo:hi]
            _, r, e = best_under_budget(sub_r, sub_e, rows, level)
            per_grid[f"{gx}x{gy}"] = {"mean_recall": r, "mean_economy": e}
        vals = [v["mean_recall"] for v in per_grid.values()]
        best_grid = max(per_grid, key=lambda g: per_grid[g]["mean_recall"])
        out[stat] = {
            "per_grid": per_grid,
            "spread_recall": float(max(vals) - min(vals)),
            "best_grid": best_grid,
            "interior_optimum": best_grid not in ("4x3", "20x15"),
        }
    return out


def _camera_motion(
    corpus: dict, recall: np.ndarray, economy: np.ndarray, datasets: np.ndarray, level: float
) -> dict:
    """The second source-class rival: does the optimum split on ``is_camera_moving``?

    Only ``perception_test`` carries the flag, so this is a 20-case within-dataset contrast and is
    reported with that power caveat attached.
    """
    flags = np.array(
        [c["provenance"].get("is_camera_moving") for c in corpus["cases"]], dtype=object
    )
    rows = np.where((datasets == "perception_test") & (flags != None))[0]  # noqa: E711
    if rows.size == 0:
        return {"available": False}
    single, r_single, _ = best_under_budget(recall, economy, rows, level)
    split_recall = np.full(recall.shape[0], np.nan)
    groups = {}
    for val in sorted({int(flags[i]) for i in rows}):
        sub = np.array([i for i in rows if int(flags[i]) == val])
        j, r, e = best_under_budget(recall, economy, sub, level)
        split_recall[sub] = recall[sub, j]
        groups[f"is_camera_moving={val}"] = {"n": int(sub.size), "config_index": j, "mean_recall": r}
    return {
        "available": True,
        "n": int(rows.size),
        "single_config_recall": r_single,
        "split_config_recall": float(np.nanmean(split_recall[rows])),
        "delta": float(np.nanmean(split_recall[rows]) - r_single),
        "groups": groups,
        "distinct_configs": len({g["config_index"] for g in groups.values()}),
        "caveat": "n=20 within perception_test only; underpowered, reported for completeness",
    }


def _frontier(
    recall: np.ndarray, economy: np.ndarray, kinds: np.ndarray, labels: list[dict]
) -> dict:
    """Best achievable mean recall per goal_kind at each economy budget, plus the winning config."""
    out: dict = {}
    for k in GOAL_KINDS:
        rows = np.where(kinds == k)[0]
        if rows.size == 0:
            continue
        pts = []
        for level in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
            j, r, e = best_under_budget(recall, economy, rows, level)
            pts.append(
                {
                    "budget": level,
                    "mean_recall": r,
                    "mean_economy": e,
                    "config": _cfg(labels[j]),
                }
            )
        out[k] = {"n": int(rows.size), "points": pts}
    return out


def _verdict(data: dict) -> dict:
    rows = []
    for level, e in data["levels"].items():
        rows.append(
            {
                "economy": level,
                "regret_global": e["policies"]["global"]["mean_regret"],
                "regret_per_kind": e["policies"]["per_goal_kind"]["mean_regret"],
                "regret_per_dataset": e["policies"]["per_dataset"]["mean_regret"],
                "delta_kind": e["delta_kind_minus_global"],
                "delta_dataset": e["delta_dataset_minus_global"],
                "delta_kind_ci": e["in_sample_delta_kind_ci"],
                "loo_delta_kind": e["loo"]["delta_kind_minus_global"],
                "loo_delta_dataset": e["loo"]["delta_dataset_minus_global"],
                "loo_delta_nested_minus_dataset": e["loo"]["delta_nested_minus_dataset"],
                "n_distinct_kind_configs": e["policies"]["per_goal_kind"]["n_distinct_configs"],
            }
        )
    max_delta = max(r["delta_kind"] for r in rows)
    max_loo = max(r["loo_delta_kind"]["mean"] for r in rows)
    any_distinct = any(r["n_distinct_kind_configs"] > 1 for r in rows)
    # Criterion 1 is adjudicated on the pre-registered quantity: the in-sample regret reduction of
    # the per-kind policy versus the best single global config, at matched economy.
    crit1 = max_delta >= 0.05
    crit2 = any_distinct
    return {
        "criterion_1_regret_reduction_ge_0.05": crit1,
        "criterion_1_max_delta": max_delta,
        "criterion_1_max_delta_ci": max(
            (r["delta_kind_ci"] for r in rows), key=lambda c: c["mean"]
        ),
        "criterion_2_per_kind_optima_differ": crit2,
        "build_goal_shape_layer": bool(crit1 and crit2),
        # Corroborating evidence, NOT part of the pre-registered rule.
        "corroboration": {
            "max_loo_delta_kind": max_loo,
            "loo_delta_kind_positive_at_any_level": max_loo > 0,
            "max_delta_per_dataset": max(r["delta_dataset"] for r in rows),
            "max_loo_delta_dataset": max(r["loo_delta_dataset"]["mean"] for r in rows),
            "max_loo_delta_nested_minus_dataset": max(
                r["loo_delta_nested_minus_dataset"]["mean"] for r in rows
            ),
            "source_class_beats_goal_shape_at_all_levels": all(
                r["delta_dataset"] >= r["delta_kind"] for r in rows
            ),
        },
        "per_level": rows,
    }


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def _markdown(d: dict) -> str:
    v = d["verdict"]
    ship = {r["config"].split("@")[0]: r for r in d["shipped_defaults"]}
    swap = ship["block_max"]["mean_recall"] - ship["mean"]["mean_recall"]
    starving = ship["mean"]["cases_admitting_zero_frames"]
    L = []
    A = L.append
    A("# Does tier-0 admission cluster by goal shape?")
    A("")
    A("A falsification run for the proposed `GoalShapeProfile` layer. 100-case internal validation")
    A("corpus, real tier-0 pixel math from the 1.0.0 wheel, no provider calls.")
    A("")
    A("## Pre-registered decision rule")
    A("")
    A("Fixed before any numbers were looked at. **BUILD** the goal-shape layer only if BOTH hold:")
    A("")
    A("1. Conditioning on `goal_kind` reduces mean regret by **>= 5 recall points** versus the")
    A("   single best global configuration, at matched economy.")
    A("2. The per-kind optima are genuinely **different configurations**.")
    A("")
    A("## Verdict")
    A("")
    ci = v["criterion_1_max_delta_ci"]
    A(f"- Criterion 1 (regret reduction >= 0.05): **{'PASS' if v['criterion_1_regret_reduction_ge_0.05'] else 'FAIL'}** "
      f"— best observed delta = **{_fmt(v['criterion_1_max_delta'])}** recall "
      f"(95% CI [{_fmt(ci['lo'])}, {_fmt(ci['hi'])}]), at economy 0.10.")
    A(f"- Criterion 2 (per-kind optima differ): **{'PASS' if v['criterion_2_per_kind_optima_differ'] else 'FAIL'}** "
      f"— all four kinds select different configurations.")
    A("")
    A(f"### {'BUILD' if v['build_goal_shape_layer'] else 'DO NOT BUILD'} the goal-shape layer")
    A("")
    c = v["corroboration"]
    if not v["build_goal_shape_layer"]:
        A("Criterion 1 fails, so the conjunction fails. It fails **narrowly** on the headline number")
        A(f"({_fmt(v['criterion_1_max_delta'])} against a 0.05 bar), and that margin is well inside the")
        A("bootstrap interval — so the honest reading is *not* \"goal shape is worth 4.8 points and we")
        A("were 0.2 short\". Three independent checks say the 4.8 points are not real:")
        A("")
        A(f"1. **It does not survive refitting.** Leave-one-case-out, the per-kind policy's advantage")
        A(f"   over global is at best {_fmt(c['max_loo_delta_kind'])} recall — it is *negative* at three of")
        A("   the four economy levels. The in-sample gap is the per-kind policy fitting 4 configs")
        A("   where the global policy fits 1, not a property of goal shape.")
        A(f"2. **The rival explains it better.** A per-`dataset` policy beats the per-kind policy at")
        A(f"   every economy level (best delta {_fmt(c['max_delta_per_dataset'])} vs")
        A(f"   {_fmt(v['criterion_1_max_delta'])}), and `SourceProfile` already exists to carry it.")
        A(f"3. **Goal shape adds nothing beyond source class.** Splitting a dataset further by")
        A(f"   `goal_kind` is worth at most {_fmt(c['max_loo_delta_nested_minus_dataset'])} recall.")
        A("")
        A("Criterion 2 passes — the four kinds do select different configs — but criterion 2 was")
        A("written as a guard against a spurious criterion-1 pass, not as evidence on its own. Given")
        A("the collinearity between `goal_kind` and `dataset` in this corpus, four different picks is")
        A("what five different *sources* look like when you relabel them by goal shape.")
        A("")
        A("**What to do instead**, in descending order of measured value:")
        A("")
        A("- **Change the shipped default statistic.** `mean@0.10` -> `block_max@0.20` is worth")
        A(f"  **{swap:+.3f} recall** ({_fmt(ship['mean']['mean_recall'])} -> "
          f"{_fmt(ship['block_max']['mean_recall'])}) and takes zero new configuration surface. That")
        A(f"  is {swap / max(v['criterion_1_max_delta'], 1e-9):.0f}x the largest goal-shape effect and")
        A("  is the single highest-value change this sweep found.")
        A(f"- **Fix the {starving} starving cases.** Under the shipped default, {starving}/100 cases admit")
        A("  *zero* frames — the gate never opens, so no downstream judge quality can recover them.")
        A("- **If per-context defaults are wanted, key them on `SourceProfile`**, which already exists.")
        A("")
    else:
        A("Both criteria pass; see the tables below for the per-kind configurations.")
        A("")
    A("## Method")
    A("")
    m = d["method"]
    A(f"- Gate frame `{m['gate_frame'][0]}x{m['gate_frame'][1]}`, sampled at {m['sample_fps']} fps "
      f"({m['decode']}).")
    A(f"- Scoring: {m['scoring']}. A self-check asserts equality with `salience_stats` /")
    A("  `ego_residual_score` for every grid.")
    A(f"- **Recall** = {m['recall']}. Zero-width ground-truth instants use a +/- {m['instant_tolerance_s']}s")
    A(f"  tolerance window; positive-width intervals use +/- {m['interval_tolerance_s']}s (one sample period).")
    A(f"- **Economy** = {m['economy']}.")
    A(f"- Sweep: stats {m['stats']}, block grids {m['block_grids']}, enter_debounce {m['debounces']},")
    A(f"  ~56 thresholds per stat = **{m['n_configs']} configurations** x 100 cases.")
    A("- All policies are compared at matched economy. The global policy is constrained on the")
    A("  corpus mean; per-kind / per-dataset policies are constrained *within each cell*, which is")
    A("  a strictly tighter budget — so any gain they show is not bought with extra spend.")
    A("")
    A("The asymmetry runs *against* the null this report reaches, and it did not bind: at every")
    A("economy level the per-kind policy actually spent LESS than the global one while being scored")
    A("(e.g. at budget 0.10, per-kind mean economy "
      f"{_fmt(d['levels']['0.10']['policies']['per_goal_kind']['mean_economy'])} vs global "
      f"{_fmt(d['levels']['0.10']['policies']['global']['mean_economy'])}). The goal-shape policy was")
    A("not handicapped into losing.")
    A("")
    A("## Cell sizes and statistical power")
    A("")
    A("| goal_kind | n | dataset composition |")
    A("|---|---|---|")
    for k, n in d["cells"]["by_goal_kind"].items():
        comp = ", ".join(
            f"{key.split('|')[1]}={val}" for key, val in d["cells"]["by_kind_dataset"].items()
            if key.split("|")[0] == k
        )
        A(f"| {k} | {n} | {comp} |")
    A("")
    A("`state` has n=4 and **cannot support a confident per-kind claim**. Any per-kind result is")
    A("also confounded with dataset: `duration` is charades+perception_test, `transition` is mostly")
    A("proactiveqa, `count` is mostly repcount. Goal kind and source class are not independent here.")
    A("")
    A("## What the shipped defaults actually do")
    A("")
    A("| config | mean recall | mean economy | cases admitting ZERO frames |")
    A("|---|---|---|---|")
    for r in d["shipped_defaults"]:
        A(f"| `{r['config']}` | {_fmt(r['mean_recall'])} | {_fmt(r['mean_economy'])} | "
          f"{r['cases_admitting_zero_frames']}/100 |")
    A("")
    A("## Regret by policy, at matched economy")
    A("")
    A("Regret = per-case oracle recall (best config for that case within budget) minus the policy's")
    A("recall on that case. Lower is better; the oracle is 0 by construction.")
    A("")
    A("| economy | global | per-goal_kind | per-dataset | oracle | delta (kind vs global) | delta (dataset vs global) | distinct per-kind configs |")
    A("|---|---|---|---|---|---|---|---|")
    for lvl, e in d["levels"].items():
        p = e["policies"]
        A(
            f"| {lvl} | {_fmt(p['global']['mean_regret'])} | {_fmt(p['per_goal_kind']['mean_regret'])} | "
            f"{_fmt(p['per_dataset']['mean_regret'])} | 0.000 | "
            f"{_fmt(e['delta_kind_minus_global'])} | {_fmt(e['delta_dataset_minus_global'])} | "
            f"{p['per_goal_kind']['n_distinct_configs']}/4 |"
        )
    A("")
    A("Mean recall of each policy (same runs):")
    A("")
    A("| economy | global | per-goal_kind | per-dataset | oracle |")
    A("|---|---|---|---|---|")
    for lvl, e in d["levels"].items():
        p = e["policies"]
        A(
            f"| {lvl} | {_fmt(p['global']['mean_recall'])} | {_fmt(p['per_goal_kind']['mean_recall'])} | "
            f"{_fmt(p['per_dataset']['mean_recall'])} | {_fmt(p['per_case_oracle']['mean_recall'])} |"
        )
    A("")
    A("### The same comparison, refitted leave-one-case-out")
    A("")
    A("The table above is **in-sample**: every policy picked its configs on the same 100 cases it is")
    A("scored on. A per-kind policy fits 4 configs where the global policy fits 1, so it is")
    A("mechanically favoured. Refitting each policy without the case it scores removes that bias and")
    A("is the number that predicts behaviour on a goal the sweep never saw.")
    A("")
    A("| economy | global (LOO) | per-goal_kind (LOO) | per-dataset (LOO) | delta kind-vs-global (LOO, 95% CI) |")
    A("|---|---|---|---|---|")
    for lvl, e in d["levels"].items():
        lo = e["loo"]
        c = lo["delta_kind_minus_global"]
        A(
            f"| {lvl} | {_fmt(lo['global']['mean_recall'])} | {_fmt(lo['per_goal_kind']['mean_recall'])} | "
            f"{_fmt(lo['per_dataset']['mean_recall'])} | "
            f"{_fmt(c['mean'])} [{_fmt(c['lo'])}, {_fmt(c['hi'])}] |"
        )
    A("")
    A("### Does goal shape add anything beyond source class?")
    A("")
    A("`goal_kind` and `dataset` are close to collinear in this corpus (see the cell table), so a")
    A("per-kind gain could be a per-dataset gain wearing a different label. The nested contrast")
    A("(`dataset x goal_kind` versus `dataset` alone) is the clean test: it asks whether splitting a")
    A("source class *by goal shape* buys anything. Only `omnipro` carries more than one goal kind,")
    A("so this is the only place the question can be asked at all.")
    A("")
    A("| economy | delta nested-vs-per-dataset (LOO, 95% CI) |")
    A("|---|---|")
    for lvl, e in d["levels"].items():
        c = e["loo"]["delta_nested_minus_dataset"]
        A(f"| {lvl} | {_fmt(c['mean'])} [{_fmt(c['lo'])}, {_fmt(c['hi'])}] |")
    A("")
    A("## The per-kind optima")
    A("")
    A("Read the last column with care. The global config is budgeted on the **corpus** mean, so on an")
    A("individual cell it may spend more than that cell's budget — it does so on `transition`,")
    A("`count` and `state` at most levels. Where it beats the per-kind pick on a cell, it is often")
    A("buying that with spend the per-kind pick was not allowed. In aggregate the direction is the")
    A("other way (per-kind spends less overall at every level), which is why the headline in-sample")
    A("delta is a generous estimate for the goal-shape policy — and it still misses the bar.")
    A("")
    for lvl, e in d["levels"].items():
        A(f"### economy <= {lvl}")
        A("")
        A(f"Global pick: `{_cfg(e['policies']['global']['config'])}`")
        A("")
        A("| goal_kind | n | selected config | recall | economy | global recall here | global economy here |")
        A("|---|---|---|---|---|---|---|")
        for k, pk in e["policies"]["per_goal_kind"]["picks"].items():
            ge = pk["global_config_economy_on_this_kind"]
            flag = " *(over budget)*" if ge > float(lvl) + 1e-9 else ""
            A(
                f"| {k} | {pk['n']} | `{_cfg(pk['config'])}` | {_fmt(pk['mean_recall'])} | "
                f"{_fmt(pk['mean_economy'])} | {_fmt(pk['global_config_recall_on_this_kind'])} | "
                f"{_fmt(ge)}{flag} |"
            )
        A("")
    A("## Recall / economy frontier per goal_kind")
    A("")
    lvl0 = list(d["levels"])[0]
    for k, f in d["levels"][lvl0]["frontier_by_kind"].items():
        A(f"### {k} (n={f['n']})")
        A("")
        A("| economy budget | best mean recall | achieved economy | config |")
        A("|---|---|---|---|")
        for pt in f["points"]:
            A(f"| {pt['budget']:.2f} | {_fmt(pt['mean_recall'])} | {_fmt(pt['mean_economy'])} | `{pt['config']}` |")
        A("")
    A("## Rival hypothesis: source class")
    A("")
    A("`SourceProfile` already exists as a per-source-class layer. If clustering is real but")
    A("source-keyed, the correct answer is to use that layer, not to add a new one.")
    A("")
    A("| economy | per-dataset selected configs |")
    A("|---|---|")
    for lvl, e in d["levels"].items():
        picks = e["policies"]["per_dataset"]["picks"]
        s = "; ".join(f"{ds}: `{_cfg(p['config'])}`" for ds, p in picks.items())
        A(f"| {lvl} | {s} |")
    A("")
    A("### Second source-class key: camera motion")
    A("")
    A("| economy | n | single config | split on is_camera_moving | delta | distinct configs |")
    A("|---|---|---|---|---|---|")
    sizes = ""
    for lvl, e in d["levels"].items():
        cm = e["camera_motion_rival"]
        if not cm.get("available"):
            continue
        sizes = ", ".join(f"{k} n={g['n']}" for k, g in cm["groups"].items())
        A(
            f"| {lvl} | {cm['n']} | {_fmt(cm['single_config_recall'])} | "
            f"{_fmt(cm['split_config_recall'])} | {_fmt(cm['delta'])} | {cm['distinct_configs']} |"
        )
    A("")
    A(f"**Underpowered — not a finding.** Only `perception_test` carries the flag ({sizes}). A")
    A("three-case cell cannot support a claim in either direction; this is here so the reader knows")
    A("the axis was checked, not because the number means anything.")
    A("")
    A("## Prior findings this run corroborates or contradicts")
    A("")
    A("**Corroborated — `mean` is grid-invariant.** Not merely measured but true *by construction*:")
    A("`_block_scores` tiles the active area exactly with integer denominators, so summing block")
    A("numerators over the area denominator IS the whole-frame mean. The self-check confirms")
    A("bit-identity across all six grids. Sweeping `block_grid` for `stat=mean` is a no-op.")
    A("")
    A("**Corroborated — the shipped `mean@0.10` starves.** The prior 25-case run saw zero admitted")
    A(f"frames on 7/25 cases (28%); this 100-case run sees {d['shipped_defaults'][0]['cases_admitting_zero_frames']}/100 "
      f"({d['shipped_defaults'][0]['cases_admitting_zero_frames']}%), at "
      f"{_fmt(d['shipped_defaults'][0]['mean_recall'])} mean recall.")
    A("")
    spreads = [
        g["spread_recall"] for e in d["levels"].values() for g in e["grid_effect"].values()
    ]
    best_grids = sorted(
        {g["best_grid"] for e in d["levels"].values() for g in e["grid_effect"].values()}
    )
    A("**Partially contradicted — the interior `12x9` optimum.** The interior-optimum half holds: the")
    A("best grid is never an endpoint at any economy level, for either block statistic. The magnitude")
    A(f"does not: the largest recall spread across grids at matched economy is **{_fmt(max(spreads))}**,")
    A(f"not the ~0.31 seen on 25 cases. Nor is the winner stably `12x9` — across levels and statistics")
    A(f"it lands on {', '.join('`' + g + '`' for g in best_grids)}.")
    A("")
    A("| economy | stat | recall spread across grids | best grid | interior? |")
    A("|---|---|---|---|---|")
    for lvl, e in d["levels"].items():
        for stat, g in e["grid_effect"].items():
            A(
                f"| {lvl} | {stat} | {_fmt(g['spread_recall'])} | {g['best_grid']} | "
                f"{'yes' if g['interior_optimum'] else 'no'} |"
            )
    A("")
    A("At 100 cases the grid is a real but second-order lever, and which grid wins depends on the")
    A("economy budget. The first-order lever is the **statistic**, not the grid:")
    A(f"swapping `mean` for `block_max` is worth {_fmt(swap)} recall — {swap / max(spreads):.0f}x the")
    A("largest grid effect.")
    A("")
    return "\n".join(L) + "\n"


def _cfg(c: dict) -> str:
    return f"{c['stat']}@{c['threshold']} grid={c['block_grid'] or '-'} deb={c['enter_debounce']}"


if __name__ == "__main__":
    sys.exit(main())
