"""Perception ablation — does a temporal STRIP change verdicts on real golden scenes? (the strip de-risk)

Reuses the scorecard path on the private golden corpus: for each sampled, approved gate_case it predicts
fire/ask/silent two ways —
    single : judge each frame alone (today)
    +strip : judge each frame WITH its preceding frames as temporal context (one multimodal call)
— and compares both to the human-labeled ``expected_verdict``. GO iff +strip raises agreement with the
labels without a control regression (the S1 discipline: prove the lift; don't assume it).

Vision-only: the golden corpus is frames (no audio), so this de-risks the **strip** lever. The acoustic
(`fuse`) lever needs audio data — record a few clips + use ``make_probe_clip`` for that one.

Run (sampled): PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision python3.12 -m eval.exp_perception_ablation [N]
Env: PERCEPT_STRIP_N (strip width, default 4) · PERCEPT_FRAME_CAP (max frames/scene, default 6)
"""

from __future__ import annotations

import asyncio
import os
import sys

from ._dotenv import load_dotenv
from .frames import frames_for_scene

STRIP_N = int(os.environ.get("PERCEPT_STRIP_N", 4))
FRAME_CAP = int(os.environ.get("PERCEPT_FRAME_CAP", 6))


def _predict(verdicts: list, threshold: float, fresh_min: float = 0.5) -> str:
    """Collapse per-frame verdicts to one gate action — identical rule to scorecard._predict."""
    from percept.gate import classify

    truths = [classify(v, 1.0, threshold, fresh_min) for v in verdicts]
    if "yes" in truths:
        return "fire"
    if "unknown" in truths:
        return "ask"
    return "silent"


def _cap(frames: list, cap: int) -> list:
    """Evenly subsample a scene's frames to bound cost (keeps the temporal span)."""
    if len(frames) <= cap:
        return frames
    step = (len(frames) - 1) / (cap - 1)
    return [frames[round(i * step)] for i in range(cap)]


async def _judge_modes(vision, condition, frames, sem):
    """Per-frame verdicts in two modes: single-frame vs each frame + its preceding strip."""
    fr = [f for _, f in frames]

    async def single(f):
        async with sem:
            return await vision.judge(condition, f)

    async def strip(i, f):
        ctx = fr[max(0, i - (STRIP_N - 1)): i]  # earlier context, oldest->newest
        async with sem:
            return await vision.judge(condition, f, frames=ctx or None)

    singles = await asyncio.gather(*(single(f) for f in fr))
    strips = await asyncio.gather(*(strip(i, f) for i, f in enumerate(fr)))
    return singles, strips


async def run(n: int) -> int:
    load_dotenv()
    from percept import registry
    from percept.config import load_settings
    from percept.eval import find_golden_root, load_golden

    settings = load_settings()
    root = find_golden_root()
    if root is None:
        raise SystemExit("no golden fixtures found (expected eval/fixtures-golden/)")
    golden = load_golden(root)
    cases = [
        g for g in golden.gate_cases
        if g.get("review_status") == "approved"
        and (root / "scenes" / g["scene_id"] / "clip.json").exists()
    ]
    # deterministic sample: dev split first, then by scene id (no RNG)
    cases = sorted(cases, key=lambda g: (g.get("split") != "dev", g["scene_id"]))[:n]
    backend = settings.models.vision if settings.models.vision != "fake" else "gemini"
    vision = registry.make_vision(backend, settings)
    sem = asyncio.Semaphore(6)
    thr = settings.gate.threshold

    rows, calls = [], 0
    for g in cases:
        frames = _cap(frames_for_scene(root / "scenes" / g["scene_id"]), FRAME_CAP)
        s, st = await _judge_modes(vision, g["goal"]["condition"], frames, sem)
        calls += 2 * len(frames)
        rows.append({
            "scene": g["scene_id"], "exp": g["expected_verdict"],
            "single": _predict(s, thr), "strip": _predict(st, thr), "n": len(frames),
            "cond": g["goal"]["condition"][:40],
        })

    sa = sum(r["single"] == r["exp"] for r in rows)
    pa = sum(r["strip"] == r["exp"] for r in rows)
    flips_better = sum(r["single"] != r["exp"] and r["strip"] == r["exp"] for r in rows)
    flips_worse = sum(r["single"] == r["exp"] and r["strip"] != r["exp"] for r in rows)

    L = ["", "=" * 84, f"  PERCEPTION STRIP ABLATION — {backend} · {len(rows)} gate-cases · strip_N={STRIP_N} "
         f"· cap={FRAME_CAP} · {calls} calls", "=" * 84,
         f"  {'scene':<12}{'expected':<10}{'single':<10}{'+strip':<10}  condition"]
    for r in rows:
        mark = "  <-- strip helped" if (r["single"] != r["exp"] and r["strip"] == r["exp"]) else (
            "  <-- strip HURT" if (r["single"] == r["exp"] and r["strip"] != r["exp"]) else "")
        L.append(f"  {r['scene']:<12}{r['exp']:<10}{r['single']:<10}{r['strip']:<10}  {r['cond']}{mark}")
    L += ["", f"  single agreement: {sa}/{len(rows)}   +strip agreement: {pa}/{len(rows)}   "
          f"(helped {flips_better}, hurt {flips_worse})"]
    verdict = "GO (strip lifts agreement)" if pa > sa and flips_better > flips_worse else (
        "NO-GO (no lift)" if pa <= sa else "WEAK (lift within noise — widen N)")
    L += [f"  VERDICT: {verdict}", "=" * 84, ""]
    print("\n".join(L))
    return 0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    return asyncio.run(run(n))


if __name__ == "__main__":  # pragma: no cover — live ablation
    raise SystemExit(main())
