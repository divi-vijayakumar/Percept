"""ROI / cropping ablation — does narrow grounded VQA (crop to the detected region) beat full-frame?

The last un-tested VLM-side lever (the strip was a NO-GO). A DIFFERENT, *spatial* mechanism: detect the
goal-relevant region (``vision.detect``), crop the frame to it + a margin, and judge the crop — the
narrow-grounded-VQA premise the backends are built on. Reuses the scorecard predict path on the golden
corpus; compares full-frame vs cropped agreement with the human labels.

Honest end-to-end: if detect picks the wrong region, the crop can HURT — that's the lever as it would
really run, so the verdict is fair. Per-scene box (detected on the middle frame) to bound cost.

Run (sampled): PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision python3.12 -m eval.exp_roi_ablation [N]
Env: PERCEPT_FRAME_CAP (max frames/scene, default 5) · PERCEPT_ROI_MARGIN (box pad, default 0.15)
"""

from __future__ import annotations

import asyncio
import io
import os
import sys

from ._dotenv import load_dotenv
from .frames import frames_for_scene

FRAME_CAP = int(os.environ.get("PERCEPT_FRAME_CAP", 5))
MARGIN = float(os.environ.get("PERCEPT_ROI_MARGIN", 0.15))


def _predict(verdicts: list, threshold: float, fresh_min: float = 0.5) -> str:
    from percept.gate import classify

    truths = [classify(v, 1.0, threshold, fresh_min) for v in verdicts]
    if "yes" in truths:
        return "fire"
    if "unknown" in truths:
        return "ask"
    return "silent"


def _cap(frames: list, cap: int) -> list:
    if len(frames) <= cap:
        return frames
    step = (len(frames) - 1) / (cap - 1)
    return [frames[round(i * step)] for i in range(cap)]


def _crop(frame, box, margin: float):
    """Crop a (media, jpeg-bytes) frame to a normalized Box + margin; re-encode to JPEG."""
    from PIL import Image

    _, raw = frame
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    x1 = max(0.0, box.x - margin) * w
    y1 = max(0.0, box.y - margin) * h
    x2 = min(1.0, box.x + box.w + margin) * w
    y2 = min(1.0, box.y + box.h + margin) * h
    px = (int(x1), int(y1), max(int(x2), int(x1) + 1), max(int(y2), int(y1) + 1))
    out = io.BytesIO()
    img.crop(px).save(out, format="JPEG", quality=85)
    return ("image/jpeg", out.getvalue())


async def run(n: int) -> int:
    load_dotenv()
    from percept import registry
    from percept.adapters import Goal
    from percept.config import load_settings
    from percept.eval import find_golden_root, load_golden

    settings = load_settings()
    root = find_golden_root()
    golden = load_golden(root)
    cases = [
        g for g in golden.gate_cases
        if g.get("review_status") == "approved"
        and (root / "scenes" / g["scene_id"] / "clip.json").exists()
    ]
    cases = sorted(cases, key=lambda g: (g.get("split") != "dev", g["scene_id"]))[:n]
    backend = settings.models.vision if settings.models.vision != "fake" else "gemini"
    vision = registry.make_vision(backend, settings)
    sem = asyncio.Semaphore(6)
    thr = settings.gate.threshold

    async def judge(cond, f):
        async with sem:
            return await vision.judge(cond, f)

    rows, calls = [], 0
    for g in cases:
        cond = g["goal"]["condition"]
        frames = _cap(frames_for_scene(root / "scenes" / g["scene_id"]), FRAME_CAP)
        fr = [f for _, f in frames]
        # per-scene region: detect on the middle frame, take the most-confident box
        mid = fr[len(fr) // 2]
        obs = await vision.detect(mid, [Goal(id="g", condition=cond, say="")])
        calls += 1
        box = max(obs, key=lambda o: o.confidence).box if obs else None

        full = await asyncio.gather(*(judge(cond, f) for f in fr))
        calls += len(fr)
        if box is not None:
            crops = [_crop(f, box, MARGIN) for f in fr]
            cropped = await asyncio.gather(*(judge(cond, c) for c in crops))
            calls += len(fr)
        else:
            cropped = full  # nothing detected -> crop can't differ from full
        rows.append({
            "scene": g["scene_id"], "exp": g["expected_verdict"],
            "full": _predict(full, thr), "crop": _predict(cropped, thr),
            "box": box is not None, "cond": cond[:38],
        })

    fa = sum(r["full"] == r["exp"] for r in rows)
    ca = sum(r["crop"] == r["exp"] for r in rows)
    helped = sum(r["full"] != r["exp"] and r["crop"] == r["exp"] for r in rows)
    hurt = sum(r["full"] == r["exp"] and r["crop"] != r["exp"] for r in rows)
    no_box = sum(1 for r in rows if not r["box"])

    L = ["", "=" * 86, f"  ROI / CROPPING ABLATION — {backend} · {len(rows)} gate-cases · margin={MARGIN} "
         f"· cap={FRAME_CAP} · {calls} calls", "=" * 86,
         f"  {'scene':<12}{'expected':<10}{'full':<10}{'+crop':<10}{'box?':<6} condition"]
    for r in rows:
        mark = "  <-- crop helped" if (r["full"] != r["exp"] and r["crop"] == r["exp"]) else (
            "  <-- crop HURT" if (r["full"] == r["exp"] and r["crop"] != r["exp"]) else "")
        L.append(f"  {r['scene']:<12}{r['exp']:<10}{r['full']:<10}{r['crop']:<10}"
                 f"{('yes' if r['box'] else 'no'):<6} {r['cond']}{mark}")
    L += ["", f"  full agreement: {fa}/{len(rows)}   +crop agreement: {ca}/{len(rows)}   "
          f"(helped {helped}, hurt {hurt}; {no_box} had no detected box)"]
    verdict = "GO (crop lifts agreement)" if ca > fa and helped > hurt else (
        "NO-GO (no lift)" if ca <= fa else "WEAK (within noise — widen N)")
    L += [f"  VERDICT: {verdict}", "=" * 86, ""]
    print("\n".join(L))
    return 0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    return asyncio.run(run(n))


if __name__ == "__main__":  # pragma: no cover — live ablation
    raise SystemExit(main())
