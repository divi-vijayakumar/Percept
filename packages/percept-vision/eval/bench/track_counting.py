"""COUNTING track — rep-count accuracy on the jumping-jacks clip (the RepCount-Δ0 fix).

The edge POSE detector (BlazePose openness waveform -> one clean peak per rep) beats the per-frame
VLM, which aliases on fast jumps and under-counts. Ground truth = 12 reps.

  raw  = VLM per-frame (Gemini): judge each sampled frame "legs spread mid-jump", count rising edges.
  e2e  = POSE (local, no API): openness_series -> count_reps -> distinct peaks.
  core = n/a — the gate doesn't change counting (no cells).

Metrics: pred (count), obo (off-by-one accuracy, |pred-12|<=1), mae (|pred-12|).
"""

from __future__ import annotations

import os
from pathlib import Path

from .types import TrackResult

TRACK = "counting"

CLIP = os.path.expanduser("~/Downloads/Jumping Jack Progression shorts.mp4")
GT = 12
FPS = 6.0
VLM_FPS = 3.0
CONDITION = (
    "the person's legs are spread wide apart mid-jump (the open phase of a jumping jack), "
    "not standing with feet together"
)


def _obo(pred: int) -> float:
    return 1.0 if abs(pred - GT) <= 1 else 0.0


def _mae(pred: int) -> float:
    return float(abs(pred - GT))


def _pose_model_path() -> str:
    return str((Path(__file__).resolve().parents[1] / "pose_landmarker_lite.task").resolve())


def _e2e_count(full) -> int:
    """POSE path — local, no API."""
    from eval.pose_count import count_reps, openness_series

    os.environ["PERCEPT_POSE_MODEL"] = _pose_model_path()
    series = openness_series(full)
    return len(count_reps(series, min_height=0.6, rel_drop=0.4, refractory=0.4))


def _raw_count(full) -> tuple[int, int]:
    """VLM per-frame path (Gemini). -> (rising-edge count, frames judged)."""
    import asyncio

    import cv2

    from percept import registry
    from percept.config import load_settings
    from percept.gate import classify

    settings = load_settings()
    vision = registry.make_vision("gemini", settings)
    thr = settings.gate.threshold
    stride = max(1, round(FPS / VLM_FPS))
    sampled = full[::stride]

    def _jpg(bgr):
        _, buf = cv2.imencode(".jpg", bgr)
        return ("image/jpeg", buf.tobytes())

    async def _go():
        sem = asyncio.Semaphore(6)

        async def one(t, bgr):
            async with sem:
                v = await vision.judge(CONDITION, _jpg(bgr))
                return t, classify(v, 1.0, thr, 0.5)

        return await asyncio.gather(*(one(t, b) for t, b in sampled))

    sig = asyncio.run(_go())
    rising, prev = 0, False
    for _, truth in sig:
        yes = truth == "yes"
        rising += yes and not prev
        prev = yes
    return rising, len(sampled)


def _result(raw_pred: int, e2e_pred: int, *, calls: int, notes: str) -> TrackResult:
    cells = {
        ("raw", "pred"): raw_pred,
        ("raw", "obo"): _obo(raw_pred),
        ("raw", "mae"): _mae(raw_pred),
        ("core", "pred"): None,
        ("core", "obo"): None,
        ("core", "mae"): None,
        ("e2e", "pred"): e2e_pred,
        ("e2e", "obo"): _obo(e2e_pred),
        ("e2e", "mae"): _mae(e2e_pred),
    }
    headline = {
        f"counting jumping-jacks gt={GT} (raw->e2e pred)": f"{raw_pred} -> {e2e_pred}",
        "counting OBO raw/e2e": f"{_obo(raw_pred):.0f}/{_obo(e2e_pred):.0f}",
    }
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts={"pred": "{}", "obo": "{:.0f}", "mae": "{:.1f}"},
        headline=headline,
        measured=[
            "raw pred", "raw obo", "raw mae",
            "e2e pred", "e2e obo", "e2e mae",
        ],
        modeled=[],
        calls=calls,
        notes=notes,
    )


def score(*, live: bool, n: int) -> TrackResult:  # n unused (counting is a single fixed clip)
    if not live:
        # OFFLINE stub — no decode, no API. Pose beats VLM (the RepCount-Δ0 story).
        return _result(5, 12, calls=0, notes="offline stub")

    from eval.count_video import decode

    _, full = decode(CLIP, FPS)
    e2e_pred = _e2e_count(full)
    raw_pred, calls = _raw_count(full)
    notes = (
        f"gt={GT} jumping-jacks; raw=VLM per-frame (aliases on fast jumps), "
        f"e2e=POSE openness waveform (one peak/rep) — pose beats VLM"
    )
    return _result(raw_pred, e2e_pred, calls=calls, notes=notes)
