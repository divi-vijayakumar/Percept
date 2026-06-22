"""COUNTING track — rep-count accuracy on the real RepCount-A dataset (replacing the single local clip).

We score two paths against the dataset's ground-truth rep counts, over a deterministic pose-friendly
sample of ``n`` videos:

  raw  = VLM per-frame (Gemini): judge each ~2.5fps frame "person at the top/peak of one rep",
         count rising edges of confident-yes.
  e2e  = POSE (local, no API): BlazePose landmarks -> PC1 waveform -> autocorrelation period -> count.
  core = n/a — the gate doesn't change counting (no cells).

Metrics (averaged over videos): obo (off-by-one accuracy, |pred-gt|<=1) and mae (|pred-gt|).
Reference = TransRAC (CVPR'22) on RepCount-A.
"""

from __future__ import annotations

from collections import Counter

from . import repcount
from .types import TrackResult

TRACK = "counting"

VLM_FPS = 2.5
DECODE_FPS = 6

# TransRAC (CVPR'22) RepCount-A test: OBO 0.29, MAE 0.44 — the published SOTA we benchmark against.
_REFERENCE = {
    "obo": {"value": 0.29, "by": "TransRAC CVPR'22"},
    "mae": {"value": 0.44, "by": "TransRAC"},
}
_FMTS = {"obo": "{:.2f}", "mae": "{:.2f}"}
_MEASURED = ["raw obo", "raw mae", "e2e obo", "e2e mae"]


def _raw_count(path, action: str) -> tuple[int, int]:
    """VLM per-frame path (Gemini). -> (rising-edge rep count, frames judged)."""
    import asyncio

    import cv2

    from percept import registry
    from percept.config import load_settings
    from percept.gate import classify

    from eval.count_video import decode

    settings = load_settings()
    vision = registry.make_vision("gemini", settings)
    thr = settings.gate.threshold

    _, full = decode(str(path), DECODE_FPS)
    stride = max(1, round(DECODE_FPS / VLM_FPS))
    sampled = full[::stride]
    condition = (
        "the person is at the top/peak of one " + action.replace("_", " ") + " repetition"
    )

    def _jpg(bgr):
        _, buf = cv2.imencode(".jpg", bgr)
        return ("image/jpeg", buf.tobytes())

    async def _go():
        sem = asyncio.Semaphore(4)

        async def one(t, bgr):
            async with sem:
                v = await vision.judge(condition, _jpg(bgr))
                return t, classify(v, 1.0, thr, 0.5)

        return await asyncio.gather(*(one(t, b) for t, b in sampled))

    sig = asyncio.run(_go())
    rising, prev = 0, False
    for _, truth in sig:
        yes = truth == "yes"
        rising += yes and not prev
        prev = yes
    return rising, len(sampled)


def _result(cells, headline, notes, *, calls, measured, modeled) -> TrackResult:
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts=_FMTS,
        headline=headline,
        measured=measured,
        modeled=modeled,
        reference=_REFERENCE,
        calls=calls,
        notes=notes,
    )


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        # OFFLINE stub — no dataset, no API. Pose (e2e) beats VLM (raw) on rep counting.
        cells = {
            ("raw", "obo"): 0.20,
            ("raw", "mae"): 3.0,
            ("core", "obo"): None,
            ("core", "mae"): None,
            ("e2e", "obo"): 0.45,
            ("e2e", "mae"): 1.8,
        }
        headline = {
            "counting RepCount-A OBO raw/e2e (N)": "0.20 / 0.45 (N=stub) [SOTA TransRAC 0.29]",
            "counting MAE raw/e2e": "3.00 / 1.80",
        }
        return _result(
            cells,
            headline,
            "offline stub",
            calls=0,
            measured=[],
            modeled=_MEASURED,
        )

    ev = repcount.sample_repcount(n)

    calls = 0
    raw_obos, raw_maes, e2e_obos, e2e_maes = [], [], [], []
    per_video = []
    for e in ev:
        path = repcount.fetch_video(e["index"], e["video_id"])
        gt = e["count"]

        e2e_pred = repcount.pose_rep_count(path)
        raw_pred, judged = _raw_count(path, e["action"])
        calls += judged

        denom = max(gt, 1)  # NORMALIZED MAE (|pred-gt|/gt) — matches TransRAC/RepCount-A's MAE, comparable to the ref
        raw_obos.append(1.0 if abs(raw_pred - gt) <= 1 else 0.0)
        raw_maes.append(abs(raw_pred - gt) / denom)
        e2e_obos.append(1.0 if abs(e2e_pred - gt) <= 1 else 0.0)
        e2e_maes.append(abs(e2e_pred - gt) / denom)
        per_video.append((e["video_id"], gt, raw_pred, e2e_pred))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    raw_obo, raw_mae = _mean(raw_obos), _mean(raw_maes)
    e2e_obo, e2e_mae = _mean(e2e_obos), _mean(e2e_maes)

    cells = {
        ("raw", "obo"): raw_obo,
        ("raw", "mae"): raw_mae,
        ("core", "obo"): None,
        ("core", "mae"): None,
        ("e2e", "obo"): e2e_obo,
        ("e2e", "mae"): e2e_mae,
    }
    headline = {
        "counting RepCount-A OBO raw/e2e (N)":
            f"{raw_obo:.2f} / {e2e_obo:.2f} (N={len(ev)}) [SOTA TransRAC 0.29]",
        "counting MAE raw/e2e": f"{raw_mae:.2f} / {e2e_mae:.2f}",
    }
    mix = Counter(e["action"] for e in ev)
    notes = (
        f"action mix: {dict(mix)}; "
        f"per-video (vid,gt,raw,e2e): {per_video}"
    )
    return _result(
        cells,
        headline,
        notes,
        calls=calls,
        measured=_MEASURED,
        modeled=[],
    )
