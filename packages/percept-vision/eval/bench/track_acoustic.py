"""ACOUSTIC track — does audio fusion catch a working-state that vision alone misses?

On the hairdryer clip, "is it switched on and running (blowing air)" is told by SOUND (the motor
whir), not a still frame: at t=11s the air is invisible so vision-only confidently says OFF, while
fused (frame + the 2 s of audio ending at t) catches the whir and flips to WORKING. At t=1s the
dryer is genuinely off and must stay off under fusion (no false fire).

  raw  = vision-only (Gemini, frame only): does it satisfy "working" at each moment?
  e2e  = fused (Gemini, frame + audio): does audio earn its place by flipping the vision miss?
  core = n/a — the gate adds no fusion (no cells).

Metrics: recall@working (vision/fused at the WORKING moment), recall_flip (fraction of vision
recall-misses that fusion rescues), fusion_fp (false fires fusion introduces on the OFF moment).
"""

from __future__ import annotations

import os

from .types import TrackResult

TRACK = "acoustic"

CLIP = os.path.expanduser("~/Downloads/hairdryer.mp4")
CONDITION = "the hairdryer is switched on and running (blowing air)"
FPS = 4.0
AUDIO_S = 2.0
# (timestamp, ground-truth working?)
MOMENTS = [(11.0, True), (1.0, False)]


def _result(sat_vision_t11, sat_fused_t11, recall_flip, fusion_fp, *, calls, notes):
    """Assemble the TrackResult from the WORKING-moment verdicts + fusion metrics."""
    rv, fv = float(sat_vision_t11), float(sat_fused_t11)
    cells = {
        ("raw", "recall@working"): rv,
        ("raw", "recall_flip"): None,
        ("raw", "fusion_fp"): None,
        ("core", "recall@working"): None,
        ("core", "recall_flip"): None,
        ("core", "fusion_fp"): None,
        ("e2e", "recall@working"): fv,
        ("e2e", "recall_flip"): recall_flip,
        ("e2e", "fusion_fp"): float(fusion_fp),
    }
    flip_s = "n/a" if recall_flip is None else f"{recall_flip:.2f}"
    headline = {
        "acoustic recall@working vision->fused": f"{rv:.0f} -> {fv:.0f}",
        "acoustic recall-flip / fusion-FP": f"{flip_s} / {fusion_fp:.0f}",
    }
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts={"recall@working": "{:.0f}", "recall_flip": "{:.2f}", "fusion_fp": "{:.0f}"},
        headline=headline,
        measured=["raw recall@working", "e2e recall@working", "e2e recall_flip", "e2e fusion_fp"],
        modeled=[],
        calls=calls,
        notes=notes,
    )


def score(*, live: bool, n: int) -> TrackResult:  # n unused (acoustic is a single fixed clip)
    if not live:
        # OFFLINE stub — no decode, no API. Audio rescues the vision miss (the fusion story).
        return _result(0.0, 1.0, 1.0, 0, calls=0, notes="offline stub")

    import asyncio

    import cv2

    from eval.fused_judge import decode, extract_audio, to_wav
    from percept import registry
    from percept.config import load_settings
    from percept.gate import classify

    settings = load_settings()
    thr = settings.gate.threshold
    vision = registry.make_vision("gemini", settings)

    _, full = decode(CLIP, FPS)
    aud, sr = extract_audio(CLIP)

    def _frame(bgr):
        _, buf = cv2.imencode(".jpg", bgr)
        return ("image/jpeg", buf.tobytes())

    async def _judge(t):
        bgr = min(full, key=lambda tb: abs(tb[0] - t))[1]
        frame = _frame(bgr)
        clip2s = to_wav(aud, sr, max(0.0, t - AUDIO_S), t)
        v_vision = await vision.judge(CONDITION, frame)
        v_fused = await vision.judge(CONDITION, frame, audio=clip2s)
        sat_vision = classify(v_vision, 1.0, thr, 0.5) == "yes"
        sat_fused = classify(v_fused, 1.0, thr, 0.5) == "yes"
        return sat_vision, sat_fused

    async def _go():
        return await asyncio.gather(*(_judge(t) for t, _ in MOMENTS))

    verdicts = asyncio.run(_go())  # [(sat_vision, sat_fused), ...] aligned to MOMENTS
    calls = 2 * len(MOMENTS)  # 4 = 2 moments x (vision + fused)

    by_t = {t: (sv, sf) for (t, _gt), (sv, sf) in zip(MOMENTS, verdicts)}
    misses = [(t, gt) for (t, gt), (sv, _sf) in zip(MOMENTS, verdicts) if gt and not sv]
    flips = [(t, gt) for (t, gt) in misses if by_t[t][1]]
    recall_flip = len(flips) / len(misses) if misses else None
    fusion_fp = sum(
        1 for (t, gt), (sv, sf) in zip(MOMENTS, verdicts) if not gt and not sv and sf
    )

    sat_vision_t11, sat_fused_t11 = by_t[11.0]
    notes = "; ".join(
        f"t={t:.0f}s gt={'WORKING' if gt else 'OFF'} vision={'Y' if by_t[t][0] else 'N'} "
        f"fused={'Y' if by_t[t][1] else 'N'}"
        for t, gt in MOMENTS
    )
    return _result(sat_vision_t11, sat_fused_t11, recall_flip, fusion_fp, calls=calls, notes=notes)
