"""ACOUSTIC at scale — does audio fusion catch SOUND events that vision-only misses?

The single-clip ``track_acoustic`` shows the fusion *story* on one hairdryer; this runs it at scale on
the Perception Test (CC-BY) sound-localisation split, restricted to events whose source is NOT visible
(``is_visible=0``) — the cleanest fusion test, since ONLY audio can tell those events apart. For each
sampled event we judge a frame two ways at the sound onset: vision-only vs fused (frame + the 2 s of
audio ending at onset). Audio earns its place when it FLIPS a vision recall-miss to a satisfied verdict.

  raw  = vision-only recall over the positives (frame at onset).
  e2e  = fused recall, the recall-flip (fraction of vision misses audio rescues), and the fusion-FP
         (fused false-fires on a silent moment where there is no sound).
  core = n/a — the gate adds no fusion (no cells).

A negative is judged at a silent early time (t=0.2 s, before the onset) for events whose onset is late
enough (>1.5 s) that t=0.2 s is plausibly pre-sound; a fused YES there is a fusion false-fire.
"""

from __future__ import annotations

from collections import Counter

from . import perception_test as pt
from .types import TrackResult

TRACK = "acoustic"

FPS = 4.0
AUDIO_S = 2.0
T_NEG = 0.2  # silent pre-onset probe time
NEG_MIN_ONSET = 1.5  # only probe a negative when the real onset is comfortably after t=0.2 s


def _cells(recall_raw, recall_e2e, recall_flip, fusion_fp):
    return {
        ("raw", "recall"): recall_raw,
        ("raw", "recall_flip"): None,
        ("raw", "fusion_fp"): None,
        ("core", "recall"): None,
        ("core", "recall_flip"): None,
        ("core", "fusion_fp"): None,
        ("e2e", "recall"): recall_e2e,
        ("e2e", "recall_flip"): recall_flip,
        ("e2e", "fusion_fp"): fusion_fp,
    }


FMTS = {"recall": "{:.2f}", "recall_flip": "{:.2f}", "fusion_fp": "{:.2f}"}
MEASURED = ["raw recall", "e2e recall", "e2e recall_flip", "e2e fusion_fp"]


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        # OFFLINE card path: use the golden-v1 PT sample count when annotations are staged, but keep
        # VLM/audio-fusion cells modeled until --live measures them.
        try:
            n_events = len(pt.golden_sound_events(require_media=False))
        except Exception:
            n_events = 0
        recall_raw, recall_e2e, recall_flip, fusion_fp = 0.30, 0.90, 0.85, 0.05
        return TrackResult(
            track=TRACK,
            cells=_cells(recall_raw, recall_e2e, recall_flip, fusion_fp),
            fmts=FMTS,
            headline={
                "acoustic recall raw->fused (N PT sound)": f"{recall_raw:.2f} -> {recall_e2e:.2f} (N={n_events})",
                "acoustic recall-flip [95% CI] / fusion-FP": f"{recall_flip:.2f} [n/a,n/a] / {fusion_fp:.2f}",
            },
            measured=[],
            modeled=["raw:recall", "e2e:recall", "e2e:recall_flip", "e2e:fusion_fp"],
            calls=0,
            notes="offline modeled; run --live to measure golden-v1 Perception Test samples",
        )

    import asyncio
    import random

    import cv2

    from eval.count_video import decode
    from eval.fused_judge import extract_audio, to_wav
    from percept import registry
    from percept.config import load_settings
    from percept.gate import classify

    settings = load_settings()
    thr = settings.gate.threshold
    vision = registry.make_vision("gemini", settings)

    events = pt.sample_events("sound", n, is_visible=0)

    def _frame(bgr):
        _, buf = cv2.imencode(".jpg", bgr)
        return ("image/jpeg", buf.tobytes())

    sem = asyncio.Semaphore(6)

    async def _judge(cond, frame, clip):
        async with sem:
            return await vision.judge(cond, frame, audio=clip)

    async def _one(ev):
        path = pt.fetch_video(ev["video_id"])
        _, full = decode(str(path), FPS)
        aud, sr = extract_audio(path)
        cond = pt.label_phrase(ev["label"])
        onset = ev["onset_s"]

        # POSITIVE @ onset: vision-only vs fused on the nearest frame.
        bgr = min(full, key=lambda tb: abs(tb[0] - onset))[1]
        frame = _frame(bgr)
        clip = to_wav(aud, sr, max(0.0, onset - AUDIO_S), onset)
        v_vision = await _judge(cond, frame, None)
        v_fused = await _judge(cond, frame, clip)
        sat_vision = classify(v_vision, 1.0, thr, 0.5) == "yes"
        sat_fused = classify(v_fused, 1.0, thr, 0.5) == "yes"

        # NEGATIVE @ a silent pre-onset moment (only if the real onset is comfortably later).
        neg_fired = None
        if onset > NEG_MIN_ONSET:
            nbgr = min(full, key=lambda tb: abs(tb[0] - T_NEG))[1]
            nframe = _frame(nbgr)
            nclip = to_wav(aud, sr, max(0.0, T_NEG - AUDIO_S), T_NEG)
            n_fused = await _judge(cond, nframe, nclip)
            neg_fired = classify(n_fused, 1.0, thr, 0.5) == "yes"

        return ev["label"], sat_vision, sat_fused, neg_fired

    async def _go():
        return await asyncio.gather(*(_one(ev) for ev in events))

    rows = asyncio.run(_go())

    positives = rows  # one positive per event
    negatives = [r for r in rows if r[3] is not None]
    # calls: 2 per positive (vision + fused) + 1 per negative (fused).
    calls = 2 * len(positives) + len(negatives)

    recall_raw = sum(r[1] for r in positives) / len(positives) if positives else 0.0
    recall_e2e = sum(r[2] for r in positives) / len(positives) if positives else 0.0
    miss_flips = [r[2] for r in positives if not r[1]]  # per-miss: did fusion rescue it?
    recall_flip = sum(miss_flips) / len(miss_flips) if miss_flips else None
    fusion_fp = (sum(1 for r in negatives if r[3]) / len(negatives)) if negatives else None

    # 95% CI on recall_flip via bootstrap of the per-miss flip booleans.
    lo = hi = None
    if len(miss_flips) >= 2:
        import numpy as np

        random.seed(0)
        means = []
        k = len(miss_flips)
        for _ in range(1000):
            sample = [miss_flips[random.randrange(k)] for _ in range(k)]
            means.append(sum(sample) / k)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    flip_s = "n/a" if recall_flip is None else f"{recall_flip:.2f}"
    ci_s = f"[{lo:.2f},{hi:.2f}]" if lo is not None else "[n/a,n/a]"
    fp_s = ("%.2f" % fusion_fp) if fusion_fp is not None else "n/a"
    mix = Counter(r[0] for r in positives)
    notes = f"{pt.manifest_hash(events)}; labels={dict(mix)}"

    return TrackResult(
        track=TRACK,
        cells=_cells(recall_raw, recall_e2e, recall_flip, fusion_fp),
        fmts=FMTS,
        headline={
            "acoustic recall raw->fused (N PT sound)": f"{recall_raw:.2f} -> {recall_e2e:.2f} (N={len(positives)})",
            "acoustic recall-flip [95% CI] / fusion-FP": f"{flip_s} {ci_s} / {fp_s}",
        },
        measured=MEASURED,
        modeled=[],
        calls=calls,
        notes=notes,
    )
