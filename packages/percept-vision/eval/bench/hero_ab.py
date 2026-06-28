"""Rhythm hero A/B — does the fully-connected hero beat the committed counter on the same N=8?

Three arms, all PRIMED with the same per-action channel prior (held constant, so the only difference is the
hero + phase-gate + VLM machinery):

  * A — committed baseline: single-subject ``perceive_rhythm`` (nearest-hip selection, today's path).
  * B — connected, edge-only: per-person ``RhythmHero`` → EntityGraph/AppearanceIdentity association →
        entity-scoped ``perceive_rhythm`` (the brain selects the performer) + a MEASURED phase-source gate
        (low prior-channel keypoint coverage → silhouette fallback). No VLM.
  * C — connected + LIVE VLM: B, but a count whose periodicity confidence is low ESCALATES to a real Gemini
        recount over the performer's active span (the two-tier "make counts honest" design).

Replays cached RTMO detections (``eval.bench.pose_cache``) so it's fast and deterministic for A/B; Arm C is
live (needs ``PERCEPT_LIVE=1`` + a Gemini key) and non-deterministic.

  PYTHONPATH=packages/percept-vision python -m eval.bench.pose_cache        # one-time
  PYTHONPATH=packages/percept-vision python -m eval.bench.hero_ab           # arms A, B
  PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision python -m eval.bench.hero_ab   # + arm C (live)
"""

from __future__ import annotations

import os
from pathlib import Path

from eval.bench.pose_cache import cache_path, load_detections, reconstruct
from eval.bench.repcount import _STAGING
from eval.bench.silhouette import dominant_period, silhouette_count

# Fixed N=8 (the handoff set) with GT and the validated per-action channel prior (scratch ablation VLM_PRIOR).
MANIFEST = [
    ("stu10_5", "bench_pressing", 16),
    ("train143", "bench_pressing", 0),
    ("stu8_23", "front_raise", 3),
    ("test538", "front_raise", 5),
    ("stu8_34", "jump_jack", 5),
    ("train1665", "jump_jack", 3),
    ("stu4_28", "others", 10),
    ("stu13_33", "pommelhorse", 33),
]
VLM_PRIOR = {"bench_pressing": "limb-vertical", "front_raise": "limb-vertical",
             "jump_jack": "ankle-spread", "others": "hip-vertical", "pommelhorse": "ankle-spread"}
# expected_rom in each channel's NATIVE (torso-normalized) units, per-CHANNEL not per-clip, from the
# committed test (limb-vertical 0.18) + design-note O6/O8 (jack ankle ≈ 0.35). Held identical across arms,
# so even an imperfect value can't bias A-vs-B-vs-C — it only sets the absolute amplitude (fidget) gate.
EXPECTED_ROM = {"limb-vertical": 0.18, "ankle-spread": 0.35, "hip-vertical": 0.20}

FPS = int(os.environ.get("PERCEPT_HERO_FPS", "6"))
MIN_CONF = 0.3                                                  # matches PoseChannels._MIN_POSE_CONF
TAU_KP = float(os.environ.get("PERCEPT_HERO_TAU_KP", "0.3"))   # prior-channel coverage < this → silhouette
TAU_SIL = float(os.environ.get("PERCEPT_HERO_TAU_SIL", "0.3")) # silhouette period conf floor to accept it
TAU_CONF = float(os.environ.get("PERCEPT_HERO_TAU_CONF", "0.4"))  # period conf < this → VLM escalation
PRIMED = os.environ.get("PERCEPT_HERO_UNSUPERVISED") != "1"    # default: primed (the reported condition)
LIVE = os.environ.get("PERCEPT_LIVE") == "1"
MOTION = os.environ.get("PERCEPT_HERO_MOTION") == "1"          # predictive continuity in the EntityGraph (Arm B/C)
VLM_WINDOW_S, VLM_FRAMES = 7.0, 6                              # windowed-direct recount (exp_repcount_windowed)


def _video(vid: str) -> Path:
    return _STAGING / f"{vid}.mp4"


def _goal_state(prior: str | None, erom: float | None) -> dict:
    state = {"type": "rhythm", "count": 0}
    if PRIMED and prior is not None:
        rhythm = {"channel": prior}
        if erom is not None:
            rhythm["expected_rom"] = erom
            rhythm["expected_rom_space"] = "channel_native"
        state["rhythm"] = rhythm
    return state


def _arm_a(cache_frames, prior, erom) -> int:
    """Committed single-subject path off cached poses: nearest-hip selection → single-subject perceive_rhythm
    (carries the last channel vector forward on a dropped pose, exactly like PoseChannels.measure())."""
    from percept import Goal, Percept
    from percept_harness.pose import hip_centre, pose_channels, select_nearest_hip

    core = Percept.create(discover_plugins=False)
    core.add_goal(Goal(id="g", condition="a completed repetition", say="rep", state=_goal_state(prior, erom)))
    anchor, last_ch = None, None
    for fr in cache_frames:
        chosen = select_nearest_hip(reconstruct(fr["dets"]), anchor, MIN_CONF)
        if chosen is not None:
            anchor = hip_centre(chosen)
            last_ch = pose_channels(chosen.keypoints)
        if last_ch:
            core.perceive_rhythm("g", t=fr["t"], channels=last_ch)
    tr = core.rhythm.get("g")
    return tr.count if tr is not None else 0


def _arm_b(cache_frames, full, prior, erom, video):
    """Connected hero (edge-only) + measured phase-source gate. Returns the result dict for arms B and C."""
    from percept import Goal, Percept
    from percept.identity import AppearanceIdentity
    from percept.rhythm_hero import RhythmHero
    from percept_harness.appearance import appearance_signature
    from percept_harness.detectors import person_measures

    core = Percept.create(discover_plugins=False, identity=AppearanceIdentity(), embedder=appearance_signature)
    core.graph.motion = MOTION  # predictive continuity (predict + coast) when enabled
    core.add_goal(Goal(id="g", condition="a completed repetition", say="rep", state=_goal_state(prior, erom)))
    hero = RhythmHero(core, "g", prior_channel=prior)
    for fr, (_t, bgr) in zip(cache_frames, full):
        measures = person_measures(reconstruct(fr["dets"]), min_conf=MIN_CONF)
        hero.observe(fr["t"], measures, frame=bgr)
    res = hero.result()

    # periodicity confidence on the PERFORMER's pose signal (autocorr — same units as the silhouette gate)
    vals = [v for _, v in res.signal]
    _, pose_conf = dominant_period(vals, FPS) if len(vals) >= 6 else (None, 0.0)

    phase_source, count, period_conf, sil = "pose", res.count, pose_conf, None
    # MEASURED gate: an occluded prior channel (low keypoint coverage) routes to the silhouette fallback —
    # never the action name. Only when primed (an unsupervised run has no prior channel to measure coverage on).
    if PRIMED and prior is not None and res.channel_coverage < TAU_KP:
        sil = silhouette_count(video)
        if sil["period_confidence"] >= TAU_SIL:
            phase_source, count, period_conf = "silhouette", sil["count"], sil["period_confidence"]

    return {
        "count": count, "phase_source": phase_source, "period_conf": period_conf or 0.0,
        "channel_coverage": res.channel_coverage, "hero_track_coverage": res.hero_track_coverage,
        "regularity": res.regularity, "subject": res.subject, "n_entities": res.n_entities,
        "reps": res.reps, "silhouette": sil,
    }


def _vlm_recount(client, model, video, action, span) -> int:
    """Live two-tier recount: split the performer's active span into windowed-direct windows, sum Gemini's
    per-window completed-rep counts. Reuses the validated windowed counter (exp_repcount_windowed)."""
    import cv2

    from eval.exp_repcount_windowed import _window_fids, gemini_window_count, grab_frames

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    label = action.replace("_", " ").replace("pommelhorse", "pommel horse")
    a, b = span
    total, s = 0, a
    while s < b:
        fids = _window_fids(nfr, fps, s, VLM_WINDOW_S, VLM_FRAMES)
        frames = grab_frames(str(video), fids)
        if len(frames) >= 2:
            total += gemini_window_count(client, model, label, frames)
        s += VLM_WINDOW_S
    return total


def run_clip(vid, action, gt, *, client=None, model=None) -> dict:
    from eval.count_video import decode

    cache_frames = load_detections(vid, FPS, "rtmo")
    if cache_frames is None:
        raise SystemExit(f"no pose cache for {vid} — run `python -m eval.bench.pose_cache` first "
                         f"(expected {cache_path(vid, FPS, 'rtmo')})")
    video = _video(vid)
    _small, full = decode(str(video), FPS)
    n = min(len(cache_frames), len(full))
    cache_frames, full = cache_frames[:n], full[:n]
    prior = VLM_PRIOR.get(action) if PRIMED else None
    erom = EXPECTED_ROM.get(prior) if prior else None
    dur = full[-1][0] if full else 0.0

    a = _arm_a(cache_frames, prior, erom)
    b = _arm_b(cache_frames, full, prior, erom, video)

    # Arm C: the two-tier policy (core) — escalate a low-confidence count to a LIVE VLM recount over the
    # performer's active span. The recount closure is the injected backstop; with no live backend it's a
    # no-op so C == B (and we still flag that it WOULD have escalated).
    from percept.rhythm_hero import escalate_count

    vlm = None

    def _recount():
        nonlocal vlm
        reps = b["reps"]
        span = (reps[0].start_t, reps[-1].stop_t) if reps else (0.0, dur)
        vlm = _vlm_recount(client, model, video, action, span)
        return vlm

    if LIVE and client is not None:
        c_count, escalated = escalate_count(b["period_conf"], b["count"], _recount, tau=TAU_CONF)
    else:
        c_count, escalated = b["count"], b["period_conf"] < TAU_CONF

    return {"vid": vid, "action": action, "gt": gt, "prior": prior or "—",
            "A": a, "B": b["count"], "C": c_count, "src": b["phase_source"],
            "conf": b["period_conf"], "cov": b["channel_coverage"], "htc": b["hero_track_coverage"],
            "subj": b["subject"], "ents": b["n_entities"], "escalated": escalated, "vlm": vlm}


def _obo(errs):
    return sum(e <= 1 for e in errs) / len(errs) if errs else 0.0


def _mae(rows, key):  # normalized |pred-gt|/max(gt,1), matching the committed repcount harness
    return sum(abs(r[key] - r["gt"]) / max(r["gt"], 1) for r in rows) / len(rows) if rows else 0.0


def main() -> None:
    client = model = None
    if LIVE:
        from eval._dotenv import load_dotenv
        from eval.exp_e2e import gemini_client
        load_dotenv()
        client = gemini_client()
        model = os.environ.get("PERCEPT_GEMINI_MODEL", "gemini-2.5-flash")

    cond = "primed" if PRIMED else "unsupervised"
    print(f"\n  Rhythm hero A/B · N={len(MANIFEST)} · {cond} · fps={FPS} · "
          f"live={'on' if LIVE else 'off'} · motion={'on' if MOTION else 'off'} · τ_kp={TAU_KP} τ_conf={TAU_CONF}")
    print("  " + "=" * 96)
    print(f"  {'clip':10s}{'action':14s}{'gt':>3} | {'A':>3}{'B':>4}{'C':>4} | {'src':>10}{'conf':>6}"
          f"{'cov':>6}{'htc':>5}{'ent':>4} {'esc/vlm':>9}")
    print("  " + "-" * 96)
    rows = []
    for vid, action, gt in MANIFEST:
        r = run_clip(vid, action, gt, client=client, model=model)
        rows.append(r)
        esc = (f"Y→{r['vlm']}" if r["vlm"] is not None else ("Y" if r["escalated"] else "-"))
        print(f"  {r['vid']:10s}{r['action']:14s}{gt:>3} | {r['A']:>3}{r['B']:>4}{r['C']:>4} | "
              f"{r['src']:>10}{r['conf']:>6.2f}{r['cov']:>6.2f}{r['htc']:>5.2f}{r['ents']:>4} {esc:>9}")

    no_gt0 = [r for r in rows if r["gt"] > 0]
    print("  " + "-" * 96)
    print(f"  {'arm':<26}{'OBO(all)':>9}{'MAE(all)':>9}   {'OBO(noGT0)':>11}{'MAE(noGT0)':>11}")
    for arm in ("A", "B", "C"):
        errs_all = [abs(r[arm] - r["gt"]) for r in rows]
        errs_ng = [abs(r[arm] - r["gt"]) for r in no_gt0]
        name = {"A": "A committed", "B": "B connected (edge)", "C": "C connected+VLM"}[arm]
        print(f"  {name:<26}{_obo(errs_all):>9.3f}{_mae(rows, arm):>9.3f}   "
              f"{_obo(errs_ng):>11.3f}{_mae(no_gt0, arm):>11.3f}")
    print("  " + "=" * 96)
    print("  committed reference (primed, repcount harness): OBO≈0.625 MAE≈0.25  ·  unsupervised ≈0.50/0.89")
    if not LIVE:
        print("  (arm C == arm B here: set PERCEPT_LIVE=1 + a Gemini key to run the live escalation)\n")


if __name__ == "__main__":  # pragma: no cover
    main()
