"""RepCount-A dataset access + a general pose-periodicity rep counter (the COUNTING track's real data).

RepCount-A ships as a Lance dataset on HuggingFace with EMBEDDED video blobs — lazily readable, so we
pull only the rows we sample (no full download). 152 test videos across 8 action types.

  * :func:`load_ds` / :func:`load_meta`  — cached dataset + ``{video_id, action_type, count}`` rows.
  * :func:`sample_repcount`              — deterministic, pose-friendly diagnostic sample.
  * :func:`sample_repcount_unbiased`     — deterministic sample across all action types.
  * :func:`fetch_video`                  — materialize one embedded blob to a cached ``.mp4``.
  * :func:`pose_rep_count`               — configured pose provider -> PC1 waveform -> autocorrelation period
                                           -> rep count. General across actions (unlike the
                                           jumping-jack-specific openness signal).
  * :func:`percept_e2e_rep_count`         — registered edge detector(s) -> core.perceive_boundary ->
                                           core-owned count. This is the benchmark-spec E2E path.
"""

from __future__ import annotations

import os
import hashlib
from functools import lru_cache
from pathlib import Path

# Pose-friendly action types (skip pommelhorse — the apparatus occludes the body / pose fails).
POSE_FRIENDLY = {
    "jump_jack",
    "squat",
    "situp",
    "push_up",
    "pull_up",
    "front_raise",
    "bench_pressing",
}

_DATASET_URI = "hf://datasets/lmms-lab-eval/repcounta-lance/data/test.lance"
REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_STAGE_ROOT = Path(__file__).resolve().parents[1] / "bench-staging"
_STAGE_ROOT = Path(os.environ.get("PERCEPT_BENCH_DATA_DIR", _DEFAULT_STAGE_ROOT)).expanduser()
_STAGING = (_STAGE_ROOT if _STAGE_ROOT.is_absolute() else REPO_ROOT / _STAGE_ROOT) / "repcount" / "raw"
_LOCAL_LANCE = _STAGING / "data" / "test.lance"
SAMPLE_SEED = os.environ.get("PERCEPT_BENCH_SAMPLE_SEED", "percept-golden-v1")


def _pose_model_path() -> str:
    return str((Path(__file__).resolve().parents[1] / "pose_landmarker_lite.task").resolve())


@lru_cache(maxsize=1)
def load_ds():
    """The RepCount-A test Lance dataset (lazy — blobs read on demand)."""
    import lance

    if _LOCAL_LANCE.exists():
        return lance.dataset(str(_LOCAL_LANCE))
    return lance.dataset(_DATASET_URI)


@lru_cache(maxsize=1)
def load_meta() -> list[dict]:
    """All 152 rows as ``{video_id, action_type, count}`` (row order = lance row index)."""
    return load_ds().to_table(columns=["video_id", "action_type", "count"]).to_pylist()


def sample_repcount(n: int) -> list[dict]:
    """Deterministic sample of ``{video_id, action, count, index}``.

    Keeps only POSE-FRIENDLY actions with ``count in [4, 20]``, sorts by ``video_id`` for stability,
    and takes the first ``n``. ``index`` is the lance row index (needed for ``take_blobs``).
    """
    meta = load_meta()
    rows = [
        {
            "video_id": m["video_id"],
            "action": m["action_type"],
            "count": int(m["count"]),
            "index": i,
        }
        for i, m in enumerate(meta)
        if m["action_type"] in POSE_FRIENDLY and 4 <= int(m["count"]) <= 20
    ]
    rows.sort(key=lambda r: hashlib.sha256(f"{SAMPLE_SEED}|repcount|{r['video_id']}".encode("utf-8")).hexdigest())
    return rows[:n]


def sample_repcount_unbiased(n: int | None = None) -> list[dict]:
    """Deterministic action-stratified sample across every valid RepCount-A action type.

    This is the sampler for publishable/counting E2E work. It does not filter by pose suitability or
    count range. ``n=None`` or ``n <= 0`` returns the full split metadata.
    """
    rows = [
        {
            "video_id": m["video_id"],
            "action": m["action_type"],
            "count": int(m["count"]),
            "index": i,
        }
        for i, m in enumerate(load_meta())
        if str(m["count"]).isdigit()
    ]
    if n is None or n <= 0:
        rows.sort(key=lambda r: hashlib.sha256(f"{SAMPLE_SEED}|repcount-full|{r['video_id']}".encode("utf-8")).hexdigest())
        return rows

    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["action"], []).append(row)
    for action_rows in buckets.values():
        action_rows.sort(
            key=lambda r: hashlib.sha256(
                f"{SAMPLE_SEED}|repcount-full|{r['video_id']}".encode("utf-8")
            ).hexdigest()
        )

    sample: list[dict] = []
    actions = sorted(buckets)
    while len(sample) < n and any(buckets.values()):
        for action in actions:
            if buckets[action]:
                sample.append(buckets[action].pop(0))
                if len(sample) >= n:
                    break
    return sample


def fetch_video(index: int, video_id: str) -> Path:
    """Materialize the embedded blob at lance row ``index`` to a cached ``<video_id>.mp4``."""
    _STAGING.mkdir(parents=True, exist_ok=True)
    path = _STAGING / f"{video_id}.mp4"
    if path.exists() and path.stat().st_size > 0:
        return path
    blob = load_ds().take_blobs("video_blob", indices=[index])[0].read()
    path.write_bytes(blob)
    return path


# COCO-17 landmark indices: wrists (9,10), ankles (15,16), knees (13,14), elbows (7,8), hips (11,12).
# These four limb pairs move once per rep across all pose-friendly actions, so their joint
# trajectory carries the rep period regardless of which action it is.
_JOINTS = (9, 10, 15, 16, 13, 14, 7, 8, 11, 12)


def pose_rep_count(path, fps: int = 6) -> int:
    """General pose-periodicity rep counter -> int.

    Pose-provider landmarks per frame -> 20-dim limb-position feature -> PC1 waveform (SVD) ->
    first autocorrelation peak = the rep period -> ``round(duration / period)``.
    """
    import cv2
    import numpy as np
    from percept.harness.pose import make_pose_provider

    from eval.count_video import decode

    _, full = decode(str(path), fps)
    if not full:
        return 0

    provider = make_pose_provider()

    feats: list[list[float]] = []
    last: list[float] | None = None
    for _t, bgr in full:
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        poses = provider.detect(rgb)
        if poses:
            lm = max(poses, key=lambda p: p.confidence).keypoints
            feat = [coord for j in _JOINTS for coord in (lm[j].x, lm[j].y)]
            last = feat
        elif last is not None:
            feat = last
        else:
            feat = [0.0] * (len(_JOINTS) * 2)
        feats.append(feat)

    T = len(feats)
    if T < 4:
        return 0

    X = np.asarray(feats, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)  # mean-center columns

    _U, _S, Vt = np.linalg.svd(X, full_matrices=False)
    sig = X @ Vt[0]  # PC1 projection

    a = sig - sig.mean()
    ac = np.correlate(a, a, "full")[len(a) - 1:]
    if ac[0] <= 0:
        return 0
    ac = ac / ac[0]

    lag = None
    for L in range(2, len(ac) - 1):
        if ac[L] > ac[L - 1] and ac[L] >= ac[L + 1] and ac[L] > 0.3:
            lag = L
            break
    if lag is None:
        return 0

    duration = (T - 1) / fps
    period_s = lag / fps
    return int(round(duration / period_s)) if period_s > 0 else 0


# The conductor's goal-set prior is a MANIFEST it READS, not a chain of ``if action == ...``. It maps an
# action LABEL to (the channel that action's rep lives on, that channel's expected range-of-motion in its
# own torso-normalized units). The brain consumes the channel as the rhythm prior and the rom as the
# absolute amplitude (anti-fidget) gate. ``hero_ab.VLM_PRIOR`` / ``EXPECTED_ROM`` is the canonical map; this
# imports it lazily (``hero_ab`` imports ``repcount``, so a module-load import would be circular). A new
# multi-person counting goal is supported by adding a row here — never a new ``if``.
def _channel_prior(action: str) -> tuple[str | None, float | None]:
    """(prior_channel, expected_rom) for ``action`` from the manifest — ``(None, None)`` if unmapped."""
    from eval.bench.hero_ab import EXPECTED_ROM, VLM_PRIOR

    channel = VLM_PRIOR.get(action)
    erom = EXPECTED_ROM.get(channel) if channel else None
    return channel, erom


def percept_e2e_rep_count(
    path, *, action: str, fps: int = 6, capture_fps: int | None = None,
    sense: dict | None = None, detectors: list[str] | None = None,
    hero: bool | None = None,
) -> dict:
    """Run the Percept edge-boundary -> core-count path for one RepCount-A routine.

    The whole video is the routine window supplied by RepCount-A. The "brain-owned" contract is encoded
    as a Percept goal: a completed repetition of the action, with a registered detector selected in
    ``goal.state['detectors']``. The edge proposes timed boundaries; the core consumes them via
    :meth:`Percept.perceive_boundary` and owns the resulting count. No VLM calls are made.

    ``detectors`` defaults to ``PERCEPT_REPCOUNT_DETECTORS`` or ``pose-channels`` (the rhythm-sense eye).
    That default is intentionally global, not action-specific, so the benchmark exposes limitations rather
    than tuning thresholds per class — and it runs with NO goal-set prior, the unsupervised case.

    The path is the **rhythm sense**: the ``pose-channels`` eye MEASURES a decomposed channel basis each
    frame; the brain's :class:`~percept.rhythm.RhythmTracker` (driven by ``core.perceive_rhythm``) selects
    the dominant channel and counts complete reps causally. No VLM calls. A legacy boundary-detector name
    (e.g. ``pose-openness``) instead routes through the old ``perceive_boundary`` blind tally for A/B.

    ``hero`` routes through the PER-PERFORMER hero: the eye measures EVERY person, the
    ``EntityGraph``/``AppearanceIdentity`` seam assigns stable ids, and the brain SELECTS the performer (most
    reps on the prior channel — a bystander/judge reps ~0). This is the fix for multi-person clips (pommel
    horse, bench press with spotters) where the single-subject nearest-hip path can lock the wrong body. The
    counting TRACK (:mod:`eval.bench.track_counting`) passes ``hero=True``; the default stays single-subject
    so the in-process rhythm contract (one eye, one body) is unchanged. ``PERCEPT_REPCOUNT_HERO`` flips the
    default globally (``1`` on, ``0`` off). The channel prior comes from :func:`_channel_prior` (the
    manifest), never per-action ``if``s.
    """
    import cv2

    from percept import Goal, Percept
    from percept.harness.detectors import DetectorBank, make_detector

    from eval.count_video import decode

    names = detectors
    if names is None:
        raw = os.environ.get("PERCEPT_REPCOUNT_DETECTORS", "pose-channels")
        names = [p.strip() for p in raw.split(",") if p.strip()]
    if not names:
        names = ["pose-channels"]
    rhythm_mode = names == ["pose-channels"]
    if hero is None:  # the counting TRACK passes hero=True; env can flip the default globally
        hero = os.environ.get("PERCEPT_REPCOUNT_HERO", "0") != "0"
    hero_mode = rhythm_mode and hero
    if hero_mode:
        return _hero_e2e_rep_count(
            path, action=action, fps=fps, capture_fps=capture_fps, sense=sense, names=names,
        )

    # Capture the cheap stream at capture_fps (the "retina" — high rate); the fovea (SamplingController)
    # decides how DENSELY to spend the expensive pose measurement. Default capture == fps (reproducible).
    capture = int(capture_fps or fps)
    small, full = decode(str(path), capture)
    if not small or not full:
        return {
            "count": 0,
            "frames": 0,
            "boundaries": [],
            "detectors": names,
            "goal": f"a completed {action} repetition",
        }

    goal_id = "repcount"
    goal_text = f"a completed {action.replace('_', ' ')} repetition"
    core = Percept.create(discover_plugins=False)
    state = {"type": "rhythm" if rhythm_mode else "watch", "count": 0, "detectors": names}
    if sense is not None:
        state["sense"] = sense
    core.add_goal(Goal(id=goal_id, condition=goal_text, say="rep counted", state=state))

    boundaries: list[dict] = []
    processed = 0
    health: dict | None = None
    if rhythm_mode:
        # The rhythm sense through the fovea: capture motion every frame, but run the expensive
        # pose measure() ONLY on frames the SamplingController admits. A ``sense`` policy makes it adaptive
        # (coast when quiet, escalate on motion); without one the default policy processes every frame
        # (dense — today's behaviour, and the reproducible benchmark).
        from percept.sampling import SamplingController, SensePolicy
        from percept.harness.tier0 import motion_score

        controller = SamplingController()
        if isinstance(sense, dict):
            controller.configure(goal_id, SensePolicy(**{
                k: sense[k] for k in ("base_fps", "max_fps", "escalate_threshold") if k in sense
            }))
        else:  # dense default: process every captured frame
            controller.configure(goal_id, SensePolicy(base_fps=capture, max_fps=capture, escalate_threshold=0.0))

        eye = make_detector("pose-channels")
        prev = None
        n_present = n_floored = 0
        kp_conf_sum = 0.0
        for (t, gate_frame), (_tf, bgr) in zip(small, full):
            motion = motion_score(prev, gate_frame)
            prev = gate_frame
            if not controller.should_process(goal_id, float(t), motion):
                continue
            processed += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            core.perceive_rhythm(goal_id, t=float(t), channels=eye.measure(rgb))
            if getattr(eye, "_last_present", False):  # accumulate pose-quality for the health trace
                n_present += 1
                kp_conf_sum += eye._last_kp_conf
                n_floored += int(eye._last_floored)
        tracker = core.rhythm.get(goal_id)
        # Health trace — surface SILENT failures (degraded provider, no-op gate, inflated scale, weak
        # periodicity) so a run announces when its count is untrustworthy instead of scoring quietly wrong.
        from eval.bench.diagnostics import from_run

        prior = state.get("rhythm") if isinstance(state.get("rhythm"), dict) else {}
        health = from_run(
            f"counting:{action}", tracker=tracker, n_frames=processed,
            frac_no_detection=(1.0 - n_present / processed) if processed else 1.0,
            mean_kp_conf=(kp_conf_sum / n_present) if n_present else 0.0,
            frac_scale_floored=(n_floored / n_present) if n_present else 0.0,
            expected_rom=prior.get("expected_rom"), amp_gate_frac=prior.get("amp_gate_frac", 0.5),
        ).as_dict()
        if tracker is not None:
            boundaries = [
                {"detector": "pose-channels", "kind": "rep", "t": float(r.stop_t),
                 "value": float(r.amplitude), "channel": r.channel,
                 "span": [float(r.start_t), float(r.peak_t), float(r.stop_t)]}
                for r in tracker.reps
            ]
    else:
        # Legacy A/B path: a boundary detector (e.g. pose-openness) -> blind perceive_boundary tally.
        from percept.harness.tier0 import motion_score

        bank = DetectorBank(names)
        prev = None
        for (t, gate_frame), (_tf, bgr) in zip(small, full):
            motion = motion_score(prev, gate_frame)
            prev = gate_frame
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for boundary in bank.push(t, motion=motion, frame=rgb):
                boundaries.append(
                    {"detector": boundary.detector, "kind": boundary.kind,
                     "t": float(boundary.t), "value": float(boundary.value)}
                )
                core.perceive_boundary(goal_id, t=boundary.t)

    return {
        "count": int(core.edge_counts.get(goal_id, 0)),
        "frames": len(full),
        "capture_fps": capture,
        "processed": processed if rhythm_mode else len(full),  # frames the expensive eye actually ran on
        "boundaries": boundaries,
        "detectors": names,
        "goal": goal_text,
        "health": health,  # None for the legacy path; the rhythm path emits a silent-failure trace
    }


# Observability gates (mirror hero_ab so a counting-track run honours the same blind-count policy).
_TAU_KP = float(os.environ.get("PERCEPT_HERO_TAU_KP", "0.3"))    # prior-channel coverage < this → silhouette
_TAU_SIL = float(os.environ.get("PERCEPT_HERO_TAU_SIL", "0.3"))  # silhouette period conf floor to accept it
_TAU_CONF = float(os.environ.get("PERCEPT_HERO_TAU_CONF", "0.4"))  # period conf < this → BLIND (escalate)


def _hero_e2e_rep_count(
    path, *, action: str, fps: int, capture_fps: int | None, sense: dict | None, names: list[str],
) -> dict:
    """The PER-PERFORMER rhythm path for the counting track (the multi-person fix).

    Mirrors ``hero_ab._arm_b`` but on a freshly decoded clip (live pose provider, no cache dependency): the
    ``pose-channels`` eye measures EVERY person each frame, the ``EntityGraph``/``AppearanceIdentity`` seam
    assigns stable ids, and :class:`~percept.rhythm_hero.RhythmHero` folds each id into its own tracker so the
    brain SELECTS the performer (most reps on the prior channel — a judge/spectator reps ~0). The reported
    count is the PERFORMER's. The channel prior + expected-ROM gate come from the manifest (:func:`_channel_prior`).

    OBSERVABILITY gate (so a BLIND count is never reported clean): if the performer's prior channel is occluded
    (``channel_coverage`` < ``_TAU_KP`` — e.g. the pommel apparatus hides the legs) we route to the measured
    silhouette phase-source instead of the action name; and a count whose periodicity confidence is low (or
    whose coverage is low) is flagged ``escalate=True`` (arm-C-worthy) rather than presented as trustworthy.
    """
    import cv2

    from percept import Goal, Percept
    from percept.identity import AppearanceIdentity
    from percept.rhythm_hero import RhythmHero
    from percept.harness.appearance import appearance_signature
    from percept.harness.detectors import make_detector

    from eval.bench.silhouette import dominant_period, silhouette_count
    from eval.count_video import decode

    capture = int(capture_fps or fps)
    small, full = decode(str(path), capture)
    goal_id = "repcount"
    goal_text = f"a completed {action.replace('_', ' ')} repetition"
    prior_channel, erom = _channel_prior(action)

    if not small or not full:
        return {
            "count": 0, "frames": 0, "capture_fps": capture, "processed": 0,
            "boundaries": [], "detectors": names, "goal": goal_text, "health": None,
            "hero": {"subject": None, "n_entities": 0, "channel_coverage": 0.0,
                     "hero_track_coverage": 0.0, "period_conf": 0.0, "phase_source": "pose",
                     "prior_channel": prior_channel, "escalate": True, "blind_reason": "no_frames"},
        }

    # Goal state carries the manifest prior (channel + native-unit expected_rom) — the brain reads it as the
    # rhythm prior + anti-fidget amplitude gate. Build-2/3 knobs (smooth_s/canonical_select) stay OFF.
    state = {"type": "rhythm", "count": 0, "detectors": names}
    if prior_channel is not None:
        rhythm = {"channel": prior_channel}
        if erom is not None:
            rhythm["expected_rom"] = erom
            rhythm["expected_rom_space"] = "channel_native"
        state["rhythm"] = rhythm
    if sense is not None:
        state["sense"] = sense

    core = Percept.create(discover_plugins=False, identity=AppearanceIdentity(), embedder=appearance_signature)
    core.add_goal(Goal(id=goal_id, condition=goal_text, say="rep counted", state=state))
    hero = RhythmHero(core, goal_id, prior_channel=prior_channel)

    eye = make_detector("pose-channels")
    processed = 0
    for _t, bgr in full:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hero.observe(float(_t), eye.measure_entities(rgb), frame=bgr)
        processed += 1
    res = hero.result()

    # periodicity confidence on the PERFORMER's prior-channel signal (autocorr — same units as silhouette).
    vals = [v for _, v in res.signal]
    _, pose_conf = dominant_period(vals, capture) if len(vals) >= 6 else (None, 0.0)

    phase_source, count, period_conf, sil = "pose", res.count, pose_conf, None
    blind_reason = None
    # MEASURED phase-source gate: an occluded prior channel routes to the silhouette fallback — never the
    # action name. Only meaningful when primed (an unsupervised run has no prior channel to measure coverage).
    if prior_channel is not None and res.channel_coverage < _TAU_KP:
        sil = silhouette_count(path)
        if sil["period_confidence"] >= _TAU_SIL:
            phase_source, count, period_conf = "silhouette", sil["count"], sil["period_confidence"]
        else:
            blind_reason = "low_channel_coverage"

    # The OBSERVABILITY verdict: a low-confidence (or still-low-coverage) count is BLIND → escalate-worthy
    # (arm C). We surface it; we do NOT silently report it as a clean count.
    escalate = (period_conf or 0.0) < _TAU_CONF or res.hero_track_coverage < _TAU_KP
    if escalate and blind_reason is None:
        blind_reason = "low_period_conf" if (period_conf or 0.0) < _TAU_CONF else "low_track_coverage"

    boundaries = [
        {"detector": "pose-channels", "kind": "rep", "t": float(r.stop_t),
         "value": float(r.amplitude), "channel": r.channel,
         "span": [float(r.start_t), float(r.peak_t), float(r.stop_t)]}
        for r in res.reps
    ]

    return {
        "count": int(count),
        "frames": len(full),
        "capture_fps": capture,
        "processed": processed,
        "boundaries": boundaries,
        "detectors": names,
        "goal": goal_text,
        "health": None,  # the hero path carries its observability in the "hero" block below
        "hero": {
            "subject": res.subject, "n_entities": res.n_entities,
            "channel_coverage": res.channel_coverage, "hero_track_coverage": res.hero_track_coverage,
            "regularity": res.regularity, "period_conf": period_conf or 0.0,
            "phase_source": phase_source, "prior_channel": prior_channel,
            "escalate": escalate, "blind_reason": blind_reason,
            "pose_count": res.count, "silhouette": sil,
        },
    }
