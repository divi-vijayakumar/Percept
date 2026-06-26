"""RepCount-A dataset access + a general pose-periodicity rep counter (the COUNTING track's real data).

RepCount-A ships as a Lance dataset on HuggingFace with EMBEDDED video blobs — lazily readable, so we
pull only the rows we sample (no full download). 152 test videos across 8 action types.

  * :func:`load_ds` / :func:`load_meta`  — cached dataset + ``{video_id, action_type, count}`` rows.
  * :func:`sample_repcount`              — deterministic, pose-friendly, count-bounded sample.
  * :func:`fetch_video`                  — materialize one embedded blob to a cached ``.mp4``.
  * :func:`pose_rep_count`               — BlazePose landmarks -> PC1 waveform -> autocorrelation period
                                           -> rep count. General across actions (unlike the
                                           jumping-jack-specific openness signal).
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


def fetch_video(index: int, video_id: str) -> Path:
    """Materialize the embedded blob at lance row ``index`` to a cached ``<video_id>.mp4``."""
    _STAGING.mkdir(parents=True, exist_ok=True)
    path = _STAGING / f"{video_id}.mp4"
    if path.exists() and path.stat().st_size > 0:
        return path
    blob = load_ds().take_blobs("video_blob", indices=[index])[0].read()
    path.write_bytes(blob)
    return path


# Landmark indices: wrists (15,16), ankles (27,28), knees (25,26), elbows (13,14), hips (23,24).
# These four limb pairs move once per rep across all pose-friendly actions, so their joint
# trajectory carries the rep period regardless of which action it is.
_JOINTS = (15, 16, 27, 28, 25, 26, 13, 14, 23, 24)


def pose_rep_count(path, fps: int = 6) -> int:
    """General pose-periodicity rep counter -> int.

    BlazePose landmarks per frame -> 20-dim limb-position feature -> PC1 waveform (SVD) ->
    first autocorrelation peak = the rep period -> ``round(duration / period)``.
    """
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    from eval.count_video import decode

    _, full = decode(str(path), fps)
    if not full:
        return 0

    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=_pose_model_path()),
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)

    feats: list[list[float]] = []
    last: list[float] | None = None
    for _t, bgr in full:
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
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
