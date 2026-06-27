"""Pose provider seam for frame-backed edge detectors.

The harness consumes pose as provider-neutral COCO-style keypoints. RTMO is the
default provider because it is multi-person and deployment-friendly; BlazePose is
kept as an optional single-person/high-detail provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


COCO_NOSE = 0
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16


@dataclass(frozen=True)
class PoseKeypoint:
    x: float
    y: float
    confidence: float = 1.0
    z: float | None = None


@dataclass(frozen=True)
class PoseDetection:
    keypoints: tuple[PoseKeypoint, ...]
    confidence: float = 1.0
    person_id: str | None = None
    provider: str = "unknown"
    bbox: tuple[float, float, float, float] | None = None


class PoseProvider(Protocol):
    name: str

    def detect(self, frame: Any) -> list[PoseDetection]:
        ...

    def warm(self) -> str:
        ...


class RTMOPoseProvider:
    """RTMO/MMPose provider.

    This is intentionally lazy and optional. Install the RTMO stack in the
    runtime image and set either:

    - ``PERCEPT_RTMO_MODEL`` to an MMPose model/config alias or local path, or
    - rely on the default ``rtmo`` alias if your MMPose install provides it.

    The provider returns COCO-17 keypoints normalized to 0..1 when frame shape is
    available. It does not import MMPose until the first detection.
    """

    name = "rtmo"

    def __init__(self, *, model: str | None = None, device: str | None = None) -> None:
        self._model = model or os.environ.get("PERCEPT_RTMO_MODEL", "rtmo")
        self._device = device or os.environ.get("PERCEPT_POSE_DEVICE", "cpu")
        self._inferencer = None

    def warm(self) -> str:
        """Build the inferencer now (downloads the RTMO checkpoint). Returns provider name."""
        self._ensure()
        return self.name

    def _ensure(self):
        if self._inferencer is not None:
            return self._inferencer
        try:
            from mmpose.apis import MMPoseInferencer
        except ModuleNotFoundError as exc:  # pragma: no cover - optional L2 dependency
            missing = exc.name or str(exc)
            raise RuntimeError(
                f"RTMO pose runtime is unresolved; missing Python module {missing!r}. "
                "Install the full RTMO/MMPose runtime, including compiled MMCV ops, and set "
                "PERCEPT_RTMO_MODEL if needed. For local fallback, set PERCEPT_POSE_PROVIDER=blazepose."
            ) from exc
        self._inferencer = MMPoseInferencer(pose2d=self._model, device=self._device)
        return self._inferencer

    def detect(self, frame: Any) -> list[PoseDetection]:
        inferencer = self._ensure()
        h, w = _shape_hw(frame)
        result_iter = inferencer(frame, return_vis=False)
        first = next(result_iter, None)
        if not first:
            return []
        predictions = first.get("predictions", first)
        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]
        out: list[PoseDetection] = []
        for i, pred in enumerate(predictions or []):
            raw = pred.get("keypoints") or pred.get("keypoints_visible")
            if raw is None:
                continue
            scores = pred.get("keypoint_scores") or pred.get("keypoint_score") or []
            kpts = []
            for j, xy in enumerate(raw[:17]):
                if len(xy) < 2:
                    continue
                score = float(scores[j]) if j < len(scores) else 1.0
                kpts.append(PoseKeypoint(_norm(float(xy[0]), w), _norm(float(xy[1]), h), score))
            if len(kpts) >= 17:
                out.append(
                    PoseDetection(
                        keypoints=tuple(kpts),
                        confidence=float(pred.get("score", pred.get("bbox_score", 1.0))),
                        person_id=str(i),
                        provider=self.name,
                        bbox=_coerce_bbox(pred.get("bbox"), w, h),
                    )
                )
        return out


class BlazePoseProvider:
    """MediaPipe/BlazePose provider, optional and single-person-first."""

    name = "blazepose"

    def __init__(self, *, model_path: str | None = None) -> None:
        self._model_path = model_path or os.environ.get("PERCEPT_POSE_MODEL", "")
        self._lmk = None
        self._mp = None

    def warm(self) -> str:
        """Build the landmarker now (loads the .task model). Returns provider name."""
        self._ensure()
        return self.name

    def _ensure(self):
        if self._lmk is not None:
            return self._lmk
        if not self._model_path:
            raise RuntimeError("BlazePose needs PERCEPT_POSE_MODEL pointing at a .task model.")
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=self._model_path,
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
        )
        self._mp = mp
        self._lmk = vision.PoseLandmarker.create_from_options(opts)
        return self._lmk

    def detect(self, frame: Any) -> list[PoseDetection]:
        lmk = self._ensure()
        res = lmk.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame))
        if not res.pose_landmarks:
            return []
        lm = res.pose_landmarks[0]
        coco = {
            COCO_NOSE: 0,
            COCO_LEFT_SHOULDER: 11,
            COCO_RIGHT_SHOULDER: 12,
            COCO_LEFT_ELBOW: 13,
            COCO_RIGHT_ELBOW: 14,
            COCO_LEFT_WRIST: 15,
            COCO_RIGHT_WRIST: 16,
            COCO_LEFT_HIP: 23,
            COCO_RIGHT_HIP: 24,
            COCO_LEFT_KNEE: 25,
            COCO_RIGHT_KNEE: 26,
            COCO_LEFT_ANKLE: 27,
            COCO_RIGHT_ANKLE: 28,
        }
        kpts = [PoseKeypoint(0.0, 0.0, 0.0) for _ in range(17)]
        for coco_i, blaze_i in coco.items():
            p = lm[blaze_i]
            kpts[coco_i] = PoseKeypoint(p.x, p.y, getattr(p, "visibility", 1.0), getattr(p, "z", None))
        return [PoseDetection(keypoints=tuple(kpts), confidence=1.0, person_id="0", provider=self.name)]


def make_pose_provider(name: str | None = None) -> PoseProvider:
    provider = (name or os.environ.get("PERCEPT_POSE_PROVIDER") or "rtmo").strip().lower()
    if provider == "rtmo":
        return RTMOPoseProvider()
    if provider in {"blazepose", "mediapipe"}:
        return BlazePoseProvider()
    raise KeyError(f"unknown pose provider {provider!r}; expected 'rtmo' or 'blazepose'")


def ensure_pose_ready(name: str | None = None) -> str:
    """Build the configured pose provider so its weights are downloaded and cached.

    Front-loads (and surfaces failures from) the pose checkpoint download before a run,
    instead of paying it lazily on the first frame. Returns the resolved provider name.
    """
    return make_pose_provider(name).warm()


def _shape_hw(frame: Any) -> tuple[float | None, float | None]:
    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return float(shape[0]), float(shape[1])
    return None, None


def _norm(v: float, denom: float | None) -> float:
    if denom and denom > 1.0 and v > 1.0:
        return v / denom
    return v


def _coerce_bbox(obj: Any, w: float | None, h: float | None) -> tuple[float, float, float, float] | None:
    if obj is None:
        return None
    raw = obj[0] if obj and isinstance(obj[0], (list, tuple)) else obj
    if len(raw) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in raw[:4])
    return (_norm(x1, w), _norm(y1, h), _norm(x2, w), _norm(y2, h))


# ---------------------------------------------------------------------------
# Decomposed motion channels — the eye's fixed, system-tunable measurement basis
# ---------------------------------------------------------------------------
#
# The rhythm sense (percept.rhythm, brain side) decides reps; the eye only MEASURES. Instead of fusing
# landmarks into one action-shaped scalar (the pose-openness mistake), we emit a small basis of
# DECOMPOSED channels — each oscillates once per rep for a family of actions — and let the brain select
# the dominant one. The basis is system-level (this function), NOT per-goal. Orientation/sign is
# irrelevant to the count: one full oscillation has exactly one min and one max per period either way.

#: The default channel basis names (env ``PERCEPT_POSE_CHANNELS`` may restrict to a subset).
POSE_CHANNEL_NAMES = ("limb-vertical", "ankle-spread", "hip-vertical", "torso-vertical")


_MIN_BODY_SCALE = 0.05  # floor on the torso-length normalizer so a foreshortened/degenerate pose can't blow up


def body_scale(keypoints: "tuple[PoseKeypoint, ...]") -> float:
    """A stable per-pose length unit: mean vertical shoulder→hip distance (≈ torso length), floored.

    Normalizing every channel by this makes their OSCILLATION amplitudes scale-comparable across channels
    (a "0.3 torso-length" bob reads the same whether it's hips or wrists) — which is what lets the brain
    select the dominant channel by amplitude without a per-channel unit bias."""
    left = abs(keypoints[COCO_LEFT_HIP].y - keypoints[COCO_LEFT_SHOULDER].y)
    right = abs(keypoints[COCO_RIGHT_HIP].y - keypoints[COCO_RIGHT_SHOULDER].y)
    return max(0.5 * (left + right), _MIN_BODY_SCALE)


def pose_channels(keypoints: "tuple[PoseKeypoint, ...]") -> dict[str, float]:
    """Decompose one pose into the motion-channel basis, each normalized by torso length so amplitudes
    are scale-comparable (COCO keypoints in 0..1).

    * ``limb-vertical``  — wrist elevation vs shoulders (press / raise / pull-up).
    * ``ankle-spread``   — ankle separation (jumping jacks / lateral).
    * ``hip-vertical``   — hip height (squats / sit-ups).
    * ``torso-vertical`` — shoulder height (push-ups / jumps / torso lifts).

    Each is a plain measurement; the brain owns selection + counting. The baseline-relative tracker uses
    only the oscillation (the DC offset cancels), so absolute-position channels are fine."""
    L = body_scale(keypoints)
    limb = 0.5 * (
        (keypoints[COCO_LEFT_SHOULDER].y - keypoints[COCO_LEFT_WRIST].y)
        + (keypoints[COCO_RIGHT_SHOULDER].y - keypoints[COCO_RIGHT_WRIST].y)
    )
    spread = abs(keypoints[COCO_LEFT_ANKLE].x - keypoints[COCO_RIGHT_ANKLE].x)
    hip = 0.5 * (keypoints[COCO_LEFT_HIP].y + keypoints[COCO_RIGHT_HIP].y)
    torso = 0.5 * (keypoints[COCO_LEFT_SHOULDER].y + keypoints[COCO_RIGHT_SHOULDER].y)
    allch = {
        "limb-vertical": float(limb / L),
        "ankle-spread": float(spread / L),
        "hip-vertical": float(hip / L),
        "torso-vertical": float(torso / L),
    }
    sel = os.environ.get("PERCEPT_POSE_CHANNELS")
    if sel:
        want = {p.strip() for p in sel.split(",") if p.strip()}
        return {k: v for k, v in allch.items() if k in want} or allch
    return allch


#: The keypoints each channel is MEASURED from — used to gate the phase source by their visibility
#: (the *measured* signal, never the action name): an occluded prior channel (low keypoint confidence,
#: e.g. pommel-horse legs ≈ 0) routes the count to the silhouette fallback instead of trusting the pose.
CHANNEL_KEYPOINTS = {
    "limb-vertical": (COCO_LEFT_WRIST, COCO_RIGHT_WRIST, COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER),
    "ankle-spread": (COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE),
    "hip-vertical": (COCO_LEFT_HIP, COCO_RIGHT_HIP),
    "torso-vertical": (COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER),
}


def channel_coverage(keypoints: "tuple[PoseKeypoint, ...]") -> dict[str, float]:
    """Per-channel keypoint visibility: mean confidence of the joints each channel is built from."""
    return {
        ch: sum(keypoints[i].confidence for i in idx) / len(idx)
        for ch, idx in CHANNEL_KEYPOINTS.items()
    }


def keypoints_bbox(keypoints: "tuple[PoseKeypoint, ...]") -> tuple[float, float, float, float]:
    """A normalized ``(x, y, w, h)`` box enclosing the pose — the fallback when a provider gives no bbox."""
    xs = [k.x for k in keypoints]
    ys = [k.y for k in keypoints]
    x0, y0 = min(xs), min(ys)
    return (x0, y0, max(xs) - x0, max(ys) - y0)


def bbox_xywh(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert a provider ``(x1, y1, x2, y2)`` box to normalized ``(x, y, w, h)``."""
    x1, y1, x2, y2 = bbox
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def hip_centre(p: "PoseDetection") -> tuple[float, float]:
    """The mid-hip point of a detection (the continuity anchor for single-subject selection)."""
    k = p.keypoints
    return (0.5 * (k[COCO_LEFT_HIP].x + k[COCO_RIGHT_HIP].x),
            0.5 * (k[COCO_LEFT_HIP].y + k[COCO_RIGHT_HIP].y))


def select_nearest_hip(poses, anchor, min_conf: float):
    """The single-subject pick: detection closest to the prior hip centre (continuity), else the
    most-confident. Module-level so it can run on any detection list, not only inside ``PoseChannels``."""
    live = [p for p in poses if p.confidence >= min_conf]
    if not live:
        return None
    if anchor is not None:
        ax, ay = anchor
        return min(live, key=lambda p: (hip_centre(p)[0] - ax) ** 2 + (hip_centre(p)[1] - ay) ** 2)
    return max(live, key=lambda p: p.confidence)


__all__ = [
    "PoseKeypoint",
    "PoseDetection",
    "PoseProvider",
    "RTMOPoseProvider",
    "BlazePoseProvider",
    "make_pose_provider",
    "ensure_pose_ready",
    "pose_channels",
    "POSE_CHANNEL_NAMES",
    "CHANNEL_KEYPOINTS",
    "channel_coverage",
    "keypoints_bbox",
    "bbox_xywh",
    "hip_centre",
    "select_nearest_hip",
]
