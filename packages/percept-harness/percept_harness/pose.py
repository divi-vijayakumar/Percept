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


__all__ = [
    "PoseKeypoint",
    "PoseDetection",
    "PoseProvider",
    "RTMOPoseProvider",
    "BlazePoseProvider",
    "make_pose_provider",
]
