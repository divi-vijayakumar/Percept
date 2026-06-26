#!/usr/bin/env python3
"""Scratch MediaPipe/BlazePose probe.

Runs each probe mode in a separate child process so native MediaPipe aborts do
not take down the parent. This is intentionally diagnostic, not production code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / "packages" / "percept-vision"
MODEL = SDK / "eval" / "pose_landmarker_lite.task"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    pp = [str(SDK)]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env.setdefault("MPLCONFIGDIR", "/private/tmp/percept-mpl-cache")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp/percept-xdg-cache")
    return env


def _synthetic_rgb():
    import numpy as np

    # Small neutral RGB frame. The goal is graph startup, not pose quality.
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 1] = 64
    return img


def mode_import() -> None:
    import mediapipe as mp

    print("mediapipe", getattr(mp, "__version__", "?"))


def mode_tasks_create(delegate_name: str) -> None:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    delegate = getattr(python.BaseOptions.Delegate, delegate_name)
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL),
            delegate=delegate,
        ),
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    print("created tasks PoseLandmarker", delegate_name, landmarker)


def mode_tasks_detect(delegate_name: str) -> None:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    delegate = getattr(python.BaseOptions.Delegate, delegate_name)
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL),
            delegate=delegate,
        ),
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=_synthetic_rgb())
    result = landmarker.detect(image)
    print("detect ok", delegate_name, "poses", len(result.pose_landmarks or []))


def mode_solutions_create() -> None:
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )
    print("created solutions.pose.Pose", pose)


def mode_solutions_detect() -> None:
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )
    result = pose.process(_synthetic_rgb())
    print("solutions detect ok", bool(result.pose_landmarks))


def run_one(mode: str) -> None:
    if mode == "import":
        mode_import()
    elif mode == "tasks-create-cpu":
        mode_tasks_create("CPU")
    elif mode == "tasks-detect-cpu":
        mode_tasks_detect("CPU")
    elif mode == "tasks-create-gpu":
        mode_tasks_create("GPU")
    elif mode == "tasks-detect-gpu":
        mode_tasks_detect("GPU")
    elif mode == "solutions-create":
        mode_solutions_create()
    elif mode == "solutions-detect":
        mode_solutions_detect()
    else:
        raise SystemExit(f"unknown mode {mode!r}")


def run_matrix() -> int:
    modes = [
        "import",
        "tasks-create-cpu",
        "tasks-detect-cpu",
        "tasks-create-gpu",
        "tasks-detect-gpu",
        "solutions-create",
        "solutions-detect",
    ]
    any_failed = False
    for mode in modes:
        print(f"\n=== {mode} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--mode", mode],
            cwd=ROOT,
            env=_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        print(proc.stdout.rstrip())
        print(f"exit={proc.returncode}")
        if proc.returncode != 0:
            any_failed = True
    return 1 if any_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="matrix")
    args = parser.parse_args()
    if args.mode == "matrix":
        return run_matrix()
    run_one(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
