"""Render protected, local-only zone overlays for frozen VLM review cases."""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np


def render_case(case: dict, output_dir: pathlib.Path) -> dict:
    capture = cv2.VideoCapture(case["media_path"])
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {case['media_path']}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    start = max(0.0, case["source_seconds"] - 3.0)
    end = min(duration, case["source_seconds"] + 3.0)
    output_width = 960
    output_height = round(height * output_width / width)
    output_path = output_dir / f"{case['case_id']}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {output_path}")
    polygon = np.array(
        [[round(x * output_width), round(y * output_height)] for x, y in case["polygon_normalized"]],
        dtype=np.int32,
    )
    footpoint = (
        round(case["footpoint_normalized"][0] * output_width),
        round(case["footpoint_normalized"][1] * output_height),
    )
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = 0
    while True:
        source_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, frame = capture.read()
        if not ok or source_seconds > end:
            break
        frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
        shade = frame.copy()
        cv2.fillPoly(shade, [polygon], (0, 200, 200))
        frame = cv2.addWeighted(shade, 0.14, frame, 0.86, 0)
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 3, cv2.LINE_AA)
        if abs(source_seconds - case["source_seconds"]) <= 0.3:
            cv2.circle(frame, footpoint, 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, footpoint, 16, (255, 255, 255), 2, cv2.LINE_AA)
        label = (
            f"{case['case_id']}  source={source_seconds:.2f}s  "
            f"FIRE={case['source_seconds']:.2f}s"
        )
        cv2.rectangle(frame, (0, 0), (output_width, 36), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        writer.write(frame)
        frames += 1
    capture.release()
    writer.release()
    return {
        **case,
        "review_clip": str(output_path),
        "clip_start_seconds": start,
        "clip_end_seconds": end,
        "clip_frames": frames,
        "clip_fps": fps,
        "clip_dimensions": [output_width, output_height],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = [render_case(case, args.output_dir) for case in payload["cases"]]
    output = {**payload, "cases": rendered}
    (args.output_dir / "review-cases.rendered.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

