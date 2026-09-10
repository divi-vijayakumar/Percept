"""Extract a fixed ordered 16-frame sequence from every frozen review clip."""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rendered_manifest", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    args = parser.parse_args()
    payload = json.loads(args.rendered_manifest.read_text())
    for case in payload["cases"]:
        case_dir = args.output_dir / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(case["review_clip"])
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [round(i * (frame_count - 1) / 15) for i in range(16)]
        paths = []
        for ordinal, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed frame {index} from {case['review_clip']}")
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            path = case_dir / f"frame-{ordinal:02d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"failed to write {path}")
            paths.append(str(path))
        capture.release()
        case["ordered_review_frames"] = paths
    (args.output_dir / "review-cases.frames.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

