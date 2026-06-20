"""L1: golden frame extraction — decode clip.json base64 frames, order, media-type, missing-skip."""

import base64
import json
import tempfile
import unittest
from pathlib import Path

from eval import frames as F

JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-body"
PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"


def _write_clip(scene_dir: Path, entries):
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "clip.json").write_text(
        json.dumps(
            {"id": scene_dir.name, "frames": [[t, base64.b64encode(b).decode()] for t, b in entries]}
        )
    )


class TestFrames(unittest.TestCase):
    def test_decodes_and_sorts(self):
        with tempfile.TemporaryDirectory() as d:
            scene = Path(d) / "s1"
            _write_clip(scene, [(1.0, JPEG), (0.5, PNG)])  # deliberately out of order
            out = F.frames_for_scene(scene)
            self.assertEqual([t for t, _ in out], [0.5, 1.0])
            self.assertEqual(out[0][1], ("image/png", PNG))
            self.assertEqual(out[1][1], ("image/jpeg", JPEG))

    def test_first_frame(self):
        with tempfile.TemporaryDirectory() as d:
            scene = Path(d) / "s1"
            _write_clip(scene, [(2.0, JPEG), (0.0, JPEG)])
            self.assertEqual(F.first_frame(scene)[0], 0.0)

    def test_first_frame_none_when_empty(self):
        with tempfile.TemporaryDirectory() as d:
            scene = Path(d) / "s1"
            _write_clip(scene, [])
            self.assertIsNone(F.first_frame(scene))

    def test_iter_skips_missing_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_clip(root / "scenes" / "s1", [(0.0, JPEG)])
            # s2 has no clip.json on disk (git-ignored binaries) -> must be skipped, not error
            got = dict(F.iter_scene_frames(root, ["s1", "s2"]))
            self.assertEqual(list(got), ["s1"])
            self.assertEqual(got["s1"][0][1][0], "image/jpeg")


if __name__ == "__main__":
    unittest.main()
