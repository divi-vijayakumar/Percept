"""CapsuleBuilder folds the recent window into one montage judge frame."""
from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from capsule import CapsuleBuilder


def _frame(val: int, h: int = 120, w: int = 160):
    return np.full((h, w, 3), val, dtype=np.uint8)


class CapsuleBuilderTests(unittest.TestCase):
    def test_montage_is_a_decodable_jpeg_pair(self):
        b = CapsuleBuilder(window=6)
        for v in (10, 40, 200, 60):
            b.push(_frame(v))
        media_type, data = b.montage_jpeg()
        self.assertEqual(media_type, "image/jpeg")
        img = Image.open(io.BytesIO(data))            # decodes without error
        self.assertEqual(img.mode, "RGB")

    def test_window_is_bounded(self):
        b = CapsuleBuilder(window=3)
        for v in range(10):
            b.push(_frame(v * 20))
        # still produces a valid montage from only the last <=3 frames
        _mt, data = b.montage_jpeg()
        self.assertGreater(len(data), 0)

    def test_peak_salience_frame_is_included(self):
        # a bright flash between dim frames is the highest-diff (peak) frame; it must ride in the montage
        b = CapsuleBuilder(window=6, tiles=3, tile_px=64)
        b.push(_frame(0))
        b.push(_frame(0))
        b.push(_frame(255))     # the peak (large frame-diff)
        b.push(_frame(0))
        _mt, data = b.montage_jpeg()
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
        self.assertGreater(int(arr.max()), 200)      # a bright tile survived into the montage

    def test_ready_reflects_buffer(self):
        b = CapsuleBuilder(window=4)
        self.assertFalse(b.ready())
        b.push(_frame(50))
        self.assertTrue(b.ready())


if __name__ == "__main__":
    unittest.main()
