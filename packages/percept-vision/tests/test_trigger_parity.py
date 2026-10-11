"""block_max parity with the 0.x implementation.

The expected values were produced once by the 0.x salience gate (``motion_score(prev, cur,
stat="block_max", region=Box(...))``, 8x6 grid) on the frames built below, and are recorded as
constants. The port must match within 1e-6.

Two things differ on purpose and are kept out of this comparison:

- Rasterising a zone. The old gate took a rectangle and used ``floor(x * W)`` / ``round((x + w)
  * W)`` pixel bounds; zones are now polygons and a pixel is in when its centre is. On pixel-
  aligned rectangles both agree (checked below); on the odd rectangles the recorded integer
  bounds are applied as a mask directly, so the partition arithmetic is still compared exactly.
- Making the gate frame. The old capture path downscaled with bilinear interpolation to uint8;
  ``gate_frame`` area-averages to float32. Inputs here are already 48x36, where the two agree.

The old gate compared each frame with the previous one; ``on_change`` compares with a moving
average. With ``baseline_s=0`` the average is the previous frame, which is checked last.
"""

import numpy as np
import pytest

from percept.contracts import Condition, Frame
from percept.triggers import GATE_H, GATE_W, OnChange, block_max
from percept.zones import zone_mask

y, x = np.mgrid[0:GATE_H, 0:GATE_W]


def texture(k):
    return ((x * 7 + y * 13 + k * 31 + (x * y) % 17) % 256).astype(np.uint8)


def patch(img, y0, y1, x0, x1, delta):
    out = img.astype(np.int16)
    out[y0:y1, x0:x1] += delta
    return np.clip(out, 0, 255).astype(np.uint8)


A = texture(0)
ALIGNED = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))

# name: (reference, current, region, recorded block_max, recorded pixel bounds (x0, y0, x1, y1))
CASES = {
    "texture_pair": (A, texture(1), None, 0.4808278867102396, None),
    "static": (A, A, None, 0.0, None),
    "local_patch": (A, patch(A, 10, 16, 20, 26, 100), None, 0.17429193899782136, None),
    "brightness": (A, patch(A, 0, GATE_H, 0, GATE_W, 10), None, 0.0392156862745098, None),
    "small_patch": (A, patch(A, 30, 33, 2, 4, -90), None, 0.058823529411764705, None),
    "region_aligned_patch": (A, patch(A, 10, 16, 20, 26, 100), ALIGNED, 0.39215686274509803, (12, 9, 36, 27)),
    "region_aligned_texture": (A, texture(3), ALIGNED, 0.6392156862745098, (12, 9, 36, 27)),
    "region_odd": (A, texture(2), "bounds", 0.7607843137254902, (4, 7, 19, 25)),
    "region_narrow": (A, patch(A, 5, 30, 24, 29, 60), "bounds", 0.23529411764705882, (24, 3, 29, 32)),
}


def rect_mask(bounds):
    x0, y0, x1, y1 = bounds
    mask = np.zeros((GATE_H, GATE_W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


@pytest.mark.parametrize("name", sorted(CASES))
def test_block_max_matches_recorded_values(name):
    ref, cur, region, expected, bounds = CASES[name]
    if region is None:
        mask = None
    elif region == "bounds":
        mask = rect_mask(bounds)
    else:
        mask = zone_mask((region,), (), GATE_W, GATE_H)
        assert np.array_equal(mask, rect_mask(bounds))  # pixel-centre rule agrees on aligned edges
    assert block_max(ref, cur, mask) == pytest.approx(expected, abs=1e-6)


def test_on_change_with_frame_to_frame_baseline_matches_recorded_sequence():
    moved = patch(A, 10, 16, 20, 26, 100)
    frames = [A, A, moved, moved, A, texture(5)]
    recorded = [0.0, 0.17429193899782136, 0.0, 0.17429193899782136, 0.596078431372549]
    trig, state, scores = OnChange(baseline_s=0.0), None, []
    c = Condition("c", "anything")
    for i, img in enumerate(frames):
        _, state = trig.step(Frame("cam", 0, i, float(i), float(i), img), (c,), state, float(i))
        if i:
            scores.append(state["conditions"]["c"]["score"])
    assert scores == pytest.approx(recorded, abs=1e-6)
