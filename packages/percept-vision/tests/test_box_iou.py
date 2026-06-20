"""L1: Box.iou geometry — deterministic, no backends."""

import unittest

from percept.adapters import Box


class TestBoxIoU(unittest.TestCase):
    def test_identical_boxes_iou_is_one(self):
        b = Box(0.1, 0.1, 0.2, 0.2)
        self.assertAlmostEqual(b.iou(b), 1.0)

    def test_disjoint_boxes_iou_is_zero(self):
        a = Box(0.0, 0.0, 0.1, 0.1)
        b = Box(0.5, 0.5, 0.1, 0.1)
        self.assertEqual(a.iou(b), 0.0)

    def test_half_overlap(self):
        # two unit-ish boxes overlapping in exactly half their area
        a = Box(0.0, 0.0, 0.2, 0.2)
        b = Box(0.1, 0.0, 0.2, 0.2)
        # intersection = 0.1*0.2 = 0.02 ; union = 0.04+0.04-0.02 = 0.06
        self.assertAlmostEqual(a.iou(b), 0.02 / 0.06)

    def test_touching_edges_is_zero(self):
        a = Box(0.0, 0.0, 0.2, 0.2)
        b = Box(0.2, 0.0, 0.2, 0.2)
        self.assertEqual(a.iou(b), 0.0)

    def test_iou_is_symmetric(self):
        a = Box(0.0, 0.0, 0.3, 0.2)
        b = Box(0.1, 0.05, 0.3, 0.2)
        self.assertAlmostEqual(a.iou(b), b.iou(a))

    def test_degenerate_zero_area_box(self):
        a = Box(0.0, 0.0, 0.0, 0.0)
        b = Box(0.0, 0.0, 0.2, 0.2)
        self.assertEqual(a.iou(b), 0.0)


if __name__ == "__main__":
    unittest.main()
