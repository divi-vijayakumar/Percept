import numpy as np
import pytest

from percept.zones import crop_box, polygon_mask, zone_mask

RIGHT_HALF = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
TOP_LEFT = ((0.0, 0.0), (0.25, 0.0), (0.25, 0.5), (0.0, 0.5))


def test_rectangle_mask_selects_pixel_centres():
    mask = polygon_mask(RIGHT_HALF, 48, 36)
    assert mask.shape == (36, 48)
    assert mask[:, 24:].all() and not mask[:, :24].any()


def test_triangle_mask_is_half_the_square():
    tri = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    mask = polygon_mask(tri, 100, 100)
    assert abs(mask.mean() - 0.5) < 0.01
    assert mask[0, 0] and not mask[99, 99]


def test_mask_is_resolution_consistent():
    small, big = polygon_mask(TOP_LEFT, 48, 36), polygon_mask(TOP_LEFT, 480, 360)
    assert small.mean() == pytest.approx(big.mean())


def test_no_zones_is_whole_frame():
    assert zone_mask((), (), 48, 36).all()


def test_include_zones_union():
    mask = zone_mask((RIGHT_HALF, TOP_LEFT), (), 48, 36)
    assert mask.sum() == 36 * 24 + 18 * 12


def test_exclude_subtracts():
    mask = zone_mask((), (RIGHT_HALF,), 48, 36)
    assert mask[:, :24].all() and not mask[:, 24:].any()
    both = zone_mask((RIGHT_HALF,), (RIGHT_HALF,), 48, 36)
    assert not both.any()


def test_zone_mask_is_read_only_and_memoised():
    a = zone_mask((RIGHT_HALF,), (), 48, 36)
    b = zone_mask((RIGHT_HALF,), (), 48, 36)
    assert a is b
    with pytest.raises(ValueError):
        a[0, 0] = False


def test_crop_box_with_margin_and_clamp():
    assert crop_box((), 640, 480) == (0, 0, 640, 480)
    assert crop_box((RIGHT_HALF,), 640, 480, margin=0.0) == (320, 0, 640, 480)
    assert crop_box((RIGHT_HALF,), 640, 480, margin=0.05) == (288, 0, 640, 480)
    assert crop_box((TOP_LEFT, RIGHT_HALF), 100, 100, margin=0.0) == (0, 0, 100, 100)
    x0, y0, x1, y1 = crop_box((((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),), 10, 10, margin=0.0)
    assert 0 <= x0 < x1 <= 10 and 0 <= y0 < y1 <= 10


def test_tiny_or_collinear_zone_still_watches_its_centre():
    tiny = ((0.501, 0.501), (0.502, 0.501), (0.502, 0.502))
    line = ((0.1, 0.1), (0.5, 0.5), (0.9, 0.9))
    for poly in (tiny, line):
        assert not polygon_mask(poly, 48, 36).any()
        assert zone_mask((poly,), (), 48, 36).sum() == 1
    assert zone_mask((tiny,), (), 48, 36)[18, 24]
    everything = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert not zone_mask((tiny,), (everything,), 48, 36).any()  # excluding everything stays legal
