from __future__ import annotations

from percept.harness.candidates import FixedCameraCollisionCandidate
from percept.harness.tier0 import Frame, gray_frame


def _patch(frame: Frame, *, x0: int, y0: int, x1: int, y1: int, value: int) -> Frame:
    pixels = list(frame.pixels)
    for y in range(y0, y1):
        for x in range(x0, x1):
            pixels[y * frame.w + x] = value
    return Frame(frame.w, frame.h, tuple(pixels))


def _detector(**overrides) -> FixedCameraCollisionCandidate:
    kwargs = {
        "region": (0.0, 0.0, 1.0, 1.0),
        "grid": (4, 4),
        "min_local_motion": 0.10,
        "min_local_shock": 0.05,
        "min_recent_motion": 0.01,
        "recent_frames": 4,
        "warmup_frames": 2,
        "cooldown_frames": 2,
    }
    kwargs.update(overrides)
    return FixedCameraCollisionCandidate(**kwargs)


def test_localized_shock_after_motion_proposes_one_candidate() -> None:
    detector = _detector()
    still = gray_frame(16, 16, 0)
    moving = _patch(still, x0=4, y0=4, x1=6, y1=6, value=100)
    impact = _patch(still, x0=4, y0=4, x1=8, y1=8, value=255)

    assert detector(still) == 0.0
    assert detector(moving) == 0.0
    assert detector(impact) == 1.0
    assert detector.last_receipt.reason == "localized_motion_shock"
    assert detector(impact) == 0.0


def test_global_cut_is_rejected() -> None:
    detector = _detector(max_global_change_fraction=0.50)
    dark = gray_frame(16, 16, 0)
    local = _patch(dark, x0=4, y0=4, x1=6, y1=6, value=100)
    cut = gray_frame(16, 16, 255)

    detector(dark)
    detector(local)
    assert detector(cut) == 0.0
    assert detector.last_receipt.reason == "global_change_rejected"


def test_no_candidate_without_preexisting_motion() -> None:
    detector = _detector(min_recent_motion=0.05)
    dark = gray_frame(16, 16, 0)
    impact = _patch(dark, x0=4, y0=4, x1=8, y1=8, value=255)

    detector(dark)
    assert detector(impact) == 0.0
    assert detector.last_receipt.reason == "no_recent_motion"


def test_reset_restores_cold_start() -> None:
    detector = _detector()
    detector(gray_frame(16, 16, 0))
    detector.reset()

    assert detector(gray_frame(16, 16, 255)) == 0.0
    assert detector.last_receipt.reason == "cold_start"
