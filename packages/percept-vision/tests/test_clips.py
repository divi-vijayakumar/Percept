import numpy as np
import pytest

from percept.clips import PreRoll
from percept.contracts import Frame

IMG = np.zeros((4, 4), np.uint8)


def frame(t_wall: float, seq: int = 0) -> Frame:
    return Frame("cam", 0, seq, t_wall, t_wall, IMG)


def feed(ring: PreRoll, start: float, end: float, fps: float) -> None:
    n = int(round((end - start) * fps))
    for i in range(n + 1):
        ring.add(frame(start + i / fps, i))


def test_samples_at_most_fps():
    ring = PreRoll(seconds=8, fps=2)
    feed(ring, 1000.0, 1004.0, fps=30)
    times = [f.t_wall for f in ring.frames]
    assert len(times) == 9  # one per half-second slot over 4 s, inclusive of both ends
    assert all(b - a >= 0.49 for a, b in zip(times, times[1:]))


def test_time_bound_drops_old_frames():
    ring = PreRoll(seconds=8, fps=2)
    feed(ring, 1000.0, 1030.0, fps=10)
    assert ring.oldest >= 1030.0 - 8
    assert ring.frames[-1].t_wall == pytest.approx(1030.0)
    assert len(ring) <= 8 * 2 + 1


def test_add_reports_kept():
    ring = PreRoll(fps=2)
    assert ring.add(frame(1000.0))
    assert not ring.add(frame(1000.2))
    assert ring.add(frame(1000.5))


def test_select_spreads_evenly_oldest_to_newest():
    ring = PreRoll(seconds=8, fps=2)
    feed(ring, 1000.0, 1007.5, fps=2)  # 16 frames, one every 0.5 s
    picked = ring.select((1001.0, 1004.0), n=4)
    assert [f.t_wall for f in picked] == [1001.0, 1002.0, 1003.0, 1004.0]
    assert ring.select((1001.0, 1004.0), n=1)[-1].t_wall == 1004.0
    few = ring.select((1001.0, 1001.6), n=4)
    assert [f.t_wall for f in few] == [1001.0, 1001.5]


def test_select_partly_older_than_the_ring():
    ring = PreRoll(seconds=4, fps=2)
    feed(ring, 1000.0, 1010.0, fps=2)
    assert ring.oldest == 1006.0
    picked = ring.select((1002.0, 1008.0), n=4)
    assert picked[0].t_wall == 1006.0 and picked[-1].t_wall == 1008.0
    assert ring.select((1000.0, 1005.0)) == ()


def test_select_validates():
    ring = PreRoll()
    with pytest.raises(ValueError):
        ring.select((2.0, 1.0))
    with pytest.raises(ValueError):
        ring.select((1.0, 2.0), n=0)
    with pytest.raises(ValueError):
        PreRoll(seconds=0)


def test_keyframe_nearest_and_default_end():
    ring = PreRoll(seconds=8, fps=2)
    feed(ring, 1000.0, 1004.0, fps=2)
    assert ring.keyframe((1000.0, 1003.2)).t_wall == 1003.0
    assert ring.keyframe((1000.0, 1004.0), at=1001.25).t_wall == 1001.5  # tie goes later
    assert ring.keyframe((900.0, 950.0)) is None


def test_wall_clock_stepping_back_keeps_ring_ordered():
    ring = PreRoll(seconds=8, fps=2)
    feed(ring, 1000.0, 1004.0, fps=2)
    assert not ring.add(frame(1002.2))  # its slot is already held by the 1002.0 frame
    assert [f.t_wall for f in ring.frames] == [1000.0, 1000.5, 1001.0, 1001.5, 1002.0]
    assert ring.add(frame(1002.6))
    assert ring.frames[-1].t_wall == 1002.6
