from zone_entry_watch import Detection, ZoneEntryWatch


ZONE = ((0.5, 0), (1, 0), (1, 1), (0.5, 1))


def detection(
    x: float, *, y: float = 80, score: float = 0.9, height: float = 20
) -> Detection:
    return Detection((x - 5, y - height, x + 5, y), score)


def watch(**overrides) -> ZoneEntryWatch:
    options = dict(polygon=ZONE, frame_width=100, frame_height=100)
    options.update(overrides)
    return ZoneEntryWatch(**options)


def test_outside_to_inside_emits_once() -> None:
    subject = watch()
    assert subject.observe([detection(40)], 0.0) == []
    assert subject.observe([detection(51)], 1.0) == [1]
    assert subject.observe([detection(60)], 2.0) == []


def test_initial_inside_is_left_censored() -> None:
    subject = watch()
    assert subject.observe([detection(70)], 0.0) == []
    assert subject.observe([detection(75)], 1.0) == []


def test_low_score_and_small_people_are_ineligible() -> None:
    subject = watch()
    subject.observe([detection(40)], 0.0)
    assert subject.observe([detection(55, score=0.09)], 1.0) == []
    assert subject.observe([detection(55, height=2)], 2.0) == []


def test_expired_track_does_not_create_entry() -> None:
    subject = watch(maximum_track_gap_seconds=2.0)
    subject.observe([detection(40)], 0.0)
    assert subject.observe([detection(55)], 3.0) == []


def test_dwell_delays_fire_and_resets_when_person_leaves() -> None:
    subject = watch(dwell_seconds=1.0)
    subject.observe([detection(45)], 0.0)
    assert subject.observe([detection(55)], 1.0) == []
    assert subject.observe([detection(45)], 2.0) == []
    assert subject.observe([detection(55)], 3.0) == []
    assert subject.observe([detection(60)], 4.0) == [1]


def test_two_people_are_tracked_independently() -> None:
    subject = watch(maximum_association_distance=0.2)
    subject.observe([detection(30, y=40), detection(45, y=80)], 0.0)
    assert subject.observe([detection(35, y=40), detection(51, y=80)], 1.0) == [2]
    assert subject.observe([detection(51, y=40), detection(60, y=80)], 2.0) == [1]
