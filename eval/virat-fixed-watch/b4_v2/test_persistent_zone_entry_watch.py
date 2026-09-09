from persistent_zone_entry_watch import Detection, PersistentZoneEntryWatch


ZONE = ((0.5, 0), (1, 0), (1, 1), (0.5, 1))


def detection(x: float, *, y: float = 80, score: float = 0.9) -> Detection:
    return Detection((x - 5, y - 20, x + 5, y), score)


def watch(**changes) -> PersistentZoneEntryWatch:
    values = dict(
        polygon=ZONE,
        frame_width=100,
        frame_height=100,
        score_threshold=0.1,
        maximum_association_distance=0.2,
        maximum_track_gap_seconds=3.1,
        minimum_outside_observations=1,
        boundary_margin_normalized=0.0,
        duplicate_footpoint_distance=0.0,
        refractory_seconds=0.0,
    )
    values.update(changes)
    return PersistentZoneEntryWatch(**values)


def actions(outcomes):
    return [outcome.action for outcome in outcomes]


def test_rising_edge_fires_once_and_requires_outside_to_rearm() -> None:
    subject = watch()
    assert actions(subject.observe([detection(40)], 0)) == []
    assert actions(subject.observe([detection(55)], 1)) == ["fire"]
    assert actions(subject.observe([detection(60)], 2)) == []
    assert actions(subject.observe([detection(40)], 3)) == []
    assert actions(subject.observe([detection(55)], 4)) == ["fire"]


def test_initial_inside_is_silent_left_censored() -> None:
    subject = watch()
    assert subject.observe([detection(60)], 0) == []
    assert subject.observe([detection(65)], 1) == []


def test_outside_observability_arms_only_after_required_hits() -> None:
    subject = watch(minimum_outside_observations=2)
    subject.observe([detection(35)], 0)
    assert subject.observe([detection(55)], 1) == []
    subject = watch(minimum_outside_observations=2)
    subject.observe([detection(40)], 2)
    subject.observe([detection(42)], 3)
    assert actions(subject.observe([detection(55)], 4)) == ["fire"]


def test_duplicate_boxes_are_suppressed_before_tracking() -> None:
    subject = watch()
    subject.observe([detection(40), detection(41)], 0)
    result = subject.observe([detection(55), detection(56)], 1)
    assert actions(result) == ["fire"]
    assert subject.suppressed_duplicate_detections == 2


def test_global_assignment_preserves_two_tracks() -> None:
    subject = watch(maximum_association_distance=0.25)
    subject.observe([detection(35, y=40), detection(45, y=80)], 0)
    result = subject.observe([detection(55, y=40), detection(60, y=80)], 1)
    assert actions(result) == ["fire", "fire"]
    assert {outcome.track_id for outcome in result} == {1, 2}


def test_missing_transition_asks_instead_of_firing() -> None:
    subject = watch()
    subject.observe([detection(40)], 0)
    subject.observe([], 1)
    result = subject.observe([detection(55)], 2)
    assert actions(result) == ["ask"]


def test_boundary_band_can_bridge_one_observable_sample() -> None:
    subject = watch(boundary_margin_normalized=0.03)
    subject.observe([detection(40)], 0)
    assert subject.observe([detection(49)], 1) == []
    assert actions(subject.observe([detection(55)], 2)) == ["fire"]


def test_unresolved_boundary_expiration_asks() -> None:
    subject = watch(boundary_margin_normalized=0.03, maximum_track_gap_seconds=1.1)
    subject.observe([detection(40)], 0)
    subject.observe([detection(49)], 1)
    assert actions(subject.observe([], 3)) == ["ask"]


def test_refractory_suppresses_local_duplicate_fire() -> None:
    subject = watch(
        refractory_seconds=2.1,
        maximum_association_distance=0.3,
        duplicate_iou_threshold=1.1,
    )
    subject.observe([detection(40, y=40), detection(42, y=60)], 0)
    result = subject.observe([detection(55, y=49), detection(56, y=51)], 1)
    assert actions(result) == ["fire"]
    assert subject.suppressed_refractory == 1
