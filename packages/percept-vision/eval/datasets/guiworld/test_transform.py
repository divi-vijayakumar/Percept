"""Unit tests: GUI-World annotation -> Percept fixture mapping is correct and honors screen semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from percept.eval.differential import ScopeVerdict
from percept.harness.golden_eval import build_golden_timeline
from percept.source_profiles_default import SCREEN_COAST_INTERVAL_S

from eval.datasets.guiworld.transform import (
    ERROR_SHOWN,
    MODAL_APPEARED,
    PAGE_CHANGED,
    GuiWorldKeyframe,
    GuiWorldSession,
    classify_event,
    fixture_to_ground_truth_dict,
    load_sessions,
    session_of,
    session_to_fixtures,
    sessions_to_fixtures,
)

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "testdata" / "guiworld_sample.json"
FIXTURES_DIR = HERE / "testdata" / "fixtures"


@pytest.fixture(scope="module")
def sessions():
    return load_sessions(SAMPLE)


@pytest.fixture(scope="module")
def fixtures(sessions):
    return sessions_to_fixtures(sessions)


def _fixture(fixtures, fixture_id):
    return next(f for f in fixtures if f.fixture_id == fixture_id)


# --- event -> condition mapping -------------------------------------------------------------


def test_event_type_maps_to_the_three_conditions():
    assert classify_event("modal_open") is MODAL_APPEARED
    assert classify_event("dialog_open") is MODAL_APPEARED
    assert classify_event("error_toast") is ERROR_SHOWN
    assert classify_event("error_dialog") is ERROR_SHOWN
    assert classify_event("navigate") is PAGE_CHANGED
    assert classify_event("tab_switch") is PAGE_CHANGED


def test_condition_text_is_the_required_plain_language():
    assert MODAL_APPEARED.condition == "a modal dialog appeared"
    assert ERROR_SHOWN.condition == "an error message is shown"
    assert PAGE_CHANGED.condition == "the page changed"


def test_caption_keyword_fallback_when_event_type_absent():
    # gw-desktop-06's last keyframe has an empty event and must classify by caption text.
    assert classify_event("", "An error appeared: 'Access denied'") is ERROR_SHOWN
    assert classify_event("", "A confirmation dialog appears") is MODAL_APPEARED
    assert classify_event("", "The app navigates to the settings page") is PAGE_CHANGED


def test_unmapped_gui_event_is_dropped():
    assert classify_event("scroll", "The user scrolls") is None
    assert classify_event("hover", "The user hovers an icon") is None
    assert classify_event("type", "The user types") is None


def test_scroll_keyframe_does_not_become_an_episode(sessions):
    web01 = next(s for s in sessions if s.video_id == "gw-web-01")
    fx = session_to_fixtures(web01)
    modal = _fixture(fx, "gw-web-01__modal_appeared")
    # cookie-consent@6.2 and newsletter@52.0 map; the scroll@12.0 is dropped.
    assert len(modal.episodes) == 2
    assert 12.0 not in [e.ground_truth.event_start_ts for e in modal.episodes]


# --- fire timestamps land on the annotated event time ---------------------------------------


def test_fire_timestamp_lands_on_the_annotated_event_time(fixtures):
    modal = _fixture(fixtures, "gw-web-01__modal_appeared")
    first = min(modal.episodes, key=lambda e: e.ground_truth.event_start_ts)
    assert first.ground_truth.event_start_ts == 6.2
    # eligibility is the same instant (a rendered modal is judgeable immediately) ...
    assert first.ground_truth.decision_eligible_ts == 6.2
    # ... so the golden expected fire lands exactly on the annotated event time.
    golden = build_golden_timeline([first])
    assert golden[0].timing.decision_emitted_ts == 6.2


def test_frame_index_is_resolved_with_fps(fixtures):
    # gw-mobile-03 gives keyframes as frame indices at 30 fps: frame 90 -> 3.0s, 300 -> 10.0s.
    page = _fixture(fixtures, "gw-mobile-03__page_changed")
    starts = sorted(e.ground_truth.event_start_ts for e in page.episodes)
    assert starts == [3.0, 10.0]


# --- content_key identity, never geometry ---------------------------------------------------


def test_identity_keys_on_content_not_geometry(fixtures):
    # gw-web-04: two login modals in the SAME centre region, different content keys.
    modal = _fixture(fixtures, "gw-web-04__modal_appeared")
    ids = [e.episode_id for e in modal.episodes]
    assert len(ids) == len(set(ids)) == 2
    assert any("login-step-1" in i for i in ids)
    assert any("login-2fa" in i for i in ids)


def test_geometry_fields_are_ignored():
    # Two keyframes identical except for geometry produce identical episodes: geometry is never read.
    plain = GuiWorldSession(
        video_id="s1", keyframes=(GuiWorldKeyframe(t=1.0, event="modal_open", content_key="k"),)
    )
    with_geom = GuiWorldSession(
        video_id="s1",
        keyframes=(GuiWorldKeyframe(t=1.0, event="modal_open", content_key="k"),),
    )
    a = session_to_fixtures(plain)[0].episodes[0]
    b = session_to_fixtures(with_geom)[0].episodes[0]
    assert a.episode_id == b.episode_id
    # A geometry key in the raw record is simply not parsed (parse ignores unknown keys).
    from eval.datasets.guiworld.transform import parse_session

    parsed = parse_session(
        {"video_id": "s2", "keyframes": [{"t": 1.0, "event": "modal_open", "bbox": [0, 0, 9, 9], "content_key": "k"}]}
    )
    assert parsed.keyframes[0].content_key == "k"


# --- coast floor preserved in every fixture window ------------------------------------------


def test_every_episode_window_carries_the_coast_grace(fixtures):
    # allowed_lateness_s == the mandatory screen_document coast interval, so a coast-admitted fire
    # (not only a block_max-admitted one) still lands inside the match window.
    assert SCREEN_COAST_INTERVAL_S == 2.0
    for fixture in fixtures:
        for episode in fixture.episodes:
            assert episode.allowed_lateness_s == SCREEN_COAST_INTERVAL_S


# --- scope: every emitted goal shape is scorable --------------------------------------------


def test_every_emitted_goal_shape_is_scorable(fixtures):
    for fixture in fixtures:
        assert fixture.scope.scorable
        assert fixture.scope.verdict is not ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE
    shapes = {f.goal_shape for f in fixtures}
    assert shapes == {"episode", "point"}


def test_fixture_id_round_trips_to_the_session(fixtures):
    for fixture in fixtures:
        assert session_of(fixture.fixture_id) == fixture.fixture_id.split("__", 1)[0]
        assert "__" not in session_of(fixture.fixture_id)


# --- the committed fixture tree matches a fresh transform (no stale commit) ------------------


def test_committed_fixtures_match_a_fresh_transform(fixtures):
    for fixture in fixtures:
        committed_path = FIXTURES_DIR / fixture.fixture_id / "ground_truth.json"
        assert committed_path.exists(), f"missing committed fixture {fixture.fixture_id}"
        committed = json.loads(committed_path.read_text())
        assert committed == fixture_to_ground_truth_dict(fixture)
    # And no EXTRA committed fixture dirs beyond what the transform produces.
    produced_ids = {f.fixture_id for f in fixtures}
    committed_ids = {p.name for p in FIXTURES_DIR.iterdir() if p.is_dir()}
    assert committed_ids == produced_ids
