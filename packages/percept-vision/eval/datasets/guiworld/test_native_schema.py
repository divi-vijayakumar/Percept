"""Unit tests: the native ONE-Lab/GUI-World record reader (``parse_guiworld_native``).

Covers the three ways the real dataset schema differs from the representative one:
session id in ``video_path``, ``frame``/fps -> seconds (per-video fps, no constant), and the
translation-artifact scrub on ``sub_goal`` (a bare "Translation failed" must never read as an error).
"""

from __future__ import annotations

import pytest

from eval.datasets.guiworld.transform import (
    ERROR_SHOWN,
    MODAL_APPEARED,
    PAGE_CHANGED,
    classify_event,
    clean_sub_goal,
    native_video_id,
    parse_guiworld_native,
    session_to_fixtures,
)


def test_native_video_id_slugs_path_and_drops_separator():
    assert native_video_id("website/79.mp4") == "website-79"
    assert native_video_id("software/296.mp4") == "software-296"
    # a "__" in the path must not survive (it is the fixture session/condition separator)
    assert "__" not in native_video_id("a__b/1.mp4")


def test_frame_to_seconds_uses_supplied_fps():
    rec = {
        "video_path": "website/79.mp4",
        "system": "Windows",
        "keyframes": [{"frame": 232, "sub_goal": "The webpage loads with pop-up ads."}],
    }
    s30 = parse_guiworld_native(rec, fps=30.0)
    assert s30.video_id == "website-79"
    assert s30.keyframes[0].t == pytest.approx(232 / 30.0)
    # a different (real) fps moves the timestamp — proving fps must be probed, not assumed
    s60 = parse_guiworld_native(rec, fps=60.0)
    assert s60.keyframes[0].t == pytest.approx(232 / 60.0)
    assert s30.keyframes[0].t != s60.keyframes[0].t


def test_missing_or_bad_fps_is_rejected():
    rec = {"video_path": "x/1.mp4", "keyframes": [{"frame": 10, "sub_goal": "a modal appears"}]}
    for bad in (0, 0.0, -1, None):
        with pytest.raises(ValueError):
            parse_guiworld_native(rec, fps=bad)


def test_translate_prefix_stripped_before_classification():
    # website/2's real sub_goal carries a leading Translate: marker + quotes
    raw = 'Translate: "Page redirects to the Worcester Living interface."'
    cleaned = clean_sub_goal(raw)
    assert cleaned == "Page redirects to the Worcester Living interface."
    assert classify_event("", cleaned) is PAGE_CHANGED


def test_translation_failed_marker_is_blanked_not_an_error():
    # software/183's real sub_goal is the bare failed-translation marker — an annotation artifact.
    assert clean_sub_goal("Translation failed") == ""
    # blanked -> classifies to nothing (NOT error_shown via the "failed" keyword)
    assert classify_event("", clean_sub_goal("Translation failed")) is None


def test_native_record_maps_to_expected_conditions():
    rec = {
        "video_path": "website/79.mp4",
        "system": "Windows",
        "keyframes": [
            {"frame": 130, "sub_goal": "Web page redirection"},
            {"frame": 232, "sub_goal": "The webpage loads with pop-up ads."},
            {"frame": 342, "sub_goal": 'Move the mouse to the "X" on the advertisement pop-up.'},
        ],
    }
    session = parse_guiworld_native(rec, fps=30.0)
    fixtures = session_to_fixtures(session)
    by_id = {f.fixture_id: f for f in fixtures}
    assert "website-79__page_changed" in by_id
    assert "website-79__modal_appeared" in by_id
    # the modal fixture has both pop-up keyframes as ordered episodes
    assert len(by_id["website-79__modal_appeared"].episodes) == 2
    assert MODAL_APPEARED and ERROR_SHOWN  # imported symbols referenced
