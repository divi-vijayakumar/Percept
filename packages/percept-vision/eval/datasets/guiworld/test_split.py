"""Unit tests: the held-out split is SESSION-disjoint and the pre-registration freezes the corpus."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from percept.eval.derivation import DerivationPreRegistration, fixture_set_hash
from percept.eval.differential import load_fixtures, load_held_out_split

from eval.datasets.guiworld.build import build
from eval.datasets.guiworld.split import build_preregistration, build_session_split
from eval.datasets.guiworld.transform import (
    load_sessions,
    session_of,
    sessions_to_fixtures,
)

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "testdata" / "guiworld_sample.json"
BARS = HERE / "bars.json"
FIXTURES_DIR = HERE / "testdata" / "fixtures"


@pytest.fixture(scope="module")
def fixtures():
    return sessions_to_fixtures(load_sessions(SAMPLE))


# --- the split is by session, never by clip -------------------------------------------------


def test_split_is_session_disjoint_in_memory(fixtures):
    splits = build_session_split(fixtures)
    test_sessions = {session_of(fid) for fid in splits["test"]}
    train_sessions = {session_of(fid) for fid in splits["train"]}
    assert test_sessions and train_sessions
    assert test_sessions.isdisjoint(train_sessions), (
        "a session appears on both sides of the split — a clip-level split leaks between arms"
    )


def test_split_round_trips_through_load_held_out_split_and_stays_session_disjoint():
    fixtures = load_fixtures(FIXTURES_DIR)
    split = load_held_out_split(FIXTURES_DIR, fixtures, split_name="test")
    assert split.declared
    held_out_sessions = {session_of(fid) for fid in split.held_out}
    exploratory_sessions = {session_of(fid) for fid in split.exploratory}
    assert held_out_sessions.isdisjoint(exploratory_sessions)


def test_split_is_fifty_fifty_by_session(fixtures):
    splits = build_session_split(fixtures)
    test_sessions = {session_of(fid) for fid in splits["test"]}
    train_sessions = {session_of(fid) for fid in splits["train"]}
    all_sessions = test_sessions | train_sessions
    assert len(all_sessions) == 6
    assert len(test_sessions) == len(train_sessions) == 3


def test_split_is_deterministic(fixtures):
    assert build_session_split(fixtures) == build_session_split(fixtures)


def test_committed_split_matches_a_fresh_build(fixtures, tmp_path):
    out = build(sample_path=SAMPLE, fixtures_dir=tmp_path / "fixtures", bars_path=BARS, prereg_path=None)
    fresh = (out / "splits.json").read_text()
    committed = (FIXTURES_DIR / "splits.json").read_text()
    assert fresh == committed


# --- the pre-registration freezes the split + bars + fixture-set hash -----------------------


def test_preregistration_is_a_real_frozen_registration():
    prereg = build_preregistration(FIXTURES_DIR, bars_path=BARS)
    assert isinstance(prereg, DerivationPreRegistration)
    # A stable, content-addressed hash and a derived (never wall-time) seed.
    assert len(prereg.registration_hash) == 64
    assert prereg.seed == int(prereg.registration_hash[:12], 16)
    # The bars round-tripped in (candidate_recall + recall_delta + false_fire_delta).
    assert {b.metric for b in prereg.bars.bars} == {
        "candidate_recall",
        "recall_delta",
        "false_fire_delta",
    }
    # Two ladder rungs over one source class (screen_document).
    assert [r.name for r in prereg.ladder.rungs] == ["coast_only", "client_change_signal"]
    assert prereg.ladder.source_profile.name == "screen_document"


def test_preregistration_hash_is_stable_across_rebuilds():
    a = build_preregistration(FIXTURES_DIR, bars_path=BARS)
    b = build_preregistration(FIXTURES_DIR, bars_path=BARS)
    assert a.registration_hash == b.registration_hash
    assert a.fixture_set_hash == b.fixture_set_hash


def test_fixture_set_hash_changes_when_a_label_moves():
    fixtures = load_fixtures(FIXTURES_DIR)
    base = fixture_set_hash(fixtures)
    # Move one episode's event window by relabelling it; the fixture-set hash must change.
    first = fixtures[0]
    moved_episode = dataclasses.replace(first.episodes[0], episode_end_ts=first.episodes[0].episode_end_ts + 5.0)
    moved_fixture = dataclasses.replace(first, episodes=(moved_episode, *first.episodes[1:]))
    perturbed = (moved_fixture, *fixtures[1:])
    assert fixture_set_hash(perturbed) != base


def test_committed_preregistration_matches_a_fresh_one():
    import json

    committed = json.loads((HERE / "testdata" / "preregistration.json").read_text())
    fresh = build_preregistration(FIXTURES_DIR, bars_path=BARS).to_dict()
    assert committed["registration_hash"] == fresh["registration_hash"]
    assert committed["fixture_set_hash"] == fresh["fixture_set_hash"]
    assert committed["bars_hash"] == fresh["bars_hash"]
