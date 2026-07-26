"""AMI Meeting Corpus (NXT) -> Percept fixture-transform tests.

Proves the public transform contract against the COMMITTED synthetic NXT sample (no real dataset,
no network):

1. annotation -> fixture mapping is correct (headings -> conditions; ABSTRACT skipped; unlinked acts
   drop; the same act linked under two headings yields two key-disjoint episodes);
2. fire timestamps land on the linked dialogue act's START;
3. the bars document + the held-out split load through Percept's OWN loaders;
4. the split is MEETING-disjoint;
plus: the pre-registration freeze detects drift, the golden matcher scores the fixtures cleanly, and
the meeting pack carries the conversation profile's experimental limits without claiming live support.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The transform lives in the top-level ``eval`` dev-tooling package (packages/percept-vision/eval),
# not under ``percept``. Ensure its parent is importable regardless of pytest's invocation dir.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from eval.datasets.ami import (  # noqa: E402
    AMI_ALLOWED_LATENESS_S,
    AMI_GOAL_SHAPE,
    MEETING_OBSERVATION_V1,
    MeetingSplitRegistration,
    bars_path,
    build_meeting_split,
    freeze_registration,
    load_meeting_bars,
    read_meeting_annotations,
    register_meeting_pack,
    transform_corpus,
    transform_meeting,
    write_fixture_tree,
)
from eval.datasets.ami.pack import CONVERSATION_EXPERIMENTAL_LIMITS  # noqa: E402
from eval.datasets.ami.split import SOURCE_PROFILE_NAME  # noqa: E402
from eval.datasets.ami.transform import SECTION_CONDITIONS  # noqa: E402

from percept.domain_pack import DEFAULT_DOMAIN_PACK_REGISTRY, DomainPackRegistry  # noqa: E402
from percept.eval.derivation import fixture_set_hash  # noqa: E402
from percept.eval.differential import (  # noqa: E402
    ConfigDriftError,
    load_bars,
    load_fixtures,
    load_held_out_split,
)
from percept.eval.differential import ScopeVerdict  # noqa: E402
from percept.harness.golden_eval import build_golden_timeline, score  # noqa: E402

_NXT = _PKG_ROOT / "eval" / "datasets" / "ami" / "testdata" / "nxt"
_FIXTURES = _PKG_ROOT / "eval" / "datasets" / "ami" / "testdata" / "fixtures"

_EXPECTED_MEETINGS = {"ES0001a", "ES0001b", "IS0001a", "TS0001a"}


@pytest.fixture(scope="module")
def meetings():
    return {m.meeting_id: m for m in transform_corpus(_NXT)}


# ---------------------------------------------------------------------------
# 1. annotation -> fixture mapping
# ---------------------------------------------------------------------------


def test_every_meeting_becomes_a_fixture(meetings):
    assert set(meetings) == _EXPECTED_MEETINGS


def test_headings_map_to_the_declared_conditions(meetings):
    # ES0001b links two DECISIONS sentences and one ACTIONS sentence.
    conditions = {e.section: e.condition for e in meetings["ES0001b"].episodes}
    assert conditions["DECISIONS"] == "a decision was made"
    assert conditions["ACTIONS"] == "an action item was assigned"
    # IS0001a links a DECISIONS and a PROBLEMS sentence.
    conditions = {e.section: e.condition for e in meetings["IS0001a"].episodes}
    assert conditions["PROBLEMS"] == "a problem was raised"
    # The mapping table itself is exactly the three fire-worthy headings — ABSTRACT is not in it.
    assert set(SECTION_CONDITIONS) == {"DECISIONS", "ACTIONS", "PROBLEMS"}


def test_abstract_links_and_unlinked_acts_produce_no_episode(meetings):
    # ES0001a: act 1 -> DECISIONS + ACTIONS, act 2 -> PROBLEMS, act 3 -> ABSTRACT (skip),
    # act 4 -> unlinked. So exactly three episodes, none of them from acts 3 or 4.
    episodes = meetings["ES0001a"].episodes
    assert len(episodes) == 3
    linked_acts = {e.dialogue_act_id for e in episodes}
    assert "ES0001a.A.dialog-act.3" not in linked_acts  # ABSTRACT-linked -> dropped
    assert "ES0001a.B.dialog-act.4" not in linked_acts  # unlinked -> dropped


def test_same_act_two_headings_yields_two_key_disjoint_episodes(meetings):
    episodes = {e.episode_id: e for e in meetings["ES0001a"].episodes}
    assert "ES0001a.A.dialog-act.1:decision" in episodes
    assert "ES0001a.A.dialog-act.1:action_item" in episodes
    dec = episodes["ES0001a.A.dialog-act.1:decision"]
    act = episodes["ES0001a.A.dialog-act.1:action_item"]
    # Same underlying act, same window, DIFFERENT keys (so score() treats them as distinct goals).
    assert dec.dialogue_act_id == act.dialogue_act_id
    assert (dec.start, dec.end) == (act.start, act.end)
    assert dec.episode_id != act.episode_id


def test_fixture_declares_episode_goal_shape_and_is_scorable(meetings):
    fixture = meetings["ES0001a"].to_fixture()
    assert fixture.goal_shape == AMI_GOAL_SHAPE == "episode"
    assert fixture.scope.verdict is ScopeVerdict.SUPPORTED
    assert fixture.scope.scorable


# ---------------------------------------------------------------------------
# 2. fire timestamps land on the linked act's start
# ---------------------------------------------------------------------------


def test_fire_timestamp_is_the_linked_acts_start():
    for meeting_id in sorted(_EXPECTED_MEETINGS):
        annotations = read_meeting_annotations(_NXT, meeting_id)
        fixture = transform_meeting(annotations)
        for episode in fixture.episodes:
            act = annotations.dialogue_acts[episode.dialogue_act_id]
            gte = episode.to_ground_truth_episode()
            # decision_eligible_ts (the instant build_golden_timeline fires on) == the act's start.
            assert gte.ground_truth.decision_eligible_ts == act.start
            assert gte.ground_truth.event_start_ts == act.start
            # the matching window ends at the act's end + the documented lateness grace.
            assert gte.episode_end_ts == act.end
            assert gte.allowed_lateness_s == AMI_ALLOWED_LATENESS_S


def test_golden_timeline_scores_the_fixture_cleanly(meetings):
    # A fixture whose golden timeline is scored against itself must be perfect: every episode matched,
    # no false fires — proof the emitted windows and keys are internally consistent for the matcher.
    fixture = meetings["TS0001a"].to_fixture()
    result = score(fixture.episodes, build_golden_timeline(fixture.episodes))
    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.false_fires == 0
    assert result.misses == 0


# ---------------------------------------------------------------------------
# 3. committed fixtures ARE what the transform emits (no drift) + Percept loaders read them
# ---------------------------------------------------------------------------


def test_committed_fixtures_match_the_transform_output(meetings, tmp_path):
    write_fixture_tree(list(meetings.values()), tmp_path)
    for meeting_id in _EXPECTED_MEETINGS:
        regenerated = json.loads((tmp_path / meeting_id / "ground_truth.json").read_text())
        committed = json.loads((_FIXTURES / meeting_id / "ground_truth.json").read_text())
        assert regenerated == committed, f"committed fixture for {meeting_id} drifted from transform"


def test_fixtures_load_through_percept_loader():
    fixtures = load_fixtures(_FIXTURES)
    assert {f.fixture_id for f in fixtures} == _EXPECTED_MEETINGS
    for f in fixtures:
        assert f.goal_shape == "episode"
        assert f.episodes  # load_fixtures refuses an empty fixture; every meeting has episodes


def test_bars_document_loads_and_declares_the_three_bars():
    bars = load_meeting_bars()
    assert bars.declared
    by_metric = {b.metric: b for b in bars.bars}
    assert set(by_metric) == {"candidate_recall", "recall_delta", "false_fire_delta"}
    assert by_metric["candidate_recall"].direction == "min"
    assert by_metric["recall_delta"].direction == "min"
    assert by_metric["recall_delta"].require_ci_excludes_zero is True
    assert by_metric["false_fire_delta"].direction == "max"
    # loads identically through the raw loader on the committed path.
    assert load_bars(bars_path()).to_dict() == bars.to_dict()


# ---------------------------------------------------------------------------
# 4. the split is MEETING-disjoint and round-trips through load_held_out_split
# ---------------------------------------------------------------------------


def test_committed_split_loads_and_is_meeting_disjoint():
    fixtures = load_fixtures(_FIXTURES)
    split = load_held_out_split(_FIXTURES, fixtures)
    assert split.declared
    held = set(split.held_out)
    exploratory = set(split.exploratory)
    # THE load-bearing assertion: no meeting id appears on both sides of the split.
    assert held.isdisjoint(exploratory)
    assert held | exploratory == _EXPECTED_MEETINGS
    # 50/50 over four meetings.
    assert len(held) == 2 and len(exploratory) == 2


def test_build_meeting_split_is_deterministic_disjoint_and_balanced():
    ids = ["m5", "m1", "m3", "m2", "m4"]  # unsorted on purpose
    split = build_meeting_split(ids)
    assert build_meeting_split(ids) == split  # deterministic
    assert split["train"] == []
    held, dev = set(split["test"]), set(split["dev"])
    assert held.isdisjoint(dev)  # meeting-disjoint
    assert held | dev == set(ids)
    assert abs(len(held) - len(dev)) <= 1  # balanced to within one meeting


def test_build_meeting_split_rejects_duplicate_meetings():
    with pytest.raises(ValueError, match="unique"):
        build_meeting_split(["m1", "m1", "m2"])


# ---------------------------------------------------------------------------
# 5. pre-registration freeze (DerivationPreRegistration-style) + drift refusal
# ---------------------------------------------------------------------------


def test_freeze_registration_is_stable_and_covers_the_fixture_set():
    bars = load_meeting_bars()
    reg = freeze_registration(_FIXTURES, bars)
    assert isinstance(reg, MeetingSplitRegistration)
    assert reg.split.declared
    assert reg.source_profile == SOURCE_PROFILE_NAME == "conversation"
    # fixture_set_hash is Percept's own hash over the fixture LABELS.
    assert reg.fixture_set_hash == fixture_set_hash(load_fixtures(_FIXTURES))
    # deterministic: same corpus + bars -> same registration hash.
    assert freeze_registration(_FIXTURES, bars).registration_hash == reg.registration_hash


def test_registration_detects_drift():
    bars = load_meeting_bars()
    reg = freeze_registration(_FIXTURES, bars)
    # matching hash passes.
    freeze_registration(
        _FIXTURES, bars, expected_registration_hash=reg.registration_hash
    ).assert_no_drift()
    # a hash that does not match the presented configuration is refused, not scored.
    drifted = freeze_registration(_FIXTURES, bars, expected_registration_hash="deadbeef" * 8)
    with pytest.raises(ConfigDriftError):
        drifted.assert_no_drift()


# ---------------------------------------------------------------------------
# 6. the meeting pack — experimental limits, no live claim, not auto-registered
# ---------------------------------------------------------------------------


def test_pack_shape_and_templates():
    assert MEETING_OBSERVATION_V1.qualified_name == "meeting_observation@1"
    assert set(MEETING_OBSERVATION_V1.templates) == {
        "decision_made",
        "action_assigned",
        "problem_raised",
    }
    plan = MEETING_OBSERVATION_V1.template("decision_made").render()
    assert plan.condition == "a decision was made"


def test_pack_notes_carry_experimental_limits_and_no_live_claim():
    notes = MEETING_OBSERVATION_V1.notes
    assert "diarization" in notes
    assert "24s/turn" in notes
    assert "no live-conversation claim" in notes
    assert "RECORDED" in notes.upper()
    # the exact experimental-limits string is quoted from the conversation profile.
    assert CONVERSATION_EXPERIMENTAL_LIMITS in notes


def test_pack_is_not_auto_registered_but_wires_in_one_line():
    # importing this dev-tooling module must NOT mutate the shared process registry.
    assert "meeting_observation@1" not in DEFAULT_DOMAIN_PACK_REGISTRY
    fresh = DomainPackRegistry()
    register_meeting_pack(fresh.register)
    assert "meeting_observation@1" in fresh
    assert fresh.resolve("meeting_observation@1") is MEETING_OBSERVATION_V1
