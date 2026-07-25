"""Card Y6 — resolver choice belongs to the CONCERN, not only the source profile.

The source profile DECLARES a default identity policy (its ``identity_strategy``); a concern SELECTS
one via ``goal.state['identity_policy']``. Concern wins; a concern that declares nothing inherits the
profile default byte-for-byte. Selection also refuses a person-identifying resolver (envelope).
"""

from __future__ import annotations

import types

import pytest

from percept.contracts import Goal
from percept.envelope import EnvelopeViolation, SubjectResolverRegistry
from percept.identity_policy import (
    STATE_KEY,
    identity_policy_of,
    select_identity_policy,
)
from percept.source_profiles_default import CONVERSATION, PHYSICAL_SCENE, SCREEN_DOCUMENT
from percept.sourceprofile import SourceProfile


def _goal(**state) -> Goal:
    base = {"type": "watch", "count": 0}
    base.update(state)
    return Goal(id="g", condition="c", say="s", state=base)


# ---------------------------------------------------------------------------
# 1. the source profile DECLARES a default that equals its identity_strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile", [PHYSICAL_SCENE, SCREEN_DOCUMENT, CONVERSATION]
)
def test_default_identity_policy_equals_identity_strategy(profile: SourceProfile):
    assert profile.default_identity_policy == profile.identity_strategy


def test_default_identity_policy_is_in_to_dict():
    d = SCREEN_DOCUMENT.to_dict()
    assert d["default_identity_policy"] == d["identity_strategy"] == "content_key"


# ---------------------------------------------------------------------------
# 2. a concern that declares NOTHING inherits the profile default (byte-for-byte)
# ---------------------------------------------------------------------------


def test_declares_nothing_inherits_profile_default():
    assert identity_policy_of(_goal()) is None
    assert select_identity_policy(_goal(), SCREEN_DOCUMENT) == "content_key"
    assert select_identity_policy(_goal(), PHYSICAL_SCENE) == "track_iou_label"
    assert select_identity_policy(_goal(), CONVERSATION) == "speaker_label"


@pytest.mark.parametrize("junk", [None, "", "   ", 123, [], {"x": 1}])
def test_blank_or_nonstring_declaration_is_treated_as_no_declaration(junk):
    g = _goal(**{STATE_KEY: junk})
    assert identity_policy_of(g) is None
    # falls back to the profile default
    assert select_identity_policy(g, SCREEN_DOCUMENT) == "content_key"


def test_goal_without_state_dict_is_no_declaration():
    obj = types.SimpleNamespace(state=None)
    assert identity_policy_of(obj) is None
    assert select_identity_policy(obj, PHYSICAL_SCENE) == "track_iou_label"


# ---------------------------------------------------------------------------
# 3. the concern WINS — one source, three different goals, three policies
# ---------------------------------------------------------------------------


def test_concern_overrides_the_source_default():
    # one screen source, three goals that legitimately want different identity policies
    po_goal = _goal(**{STATE_KEY: "content_key"})
    cursor_goal = _goal(**{STATE_KEY: "track_iou_label"})
    workflow_goal = _goal(**{STATE_KEY: "none"})

    assert select_identity_policy(po_goal, SCREEN_DOCUMENT) == "content_key"
    assert select_identity_policy(cursor_goal, SCREEN_DOCUMENT) == "track_iou_label"
    # a workflow-state goal wants NO subject — it selects it explicitly, not by re-profiling
    assert select_identity_policy(workflow_goal, SCREEN_DOCUMENT) == "none"


def test_concern_wins_even_when_it_matches_a_different_source_default():
    # physical_scene defaults to track_iou_label; this concern selects the document key instead
    g = _goal(**{STATE_KEY: "content_key"})
    assert select_identity_policy(g, PHYSICAL_SCENE) == "content_key"


def test_declared_policy_is_stripped():
    assert identity_policy_of(_goal(**{STATE_KEY: "  content_key  "})) == "content_key"


def test_selection_allows_a_client_registered_custom_resolver_name():
    # ADR-005: identity policy is open free text; a client's own resolver name is selectable.
    g = _goal(**{STATE_KEY: "my_phash_key"})
    assert select_identity_policy(g, SCREEN_DOCUMENT) == "my_phash_key"


# ---------------------------------------------------------------------------
# 4. envelope: selection refuses a person-identifying resolver (coordinate with Z7)
# ---------------------------------------------------------------------------


class _PersonResolverStub:
    """A resolver that identifies persons. The real registry's constructor/register refuse to build
    or accept one; this stub simulates a future biometric resolver that reached the registry by some
    other door, so we can prove SELECTION refuses it at the point of use."""

    name = "biometric_reid"
    identifies_persons = True


def test_selection_refuses_a_person_identifying_default():
    reg = SubjectResolverRegistry()
    reg._resolvers["biometric_reid"] = _PersonResolverStub()  # bypass register()'s own refusal
    # a source profile that names the biometric resolver as its default must NOT activate globally
    biometric_source = SourceProfile(
        name="security_camera",
        modalities=frozenset({"video"}),
        temporal_unit="frame",
        admission_strategy="tier0_motion_salience",
        evidence_strategy="crop",
        identity_strategy="biometric_reid",
    )
    with pytest.raises(EnvelopeViolation, match="person-identifying"):
        select_identity_policy(_goal(), biometric_source, resolvers=reg)


def test_selection_refuses_a_person_identifying_concern_choice():
    reg = SubjectResolverRegistry()
    reg._resolvers["biometric_reid"] = _PersonResolverStub()
    g = _goal(**{STATE_KEY: "biometric_reid"})
    with pytest.raises(EnvelopeViolation, match="person-identifying"):
        select_identity_policy(g, PHYSICAL_SCENE, resolvers=reg)


def test_shipped_defaults_are_not_person_identifying():
    # the whole point: no shipped source profile's default resolves to a person resolver
    for profile in (PHYSICAL_SCENE, SCREEN_DOCUMENT, CONVERSATION):
        # must not raise against the shipped registry
        assert select_identity_policy(_goal(), profile) == profile.identity_strategy
