"""The concern registry (percept.harness.concerns) is DEPRECATED, per the enricher playbook.

It has zero runtime consumers — registering a concern here has never changed any watch's
behavior. This module is not removed outright (a MAJOR-boundary courtesy), but it now warns
loudly so a client cannot mistake "it accepted my registration" for "it did something." Covers:
the warning fires on register_concern, the registry still functions exactly as documented while
deprecated, and the live Concern.from_goal / ConcernContext path is untouched by any of this.
"""

from __future__ import annotations

import warnings

import pytest

from percept.adapters import Concern, Goal, Subject
from percept.behavior import ConcernContext
from percept.harness.concerns import clear_concerns, register_concern, registered_concerns


def setup_function(_fn) -> None:
    clear_concerns()


def teardown_function(_fn) -> None:
    clear_concerns()


def _concern(cid: str) -> Concern:
    condition = f"condition for {cid}"
    return Concern(id=cid, identity=Subject(kind="entity", ref=condition), condition=condition, say="heads up")


def test_register_concern_warns_deprecation():
    with pytest.warns(DeprecationWarning, match="no runtime consumer"):
        register_concern(_concern("probe"))


def test_registry_still_functions_while_deprecated():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        register_concern(_concern("still-works"))
    assert [c.id for c in registered_concerns()] == ["still-works"]


def test_goal_shaped_registration_still_warns_and_still_converts():
    with pytest.warns(DeprecationWarning):
        register_concern(Goal(id="g1", condition="door open", say="hey"))
    assert registered_concerns()[0].id == "g1"


def test_concern_from_goal_and_concern_context_are_untouched_by_the_deprecation():
    # The live, non-deprecated path: building a Concern / ConcernContext directly never touches
    # the deprecated registry and never warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        concern = Concern.from_goal(Goal(id="g2", condition="door open", say="hey"))
        ctx = ConcernContext()
    assert concern.id == "g2"
    assert ctx.count == 0
