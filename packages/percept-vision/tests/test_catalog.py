"""ComponentCatalog — a read-only listing over the five component registries.

Covers: every role is non-empty out of the box (nothing silently returns an empty list), the
snapshot hash is stable across calls and changes when a registry gains a member, filtering by
role matches the unfiltered listing, and a descriptor never carries a measured-performance field
(that is what QualificationRecord is for).
"""

from __future__ import annotations

import pytest

from percept.catalog import (
    ROLES,
    ComponentDescriptor,
    QualificationRecord,
    list_components,
    snapshot,
)


def test_every_role_is_populated():
    comps = list_components()
    seen_roles = {c.role for c in comps}
    assert seen_roles == set(ROLES), f"missing roles: {set(ROLES) - seen_roles}"


def test_role_filter_matches_unfiltered_listing():
    everything = list_components()
    for role in ROLES:
        filtered = list_components(role=role)
        assert filtered == [c for c in everything if c.role == role]
        assert filtered  # every shipped role has at least one built-in member


def test_list_components_rejects_unknown_role():
    with pytest.raises(ValueError):
        list_components(role="not-a-real-role")


def test_snapshot_hash_is_stable_across_calls():
    a = snapshot()
    b = snapshot()
    assert a.component_hash == b.component_hash
    assert a.components == b.components


def test_snapshot_hash_changes_when_a_registry_gains_a_member():
    from percept.harness import detectors as det

    before = snapshot()
    det.register_detector("test-catalog-probe-detector", object, when_to_use="test only")
    try:
        after = snapshot()
        assert after.component_hash != before.component_hash
        assert "test-catalog-probe-detector" in after.names(role="detector")
    finally:
        det.unregister("test-catalog-probe-detector")


def test_descriptor_carries_no_measured_performance_field():
    # A ComponentDescriptor's fields are identity + description only; asserted here as an
    # explicit contract so a future field addition trips this test into a real review.
    field_names = {f for f in ComponentDescriptor.__dataclass_fields__}
    perf_shaped = {"recall", "precision", "latency", "accuracy", "score", "metrics"}
    assert not (field_names & perf_shaped)


def test_qualification_record_is_the_separate_measured_type():
    rec = QualificationRecord(
        component_role="detector",
        component_name="motion-periodicity",
        fixture_hash="deadbeef",
        metrics={"recall": 0.9},
    )
    assert rec.to_dict()["metrics"] == {"recall": 0.9}


def test_by_role_and_names_helpers_agree():
    snap = snapshot()
    for role in ROLES:
        names_via_helper = snap.names(role=role)
        names_via_by_role = sorted(c.name for c in snap.by_role(role))
        assert names_via_helper == names_via_by_role
