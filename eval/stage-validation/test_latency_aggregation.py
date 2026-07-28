"""Latency aggregation unit: one GT OCCURRENCE, not one Incident.

The defect these tests pin: ``latency_summary`` used to average ``onset_to_incident_s`` over every
correct Incident, while ``quality()`` deduplicated timely occurrences by
``(case_id, matched_occurrence)``.  A long GT interval emits many correct Incidents, so one
occurrence was weighted dozens of times and the report answered "how late were all the repeated
messages?" instead of "how soon did we first alert for each occurrence?".

Run: ``python -m pytest eval/stage-validation/test_latency_aggregation.py -q``
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import live_run as LR  # noqa: E402


def row(case: str, occ: tuple, onset: float | None, *, kind: str = "duration",
        dataset: str = "charades", **extra) -> dict:
    r = {
        "case_id": case,
        "goal_kind": kind,
        "dataset": dataset,
        "matched_occurrence": list(occ) if occ else None,
        "onset_to_incident_s": onset,
    }
    r.update(extra)
    return r


# -- the core defect ---------------------------------------------------------------------------

def test_repeated_incidents_for_one_occurrence_collapse_to_the_earliest():
    """[1, 10, 20] for ONE occurrence is n=1 at 1.0 -- not n=3 with a median of 10."""
    lat = [row("c1", (5.0, 40.0), t) for t in (1.0, 10.0, 20.0)]
    out = LR.latency_summary(lat)
    assert out["legs"]["onset_to_incident_s"]["n"] == 1
    assert out["legs"]["onset_to_incident_s"]["p50"] == 1.0
    assert out["legs"]["onset_to_incident_s"]["max"] == 1.0
    assert out["matched_occurrences"] == 1


def test_earliest_wins_regardless_of_emission_order():
    for order in ((20.0, 1.0, 10.0), (10.0, 20.0, 1.0), (1.0, 20.0, 10.0)):
        out = LR.latency_summary([row("c1", (5.0, 40.0), t) for t in order])
        assert out["legs"]["onset_to_incident_s"]["p50"] == 1.0, order


def test_two_distinct_occurrences_remain_two_samples():
    lat = [row("c1", (5.0, 10.0), 3.0), row("c1", (50.0, 60.0), 7.0)]
    out = LR.latency_summary(lat)
    assert out["legs"]["onset_to_incident_s"]["n"] == 2
    assert out["matched_occurrences"] == 2


def test_identical_intervals_in_different_cases_are_not_merged():
    """case_id is part of the key -- two clips can share an interval by coincidence."""
    lat = [row("c1", (5.0, 10.0), 2.0), row("c2", (5.0, 10.0), 9.0)]
    out = LR.latency_summary(lat)
    assert out["legs"]["onset_to_incident_s"]["n"] == 2
    assert out["matched_occurrences"] == 2


# -- population boundaries ---------------------------------------------------------------------

def test_unmatched_incidents_are_excluded_from_matched_latency():
    """An Incident outside every GT window is precision evidence, never latency evidence."""
    lat = [row("c1", (5.0, 10.0), 3.0), row("c1", None, 99.0)]
    out = LR.latency_summary(lat)
    assert out["legs"]["onset_to_incident_s"]["n"] == 1
    assert out["legs"]["onset_to_incident_s"]["max"] == 3.0
    assert out["matched_occurrences"] == 1


def test_missed_occurrences_stay_visible_in_the_denominator():
    """Latency is CONDITIONAL ON DETECTION -- the misses must remain readable beside it."""
    rows = [{"stages": {"stage4_cognition": {"gt_occurrences": 10}}}]
    # one occurrence found and alerted inside the gate; nine never found at all
    out = LR.latency_summary([row("c1", (5.0, 10.0), 1.0)], rows)
    assert out["gt_occurrences"] == 10
    assert out["matched_occurrences"] == 1
    assert out["missed_occurrences"] == 9
    assert out["timely_recall_all_gt"] == pytest.approx(0.1)   # 1/10 -- the honest number
    assert out["timely_rate_matched_only"] == pytest.approx(1.0)  # 1/1 -- flattering, and why both ship


def test_a_matched_but_late_occurrence_counts_in_neither_timely_rate():
    rows = [{"stages": {"stage4_cognition": {"gt_occurrences": 10}}}]
    out = LR.latency_summary([row("c1", (5.0, 10.0), 3.0)], rows)  # past the 2.0s gate
    assert out["matched_occurrences"] == 1
    assert out["timely_occurrences"] == 0
    assert out["timely_recall_all_gt"] == pytest.approx(0.0)
    assert out["timely_rate_matched_only"] == pytest.approx(0.0)
    assert out["legs"]["onset_to_incident_s"]["n"] == 1  # still in the latency population


def test_pre_onset_incidents_are_dropped_and_counted_not_silently_minimised():
    """A negative latency is a matcher artifact; min() would promote it to best-case."""
    lat = [row("c1", (5.0, 10.0), -2.0), row("c1", (5.0, 10.0), 4.0)]
    out = LR.latency_summary(lat)
    assert out["pre_onset_rows_dropped"] == 1
    assert out["legs"]["onset_to_incident_s"]["n"] == 1
    assert out["legs"]["onset_to_incident_s"]["p50"] == 4.0


# -- gate boundary -----------------------------------------------------------------------------

def test_exactly_the_gate_counts_as_timely_matching_quality():
    """quality() uses <=; latency_summary used < . They must agree on the boundary."""
    out = LR.latency_summary([row("c1", (5.0, 10.0), LR.ACTION_TO_INCIDENT_GATE_S)])
    assert out["timely_occurrences"] == 1
    assert out["n_under_gate"] == 1


def test_just_over_the_gate_is_not_timely():
    out = LR.latency_summary([row("c1", (5.0, 10.0), LR.ACTION_TO_INCIDENT_GATE_S + 1e-6)])
    assert out["timely_occurrences"] == 0


# -- breakdowns --------------------------------------------------------------------------------

def test_by_goal_kind_uses_earliest_rows():
    lat = [row("c1", (5.0, 40.0), t, kind="count") for t in (1.0, 30.0)]
    lat += [row("c2", (5.0, 40.0), t, kind="duration") for t in (2.0, 40.0)]
    out = LR.latency_summary(lat)
    assert out["by_goal_kind"]["count"]["n"] == 1
    assert out["by_goal_kind"]["count"]["p50"] == 1.0
    assert out["by_goal_kind"]["duration"]["p50"] == 2.0


def test_by_dataset_exists_and_uses_earliest_rows():
    lat = [row("c1", (5.0, 40.0), t, dataset="omnipro") for t in (3.0, 33.0)]
    out = LR.latency_summary(lat)
    assert out["by_dataset"]["omnipro"]["n"] == 1
    assert out["by_dataset"]["omnipro"]["p50"] == 3.0


def test_every_onset_anchored_leg_is_occurrence_aggregated():
    """onset_to_ingest and onset_to_admission share the defect -- fixing only one is half a fix."""
    lat = [
        row("c1", (5.0, 40.0), 1.0, onset_to_ingest_s=0.5, onset_to_admission_s=0.7),
        row("c1", (5.0, 40.0), 20.0, onset_to_ingest_s=19.0, onset_to_admission_s=19.5),
    ]
    out = LR.latency_summary(lat)
    for leg in LR.ONSET_ANCHORED_LEGS:
        assert out["legs"][leg]["n"] == 1, leg
    assert out["legs"]["onset_to_ingest_s"]["p50"] == 0.5
    assert out["legs"]["onset_to_admission_s"]["p50"] == 0.7


def test_onset_legs_come_from_the_same_emission_so_they_decompose_coherently():
    """The winning ROW wins, not a per-leg minimum -- otherwise the breakdown describes no
    delivery that ever happened."""
    lat = [
        row("c1", (5.0, 40.0), 1.0, onset_to_ingest_s=0.9),   # earliest incident, LATER ingest
        row("c1", (5.0, 40.0), 20.0, onset_to_ingest_s=0.1),  # earlier ingest, later incident
    ]
    out = LR.latency_summary(lat)
    assert out["legs"]["onset_to_incident_s"]["p50"] == 1.0
    assert out["legs"]["onset_to_ingest_s"]["p50"] == 0.9  # NOT 0.1


def test_per_call_legs_stay_weighted_per_emission():
    """One judge call is one genuine sample of provider latency."""
    lat = [row("c1", (5.0, 40.0), t, dispatch_to_verdict_s=d)
           for t, d in ((1.0, 1.0), (10.0, 2.0), (20.0, 3.0))]
    out = LR.latency_summary(lat)
    assert out["legs"]["dispatch_to_verdict_s"]["n"] == 3
    assert out["legs"]["onset_to_incident_s"]["n"] == 1


# -- audit trail + hygiene ---------------------------------------------------------------------

def test_raw_per_emission_diagnostics_remain_available_but_are_not_called_product_latency():
    lat = [row("c1", (5.0, 40.0), t) for t in (1.0, 10.0, 20.0)]
    out = LR.latency_summary(lat)
    raw = out["raw_incident_emissions"]
    assert raw["legs"]["onset_to_incident_s"]["n"] == 3
    assert raw["n_matched_emissions"] == 3
    assert raw["aggregation_unit"] == "incident_emission"
    assert "AUDIT ONLY" in raw["note"]
    assert "product latency" not in raw["note"].lower().replace("not product latency", "")


def test_schema_and_aggregation_unit_are_declared():
    out = LR.latency_summary([row("c1", (5.0, 10.0), 1.0)])
    assert out["schema"] == LR.LATENCY_SCHEMA
    assert out["aggregation_unit"]["onset_to_incident_s"] == "occurrence:earliest_incident"
    assert out["aggregation_unit"]["dispatch_to_verdict_s"] == "incident_emission"


def test_input_rows_are_not_mutated():
    lat = [row("c1", (5.0, 40.0), 1.0), row("c1", (5.0, 40.0), 9.0)]
    before = [dict(r) for r in lat]
    LR.latency_summary(lat)
    assert lat == before


def test_empty_input_is_safe():
    out = LR.latency_summary([])
    assert out["legs"]["onset_to_incident_s"]["n"] == 0
    assert out["matched_occurrences"] == 0
    assert out["timely_recall_all_gt"] is None


# -- the numerator must agree with quality() ----------------------------------------------------

def test_timely_count_agrees_with_quality_on_the_same_rows():
    """quality() and latency_summary() must describe the same population, or the report contradicts
    itself. Same key, same <= boundary, same dedup."""
    lat = [
        row("c1", (5.0, 40.0), 1.0), row("c1", (5.0, 40.0), 30.0),  # timely, repeated
        row("c1", (50.0, 60.0), 5.0),                                # matched, not timely
        row("c2", (5.0, 40.0), 2.0),                                 # timely, exactly at gate-ish
    ]
    rows = [{"stages": {"stage4_cognition": {
        "gt_occurrences": 5, "occurrences_matched": 3, "incidents": 4, "incidents_outside_gt": 0}}}]
    q = LR.quality(rows, lat)
    out = LR.latency_summary(lat, rows)
    assert out["timely_occurrences"] == q["occurrences_timely_2s"]
    assert out["timely_recall_all_gt"] == pytest.approx(q["timely_recall_2s"])
