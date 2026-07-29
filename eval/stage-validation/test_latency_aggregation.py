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

def case(case_id: str, *, media_end: float, gt: list[tuple[float, float]],
         incidents: list[float], gt_time: float | None = None, **extra) -> dict:
    """A minimal case record shaped like the ones ``run_case`` writes.

    ``latency_rows`` is the per-Incident record the scorers read; ``gt_intervals`` is the ground
    truth the case was scored against. Both are what a real artifact carries.
    """
    matched = [next(((a, b) for a, b in gt if a <= t < b), None) for t in incidents]
    rec = {
        "case_id": case_id,
        "dataset": "charades",
        "goal_kind": "duration",
        "measurable": True,
        "media_end_s": media_end,
        "gt_time_s": gt_time if gt_time is not None else sum(b - a for a, b in gt),
        "gt_intervals": [[a, b] for a, b in gt],
        "latency_rows": [
            {"case_id": case_id, "dataset": "charades", "goal_kind": "duration",
             "incident_media_ts": t, "matched_occurrence": list(m) if m else None,
             "onset_to_incident_s": (t - m[0]) if m else None}
            for t, m in zip(incidents, matched)
        ],
        "stages": {"stage4_cognition": {
            "gt_occurrences": len(gt),
            "occurrences_matched": sum(1 for a, b in gt if any(a <= t < b for t in incidents)),
            "incidents": len(incidents),
            "incidents_outside_gt": sum(1 for m in matched if m is None),
        }},
    }
    rec.update(extra)
    return rec


def test_timely_count_agrees_with_quality_on_the_same_rows():
    """quality() and latency_summary() must describe the same population, or the report contradicts
    itself. Same key, same <= boundary, same dedup."""
    lat = [
        row("c1", (5.0, 40.0), 1.0), row("c1", (5.0, 40.0), 30.0),  # timely, repeated
        row("c1", (50.0, 60.0), 5.0),                                # matched, not timely
        row("c2", (5.0, 40.0), 2.0),                                 # timely, exactly at gate-ish
    ]
    rows = [case("c1", media_end=100.0, gt=[(5.0, 40.0), (50.0, 60.0)], incidents=[6.0, 35.0, 55.0]),
            case("c2", media_end=100.0, gt=[(5.0, 40.0)], incidents=[7.0])]
    q = LR.quality(rows, lat)
    out = LR.latency_summary(lat, rows)
    assert out["timely_occurrences"] == q["occurrences_timely_2s"]
    assert out["timely_recall_all_gt"] == pytest.approx(q["timely_recall_2s"])


# ==============================================================================================
# PRECISION UNITS -- the defect: recall counted occurrences while precision counted emissions
# ==============================================================================================

def test_one_long_occurrence_weights_recall_once_and_emission_precision_many_times():
    """THE defect, in one case: 1 occurrence, 127 Incidents inside it, 1 outside.

    Per-emission precision sees 127 correct answers and 1 wrong one (0.992). Per-occurrence
    precision sees one occurrence found and one clean stretch of media wrongly flagged (0.500).
    Recall sees exactly one occurrence either way. The first number is not wrong, it is a different
    question -- and publishing only it beside an occurrence-weighted recall is what this pins.
    """
    rows = [case("c1", media_end=200.0, gt=[(10.0, 186.0)],
                 incidents=[10.0 + i for i in range(127)] + [195.0])]
    q = LR.quality(rows, [])
    assert q["occurrence_recall"] == 1.0
    assert q["incidents"] == 128
    assert q["precision_per_emission"] == pytest.approx(127 / 128)
    # two negative intervals ([0,10) and [186,200)), one of them flagged
    assert q["negative_intervals"] == 2
    assert q["negative_intervals_hit"] == 1
    assert q["precision_per_occurrence"] == pytest.approx(1 / 2)
    assert q["precision_per_emission_unit"] == "incident_emission"


def test_the_mislabelled_schema_1_key_is_gone_not_silently_repointed():
    """`occurrence_precision` was emission-weighted. Re-pointing the name at the corrected figure
    would make every historical artifact silently non-comparable under an unchanged key."""
    q = LR.quality([case("c1", media_end=10.0, gt=[(1.0, 2.0)], incidents=[1.5])], [])
    assert "occurrence_precision" not in q
    assert q["schema"] == LR.QUALITY_SCHEMA


def test_recall_and_occurrence_precision_share_a_denominator_convention():
    """Two occurrences, one long and one short, each found once: both weigh the same on both sides."""
    rows = [case("c1", media_end=300.0, gt=[(10.0, 200.0), (250.0, 251.0)],
                 incidents=[11.0] * 50 + [250.5])]
    q = LR.quality(rows, [])
    assert q["occurrences_matched"] == 2
    assert q["occurrence_recall"] == 1.0
    assert q["negative_intervals_hit"] == 0
    assert q["precision_per_occurrence"] == 1.0
    # ...while the emission figure is dominated by the long occurrence's 50 messages
    assert q["precision_per_emission"] == 1.0


# ==============================================================================================
# MATCHING TOLERANCE
# ==============================================================================================

def test_tolerance_is_zero_by_default_so_the_headline_is_the_strict_rule():
    assert LR.MATCH_TOLERANCE_S == 0.0
    assert 0.0 in LR.MATCH_TOLERANCE_SENSITIVITY_S


def test_an_incident_just_before_onset_matches_only_under_tolerance():
    rows = [case("c1", media_end=100.0, gt=[(10.0, 20.0)], incidents=[9.5])]
    assert LR.score_matching(rows, 0.0)["occurrences_matched"] == 0
    assert LR.score_matching(rows, 1.0)["occurrences_matched"] == 1


def test_tolerance_is_monotone_so_a_flattering_value_cannot_hide():
    """Padding an interval can only ADD matches. If this ever failed, the sensitivity table would
    not bound the effect of the choice and the default would stop being the conservative one."""
    rows = [case("c1", media_end=100.0, gt=[(10.0, 20.0), (60.0, 61.0)],
                 incidents=[9.6, 15.0, 30.0, 59.0, 80.0])]
    matched = [LR.score_matching(rows, t)["occurrences_matched"]
               for t in LR.MATCH_TOLERANCE_SENSITIVITY_S]
    assert matched == sorted(matched)
    inside = [LR.score_matching(rows, t)["incidents_inside_gt"]
              for t in LR.MATCH_TOLERANCE_SENSITIVITY_S]
    assert inside == sorted(inside)


def test_sensitivity_table_reports_every_declared_value():
    rows = [case("c1", media_end=100.0, gt=[(10.0, 20.0)], incidents=[9.5, 15.0])]
    values = LR.tolerance_sensitivity(rows)["values"]
    assert set(values) == {f"{t:g}" for t in LR.MATCH_TOLERANCE_SENSITIVITY_S}


# ==============================================================================================
# LATENESS -- an Incident 80 s into a 176 s occurrence is not the same as one at its onset
# ==============================================================================================

def test_lateness_is_measured_against_the_onset_not_the_interval():
    rows = [case("c1", media_end=200.0, gt=[(10.0, 186.0)], incidents=[90.0, 120.0])]
    q = LR.quality(rows, [])
    assert q["occurrence_recall"] == 1.0          # containment says "found it"
    assert q["lateness_media_s"]["max"] == pytest.approx(80.0)  # ...80 s after it started
    assert q["lateness_beyond_gate"] == 1
    assert q["lateness_beyond_10s"] == 1


def test_lateness_uses_the_first_matching_incident():
    rows = [case("c1", media_end=200.0, gt=[(10.0, 186.0)], incidents=[120.0, 12.0, 90.0])]
    q = LR.quality(rows, [])
    assert q["lateness_media_s"]["max"] == pytest.approx(2.0)


# ==============================================================================================
# CHANCE FLOOR
# ==============================================================================================

def test_full_coverage_ground_truth_has_a_chance_precision_of_one():
    """A case whose GT covers 100 % of its media gives an always-fire detector a perfect score.
    The chance floor must SAY so rather than letting 1.000 read as a result."""
    rows = [case("c1", media_end=30.0, gt=[(0.0, 30.0)], incidents=[1.0, 2.0, 3.0])]
    q = LR.quality(rows, [])
    assert q["precision_per_emission"] == 1.0
    assert q["chance"]["precision_per_emission"] == pytest.approx(1.0)
    assert q["lift"]["precision_per_emission"] is None  # no headroom exists to be captured


def test_lift_is_withheld_when_there_is_no_headroom_left_to_capture():
    """Near-total coverage makes lift a ratio of two vanishing quantities that reads 1.000 on a
    rounding difference. A cell that cannot discriminate must print no skill score at all."""
    rows = [case("c1", media_end=100.0, gt=[(0.0, 99.5)], incidents=[float(t) for t in range(90)])]
    q = LR.quality(rows, [])
    assert q["chance"]["occurrence_recall"] > 1.0 - LR.LIFT_MIN_HEADROOM
    assert q["lift"]["occurrence_recall"] is None


def test_lift_is_zero_when_the_detector_matches_the_null():
    """Half-covered media, emissions spread evenly: precision lands on the floor, lift ~0."""
    rows = [case("c1", media_end=100.0, gt=[(0.0, 50.0)],
                 incidents=[10.0, 20.0, 60.0, 70.0])]
    q = LR.quality(rows, [])
    assert q["precision_per_emission"] == pytest.approx(0.5)
    assert q["chance"]["precision_per_emission"] == pytest.approx(0.5)
    assert q["lift"]["precision_per_emission"] == pytest.approx(0.0)


def test_chance_recall_rises_with_the_firing_rate():
    """The null is RATE-MATCHED: a detector that fires more often is more likely to hit an
    occurrence by luck, and the floor it must beat has to rise with it."""
    sparse = [case("c1", media_end=100.0, gt=[(10.0, 20.0)], incidents=[15.0])]
    busy = [case("c1", media_end=100.0, gt=[(10.0, 20.0)],
                 incidents=[float(t) for t in range(0, 100, 2)])]
    assert (LR.score_matching(busy, 0.0)["chance"]["occurrence_recall"]
            > LR.score_matching(sparse, 0.0)["chance"]["occurrence_recall"])


# ==============================================================================================
# SATURATED CASES -- named, never averaged
# ==============================================================================================

def test_saturated_cases_are_named_with_their_coverage():
    rows = [case("sat", media_end=30.0, gt=[(0.0, 30.0)], incidents=[1.0]),
            case("ok", media_end=100.0, gt=[(10.0, 20.0)], incidents=[15.0])]
    sat = LR.saturation(rows)
    assert sat["n_saturated"] == 1
    assert [c["case_id"] for c in sat["saturated_cases"]] == ["sat"]
    assert sat["saturated_cases"][0]["gt_coverage"] == pytest.approx(1.0)
    assert sat["threshold"] == LR.SATURATED_COVERAGE


def test_coverage_is_measured_against_media_actually_seen():
    assert LR.coverage_of({"media_end_s": 40.0, "gt_time_s": 10.0}) == pytest.approx(0.25)
    assert LR.coverage_of({"media_end_s": None, "gt_time_s": 10.0}) is None


# ==============================================================================================
# GROUND TRUTH PROVENANCE -- a re-scoring must not silently adopt a newer corpus
# ==============================================================================================

def test_stored_gt_wins_over_the_corpus():
    rec = case("c1", media_end=100.0, gt=[(1.0, 2.0)], incidents=[])
    events, source = LR.gt_intervals_for(rec)
    assert source == "artifact"
    assert events == [(1.0, 2.0)]


def test_missing_gt_is_reported_not_guessed():
    events, source = LR.gt_intervals_for({"case_id": "not-in-any-corpus", "media_end_s": 10.0})
    assert events == []
    assert source in {"unavailable", "rederived_from_corpus"}


# ==============================================================================================
# PER-VERDICT PERSISTENCE
# ==============================================================================================

def test_verdict_schema_names_its_source_stage():
    assert LR.VERDICT_SCHEMA == "percept.verdicts/1-gate-classified"


# ==============================================================================================
# PROVENANCE -- both SHAs, and the corpus revision
# ==============================================================================================

def test_both_shas_and_the_corpus_version_are_stamped():
    hp = LR.harness_provenance()
    assert hp["scorer_sha256"] and len(hp["scorer_sha256"]) == 64
    assert hp["corpus_version"] in {"1.0", "1.1"} or hp["corpus_version"] == "unknown"
    assert hp["corpus_sha256"]


def test_scoring_parameters_carry_every_knob_the_score_depends_on():
    sp = LR.scoring_parameters()
    for key in ("quality_schema", "latency_schema", "verdict_schema", "match_tolerance_s",
                "saturated_coverage_threshold", "action_to_incident_gate_s"):
        assert key in sp


def test_measurement_provenance_keys_never_include_the_scorer():
    """`harness` describes the SCORER and must never be inherited from the measured artifact, or a
    re-scored file would claim the old scorer produced its new numbers."""
    assert "build_commit" in LR.MEASUREMENT_META_KEYS
    assert "scoring" not in LR.MEASUREMENT_META_KEYS
