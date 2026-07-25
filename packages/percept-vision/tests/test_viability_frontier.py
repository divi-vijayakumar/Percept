"""Adversarial contract tests for the matched viability-frontier evaluator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from percept.eval.viability_frontier import (
    ViabilityInputError,
    evaluate_viability_frontier,
    load_viability_evidence,
    load_viability_manifest,
    main,
    render_text,
)

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "packages" / "percept-vision" / "tests" / "fixtures" / "viability_frontier"
MANIFEST = FIXTURE / "manifest.json"
EVIDENCE = FIXTURE / "observations.jsonl"


@pytest.fixture()
def manifest() -> dict:
    return load_viability_manifest(MANIFEST)


@pytest.fixture()
def rows() -> list[dict]:
    return load_viability_evidence(EVIDENCE)


def _row(rows: list[dict], case: str, point: str, arm: str) -> dict:
    return next(
        row
        for row in rows
        if (row["case_id"], row["operating_point"], row["arm"]) == (case, point, arm)
    )


def test_manifest_derived_frontier_has_exact_denominators(manifest, rows):
    report = evaluate_viability_frontier(manifest, rows)

    assert report.policy_cleared is True
    assert report.launch_qualified is False
    assert report.evidence_kind == "synthetic"
    assert len(report.archetypes) == len(manifest["archetypes"]) == 2

    state = report.archetype("state_change")
    assert len(state.case_ids) == len(manifest["archetypes"][0]["cases"]) == 2
    conservative, sparse = state.operating_points
    assert conservative.adaptive.candidate_recall.to_dict() == {
        "numerator": 4,
        "denominator": 8,
        "value": 0.5,
    }
    assert conservative.adaptive.event_recall.to_dict() == {
        "numerator": 4,
        "denominator": 4,
        "value": 1.0,
    }
    assert conservative.adaptive.admitted_rate.to_dict() == {
        "numerator": 8,
        "denominator": 16,
        "value": 0.5,
    }
    assert conservative.adaptive.judge_call_rate.to_dict() == {
        "numerator": 8,
        "denominator": 16,
        "value": 0.5,
    }
    assert conservative.call_reduction.to_dict() == {
        "numerator": 8,
        "denominator": 16,
        "value": 0.5,
    }
    assert sparse.adaptive.event_recall.value == pytest.approx(0.75)
    assert sparse.call_reduction.value == pytest.approx(0.75)


def test_policies_are_per_archetype_and_select_different_points(manifest, rows):
    report = evaluate_viability_frontier(manifest, rows)
    state = report.archetype("state_change")
    brief = report.archetype("brief_action")

    assert state.policy != brief.policy
    assert [point.id for point in state.operating_points if point.clears_policy] == ["sparse"]
    assert [point.id for point in brief.operating_points if point.clears_policy] == ["conservative"]


def test_pareto_frontier_marks_a_dominated_point(manifest, rows):
    mutated = copy.deepcopy(rows)
    for case_id in ("state-01", "state-02"):
        row = _row(mutated, case_id, "conservative", "adaptive")
        row["candidate_windows_admitted"] = 6
        row["judge_calls"] = 6
        row["decisions"] = {"fire": 2, "ask": 0, "silent": 4, "missing": 0}
    state_02 = _row(mutated, "state-02", "conservative", "adaptive")
    state_02["decisions"] = {"fire": 1, "ask": 0, "silent": 5, "missing": 0}
    state_02["event_windows"][1]["outcome"] = "silent"

    state = evaluate_viability_frontier(manifest, mutated).archetype("state_change")
    assert [point.id for point in state.frontier] == ["sparse"]
    assert state.operating_points[0].pareto_frontier is False


def test_abstentions_and_missing_work_are_reported_not_erased(manifest, rows):
    mutated = copy.deepcopy(rows)
    row = _row(mutated, "brief-01", "conservative", "adaptive")
    row["judge_calls"] = 3
    row["decisions"] = {"fire": 1, "ask": 1, "silent": 0, "missing": 1}
    row["event_windows"][1]["outcome"] = "ask"

    point = evaluate_viability_frontier(manifest, mutated).archetype(
        "brief_action"
    ).operating_points[0]
    # One admitted candidate was never judged and one judge call produced no decision.
    assert point.adaptive.execution_missing_rate.numerator == 2
    assert point.adaptive.execution_missing_rate.denominator == 8
    assert point.adaptive.event_missing_rate.numerator == 0
    assert point.adaptive.decision_abstention_rate.numerator == 3
    assert point.adaptive.decision_abstention_rate.denominator == 7
    assert point.adaptive.event_abstention_rate.numerator == 1
    assert point.clears_policy is False


def test_missing_event_window_is_separate_from_missing_execution(manifest, rows):
    row = _row(rows, "brief-01", "sparse", "adaptive")
    row["event_windows"][1]["outcome"] = "missing"

    point = evaluate_viability_frontier(manifest, rows).archetype(
        "brief_action"
    ).operating_points[1]
    assert point.adaptive.event_missing_rate.to_dict() == {
        "numerator": 1,
        "denominator": 4,
        "value": 0.25,
    }
    assert point.adaptive.execution_missing_rate.value == 0.0
    assert any("event_missing_rate" in failure for failure in point.policy_failures)


def test_synthetic_can_clear_policy_but_never_qualify_launch(manifest, rows):
    report = evaluate_viability_frontier(manifest, rows)
    assert report.policy_cleared is True
    assert report.launch_qualified is False
    assert "synthetic" in report.qualification_reason


def test_render_is_scoped_and_contains_no_cost_or_backend_claim(manifest, rows):
    text = render_text(evaluate_viability_frontier(manifest, rows))
    assert "programmable Watches over live multimodal streams" in text
    assert "Manifest-derived coverage: 2 archetype(s), 4 case(s), 4 operating point(s)" in text
    assert "candidate_recall=0.500 (4/8)" in text
    assert "call_reduction=0.750 (12/16)" in text
    for forbidden in ("$", "provider", "api key", "universal threshold"):
        assert forbidden not in text.lower()


@pytest.mark.parametrize("kind", ["modeled", "stub"])
def test_manifest_refuses_modeled_or_stub_data(manifest, rows, kind):
    manifest["evidence_kind"] = kind
    with pytest.raises(ViabilityInputError, match="modeled and stub data"):
        evaluate_viability_frontier(manifest, rows)


@pytest.mark.parametrize("kind", ["modeled", "stub"])
def test_rows_refuse_modeled_or_stub_data(manifest, rows, kind):
    rows[0]["evidence_kind"] = kind
    with pytest.raises(ViabilityInputError, match="modeled and stub data"):
        evaluate_viability_frontier(manifest, rows)


def test_empty_manifest_archetypes_fail_closed(manifest, rows):
    manifest["archetypes"] = []
    with pytest.raises(ViabilityInputError, match="archetypes must be non-empty"):
        evaluate_viability_frontier(manifest, rows)


def test_empty_archetype_cases_fail_closed(manifest, rows):
    manifest["archetypes"][0]["cases"] = []
    with pytest.raises(ViabilityInputError, match="at least 1"):
        evaluate_viability_frontier(manifest, rows)


def test_fewer_than_two_operating_points_fail_closed(manifest, rows):
    manifest["archetypes"][0]["operating_points"] = ["only"]
    with pytest.raises(ViabilityInputError, match="at least 2"):
        evaluate_viability_frontier(manifest, rows)


@pytest.mark.parametrize("field", sorted({
    "min_candidate_recall",
    "min_event_recall",
    "min_call_reduction",
    "max_event_abstention_rate",
    "max_missing_rate",
}))
def test_every_policy_field_is_required_with_no_default(manifest, rows, field):
    manifest["archetypes"][0]["policy"].pop(field)
    with pytest.raises(ViabilityInputError, match="missing required fields"):
        evaluate_viability_frontier(manifest, rows)


def test_policy_must_be_preregistered(manifest, rows):
    manifest["preregistered"] = False
    with pytest.raises(ViabilityInputError, match="preregistered must be true"):
        evaluate_viability_frontier(manifest, rows)


def test_duplicate_archetype_fails_closed(manifest, rows):
    manifest["archetypes"].append(copy.deepcopy(manifest["archetypes"][0]))
    with pytest.raises(ViabilityInputError, match="duplicate archetype"):
        evaluate_viability_frontier(manifest, rows)


def test_duplicate_case_fails_closed(manifest, rows):
    manifest["archetypes"][0]["cases"].append(manifest["archetypes"][0]["cases"][0])
    with pytest.raises(ViabilityInputError, match="duplicate ids"):
        evaluate_viability_frontier(manifest, rows)


def test_duplicate_operating_point_fails_closed(manifest, rows):
    manifest["archetypes"][0]["operating_points"][1] = "conservative"
    with pytest.raises(ViabilityInputError, match="duplicate ids"):
        evaluate_viability_frontier(manifest, rows)


def test_empty_evidence_fails_closed(manifest):
    with pytest.raises(ViabilityInputError, match="must be non-empty"):
        evaluate_viability_frontier(manifest, [])


def test_duplicate_evidence_row_fails_closed(manifest, rows):
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ViabilityInputError, match="duplicate evidence row"):
        evaluate_viability_frontier(manifest, rows)


@pytest.mark.parametrize("arm", ["baseline", "adaptive"])
def test_missing_arm_fails_closed(manifest, rows, arm):
    rows[:] = [
        row
        for row in rows
        if not (
            row["case_id"] == "state-01"
            and row["operating_point"] == "conservative"
            and row["arm"] == arm
        )
    ]
    with pytest.raises(ViabilityInputError, match="missing baseline/adaptive"):
        evaluate_viability_frontier(manifest, rows)


def test_case_mismatch_fails_closed(manifest, rows):
    rows[0]["case_id"] = "not-preregistered"
    with pytest.raises(ViabilityInputError, match="missing baseline/adaptive|mismatch"):
        evaluate_viability_frontier(manifest, rows)


def test_archetype_mismatch_fails_closed(manifest, rows):
    rows[0]["archetype"] = "brief_action"
    with pytest.raises(ViabilityInputError, match="belongs to archetype"):
        evaluate_viability_frontier(manifest, rows)


def test_event_window_mismatch_between_arms_fails_closed(manifest, rows):
    row = _row(rows, "state-01", "conservative", "adaptive")
    row["event_windows"][0]["id"] = "different-event"
    with pytest.raises(ViabilityInputError, match="exact candidate/event-window denominators"):
        evaluate_viability_frontier(manifest, rows)


def test_candidate_denominator_mismatch_between_arms_fails_closed(manifest, rows):
    row = _row(rows, "state-01", "conservative", "adaptive")
    row["candidate_windows_total"] = 9
    with pytest.raises(ViabilityInputError, match="exact candidate/event-window denominators"):
        evaluate_viability_frontier(manifest, rows)


def test_baseline_must_be_identical_across_operating_points(manifest, rows):
    row = _row(rows, "state-01", "sparse", "baseline")
    row["candidate_windows_admitted"] = 7
    row["judge_calls"] = 7
    row["decisions"] = {"fire": 2, "ask": 0, "silent": 5, "missing": 0}
    with pytest.raises(ViabilityInputError, match="baseline arm changed"):
        evaluate_viability_frontier(manifest, rows)


def test_missing_denominator_field_fails_closed(manifest, rows):
    rows[0].pop("candidate_windows_total")
    with pytest.raises(ViabilityInputError, match="missing required fields"):
        evaluate_viability_frontier(manifest, rows)


def test_incomplete_decision_denominator_fails_closed(manifest, rows):
    rows[0]["decisions"]["silent"] -= 1
    with pytest.raises(ViabilityInputError, match="decision counts sum"):
        evaluate_viability_frontier(manifest, rows)


def test_event_recall_cannot_exceed_fire_decisions(manifest, rows):
    rows[0]["decisions"] = {"fire": 0, "ask": 0, "silent": 8, "missing": 0}
    with pytest.raises(ViabilityInputError, match="FIRE outcomes exceed FIRE decisions"):
        evaluate_viability_frontier(manifest, rows)


def test_duplicate_event_window_fails_closed(manifest, rows):
    rows[0]["event_windows"].append(copy.deepcopy(rows[0]["event_windows"][0]))
    with pytest.raises(ViabilityInputError, match="duplicate event-window"):
        evaluate_viability_frontier(manifest, rows)


def test_evidence_kind_must_match_manifest(manifest, rows):
    rows[0]["evidence_kind"] = "measured"
    with pytest.raises(ViabilityInputError, match="does not match manifest"):
        evaluate_viability_frontier(manifest, rows)


def test_negative_call_reduction_is_reported_and_fails_policy(manifest, rows):
    mutated = copy.deepcopy(rows)
    # Lower the repeated matched baseline to three calls per case at one point while leaving the
    # adaptive arm at four. Apply the same baseline to both operating points to preserve matching.
    for point in ("conservative", "sparse"):
        for case_id in ("state-01", "state-02"):
            baseline = _row(mutated, case_id, point, "baseline")
            baseline["candidate_windows_admitted"] = 3
            baseline["judge_calls"] = 3
            baseline["decisions"] = {"fire": 2, "ask": 0, "silent": 1, "missing": 0}
            for event in baseline["event_windows"]:
                event["admitted_candidates"] = 1
    # Sparse must stay within its baseline; conservative intentionally exceeds it.
    state = evaluate_viability_frontier(manifest, mutated).archetype("state_change")
    conservative = state.operating_points[0]
    assert conservative.call_reduction.numerator == -2
    assert conservative.call_reduction.denominator == 6
    assert conservative.call_reduction.value == pytest.approx(-1 / 3)
    assert conservative.clears_policy is False


def test_loaders_refuse_empty_evidence_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ViabilityInputError, match="evidence is empty"):
        load_viability_evidence(empty)


def test_json_report_counts_come_from_manifest(manifest, rows):
    payload = json.loads(evaluate_viability_frontier(manifest, rows).to_json())
    assert payload["archetype_count"] == len(manifest["archetypes"])
    assert payload["case_count"] == sum(len(item["cases"]) for item in manifest["archetypes"])
    assert payload["operating_point_count"] == sum(
        len(item["operating_points"]) for item in manifest["archetypes"]
    )


def test_cli_can_validate_synthetic_mechanics_but_launch_gate_refuses(capsys):
    args = ["--manifest", str(MANIFEST), "--evidence", str(EVIDENCE)]
    assert main(args) == 0
    assert "LAUNCH QUALIFIED: FALSE" in capsys.readouterr().out
    assert main([*args, "--require-launch-qualification"]) == 1
