from __future__ import annotations

from dataclasses import replace

import pytest

from percept.eval.matched_watch import (
    MatchedWatchInputError,
    derive_viability_rows,
    run_matched_watch,
)
from tests.test_matched_watch import _write_case
from tests.test_matched_watch import _documents


def test_derivation_revalidates_config_instead_of_trusting_comparable_flag(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    resolved = dict(receipt.adaptive.resolved_watch)
    effective = dict(resolved["effective"])
    effective["Settings.gate.threshold"] = 0.987
    resolved["effective"] = effective
    adaptive = replace(receipt.adaptive, resolved_watch=resolved)
    tampered = replace(
        receipt,
        adaptive=adaptive,
        comparable=True,
        comparability_blockers=(),
    )

    with pytest.raises(MatchedWatchInputError, match="configuration|comparable|matched"):
        derive_viability_rows(tampered)


def test_derivation_rejects_media_identity_tamper(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    media_receipt = dict(receipt.media_input_receipt)
    media_receipt["sha256"] = "sha256:" + "0" * 64
    tampered = replace(receipt, media_input_receipt=media_receipt)

    with pytest.raises(MatchedWatchInputError, match="media"):
        derive_viability_rows(tampered)


def test_derivation_rejects_incident_candidate_disagreement(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    adaptive = replace(receipt.adaptive, incidents=())
    tampered = replace(
        receipt,
        adaptive=adaptive,
        comparable=True,
        comparability_blockers=(),
    )

    with pytest.raises(MatchedWatchInputError, match="incident|decision|FIRE"):
        derive_viability_rows(tampered)


def test_derivation_rejects_runtime_stats_tamper(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    runtime_stats = dict(receipt.adaptive.runtime_stats)
    runtime_stats["judge_calls"] = 0
    runtime_stats["perceive_dispatches"] = 0
    adaptive = replace(receipt.adaptive, runtime_stats=runtime_stats)
    tampered = replace(
        receipt,
        adaptive=adaptive,
        comparable=True,
        comparability_blockers=(),
    )

    with pytest.raises(MatchedWatchInputError, match="RuntimeStats|dispatch|runtime"):
        derive_viability_rows(tampered)


def test_first_transition_subcall_failure_is_costed_and_cannot_fire(tmp_path):
    media = _documents()
    media["frames"][2]["from"]["error"] = "scripted from timeout"
    receipt = run_matched_watch(_write_case(tmp_path, media=media))
    candidate = receipt.adaptive.candidates[2]

    assert [(call.predicate, call.status) for call in candidate.provider_calls] == [
        ("from", "error")
    ]
    assert candidate.decision == "missing"
    assert candidate.incident_ids == ()
    assert receipt.adaptive.provider_call_attempts == 5
    assert receipt.adaptive.provider_call_failures == 1
    assert receipt.comparable is False


def test_joined_in_to_state_never_fires_without_observed_from(tmp_path):
    media = _documents()
    for frame in media["frames"]:
        frame["from"]["satisfied"] = False
        frame["to"]["satisfied"] = True

    receipt = run_matched_watch(_write_case(tmp_path, media=media))

    assert receipt.baseline.incidents == ()
    assert receipt.adaptive.incidents == ()
    assert all(candidate.incident_ids == () for candidate in receipt.adaptive.candidates)


def test_low_confidence_to_does_not_become_a_fire(tmp_path):
    media = _documents()
    media["frames"][2]["to"]["confidence"] = 0.2

    receipt = run_matched_watch(_write_case(tmp_path, media=media))

    assert not any(
        incident["ts"] == 2.0 for incident in receipt.baseline.incidents
    )
    assert not any(
        incident["ts"] == 2.0 for incident in receipt.adaptive.incidents
    )


def test_receipt_nested_documents_are_immutable(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))

    with pytest.raises(TypeError):
        receipt.manifest["adaptive"]["threshold"] = 0.0
    with pytest.raises(TypeError):
        receipt.adaptive.runtime_stats["judge_calls"] = 0
    with pytest.raises(TypeError):
        receipt.adaptive.candidates[0].edge_decision["goal_ids"] = ()
