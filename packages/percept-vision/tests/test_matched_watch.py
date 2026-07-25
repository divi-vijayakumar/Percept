"""L1 contract tests for the real compiled matched-Watch replay path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from percept.eval.matched_watch import (
    MatchedWatchInputError,
    derive_viability_rows,
    load_matched_watch_manifest,
    run_matched_watch,
    validate_matched_arms,
)


def _sha(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _verdict(satisfied: bool, evidence: str, *, error: str | None = None) -> dict:
    return {
        "satisfied": satisfied,
        "confidence": 0.99,
        "evidence": evidence,
        "error": error,
    }


def _documents(*, fills=(0, 255, 0, 255), fail_to_at: int | None = None):
    frames = []
    for index, fill in enumerate(fills):
        present = index >= 2
        frames.append(
            {
                "t": float(index),
                "fill": fill,
                "token": f"candidate-{index}",
                "from": _verdict(not present, "scripted: package absent"),
                "to": _verdict(
                    present,
                    "scripted: package visible",
                    error=("scripted timeout" if fail_to_at == index else None),
                ),
            }
        )
    media = {
        "schema": "percept.eval.matched_watch.synthetic_video/v1",
        "width": 8,
        "height": 8,
        "frames": frames,
    }
    return media


def _write_case(
    tmp_path,
    *,
    media=None,
    threshold=0.5,
    media_sha=None,
    event_start=2.0,
    event_end=3.0,
):
    media_doc = media or _documents()
    media_bytes = json.dumps(media_doc, sort_keys=True, separators=(",", ":")).encode()
    media_path = tmp_path / "media.json"
    media_path.write_bytes(media_bytes)
    manifest = {
        "schema": "percept.eval.matched_watch/v1",
        "case_id": "physical-package-001",
        "archetype": "physical_scene/package_appeared",
        "split": "dev",
        "media": {
            "path": "media.json",
            "sha256": media_sha or _sha(media_bytes),
        },
        "replay": {"fps": 4.0, "timebase": "media", "source_id": "camera"},
        "watch": {
            "template_ref": "facility_observation@1/package_appeared",
            "parameters": {"zone_name": "doorstep"},
            "domain_pack": "facility_observation@1",
        },
        "judge": {
            "adapter": "scripted_transition/v1",
            "model": "scripted-contract",
            "prompt_sha256": _sha(b"fixed transition prompt"),
        },
        "adaptive": {
            "operating_point": "block-max-0.5",
            "salience_stat": "block_max",
            "threshold": threshold,
        },
        "events": [
            {
                "id": "physical-package-001/e1",
                "eligible_at_s": 2.0,
                "window_start_s": event_start,
                "window_end_s": event_end,
                "review_authority": "fixture-authority/v1",
                "label_digest": _sha(b"package visible at t=2"),
            }
        ],
        "evidence_classification": "scripted_contract",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_full_compiled_transition_path_records_dispatches_and_subcalls(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    baseline, adaptive = derive_viability_rows(receipt)

    assert receipt.comparable is True
    assert receipt.launch_qualified is False
    assert (
        receipt.baseline.compiled_binding["watchspec_kwargs"]["stat_thresholds"]["block_max"]
        == 0.0
    )
    assert receipt.adaptive.compiled_binding["watchspec_kwargs"]["salience_stat"] == "block_max"
    assert (
        receipt.adaptive.compiled_binding["watchspec_kwargs"]["stat_thresholds"]["block_max"]
        == 0.5
    )
    assert baseline["candidate_windows_admitted"] == 4
    assert baseline["runtime_dispatches"] == 4
    assert baseline["judge_calls"] == 8
    assert adaptive["candidate_windows_admitted"] == 3
    assert adaptive["runtime_dispatches"] == 3
    assert adaptive["judge_calls"] == 6
    assert adaptive["decisions"] == {"fire": 1, "ask": 0, "silent": 2, "missing": 0}
    assert adaptive["fire_event_ids"] == ["physical-package-001/e1"]
    assert len(receipt.adaptive.incidents) == 1
    assert [item["stage"] for item in receipt.adaptive.traces].count("edge.decision") == 4
    assert [item["stage"] for item in receipt.adaptive.traces].count("transition.decision") == 3
    assert all(
        [call.predicate for call in candidate.provider_calls] == ["from", "to"]
        for candidate in receipt.adaptive.candidates
        if candidate.runtime_dispatched
    )


def test_sustained_true_fires_once_and_boundary_candidates_are_single_owned(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path, event_start=2.0, event_end=3.0))
    _, adaptive = derive_viability_rows(receipt)

    assert len(receipt.adaptive.incidents) == 1
    assert receipt.adaptive.incidents[0]["ts"] == 2.0
    assert adaptive["event_windows"] == [
        {
            "id": "physical-package-001/e1",
            "candidate_windows": 2,
            "admitted_candidates": 2,
            "outcome": "fire",
        }
    ]


def test_adaptive_zero_admission_and_miss_are_preserved(tmp_path):
    media = _documents(fills=(0, 0, 0, 0))
    receipt = run_matched_watch(_write_case(tmp_path, media=media, threshold=0.5))
    baseline, adaptive = derive_viability_rows(receipt)

    assert baseline["candidate_windows_admitted"] == 4
    assert adaptive["candidate_windows_admitted"] == 0
    assert adaptive["runtime_dispatches"] == 0
    assert adaptive["judge_calls"] == 0
    assert adaptive["decisions"] == {"fire": 0, "ask": 0, "silent": 0, "missing": 0}
    assert adaptive["event_windows"][0]["outcome"] == "silent"


def test_second_transition_subcall_failure_is_costed_and_never_partial_success(tmp_path):
    receipt = run_matched_watch(
        _write_case(tmp_path, media=_documents(fail_to_at=2))
    )
    candidate = receipt.adaptive.candidates[2]

    assert [call.status for call in candidate.provider_calls] == ["success", "error"]
    assert candidate.decision == "missing"
    assert candidate.incident_ids == ()
    assert receipt.adaptive.provider_call_attempts == 6
    assert receipt.adaptive.provider_call_failures == 1
    assert receipt.comparable is False
    with pytest.raises(MatchedWatchInputError, match="not comparable"):
        derive_viability_rows(receipt)


def test_comparability_rejects_timestamp_and_unapproved_config_mismatches(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    changed_timestamps = replace(
        receipt.adaptive,
        candidate_timestamps=(0.0, 1.0, 2.0, 4.0),
    )
    assert "candidate timestamp sequences differ" in validate_matched_arms(
        receipt.baseline, changed_timestamps
    )

    changed_resolved = dict(receipt.adaptive.resolved_watch)
    changed_effective = dict(changed_resolved["effective"])
    changed_effective["Settings.gate.threshold"] = 0.123
    changed_resolved["effective"] = changed_effective
    changed_config = replace(receipt.adaptive, resolved_watch=changed_resolved)
    assert "arms differ outside approved admission configuration" in validate_matched_arms(
        receipt.baseline, changed_config
    )


def test_canonical_content_excludes_volatile_timestamp(tmp_path):
    manifest = _write_case(tmp_path)
    first = run_matched_watch(manifest)
    second = run_matched_watch(manifest)

    assert first.content_sha256 == second.content_sha256
    assert first.hash_content_dict() == second.hash_content_dict()
    assert "run_timestamp" not in first.content_dict()


def test_manifest_and_media_fail_closed_at_safe_input_boundary(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(MatchedWatchInputError, match="unsafe matched-Watch manifest"):
        load_matched_watch_manifest(missing)

    manifest = _write_case(tmp_path, media_sha="sha256:" + "0" * 64)
    with pytest.raises(MatchedWatchInputError, match="media digest"):
        load_matched_watch_manifest(manifest)


def test_nonmonotonic_media_and_overlapping_event_ownership_fail_closed(tmp_path):
    media = _documents()
    media["frames"][2]["t"] = 1.0
    with pytest.raises(MatchedWatchInputError, match="strictly increasing"):
        load_matched_watch_manifest(_write_case(tmp_path, media=media))
