"""Real-MP4 mechanics for the existing matched-Watch evaluator.

The generated clips and offline-oracle answers are deterministic contract fixtures. They establish
decoder/runtime/receipt mechanics only and can never qualify pack accuracy, latency, or cost.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import percept.eval.matched_watch as matched_watch_module
from percept.domain_pack import watch_template
from percept.eval.matched_watch import (
    MatchedWatchInputError,
    derive_viability_rows,
    load_matched_watch_manifest,
    load_matched_watch_receipt,
    run_matched_watch,
)
from percept.eval.measured_evidence import derive_measured_evidence_bundle
from percept.runtime.source import open_source
from percept.tracing import content_hash


def _sha(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verdict(satisfied: bool, evidence: str) -> dict[str, object]:
    return {
        "satisfied": satisfied,
        "confidence": 0.99,
        "evidence": evidence,
        "error": None,
    }


def _reseal(receipt, **changes):
    changed = replace(receipt, origin_evidence_sha256="", **changes)
    return replace(changed, origin_evidence_sha256=changed.evidence_sha256)


def _write_mp4(
    path: Path,
    values: tuple[int, ...] = (8, 30, 230, 232, 25, 27),
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        (32, 24),
    )
    assert writer.isOpened(), "test environment cannot create the required MP4 mechanics fixture"
    for value in values:
        frame = np.full((24, 32, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.is_file() and path.stat().st_size > 0


async def _decoded_candidates(path: Path) -> list[dict[str, object]]:
    result = []
    async for envelope in open_source(
        str(path),
        fps=4.0,
        maxsize=64,
        reconnects=0,
        timebase="media",
    ):
        mime, payload = envelope.payload
        assert mime == "image/jpeg"
        result.append(
            {
                "sequence": envelope.sequence,
                "t": envelope.event_time,
                "frame_sha256": _sha(payload),
            }
        )
    return result


#: Which decoded frame a TWO-event case stages its first event on. At the fixture's 4fps replay,
#: index 1 (t=0.25s) puts it 0.75s before the second event at index 4 (t=1.0s) — outside the shipped
#: 0.7s delivery resolution, so the two events are separately resolvable notifications rather than
#: one occurrence re-reported. See the note at the ``events`` block below for why the fixture moved
#: and the calibrated window did not.
_FIRST_EVENT_INDEX_TWO = 1


def _write_real_case(
    root: Path,
    *,
    template_ref: str = "facility_observation@1/package_appeared",
    archetype: str = "physical_scene/package_appeared",
    case_id: str = "real-case-001",
    source_group_id: str = "generated-session-001",
    operating_point: str = "block-max-0.20",
    threshold: float = 0.20,
    frame_values: tuple[int, ...] = (8, 30, 230, 232, 25, 27),
    states: tuple[bool, ...] = (False, False, True, True, False, False),
    two_events: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    media_path = root / "case.mp4"
    _write_mp4(media_path, frame_values)
    decoded = asyncio.run(_decoded_candidates(media_path))
    assert len(decoded) == len(states)
    plan = watch_template(template_ref, {"zone_name": "receiving"})
    pack_name = template_ref.partition("/")[0]
    route = str(plan.explicit.get("Goal.state.type", "watch"))
    if route == "transition":
        predicates = (
            (
                f"sha256:{content_hash(plan.explicit['Goal.state.from'])}",
                tuple(not state for state in states),
                "from",
            ),
            (
                f"sha256:{content_hash(plan.explicit['Goal.state.to'])}",
                states,
                "to",
            ),
        )
    else:
        predicates = (
            (
                f"sha256:{content_hash(plan.condition)}",
                states,
                "condition",
            ),
        )
    oracle_candidates = []
    for item in decoded:
        index = int(item["sequence"])
        oracle_candidates.append(
            {
                **item,
                "token": f"decoded-candidate-{index}",
                "verdicts": {
                    condition_sha: _verdict(
                        states[index],
                        f"offline oracle {label} candidate {index}",
                    )
                    for condition_sha, states, label in predicates
                },
            }
        )
    oracle = {
        "schema": "percept.eval.matched_watch.offline_oracle/v1",
        "candidates": oracle_candidates,
    }
    oracle_path = root / "oracle.json"
    oracle_path.write_bytes(_json_bytes(oracle))
    media_sha = _sha(media_path.read_bytes())
    oracle_sha = _sha(oracle_path.read_bytes())
    corpus = {
        "schema": "percept.eval.matched_watch.corpus/v1",
        "corpus_id": "generated-real-media-mechanics",
        "version": "1",
        "cases": [
            {
                "case_id": case_id,
                "split": "mechanics",
                "source_group_id": source_group_id,
                "media_sha256": media_sha,
                "oracle_sha256": oracle_sha,
            }
        ],
        "authorities": {
            "license": None,
            "label_review": None,
            "provider_transmission": None,
            "preregistration": None,
        },
    }
    corpus_path = root / "corpus.json"
    corpus_path.write_bytes(_json_bytes(corpus))
    manifest = {
        "schema": "percept.eval.matched_watch.real_video/v1",
        "case_id": case_id,
        "archetype": archetype,
        "split": "mechanics",
        "media": {
            "path": "case.mp4",
            "sha256": media_sha,
            "media_type": "video/mp4",
        },
        "replay": {"fps": 4.0, "timebase": "media", "source_id": "camera"},
        "watch": {
            "template_ref": template_ref,
            "parameters": {"zone_name": "receiving"},
            "domain_pack": pack_name,
        },
        "judge": {
            "adapter": "offline_oracle/v1",
            "model": "deterministic-test-oracle",
            "prompt_sha256": _sha(b"fixed mechanics oracle contract"),
            "judge_mode": "offline_oracle",
        },
        "adaptive": {
            "operating_point": operating_point,
            "salience_stat": "block_max",
            "threshold": threshold,
        },
        # Two-event cases stage event-1 one frame EARLIER (index 1, not 2). At the fixture's 4fps
        # replay that puts the two events 0.75s apart instead of 0.5s.
        #
        # Why: 0.5s is INSIDE the shipped delivery resolution (percept.behavior
        # INCIDENT_MERGE_WINDOW_S = 0.7s), so the fixture was asserting that the product resolves two
        # separate notifications closer together than the product claims to. Nothing downstream ever
        # noticed, because the transition route used to return its fire around the delivery seam
        # entirely; now that it goes through the same seam as every other sense, the fixture's own
        # spacing is what fails. The window is NOT the thing to change here — 0.7s is calibrated
        # (it clears a measured rep-to-rep floor and catches the judgment strides, see the constant's
        # docstring) whereas transition event spacing has never been measured, so widening a
        # calibrated constant to fit an arbitrary fixture would trade a real number for an invented
        # one. The fixture moves instead. Single-event cases are untouched (index 2), and the
        # adaptive arm's admitted frames and incident count are unchanged either way.
        "events": [
            {
                "id": f"{case_id}/event-1",
                "eligible_at_s": decoded[_FIRST_EVENT_INDEX_TWO if two_events else 2]["t"],
                "window_start_s": decoded[_FIRST_EVENT_INDEX_TWO if two_events else 2]["t"],
                "window_end_s": (
                    decoded[_FIRST_EVENT_INDEX_TWO]["t"] if two_events else decoded[3]["t"]
                ),
                "review_authority": "generated-fixture-author",
                "label_digest": _sha(b"mechanics-only positive interval"),
            }
        ]
        + (
            [
                {
                    "id": f"{case_id}/event-2",
                    "eligible_at_s": decoded[4]["t"],
                    "window_start_s": decoded[4]["t"],
                    "window_end_s": decoded[5]["t"],
                    "review_authority": "generated-fixture-author",
                    "label_digest": _sha(b"second mechanics-only positive interval"),
                }
            ]
            if two_events
            else []
        ),
        "evidence_classification": "admission_only",
        "corpus": {"path": "corpus.json", "sha256": _sha(corpus_path.read_bytes())},
        "oracle": {"path": "oracle.json", "sha256": oracle_sha},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest_path


def _set_oracle_error(manifest_path: Path, *, index: int, predicate: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = watch_template(
        manifest["watch"]["template_ref"],
        manifest["watch"]["parameters"],
    )
    condition = plan.explicit[f"Goal.state.{predicate}"]
    condition_sha = f"sha256:{content_hash(condition)}"
    oracle_path = manifest_path.parent / manifest["oracle"]["path"]
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle["candidates"][index]["verdicts"][condition_sha]["error"] = (
        f"offline oracle {predicate} failure"
    )
    oracle_path.write_bytes(_json_bytes(oracle))
    oracle_sha = _sha(oracle_path.read_bytes())
    corpus_path = manifest_path.parent / manifest["corpus"]["path"]
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["cases"][0]["oracle_sha256"] = oracle_sha
    corpus_path.write_bytes(_json_bytes(corpus))
    manifest["oracle"]["sha256"] = oracle_sha
    manifest["corpus"]["sha256"] = _sha(corpus_path.read_bytes())
    manifest_path.write_bytes(_json_bytes(manifest))


def _set_ordinary_oracle_error(manifest_path: Path, *, index: int) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    oracle_path = manifest_path.parent / manifest["oracle"]["path"]
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    condition_sha = next(iter(oracle["candidates"][index]["verdicts"]))
    oracle["candidates"][index]["verdicts"][condition_sha]["error"] = (
        "offline ordinary-Watch failure"
    )
    oracle_path.write_bytes(_json_bytes(oracle))
    oracle_sha = _sha(oracle_path.read_bytes())
    corpus_path = manifest_path.parent / manifest["corpus"]["path"]
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["cases"][0]["oracle_sha256"] = oracle_sha
    corpus_path.write_bytes(_json_bytes(corpus))
    manifest["oracle"]["sha256"] = oracle_sha
    manifest["corpus"]["sha256"] = _sha(corpus_path.read_bytes())
    manifest_path.write_bytes(_json_bytes(manifest))


def test_real_transition_decodes_actual_mp4_and_executes_both_arms(tmp_path):
    manifest_path = _write_real_case(tmp_path)

    receipt = run_matched_watch(manifest_path)
    baseline, adaptive = derive_viability_rows(receipt)

    assert receipt.receipt_schema == "percept.eval.matched_watch.receipt/v2"
    assert receipt.judge_identity["judge_mode"] == "offline_oracle"
    assert receipt.launch_qualified is False
    assert receipt.authority_artifact_presence == {
        "license": False,
        "label_review": False,
        "provider_transmission": False,
        "preregistration": False,
    }
    assert baseline["candidate_windows_total"] == 6
    assert baseline["runtime_dispatches"] == 6
    assert baseline["judge_calls"] == 12
    assert adaptive["runtime_dispatches"] <= baseline["runtime_dispatches"]
    assert all(
        candidate.source_sequence == candidate.index
        and candidate.frame_sha256.startswith("sha256:")
        for candidate in receipt.baseline.candidates
    )
    decoder = receipt.baseline.media_identity["decoder_identity"]
    assert decoder["library"] == "opencv"
    assert decoder["build_sha256"].startswith("sha256:")
    assert receipt.adaptive.media_identity["decoder_identity"] == decoder
    assert all(
        call.oracle_lookup_sha256.startswith("sha256:")
        for candidate in receipt.baseline.candidates
        for call in candidate.provider_calls
    )
    persisted = tmp_path / "receipt.json"
    persisted.write_text(receipt.to_json(), encoding="utf-8")
    loaded = load_matched_watch_receipt(persisted)
    assert loaded.receipt.to_dict() == receipt.to_dict()


def test_real_transition_does_not_renotify_a_flickering_held_state(tmp_path):
    """One continuous action, judged off real decoded media, delivers ONE notification.

    The oracle script here is the defect shape: the judge reads B at t=0.5s, flickers back to A at
    t=0.75s, and reads B again at t=1.0s and t=1.25s. Nothing about the world changed — only the
    judge's mind did. The runtime used to treat that flicker as a fresh ordering edge and emit a
    second Incident 0.5s after the first, because the transition route returned its fire without
    ever passing the delivery seam.

    The re-notification lands INSIDE the shipped 0.7s delivery resolution, so it is a re-report of
    the occurrence already delivered. It is the counterpart of
    ``test_real_receipts_feed_existing_measured_frontier_without_qualification``, whose two events
    are deliberately staged 0.75s apart — OUTSIDE that resolution — and are both delivered. Together
    they pin both sides of the boundary on real media, and this one is what keeps the oracle's
    ``merged_duplicate`` branch live rather than dead.
    """
    manifest_path = _write_real_case(
        tmp_path,
        frame_values=(9, 31, 229, 41, 219, 221),
        states=(False, False, True, False, True, True),
    )

    receipt = run_matched_watch(manifest_path)
    derive_viability_rows(receipt)  # the reconstruction oracle must accept this trace

    outcomes = [
        item["data"]["outcome"]
        for item in receipt.baseline.traces
        if item["stage"] == "transition.decision"
    ]
    assert outcomes == [
        "armed",
        "rearmed",
        "fire",
        "rearmed",          # the judge flickers back to A — a re-arm, not a new occurrence
        "merged_duplicate",  # ...so the edge it re-opens is withheld at the delivery seam
        "suppressed_latched",
    ]
    assert len(receipt.baseline.incidents) == 1
    assert [incident["reason"] for incident in receipt.baseline.incidents] == ["rising_edge"]


def test_real_ordinary_template_uses_one_executed_judgment_per_dispatch(tmp_path):
    manifest_path = _write_real_case(
        tmp_path,
        template_ref="facility_observation@1/zone_occupied",
        archetype="physical_scene/zone_occupied",
    )

    receipt = run_matched_watch(manifest_path)
    baseline, _ = derive_viability_rows(receipt)

    assert baseline["runtime_dispatches"] == 6
    assert baseline["judge_calls"] == 6
    assert all(
        [call.predicate for call in candidate.provider_calls] == ["condition"]
        for candidate in receipt.baseline.candidates
        if candidate.runtime_dispatched
    )
    started = [
        trace
        for trace in receipt.baseline.traces
        if trace["stage"] == "judge.started"
    ]
    assert [dict(trace["data"]) for trace in started] == [
        {"temporal": "single", "roi": False}
        for candidate in receipt.baseline.candidates
        if candidate.runtime_dispatched
    ]
    assert len(receipt.baseline.incidents) == 1


def test_v1_synthetic_wire_shape_remains_exactly_compatible(tmp_path):
    from tests.test_matched_watch import _write_case

    receipt = run_matched_watch(_write_case(tmp_path))
    payload = receipt.to_dict()

    assert payload["schema"] == "percept.eval.matched_watch.receipt/v1"
    assert "input_artifacts" not in payload
    assert "authority_artifact_presence" not in payload
    assert "source_sequence" not in payload["baseline"]["candidates"][0]
    assert "oracle_lookup_sha256" not in payload["baseline"]["candidates"][0]["provider_calls"][0]


def test_v1_synthetic_validation_never_imports_the_optional_decoder(tmp_path, monkeypatch):
    from tests.test_matched_watch import _write_case

    monkeypatch.setitem(sys.modules, "cv2", None)
    receipt = run_matched_watch(_write_case(tmp_path))

    baseline, adaptive = derive_viability_rows(receipt)

    assert baseline["candidate_windows_total"] == adaptive["candidate_windows_total"]
    assert receipt.receipt_schema == "percept.eval.matched_watch.receipt/v1"


def test_v1_synthetic_manifest_remains_portable_without_dir_fd_open(
    tmp_path,
    monkeypatch,
):
    from tests.test_matched_watch import _write_case

    manifest_path = _write_case(tmp_path)
    monkeypatch.setattr(matched_watch_module.os, "supports_dir_fd", set())

    loaded = load_matched_watch_manifest(manifest_path)

    assert loaded.schema == "percept.eval.matched_watch/v1"


@pytest.mark.parametrize(
    "attack",
    [
        "duplicate_media",
        "group_leakage",
        "path_escape",
        "oracle_tamper",
        "media_symlink",
        "media_hardlink",
    ],
)
def test_real_manifest_structure_and_artifacts_fail_closed(tmp_path, attack):
    case_root = tmp_path / "case"
    manifest_path = _write_real_case(case_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_path = case_root / "corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if attack == "duplicate_media":
        second = dict(corpus["cases"][0])
        second["case_id"] = "real-case-002"
        second["source_group_id"] = "generated-session-002"
        corpus["cases"].append(second)
        corpus_path.write_bytes(_json_bytes(corpus))
        manifest["corpus"]["sha256"] = _sha(corpus_path.read_bytes())
    elif attack == "group_leakage":
        second = dict(corpus["cases"][0])
        second["case_id"] = "real-case-002"
        second["split"] = "other"
        second["media_sha256"] = "sha256:" + "1" * 64
        second["oracle_sha256"] = "sha256:" + "2" * 64
        corpus["cases"].append(second)
        corpus_path.write_bytes(_json_bytes(corpus))
        manifest["corpus"]["sha256"] = _sha(corpus_path.read_bytes())
    elif attack == "path_escape":
        outside = tmp_path / "outside.mp4"
        outside.write_bytes((case_root / "case.mp4").read_bytes())
        manifest["media"]["path"] = "../outside.mp4"
    elif attack == "oracle_tamper":
        oracle = case_root / "oracle.json"
        oracle.write_bytes(oracle.read_bytes() + b"\n")
    elif attack == "media_symlink":
        original = case_root / "case.mp4"
        moved = case_root / "original.mp4"
        original.rename(moved)
        original.symlink_to(moved.name)
    else:
        original = case_root / "case.mp4"
        hardlink = case_root / "hardlinked.mp4"
        os.link(original, hardlink)
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(MatchedWatchInputError):
        load_matched_watch_manifest(manifest_path)


def test_persisted_v2_revalidates_descriptor_backed_media(tmp_path):
    manifest_path = _write_real_case(tmp_path)
    receipt = run_matched_watch(manifest_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(receipt.to_json(), encoding="utf-8")
    media = tmp_path / "case.mp4"
    media.write_bytes(media.read_bytes() + b"tamper")

    with pytest.raises(MatchedWatchInputError, match="artifact|snapshot|media"):
        load_matched_watch_receipt(receipt_path)


def test_real_arms_decode_the_same_frozen_mp4_independently(tmp_path, monkeypatch):
    import percept.runtime.source as source_module

    manifest_path = _write_real_case(tmp_path)
    real_open_source = source_module.open_source
    opens = 0
    queue_bounds = []

    async def counted_open_source(*args, **kwargs):
        nonlocal opens
        opens += 1
        queue_bounds.append(kwargs["maxsize"])
        async for envelope in real_open_source(*args, **kwargs):
            yield envelope

    monkeypatch.setattr(source_module, "open_source", counted_open_source)
    receipt = run_matched_watch(manifest_path)

    assert opens == 2
    assert queue_bounds == [64, 64]
    assert [
        (item.source_sequence, item.t, item.frame_sha256) for item in receipt.baseline.candidates
    ] == [(item.source_sequence, item.t, item.frame_sha256) for item in receipt.adaptive.candidates]


def test_decoder_absence_fails_loudly_with_runtime_install_guidance(tmp_path, monkeypatch):
    manifest_path = _write_real_case(tmp_path)
    monkeypatch.setitem(sys.modules, "cv2", None)

    with pytest.raises(MatchedWatchInputError, match=r"percept-vision\[runtime\]"):
        run_matched_watch(manifest_path)


def test_event_labels_never_answer_offline_oracle_judgments(tmp_path):
    manifest_path = _write_real_case(tmp_path)
    first = run_matched_watch(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events"][0]["review_authority"] = "different-caller-string"
    manifest["events"][0]["label_digest"] = _sha(b"different event metadata")
    manifest_path.write_bytes(_json_bytes(manifest))

    second = run_matched_watch(manifest_path)

    assert [
        call.to_dict()
        for candidate in first.baseline.candidates
        for call in candidate.provider_calls
    ] == [
        call.to_dict()
        for candidate in second.baseline.candidates
        for call in candidate.provider_calls
    ]


def test_authority_artifact_presence_is_not_authority_or_qualification(tmp_path):
    manifest_path = _write_real_case(tmp_path)
    license_path = tmp_path / "license.txt"
    license_path.write_text("generated fixture license declaration", encoding="utf-8")
    corpus_path = tmp_path / "corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["authorities"]["license"] = {
        "path": "license.txt",
        "sha256": _sha(license_path.read_bytes()),
    }
    corpus_path.write_bytes(_json_bytes(corpus))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus"]["sha256"] = _sha(corpus_path.read_bytes())
    manifest_path.write_bytes(_json_bytes(manifest))

    receipt = run_matched_watch(manifest_path)

    assert receipt.authority_artifact_presence["license"] is True
    assert receipt.authority_artifact_presence["label_review"] is False
    assert receipt.launch_qualified is False
    assert "authority:license" in {item["role"] for item in receipt.input_artifacts}


def test_registered_future_pack_template_uses_same_engine_without_hardcoding(tmp_path):
    from percept.domain_pack import (
        DEFAULT_DOMAIN_PACK_REGISTRY,
        DomainPack,
        GoalTemplate,
        register_domain_pack,
    )

    pack = DomainPack(
        id="test_future_pack",
        version="1",
        templates={
            "presence": GoalTemplate(
                name="presence",
                condition="an item is present in {zone_name}",
                parameters={"zone_name": "zone"},
                target="item",
            )
        },
    )
    register_domain_pack(pack)
    try:
        manifest_path = _write_real_case(
            tmp_path,
            template_ref="test_future_pack@1/presence",
            archetype="future/item_presence",
        )
        receipt = run_matched_watch(manifest_path)
        baseline, _ = derive_viability_rows(receipt)

        assert receipt.manifest["watch"]["domain_pack"] == "test_future_pack@1"
        assert baseline["judge_calls"] == baseline["runtime_dispatches"] == 6
    finally:
        DEFAULT_DOMAIN_PACK_REGISTRY.unregister("test_future_pack@1")


@pytest.mark.parametrize(
    ("predicate", "statuses"),
    [
        ("from", ["error"]),
        ("to", ["success", "error"]),
    ],
)
def test_transition_predicate_failure_short_circuits_to_one_missing_candidate(
    tmp_path,
    predicate,
    statuses,
):
    manifest_path = _write_real_case(tmp_path)
    _set_oracle_error(manifest_path, index=2, predicate=predicate)

    receipt = run_matched_watch(manifest_path)
    candidate = receipt.baseline.candidates[2]

    assert [call.status for call in candidate.provider_calls] == statuses
    assert candidate.decision == "missing"
    assert sum(item.decision == "missing" for item in receipt.baseline.candidates) == 1
    assert receipt.baseline.provider_call_failures == 1
    matched_watch_module._validate_receipt(receipt)
    with pytest.raises(MatchedWatchInputError, match="not comparable"):
        derive_viability_rows(receipt)


def test_decoder_identity_is_required_for_exact_frame_reconstruction(tmp_path, monkeypatch):
    manifest_path = _write_real_case(tmp_path)
    receipt = run_matched_watch(manifest_path)
    active = dict(receipt.baseline.media_identity["decoder_identity"])
    changed = {**active, "version": active["version"] + "-different-build"}
    monkeypatch.setattr(
        matched_watch_module,
        "_decoder_identity",
        lambda: changed,
    )

    with pytest.raises(MatchedWatchInputError, match="decoder identity"):
        derive_viability_rows(receipt)


def test_v2_authority_presence_cannot_be_relabelled_in_the_receipt(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    forged = replace(
        receipt,
        authority_artifact_presence={
            **receipt.authority_artifact_presence,
            "license": True,
        },
    )

    with pytest.raises(MatchedWatchInputError, match="authority artifact presence"):
        derive_viability_rows(forged)


def test_v2_candidate_evidence_must_reconstruct_from_the_frozen_oracle(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    forged_sha = "sha256:" + "0" * 64

    def forge_arm(arm):
        candidate = replace(arm.candidates[0], frame_sha256=forged_sha)
        return replace(arm, candidates=(candidate, *arm.candidates[1:]))

    forged = replace(
        receipt,
        baseline=forge_arm(receipt.baseline),
        adaptive=forge_arm(receipt.adaptive),
    )

    with pytest.raises(MatchedWatchInputError, match="decoded candidate evidence"):
        derive_viability_rows(forged)


def test_v2_provider_answer_must_reconstruct_from_the_frozen_oracle(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    first = receipt.baseline.candidates[0]
    forged_call = replace(first.provider_calls[0], evidence="caller-forged answer")
    forged_candidate = replace(
        first,
        provider_calls=(forged_call, *first.provider_calls[1:]),
    )
    forged_baseline = replace(
        receipt.baseline,
        candidates=(forged_candidate, *receipt.baseline.candidates[1:]),
    )
    forged = replace(receipt, baseline=forged_baseline)

    with pytest.raises(MatchedWatchInputError, match="provider calls disagree"):
        derive_viability_rows(forged)


def test_run_fails_if_oracle_changes_between_independent_arms(tmp_path, monkeypatch):
    manifest_path = _write_real_case(tmp_path)
    original = matched_watch_module._run_arm

    async def mutate_after_baseline(manifest, plan, arm):
        result = await original(manifest, plan, arm)
        if arm == "baseline":
            oracle = next(
                Path(item["path"]) for item in manifest.input_artifacts if item["role"] == "oracle"
            )
            oracle.write_bytes(oracle.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(matched_watch_module, "_run_arm", mutate_after_baseline)

    with pytest.raises(MatchedWatchInputError, match="artifact|oracle|snapshot|changed"):
        run_matched_watch(manifest_path)


def test_decoder_cannot_use_a_transient_unrecorded_media_path(tmp_path, monkeypatch):
    import percept.runtime.source as source_module

    case_root = tmp_path / "case"
    manifest_path = _write_real_case(case_root)
    media = case_root / "case.mp4"
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(media.read_bytes() + b"unrecorded trailing container bytes")
    real_open_source = source_module.open_source
    opens = 0

    async def swap_only_while_decoder_opens(uri, *args, **kwargs):
        nonlocal opens
        opens += 1
        saved = case_root / f"case.saved-{opens}.mp4"
        media.rename(saved)
        media.symlink_to(outside)
        restored = False
        try:
            async for envelope in real_open_source(uri, *args, **kwargs):
                if not restored:
                    media.unlink()
                    saved.rename(media)
                    restored = True
                yield envelope
        finally:
            if not restored:
                media.unlink(missing_ok=True)
                saved.rename(media)

    monkeypatch.setattr(source_module, "open_source", swap_only_while_decoder_opens)

    with pytest.raises(MatchedWatchInputError, match="media|artifact|snapshot|changed|symlink"):
        run_matched_watch(manifest_path)


def test_directory_swap_to_symlink_between_containment_check_and_open_fails(
    tmp_path,
    monkeypatch,
):
    case_root = tmp_path / "case"
    manifest_path = _write_real_case(case_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = case_root / "assets"
    assets.mkdir()
    (case_root / "case.mp4").rename(assets / "case.mp4")
    manifest["media"]["path"] = "assets/case.mp4"
    manifest_path.write_bytes(_json_bytes(manifest))

    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copy2(assets / "case.mp4", outside / "case.mp4")
    saved_assets = case_root / "assets-before-race"
    original = matched_watch_module._resolve_contained_path
    swapped = False

    def swap_after_check(raw, *, root, context):
        nonlocal swapped
        result = original(raw, root=root, context=context)
        if context == "manifest.media.path" and not swapped:
            swapped = True
            assets.rename(saved_assets)
            assets.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(
        matched_watch_module,
        "_resolve_contained_path",
        swap_after_check,
    )

    with pytest.raises(MatchedWatchInputError, match="symlink|escape|changed|unsafe"):
        load_matched_watch_manifest(manifest_path)


def test_duplicate_oracle_digest_across_corpus_splits_fails_closed(tmp_path):
    manifest_path = _write_real_case(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_path = tmp_path / "corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    second = dict(corpus["cases"][0])
    second.update(
        {
            "case_id": "real-case-other-split",
            "split": "held-out",
            "source_group_id": "different-source-group",
            "media_sha256": "sha256:" + "1" * 64,
        }
    )
    corpus["cases"].append(second)
    corpus_path.write_bytes(_json_bytes(corpus))
    manifest["corpus"]["sha256"] = _sha(corpus_path.read_bytes())
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(MatchedWatchInputError, match="oracle|split|duplicate|leak"):
        load_matched_watch_manifest(manifest_path)


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("baseline", "source_missing_sequences", 1, "counter|source|gap|stats"),
        ("adaptive", "frames_dropped", 1, "drop|counter|stats"),
    ],
)
def test_v2_runtime_counters_must_reconstruct_from_candidates(
    tmp_path,
    target,
    field,
    value,
    message,
):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    arm = getattr(receipt, target)
    runtime_stats = {**arm.runtime_stats, field: value}
    forged = _reseal(receipt, **{target: replace(arm, runtime_stats=runtime_stats)})

    with pytest.raises(MatchedWatchInputError, match=message):
        derive_viability_rows(forged)


def test_v2_judgment_action_must_reconstruct_from_fire_candidate(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    traces = list(receipt.baseline.traces)
    index = next(index for index, trace in enumerate(traces) if trace["stage"] == "judgment.event")
    traces[index] = {
        **traces[index],
        "data": {**traces[index]["data"], "action": "ask"},
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="trace|action|judgment"):
        derive_viability_rows(forged)


def test_v2_escalation_health_must_reconstruct_from_edge_admission(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    health = list(receipt.baseline.health)
    index = next(index for index, event in enumerate(health) if event["kind"] == "escalation")
    health[index] = {
        **health[index],
        "data": {**health[index]["data"], "cause": "forged-cause"},
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, health=tuple(health)),
    )

    with pytest.raises(MatchedWatchInputError, match="health|escalation|cause"):
        derive_viability_rows(forged)


def test_v2_incident_message_must_reconstruct_from_compiled_watch(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    incidents = list(receipt.baseline.incidents)
    incidents[0] = {**incidents[0], "message": "caller-forged incident message"}
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, incidents=tuple(incidents)),
    )

    with pytest.raises(MatchedWatchInputError, match="Incident|message|Watch"):
        derive_viability_rows(forged)


def test_v2_edge_trace_outer_identity_must_reconstruct_from_candidate(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    traces = list(receipt.baseline.traces)
    index = next(index for index, trace in enumerate(traces) if trace["stage"] == "edge.decision")
    traces[index] = {
        **traces[index],
        "trace_id": "caller-forged-edge-coordinate",
        "goal_id": "caller-forged-goal",
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="edge|trace|identity|coordinate|goal"):
        derive_viability_rows(forged)


def test_v2_global_trace_append_order_must_reconstruct_from_execution(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    traces = list(receipt.baseline.traces)
    source_index = next(
        index for index, trace in enumerate(traces) if trace["stage"] == "source.frame"
    )
    transition_index = next(
        index
        for index, trace in enumerate(traces)
        if trace["stage"] == "transition.decision" and trace["t"] == traces[source_index]["t"]
    )
    traces[source_index], traces[transition_index] = (
        traces[transition_index],
        traces[source_index],
    )
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="trace|order|execution|append"):
        derive_viability_rows(forged)


def test_v2_health_append_order_must_reconstruct_from_execution(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    health = list(receipt.baseline.health)
    ended_index = next(
        index for index, event in enumerate(health) if event["kind"] == "runtime_ended"
    )
    ended = health.pop(ended_index)
    health.insert(0, ended)
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, health=tuple(health)),
    )

    with pytest.raises(MatchedWatchInputError, match="health|order|runtime|execution|append"):
        derive_viability_rows(forged)


def test_v2_rows_can_be_derived_inside_an_active_event_loop(tmp_path):
    manifest_path = _write_real_case(tmp_path)

    async def exercise():
        receipt = await matched_watch_module.run_matched_watch_async(manifest_path)
        return derive_viability_rows(receipt)

    baseline, adaptive = asyncio.run(exercise())

    assert baseline["arm"] == "baseline"
    assert adaptive["arm"] == "adaptive"


def test_v2_ordinary_judge_finished_trace_reconstructs_from_oracle(tmp_path):
    receipt = run_matched_watch(
        _write_real_case(
            tmp_path,
            template_ref="facility_observation@1/zone_occupied",
            archetype="physical_scene/zone_occupied",
        )
    )
    traces = list(receipt.baseline.traces)
    index = next(
        index
        for index, trace in enumerate(traces)
        if trace["stage"] == "judge.finished"
    )
    traces[index] = {
        **traces[index],
        "trace_id": "caller-forged-judge-coordinate",
        "data": {
            **traces[index]["data"],
            "satisfied": not traces[index]["data"]["satisfied"],
            "confidence": 0.01,
        },
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="judge|trace|oracle|identity|execution"):
        derive_viability_rows(forged)


def test_v2_ordinary_provider_failure_receipt_validates_before_not_comparable(
    tmp_path,
):
    manifest_path = _write_real_case(
        tmp_path,
        template_ref="facility_observation@1/zone_occupied",
        archetype="physical_scene/zone_occupied",
    )
    _set_ordinary_oracle_error(manifest_path, index=0)
    receipt = run_matched_watch(manifest_path)

    matched_watch_module._validate_receipt(receipt)
    degraded = [
        trace
        for trace in receipt.baseline.traces
        if trace["stage"] == "judge.degraded"
    ]
    assert [(trace["t"], dict(trace["data"])) for trace in degraded] == [
        (
            receipt.baseline.candidates[0].t,
            {"cause": "RuntimeError('judgment provider unavailable')"},
        )
    ]
    degraded_health = [
        event
        for event in receipt.baseline.health
        if event["kind"] == "judge_degraded"
    ]
    assert [(event["ts"], dict(event["data"])) for event in degraded_health] == [
        (
            receipt.baseline.candidates[0].t,
            {
                "key": repr(
                    (
                        receipt.baseline.compiled_binding["goal_id"],
                        None,
                    )
                ),
                "cause": repr("judgment provider unavailable"),
            },
        )
    ]
    receipt_path = tmp_path / "ordinary-error-receipt.json"
    receipt_path.write_text(receipt.to_json(), encoding="utf-8")

    with pytest.raises(MatchedWatchInputError, match="not comparable"):
        load_matched_watch_receipt(receipt_path)
    with pytest.raises(MatchedWatchInputError, match="not comparable"):
        derive_viability_rows(receipt)


def test_v2_consecutive_provider_failures_reconstruct_health_at_each_completion(
    tmp_path,
):
    manifest_path = _write_real_case(
        tmp_path,
        template_ref="facility_observation@1/zone_occupied",
        archetype="physical_scene/zone_occupied",
    )
    for index in range(5):
        _set_ordinary_oracle_error(manifest_path, index=index)

    receipt = run_matched_watch(manifest_path)
    matched_watch_module._validate_receipt(receipt)

    degraded = [
        (event["kind"], event["ts"])
        for event in receipt.baseline.health
        if event["kind"] in {
            "judge_degraded",
            "judge_persistent_failure",
        }
    ]
    candidate_times = [
        candidate.t for candidate in receipt.baseline.candidates[:5]
    ]
    assert degraded == [
        *[("judge_degraded", candidate_t) for candidate_t in candidate_times[:4]],
        ("judge_degraded", candidate_times[4]),
        ("judge_persistent_failure", candidate_times[4]),
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("from_evidence", "caller-forged verdict evidence"),
        ("from_confidence", 0.01),
    ],
)
def test_v2_transition_verdict_trace_must_reconstruct_from_oracle_call(
    tmp_path,
    field,
    value,
):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    traces = list(receipt.baseline.traces)
    index = next(
        index for index, trace in enumerate(traces) if trace["stage"] == "transition.decision"
    )
    traces[index] = {
        **traces[index],
        "data": {**traces[index]["data"], field: value},
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="transition trace verdict|oracle"):
        derive_viability_rows(forged)


def test_v2_transition_trace_identity_must_reconstruct_from_dispatch(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    traces = list(receipt.baseline.traces)
    index = next(
        index for index, trace in enumerate(traces) if trace["stage"] == "transition.decision"
    )
    traces[index] = {
        **traces[index],
        "trace_id": "caller-forged-transition-coordinate",
        "goal_id": "caller-forged-goal",
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="transition trace identity"):
        derive_viability_rows(forged)


def test_v2_transition_latch_state_must_reconstruct_from_oracle_sequence(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    traces = list(receipt.baseline.traces)
    index = next(
        index for index, trace in enumerate(traces) if trace["stage"] == "transition.decision"
    )
    traces[index] = {
        **traces[index],
        "data": {
            **traces[index]["data"],
            "armed": False,
            "fired_latch": True,
        },
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="transition trace"):
        derive_viability_rows(forged)


def test_v2_ordinary_gate_trace_must_reconstruct_from_oracle_and_gate(tmp_path):
    receipt = run_matched_watch(
        _write_real_case(
            tmp_path,
            template_ref="facility_observation@1/zone_occupied",
            archetype="physical_scene/zone_occupied",
        )
    )
    traces = list(receipt.baseline.traces)
    index = next(index for index, trace in enumerate(traces) if trace["stage"] == "gate.classified")
    traces[index] = {
        **traces[index],
        "data": {
            **traces[index]["data"],
            "satisfied": True,
            "confidence": 0.01,
        },
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="ordinary gate trace"):
        derive_viability_rows(forged)


def test_v2_runtime_ended_counters_must_reconstruct_from_candidates(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    health = list(receipt.baseline.health)
    index = next(index for index, event in enumerate(health) if event["kind"] == "runtime_ended")
    health[index] = {
        **health[index],
        "data": {
            **health[index]["data"],
            "signal_admits": health[index]["data"]["signal_admits"] + 1,
        },
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, health=tuple(health)),
    )

    with pytest.raises(MatchedWatchInputError, match="runtime-ended|RuntimeStats"):
        derive_viability_rows(forged)


def test_v2_runtime_effective_config_must_reconstruct_from_compiled_replay(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    effective = {
        **receipt.baseline.runtime_stats["effective_config"],
        "coast_enabled": False,
    }
    runtime_stats = {
        **receipt.baseline.runtime_stats,
        "effective_config": effective,
    }
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, runtime_stats=runtime_stats),
    )

    with pytest.raises(MatchedWatchInputError, match="effective runtime configuration"):
        derive_viability_rows(forged)


def test_v2_source_frame_identity_must_reconstruct_from_dispatch(tmp_path):
    receipt = run_matched_watch(_write_real_case(tmp_path))
    traces = list(receipt.baseline.traces)
    index = next(index for index, trace in enumerate(traces) if trace["stage"] == "source.frame")
    traces[index] = {**traces[index], "trace_id": "caller-forged@0.0"}
    forged = _reseal(
        receipt,
        baseline=replace(receipt.baseline, traces=tuple(traces)),
    )

    with pytest.raises(MatchedWatchInputError, match="source-frame trace identity"):
        derive_viability_rows(forged)


def test_real_manifest_rejects_a_symlinked_parent_path(tmp_path):
    real_root = tmp_path / "real"
    manifest_path = _write_real_case(real_root)
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(MatchedWatchInputError, match="symlink|unsafe"):
        load_matched_watch_manifest(alias / manifest_path.name)


def test_real_receipts_feed_existing_measured_frontier_without_qualification(tmp_path):
    receipt_paths = []
    cases = []
    operating_points = {}
    for case_number in (1, 2):
        case_id = f"real-bundle-{case_number:02d}"
        cases.append(
            {
                "id": case_id,
                "candidate_windows_total": 6,
                "events": [
                    {"id": f"{case_id}/event-1", "candidate_windows": 1},
                    {"id": f"{case_id}/event-2", "candidate_windows": 2},
                ],
            }
        )
        for point, threshold in (("observe", 0.20), ("sparse", 0.60)):
            directory = tmp_path / f"{case_id}-{point}"
            manifest_path = _write_real_case(
                directory,
                case_id=case_id,
                source_group_id=f"session-{case_number:02d}",
                operating_point=point,
                threshold=threshold,
                frame_values=(
                    8 + case_number,
                    30 + case_number,
                    230 - case_number,
                    40 + case_number,
                    220 - case_number,
                    222 - case_number,
                ),
                # The first positive moves to index 1 (t=0.25s), so the two staged events sit 0.75s
                # apart instead of 0.5s — outside the shipped 0.7s delivery resolution. Same two
                # events, same frame_values (so admission and the adaptive arm are untouched); only
                # the spacing now matches a separation the product actually claims to resolve.
                states=(False, True, False, False, True, True),
                two_events=True,
            )
            receipt = run_matched_watch(manifest_path)
            receipt_path = directory / "receipt.json"
            receipt_path.write_text(receipt.to_json(), encoding="utf-8")
            receipt_paths.append(receipt_path)
            if point not in operating_points:
                arm = receipt.adaptive.to_dict()
                configuration = {
                    "effective": arm["resolved_watch"]["effective"],
                    "source_profile": arm["source_profile"],
                    "compiled_binding": arm["compiled_binding"],
                }
                operating_points[point] = {
                    "id": point,
                    "configuration": configuration,
                    "configuration_sha256": receipt.adaptive.configuration_sha256,
                }
    viability = {
        "schema_version": "3",
        "track": "viability_frontier",
        "scope": "programmable Watches over live multimodal streams",
        "evidence_kind": "synthetic",
        "preregistered": False,
        "archetypes": [
            {
                "id": "physical_scene/package_appeared",
                "provider_call_contract": {
                    "completed_dispatch_calls": 2,
                    "missing_dispatch_min_calls": 1,
                    "missing_dispatch_max_calls": 2,
                },
                "cases": cases,
                "operating_points": list(operating_points.values()),
                "policy": {
                    "min_candidate_recall": 0.0,
                    "min_event_recall": 0.0,
                    "min_precision": 0.0,
                    "min_call_reduction": 0.0,
                    "max_event_abstention_rate": 1.0,
                    "max_missing_rate": 1.0,
                },
            }
        ],
    }
    viability_path = tmp_path / "viability.json"
    viability_path.write_bytes(_json_bytes(viability))

    report = derive_measured_evidence_bundle(
        viability_manifest_path=viability_path,
        receipt_paths=receipt_paths,
    )

    assert len(report.coverage) == 4
    assert len(report.derived_rows) == 8
    assert report.judge_identity["judge_mode"] == "offline_oracle"
    assert report.authority_verified is False
    assert report.launch_qualified is False
    assert report.frontier.launch_qualified is False
