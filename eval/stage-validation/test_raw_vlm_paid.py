"""Offline-only tests for paid-admission, provenance, and crash recovery."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import raw_vlm_baseline as RV


def frozen_fixture(tmp_path: Path, *, cases: int = 2) -> RV.FrozenCorpus:
    media = b"offline mp4 fixture bytes"
    media_path = tmp_path / "source.mp4"
    media_path.write_bytes(media)
    media_sha = hashlib.sha256(media).hexdigest()
    raw_cases = [
        {
            "case_id": f"paid::{index}",
            "dataset": "offline",
            "duration_s": 10.0,
            "goal_kind": "event",
            "goal_text": f"event number {index}",
            "positives": [[1.0, 2.0]],
            "video_path": "source.mp4",
            "video_sha256": media_sha,
        }
        for index in range(cases)
    ]
    corpus_doc = {"corpus_version": "1.2", "n_cases": cases, "cases": raw_cases}
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus_doc))
    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    authoring = {
        "corpus": {
            "sha256": corpus_sha,
            "version": "1.2",
            "n_cases": cases,
            "n_unique_goals": cases,
        },
        "entries": {
            str(index): {"goal_text": item["goal_text"]}
            for index, item in enumerate(raw_cases)
        },
    }
    authoring_path = tmp_path / "authoring.json"
    authoring_path.write_text(json.dumps(authoring))
    return RV.preflight_frozen_corpus(corpus_path, authoring_path, repo_root=tmp_path)


def pricing(*, model: str = "gemini-offline-model") -> RV.PricingStamp:
    return RV.PricingStamp(
        backend="gemini",
        model=model,
        pricing_version="offline-prices-v1",
        pricing_source="offline immutable test fixture",
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
    )


def build() -> RV.BuildStamp:
    return RV.BuildStamp("1.0.0", "abc123", "1", "2026-08-01T00:00:00Z")


def profile() -> RV.ProfileStamp:
    return RV.ProfileStamp(
        "offline-profile", hashlib.sha256(b"offline profile").hexdigest(), "test fixture"
    )


def policy(*, max_usd: float = 0.10, max_calls: int = 2, bound: float = 0.04):
    return RV.PaidRunPolicy(max_usd, max_calls, bound, 5.0)


def accepted() -> RV.ProviderResponse:
    return RV.ProviderResponse(
        '{"events":[{"start_s":1.0,"end_s":2.0}]}',
        input_tokens=100,
        output_tokens=20,
    )


class FakeBudgetedProvider:
    backend = "gemini"
    supports_native_video = True
    budget_reservation_before_transport = True

    def __init__(self, responses, *, model: str = "gemini-offline-model"):
        self.model = model
        self.responses = list(responses)
        self.transport_calls = 0
        self.complete_invocations = 0
        self.budget = None
        self.checkpoint_path: Path | None = None
        self.reservation_was_durable = []

    def configure_request_budget(self, request_budget):
        self.budget = request_budget

    def complete(self, request):
        self.complete_invocations += 1
        reservation = self.budget.begin_attempt()
        if self.checkpoint_path is not None:
            checkpoint = json.loads(self.checkpoint_path.read_text())
            self.reservation_was_durable.append(
                checkpoint["status"] == "provider_attempt_reserved"
                and checkpoint["pending_call"]["case_id"] == request.case_id
            )
        self.transport_calls += 1
        value = self.responses.pop(0)
        if isinstance(value, KeyboardInterrupt):
            # Deliberately simulate process loss after transport and before budget reconciliation.
            raise value
        if isinstance(value, BaseException):
            self.budget.fail_attempt(reservation)
            raise value
        usage = {
            "input_tokens": value.input_tokens or 0,
            "output_tokens": value.output_tokens or 0,
        }
        if value.reported_cost_usd is None:
            self.budget.complete_unpriced_attempt(reservation, usage)
        else:
            self.budget.complete_attempt(
                reservation, {**usage, "cost_usd": value.reported_cost_usd}
            )
        return value


def paid_run(tmp_path: Path, corpus, provider, **overrides):
    checkpoint = overrides.pop("checkpoint_path", tmp_path / "paid.checkpoint.json")
    provider.checkpoint_path = checkpoint
    return RV.run_paid_baseline(
        corpus,
        provider,
        overrides.pop("pricing", pricing()),
        policy=overrides.pop("policy", policy()),
        build=overrides.pop("build", build()),
        profile=overrides.pop("profile", profile()),
        checkpoint_path=checkpoint,
        **overrides,
    )


def test_reservation_is_durable_before_transport_and_call_cap_counts_unprocessed_as_miss(tmp_path):
    corpus = frozen_fixture(tmp_path)
    provider = FakeBudgetedProvider([accepted()])
    result = paid_run(
        tmp_path, corpus, provider, policy=policy(max_calls=1, bound=0.04)
    )

    assert provider.transport_calls == 1
    assert provider.reservation_was_durable == [True]
    assert result["status"] == "budget_exhausted"
    assert result["complete_corpus"] is False
    assert result["execution"]["unprocessed_cases"] == ["paid::1"]
    assert result["receiver_independent_quality"]["false_negatives"] == 1
    assert result["receiver_independent_quality"]["recall"] == 0.5
    assert result["economy"]["budget"]["physical_attempts"] == 1


def test_usd_cap_stops_before_overspend(tmp_path):
    corpus = frozen_fixture(tmp_path)
    provider = FakeBudgetedProvider([accepted()])
    result = paid_run(
        tmp_path,
        corpus,
        provider,
        policy=policy(max_usd=0.10, max_calls=10, bound=0.06),
    )
    assert provider.transport_calls == 1
    assert result["status"] == "budget_exhausted"
    assert result["economy"]["budget"]["committed_usd"] == pytest.approx(0.06)
    assert result["economy"]["budget"]["committed_usd"] <= 0.10


def test_failed_attempt_retains_conservative_cost_and_usage_is_unknown(tmp_path):
    corpus = frozen_fixture(tmp_path)
    provider = FakeBudgetedProvider([RuntimeError("offline failure"), accepted()])
    result = paid_run(tmp_path, corpus, provider)
    budget = result["economy"]["budget"]
    assert budget["failed_attempts"] == 1
    assert budget["physical_attempts"] == 2
    assert budget["committed_usd"] == pytest.approx(0.08)
    assert result["calls"][0]["response"]["status"] == "provider_error"
    assert result["calls"][0]["usage"]["complete"] is False
    assert result["economy"]["usage_complete_for_full_corpus"] is False


def test_resume_recovers_unknown_paid_attempt_without_replay_and_keeps_raw_response(tmp_path):
    corpus = frozen_fixture(tmp_path)
    checkpoint = tmp_path / "crash.checkpoint.json"
    crashing = FakeBudgetedProvider([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        paid_run(tmp_path, corpus, crashing, checkpoint_path=checkpoint)
    crashed = json.loads(checkpoint.read_text())
    assert crashed["pending_call"]["case_id"] == "paid::0"
    assert crashed["budget"]["physical_attempts"] == 1
    assert crashed["budget"]["inflight"]

    resumed = FakeBudgetedProvider([accepted()])
    result = paid_run(
        tmp_path, corpus, resumed, checkpoint_path=checkpoint, resume=True
    )
    assert resumed.transport_calls == 1
    assert result["calls"][0]["response"]["provider_error"].startswith(
        "RecoveredUnknownOutcome"
    )
    assert result["calls"][1]["response"]["raw_response"] == accepted().raw_response
    assert result["economy"]["budget"]["physical_attempts"] == 2
    persisted = json.loads(checkpoint.read_text())
    assert persisted["calls"][1]["response"]["raw_response"] == accepted().raw_response


def test_resume_rejects_any_compatibility_drift_before_provider(tmp_path):
    corpus = frozen_fixture(tmp_path, cases=1)
    checkpoint = tmp_path / "exact.checkpoint.json"
    paid_run(tmp_path, corpus, FakeBudgetedProvider([accepted()]), checkpoint_path=checkpoint)
    drifted_pricing = pricing(model="gemini-different-model")
    provider = FakeBudgetedProvider([], model="gemini-different-model")
    with pytest.raises(RV.BaselineInputError, match="compatibility differs"):
        paid_run(
            tmp_path,
            corpus,
            provider,
            checkpoint_path=checkpoint,
            pricing=drifted_pricing,
            resume=True,
        )
    assert provider.transport_calls == 0
    assert provider.budget is None


def approved_contract(tmp_path: Path) -> tuple[Path, str]:
    document = {
        "kind": "percept-launch-qualification-contract",
        "version": 1,
        "status": "approved",
        "approved_by": "offline-test",
        "approved_at": "2026-08-01T00:00:00Z",
        "quality": {
            "recall_target": 0.99,
            "recall_confidence": 0.95,
            "recall_unit": "independent_ground_truth_occurrence",
            "precision_target": 0.99,
            "precision_confidence": 0.95,
            "precision_unit": "receiver_observed_logical_notification",
            "false_alerts_per_fully_labelled_negative_source_hour": 0.01,
            "misses_include": [
                "no_receiver_arrival",
                "no_successful_receiver_ack",
                "budget_suppressed_judgment",
            ],
        },
        "latency": {
            "primary": "ground_truth_occurrence_onset_to_successful_webhook_receiver_ack",
            "timely_deadline_s": 5.0,
        },
        "economy": {
            "absolute_ceiling_usd": 0.10,
            "denominator": "source_stream_hour",
            "scope": "all_in",
        },
        "statistics": {
            "minimum_zero_miss_independent_occurrences_per_unadjusted_cell": 299,
            "minimum_zero_miss_independent_occurrences_per_12_cell_family": 546,
        },
        "supported_primary_cells": {"mechanics": ["event"], "source_classes": ["file"]},
    }
    path = tmp_path / "launch-contract.json"
    path.write_text(json.dumps(document, sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualification_requires_exact_approved_contract_hash_before_provider(tmp_path):
    corpus = frozen_fixture(tmp_path, cases=1)
    contract, digest = approved_contract(tmp_path)
    rejected = FakeBudgetedProvider([])
    with pytest.raises(RV.BaselineInputError, match="hash mismatch"):
        paid_run(
            tmp_path,
            corpus,
            rejected,
            checkpoint_path=tmp_path / "wrong.json",
            qualification_mode=True,
            contract_path=contract,
            approved_contract_sha256="0" * 64,
        )
    assert rejected.budget is None
    accepted_provider = FakeBudgetedProvider([accepted()])
    result = paid_run(
        tmp_path,
        corpus,
        accepted_provider,
        checkpoint_path=tmp_path / "approved.json",
        qualification_mode=True,
        contract_path=contract,
        approved_contract_sha256=digest,
    )
    assert result["qualification"]["mode"] == "qualification"
    assert result["qualification"]["approved_launch_contract"]["sha256"] == digest


def test_credential_values_are_redacted_from_result_and_checkpoint(tmp_path, monkeypatch):
    secret = "offline-secret-must-never-be-written"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    corpus = frozen_fixture(tmp_path, cases=1)
    provider = FakeBudgetedProvider([RuntimeError(f"provider echoed {secret}")])
    checkpoint = tmp_path / "redacted.json"
    result = paid_run(tmp_path, corpus, provider, checkpoint_path=checkpoint)
    assert secret not in json.dumps(result)
    assert secret not in checkpoint.read_text()
    assert "[REDACTED_CREDENTIAL]" in checkpoint.read_text()


def test_protocol_stamp_refuses_implicit_or_mislabelled_chunking(tmp_path):
    with pytest.raises(RV.BaselineInputError, match="chunk_duration_s"):
        RV.ComparisonProtocolStamp(mode="chunked")
    declared = RV.ComparisonProtocolStamp(
        mode="chunked",
        chunk_duration_s=8.0,
        chunk_overlap_s=2.0,
        source_clock_offset_mapping="chunk-relative plus exact source start offset",
        global_dedup_scoring="maximum-cardinality one-to-one on source clock",
    )
    corpus = frozen_fixture(tmp_path, cases=1)
    provider = FakeBudgetedProvider([])
    with pytest.raises(RV.BaselineInputError, match="whole_frozen_case only"):
        paid_run(tmp_path, corpus, provider, protocol=declared)
    assert provider.budget is None


def test_shipped_event_chain_schema_rejects_out_of_source_bounds():
    valid = json.dumps(
        {
            "satisfied": True,
            "confidence": 0.9,
            "evidence": "offline",
            "action_present": True,
            "phase": "post",
            "events": [{"t_start": 1.0, "t_end": None, "label": "event"}],
        }
    )
    assert RV.parse_percept_event_chain_response(valid, 10.0) == ((1.0, 1.0),)
    invalid = valid.replace('"t_start": 1.0', '"t_start": 11.0')
    with pytest.raises(RV.BaselineInputError, match="out of bounds"):
        RV.parse_percept_event_chain_response(invalid, 10.0)


def test_default_cli_never_imports_provider_adapter(tmp_path, monkeypatch):
    corpus = frozen_fixture(tmp_path, cases=1)
    output = tmp_path / "preflight.json"
    real_import = __import__
    imported = []

    def guarded_import(name, *args, **kwargs):
        if name == "raw_vlm_percept_adapter":
            imported.append(name)
            raise AssertionError("default CLI imported paid provider")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    assert RV.main(
        [
            "--corpus", str(corpus.corpus_path),
            "--authoring", str(corpus.authoring_path),
            "--repo-root", str(tmp_path),
            "--out", str(output),
        ]
    ) == 0
    assert imported == []
    assert json.loads(output.read_text())["provider_called"] is False


def test_paid_cli_main_module_shares_exact_response_type_without_transport(
    tmp_path,
    monkeypatch,
):
    corpus = frozen_fixture(tmp_path, cases=1)
    output = tmp_path / "paid-cli.json"
    checkpoint = tmp_path / "paid-cli.checkpoint.json"
    fake_adapter = ModuleType("raw_vlm_percept_adapter")

    class OfflineCliProvider:
        supports_native_video = True
        budget_reservation_before_transport = True

        def __init__(self, *, backend, model):
            self.backend = backend
            self.model = model
            self.budget = None

        def configure_request_budget(self, request_budget):
            self.budget = request_budget

        def complete(self, request):
            assert self.budget is not None
            reservation = self.budget.begin_attempt()
            usage = {"input_tokens": 11, "output_tokens": 3}
            self.budget.complete_unpriced_attempt(reservation, usage)
            baseline_module = sys.modules["raw_vlm_baseline"]
            return baseline_module.ProviderResponse(
                raw_response='{"events":[]}',
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )

    fake_adapter.PerceptBackendProvider = OfflineCliProvider
    monkeypatch.setitem(sys.modules, "raw_vlm_percept_adapter", fake_adapter)
    monkeypatch.delitem(sys.modules, "raw_vlm_baseline", raising=False)
    for name in RV._CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    profile_sha = hashlib.sha256(b"offline profile").hexdigest()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(RV.__file__).resolve()),
            "--corpus", str(corpus.corpus_path),
            "--authoring", str(corpus.authoring_path),
            "--repo-root", str(tmp_path),
            "--out", str(output),
            "--execute-paid",
            "--confirm-paid-run",
            "--backend", "gemini",
            "--model", "gemini-offline-model",
            "--pricing-version", "offline-prices-v1",
            "--pricing-source", "offline immutable test fixture",
            "--input-usd-per-million-tokens", "1.0",
            "--output-usd-per-million-tokens", "2.0",
            "--max-usd", "0.10",
            "--max-calls", "1",
            "--conservative-usd-per-call", "0.04",
            "--request-timeout-s", "5.0",
            "--checkpoint", str(checkpoint),
            "--profile-name", "offline-profile",
            "--profile-sha256", profile_sha,
            "--profile-source", "offline test fixture",
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("raw_vlm_baseline", run_name="__main__", alter_sys=True)

    assert stopped.value.code == 0
    report = json.loads(output.read_text())
    assert report["receiver_independent_quality"]["accepted_calls"] == 1
    assert report["receiver_independent_quality"]["provider_error_calls"] == 0
    assert report["calls"][0]["response"]["raw_response"] == '{"events":[]}'


def test_percept_adapter_reuses_shipped_metered_native_video_transport():
    from raw_vlm_percept_adapter import PerceptBackendProvider, _RawGeminiVision

    transport = []

    class Parts:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"data": data, "mime_type": mime_type}

    class Types:
        Part = Parts

        @staticmethod
        def GenerateContentConfig(**kwargs):
            return kwargs

    class Models:
        async def generate_content(self, **kwargs):
            transport.append(kwargs)
            usage = SimpleNamespace(
                prompt_token_count=12,
                tool_use_prompt_token_count=None,
                candidates_token_count=4,
                thoughts_token_count=None,
            )
            return SimpleNamespace(text='{"events":[]}', usage_metadata=usage)

    client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    vision = _RawGeminiVision(
        settings=SimpleNamespace(
            models=SimpleNamespace(vision_model="gemini-offline-model")
        ),
        client=client,
        transport_types=Types,
    )
    provider = PerceptBackendProvider(
        backend="gemini", model="gemini-offline-model", vision=vision
    )
    ledger = RV.PaidBudgetLedger(policy(), pricing())
    provider.configure_request_budget(ledger)
    request = RV.RawVlmRequest(
        case_id="offline",
        condition="event",
        source_path="source.mp4",
        source_sha256=hashlib.sha256(b"video").hexdigest(),
        duration_s=10.0,
        media_bytes=b"video",
        prompt="exact offline prompt",
        prompt_sha256=hashlib.sha256(b"exact offline prompt").hexdigest(),
        backend="gemini",
        model="gemini-offline-model",
    )
    response = provider.complete(request)
    assert response.raw_response == '{"events":[]}'
    assert (response.input_tokens, response.output_tokens) == (12, 4)
    assert transport[0]["model"] == "gemini-offline-model"
    assert transport[0]["contents"][0] == {"data": b"video", "mime_type": "video/mp4"}
    assert transport[0]["contents"][1] == "exact offline prompt"
    assert ledger.snapshot()["physical_attempts"] == 1
    assert ledger.snapshot()["inflight"] == []
