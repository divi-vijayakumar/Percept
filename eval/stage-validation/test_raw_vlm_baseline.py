"""Offline contract tests for the strict raw-VLM comparison lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import raw_vlm_baseline as RV


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def frozen_fixture(tmp_path: Path, *, two_goals: bool = False):
    media = b"exact frozen source bytes"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    media_path = media_dir / "source.mp4"
    media_path.write_bytes(media)
    media_sha = hashlib.sha256(media).hexdigest()
    conditions = ["the red light turns on"]
    if two_goals:
        conditions.append("the door opens")
    cases = [
        {
            "case_id": f"fixture::{index}",
            "dataset": "fixture",
            "duration_s": 10.0,
            "goal_kind": "transition",
            "goal_text": condition,
            "positives": [[1.0, 2.0]],
            "video_path": "media/source.mp4",
            "video_sha256": media_sha,
        }
        for index, condition in enumerate(conditions)
    ]
    corpus_doc = {"corpus_version": "1.2", "n_cases": len(cases), "cases": cases}
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus_doc))
    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    authoring_doc = {
        "corpus": {
            "path": "corpus.json",
            "sha256": corpus_sha,
            "version": "1.2",
            "n_cases": len(cases),
            "n_unique_goals": len(conditions),
        },
        "entries": {
            str(index): {"goal_text": condition}
            for index, condition in enumerate(conditions)
        },
    }
    authoring_path = tmp_path / "authoring.json"
    authoring_path.write_text(json.dumps(authoring_doc))
    frozen = RV.preflight_frozen_corpus(
        corpus_path,
        authoring_path,
        repo_root=tmp_path,
    )
    return frozen, media_path, media


def pricing():
    return RV.PricingStamp(
        backend="fake-provider",
        model="fake-video-model-v1",
        pricing_version="fixture-prices-2026-08-01",
        pricing_source="offline test fixture",
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
    )


def test_same_bytes_same_conditions_one_goal_per_call_and_explicit_cost_denominators(tmp_path):
    frozen, _media_path, media = frozen_fixture(tmp_path, two_goals=True)
    raw = '{"events":[{"start_s":1.1,"end_s":1.9}]}'
    provider = FakeProvider(
        [
            RV.ProviderResponse(raw, input_tokens=100, output_tokens=50),
            RV.ProviderResponse(raw, input_tokens=100, output_tokens=50),
        ]
    )
    ticks = iter([0.0, 0.25, 1.0, 1.5])
    result = RV.run_baseline(frozen, provider, pricing(), clock=lambda: next(ticks))

    assert len(provider.requests) == 2
    assert [request.condition for request in provider.requests] == [
        "the red light turns on",
        "the door opens",
    ]
    assert all(request.media_bytes == media for request in provider.requests)
    assert all(request.source_sha256 == hashlib.sha256(media).hexdigest()
               for request in provider.requests)
    assert all(call["request"]["goals_in_call"] == 1 for call in result["calls"])
    assert result["execution"]["default_goals_per_call"] == 1
    assert result["calls"][0]["response"]["raw_response"] == raw
    assert result["calls"][0]["provider_elapsed_s"] == pytest.approx(0.25)
    assert result["provider"] == {
        "backend": "fake-provider",
        "model": "fake-video-model-v1",
    }
    assert result["prompt"]["version"] == RV.DEFAULT_PROMPT_VERSION
    assert result["prompt"]["sha256"] == hashlib.sha256(
        RV.DEFAULT_PROMPT_TEMPLATE.encode()
    ).hexdigest()

    quality = result["receiver_independent_quality"]
    assert quality["boundary"].startswith("accepted raw provider response")
    assert quality["precision"] == 1.0
    assert quality["recall"] == 1.0
    assert result["not_measured"][0] == "webhook delivery"

    economy = result["economy"]
    # The same physical source is watched for two goals: source time is counted once, scheduled
    # watch time twice.  Two calls cost 2 * ((100 * $1 + 50 * $2) / 1M) = $0.0004.
    assert economy["source_stream_hours"] == pytest.approx(10.0 / 3600.0)
    assert economy["scheduled_watch_hours"] == pytest.approx(20.0 / 3600.0)
    assert economy["estimated_cost_usd"] == pytest.approx(0.0004)
    assert economy["cost_usd_per_source_stream_hour"] == pytest.approx(0.144)
    assert economy["cost_usd_per_scheduled_watch_hour"] == pytest.approx(0.072)


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        '{"events":[],"summary":"extra"}',
        '{"events":"not-a-list"}',
        '{"events":[{"start_s":true,"end_s":2.0}]}',
        '{"events":[{"start_s":NaN,"end_s":2.0}]}',
        '{"events":[{"start_s":1e10000,"end_s":2.0}]}',
        '{"events":[{"start_s":-0.1,"end_s":2.0}]}',
        '{"events":[{"start_s":3.0,"end_s":2.0}]}',
        '{"events":[{"start_s":9.0,"end_s":10.1}]}',
    ],
)
def test_malformed_or_out_of_bounds_response_is_rejected_not_clamped(tmp_path, raw):
    frozen, _media_path, _media = frozen_fixture(tmp_path)
    provider = FakeProvider([RV.ProviderResponse(raw, input_tokens=10, output_tokens=5)])
    result = RV.run_baseline(frozen, provider, pricing(), clock=lambda: 0.0)

    call = result["calls"][0]
    assert call["response"]["status"] == "rejected"
    assert call["response"]["raw_response"] == raw
    assert call["response"]["raw_response_retained"] is True
    assert call["response"]["parsed_events"] == []
    assert call["response"]["schema_error"]
    assert call["score"]["true_positives"] == 0
    assert call["score"]["false_negatives"] == 1
    assert result["receiver_independent_quality"]["recall"] == 0.0


def test_scorer_uses_maximum_cardinality_one_to_one_matching():
    scored = RV.score_temporal_events(
        [(1.0, 3.0), (3.0, 5.0)],
        # Broad prediction can match either truth; the narrow one can only match the first.  A
        # local greedy scorer can produce one TP here, while augmenting-path matching produces two.
        [(2.0, 4.0), (1.0, 2.0), (8.0, 9.0)],
        duration_s=10.0,
    )
    assert scored["matched_pairs"] == [
        {"prediction_index": 0, "ground_truth_index": 1},
        {"prediction_index": 1, "ground_truth_index": 0},
    ]
    assert scored["true_positives"] == 2
    assert scored["false_positives"] == 1
    assert scored["false_negatives"] == 0
    assert scored["precision"] == pytest.approx(2 / 3)
    assert scored["recall"] == 1.0


def test_only_point_ground_truth_receives_declared_tolerance():
    assert RV.score_temporal_events(
        [(5.0, 5.0)], [(4.5, 4.5)], duration_s=10.0, point_tolerance_s=1.0
    )["true_positives"] == 1
    assert RV.score_temporal_events(
        [(5.0, 5.0)], [(6.0, 6.0)], duration_s=10.0, point_tolerance_s=1.0
    )["true_positives"] == 1
    assert RV.score_temporal_events(
        [(5.0, 5.0)], [(4.5, 4.5)], duration_s=10.0, point_tolerance_s=0.0
    )["true_positives"] == 0
    assert RV.score_temporal_events(
        [(5.0, 6.0)], [(4.5, 4.5)], duration_s=10.0, point_tolerance_s=1.0
    )["true_positives"] == 0


def test_missing_usage_does_not_become_zero_cost(tmp_path):
    frozen, _media_path, _media = frozen_fixture(tmp_path)
    provider = FakeProvider(
        [
            RV.ProviderResponse(
                '{"events":[{"start_s":1.0,"end_s":2.0}]}',
                input_tokens=None,
                output_tokens=None,
            )
        ]
    )
    result = RV.run_baseline(frozen, provider, pricing(), clock=lambda: 0.0)
    assert result["receiver_independent_quality"]["recall"] == 1.0
    assert result["economy"]["usage_complete_for_all_calls"] is False
    assert result["economy"]["estimated_cost_usd"] is None
    assert result["economy"]["cost_usd_per_source_stream_hour"] is None
    assert result["economy"]["known_cost_lower_bound_usd"] == 0.0


def test_provider_error_is_a_quality_miss_and_unknown_cost_not_a_dropped_case(tmp_path):
    frozen, _media_path, _media = frozen_fixture(tmp_path)
    result = RV.run_baseline(
        frozen,
        FakeProvider([RuntimeError("offline fake failure")]),
        pricing(),
        clock=lambda: 0.0,
    )
    call = result["calls"][0]
    assert call["response"]["status"] == "provider_error"
    assert call["response"]["provider_error"] == "RuntimeError: offline fake failure"
    assert call["score"]["false_negatives"] == 1
    assert result["receiver_independent_quality"]["provider_error_calls"] == 1
    assert result["receiver_independent_quality"]["recall"] == 0.0
    assert result["economy"]["estimated_cost_usd"] is None


def test_execution_rehashes_and_refuses_changed_source_before_provider_call(tmp_path):
    frozen, media_path, _media = frozen_fixture(tmp_path)
    media_path.write_bytes(b"changed after preflight")
    provider = FakeProvider([])
    with pytest.raises(RV.BaselineInputError, match="changed after preflight"):
        RV.run_baseline(frozen, provider, pricing())
    assert provider.requests == []


def test_preflight_rejects_authoring_goal_drift(tmp_path):
    frozen, _media_path, _media = frozen_fixture(tmp_path)
    authoring = json.loads(frozen.authoring_path.read_text())
    authoring["entries"]["0"]["goal_text"] = "a different condition"
    frozen.authoring_path.write_text(json.dumps(authoring))
    with pytest.raises(RV.BaselineInputError, match="goal conditions differ"):
        RV.preflight_frozen_corpus(
            frozen.corpus_path,
            frozen.authoring_path,
            repo_root=tmp_path,
        )
