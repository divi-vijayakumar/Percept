"""Paid-arm preflight: wrong mechanics must fail before provider work starts."""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import live_run as LR  # noqa: E402
from percept.conductor import AuthoredPrimitive  # noqa: E402


def primitive(text: str, behavior_type: str, explicit: dict | None = None) -> AuthoredPrimitive:
    return AuthoredPrimitive(
        goal_text=text,
        behavior_type=behavior_type,
        explicit=explicit or {},
        origin="cache",
    )


def provider_budget(tmp_path: pathlib.Path, *, max_usd: float = 2.0,
                    max_calls: int = 5, bound: float = 0.4,
                    prior: float = 98.0, official_settlement: bool = False,
                    prior_source: str = "raw-run.result.json",
                    prior_basis: str = "official-token-settled") -> LR.GlobalProviderBudget:
    return LR.GlobalProviderBudget.open(
        policy=LR.PaidRunPolicy(
            max_usd=max_usd,
            max_calls=max_calls,
            conservative_usd_per_call=bound,
            request_timeout_s=12.0,
        ),
        pricing=LR.PricingStamp(
            backend="gemini",
            model="gemini-test",
            pricing_version="test-v1",
            pricing_source="offline fixture",
            input_usd_per_million_tokens=1.0,
            output_usd_per_million_tokens=2.0,
        ),
        checkpoint_path=tmp_path / "provider-budget.json",
        global_prior_committed_usd=prior,
        global_prior_commitment_source=prior_source,
        global_prior_commitment_basis=prior_basis,
        settle_from_official_token_prices=official_settlement,
        resume=False,
        case_checkpoint_exists=False,
    )


def test_matching_authored_mechanics_pass_preflight():
    cases = [
        {"case_id": "s", "goal_text": "monitor open and closed", "goal_kind": "state"},
        {"case_id": "t", "goal_text": "door becomes open", "goal_kind": "transition"},
        {"case_id": "d", "goal_text": "holding a book", "goal_kind": "duration"},
        {"case_id": "c", "goal_text": "count distinct cups", "goal_kind": "count"},
    ]
    authored = [
        primitive("monitor open and closed", "watch", {"Goal.state.rearm": "each_event"}),
        primitive("door becomes open", "transition"),
        primitive("holding a book", "watch"),
        primitive("count distinct cups", "count"),
    ]
    assert LR.authored_plan_issues(cases, authored) == []


def test_wrong_primitive_is_named_and_never_silently_repaired():
    cases = [{"case_id": "count-1", "goal_text": "count cups", "goal_kind": "count"}]
    issues = LR.authored_plan_issues(cases, [primitive("count cups", "watch")])
    assert len(issues) == 1
    assert "count-1" in issues[0]
    assert "selected 'duration'" in issues[0]
    assert "expected declared goal_kind 'count'" in issues[0]


def test_pin_goal_text_identity_is_checked():
    cases = [{"case_id": "x", "goal_text": "expected", "goal_kind": "duration"}]
    issues = LR.authored_plan_issues(cases, [primitive("different", "watch")])
    assert issues == ["x: pin text 'different' != selected goal 'expected'"]


def test_plan_count_mismatch_fails_closed():
    cases = [{"case_id": "x", "goal_text": "expected", "goal_kind": "duration"}]
    assert LR.authored_plan_issues(cases, []) == [
        "authored plan count 0 != selected case count 1"
    ]


def test_authored_plain_watch_does_not_inherit_legacy_each_event_rearm():
    authored = primitive("holding a book", "watch")
    conductor_plan = LR.watch_plan_for("g", authored.goal_text, authored)
    legacy_plan = LR.watch_plan_for(
        "g", authored.goal_text, None, legacy_occurrence_arm=True
    )

    assert conductor_plan.rearm is None
    assert "Goal.state.rearm" not in conductor_plan.explicit
    assert legacy_plan.rearm == "each_event"


def test_post_merge_mechanic_drift_is_a_preflight_failure(monkeypatch):
    authored = primitive("holding a book", "watch")
    original = LR.watch_plan_for

    def contaminated(goal_id, condition, primitive, **_kwargs):
        clean = original(goal_id, condition, primitive, **_kwargs)
        return LR.WatchPlan(
            id=clean.id,
            condition=clean.condition,
            sense=clean.sense,
            rearm="each_event",
            explicit=clean.explicit,
        )

    monkeypatch.setattr(LR, "watch_plan_for", contaminated)
    issues = LR.authored_plan_armability_issues(
        [{"case_id": "d", "goal_text": authored.goal_text, "goal_kind": "duration"}],
        [authored],
    )

    assert len(issues) == 1
    assert "compiled mechanic drift after WatchPlan merge" in issues[0]
    assert "authored type='watch' rearm='on_false'" in issues[0]
    assert "effective type='watch' rearm='each_event'" in issues[0]


def test_duration_evidence_contract_has_no_universal_brief_prior_and_authorship_wins():
    authored = primitive(
        "holding a book",
        "watch",
        {"Goal.state.perception.n_frames": 3},
    )
    contract = LR.evidence_contract_for({"dataset": "charades", "goal_kind": "duration"})
    plan = LR.watch_plan_for("g", authored.goal_text, authored, evidence_contract=contract)

    assert contract["sense"] == {}
    assert "Goal.state.sense.brief_window_s" not in plan.explicit
    assert plan.explicit["Goal.state.perception.n_frames"] == 3
    assert plan.explicit["Goal.state.perception.temporal"] == "strip"


def test_checkpoint_dispatch_arms_cannot_be_mixed_or_silently_resumed():
    assert LR.checkpoint_dispatch_issue(
        [{"judge_dispatch": "async"}, {"judge_dispatch": "inline"}]
    ) == "checkpoint mixes judge-dispatch arms ['async', 'inline']"
    assert "differs from requested" in LR.checkpoint_dispatch_issue(
        [{"judge_dispatch": "inline"}], requested="async"
    )
    assert LR.checkpoint_dispatch_issue(
        [{"judge_dispatch": "inline"}], prior_meta_dispatch="inline"
    ) is None


def test_run_mode_threads_inline_dispatch_without_constructing_provider(monkeypatch):
    seen = []

    async def fake_run_case(case, mode, **kwargs):
        seen.append(kwargs["judge_dispatch"])
        return {"case_id": case["case_id"], "mode": mode, "error": None}

    class BudgetDouble:
        terminal = False

        @staticmethod
        def snapshot():
            return {}

    monkeypatch.setattr(LR, "run_case", fake_run_case)
    rows = asyncio.run(LR.run_mode(
        [{"case_id": "c", "dataset": "d", "goal_kind": "duration", "duration_s": 1}],
        "reflex",
        backend="gemini",
        concurrency=1,
        provider_budget=BudgetDouble(),
        judge_dispatch="inline",
        progress=False,
    ))
    assert seen == ["inline"]
    assert rows[0]["case_id"] == "c"


def test_transition_trace_contract_uses_transition_decisions():
    goal = type("GoalDouble", (), {"state": {"type": "transition"}})()
    contract = LR.primitive_trace_contract(goal)
    assert contract["completion_stage"] == "transition.decision"
    assert contract["decision_stage"] == "transition.decision"
    assert contract["passed"]({"outcome": "fire"}) is True
    assert contract["passed"]({"outcome": "armed"}) is False
    assert contract["degraded"]({"outcome": "judge_degraded"}) is True


def test_explicit_proactiveqa_source_contract_enables_ego_compensation():
    case = {"dataset": "proactiveqa"}
    motion = LR.source_motion_contract_for(case)
    settings = LR.settings_for_case(LR.load_settings(), case)

    assert motion["camera_motion"] == "ego"
    assert "EGO_videos" in motion["evidence"]
    assert settings.edge.ego_compensate is True


def test_preflight_rejects_declared_ego_source_without_compensation(monkeypatch):
    authored = primitive("holding a book", "watch")
    monkeypatch.setattr(LR, "settings_for_case", lambda settings, _case: settings)
    monkeypatch.setattr(LR, "situation_for", lambda _case: None)
    settings = LR.load_settings()
    settings = LR.replace(settings, edge=LR.replace(settings.edge, ego_compensate=False))

    issues = LR.authored_plan_armability_issues(
        [{"case_id": "ego", "dataset": "proactiveqa", "goal_text": authored.goal_text,
          "goal_kind": "duration"}],
        [authored],
        settings=settings,
    )

    assert any("compiled source-motion drift" in issue for issue in issues)


def test_proactiveqa_uses_seeded_handheld_situation_route_not_ad_hoc_plan_fields():
    case = {
        "case_id": "ego-duration",
        "dataset": "proactiveqa",
        "goal_text": "waste is being disposed",
        "goal_kind": "duration",
    }
    authored = primitive(case["goal_text"], "watch")
    situation = LR.situation_for(case)
    profile, overrides = LR.resolve_profile(
        *situation, registry=LR.watch_profile_registry()
    )
    plan = LR.watch_plan_for(
        "g", case["goal_text"], authored,
        evidence_contract=None if situation is not None else LR.evidence_contract_for(case),
    )
    settings = LR.settings_for_case(LR.load_settings(), case)

    assert situation == ("dynamic", "handheld", "transition")
    assert profile.name == "handheld"
    assert overrides == {
        "plan_mode": "window",
        "gate": {"require_false_first": True},
        "ego_compensate": True,
    }
    assert settings.edge.ego_compensate is True
    assert settings.gate.require_false_first is True
    assert profile.cadence.base_fps == 4.0
    assert profile.cadence.max_fps == 15.0
    assert "WatchProfile.salience_stat" not in plan.explicit
    assert not any(key.startswith("Goal.state.perception.") for key in plan.explicit)
    assert LR.authored_plan_armability_issues([case], [authored]) == []


def test_launch_qualification_refuses_to_start_without_approved_bar_contract():
    with pytest.raises(SystemExit, match="requires an approved --qualification-contract"):
        LR.main(["--qualification", "--no-webhook-probe"])


def test_launch_qualification_refuses_inline_diagnostic_dispatch_before_setup():
    with pytest.raises(SystemExit, match="qualification headline requires --judge-dispatch async"):
        LR.main([
            "--qualification",
            "--qualification-contract", "not-opened-before-dispatch-check.json",
            "--judge-dispatch", "inline",
            "--no-webhook-probe",
        ])


def test_paid_live_run_refuses_missing_explicit_global_budget_before_setup():
    with pytest.raises(SystemExit, match="paid live run requires explicit --max-provider-usd"):
        LR.main(["--no-webhook-probe"])


def test_official_settlement_requires_complete_pinned_price_stamp_before_setup():
    with pytest.raises(SystemExit, match="requires --price-in, --price-out"):
        LR.main([
            "--no-webhook-probe",
            "--max-provider-usd", "1",
            "--max-provider-calls", "1",
            "--conservative-usd-per-call", "1",
            "--global-prior-committed-usd", "0.21945",
            "--global-prior-commitment-source", "raw-run.result.json",
            "--global-prior-commitment-basis", "official-token-settled",
            "--settle-from-official-token-prices",
        ])


def test_provider_reservation_is_durable_before_transport_and_stamps_global_allocation(tmp_path):
    budget = provider_budget(tmp_path)

    reservation = budget.begin_attempt()

    artifact = json.loads((tmp_path / "provider-budget.json").read_text())
    assert artifact["global_allocation"] == {
        "prior_committed_usd": 98.0,
        "prior_commitment_provenance": {
            "source": "raw-run.result.json",
            "basis": "official-token-settled",
        },
        "remaining_before_live_run_usd": 2.0,
        "live_run_local_max_usd": 2.0,
        "unallocated_after_live_run_usd": 0.0,
    }
    assert artifact["state"]["physical_attempts"] == 1
    assert artifact["state"]["committed_usd"] == pytest.approx(0.4)
    assert artifact["state"]["inflight"] == [reservation]

    budget.complete_unpriced_attempt(reservation, {"input_tokens": 3, "output_tokens": 1})
    settled = json.loads((tmp_path / "provider-budget.json").read_text())["state"]
    assert settled["inflight"] == []
    assert settled["unpriced_attempts"] == 1
    assert settled["committed_usd"] == pytest.approx(0.4)


def test_completed_response_settles_to_exact_official_token_price_and_refunds_difference(tmp_path):
    budget = provider_budget(tmp_path, official_settlement=True)
    reservation = budget.begin_attempt()

    budget.complete_unpriced_attempt(
        reservation, {"input_tokens": 100_000, "output_tokens": 10_000}
    )

    state = json.loads((tmp_path / "provider-budget.json").read_text())["state"]
    assert state["metered_attempts"] == 1
    assert state["unpriced_attempts"] == 0
    assert state["input_tokens"] == 100_000
    assert state["output_tokens"] == 10_000
    assert state["billed_usd"] == pytest.approx(0.12)
    assert state["committed_usd"] == pytest.approx(0.12)


def test_incomplete_usage_and_errors_retain_full_reservations(tmp_path):
    budget = provider_budget(tmp_path, official_settlement=True)
    incomplete = budget.begin_attempt()
    failed = budget.begin_attempt()

    budget.complete_attempt(incomplete, {"input_tokens": 100})
    budget.fail_attempt(failed)

    state = budget.snapshot()
    assert state["metered_attempts"] == 0
    assert state["unpriced_attempts"] == 1
    assert state["failed_attempts"] == 1
    assert state["committed_usd"] == pytest.approx(0.8)


def test_provider_budget_is_atomic_across_concurrent_cases_and_modes(tmp_path):
    budget = provider_budget(tmp_path)

    def reserve_once():
        try:
            return budget.begin_attempt()
        except LR.ProviderBudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        reservations = [item for item in pool.map(lambda _i: reserve_once(), range(40)) if item]

    assert len(reservations) == 5
    state = budget.snapshot()
    assert state["physical_attempts"] == 5
    assert state["committed_usd"] == pytest.approx(2.0)
    assert len(state["inflight"]) == 5
    for reservation in reservations:
        budget.complete_unpriced_attempt(reservation, {})
    assert budget.snapshot()["committed_usd"] == pytest.approx(2.0)
    with pytest.raises(LR.ProviderBudgetExceeded):
        budget.begin_attempt()


def test_provider_budget_resume_recovers_inflight_as_failed_without_refunding(tmp_path):
    first = provider_budget(tmp_path)
    first.begin_attempt()
    policy = first.policy
    pricing = first.pricing

    resumed = LR.GlobalProviderBudget.open(
        policy=policy,
        pricing=pricing,
        checkpoint_path=tmp_path / "provider-budget.json",
        global_prior_committed_usd=98.0,
        global_prior_commitment_source="raw-run.result.json",
        global_prior_commitment_basis="official-token-settled",
        settle_from_official_token_prices=False,
        resume=True,
        case_checkpoint_exists=True,
    )

    provenance = resumed.provenance()
    assert provenance["recovered_inflight_attempts"] == 1
    assert provenance["state"]["failed_attempts"] == 1
    assert provenance["state"]["inflight"] == []
    assert provenance["state"]["committed_usd"] == pytest.approx(0.4)


def test_provider_budget_resume_preserves_prior_exact_token_settlement(tmp_path):
    first = provider_budget(tmp_path, official_settlement=True)
    reservation = first.begin_attempt()
    first.complete_attempt(
        reservation, {"input_tokens": 100_000, "output_tokens": 10_000, "cost_usd": 999.0}
    )

    resumed = LR.GlobalProviderBudget.open(
        policy=first.policy,
        pricing=first.pricing,
        checkpoint_path=tmp_path / "provider-budget.json",
        global_prior_committed_usd=98.0,
        global_prior_commitment_source="raw-run.result.json",
        global_prior_commitment_basis="official-token-settled",
        settle_from_official_token_prices=True,
        resume=True,
        case_checkpoint_exists=True,
    )

    assert resumed.snapshot()["committed_usd"] == pytest.approx(0.12)
    assert resumed.snapshot()["billed_usd"] == pytest.approx(0.12)


def test_provider_budget_refuses_allocation_beyond_shared_one_hundred_dollars(tmp_path):
    with pytest.raises(LR.ProviderBudgetInputError, match="total user cap"):
        provider_budget(tmp_path, max_usd=2.01, prior=98.0)


def test_reconciled_raw_token_cost_can_replace_stale_reservation_only_explicitly(tmp_path):
    prior = 0.21945
    budget = provider_budget(
        tmp_path,
        max_usd=99.78055,
        max_calls=1,
        bound=1.0,
        prior=prior,
        prior_source="raw-vlm-full+smokes.result.json",
        prior_basis="official-token-settled",
    )
    allocation = budget.provenance()["global_allocation"]
    assert allocation["prior_committed_usd"] == prior
    assert allocation["remaining_before_live_run_usd"] == pytest.approx(99.78055)
    assert allocation["prior_commitment_provenance"] == {
        "source": "raw-vlm-full+smokes.result.json",
        "basis": "official-token-settled",
    }
    with pytest.raises(LR.ProviderBudgetInputError, match="total user cap"):
        provider_budget(
            tmp_path / "too-much",
            max_usd=99.780551,
            prior=prior,
            prior_source="raw-vlm-full+smokes.result.json",
        )


def test_provider_token_estimate_cannot_pass_the_all_in_cost_gate():
    stage3 = {
        "terminal_outcomes": {"completed": 2},
        "input_tokens": 1_000,
        "output_tokens": 100,
        "logical_calls": 2,
        "physical_attempts": 2,
        "route_attempts": 2,
        "judge_calls_batches": 2,
        "finished": 2,
        "degraded": 0,
        "adapter_errors": 0,
        "adapter_cancelled": 0,
        "failed_route_calls": 0,
        "usage_rows": 2,
        "input_bytes": 100,
        "reflex_context_stamped": 2,
        "verdicts_with_evidence_text": 0,
    }
    out = LR.accounting(
        [{"model": "m", "media_end_s": 3600.0, "stages": {"stage3_judge": stage3}}],
        {"m": {"input_usd_per_million": 1.0, "output_usd_per_million": 2.0}},
    )

    assert out["source_stream_hours"] == 1.0
    assert out["scheduled_watch_hours"] == 1.0
    assert out["estimated_provider_cost_usd_per_source_hour"] == 0.0012
    assert out["provider_cost_estimate_is_lower_bound"] is False
    assert out["launch_cost_gate"]["ceiling_usd_per_source_hour"] == 0.10
    assert out["launch_cost_gate"]["eligible"] is False


def test_accounting_includes_spend_from_quality_ineligible_error_rows():
    stage3 = {
        "terminal_outcomes": {"provider_error": 1},
        "input_tokens": 1_000,
        "output_tokens": 0,
        "logical_calls": 1,
        "physical_attempts": 1,
        "route_attempts": 1,
        "judge_calls_batches": 1,
        "finished": 0,
        "degraded": 1,
        "adapter_errors": 1,
        "adapter_cancelled": 0,
        "failed_route_calls": 1,
        "usage_rows": 1,
        "input_bytes": 10,
        "reflex_context_stamped": 1,
        "verdicts_with_evidence_text": 0,
    }
    report = LR.mode_report(
        [{
            "case_id": "errored-after-spend",
            "error": "provider failed",
            "measurable": False,
            "media_end_s": 60.0,
            "model": "m",
            "stages": {"stage3_judge": stage3},
        }],
        {"m": {"input_usd_per_million": 1.0, "output_usd_per_million": 1.0}},
    )

    assert report["n_cases_measured"] == 0
    assert report["accounting"]["input_tokens"] == 1_000
    assert report["accounting"]["estimated_cost_usd"] == 0.001
    assert report["launch_recall_gate"]["eligible"] is False
    assert report["launch_recall_gate"]["minimum_independent_occurrences_with_zero_misses"] == 299
