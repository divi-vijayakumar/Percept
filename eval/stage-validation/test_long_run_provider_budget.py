"""No-network proofs for the long-horizon paid-request boundary."""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import live_run as LIVE  # noqa: E402
import long_run as LONG  # noqa: E402
from percept.contracts import Verdict  # noqa: E402


def args(**overrides):
    values = {
        "max_provider_usd": 2.0,
        "max_provider_calls": 100,
        "conservative_usd_per_call": 0.005,
        "global_prior_committed_usd": 98.0,
        "global_prior_commitment_source": "prior-result.json",
        "global_prior_commitment_basis": "official-token-settled",
        "provider_request_timeout_s": 120.0,
        "price_in": 0.25,
        "price_out": 1.5,
        "price_source": "official pricing page",
        "price_version": "2026-07-21",
        "settle_from_official_token_prices": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def budget(tmp_path: pathlib.Path, *, max_usd=2.0, prior=98.0, resume=False):
    policy = LIVE.PaidRunPolicy(
        max_usd=max_usd,
        max_calls=100,
        conservative_usd_per_call=0.005,
        request_timeout_s=120.0,
    )
    pricing = LIVE.PricingStamp(
        backend="gemini",
        model="gemini-3.1-flash-lite",
        pricing_version="2026-07-21",
        pricing_source="official pricing page",
        input_usd_per_million_tokens=0.25,
        output_usd_per_million_tokens=1.5,
    )
    return LIVE.GlobalProviderBudget.open(
        policy=policy,
        pricing=pricing,
        checkpoint_path=tmp_path / "long.provider-budget.json",
        global_prior_committed_usd=prior,
        global_prior_commitment_source="prior-result.json",
        global_prior_commitment_basis="official-token-settled",
        settle_from_official_token_prices=True,
        resume=resume,
        case_checkpoint_exists=False,
    )


def test_long_policy_requires_explicit_prior_provenance_and_enforces_global_100_cap():
    with pytest.raises(LIVE.ProviderBudgetInputError, match="source"):
        LONG.provider_budget_policy_from_args(args(global_prior_commitment_source=""))
    with pytest.raises(LIVE.ProviderBudgetInputError, match=r"\$100\.00 total user cap"):
        LONG.provider_budget_policy_from_args(
            args(max_provider_usd=2.01, global_prior_committed_usd=98.0)
        )


def test_official_settlement_requires_a_complete_pinned_price_stamp():
    with pytest.raises(LIVE.ProviderBudgetInputError, match="price-version"):
        LONG.provider_budget_policy_from_args(args(price_version=""))


def test_wrapper_reserves_durably_before_transport_and_settles_exact_tokens(tmp_path):
    ledger = budget(tmp_path)
    checkpoint = tmp_path / "long.provider-budget.json"

    class FakeGemini:
        SUPPORTS_NATIVE_VIDEO_TRANSPORT = True
        PARSES_CLAIM_CHANNEL = True

        async def judge(self, condition, frame, **kwargs):
            reservation = ledger.begin_attempt()
            durable = json.loads(checkpoint.read_text())["state"]
            assert durable["physical_attempts"] == 1
            assert len(durable["inflight"]) == 1
            ledger.complete_unpriced_attempt(
                reservation, {"input_tokens": 1_000, "output_tokens": 100}
            )
            return Verdict(
                satisfied=True,
                confidence=0.9,
                evidence="seen",
                usage={"input_tokens": 1_000, "output_tokens": 100},
            )

    wrapper = LONG.BudgetedVision(FakeGemini(), request_budget=ledger)
    asyncio.run(wrapper.judge("condition", b"jpeg"))

    state = json.loads(checkpoint.read_text())["state"]
    assert state["physical_attempts"] == 1
    assert state["metered_attempts"] == 1
    assert state["unpriced_attempts"] == 0
    assert state["committed_usd"] == pytest.approx(0.0004)
    assert state["billed_usd"] == pytest.approx(0.0004)
    assert wrapper.calls == 1


def test_error_cancel_and_crash_paths_never_refund_the_reservation(tmp_path):
    ledger = budget(tmp_path)
    reservation = ledger.begin_attempt()
    ledger.fail_attempt(reservation)
    state = ledger.snapshot()
    assert state["failed_attempts"] == 1
    assert state["committed_usd"] == pytest.approx(0.005)

    ledger.begin_attempt()  # Simulated process death after durable reserve, before settlement.
    resumed = budget(tmp_path, resume=True)
    recovered = resumed.snapshot()
    assert recovered["failed_attempts"] == 2
    assert recovered["inflight"] == []
    assert recovered["committed_usd"] == pytest.approx(0.010)
    assert resumed.provenance()["recovered_inflight_attempts"] == 1


def test_exhausted_budget_refuses_before_fake_transport(tmp_path):
    ledger = budget(tmp_path, max_usd=0.005, prior=0.0)
    reservation = ledger.begin_attempt()
    ledger.fail_attempt(reservation)

    class NeverTransport:
        SUPPORTS_NATIVE_VIDEO_TRANSPORT = True
        PARSES_CLAIM_CHANNEL = True

        def __init__(self):
            self.transports = 0

        async def judge(self, condition, frame, **kwargs):
            reservation = ledger.begin_attempt()
            self.transports += 1
            ledger.fail_attempt(reservation)
            return Verdict(satisfied=False, confidence=1.0, evidence="")

    inner = NeverTransport()
    wrapper = LONG.BudgetedVision(inner, request_budget=ledger)
    with pytest.raises(LIVE.ProviderBudgetExceeded):
        asyncio.run(wrapper.judge("condition", b"jpeg"))
    assert inner.transports == 0
    assert wrapper.budget_capped == 1
    assert ledger.snapshot()["physical_attempts"] == 1
