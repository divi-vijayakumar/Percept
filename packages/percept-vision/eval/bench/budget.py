"""Fail-closed budget-plan validation for live programmable-Watch benchmarks."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BudgetPlanError(ValueError):
    """A live benchmark budget plan is absent, malformed, or does not match the provider."""


class BudgetExceeded(RuntimeError):
    """A request cannot enter the provider because its approved budget is exhausted."""


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise BudgetPlanError(f"{name} must be a non-empty object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BudgetPlanError(f"{name} must be a non-empty string")
    return value.strip()


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise BudgetPlanError(f"{name} must be positive")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BudgetPlanError(f"{name} must be a positive integer")
    return value


def _load(raw: str) -> dict[str, Any]:
    if not raw.strip():
        raise BudgetPlanError(
            "PERCEPT_BENCH_BUDGET_PLAN must contain a JSON object or path to a JSON file"
        )
    try:
        if raw.lstrip().startswith("{"):
            value = json.loads(raw)
        else:
            path = Path(raw).expanduser()
            if not path.is_file() or path.stat().st_size == 0:
                raise BudgetPlanError(f"budget plan file is missing or empty: {path}")
            value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise BudgetPlanError(f"budget plan is invalid JSON: {exc}") from exc
    return _object(value, "budget plan")


def validate_budget_plan(raw: str, *, provider: str, model: str | None = None) -> dict[str, Any]:
    """Validate the complete declared plan used by the live admission controller."""
    plan = _load(raw)
    if plan.get("version") != 1:
        raise BudgetPlanError("budget plan version must be 1")
    plan_provider = _text(plan.get("provider"), "provider")
    if plan_provider != provider:
        raise BudgetPlanError(
            f"budget provider {plan_provider!r} does not match configured provider {provider!r}"
        )
    plan_model = _text(plan.get("model"), "model")
    if model and plan_model != model:
        raise BudgetPlanError(
            f"budget model {plan_model!r} does not match configured model {model!r}"
        )
    _text(plan.get("region"), "region")

    prices = _object(plan.get("price_snapshot"), "price_snapshot")
    _text(prices.get("as_of"), "price_snapshot.as_of")
    if prices.get("currency") != "USD":
        raise BudgetPlanError("price_snapshot.currency must be USD")
    _text(prices.get("source"), "price_snapshot.source")
    for key in (
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "media_usd_per_minute",
    ):
        value = prices.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise BudgetPlanError(f"price_snapshot.{key} must be non-negative")

    caps = _object(plan.get("caps"), "caps")
    for key in ("usd", "media_minutes", "wall_clock_seconds", "request_timeout_seconds"):
        _positive(caps.get(key), f"caps.{key}")
    for key in ("requests", "tokens", "concurrency"):
        _positive_int(caps.get(key), f"caps.{key}")

    design = _object(plan.get("design"), "design")
    arms = design.get("arms")
    if not isinstance(arms, list) or len(arms) < 2 or any(not isinstance(x, str) or not x for x in arms):
        raise BudgetPlanError("design.arms must name at least two benchmark arms")
    seeds = design.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(x, bool) or not isinstance(x, int) for x in seeds):
        raise BudgetPlanError("design.seeds must be a non-empty integer list")
    retries = design.get("retries")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise BudgetPlanError("design.retries must be a non-negative integer")
    cache = _object(design.get("cache"), "design.cache")
    if not isinstance(cache.get("enabled"), bool):
        raise BudgetPlanError("design.cache.enabled must be boolean")
    _text(cache.get("policy"), "design.cache.policy")

    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise BudgetPlanError("stages must be a non-empty list")
    for index, stage_value in enumerate(stages):
        stage = _object(stage_value, f"stages[{index}]")
        _text(stage.get("name"), f"stages[{index}].name")
        stage_caps = _object(stage.get("caps"), f"stages[{index}].caps")
        for key, value in stage_caps.items():
            _positive(value, f"stages[{index}].caps.{key}")

    early_stop = _object(plan.get("early_stop"), "early_stop")
    if early_stop.get("stop_on_cap") is not True:
        raise BudgetPlanError("early_stop.stop_on_cap must be true")
    max_error_rate = early_stop.get("max_error_rate")
    if (
        isinstance(max_error_rate, bool)
        or not isinstance(max_error_rate, (int, float))
        or not math.isfinite(float(max_error_rate))
        or not 0 <= max_error_rate <= 1
    ):
        raise BudgetPlanError("early_stop.max_error_rate must be between 0 and 1")
    _positive_int(
        early_stop.get("min_successful_cases"),
        "early_stop.min_successful_cases",
    )

    approval = _object(plan.get("approval"), "approval")
    if approval.get("approved") is not True:
        raise BudgetPlanError("approval.approved must be true")
    for key in ("approved_by", "approved_at", "scope"):
        _text(approval.get(key), f"approval.{key}")

    enforcement = _object(plan.get("enforcement"), "enforcement")
    if enforcement.get("mode") != "reserve_per_attempt":
        raise BudgetPlanError("enforcement.mode must be 'reserve_per_attempt'")
    enforcement_stage = _text(enforcement.get("stage"), "enforcement.stage")
    stage_names = {stage["name"] for stage in stages}
    if enforcement_stage not in stage_names:
        raise BudgetPlanError(
            f"enforcement.stage {enforcement_stage!r} does not name a configured stage"
        )
    reserved_tokens = _positive_int(
        enforcement.get("tokens_per_attempt"),
        "enforcement.tokens_per_attempt",
    )
    reserved_usd = _positive(
        enforcement.get("usd_per_attempt"),
        "enforcement.usd_per_attempt",
    )
    if reserved_tokens > caps["tokens"]:
        raise BudgetPlanError("enforcement.tokens_per_attempt exceeds caps.tokens")
    if reserved_usd > caps["usd"]:
        raise BudgetPlanError("enforcement.usd_per_attempt exceeds caps.usd")
    return plan


@dataclass
class AttemptReservation:
    """One pessimistic request admission, reconciled only from provider-reported usage."""

    sequence: int
    tokens: int
    usd: float
    closed: bool = False


class BudgetEnforcer:
    """Thread-safe, fail-closed admission controller for physical provider attempts.

    Every physical attempt reserves its approved worst-case token and USD allowance before
    transport starts. A successful response releases unused allowance only when it carries
    input tokens, output tokens, and billed cost. Failed or unmetered attempts retain the full
    reservation because their spend is unknown.
    """

    def __init__(self, plan: dict[str, Any], *, clock=time.monotonic) -> None:
        self.plan = plan
        self._clock = clock
        self._started_at = clock()
        self._lock = threading.Lock()
        self._caps = dict(plan["caps"])
        selected_stage = next(
            stage
            for stage in plan["stages"]
            if stage["name"] == plan["enforcement"]["stage"]
        )
        for key, value in selected_stage["caps"].items():
            if key in self._caps:
                self._caps[key] = min(self._caps[key], value)
        self._reservation = plan["enforcement"]
        self._attempts = 0
        self._inflight = 0
        self._failed_attempts = 0
        self._metered_attempts = 0
        self._committed_tokens = 0
        self._committed_usd = 0.0
        self._input_tokens = 0
        self._output_tokens = 0
        self._billed_usd = 0.0
        self._unpriced_attempts = 0
        self._estimated_usd = 0.0

    @property
    def attempts_per_call(self) -> int:
        return int(self.plan["design"]["retries"]) + 1

    @property
    def request_timeout_s(self) -> float:
        return float(self._caps["request_timeout_seconds"])

    @property
    def media_minutes_cap(self) -> float:
        return float(self._caps["media_minutes"])

    def begin_attempt(self) -> AttemptReservation:
        """Atomically reserve one request before the transport can be invoked."""
        with self._lock:
            elapsed = self._clock() - self._started_at
            if elapsed >= float(self._caps["wall_clock_seconds"]):
                raise BudgetExceeded("wall-clock budget exhausted before provider attempt")
            if self._attempts + 1 > int(self._caps["requests"]):
                raise BudgetExceeded("request budget exhausted before provider attempt")
            if self._inflight + 1 > int(self._caps["concurrency"]):
                raise BudgetExceeded("concurrency budget exhausted before provider attempt")

            tokens = int(self._reservation["tokens_per_attempt"])
            usd = float(self._reservation["usd_per_attempt"])
            if self._committed_tokens + tokens > int(self._caps["tokens"]):
                raise BudgetExceeded("token budget exhausted before provider attempt")
            if self._committed_usd + usd > float(self._caps["usd"]) + 1e-12:
                raise BudgetExceeded("USD budget exhausted before provider attempt")

            self._attempts += 1
            self._inflight += 1
            self._committed_tokens += tokens
            self._committed_usd += usd
            return AttemptReservation(sequence=self._attempts, tokens=tokens, usd=usd)

    def complete_attempt(self, reservation: AttemptReservation, usage: dict[str, Any]) -> None:
        """Reconcile a successful request; missing/oversized usage fails closed."""
        if self._clock() - self._started_at >= float(self._caps["wall_clock_seconds"]):
            self.fail_attempt(reservation)
            raise BudgetExceeded(
                "wall-clock budget exhausted before provider response reconciliation"
            )
        try:
            input_tokens = int(usage["input_tokens"])
            output_tokens = int(usage["output_tokens"])
            billed_usd = float(usage["cost_usd"])
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            self.fail_attempt(reservation)
            raise BudgetExceeded(
                "provider response omitted enforceable input/output token or billed-cost usage"
            ) from exc
        actual_tokens = input_tokens + output_tokens
        if (
            input_tokens < 0
            or output_tokens < 0
            or not math.isfinite(billed_usd)
            or billed_usd < 0
        ):
            self.fail_attempt(reservation)
            raise BudgetExceeded(
                "provider returned invalid token or billed-cost usage"
            )

        with self._lock:
            self._ensure_open(reservation)
            reservation.closed = True
            self._inflight -= 1
            if actual_tokens > reservation.tokens or billed_usd > reservation.usd + 1e-12:
                self._committed_tokens += actual_tokens - reservation.tokens
                self._committed_usd += billed_usd - reservation.usd
                self._input_tokens += input_tokens
                self._output_tokens += output_tokens
                self._billed_usd += billed_usd
                self._failed_attempts += 1
                raise BudgetExceeded(
                    "provider usage exceeded the approved per-attempt token/USD reservation"
                )
            self._committed_tokens += actual_tokens - reservation.tokens
            self._committed_usd += billed_usd - reservation.usd
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._billed_usd += billed_usd
            self._metered_attempts += 1

    def fail_attempt(self, reservation: AttemptReservation) -> None:
        """Close a failed attempt while retaining its pessimistic token/USD reservation."""
        with self._lock:
            if reservation.closed:
                return
            reservation.closed = True
            self._inflight -= 1
            self._failed_attempts += 1

    def complete_unpriced_attempt(
        self, reservation: AttemptReservation, usage: dict[str, Any]
    ) -> None:
        """Close token-metered success when the provider supplies no billed USD.

        Keep the approved USD reservation committed rather than presenting a
        plan estimate as a bill.  The optional estimate is diagnostic only.
        """
        try:
            input_tokens = int(usage["input_tokens"])
            output_tokens = int(usage["output_tokens"])
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            self.fail_attempt(reservation)
            raise BudgetExceeded("provider response omitted enforceable token usage") from exc
        if input_tokens < 0 or output_tokens < 0:
            self.fail_attempt(reservation)
            raise BudgetExceeded("provider returned invalid token usage")
        estimate = usage.get("estimated_usd")
        if estimate is not None and (not isinstance(estimate, (int, float)) or not math.isfinite(float(estimate)) or float(estimate) < 0):
            self.fail_attempt(reservation)
            raise BudgetExceeded("provider returned invalid estimated usage")
        with self._lock:
            self._ensure_open(reservation)
            reservation.closed = True
            self._inflight -= 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._unpriced_attempts += 1
            self._estimated_usd += float(estimate or 0.0)

    @staticmethod
    def _ensure_open(reservation: AttemptReservation) -> None:
        if reservation.closed:
            raise RuntimeError(f"attempt reservation {reservation.sequence} already closed")

    def snapshot(self) -> dict[str, int | float | bool]:
        with self._lock:
            return {
                "physical_attempts": self._attempts,
                "metered_attempts": self._metered_attempts,
                "failed_attempts": self._failed_attempts,
                "inflight_attempts": self._inflight,
                "committed_tokens": self._committed_tokens,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "committed_usd": self._committed_usd,
                "billed_usd": self._billed_usd,
                "billed_usd_known": self._unpriced_attempts == 0,
                "unpriced_attempts": self._unpriced_attempts,
                "estimated_usd": self._estimated_usd,
            }


def build_budget_enforcer(
    raw: str,
    *,
    provider: str,
    model: str | None = None,
    clock=time.monotonic,
) -> BudgetEnforcer:
    """Validate a plan and construct its fresh per-process admission controller."""
    return BudgetEnforcer(
        validate_budget_plan(raw, provider=provider, model=model),
        clock=clock,
    )


def live_budget_blockers(*, provider: str, model: str | None = None, raw: str | None) -> list[str]:
    """Return validation and enforcement blockers before any provider is constructed."""
    try:
        plan = validate_budget_plan(raw or "", provider=provider, model=model)
    except BudgetPlanError as exc:
        return [f"budget plan invalid: {exc}"]
    BudgetEnforcer(plan)
    return []


__all__ = [
    "AttemptReservation",
    "BudgetEnforcer",
    "BudgetExceeded",
    "BudgetPlanError",
    "build_budget_enforcer",
    "live_budget_blockers",
    "validate_budget_plan",
]
