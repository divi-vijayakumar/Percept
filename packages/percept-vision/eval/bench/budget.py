"""Fail-closed budget-plan validation for live programmable-Watch benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BudgetPlanError(ValueError):
    """A live benchmark budget plan is absent, malformed, or does not match the provider."""


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise BudgetPlanError(f"{name} must be a non-empty object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BudgetPlanError(f"{name} must be a non-empty string")
    return value.strip()


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
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
    """Validate the complete declared plan; this does not enforce spend at runtime."""
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
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
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
    return plan


def live_budget_blockers(*, provider: str, model: str | None = None, raw: str | None) -> list[str]:
    """Return validation and enforcement blockers before any provider is constructed."""
    blockers: list[str] = []
    try:
        validate_budget_plan(raw or "", provider=provider, model=model)
    except BudgetPlanError as exc:
        blockers.append(f"budget plan invalid: {exc}")
    # Validation describes intended spend. Until every request updates and checks those caps
    # atomically, it is not a spend control and cannot qualify live-stream economics.
    blockers.append(
        "hard spend enforcement is not implemented; Cloud provider cost qualification is BLOCKED"
    )
    return blockers


__all__ = [
    "BudgetPlanError",
    "live_budget_blockers",
    "validate_budget_plan",
]
