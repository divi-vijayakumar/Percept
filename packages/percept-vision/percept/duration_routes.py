"""Executable contract for sustained-duration frame Watches.

The public duration shape is deliberately small: a positive ``Goal.state.hold_s`` may run only on
an ordinary frame Watch. The in-process worker additionally requires a positive fixed observation
cadence; bring-your-own-capture callers already own observation timing and may omit it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .behavior_routes import BehaviorRoute, behavior_route, goal_behavior_route


UNSUPPORTED_DURATION_ROUTE = "unsupported_duration_route"
_MISSING = object()
_CADENCE_KEYS = frozenset({"min_judge_interval", "gate_on"})


@dataclass(frozen=True)
class DurationRouteIssue:
    """One exact duration-shape violation."""

    path: str
    code: str
    message: str
    goal_id: str | None = None


class UnsupportedDurationRouteError(ValueError):
    """A duration Watch cannot execute on the selected runtime seam."""

    def __init__(self, issues: Iterable[DurationRouteIssue], *, driver: str) -> None:
        self.issues = tuple(issues)
        self.driver = driver
        rendered = "; ".join(
            f"{issue.path}: {issue.message}"
            + (f" (goal {issue.goal_id!r})" if issue.goal_id is not None else "")
            for issue in self.issues
        )
        super().__init__(f"{driver} rejected sustained-duration Watch shape: {rendered}")


def _positive_finite(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def duration_route_issues(
    state: Any,
    *,
    goal_id: str | None = None,
    require_worker_cadence: bool,
    canonical_goal: bool = False,
    validate_cadence_without_hold: bool = False,
) -> tuple[DurationRouteIssue, ...]:
    """Validate one raw WatchPlan state or canonical armed Goal state.

    Explicit cadence is always shape-checked. Once a hold is authored, route, scheduling, and fixed
    worker admission also fail closed instead of being coerced to immediate firing or motion-only
    admission.
    """

    if not isinstance(state, Mapping):
        return ()

    has_hold = "hold_s" in state
    has_cadence = "cadence" in state
    if not has_hold and (not has_cadence or not validate_cadence_without_hold):
        return ()

    issues: list[DurationRouteIssue] = []
    hold = state.get("hold_s")
    if has_hold and not _positive_finite(hold):
        issues.append(
            DurationRouteIssue(
                path="Goal.state.hold_s",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message="must be a finite real number greater than zero",
            )
        )

    route = goal_behavior_route(_GoalState(state)) if canonical_goal else behavior_route(state)
    if has_hold and route is not BehaviorRoute.FRAME_WATCH:
        issues.append(
            DurationRouteIssue(
                path="Goal.state.hold_s",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message=(
                    "is supported only on an ordinary frame Watch, "
                    f"not route {route.value!r}"
                ),
            )
        )

    if has_hold and state.get("schedule") is not None:
        issues.append(
            DurationRouteIssue(
                path="Goal.state.schedule",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message="cannot be combined with hold_s until schedule-boundary reset is executable",
            )
        )

    cadence = state.get("cadence", _MISSING)
    if cadence is _MISSING:
        if has_hold and require_worker_cadence:
            issues.append(
                DurationRouteIssue(
                    path="Goal.state.cadence.min_judge_interval",
                    code=UNSUPPORTED_DURATION_ROUTE,
                    goal_id=goal_id,
                    message="is required for a worker-driven hold_s Watch",
                )
            )
        return tuple(issues)

    if not isinstance(cadence, Mapping):
        issues.append(
            DurationRouteIssue(
                path="Goal.state.cadence.min_judge_interval",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message="cadence must be a mapping with a positive finite interval",
            )
        )
        return tuple(issues)

    unknown = set(cadence) - _CADENCE_KEYS
    if unknown:
        issues.append(
            DurationRouteIssue(
                path="Goal.state.cadence.gate_on",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message=f"cadence contains unsupported field(s) {sorted(unknown)!r}",
            )
        )

    gate_on = cadence.get("gate_on", "motion")
    interval = cadence.get("min_judge_interval", _MISSING)
    if interval is _MISSING and (
        (has_hold and require_worker_cadence) or gate_on == "fixed"
    ):
        issues.append(
            DurationRouteIssue(
                path="Goal.state.cadence.min_judge_interval",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message="must be present and a finite real number greater than zero",
            )
        )
    elif interval is not _MISSING and not _positive_finite(interval):
        issues.append(
            DurationRouteIssue(
                path="Goal.state.cadence.min_judge_interval",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message="must be a finite real number greater than zero",
            )
        )

    if type(gate_on) is not str or gate_on not in {
        "motion", "structural", "semantic", "fixed"
    }:
        issues.append(
            DurationRouteIssue(
                path="Goal.state.cadence.gate_on",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message="must name motion, structural, semantic, or fixed admission",
            )
        )
    elif has_hold and gate_on != "fixed":
        issues.append(
            DurationRouteIssue(
                path="Goal.state.cadence.gate_on",
                code=UNSUPPORTED_DURATION_ROUTE,
                goal_id=goal_id,
                message="must be exactly 'fixed' for a sustained-duration Watch",
            )
        )
    return tuple(issues)


@dataclass(frozen=True)
class _GoalState:
    """Minimal Goal-shaped wrapper for canonical-route validation."""

    state: Any


def require_duration_routes(
    goals: Iterable[Any],
    *,
    driver: str,
    require_worker_cadence: bool,
) -> None:
    """Raise one controlled error containing every invalid selected duration Watch."""

    issues: list[DurationRouteIssue] = []
    for goal in goals:
        issues.extend(
            duration_route_issues(
                getattr(goal, "state", None),
                goal_id=str(getattr(goal, "id", "<unknown>")),
                require_worker_cadence=require_worker_cadence,
                canonical_goal=True,
                validate_cadence_without_hold=True,
            )
        )
    if issues:
        raise UnsupportedDurationRouteError(issues, driver=driver)


def require_in_process_duration_topology(
    goals: Iterable[Any],
    *,
    driver: str,
    remote: bool,
) -> None:
    """Reject duration Watches on the remote edge contract, which lacks goal/cause attribution."""

    if not remote:
        return
    issues = [
        DurationRouteIssue(
            path="Goal.state.hold_s",
            code=UNSUPPORTED_DURATION_ROUTE,
            goal_id=str(getattr(goal, "id", "<unknown>")),
            message=(
                "requires the in-process Python admission topology; remote percept-edge "
                "does not carry ordered per-goal fixed-admission causes"
            ),
        )
        for goal in goals
        if isinstance(getattr(goal, "state", None), Mapping)
        and "hold_s" in goal.state
    ]
    if issues:
        raise UnsupportedDurationRouteError(issues, driver=driver)


def duration_policy(state: Any) -> tuple[float | None, float | None]:
    """Return validated ``(hold_s, max_support_gap_s)`` for a canonical Goal state."""

    if not isinstance(state, Mapping) or "hold_s" not in state:
        return None, None
    issues = duration_route_issues(
        state,
        require_worker_cadence=False,
        canonical_goal=True,
    )
    if issues:
        raise UnsupportedDurationRouteError(issues, driver="duration_policy")
    hold = float(state["hold_s"])
    cadence = state.get("cadence")
    interval = (
        float(cadence["min_judge_interval"])
        if isinstance(cadence, Mapping) and "min_judge_interval" in cadence
        else None
    )
    return hold, None if interval is None else 2.0 * interval


__all__ = [
    "DurationRouteIssue",
    "UNSUPPORTED_DURATION_ROUTE",
    "UnsupportedDurationRouteError",
    "duration_policy",
    "duration_route_issues",
    "require_in_process_duration_topology",
    "require_duration_routes",
]
