"""Closed routing contract between authored behavior, tier-0 admission, and frame dispatch."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class BehaviorRoute(str, Enum):
    """The execution route selected by a Goal's authored behavior."""

    FRAME_WATCH = "frame_watch"
    FRAME_TRANSITION = "frame_transition"
    FRAME_PROCESS = "frame_process"
    EXTERNAL_EVENT_WINDOW = "external_event_window"
    EXTERNAL_BOUNDARY = "external_boundary"
    EXTERNAL_RHYTHM = "external_rhythm"
    EXTERNAL_SCHEDULER = "external_scheduler"
    UNSUPPORTED = "unsupported"


_TYPE_ROUTES = {
    "watch": BehaviorRoute.FRAME_WATCH,
    "transition": BehaviorRoute.FRAME_TRANSITION,
    "process": BehaviorRoute.FRAME_PROCESS,
    "event": BehaviorRoute.EXTERNAL_EVENT_WINDOW,
    "count": BehaviorRoute.EXTERNAL_BOUNDARY,
    "rhythm": BehaviorRoute.EXTERNAL_RHYTHM,
    "remind": BehaviorRoute.EXTERNAL_SCHEDULER,
    "scheduler": BehaviorRoute.EXTERNAL_SCHEDULER,
}
BEHAVIOR_TYPES = tuple(_TYPE_ROUTES)
_FRAME_ROUTES = frozenset(
    {
        BehaviorRoute.FRAME_WATCH,
        BehaviorRoute.FRAME_TRANSITION,
        BehaviorRoute.FRAME_PROCESS,
    }
)
_MISSING = object()


def authored_behavior_type(state: Any) -> Any:
    """Return the authored route token from raw or armed state, preserving unknown values."""

    if not isinstance(state, Mapping):
        return None
    authored = state.get("behavior", _MISSING)
    if authored is _MISSING:
        return state.get("type", "watch")
    if not isinstance(authored, Mapping):
        return authored
    return authored.get("type", state.get("type", "watch"))


def _safe_type_repr(value: Any) -> str:
    """Bounded, side-effect-free display for an authored route token."""

    if type(value) is str:
        return repr(value)
    return f"<{type(value).__name__}>"


def authored_behavior_type_display(state: Any) -> str:
    """Describe flat/nested authored route tokens without invoking hostile ``repr`` methods."""

    if not isinstance(state, Mapping):
        return f"<{type(state).__name__}>"
    authored = state.get("behavior", _MISSING)
    flat_present = "type" in state
    flat = state.get("type", "watch")
    if not isinstance(authored, Mapping) or "type" not in authored:
        return _safe_type_repr(flat if flat_present else authored_behavior_type(state))
    nested = authored.get("type")
    if (
        flat_present
        and type(flat) is str
        and type(nested) is str
        and flat != "watch"
        and nested != "watch"
        and flat != nested
    ):
        return f"flat={_safe_type_repr(flat)}, nested={_safe_type_repr(nested)}"
    return _safe_type_repr(nested)


def is_known_behavior_type(value: Any) -> bool:
    """Whether ``value`` names one of the closed behavior routes."""

    return type(value) is str and value in _TYPE_ROUTES


def behavior_route(state: Any) -> BehaviorRoute:
    """Classify a runtime or pre-coercion ``Goal.state`` without inventing a fallback route.

    ``runtime_compile`` carries authored behavior in ``state["behavior"]`` until apply-time
    coercion, while armed Goals carry the same fields flat in ``state``. This classifier accepts
    both representations so apply validation, tier-0 compilation, and frame dispatch use one
    closed route table.
    """

    if not isinstance(state, Mapping):
        return BehaviorRoute.UNSUPPORTED
    authored = state.get("behavior", _MISSING)
    flat_present = "type" in state
    flat_type = state.get("type", "watch")
    if flat_present and type(flat_type) is not str:
        return BehaviorRoute.UNSUPPORTED

    if authored is _MISSING:
        route_state = state
        behavior_type = flat_type
    elif not isinstance(authored, Mapping):
        return BehaviorRoute.UNSUPPORTED
    else:
        nested_present = "type" in authored
        nested_type = authored.get("type", flat_type)
        if nested_present and type(nested_type) is not str:
            return BehaviorRoute.UNSUPPORTED
        flat_nondefault = flat_present and flat_type != "watch"
        nested_nondefault = nested_present and nested_type != "watch"
        if flat_nondefault and nested_nondefault and flat_type != nested_type:
            return BehaviorRoute.UNSUPPORTED
        if nested_nondefault:
            behavior_type = nested_type
            route_state = authored
        else:
            behavior_type = flat_type
            route_state = state

    if type(behavior_type) is not str:
        return BehaviorRoute.UNSUPPORTED
    route = _TYPE_ROUTES.get(behavior_type, BehaviorRoute.UNSUPPORTED)
    if route is BehaviorRoute.FRAME_TRANSITION:
        before = route_state.get("from", state.get("from"))
        after = route_state.get("to", state.get("to"))
        if (
            type(before) is not str
            or not before.strip()
            or type(after) is not str
            or not after.strip()
        ):
            return BehaviorRoute.UNSUPPORTED
    return route


def goal_behavior_route(goal: Any) -> BehaviorRoute:
    """Return the closed execution route for a Goal-shaped object."""

    return behavior_route(getattr(goal, "state", None))


def needs_watch_condition(route: BehaviorRoute) -> bool:
    """Whether ``route`` is admitted and driven by the frame worker."""

    return route in _FRAME_ROUTES


def worker_can_drive(route: BehaviorRoute) -> bool:
    """Whether ``percept.run`` and ``GatedWatchLoop`` execute ``route``."""

    return route in _FRAME_ROUTES


@dataclass(frozen=True)
class UnsupportedBehavior:
    """One Goal whose behavior has no frame-worker driver."""

    goal_id: str
    route: BehaviorRoute
    authored_type: Any
    authored_type_display: str


class UnsupportedBehaviorRouteError(ValueError):
    """Raised before source consumption when a frame worker cannot drive its Goals."""

    def __init__(self, unsupported: Iterable[UnsupportedBehavior], *, driver: str) -> None:
        self.unsupported = tuple(unsupported)
        self.driver = driver
        rendered = ", ".join(
            f"{item.goal_id!r} (type={item.authored_type_display}, route={item.route.value})"
            for item in self.unsupported
        )
        external = all(item.route is not BehaviorRoute.UNSUPPORTED for item in self.unsupported)
        guidance = (
            " Use the matching direct event, boundary, rhythm, or scheduler primitive instead."
            if external
            else ""
        )
        super().__init__(f"{driver} cannot drive behavior route(s): {rendered}.{guidance}")


def require_worker_routes(goals: Iterable[Any], *, driver: str) -> None:
    """Fail before capture when any supplied Goal lacks a frame-worker execution route."""

    unsupported: list[UnsupportedBehavior] = []
    for goal in goals:
        route = goal_behavior_route(goal)
        if not worker_can_drive(route):
            unsupported.append(
                UnsupportedBehavior(
                    goal_id=str(getattr(goal, "id", "<unknown>")),
                    route=route,
                    authored_type=authored_behavior_type(getattr(goal, "state", None)),
                    authored_type_display=authored_behavior_type_display(
                        getattr(goal, "state", None)
                    ),
                )
            )
    if unsupported:
        raise UnsupportedBehaviorRouteError(unsupported, driver=driver)


__all__ = [
    "BEHAVIOR_TYPES",
    "BehaviorRoute",
    "UnsupportedBehavior",
    "UnsupportedBehaviorRouteError",
    "authored_behavior_type",
    "authored_behavior_type_display",
    "behavior_route",
    "goal_behavior_route",
    "is_known_behavior_type",
    "needs_watch_condition",
    "require_worker_routes",
    "worker_can_drive",
]
