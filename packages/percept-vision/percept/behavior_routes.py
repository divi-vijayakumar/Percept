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
#: Routes ``percept.run`` / :class:`~percept.harness.tier0.GatedWatchLoop` execute. A superset of
#: :data:`_FRAME_ROUTES`: a frame route is admitted by tier-0 and JUDGED, while ``EXTERNAL_RHYTHM``
#: is driven by the same frame loop but never becomes a :class:`~percept.adapters.WatchCondition` —
#: it is MEASURED per frame (decomposed channels) and counted by
#: :class:`~percept.rhythm.RhythmTracker`, spending no judge. Keeping the two sets distinct is what
#: lets :func:`needs_watch_condition` stay "compiles to a tier-0 admission condition" while
#: :func:`worker_can_drive` answers the different question "the worker has a driver for this".
_WORKER_ROUTES = frozenset(_FRAME_ROUTES | {BehaviorRoute.EXTERNAL_RHYTHM})
_MISSING = object()

#: The public, host-facing API that DRIVES each route ``percept.run`` does not.
#:
#: These three routes are **not unimplemented**. Each has a complete, shipped brain side that the
#: HOST feeds rather than the frame loop: the boundary counter is the "brain-counts" half of
#: edge-proposes·brain-counts·VLM-confirms, the event window is fed by pushed frames plus the
#: vision adapter's own ``detect_events``, and the scheduler is fed by a clock the host owns. So a
#: deployment that authored one of them and then saw nothing fire is missing WIRING, not a feature —
#: which is why arming one is a loud WARNING naming the exact call to make, never a refusal claiming
#: the goal is unsupported. ``{goal_id}`` is substituted with the armed goal's id.
_HOST_DRIVEN_ROUTES: dict[BehaviorRoute, str] = {
    BehaviorRoute.EXTERNAL_BOUNDARY: (
        "Drive it from the host: call Percept.perceive_boundary({goal_id!r}, t=...) once per counted "
        "occurrence (a detector edge, an audio onset, your own signal) — the count comes from that "
        "cheap signal, not from a per-frame judge. If you meant \"watch the frames and count "
        "repetitions of a body motion\", author type='rhythm' instead: the frame worker drives that one."
    ),
    BehaviorRoute.EXTERNAL_EVENT_WINDOW: (
        "Drive it from the host: call Percept.push_frame(frame, source=..., t=...) to fill the rolling "
        "window, then await Percept.perceive_events(source=...). Set Goal.state['event_type'] to the "
        "event name your Vision.detect_events emits — the goal fires only on a matching type. If you "
        "meant \"tell me whenever a plain-language condition becomes true\", author type='watch' with "
        "rearm='each_event' instead: the frame worker drives that one."
    ),
    BehaviorRoute.EXTERNAL_SCHEDULER: (
        "Drive it from the host: call Percept.schedule_every(interval_s, {goal_id!r}) or "
        "Percept.schedule_at(when_s, {goal_id!r}) to register the timer, then await Percept.tick(t=...) "
        "from your own clock. percept.run never ticks the scheduler, so a scheduler goal armed inside a "
        "frame run alone will not fire."
    ),
}

#: What to say when a route is not merely unwired but *undrivable* — nothing implements it and no
#: host feed exists, so there is no call the caller can make. Never claim a "matching primitive"
#: here: an unknown or malformed authored type has no primitive to match.
_UNSUPPORTED_REMEDY = (
    "No primitive implements this. Author one of the closed behavior types instead: "
    + ", ".join(repr(name) for name in BEHAVIOR_TYPES)
    + " (a 'transition' additionally requires non-empty 'from' and 'to' states). A goal still "
    "carrying a nested state['behavior'] block is the pre-apply compiler shape — arm it through "
    "percept.watchplan_apply.apply_watch_plan, which coerces it to the flat runtime shape."
)


class UndrivenBehaviorRouteWarning(RuntimeWarning):
    """An armed Goal whose route the frame worker does not drive, but the host can.

    Raised as a warning and not an error on purpose: :data:`_HOST_DRIVEN_ROUTES` are real, shipped
    capabilities whose driving API *requires the goal to already be armed*, so refusing the arm would
    delete those capabilities rather than protect anyone. A deployment that wants an unwired route to
    be fatal opts in explicitly::

        warnings.filterwarnings("error", category=percept.UndrivenBehaviorRouteWarning)
    """


def host_feed_api(route: BehaviorRoute) -> str | None:
    """The host-facing driving API for ``route``, or ``None`` when nothing can drive it."""

    return _HOST_DRIVEN_ROUTES.get(route)


def route_remedy(route: BehaviorRoute, *, goal_id: str = "<goal_id>") -> str:
    """What the caller can DO about ``route`` — the exact call, or the closed vocabulary.

    Every seam that refuses or flags a non-frame-worker route renders its guidance from here, so an
    apply rejection, a worker preflight failure, and an arm-time warning cannot drift into three
    different (or three differently wrong) recommendations.
    """

    feed = _HOST_DRIVEN_ROUTES.get(route)
    if feed is not None:
        return feed.format(goal_id=goal_id)
    if route is BehaviorRoute.UNSUPPORTED:
        return _UNSUPPORTED_REMEDY
    return ""


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
    """Return the route for a canonical, apply-coerced Goal-shaped object.

    Nested ``state["behavior"]`` is the pre-coercion WatchPlan compiler shape, never valid armed
    Goal state. Keep :func:`behavior_route` able to validate that raw preparation shape while making
    every runtime consumer fail closed if it escapes the apply boundary.
    """

    state = getattr(goal, "state", None)
    if isinstance(state, Mapping) and "behavior" in state:
        return BehaviorRoute.UNSUPPORTED
    return behavior_route(state)


def needs_watch_condition(route: BehaviorRoute) -> bool:
    """Whether ``route`` is admitted and driven by the frame worker."""

    return route in _FRAME_ROUTES


def worker_can_drive(route: BehaviorRoute) -> bool:
    """Whether ``percept.run`` and ``GatedWatchLoop`` execute ``route``.

    Broader than :func:`needs_watch_condition`: the frame routes are driven by the JUDGE path
    (tier-0 admission -> ``perceive_judged``), ``EXTERNAL_RHYTHM`` by the MEASUREMENT path
    (:class:`~percept.rhythm.RhythmDrive` -> ``core.perceive_rhythm``) on the very same frame loop.
    """

    return route in _WORKER_ROUTES


def drives_rhythm(route: BehaviorRoute) -> bool:
    """Whether ``route`` is executed by the frame loop's rhythm (counting) driver, not the judge."""

    return route is BehaviorRoute.EXTERNAL_RHYTHM


@dataclass(frozen=True)
class UnsupportedBehavior:
    """One Goal whose behavior has no frame-worker driver."""

    goal_id: str
    route: BehaviorRoute
    authored_type: Any
    authored_type_display: str


class UnsupportedBehaviorRouteError(ValueError):
    """Raised before source consumption when a frame worker cannot drive its Goals."""

    def __init__(
        self,
        unsupported: Iterable[UnsupportedBehavior],
        *,
        driver: str,
        verb: str = "drive",
    ) -> None:
        self.unsupported = tuple(unsupported)
        self.driver = driver
        rendered = ", ".join(
            f"{item.goal_id!r} (type={item.authored_type_display}, route={item.route.value})"
            for item in self.unsupported
        )
        # One remedy per DISTINCT route, rendered from the single route->remedy table so this message
        # cannot drift from the apply receipt's or the arm-time warning's advice.
        seen: list[BehaviorRoute] = []
        remedies: list[str] = []
        for item in self.unsupported:
            if item.route in seen:
                continue
            seen.append(item.route)
            remedy = route_remedy(item.route, goal_id=item.goal_id)
            if remedy:
                remedies.append(remedy)
        guidance = (" " + " ".join(remedies)) if remedies else ""
        super().__init__(f"{driver} cannot {verb} behavior route(s): {rendered}.{guidance}")


def require_worker_routes(goals: Iterable[Any], *, driver: str) -> None:
    """Fail before capture when any supplied Goal lacks a frame-worker execution route.

    The STRICT form: every supplied Goal must be one the frame worker itself executes. Callers that
    hand this their whole source partition are asserting "this driver drives all of these" — which is
    false for a host-driven route sharing a source key, and is why
    :func:`require_drivable_routes` exists.
    """

    unsupported = [
        _describe(goal) for goal in goals if not worker_can_drive(goal_behavior_route(goal))
    ]
    if unsupported:
        raise UnsupportedBehaviorRouteError(unsupported, driver=driver)


def require_drivable_routes(
    goals: Iterable[Any], *, driver: str
) -> tuple[UnsupportedBehavior, ...]:
    """Refuse only the UNDRIVABLE; hand back the host-driven Goals the caller must SKIP.

    The composition rule for a frame run, and the reason it is not
    :func:`require_worker_routes`:

    * ``UNSUPPORTED`` — no driver exists anywhere and no host API can feed it, so the goal cannot
      fire by any means a caller could reach. A run carrying one is incoherent. **Raise**, before a
      frame is pulled, exactly as before.
    * a **host-driven** route (:data:`_HOST_DRIVEN_ROUTES`) — a complete, shipped capability whose
      driver is the HOST, not this frame loop. It CAN fire; this driver is simply not the thing that
      fires it. Aborting the run for it destroys every unrelated goal the driver *can* execute — the
      client who followed our own advice ("drive a count goal with ``Percept.perceive_boundary``")
      and armed it beside a watch on the same source lost the watch, with no obvious cause. So it is
      **returned, not raised**: the caller excludes it from the partitions it drives (no judge is
      ever spent on it) and reports it through :func:`skipped_route_message`.

    The preflight's question is "can this goal ever fire?", not "is this run worthwhile?". Answering
    the second question with the first is what made one goal's route decide another goal's fate.
    A deployment that wants an unwired route to stay fatal keeps that: escalate
    :class:`UndrivenBehaviorRouteWarning` to an error and the caller's report becomes the raise.
    """

    undrivable, host_driven = classify_armed_routes(goals)
    if undrivable:
        raise UnsupportedBehaviorRouteError(undrivable, driver=driver)
    return host_driven


def _describe(goal: Any) -> UnsupportedBehavior:
    """One Goal rendered as the structured, repr-safe route record every seam reports."""

    state = getattr(goal, "state", None)
    return UnsupportedBehavior(
        goal_id=str(getattr(goal, "id", "<unknown>")),
        route=goal_behavior_route(goal),
        authored_type=authored_behavior_type(state),
        authored_type_display=authored_behavior_type_display(state),
    )


def classify_armed_routes(
    goals: Iterable[Any],
) -> tuple[tuple[UnsupportedBehavior, ...], tuple[UnsupportedBehavior, ...]]:
    """Split Goals into ``(undrivable, host_driven)`` — the whole raise-vs-warn decision.

    Named for its first caller (the arm seam), but the split is route-only and carries no arm-time
    state, so the frame-run preflight (:func:`require_drivable_routes`) reaches the same verdict from
    the same table rather than re-deriving a second, differently-wrong one.

    The split is the entire raise-vs-warn argument, in one place:

    * **undrivable** — :attr:`BehaviorRoute.UNSUPPORTED`. No driver exists anywhere in the SDK and no
      host API can feed it, so the goal cannot fire by any means a caller could reach. Arming it is a
      goal that is dead on arrival; the arm seam raises.
    * **host_driven** — a route the frame worker does not drive but a documented public API does
      (:data:`_HOST_DRIVEN_ROUTES`). Arming is legitimate and in fact *required* — every one of those
      APIs looks the goal up by id — so the arm seam warns loudly and names the call, and does not
      refuse.

    Frame-worker routes appear in neither tuple; arming those is silent and unchanged.
    """

    undrivable: list[UnsupportedBehavior] = []
    host_driven: list[UnsupportedBehavior] = []
    for goal in goals:
        item = _describe(goal)
        if worker_can_drive(item.route):
            continue
        (host_driven if host_feed_api(item.route) is not None else undrivable).append(item)
    return tuple(undrivable), tuple(host_driven)


def undriven_route_message(item: UnsupportedBehavior) -> str:
    """The arm-time warning a client reads for a host-driven route with no host feed wired."""

    return (
        f"undriven_behavior_route: goal {item.goal_id!r} armed behavior type "
        f"{item.authored_type_display} (route {item.route.value!r}), which the frame worker "
        "(percept.run / GatedWatchLoop) does not drive — frames alone will NEVER fire it. "
        + route_remedy(item.route, goal_id=item.goal_id)
        + " Nothing is wrong with the goal; it is armed and waiting for that call. To make an "
        "unwired route fatal instead, filter "
        "percept.UndrivenBehaviorRouteWarning to 'error'."
    )


def skipped_route_message(item: UnsupportedBehavior, *, driver: str, source: str) -> str:
    """The RUN-time report for a host-driven Goal a frame driver is skipping rather than aborting on.

    The arm-time twin (:func:`undriven_route_message`) says "nothing is feeding this yet". This one
    says the different thing a client needs at run time: *this run started, it is not driving your
    goal, the goals it can drive are running normally, and here is the call that drives yours.* Both
    render their remedy from :func:`route_remedy`, so neither can drift from the other.
    """

    return (
        f"undriven_behavior_route: {driver} is running source {source!r} but does NOT drive goal "
        f"{item.goal_id!r} (behavior type {item.authored_type_display}, route "
        f"{item.route.value!r}), so this run SKIPS it: no frame, no tier-0 admission and no judge "
        "call is ever spent on it, and it can never fire from frames. Every other goal on this "
        "source runs normally — one undriven route no longer aborts the run. "
        + route_remedy(item.route, goal_id=item.goal_id)
        + " To keep it out of this run's report entirely, bind it to its own source key "
        "(Goal(..., source='external')). To make an undriven route fatal to the run instead, "
        "filter percept.UndrivenBehaviorRouteWarning to 'error'."
    )


__all__ = [
    "BEHAVIOR_TYPES",
    "BehaviorRoute",
    "UndrivenBehaviorRouteWarning",
    "UnsupportedBehavior",
    "UnsupportedBehaviorRouteError",
    "authored_behavior_type",
    "authored_behavior_type_display",
    "behavior_route",
    "classify_armed_routes",
    "drives_rhythm",
    "goal_behavior_route",
    "host_feed_api",
    "is_known_behavior_type",
    "needs_watch_condition",
    "require_drivable_routes",
    "require_worker_routes",
    "route_remedy",
    "skipped_route_message",
    "undriven_route_message",
    "worker_can_drive",
]
