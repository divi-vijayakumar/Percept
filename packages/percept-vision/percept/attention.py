"""Attention: which conditions a frame makes due for a look.

The attention step of the runtime as one pure function, so the live :class:`~percept.monitor.Monitor`
and an offline evaluator decide when to look the same way. Given a frame and the explicit state
of one stream, :func:`step` applies, in order:

1. the processing cap (at most one processed frame per ``period`` seconds of stream time);
2. the :class:`~percept.triggers.QualityGate` (a corrupt or blurry frame is vetoed);
3. the :class:`~percept.triggers.SceneMonitor` (a scene reset suppresses triggers until the
   scene settles, then every active condition gets one settled look);
4. each trigger group, in order, and the priority of every look it makes due.

It returns a :class:`Step` describing everything that happened and the new :class:`State`. No
I/O, no clock (``frame.t`` is the time), no logging: a custom trigger that raises is reported in
:attr:`Step.failed_triggers` and its group is skipped for the frame. The caller applies the side
effects (health, blind intervals, fates, the look queue).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from percept.contracts import Condition, Frame
from percept.triggers import Every, OnChange, QualityGate, SceneMonitor

EVENT, HEARTBEAT, STATE = "event", "heartbeat", "state"     # the priority kind of a look
# The runtime's queue priority for each kind of look, lower first (verifications come before
# all of them, rechecks after).
PRIORITY = {EVENT: 2, STATE: 3, HEARTBEAT: 4}


def trigger_key(trigger: Any) -> Any:
    """Triggers group (and keep their state) by value when hashable, else by identity: two
    conditions with equal triggers share one step and one state."""
    try:
        hash(trigger)
    except TypeError:
        return ("id", id(trigger))
    return trigger


def trigger_label(trigger: Any) -> str:
    """A short, stable name for a trigger in coverage: ``on_change(0.2)``, ``every(5)``, or the
    class name of a custom trigger. Runtime looks use ``poke``, ``settled`` (the look after a
    scene reset), ``recheck`` and ``verify``."""
    if isinstance(trigger, OnChange):
        extra = "" if trigger.exit_threshold == trigger.threshold / 2 else f", exit={trigger.exit_threshold:g}"
        extra += "" if trigger.baseline_s == 5.0 else f", baseline_s={trigger.baseline_s:g}"
        return f"on_change({trigger.threshold:g}{extra})"
    if isinstance(trigger, Every):
        return repr(trigger)
    return type(trigger).__name__


def build(conditions: Iterable[Condition], *, default_change: Any, default_heartbeat: Any,
          paused: Iterable[str] = ()) -> tuple[list[tuple[Any, Any, tuple[Condition, ...]]], dict[Any, str]]:
    """The trigger groups and their labels for :func:`step`, from the conditions in order.

    A condition's trigger, or each member of a tuple of triggers, joins the group of its
    :func:`trigger_key`; a condition without a trigger uses ``default_change`` (an event) or
    ``default_change`` and ``default_heartbeat`` (a state). Pass the same default instances every
    time so their groups keep their state. Conditions whose id is in ``paused`` are left out.
    Returns ``(groups, labels)``: ``(key, trigger, conditions)`` per group, and each group's label."""
    skip = set(paused)
    groups: dict[Any, tuple[Any, list[Condition]]] = {}
    for cond in conditions:
        if cond.id in skip:
            continue
        if isinstance(cond.trigger, tuple):     # any of them
            triggers: tuple = cond.trigger
        elif cond.trigger is not None:
            triggers = (cond.trigger,)
        elif cond.kind == "event":
            triggers = (default_change,)
        else:
            triggers = (default_change, default_heartbeat)
        for trigger in triggers:
            group = groups.setdefault(trigger_key(trigger), (trigger, []))[1]
            if cond not in group:       # a trigger listed twice in one tuple steps once
                group.append(cond)
    return ([(k, t, tuple(cs)) for k, (t, cs) in groups.items()],
            {k: trigger_label(t) for k, (t, _) in groups.items()})


@dataclass(frozen=True)
class State:
    """One stream's attention state. ``due``: when the processing cap admits the next frame;
    ``quality`` and ``scene``: the gate's and the scene monitor's state; ``triggers``: each
    trigger group's state by group key; ``degraded``: the feed-level quality state last seen."""

    due: Optional[float] = None
    quality: Any = None
    scene: Any = None
    triggers: Mapping[Any, Any] = field(default_factory=dict)
    degraded: bool = False


@dataclass(frozen=True)
class Look:
    """A look made due by a trigger: ``kind`` is ``event``, ``heartbeat`` or ``state``;
    ``trigger`` is the group's label; ``t_trigger`` the frame's wall time."""

    condition_id: str
    kind: str
    trigger: Optional[str]
    t_trigger: float


@dataclass(frozen=True)
class TriggerFailure:
    """A trigger that raised on this frame; its group was skipped. ``before`` is the number of
    looks made due before it, so a caller can report it in order."""

    key: Any
    trigger: Any
    error: BaseException
    before: int


@dataclass(frozen=True)
class Step:
    """Everything one frame did to attention.

    ``capped``: dropped by the processing cap (nothing else ran). ``vetoed``: the quality gate
    refused the frame (the scene monitor and triggers did not run). ``reseeded``: a lasting
    blur became the new normal. ``degraded``: the new feed-level quality state when it changed
    on this frame, else None. ``scene_reset``, ``suppressed``, ``scene_ended`` and ``changed``:
    the scene monitor's signal. ``looks``: the looks made due, in order. ``failed_triggers``:
    triggers that raised. ``settled``: every active condition gets one settled look on this frame.
    """

    capped: bool = False
    vetoed: bool = False
    reseeded: bool = False
    degraded: Optional[bool] = None
    scene_reset: bool = False
    suppressed: bool = False
    scene_ended: bool = False
    changed: float = 0.0
    looks: tuple[Look, ...] = ()
    failed_triggers: tuple[TriggerFailure, ...] = ()
    settled: bool = False


def step(frame: Frame, state: State, *, groups: Sequence[tuple[Any, Any, tuple[Condition, ...]]],
         conditions: Mapping[str, Condition], labels: Mapping[Any, Optional[str]],
         quality: QualityGate, scene: SceneMonitor, period: float = 0.0) -> tuple[Step, State]:
    """Run attention on one frame. ``groups``: ``(key, trigger, conditions)`` per trigger group;
    ``conditions``: every condition by id (a due id that is not here makes no look);
    ``labels``: each group's trigger label by key; ``period``: the processing cap in seconds
    (0 for none)."""
    due = state.due
    if period:      # the processing cap
        if due is not None and frame.t < due - 0.1 * period:
            return Step(capped=True), state
        due = (frame.t if due is None or frame.t - due > period else due) + period
    verdict, qstate = quality.check(frame, state.quality, frame.t)
    degraded = verdict.degraded if verdict.degraded != state.degraded else None
    state = State(due, qstate, state.scene, state.triggers, verdict.degraded)
    if not verdict.ok:
        return Step(vetoed=True, reseeded=verdict.reseeded, degraded=degraded), state
    signal, sstate = scene.step(frame, state.scene, frame.t)
    base = dict(reseeded=verdict.reseeded, degraded=degraded, scene_reset=signal.reset,
                suppressed=signal.suppressed, scene_ended=signal.ended, changed=signal.changed)
    if signal.suppressed:
        return Step(**base), State(due, qstate, sstate, state.triggers, verdict.degraded)
    tstates = {} if signal.ended else dict(state.triggers)
    looks: list[Look] = []
    failed: list[TriggerFailure] = []
    for key, trigger, conds in groups:
        try:
            made_due, tstates[key] = trigger.step(frame, conds, tstates.get(key), frame.t)
        except Exception as exc:  # noqa: BLE001 - a broken custom trigger skips this frame
            failed.append(TriggerFailure(key, trigger, exc, len(looks)))
            continue
        label = labels.get(key)
        for cid in made_due:
            cond = conditions.get(cid)
            if cond is not None:
                kind = EVENT if cond.kind == "event" else HEARTBEAT if isinstance(trigger, Every) else STATE
                looks.append(Look(cid, kind, label, frame.t_wall))
    return (Step(**base, looks=tuple(looks), failed_triggers=tuple(failed), settled=signal.ended),
            State(due, qstate, sstate, tstates, verdict.degraded))
