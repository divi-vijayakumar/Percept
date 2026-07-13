"""Goal registry — remembers the *goals* (standing state the conductor authors).

The :class:`~percept.adapters.Goal` DTO is frozen in ``adapters.py``; this module owns the
mutable registry and the op vocabulary the conductor emits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from .adapters import Goal
from .observability import get_logger

_log = get_logger("percept.goals")

GoalStatus = Literal["active", "complete", "cancelled"]


@dataclass
class GoalOp:
    """A single change to the registry, as authored by the conductor."""

    op: Literal["create", "update", "complete", "cancel"]
    goal: Goal | None = None  # for create
    goal_id: str | None = None  # for update/complete/cancel
    fields: dict = field(default_factory=dict)  # for update


class GoalRegistry:
    """Holds goals + their lifecycle status; ``apply`` folds in a batch of ops."""

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}
        self._status: dict[str, GoalStatus] = {}
        # Monotonic version bumped on every goal mutation. The runtime worker watches it to re-arm the
        # tier-0 gate when a goal is armed/cancelled MID-RUN (e.g. by voice) — the gate's WatchSpec is
        # otherwise compiled once at run() start, so a voice-armed goal never reaches the gate. (PR 2.1.)
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def apply(self, ops: list[GoalOp]) -> None:
        for op in ops:
            if op.op == "create":
                if op.goal is None:
                    raise ValueError("create op requires a goal")
                self.create(op.goal)
            elif op.op == "update":
                self._require(op.goal_id)
                goal = self._goals[op.goal_id]  # type: ignore[index]
                for k, v in op.fields.items():
                    setattr(goal, k, v)
                self._revision += 1
            elif op.op == "complete":
                self._set_status(op.goal_id, "complete")
            elif op.op == "cancel":
                self._set_status(op.goal_id, "cancelled")
            else:  # pragma: no cover — exhaustive over the Literal
                raise ValueError(f"unknown goal op {op.op!r}")

    def create(self, goal: Goal) -> Goal:
        # A bare dict write is last-writer-wins: re-using an id silently DROPS the prior goal (a lost
        # watch instruction — e.g. two voice conditions that slugged to the same id). Report it loudly;
        # never overwrite in silence. (D3b: the slug is now hash-suffixed, so a genuine collision here is
        # a real re-arm/override — still worth surfacing so it's never invisible.)
        prior = self._goals.get(goal.id)
        if prior is not None and self._status.get(goal.id) == "active":
            _log.warning(
                "goal_overwritten: id %r is already armed (%r); replacing it last-writer-wins with %r. "
                "If these are different watches, give them distinct ids — one instruction is being lost.",
                goal.id, getattr(prior, "condition", ""), getattr(goal, "condition", ""),
            )
        self._goals[goal.id] = goal
        self._status[goal.id] = "active"
        self._revision += 1
        return goal

    def _require(self, goal_id: str | None) -> None:
        if goal_id is None or goal_id not in self._goals:
            raise KeyError(f"no goal {goal_id!r}")

    def _set_status(self, goal_id: str | None, status: GoalStatus) -> None:
        self._require(goal_id)
        self._status[goal_id] = status  # type: ignore[index]
        self._revision += 1

    def status(self, goal_id: str) -> GoalStatus | None:
        return self._status.get(goal_id)

    def get(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def active(self) -> list[Goal]:
        return [g for gid, g in self._goals.items() if self._status.get(gid) == "active"]

    def all(self) -> list[Goal]:
        return list(self._goals.values())

    # -- persistence (durable goals; spec §9 / impl plan §7) -----------------
    def snapshot(self) -> list[dict]:
        return [
            {"goal": asdict(g), "status": self._status.get(gid, "active")}
            for gid, g in self._goals.items()
        ]

    def restore(self, items: list[dict]) -> None:
        self._goals = {}
        self._status = {}
        for item in items:
            goal = Goal(**item["goal"])
            self._goals[goal.id] = goal
            self._status[goal.id] = item.get("status", "active")


__all__ = ["GoalStatus", "GoalOp", "GoalRegistry"]
