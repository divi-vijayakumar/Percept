"""Goal registry — remembers the *goals* (standing state the conductor authors).

The :class:`~percept.adapters.Goal` DTO is frozen in ``adapters.py``; this module owns the
mutable registry and the op vocabulary the conductor emits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Literal

from .adapters import Goal, GoalOrigin

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
        # Revision-change subscribers — the AUTHORITATIVE mutation seam. Every accepted mutation path
        # (structured arm, language-authored create/safe-say-update, complete/cancel, restore)
        # routes through _bump(), so a subscriber fires on EVERY change — not just at RuntimeSession.arm,
        # which the rich `arm_from_language` path bypasses. Purpose: snapshot invalidation for consumers
        # (e.g. a voice grounding cache) that must never go stale on the language command path.
        self._subscribers: list[Callable[[int], None]] = []
        # One owning runtime may bind a required pre-commit invariant hook. It is deliberately singular:
        # multiple side-effecting pre-hooks could otherwise let an early hook mutate runtime state and a
        # later failure abort the registry commit.
        self._owner_mutation_hook: Callable[[tuple[GoalOp, ...]], None] | None = None

    @property
    def revision(self) -> int:
        return self._revision

    def subscribe_revision(self, cb: "Callable[[int], None]") -> "Callable[[], None]":
        """Register ``cb(revision)``, called AFTER every goal-registry mutation. Returns ``unsubscribe()``.

        Additive + ISOLATED: a subscriber that raises is swallowed (:meth:`_bump`), so one bad consumer
        can never break another or corrupt the mutation. This is the seam ``RuntimeSession.on_goal_change``
        binds to — subscribing here (not at ``arm``/``disarm``) is what makes it fire for the
        ``arm_from_language`` / ``apply_goal_ops`` path too."""
        self._subscribers.append(cb)

        def _unsub() -> None:
            try:
                self._subscribers.remove(cb)
            except ValueError:
                pass

        return _unsub

    def _bump(self) -> None:
        """Advance the revision and notify subscribers — the single choke point every mutation calls."""
        self._revision += 1
        rev = self._revision
        for cb in list(self._subscribers):
            try:
                cb(rev)
            except Exception:  # noqa: BLE001 - a revision subscriber must never break the mutation
                pass

    def _bind_owner_mutation_hook(
        self, cb: "Callable[[tuple[GoalOp, ...]], None]"
    ) -> None:
        """Bind the registry owner's required, no-fail pre-commit invariant hook exactly once."""
        if self._owner_mutation_hook is not None:
            raise RuntimeError("GoalRegistry already has an owner mutation hook")
        self._owner_mutation_hook = cb

    def validate(self, ops: list[GoalOp]) -> None:
        """Validate a complete batch without mutation.

        Goal ids are lifetime-unique inside a registry. Updates intentionally expose only ``say``:
        condition/source/state/id/origin change executable meaning while leaving gate, episode,
        detector, and runtime state behind.
        """
        known_ids = set(self._goals)
        for op in ops:
            if op.op == "create":
                if op.goal is None:
                    raise ValueError("create op requires a goal")
                if op.goal.id in known_ids:
                    raise ValueError(
                        f"goal id {op.goal.id!r} already exists; goal ids are lifetime-unique "
                        "within a registry. Use a new goal id."
                    )
                known_ids.add(op.goal.id)
                continue
            if op.op not in ("update", "complete", "cancel"):
                raise ValueError(f"unknown goal op {op.op!r}")
            if op.goal_id is None or op.goal_id not in known_ids:
                raise KeyError(f"no goal {op.goal_id!r}")
            if op.op == "update":
                unsupported = set(op.fields) - {"say"}
                if unsupported:
                    raise ValueError(
                        "goal update supports only the non-executable 'say' field; "
                        f"unsupported fields: {sorted(unsupported)!r}"
                    )
                if "say" in op.fields and not isinstance(op.fields["say"], str):
                    raise TypeError("goal update field 'say' must be a string")

    def apply(self, ops: list[GoalOp]) -> None:
        # Validate the WHOLE batch first. A bad later operation must not leave an earlier mutation.
        self.validate(ops)
        frozen_ops = tuple(
            GoalOp(op=op.op, goal=op.goal, goal_id=op.goal_id, fields=dict(op.fields))
            for op in ops
        )
        if self._owner_mutation_hook is not None:
            self._owner_mutation_hook(frozen_ops)
        for op in ops:
            if op.op == "create":
                goal = op.goal
                assert goal is not None  # validated above
                self._goals[goal.id] = goal
                self._status[goal.id] = "active"
                self._bump()
            elif op.op == "update":
                goal = self._goals[op.goal_id]  # type: ignore[index]
                if "say" in op.fields:
                    # Goal's top-level executable fields are frozen. Replace only the proven-safe
                    # presentation field; dataclasses.replace keeps the same mutable state object, so
                    # runtime-owned count/deadline cognition is preserved.
                    self._goals[op.goal_id] = replace(goal, say=op.fields["say"])  # type: ignore[index]
                self._bump()
            elif op.op == "complete":
                self._set_status(op.goal_id, "complete")
            elif op.op == "cancel":
                self._set_status(op.goal_id, "cancelled")

    def create(self, goal: Goal) -> Goal:
        self.apply([GoalOp(op="create", goal=goal)])
        return self._goals[goal.id]

    def _require(self, goal_id: str | None) -> None:
        if goal_id is None or goal_id not in self._goals:
            raise KeyError(f"no goal {goal_id!r}")

    def _set_status(self, goal_id: str | None, status: GoalStatus) -> None:
        self._require(goal_id)
        self._status[goal_id] = status  # type: ignore[index]
        self._bump()

    def status(self, goal_id: str) -> GoalStatus | None:
        return self._status.get(goal_id)

    def get(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def active(self) -> list[Goal]:
        return [g for gid, g in self._goals.items() if self._status.get(gid) == "active"]

    def all(self) -> list[Goal]:
        return list(self._goals.values())

    # -- persistence (durable goals) -----------------
    def snapshot(self) -> list[dict]:
        out: list[dict] = []
        for gid, g in self._goals.items():
            gd = asdict(g)
            # Provenance is durable: persist the ENUM's wire value (a string) so a JSON round-trip keeps
            # it. ``asdict`` leaves the GoalOrigin instance in place; render it to ``.value`` here.
            origin = gd.get("origin")
            if isinstance(origin, GoalOrigin):
                gd["origin"] = origin.value
            out.append({"goal": gd, "status": self._status.get(gid, "active")})
        return out

    def restore(self, items: list[dict]) -> None:
        restored_goals: dict[str, Goal] = {}
        restored_status: dict[str, GoalStatus] = {}
        for item in items:
            gd = dict(item["goal"])
            # Keep a PERSISTED origin (that is the point of durability — trust survives restart); a goal
            # from a pre-provenance snapshot carries none, so it is neutral RESTORED (never trusted-by-
            # default). A caller cannot smuggle trust in either way — the value is validated to the enum.
            raw = gd.pop("origin", None)
            if raw is None:
                origin = GoalOrigin.RESTORED
            elif isinstance(raw, GoalOrigin):
                origin = raw
            else:
                try:
                    origin = GoalOrigin(raw)
                except ValueError:
                    origin = GoalOrigin.RESTORED
            goal = Goal(**gd, origin=origin)
            if goal.id in restored_goals:
                raise ValueError(f"duplicate goal id {goal.id!r} in snapshot")
            status = item.get("status", "active")
            if status not in ("active", "complete", "cancelled"):
                raise ValueError(f"invalid goal status {status!r} for {goal.id!r}")
            restored_goals[goal.id] = goal
            restored_status[goal.id] = status

        # Parse and validate the entire snapshot before touching either registry or owner state. Then
        # notify the singular no-fail owner hook that every prior id is leaving; Percept uses this to
        # release compiled runtime/judge bindings and apply receipts even when callers invoke
        # ``core.goals.restore`` directly.
        if self._owner_mutation_hook is not None and self._goals:
            self._owner_mutation_hook(
                tuple(GoalOp(op="cancel", goal_id=goal_id) for goal_id in self._goals)
            )
        self._goals = restored_goals
        self._status = restored_status
        # A restore IS a goal-set mutation — fire subscribers so a grounding snapshot bound via
        # on_goal_change is invalidated on rehydrate, exactly as it is on arm/disarm.
        self._bump()


__all__ = ["GoalStatus", "GoalOp", "GoalRegistry"]
