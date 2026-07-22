"""Per-key evidence sequencing + a serialized decision reducer (ADR-001 §8).

Single-flight judging already gives sequencing safety "for free" today: ``Percept.perceive_judged``
awaits one judge call before starting the next, and ``expected_goals_revision`` aborts a stale
in-flight judge before it can mutate anything (see ``percept/agent.py``). The moment a WORKER POOL
dispatches multiple judge jobs for the *same key* concurrently, that free safety disappears — an
OLDER job admitted first can finish AFTER a NEWER job admitted second, and without an explicit
sequence the older result would be applied last, silently clobbering the newer one's gate state.

``evidence_seq < last_applied_sequence`` is NOT enough (ADR-001 §8): picture seq 3 in flight when
seq 5 is admitted and applied first (a faster worker, a retried job, whatever). A naive
``seq < last_applied`` check compares only against what has been APPLIED so far — job 3 still looks
"new" relative to a ``last_applied`` that hasn't caught up to 5 yet, and would be applied anyway. The
correct gate is ``sequence < latest_admitted_sequence``: a result is stale the instant a NEWER job has
been *dispatched* for that key, whether or not that newer job has itself completed.

This module supplies the two missing primitives:

- :class:`EvidenceSequencer` — a monotonic per-key counter. Call ``.admit(key)`` at job DISPATCH time
  (not completion) to reserve the job's ``evidence_seq`` and bump ``latest_admitted_sequence[key]``.
- :class:`SerializedDecisionReducer` — wraps a :class:`percept.gate.Gate` and applies each judge
  result through it ONE AT A TIME per key (an ``asyncio.Lock`` per key), gated on
  ``evidence_seq >= latest_admitted_sequence`` at commit time. A result that lost the race is DROPPED
  — reported as a :class:`GateDecision` with ``dropped=True`` and a ``drop_reason`` (a TRACED stale
  completion), and the underlying ``Gate`` is never touched, instead of being silently applied out of
  order.

:class:`ResultPolicy` controls how "stale" is judged:

- ``ordering="latest_only"`` (default) — global last-writer-wins per key: any result whose
  ``evidence_seq`` is behind the key's ``latest_admitted_sequence`` is dropped as superseded,
  regardless of episode.
- ``ordering="episode_ordered"`` — ordering is scoped to the result's ``episode_id`` rather than the
  key's all-time latest admission: a straggler from episode N is compared only against episode N's
  admissions and is dropped only when a NEWER job *within the same episode* has since been admitted —
  a fresh episode's low sequence is never penalized just because a prior episode ran further ahead.
- ``stale_after_ms`` — an independent, orthogonal drop: a result older than this (measured from its
  own admission time) is dropped as a timeout even with no superseding job. ``None`` disables it.
- ``apply_after_supersession`` — the escape hatch: when True, a result that WOULD have been dropped is
  applied anyway (flagged ``applied_after_supersession=True`` on the decision) instead of discarded.
  Off by default — a dropped result never silently mutates gate state.

Today's caller (``Percept.perceive_judged``, strictly single-flight) does not need any of this and is
left untouched — every existing gate test stays byte-for-byte. This module is the mechanism a future
worker pool wires in (per the card: "you don't need to enable a real worker pool").
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Hashable, Literal

from .adapters import Verdict
from .gate import Gate, GateAction, SILENT
from .observability import get_logger, log

ResultOrdering = Literal["latest_only", "episode_ordered"]


@dataclass(frozen=True)
class ResultPolicy:
    """How the reducer decides a completed judge result arrived too late to apply.

    ``ordering``:
      * ``"latest_only"`` (default) — drop a result the instant ANY newer job has been admitted for
        its key, regardless of episode.
      * ``"episode_ordered"`` — drop only when a newer job WITHIN THE SAME EPISODE has been admitted;
        a fresh episode is never penalized by a straggler from a prior one.
    ``stale_after_ms`` — independent timeout drop: a result whose age (now - admitted_at, in ms)
      exceeds this is dropped as stale even with no superseding job. ``None`` disables the check.
    ``apply_after_supersession`` — when True, a would-be-dropped result is still applied to the gate
      (marked ``applied_after_supersession=True``) instead of discarded. Default False: a dropped
      result NEVER mutates gate state.
    """

    ordering: ResultOrdering = "latest_only"
    stale_after_ms: float | None = None
    apply_after_supersession: bool = False


@dataclass(frozen=True)
class GateDecision:
    """The outcome of one :meth:`SerializedDecisionReducer.submit` call.

    ``action`` is the underlying three-state :data:`~percept.gate.GateAction` when the result was
    applied, or ``SILENT`` as a neutral fill when it was dropped — never FIRE/ASK for a dropped
    result, so a caller that only checks ``action == FIRE`` can never be fooled by a drop into acting
    on stale evidence. ``dropped``/``drop_reason`` are the TRACED stale-completion signal the card
    requires: every drop names why.
    """

    key: Hashable
    evidence_seq: int
    action: GateAction
    dropped: bool = False
    drop_reason: str | None = None
    applied_after_supersession: bool = False


class EvidenceSequencer:
    """Per-key monotonic ``evidence_seq`` — one counter per gate key, starting at 1."""

    def __init__(self) -> None:
        self._next: dict[Hashable, int] = {}
        self._latest_admitted: dict[Hashable, int] = {}
        self._episode_latest_admitted: dict[tuple[Hashable, Hashable], int] = {}

    def admit(self, key: Hashable, *, episode_id: Hashable | None = None) -> int:
        """Reserve the next sequence for ``key`` at JOB-DISPATCH time (not completion).

        Bumps ``latest_admitted_sequence[key]`` immediately — a job in flight already counts as "the
        latest admission" the instant it's dispatched. That is what makes the
        ``sequence < latest_admitted_sequence`` gate correct (ADR-001 §8): an older job can still be
        in flight when a newer one is admitted, and the newer admission alone — whether or not it has
        completed — is enough to mark the older one stale on arrival.
        """
        seq = self._next.get(key, 0) + 1
        self._next[key] = seq
        self._latest_admitted[key] = seq
        if episode_id is not None:
            self._episode_latest_admitted[(key, episode_id)] = seq
        return seq

    def latest_admitted(self, key: Hashable) -> int:
        """The highest ``evidence_seq`` ever admitted for ``key`` (0 if none yet)."""
        return self._latest_admitted.get(key, 0)

    def latest_admitted_in_episode(self, key: Hashable, episode_id: Hashable) -> int:
        """The highest ``evidence_seq`` admitted for ``key`` WITHIN ``episode_id`` (0 if none yet)."""
        return self._episode_latest_admitted.get((key, episode_id), 0)

    def tracked_keys(self) -> tuple[Hashable, ...]:
        """Every key this sequencer still holds a counter for, in first-admission order.

        The bound seam: a long-running caller (``percept.agent``'s judgment stamping) reads this
        to shed its oldest keys, the way :class:`percept.gate.Gate` bounds ``_armed``. Read-only.
        """
        return tuple(self._next)

    def forget(self, key: Hashable) -> None:
        """Drop all sequencing state for ``key`` (mirrors ``Gate.reset_goal``/``reset_entity``)."""
        self._next.pop(key, None)
        self._latest_admitted.pop(key, None)
        for ek in [ek for ek in self._episode_latest_admitted if ek[0] == key]:
            self._episode_latest_admitted.pop(ek, None)


class SerializedDecisionReducer:
    """Serializes judge-result application per key and gates on ``latest_admitted_sequence``.

    Wraps a :class:`~percept.gate.Gate`; :meth:`submit` is the only path that should call
    ``Gate.evaluate`` for a sequenced result. Concurrent ``submit`` calls for the SAME key are
    serialized through a per-key ``asyncio.Lock`` so a worker pool's
    overlapping completions can never interleave their gate mutations — the second submit to reach
    the lock always sees the first one's up-to-date ``latest_admitted``/applied state.
    """

    def __init__(
        self,
        gate: Gate,
        sequencer: EvidenceSequencer | None = None,
        *,
        policy: ResultPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gate = gate
        self.sequencer = sequencer or EvidenceSequencer()
        self.default_policy = policy or ResultPolicy()
        self._clock = clock
        self._locks: dict[Hashable, asyncio.Lock] = {}
        self._admitted_at: dict[tuple[Hashable, int], float] = {}
        self._last_applied: dict[Hashable, int] = {}

    def _lock_for(self, key: Hashable) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def admit(self, key: Hashable, *, episode_id: Hashable | None = None) -> int:
        """Dispatch-time admission: reserve ``evidence_seq`` and stamp its admitted-at clock."""
        seq = self.sequencer.admit(key, episode_id=episode_id)
        self._admitted_at[(key, seq)] = self._clock()
        return seq

    def last_applied(self, key: Hashable) -> int:
        """The highest ``evidence_seq`` actually applied through the gate for ``key`` (0 if none)."""
        return self._last_applied.get(key, 0)

    async def submit(
        self,
        key: Hashable,
        evidence_seq: int,
        verdict: Verdict,
        *,
        episode_id: Hashable | None = None,
        freshness: float = 1.0,
        hold_s: float | None = None,
        now: float | None = None,
        no_evict: bool = False,
        policy: ResultPolicy | None = None,
    ) -> GateDecision:
        """Apply one judge RESULT through the gate — serialized per key, gated on admitted sequence.

        Requirement 3: gates on ``latest_admitted_sequence`` (not ``last_applied``) — a result whose
        sequence lost the admission race is dropped as TRACED stale (``dropped=True`` +
        ``drop_reason``) and the gate is never mutated, unless ``policy.apply_after_supersession``
        opts back in.
        """
        pol = policy or self.default_policy
        lock = self._lock_for(key)
        async with lock:
            reason = self._stale_reason(key, evidence_seq, episode_id, pol)
            if reason is not None and not pol.apply_after_supersession:
                log(
                    get_logger("percept.sequencing"), "reducer.dropped",
                    key=repr(key), evidence_seq=evidence_seq, reason=reason,
                )
                return GateDecision(
                    key=key, evidence_seq=evidence_seq, action=SILENT,
                    dropped=True, drop_reason=reason,
                )
            action = self.gate.evaluate(
                key, verdict, freshness=freshness, hold_s=hold_s, now=now, no_evict=no_evict,
            )
            applied_after = reason is not None and pol.apply_after_supersession
            if applied_after:
                log(
                    get_logger("percept.sequencing"), "reducer.applied_after_supersession",
                    key=repr(key), evidence_seq=evidence_seq, reason=reason,
                )
            self._last_applied[key] = max(self._last_applied.get(key, 0), evidence_seq)
            return GateDecision(
                key=key, evidence_seq=evidence_seq, action=action,
                dropped=False, drop_reason=reason if applied_after else None,
                applied_after_supersession=applied_after,
            )

    def _stale_reason(
        self, key: Hashable, evidence_seq: int, episode_id: Hashable | None, pol: ResultPolicy,
    ) -> str | None:
        """The drop reason for this result under ``pol``, or ``None`` if it should apply."""
        if pol.stale_after_ms is not None:
            admitted_at = self._admitted_at.get((key, evidence_seq))
            if admitted_at is not None:
                age_ms = (self._clock() - admitted_at) * 1000.0
                if age_ms > pol.stale_after_ms:
                    return "stale_timeout"
        if pol.ordering == "episode_ordered" and episode_id is not None:
            latest = self.sequencer.latest_admitted_in_episode(key, episode_id)
            if evidence_seq < latest:
                return "superseded_within_episode"
            return None
        latest = self.sequencer.latest_admitted(key)
        if evidence_seq < latest:
            return "superseded_by_newer_admission"
        return None

    def forget(self, key: Hashable) -> None:
        """Drop all reducer state for ``key`` (sequencing + lock + applied bookkeeping)."""
        self.sequencer.forget(key)
        self._locks.pop(key, None)
        self._last_applied.pop(key, None)
        for k in [k for k in self._admitted_at if k[0] == key]:
            self._admitted_at.pop(k, None)


__all__ = [
    "ResultOrdering",
    "ResultPolicy",
    "GateDecision",
    "EvidenceSequencer",
    "SerializedDecisionReducer",
]
