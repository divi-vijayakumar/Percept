"""The three-state gate — the trust surface (invariant 1).

One firing path, one gate (invariant 3). A :class:`~percept.adapters.Verdict` plus a freshness
value maps to exactly one action:

| condition                                   | action   |
|---------------------------------------------|----------|
| confident yes, fresh, on a *rising* edge    | ``FIRE`` |
| confident yes, but already fired / not rising | ``SILENT`` |
| confident no                                | ``SILENT`` (re-arms the edge) |
| low-confidence **or** stale                 | ``ASK``  (never a confident-wrong nag) |

Rising-edge + refractory are implemented as a single armed/disarmed bit per key. A *key* is
``(goal_id, entity_or_episode_id)`` so a brand-new entity gets a fresh edge (a new cup fires even
though the old cup already did), while the same entity never double-fires until it goes
confident-false and re-arms.
"""

from __future__ import annotations

from typing import Callable, Hashable, Literal

from .adapters import Verdict
from .observability import get_logger, log

# M0/GATE_STATE_LEAK defaults — a bare ``Gate()`` must not be able to grow without bound (ADR-001
# §9, "Gate keys": count + TTL, LRU/TTL eviction except explicitly pinned). Chosen so neither bound
# interferes with any known caller's key volume, while still capping runaway 24/7 growth:
#   * _DEFAULT_MAX_KEYS  — the LRU backstop. It is enforced unconditionally (no clock required), so
#     it is the bound that actually protects a caller that never passes ``now``/a clock. 10,000 keys
#     is orders of magnitude above realistic (goal x entity) cardinality (EntityGraph itself defaults
#     to evicting at 64 entities, see percept/entity.py) yet trivial memory (a handful of small dict
#     entries per key) — generous enough to never truncate legitimate traffic, small enough to cap a
#     leak.
#   * _DEFAULT_KEY_TTL_S — 3600.0s (1 hour), matching the production default already chosen for
#     ``GateSettings.key_ttl_s`` in percept/config.py, so the raw primitive and the settings-driven
#     default agree. Only takes effect when ``evaluate(..., now=...)`` or a ``clock`` is supplied (the
#     main watch path, ``BehaviorExecutor.fire_judged``, always passes ``now``); a caller that
#     supplies neither relies on ``_DEFAULT_MAX_KEYS`` alone, per the docstring below.
# Either bound can still be disabled explicitly with ``max_keys=None`` / ``key_ttl_s=None`` (opt-out,
# not the default) for callers that deliberately want the old unbounded gate (e.g. reconstructing
# legacy behavior in a test).
_DEFAULT_MAX_KEYS = 10_000
_DEFAULT_KEY_TTL_S = 3600.0

GateAction = Literal["fire", "ask", "silent"]
FIRE: GateAction = "fire"
ASK: GateAction = "ask"
SILENT: GateAction = "silent"

# verdict -> coarse truth, before edge logic
_Truth = Literal["yes", "no", "unknown"]


def classify(
    verdict: Verdict, freshness: float, threshold: float, fresh_min: float
) -> _Truth:
    """Map a verdict + freshness to yes / no / unknown (the pre-edge decision).

    Low confidence OR a stale supporting fact is ``unknown`` -> ASK. This is the refusal-to-guess
    rule: ambiguity never becomes a confident fire.
    """
    if verdict.confidence < threshold:
        return "unknown"
    if freshness < fresh_min:
        return "unknown"
    return "yes" if verdict.satisfied else "no"


class Gate:
    """Stateful three-state gate. ``evaluate`` is the only firing path."""

    def __init__(
        self,
        threshold: float = 0.7,
        fresh_min: float = 0.5,
        *,
        max_keys: int | None = _DEFAULT_MAX_KEYS,
        key_ttl_s: float | None = _DEFAULT_KEY_TTL_S,
        clock: Callable[[], float] | None = None,
        on_evict: Callable[[Hashable, str], None] | None = None,
    ) -> None:
        self.threshold = threshold
        self.fresh_min = fresh_min
        self._armed: dict[Hashable, bool] = {}  # key -> ready-to-fire
        # R-1 sustained-duration (dwell): key -> the time it first became continuously true. Only
        # populated when a goal sets ``hold_s``; empty otherwise, so the no-dwell path is byte-for-byte.
        self._dwell_since: dict[Hashable, float] = {}
        # R-1/M0 growth bound (24/7). `_armed`/`_dwell_since` are fed by unbounded key sources (per-entity,
        # per-time-bucket, per-answer); on a 24/7 watch they grow without limit. Eviction is ON BY DEFAULT
        # (see `_DEFAULT_MAX_KEYS`/`_DEFAULT_KEY_TTL_S` above) — pass ``max_keys=None``/``key_ttl_s=None``
        # explicitly to opt back out to the old unbounded gate (no `_touched` bookkeeping at all).
        #   * key_ttl_s — a key untouched (unevaluated) this long is dropped (the primary bound; TTL sweep
        #     runs on the stream `now` the judge path already passes).
        #   * max_keys  — a hard LRU ceiling on the number of keys (the backstop when there is no clock).
        # `no_evict` keys (never-goals, the per-goal absence/resolved singletons) are EXEMPT from both — a
        # dropped-then-re-armed never-key silently converts a fire-once goal into an interval goal (the trap).
        self.max_keys = max_keys
        self.key_ttl_s = key_ttl_s
        self._clock = clock
        self._touched: dict[Hashable, float] = {}  # key -> last time it was evaluated (eviction stamp)
        self._no_evict: set[Hashable] = set()       # keys exempt from TTL + LRU (never-once semantics)
        # M0/GATE_STATE_LEAK: eviction must never be silent (ADR-001 §9). `on_evict(key, reason)` fires for
        # every bound-driven drop (reason is "ttl" or "max_keys"); default logs a structured line so a drop
        # is always observable even when the caller wires nothing.
        self.on_evict = on_evict

    @property
    def _evicting(self) -> bool:
        """Eviction bookkeeping is live whenever a bound is set — true by default (M0/GATE_STATE_LEAK);
        only an explicit ``max_keys=None, key_ttl_s=None`` disables it and restores the old unbounded,
        byte-for-byte gate."""
        return self.max_keys is not None or self.key_ttl_s is not None

    def _drop(self, key: Hashable) -> None:
        self._armed.pop(key, None)
        self._dwell_since.pop(key, None)
        self._touched.pop(key, None)
        self._no_evict.discard(key)

    def _evict(self, key: Hashable, reason: str) -> None:
        """Drop a key as a BOUND-DRIVEN eviction (TTL or max_keys) — as opposed to `_drop` used directly
        by `reset_goal`/`reset_entity`, which are intentional, caller-requested resets, not capacity
        pressure. M0/GATE_STATE_LEAK: this is the one path that must never be silent (ADR-001 §9)."""
        self._drop(key)
        if self.on_evict is not None:
            self.on_evict(key, reason)
        else:
            log(get_logger("percept.gate"), "gate.evict", key=repr(key), reason=reason)

    def _bound(self, current: Hashable, ref: float | None) -> None:
        """Sweep expired keys (TTL, needs a time ref) and enforce the LRU ceiling (max_keys). Never touches
        a `no_evict` key. Called after the CURRENT key is stamped but BEFORE the edge logic writes it into
        ``_armed``, so reserve a slot for it (``pending``) — otherwise the ceiling settles one key high."""
        if self.key_ttl_s is not None and ref is not None:
            cutoff = ref - self.key_ttl_s
            for key in [k for k, ts in self._touched.items()
                        if ts < cutoff and k not in self._no_evict and k != current]:
                self._evict(key, "ttl")
        if self.max_keys is not None:
            pending = 0 if current in self._armed else 1  # the current key will be armed right after this
            overflow = len(self._armed) + pending - self.max_keys
            if overflow > 0:
                evictable = [k for k in self._armed if k not in self._no_evict and k != current]
                # least-recently-touched first (unstamped keys sort oldest -> shed before live ones)
                evictable.sort(key=lambda k: self._touched.get(k, float("-inf")))
                for key in evictable[:overflow]:
                    self._evict(key, "max_keys")

    def evaluate(
        self,
        key: Hashable,
        verdict: Verdict,
        *,
        freshness: float = 1.0,
        hold_s: float | None = None,
        now: float | None = None,
        no_evict: bool = False,
        require_false_first: bool = False,
    ) -> GateAction:
        """Return FIRE / ASK / SILENT for ``key``, updating edge state.

        ``hold_s`` (R-1) is an optional sustained-duration/dwell: with it set, an armed key does not fire
        the instant it becomes true — it must stay confidently true for ``hold_s`` seconds (measured on
        ``now``, the stream timeline) first. A confident-false in between resets the dwell (a transient
        blip never fires). ``hold_s=None`` (the default) is byte-for-byte the rising-edge gate.

        ``no_evict`` marks this key exempt from TTL/LRU eviction — set by the executor for never-goals and
        the per-goal absence/resolved singletons, whose edge must survive an idle gap (dropping it would
        re-arm a fire-once goal). Ignored entirely when no eviction bound is configured.

        ``require_false_first`` (opt-in, default off) enforces the v0.1 "already present at start ->
        SILENT" rule: a key whose FIRST observation is confidently-true does not fire — only a genuine
        false->true edge does. Off = byte-for-byte the historical rising-edge gate (fire on first true).
        """
        # Eviction bookkeeping (only when a bound is set — bounded by default, see `_evicting`). Stamp the
        # key BEFORE sweeping so the key in play is the freshest and can never evict itself.
        if self._evicting:
            ref = now if now is not None else (self._clock() if self._clock is not None else None)
            if no_evict:
                self._no_evict.add(key)
            if ref is not None:
                self._touched[key] = ref
            self._bound(key, ref)
        truth = classify(verdict, freshness, self.threshold, self.fresh_min)
        if truth == "unknown":
            return ASK  # asking never changes the edge — the fact is still pending
        if truth == "no":
            self._armed[key] = True  # confident-false re-arms
            self._dwell_since.pop(key, None)  # ...and breaks any in-progress dwell (blip resets the clock)
            return SILENT
        # truth == "yes"
        if require_false_first and key not in self._armed:
            # Spec (v0.1 hero matrix: "already present when the runtime starts -> SILENT"): a condition
            # that is ALREADY true on the first observation of this key is not a rising edge we watched
            # become true. Record the key as seen-but-DISARMED (never fire on it); only a subsequent
            # confident-false (which arms it via the branch above) establishes the false->true baseline
            # a genuine rising-edge fire requires. Opt-in: without this flag a new key defaults to armed
            # (byte-for-byte the historical rising-edge gate) so existing callers are unchanged.
            self._armed[key] = False
            return SILENT
        if not self._armed.get(key, True):  # new keys default to armed -> fresh edge
            return SILENT  # already true / already fired this episode
        if hold_s is None or hold_s <= 0 or now is None:
            self._armed[key] = False
            return FIRE  # no dwell -> the rising-edge fire (byte-for-byte)
        # dwell: require the condition to have held continuously for hold_s before firing.
        t0 = self._dwell_since.get(key)
        if t0 is None:
            self._dwell_since[key] = now  # first sustained-true tick -> start the dwell timer
            return SILENT
        if now - t0 >= hold_s:
            self._armed[key] = False
            self._dwell_since.pop(key, None)
            return FIRE
        return SILENT  # still accumulating dwell — true, but not long enough yet

    def rearm(self, key: Hashable) -> None:
        """Force a key back to armed (e.g. after a correction clears the edge)."""
        self._armed[key] = True
        self._dwell_since.pop(key, None)

    def reset_goal(self, goal_id: str) -> None:
        """Drop all edge state for a goal so its next evaluation starts fresh."""
        for key in [k for k in self._armed if isinstance(k, tuple) and k and k[0] == goal_id]:
            self._drop(key)
        # `_drop` clears `_armed`/`_dwell_since`/`_touched`/`_no_evict` together; sweep any dwell/touched
        # keys that never had an `_armed` entry too (e.g. an in-progress dwell), keeping the maps aligned.
        for key in [k for k in self._dwell_since if isinstance(k, tuple) and k and k[0] == goal_id]:
            self._drop(key)

    def reset_entity(self, entity_id: Hashable) -> None:
        """R3: drop every gate key belonging to a dead entity (``(goal_id, entity_id)``), so an evicted
        entity leaves no orphan edge behind. Wired to :meth:`EntityGraph` eviction by :class:`Percept`.
        The per-goal singletons (``(goal_id, "__resolved__")`` etc.) never match — their second element is
        a sentinel string, not an entity id — so they are untouched."""
        for key in [
            k for k in self._armed
            if isinstance(k, tuple) and len(k) >= 2 and k[1] == entity_id
        ]:
            self._drop(key)

    # -- persistence (R4: the armed edge must survive a restart, or a 24/7 watch re-fires) ---------
    def snapshot(self) -> dict:
        """Durable edge state. Keys are tuples ``(goal_id, entity[, answer])`` -> serialized as lists so
        the whole snapshot stays JSON-able through the persistence backend. The in-progress dwell timer
        is included for inspection but is NOT restored (see :meth:`restore`)."""
        return {
            "armed": [[list(k), bool(v)] for k, v in self._armed.items()],
            "dwell_since": [[list(k), float(t)] for k, t in self._dwell_since.items()],
            # R-1: the last-touched eviction stamps + the no-evict exemption set travel with the edge, so a
            # restarted 24/7 watch keeps its TTL budget and its never-once exemptions (a fresh restore of a
            # legacy snapshot lacking these keys is a no-op — see :meth:`restore`).
            "touched": [[list(k), float(t)] for k, t in self._touched.items()],
            "no_evict": [list(k) for k in self._no_evict],
        }

    def restore(self, snap: "dict | None") -> None:
        """Rehydrate the armed/disarmed edge per key (the anti-re-fire bit). A missing/empty snapshot is a
        no-op (byte-for-byte a fresh gate).

        The in-progress dwell (``_dwell_since``) is EXPLICITLY DROPPED: its start is a wall-clock stamp on
        the pre-restart timeline with no anchor to rebase from, so a half-completed dwell cannot be
        resumed correctly. Dropping it re-arms that edge's dwell from now — the loud, correct failure — in
        preference to silently resurrecting a stamp that would fire early or late."""
        if not snap:
            return
        self._armed = {tuple(k): bool(v) for k, v in snap.get("armed", [])}
        self._dwell_since = {}
        self._touched = {tuple(k): float(t) for k, t in snap.get("touched", [])}
        self._no_evict = {tuple(k) for k in snap.get("no_evict", [])}


__all__ = ["GateAction", "FIRE", "ASK", "SILENT", "classify", "Gate"]
