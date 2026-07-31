"""The three-state gate — the trust surface (invariant 1).

One firing path, one gate (invariant 3). A :class:`~percept.adapters.Verdict` plus a freshness
value maps to exactly one action:

| condition                                   | action   |
|---------------------------------------------|----------|
| confident yes, fresh, on a *rising* edge    | ``FIRE`` |
| confident yes, but already fired / not rising | ``SILENT`` |
| confident no                                | ``SILENT`` (re-arms the edge) |
| low-confidence **or** stale                 | ``ASK``  (never a confident-wrong nag) |

Rising-edge + the armed/disarmed re-fire guard are implemented as a single bit per key. A *key* is
``(goal_id, entity_or_episode_id)`` so a brand-new entity gets a fresh edge (a new cup fires even
though the old cup already did), while the same entity never double-fires until it goes
confident-false and re-arms.

``refractory_s`` (precision track) is a SEPARATE, time-based guard on top of that bit: it
suppresses a fire landing within N seconds of the same key's last real fire, even when a spurious
confident-false blip re-armed the edge in between (mid-episode judge jitter). Off by default
(``None``/``0`` = byte-for-byte the historical gate); see :meth:`Gate.evaluate`.

GA of the decision model
-------------------------
The three actions and the rising-edge/refractory rules above are the SDK's documented, stable
decision surface. Two things make that surface usable by an integrator rather than merely true:

1. **A typed reason vocabulary.** :data:`GATE_REASONS` is the closed set of WHY codes the gate can
   produce, each mapping to exactly one action (:data:`REASON_ACTION`). :meth:`Gate.decide` returns
   the typed :class:`GateOutcome` (action + reason); :meth:`Gate.evaluate` is unchanged and still
   returns the bare :data:`GateAction`, so every existing caller is byte-for-byte.
2. **Suppression is observable.** :data:`SUPPRESSION_REASONS` names the codes where a fire the
   BASELINE rising-edge gate would have emitted was deliberately withheld by an opt-in precision
   guard (``require_false_first`` -> ``start_anchored``; ``refractory_s`` -> ``refractory``). Those
   — and only those — invoke the optional ``on_suppress`` hook, which
   :meth:`percept.agent.Percept.create` wires into the runtime's health channel so the suppression
   surfaces as a ``RuntimeHealthEvent("gate_suppressed")`` instead of a silent non-event (ADR-001
   §9: a bound-driven drop is never silent). ``already_fired`` / ``dwell_pending`` /
   ``not_satisfied`` are deliberately NOT suppressions — they are the base contract (every tick of a
   continuing true state produces ``already_fired``), and reporting them would be per-frame noise.

``SILENT`` remains INTERNAL-ONLY: it is a value of this module's decision model, never an action on
a delivered event. Nothing on the public delivery path (``watch()``, ``on_incident``) ever carries
it — the firing path drops a silent decision rather than surfacing it (see
:meth:`percept.behavior.BehaviorExecutor.fire_judged`, which returns ``None``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Hashable, Literal

from ._snapshot import (SnapshotError, as_bool, as_float, key_tuple, parse_keys, parse_pairs,
                        parse_records, reject)
from .adapters import Verdict
from .identity_policy import is_entity_scoped
from .observability import get_logger, log

# GATE_STATE_LEAK defaults — a bare ``Gate()`` must not be able to grow without bound (ADR-001
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

#: Every action the gate can produce, in delivery-precedence order. ``SILENT`` is INTERNAL-ONLY (see
#: the module docstring) — it is in the vocabulary because the decision model has three states, not
#: because three states are delivered.
GATE_ACTIONS: tuple[GateAction, ...] = (FIRE, ASK, SILENT)

#: Actions that reach a subscriber. The public delivery contract: a gate decision is delivered iff
#: its action is in this tuple.
DELIVERED_ACTIONS: tuple[GateAction, ...] = (FIRE, ASK)

GateReason = Literal[
    # -- FIRE
    "rising_edge",       # confident-true on an armed edge — the canonical proactive fire
    "dwell_satisfied",   # ...and it stayed continuously true for the goal's hold_s first (R-1 dwell)
    # -- ASK (refusal to guess; never changes the edge)
    "low_confidence",    # verdict.confidence < threshold
    "stale",             # the supporting perception/fact is older than fresh_min allows
    # -- SILENT (internal-only)
    "not_satisfied",     # confident-false — re-arms the edge
    "already_fired",     # confident-true but the edge is spent (this episode already fired)
    "start_anchored",    # SUPPRESSION: already true on the FIRST observation (require_false_first)
    "refractory",        # SUPPRESSION: a fire was due but lands inside this key's refractory window
    "dwell_pending",     # true, but hasn't held for hold_s yet — a deferral, not a suppression
]

#: The closed reason vocabulary. Stable, documented surface: a code is never renamed or repurposed;
#: a new decision path adds a NEW code (and a new entry in :data:`REASON_ACTION`).
GATE_REASONS: tuple[GateReason, ...] = (
    "rising_edge",
    "dwell_satisfied",
    "low_confidence",
    "stale",
    "not_satisfied",
    "already_fired",
    "start_anchored",
    "refractory",
    "dwell_pending",
)

#: Each reason maps to exactly ONE action — the reason is strictly finer-grained than the action, so
#: an integrator can key off either without them ever disagreeing.
REASON_ACTION: dict[GateReason, GateAction] = {
    "rising_edge": FIRE,
    "dwell_satisfied": FIRE,
    "low_confidence": ASK,
    "stale": ASK,
    "not_satisfied": SILENT,
    "already_fired": SILENT,
    "start_anchored": SILENT,
    "refractory": SILENT,
    "dwell_pending": SILENT,
}

#: The reasons that count as GATE SUPPRESSION: a fire the baseline rising-edge gate WOULD have
#: emitted, withheld by an opt-in precision guard. Both are off by default, so a default-configured
#: gate never suppresses and never emits a suppression health event. Deliberately excludes
#: ``already_fired`` (the base re-fire contract, one per tick of any continuing true state),
#: ``dwell_pending`` (a deferral — the same edge still fires once the dwell completes) and
#: ``not_satisfied`` (no fire was ever due).
SUPPRESSION_REASONS: tuple[GateReason, ...] = ("start_anchored", "refractory")

# verdict -> coarse truth, before edge logic
_Truth = Literal["yes", "no", "unknown"]


@dataclass(frozen=True)
class GateOutcome:
    """One typed gate decision: WHAT (:attr:`action`) and WHY (:attr:`reason`), plus the key it was
    made for and the stream time it was made at (``None`` when the caller passed no ``now``).

    Returned by :meth:`Gate.decide`. :meth:`Gate.evaluate` returns ``decision.action`` alone, so the
    typed form is purely additive — no existing caller changes.

    Named ``GateOutcome``, not ``GateDecision``, on purpose: two OTHER, unrelated types in this
    package already own that name — :class:`percept.sequencing.GateDecision` (a reducer submission
    result: this action plus sequencing/drop metadata) and
    :class:`percept.harness.tier0.GateDecision` (the tier-0 EDGE decision, escalate-or-not, which
    happens strictly before any of this). This is the consequence gate's own outcome, and it is the
    one of the three exported from the top-level ``percept`` namespace.
    """

    action: GateAction
    reason: GateReason
    key: Hashable = None
    at: float | None = None

    def __post_init__(self) -> None:
        expected = REASON_ACTION.get(self.reason)
        if expected is None:
            raise ValueError(f"unknown gate reason {self.reason!r}; expected one of {GATE_REASONS}")
        if expected != self.action:
            raise ValueError(
                f"gate reason {self.reason!r} implies action {expected!r}, got {self.action!r}"
            )

    @property
    def suppressed(self) -> bool:
        """True iff a fire the baseline rising-edge gate would have emitted was withheld here."""
        return self.reason in SUPPRESSION_REASONS

    @property
    def delivered(self) -> bool:
        """True iff this decision reaches a subscriber (FIRE/ASK). SILENT is internal-only."""
        return self.action in DELIVERED_ACTIONS


def _confidence_is_usable(verdict: Verdict) -> bool:
    """Is ``verdict.confidence`` a number the ``< threshold`` comparison can actually refuse?

    Defence in depth for ADR-001 §10 invariant 4, and EXACTLY ONE TERM of the verdict contract —
    the single one whose failure this function's own arithmetic cannot detect. Every comparison
    against NaN is ``False``, so a ``nan`` confidence walks straight through ``confidence <
    threshold`` and out the other side as a confident ``"yes"``: the refusal-to-guess rule silently
    inverts into a confident wrong fire. ``inf`` and a ``bool`` (``isinstance(True, int)`` is
    ``True``, and ``True < 0.7`` is ``False``) do the same. A non-numeric confidence raises
    ``TypeError`` from the comparison instead — a crash, which invariant 4 also forbids.

    The real fix is the boundary: :func:`percept.contracts.verdict_defect` refuses all of this
    before a malformed verdict ever reaches the gate. This is the second line, for the paths that
    construct a ``Verdict`` internally and reach ``classify`` without crossing a judge seam.

    **The range check is deliberately NOT here.** A finite ``7.5`` stays classifiable. Refusing an
    out-of-range confidence inside ``classify`` would silently change behaviour for a deployment
    whose backend reports on a 0-100 scale — a scale error belongs at the boundary, where it is
    reported as ``confidence_out_of_range`` and fails the call over, not swallowed into an ``ASK``
    the operator never sees. Restricted to non-finiteness, this guard is provably behaviour-
    preserving on EVERY finite numeric input, which is what makes it safe to put on the hot path.
    """
    confidence = verdict.confidence
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    # Only a float can be NaN/inf; an int is finite by construction, and ``math.isfinite`` on a
    # very large one raises OverflowError converting it — so never ask.
    return math.isfinite(confidence) if isinstance(confidence, float) else True


def classify(
    verdict: Verdict, freshness: float, threshold: float, fresh_min: float
) -> _Truth:
    """Map a verdict + freshness to yes / no / unknown (the pre-edge decision).

    Low confidence OR a stale supporting fact is ``unknown`` -> ASK. This is the refusal-to-guess
    rule: ambiguity never becomes a confident fire — and an UNUSABLE confidence (NaN/inf/bool/
    non-numeric, see :func:`_confidence_is_usable`) is the most ambiguous input of all, so it takes
    the same refusal rather than defeating the comparison that implements it.
    """
    if not _confidence_is_usable(verdict):
        return "unknown"
    if verdict.confidence < threshold:
        return "unknown"
    if freshness < fresh_min:
        return "unknown"
    return "yes" if verdict.satisfied else "no"


@dataclass
class _KeyState:
    """Everything the gate remembers about ONE key, in ONE record.

    This used to be SIX parallel dicts, and every iterator over them had drifted apart: ``_drop``
    covered all six, ``reset_goal`` swept three, ``reset_entity`` swept exactly one, and the
    ``max_keys`` ceiling counted a seventh population (``len(_armed)``) that excluded any key the
    dwell branches had touched. The bugs that produced were not independent — they were one bug,
    "the keyspace is defined by whichever map you happened to iterate", wearing three faces:
    orphaned dwell state for a dead entity, an LRU ceiling that under-bounded by exactly the
    dwell-pending population, and a per-key eviction exemption that a sibling reset threw away.

    With one record per key the question "which keys exist?" has exactly one answer (``_keys``), so
    ``_drop`` is total by construction, the ceiling's denominator is right by construction, and no
    sweep can leave a map behind because there are no other maps.

    ``armed`` is ``None``, not ``False``, when the bit was never written. That distinction is
    load-bearing twice over: an absent bit means "fresh edge, default ARMED"
    (``self._armed.get(key, True)`` historically), and ``require_false_first`` keys off "was this
    key ever armed?" (``key not in self._armed``). A dwell-pending key legitimately has state
    without ever having had an armed bit, which is why it cannot collapse to a plain ``bool``.
    """

    armed: bool | None = None            # `_armed`: None = never written = a fresh edge -> ARMED
    dwell_since: float | None = None     # `_dwell_since`: when it first became continuously true
    dwell_last_supported: float | None = None  # `_dwell_last_supported`: its last supported read
    touched: float | None = None         # `_touched`: eviction stamp, when it was last evaluated
    no_evict: bool = False               # `_no_evict`: exempt from TTL + LRU (never-once semantics)
    last_fire: float | None = None       # `_last_fire`: this key's last REAL fire (precision track)

    def is_empty(self) -> bool:
        """True iff this record carries nothing — the gate then forgets the key entirely, so a
        record can never linger as an invisible occupant of the ``max_keys`` ceiling."""
        return (
            self.armed is None
            and self.dwell_since is None
            and self.dwell_last_supported is None
            and self.touched is None
            and not self.no_evict
            and self.last_fire is None
        )


#: Snapshot payload version. v2 writes self-describing per-key RECORDS (each field named on the
#: entry that owns it) alongside the v1 pair-lists, so an older build can still read a newer
#: snapshot. :meth:`Gate.restore` prefers the records and falls back to the pair-lists.
SNAPSHOT_VERSION = 2


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
        require_false_first: bool = False,
        refractory_s: float | None = None,
        on_suppress: "Callable[[GateOutcome], None] | None" = None,
    ) -> None:
        # Called with the :class:`GateOutcome` for every SUPPRESSED decision (reason in
        # :data:`SUPPRESSION_REASONS`) — never for a plain already_fired/dwell_pending/not_satisfied
        # SILENT. ``None`` (the default) = no hook = byte-for-byte the earlier gate; ``Percept.create``
        # wires it to the runtime health channel. Best-effort by contract: a raising hook must never
        # break the firing path (see :meth:`_note_suppressed`).
        self.on_suppress = on_suppress
        self.threshold = threshold
        self.fresh_min = fresh_min
        self.require_false_first = require_false_first  # gate-wide default for the baseline rule
        # Precision track: gate-wide default time-refractory (WatchProfile ``gate.refractory_s``).
        # ``None``/``0``/negative = off = byte-for-byte the historical gate (no per-call override needed).
        self.refractory_s = refractory_s
        # THE keyspace: one :class:`_KeyState` per key, and nothing else. Its fields are the six
        # per-key facts the gate holds — the armed bit (`_armed`), the R-1 sustained-duration timer
        # and its support-continuity stamp (`_dwell_since` / `_dwell_last_supported`), the eviction
        # stamp (`_touched`), the never-once exemption (`_no_evict`), and the precision track's
        # last-real-fire stamp (`_last_fire`). Read them individually through the same-named
        # read-only views below.
        self._keys: dict[Hashable, _KeyState] = {}
        # R-1 growth bound (24/7). The keyspace is fed by unbounded key sources (per-entity,
        # per-time-bucket, per-answer); on a 24/7 watch it grows without limit. Eviction is ON BY DEFAULT
        # (see `_DEFAULT_MAX_KEYS`/`_DEFAULT_KEY_TTL_S` above) — pass ``max_keys=None``/``key_ttl_s=None``
        # explicitly to opt back out to the old unbounded gate (no eviction bookkeeping at all).
        #   * key_ttl_s — a key untouched (unevaluated) this long is dropped (the primary bound; TTL sweep
        #     runs on the stream `now` the judge path already passes).
        #   * max_keys  — a hard LRU ceiling on the number of keys (the backstop when there is no clock).
        # `no_evict` keys (never-goals, the per-goal absence/resolved singletons) are EXEMPT from both — a
        # dropped-then-re-armed never-key silently converts a fire-once goal into an interval goal (the trap).
        self.max_keys = max_keys
        self.key_ttl_s = key_ttl_s
        self._clock = clock
        # GATE_STATE_LEAK: eviction must never be silent (ADR-001 §9). `on_evict(key, reason)` fires for
        # every bound-driven drop (reason is "ttl" or "max_keys"); default logs a structured line so a drop
        # is always observable even when the caller wires nothing.
        self.on_evict = on_evict

    # -- the per-field views ------------------------------------------------------------------
    # Read-only projections of `_keys`, one per `_KeyState` field, under the names the six parallel
    # dicts used to carry. They exist so a reader (a test, a debugger, an operator at a REPL) can
    # still ask "which keys are armed?" without knowing the record layout — and they are DERIVED, so
    # they can never drift out of agreement with each other the way the six real dicts did. A field
    # whose value is `None` means "never written", so that key is simply absent from its view,
    # exactly as it was absent from the old dict.

    @property
    def _armed(self) -> dict[Hashable, bool]:
        return {k: st.armed for k, st in self._keys.items() if st.armed is not None}

    @property
    def _dwell_since(self) -> dict[Hashable, float]:
        return {k: st.dwell_since for k, st in self._keys.items() if st.dwell_since is not None}

    @property
    def _dwell_last_supported(self) -> dict[Hashable, float]:
        return {
            k: st.dwell_last_supported
            for k, st in self._keys.items()
            if st.dwell_last_supported is not None
        }

    @property
    def _touched(self) -> dict[Hashable, float]:
        return {k: st.touched for k, st in self._keys.items() if st.touched is not None}

    @property
    def _no_evict(self) -> set[Hashable]:
        return {k for k, st in self._keys.items() if st.no_evict}

    @property
    def _last_fire(self) -> dict[Hashable, float]:
        return {k: st.last_fire for k, st in self._keys.items() if st.last_fire is not None}

    @property
    def _evicting(self) -> bool:
        """Eviction bookkeeping is live whenever a bound is set — true by default (GATE_STATE_LEAK guard);
        only an explicit ``max_keys=None, key_ttl_s=None`` disables it and restores the old unbounded,
        byte-for-byte gate."""
        return self.max_keys is not None or self.key_ttl_s is not None

    def _state(self, key: Hashable) -> _KeyState:
        """This key's record, minted empty on first write. The only place a key enters the keyspace."""
        state = self._keys.get(key)
        if state is None:
            state = _KeyState()
            self._keys[key] = state
        return state

    def _forget_if_empty(self, key: Hashable) -> None:
        """Drop a record that has been emptied out, so clearing a key's last fact really removes the
        key rather than parking a blank record inside the ``max_keys`` ceiling."""
        state = self._keys.get(key)
        if state is not None and state.is_empty():
            del self._keys[key]

    def _drop(self, key: Hashable) -> None:
        """Forget a key COMPLETELY. Total by construction now: there is one record, and it is gone."""
        self._keys.pop(key, None)

    def _evict(self, key: Hashable, reason: str) -> None:
        """Drop a key as a BOUND-DRIVEN eviction (TTL or max_keys) — as opposed to `_drop` used directly
        by `reset_goal`/`reset_entity`, which are intentional, caller-requested resets, not capacity
        pressure. GATE_STATE_LEAK: this is the one path that must never be silent (ADR-001 §9)."""
        self._drop(key)
        if self.on_evict is not None:
            self.on_evict(key, reason)
        else:
            log(get_logger("percept.gate"), "gate.evict", key=repr(key), reason=reason)

    def _bound(self, current: Hashable, ref: float | None) -> None:
        """Sweep expired keys (TTL, needs a time ref) and enforce the LRU ceiling (max_keys). Never touches
        a `no_evict` key.

        The ceiling counts ``_keys`` — the WHOLE keyspace. It used to count ``len(self._armed)``, one of
        the six parallel maps, while ``_drop`` freed all six: a dwell-pending key therefore consumed
        memory that the bound could not see, so ``max_keys`` under-bounded by exactly the population of
        keys in flight on a duration goal. On a duration-heavy 24/7 watch that is unbounded growth wearing
        a bound's clothes.

        ``pending`` reserves a slot for a current key that has not entered the keyspace yet — which is
        now only possible OFF-CLOCK, since a stamped key was already minted by ``_touched``. Without the
        reservation the ceiling settles one key high forever.
        """
        if self.key_ttl_s is not None and ref is not None:
            cutoff = ref - self.key_ttl_s
            for key in [k for k, st in self._keys.items()
                        if st.touched is not None and st.touched < cutoff
                        and not st.no_evict and k != current]:
                self._evict(key, "ttl")
        if self.max_keys is not None:
            pending = 0 if current in self._keys else 1  # off-clock: it enters the keyspace right after
            overflow = len(self._keys) + pending - self.max_keys
            if overflow > 0:
                evictable = [k for k, st in self._keys.items() if not st.no_evict and k != current]
                # least-recently-touched first (unstamped keys sort oldest -> shed before live ones)
                evictable.sort(
                    key=lambda k: (
                        self._keys[k].touched
                        if self._keys[k].touched is not None
                        else float("-inf")
                    )
                )
                for key in evictable[:overflow]:
                    self._evict(key, "max_keys")

    def _note_suppressed(self, decision: "GateOutcome") -> None:
        """Announce a SUPPRESSED decision through ``on_suppress`` (and always log it).

        Best-effort by contract — telemetry must never break the trust surface, so a hook that raises
        is swallowed and logged. The structured log line is emitted regardless of whether a hook is
        wired, so suppression is observable even in a bare ``Gate()`` (ADR-001 §9: never silent)."""
        log(
            get_logger("percept.gate"), "gate.suppressed",
            key=repr(decision.key), reason=decision.reason, at=decision.at,
        )
        if self.on_suppress is None:
            return
        try:
            self.on_suppress(decision)
        except Exception as exc:  # noqa: BLE001 - a telemetry hook never breaks the firing path
            log(get_logger("percept.gate"), "gate.suppress_hook_error", error=repr(exc))

    def evaluate(
        self,
        key: Hashable,
        verdict: Verdict,
        *,
        freshness: float = 1.0,
        hold_s: float | None = None,
        max_support_gap_s: float | None = None,
        now: float | None = None,
        no_evict: bool = False,
        require_false_first: bool = False,
        refractory_s: float | None = None,
    ) -> GateAction:
        """Return FIRE / ASK / SILENT for ``key`` — :attr:`GateOutcome.action` of :meth:`decide`.

        The historical entry point, unchanged: same signature, same return type, same edge effects.
        Callers that want the typed WHY as well call :meth:`decide` instead."""
        return self.decide(
            key, verdict, freshness=freshness, hold_s=hold_s,
            max_support_gap_s=max_support_gap_s, now=now, no_evict=no_evict,
            require_false_first=require_false_first, refractory_s=refractory_s,
        ).action

    def decide(
        self,
        key: Hashable,
        verdict: Verdict,
        *,
        freshness: float = 1.0,
        hold_s: float | None = None,
        max_support_gap_s: float | None = None,
        now: float | None = None,
        no_evict: bool = False,
        require_false_first: bool = False,
        refractory_s: float | None = None,
    ) -> "GateOutcome":
        """Return the typed :class:`GateOutcome` (FIRE / ASK / SILENT **plus** the reason code) for
        ``key``, updating edge state.

        ``hold_s`` (R-1) is an optional sustained-duration/dwell: with it set, an armed key does not fire
        the instant it becomes true — it must stay confidently true for ``hold_s`` seconds (measured on
        ``now``, the stream timeline) first. A confident-false, low-confidence read, stale read, or ASK
        in between resets pending dwell support (a transient or unsupported interval never fires).
        ``hold_s=None`` (the default) is byte-for-byte the rising-edge gate.

        ``no_evict`` marks this key exempt from TTL/LRU eviction — set by the executor for never-goals and
        the per-goal absence/resolved singletons, whose edge must survive an idle gap (dropping it would
        re-arm a fire-once goal). Ignored entirely when no eviction bound is configured.

        ``require_false_first`` (opt-in, default off) enforces the v0.1 "already present at start ->
        SILENT" rule: a key whose FIRST observation is confidently-true does not fire — only a genuine
        false->true edge does. Off = byte-for-byte the historical rising-edge gate (fire on first true).

        ``refractory_s`` (precision track, opt-in) suppresses a fire that would otherwise land
        within ``refractory_s`` seconds of THIS key's last real fire — even if a confident-false blip
        re-armed the edge in between. Precision forensic finding: ~12% of false fires on the
        charades-e2e track were RE-FIRES — the gate re-arming and firing again within one true episode
        because the judge briefly flickered to "no" mid-episode. A time-refractory is the fix: a
        genuine SECOND episode (a real absence, then a real return) still fires once refractory_s has
        elapsed since the last fire; only fires closer together than that are suppressed. Falls back to
        the gate-wide ``self.refractory_s`` (a WatchProfile ``gate.refractory_s`` knob) when not passed
        per-call. Requires ``now``; ``None``/``0``/negative (the default either way) = byte-for-byte the
        historical gate (no refractory).

        The reason codes are the closed :data:`GATE_REASONS` vocabulary; the two SUPPRESSION
        codes (``start_anchored``, ``refractory``) additionally invoke ``on_suppress``. Neither the
        actions nor the edge effects differ from :meth:`evaluate` — this method IS the implementation
        and ``evaluate`` is its ``.action``.
        """
        def _d(action: GateAction, reason: GateReason) -> GateOutcome:
            decision = GateOutcome(action=action, reason=reason, key=key, at=now)
            if decision.suppressed:
                self._note_suppressed(decision)
            return decision

        # Eviction bookkeeping (only when a bound is set — bounded by default, see `_evicting`). Stamp the
        # key BEFORE sweeping so the key in play is the freshest and can never evict itself.
        if self._evicting:
            ref = now if now is not None else (self._clock() if self._clock is not None else None)
            if no_evict:
                self._state(key).no_evict = True
            if ref is not None:
                self._state(key).touched = ref
            self._bound(key, ref)
        truth = classify(verdict, freshness, self.threshold, self.fresh_min)
        if truth == "unknown":
            if hold_s is not None and hold_s > 0:
                self.clear_pending_dwell(key)
            # ASK preserves the armed/fired rising-edge bit. For positive dwell it invalidates only
            # the unproven continuity timer above. The two ASK codes split the single "unknown" truth
            # into WHICH arm refused: low confidence or stale support.
            #
            # This dereferences ``.confidence`` a SECOND time, so it needs the same one-term guard
            # ``classify`` has (see :func:`_confidence_is_usable`) — without it a NaN confidence
            # would answer ``False`` here too and get reported as ``stale``, blaming the freshness
            # arm for a refusal the confidence arm made. Unusable confidence IS the confidence arm
            # refusing, so it reads ``low_confidence``.
            reason: GateReason = (
                "low_confidence"
                if not _confidence_is_usable(verdict) or verdict.confidence < self.threshold
                else "stale"
            )
            return _d(ASK, reason)
        if truth == "no":
            self._state(key).armed = True  # confident-false re-arms
            self.clear_pending_dwell(key)  # ...and breaks any in-progress dwell (blip resets the clock)
            return _d(SILENT, "not_satisfied")
        # truth == "yes"
        state = self._keys.get(key)
        if (require_false_first or self.require_false_first) and (
            state is None or state.armed is None
        ):
            # Spec (v0.1 hero matrix: "already present when the runtime starts -> SILENT"): a condition
            # that is ALREADY true on the first observation of this key is not a rising edge we watched
            # become true. Record the key as seen-but-DISARMED (never fire on it); only a subsequent
            # confident-false (which arms it via the branch above) establishes the false->true baseline
            # a genuine rising-edge fire requires. Opt-in: without this flag a new key defaults to armed
            # (byte-for-byte the historical rising-edge gate) so existing callers are unchanged.
            self._state(key).armed = False
            if hold_s is not None and hold_s > 0:
                self.clear_pending_dwell(key)
            return _d(SILENT, "start_anchored")
        if state is not None and state.armed is False:  # an absent bit defaults to armed -> fresh edge
            if hold_s is not None and hold_s > 0:
                self.clear_pending_dwell(key)
            return _d(SILENT, "already_fired")  # already true / already fired this episode
        # Precision track: time-refractory, checked AFTER the edge/require-false-first logic (a fire must already be
        # "due") and BEFORE it is actually emitted — so a suppressed fire still counts as this episode's
        # fire (armed -> False) rather than leaving the edge live to fire again next tick.
        effective_refractory = self.refractory_s if refractory_s is None else refractory_s
        if effective_refractory is not None and effective_refractory > 0 and now is not None:
            last_fire = None if state is None else state.last_fire
            if last_fire is not None and (now - last_fire) < effective_refractory:
                self._state(key).armed = False  # already-fired; a later confident-false re-arms it
                self.clear_pending_dwell(key)
                return _d(SILENT, "refractory")
        if hold_s is None or hold_s <= 0 or now is None:
            fired = self._state(key)
            fired.armed = False
            if now is not None:
                fired.last_fire = now
            return _d(FIRE, "rising_edge")  # no dwell -> the rising-edge fire (byte-for-byte)
        # dwell: require the condition to have held continuously for hold_s before firing.
        t0 = None if state is None else state.dwell_since
        last_supported = None if state is None else state.dwell_last_supported
        if (
            t0 is not None
            and last_supported is not None
            and max_support_gap_s is not None
            and max_support_gap_s > 0
            and now - last_supported > max_support_gap_s
        ):
            # An unmarked stall longer than the supported-read budget cannot prove continuity.
            t0 = None
            self.clear_pending_dwell(key)
        if t0 is None:
            pending = self._state(key)
            pending.dwell_since = now  # first sustained-true tick -> start the dwell timer
            pending.dwell_last_supported = now
            return _d(SILENT, "dwell_pending")
        held = self._state(key)
        held.dwell_last_supported = now
        if now - t0 >= hold_s:
            held.armed = False
            self.clear_pending_dwell(key)
            self._state(key).last_fire = now
            return _d(FIRE, "dwell_satisfied")
        return _d(SILENT, "dwell_pending")  # still accumulating dwell — true, but not long enough yet

    def rearm(self, key: Hashable) -> None:
        """Force a key back to armed (e.g. after a correction clears the edge). Also clears any
        refractory memory — an explicit rearm is a fresh episode boundary, so a stale last-fire stamp
        must not suppress the next legitimate fire."""
        state = self._state(key)
        state.armed = True
        state.last_fire = None
        self.clear_pending_dwell(key)

    def clear_pending_dwell(self, key: Hashable) -> None:
        """Drop only unproven duration support for ``key``; never re-arm or erase a prior fire."""

        state = self._keys.get(key)
        if state is None:
            return
        state.dwell_since = None
        state.dwell_last_supported = None
        self._forget_if_empty(key)

    def clear_pending_dwell_goal(self, goal_id: str) -> None:
        """Drop pending duration support for every reducer key owned by ``goal_id``."""

        for key in [
            k for k, st in self._keys.items()
            if (st.dwell_since is not None or st.dwell_last_supported is not None)
            and isinstance(k, tuple) and k and k[0] == goal_id
        ]:
            self.clear_pending_dwell(key)

    def _goal_keys(self, goal_id: str) -> list:
        return [k for k in self._keys if isinstance(k, tuple) and k and k[0] == goal_id]

    def reset_goal(self, goal_id: str) -> None:
        """Drop all edge state for a goal so its next evaluation starts fresh.

        ONE sweep over ONE keyspace. This used to be three sweeps over three of the six maps —
        `_armed`, then `_dwell_since`, then `_dwell_last_supported` — precisely because a key can
        legitimately exist in one and not the others, and `reset_entity` (which had only the first)
        is what that arrangement eventually broke."""
        for key in self._goal_keys(goal_id):
            self._drop(key)

    def reset_entity(self, entity_id: Hashable) -> None:
        """Drop every gate key belonging to a dead entity (``(goal_id, entity_id)``), so an evicted
        entity leaves no orphan edge behind. Wired to :meth:`EntityGraph` eviction by :class:`Percept`.
        The per-goal singletons (``(goal_id, "__resolved__")`` etc.) never match — their second element is
        a sentinel string, not an entity id — so they are untouched.

        A NON-ENTITY scope is refused outright (:func:`~percept.identity_policy.is_entity_scoped`).
        The scene sentinel is not an entity id, it is a per-goal singleton, and dropping its record
        would discard the `no_evict` exemption the transition path registered — an absent gate key
        defaults to ARMED, so the spent edge re-opens and a still-held ``to`` state re-notifies. The
        primary guard is at the shared eviction hook (``Percept._on_entity_evicted``); this one
        makes a DIRECT call safe too, and mirrors the identical refusal in
        :meth:`percept.behavior.BehaviorExecutor.forget_entity` so the two halves of one latch can
        no longer disagree about the same collision.

        Note what this deliberately does NOT do: honour `no_evict` in general. Per-entity never-goal
        keys are marked `no_evict` and MUST die with their entity — a blanket exemption would leak
        every one of them."""
        if not is_entity_scoped(entity_id):
            return
        for key in [
            k for k in self._keys
            if isinstance(k, tuple) and len(k) >= 2 and k[1] == entity_id
        ]:
            self._drop(key)

    # -- persistence (the armed edge must survive a restart, or a 24/7 watch re-fires) ---------
    @staticmethod
    def _wire_key(key: Hashable) -> list:
        """A key on the wire. ADR-006 §6 pins the gate key tuple as a PERSISTENCE contract: it
        serializes as ``list(k)`` and re-tuples on the way back in, and widening it silently
        invalidates every stored snapshot."""
        return list(key) if isinstance(key, (list, tuple)) else [key]

    def snapshot(self) -> dict:
        """Durable edge state, in TWO representations of the same keyspace.

        * ``keys`` (v2, authoritative) — one SELF-DESCRIBING record per key, each field named on the
          entry that owns it. This is what :meth:`restore` reads. The point of the shape is that a
          lost nesting level becomes *unrepresentable* rather than merely detectable: there is no
          positional slot a flattened key can slide into and be mistaken for a (key, value) pair.
        * ``armed`` / ``dwell_since`` / ``touched`` / ``no_evict`` / ``last_fire`` (v1 pair-lists) —
          kept so a snapshot written by THIS build still restores on an OLDER one. They are a mirror,
          never an input to this build's own restore, which is exactly why tampering with one cannot
          corrupt a round trip.

        Keys are tuples ``(goal_id, entity[, answer])`` serialized as lists, so the whole payload
        stays JSON-able through the persistence backend. The in-progress dwell timer is included for
        inspection but is NOT restored (see :meth:`restore`)."""
        items = list(self._keys.items())
        return {
            "version": SNAPSHOT_VERSION,
            "keys": [
                {
                    "key": self._wire_key(k),
                    "armed": st.armed,
                    "dwell_since": st.dwell_since,
                    "touched": st.touched,
                    "no_evict": st.no_evict,
                    "last_fire": st.last_fire,
                }
                for k, st in items
            ],
            # -- v1 mirror, for an older reader ---------------------------------------------
            "armed": [[self._wire_key(k), bool(st.armed)]
                      for k, st in items if st.armed is not None],
            "dwell_since": [[self._wire_key(k), float(st.dwell_since)]
                            for k, st in items if st.dwell_since is not None],
            # R-1: the last-touched eviction stamps + the no-evict exemption set travel with the edge, so a
            # restarted 24/7 watch keeps its TTL budget and its never-once exemptions (a fresh restore of a
            # legacy snapshot lacking these keys is a no-op — see :meth:`restore`).
            "touched": [[self._wire_key(k), float(st.touched)]
                        for k, st in items if st.touched is not None],
            "no_evict": [self._wire_key(k) for k, st in items if st.no_evict],
            # Precision track: last-real-fire stamps travel too, so a restarted 24/7 watch keeps its refractory memory
            # (otherwise a restart mid-refractory-window would silently re-open it to fire early).
            "last_fire": [[self._wire_key(k), float(st.last_fire)]
                          for k, st in items if st.last_fire is not None],
        }

    @staticmethod
    def _record_to_state(record: dict) -> tuple:
        """One self-describing ``keys`` record -> ``(key, _KeyState)``. Raises on an unusable record,
        which :func:`~percept._snapshot.parse_records` turns into a logged skip of that record alone."""
        key = key_tuple(record["key"])
        armed = record.get("armed")
        touched = record.get("touched")
        last_fire = record.get("last_fire")
        return key, _KeyState(
            armed=None if armed is None else as_bool(armed),
            # dwell is deliberately never rehydrated — see :meth:`restore`
            dwell_since=None,
            dwell_last_supported=None,
            touched=None if touched is None else as_float(touched),
            no_evict=bool(record.get("no_evict", False)),
            last_fire=None if last_fire is None else as_float(last_fire),
        )

    @classmethod
    def _from_records(cls, raw: object) -> dict:
        return dict(parse_records(raw, "keys", cls._record_to_state))

    @classmethod
    def _from_legacy_pairs(cls, snap: dict) -> dict:
        """The v1 reader: five independent pair-lists, re-assembled into one keyspace.

        Kept because ADR-006 §6 makes the serialized key tuple a persistence contract — a snapshot
        written before v2 must still restore. Each field is parsed independently and a bad ENTRY
        costs only that entry; a flattened ``['goal-a', 'e1']`` is SKIPPED WITH A LOG rather than
        reinterpreted as ``tuple('goal-a') -> True``, which is how it used to manufacture a
        per-character key and leave the real one absent (and therefore ARMED).

        One coupling rule spans the five lists, and it is the whole of BUG-D-02-01: **a key whose
        value was PRESENT BUT UNREADABLE in any field is dropped entirely.** Skipping just that field
        would restore the rest of the key — and "the disarmed latch survived, the refractory window
        did not" is the one combination that re-notifies, because the next confident-false blip
        unlatches an edge nothing is left to suppress. A key that was never in a field at all is a
        different case (a legacy payload written before that field existed) and restores normally.
        The v2 record reader gets the same rule for free: a bad field fails its whole record."""
        keys: dict[Hashable, _KeyState] = {}
        poisoned: set[Hashable] = set()

        def state(key: Hashable) -> _KeyState:
            return keys.setdefault(key, _KeyState())

        def poison(key: Hashable) -> None:
            if key is not None:
                poisoned.add(key)

        for key, value in parse_pairs(
            snap.get("armed"), "armed", key_tuple, as_bool, on_skip=poison
        ).items():
            state(key).armed = value
        for key, value in parse_pairs(
            snap.get("touched"), "touched", key_tuple, as_float, on_skip=poison
        ).items():
            state(key).touched = value
        for key in parse_keys(snap.get("no_evict"), "no_evict", on_skip=poison):
            state(key).no_evict = True
        # Precision track: restore last-fire stamps; missing field (legacy pre-refractory snapshot) ->
        # nothing to resume, byte-for-byte the earlier restore.
        for key, value in parse_pairs(
            snap.get("last_fire"), "last_fire", key_tuple, as_float, on_skip=poison
        ).items():
            state(key).last_fire = value
        # ``dwell_since`` is present in the payload for inspection and deliberately not read.
        for key in poisoned:
            keys.pop(key, None)
        return keys

    def restore(self, snap: "dict | None") -> None:
        """Rehydrate the armed/disarmed edge per key (the anti-re-fire bit). A missing/empty snapshot is a
        no-op (byte-for-byte a fresh gate).

        **Validate, then swap — never assign as you go.** The keyspace is rebuilt into a local and
        installed in ONE statement at the end, so a corrupt payload can never leave the gate holding
        half of it. That mattered more here than anywhere else in the package, because the gate runs
        TWO INDEPENDENT suppression belts: the disarmed latch (``armed``) and the refractory memory
        (``last_fire``). Committing the first and failing before the second left a gate that is
        latched shut but has forgotten why — and the next confident-false blip unlatches it and
        re-notifies inside a window that had not expired. Losing exactly one belt is strictly worse
        than losing both.

        Policy on bad input, matching the ledger's long-standing per-item behaviour: a bad ENTRY is
        skipped and logged; an unusable TOP-LEVEL payload is logged and the whole restore is declined,
        leaving the gate exactly as it was. ``restore`` never raises on a restart path — a snapshot a
        backend garbled must not be able to stop a process coming back up.

        The in-progress dwell (``dwell_since``) is EXPLICITLY DROPPED: its start is a wall-clock stamp on
        the pre-restart timeline with no anchor to rebase from, so a half-completed dwell cannot be
        resumed correctly. Dropping it re-arms that edge's dwell from now — the loud, correct failure — in
        preference to silently resurrecting a stamp that would fire early or late."""
        if not snap:
            return
        if not isinstance(snap, dict):
            reject("<root>", "gate snapshot is not a mapping", snap)
            return
        try:
            if snap.get("keys") is not None:
                restored = self._from_records(snap.get("keys"))
            else:
                restored = self._from_legacy_pairs(snap)
        except SnapshotError as exc:
            reject(exc.field, str(exc), snap)
            return
        self._keys = restored  # the swap: one statement, or nothing at all


__all__ = [
    "GateAction", "FIRE", "ASK", "SILENT", "GATE_ACTIONS", "DELIVERED_ACTIONS",
    "GateReason", "GATE_REASONS", "REASON_ACTION", "SUPPRESSION_REASONS", "GateOutcome",
    "classify", "Gate",
]
