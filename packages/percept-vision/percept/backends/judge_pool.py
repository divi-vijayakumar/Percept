"""Deadline-bounded, capability-aware routing across constructed Vision judges.

``JudgePool`` is a Vision adapter, so it composes with the existing SDK instead
of creating a second judgment path.  The cognition core still consumes exactly
one :class:`~percept.contracts.Verdict`; this adapter only decides which
deployment-provided judge may produce it.

The pool has four load-bearing rules:

* evidence shape is exact -- a native-video or context-holding request never
  silently falls back to a sprite or a stateless ordered-image prompt;
* judge CONTEXT is declared -- a route serves a request only when every key the
  request carries is one the judge-context registry declares AND one this route
  accepts (:attr:`JudgeCapability.context_keys`), so a caller's arbitrary state
  cannot arrive at a model framed as a fact;
* the deadline covers route selection, hedge delay, and provider execution;
* capacity exhaustion and deadline expiry fail closed to an UNKNOWN-shaped
  verdict while retaining bounded route occupancy and an inspectable receipt.

The receipt is the DECLARATION that a decision was made, and its ring is bounded,
so under sustained load some declarations must be discarded.  That discard is
itself reported (ADR-001 §9) -- ``snapshot()['receipts_dropped']``, a per-route
breakdown, a named reason and a ``judge_pool.receipt_evicted`` log line -- because
a receipt ring that quietly scrolls is indistinguishable from a pool that never
refused anything, which is the one reading this whole surface exists to prevent.

Provider adapters can still own their retry/circuit-breaker mechanics.  The
pool's outer deadline is authoritative for the Watch: retries that cannot
finish within it are cancelled, and a cancellation-hostile adapter continues
to occupy its bounded route slot instead of allowing an unbounded task leak.

WHAT THE DEADLINE IS, AND IS NOT
--------------------------------
``deadline_s`` is a **hang guard**, not a latency promise.  It exists so a wedged
or pathologically slow provider cannot hold a route slot indefinitely; it is not
a figure any caller should read as "a judgment arrives within this long".  The
latency Percept actually reports is a measured DISTRIBUTION — see
:mod:`percept.slo` §5 (``JUDGE_ROUND_TRIP_LATENCY`` and friends), which publishes
p25/p50/p95 + max + n + the conditions they were measured under.

This distinction was learned the expensive way.  The deadline was once set BELOW
the median provider call time (1.5s against a p50 of 1.653s), which meant over
half of all reflex calls physically could not finish: on an n=100 corpus 5,224 of
7,304 reflex calls expired and only 960 returned decisive, dragging end-to-end
recall to 0.137 against 0.684 for the same frames on the deliberate path.  The
gap was cancellation, not judgment quality — and the "measured" reflex latency
distribution of that run (p50 1.365s / p95 1.4726s / max 1.5155s) was simply the
deadline's own shadow, since every sample above 1.5s had been thrown away.

WHAT REFLEX IS
--------------
``mode='reflex'`` is the **cheap, terse, bounded-evidence** path.  Positively, and
independent of any clock, a reflex route:

* sends BOUNDED evidence — the most recent frames only (``_REFLEX_MAX_FRAMES``,
  ``percept/backends/anthropic.py``), never the full retained strip;
* asks for a BOUNDED reply — the two fields the gate actually reads
  (``satisfied``/``confidence``), not prose (``_REFLEX_MAX_TOKENS``);
* is ROUTE-SELECTIVE — only adapters that declare ``'reflex'`` in
  :attr:`JudgeCapability.modes` may serve it.

It is therefore the low-token, low-cost lane, and it stays a distinct mode.  What
it is no longer is "the lane defined by a deadline it loses".
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from ..contracts import Verdict, is_valid_verdict
from ..judge_context_producers import (
    RESERVED_CONTEXT_KEYS,
    spec_for,
    unregistered,
)
from ..observability import get_logger, log

EvidenceShape = Literal[
    "single_image",
    "ordered_images",
    "video_clip",
    "context_stream",
    "audio",
    "fused",
]
LatencyMode = Literal["reflex", "deliberate"]

_EVIDENCE_SHAPES = frozenset(
    {
        "single_image",
        "ordered_images",
        "video_clip",
        "context_stream",
        "audio",
        "fused",
    }
)
_LATENCY_MODES = frozenset({"reflex", "deliberate"})
_EVIDENCE_SHAPE_CONTEXT_KEY = "_percept_evidence_shape"
#: Reserved context key naming the LATENCY MODE of the route that dispatched a judge call.
#:
#: Output verbosity is a property of the ROUTE, not of the provider. A reflex route must ask for a
#: verdict it can finish generating inside the pool's deadline; a deliberate route may still pay
#: for prose. The pool is the only component that knows which one it is, so it stamps this key on
#: the context it hands the adapter, and a verbosity-aware backend (openrouter/gemini) reads it.
#:
#: It is stamped ONLY for ``mode='reflex'``. Its ABSENCE means deliberate: an adapter called
#: directly, or through a deliberate pool, keeps the historical verbose contract byte-for-byte.
#: Like every ``_percept_`` key it is filtered out of the prompt's specialist-facts line, so it is
#: never surfaced to the model as a fact.
LATENCY_MODE_CONTEXT_KEY = "_percept_latency_mode"
REFLEX_LATENCY_MODE = "reflex"


# --- the reflex operating point: a hang guard, derived from the measured distribution ------------
#
# DERIVATION (measured, not chosen — build 2d5097d, gemini-3.1-flash-lite, n=100 corpus).
#
# The only UNCENSORED provider-latency figures we have for this corpus are the deliberate-mode ones
# (p50 1.653s, p95 2.2834s), because the reflex distribution of that same run was truncated by the
# very deadline being replaced.  A reflex call sends strictly LESS than a deliberate one (bounded
# frames, bounded reply), so the deliberate p95 is a CONSERVATIVE upper bound on the reflex p95 and
# is used as the working figure until the reflex-uncensored run lands.
#
# The floor the ceiling must clear.  The adapters ship a resilience policy of
# ``with_retry(attempts=3, base_delay_s=0.2)`` (``percept/backends/_resilience.py``).  At 1.5s that
# policy was UNREACHABLE: the pool cancelled partway through the FIRST attempt, so the two retries
# the code pays for could never run.  A deadline that lets the shipped policy actually execute must
# cover three attempts at the measured p95 plus the deterministic backoff between them:
#
#     3 x 2.2834s  +  (0.2s + 0.4s)  =  7.4502s   ->  rounded up to the next whole second = 8.0s
#
# The ceiling it must stay under.  Two guards already exist above this one and must keep their
# ordering, or this deadline becomes dead code:
#
#   * the runtime's own "this judgment has hung" health threshold, ``stall_after_s`` (default 20.0s,
#     ``percept/runtime/worker.py``) — 8.0s is 2.5x below it, so the pool still cancels FIRST and the
#     stall event stays what it is, a report on something the pool failed to bound;
#   * the transport-level per-attempt socket guard, ``_REQUEST_TIMEOUT_MS = 120_000``
#     (``percept/backends/gemini.py``) — the real wedged-socket protection, 15x above this.
#
# Sanity-check against the tail, which is what "generous" has to mean here: 8.0s is 3.50x the
# measured p95 and 3.06x the p99 implied by fitting a log-normal to the measured (p50, p95) pair
# (2.611s; p99.9 = 3.033s).  Comfortably above the tail, well inside the hang guards.
#
# THIS IS A DECLARED COST, NOT A TUNING.  A pipeline that answers in seconds is a different product
# from one that fails to answer in 1.5s; the trade is slower and honest, and no metric was consulted
# in picking the number.
REFLEX_PROVIDER_DEADLINE_S = 8.0

#: The NON-PROVIDER share of one decision — admission queue wait plus decision overhead — measured
#: at 0.241s on the same corpus (see ``percept/backends/anthropic.py``). It does not scale with
#: provider latency, so the old 1.5s/1.9s pair's intent is preserved as an ABSOLUTE +0.4s margin
#: (~1.66x the measured overhead), never as that pair's 1.267x RATIO, which would have inflated the
#: allowance to ~2.1s for no measured reason.
REFLEX_DECISION_MARGIN_S = 0.4

#: The absolute per-decision budget measured from admission, for a caller wiring the async lane's
#: ``judge_decision_deadline_s``. Provider deadline + the non-provider margin above.
REFLEX_DECISION_DEADLINE_S = REFLEX_PROVIDER_DEADLINE_S + REFLEX_DECISION_MARGIN_S


#: How many refused context keys a single receipt may name.  A receipt is retained in a bounded
#: deque, and the refused set is derived from a CALLER's dict, whose size the pool does not control.
#: Naming the first few sorted keys is what makes the refusal diagnosable; naming all of an
#: arbitrary number of them would make trace retention unbounded for no extra diagnostic value.
_MAX_RECEIPT_CONTEXT_KEYS = 8

#: How many undrained terminal routing failures the pool retains.  Unlike the receipt ring this one
#: is CONSUMED (:meth:`JudgePool.take_terminal_error` pops), so it only fills when the deployment
#: that opted into terminal outcomes has stopped draining them — which is itself the condition worth
#: reporting.  A silently evicted terminal error is a routing failure the deployment declared must
#: stop pending work, lost; the drop is counted and logged for that reason.
_TERMINAL_ERROR_LIMIT = 256

#: The reasons this pool's two bounded rings drop with.  ADR-001 §9: a bound-forced drop always
#: names one.
_RECEIPT_BOUND_REASON = "receipt_limit"
_TERMINAL_ERROR_BOUND_REASON = "terminal_error_limit"

_LOGGER = "percept.backends.judge_pool"


@dataclass(frozen=True)
class JudgeCapability:
    """Deployment-reviewed eligibility for one concrete judge adapter.

    ``latency_slo_s`` is a declared/measured routing ceiling, not a timeout.
    Reflex routing considers a route only when this value fits the pool's hard
    deadline.  The hard deadline is enforced independently at runtime.

    ``context_keys`` is the JUDGE CONTEXT this route has agreed to receive.  It defaults to
    :data:`~percept.judge_context_producers.RESERVED_CONTEXT_KEYS` -- the registry's public keys --
    which admits everything the runtime actually emits while refusing anything unregistered.  That
    unregistered case is the leak the registry exists for: an arbitrary caller key had no structural
    barrier and rendered into the prompt under a sentence telling the model to trust it over its own
    read.  Narrowing a route BELOW the default (a judge that cannot honour ``claim_mode``, say) is
    an opt-in a deployment chooses; it is deliberately not a migration every caller is forced
    through, because a ``frozenset()`` default would break every existing declaration and land most
    of them at "paste the full set" anyway.

    A key declared here must itself be registered.  A route declaring a key nothing emits is a typo,
    and construction is a cheaper place to learn that than selection, where it would present as an
    unexplained ``no_route``.  ``_percept_`` transport keys need not be declared -- they are
    admitted by prefix at selection -- and declaring one is redundant rather than wrong.
    """

    evidence_shapes: frozenset[EvidenceShape]
    latency_slo_s: float
    modes: frozenset[LatencyMode] = frozenset({"deliberate"})
    max_inflight: int = 1
    context_keys: frozenset[str] = RESERVED_CONTEXT_KEYS

    def __post_init__(self) -> None:
        shapes = frozenset(self.evidence_shapes)
        modes = frozenset(self.modes)
        context_keys = frozenset(self.context_keys)
        if not shapes or not shapes.issubset(_EVIDENCE_SHAPES):
            raise ValueError(
                "JudgeCapability.evidence_shapes must contain only supported "
                f"shapes, got {sorted(shapes)!r}"
            )
        if not modes or not modes.issubset(_LATENCY_MODES):
            raise ValueError(
                "JudgeCapability.modes must contain only 'reflex'/'deliberate'"
            )
        if (
            isinstance(self.latency_slo_s, bool)
            or not isinstance(self.latency_slo_s, (int, float))
            or not math.isfinite(float(self.latency_slo_s))
            or float(self.latency_slo_s) <= 0
        ):
            raise ValueError(
                "JudgeCapability.latency_slo_s must be a positive finite number"
            )
        if (
            isinstance(self.max_inflight, bool)
            or not isinstance(self.max_inflight, int)
            or self.max_inflight < 1
        ):
            raise ValueError(
                "JudgeCapability.max_inflight must be a positive integer"
            )
        if not all(isinstance(key, str) for key in context_keys):
            raise ValueError(
                "JudgeCapability.context_keys must contain only strings"
            )
        # Asked of the REGISTRY, never of a local list: `unregistered` is where the
        # public-key set and the `_percept_` prefix rule both live, and a second spelling of
        # either here is how the three backends' deny-lists drifted apart in the first place.
        undeclared = unregistered({key: None for key in context_keys})
        if undeclared:
            raise ValueError(
                "JudgeCapability.context_keys must contain only keys the judge-context "
                "registry declares (percept/judge_context_producers.py); "
                f"got undeclared {sorted(undeclared)!r}"
            )
        object.__setattr__(self, "evidence_shapes", shapes)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "context_keys", context_keys)
        object.__setattr__(self, "latency_slo_s", float(self.latency_slo_s))


@dataclass(frozen=True)
class JudgeRoute:
    """One named, already-constructed judge and its reviewed capability."""

    name: str
    adapter: Any
    capability: JudgeCapability
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("JudgeRoute.name must be a non-empty string")
        if not callable(getattr(self.adapter, "judge", None)):
            raise ValueError(
                f"JudgeRoute {self.name!r} adapter has no callable judge()"
            )
        if not isinstance(self.capability, JudgeCapability):
            raise ValueError(
                f"JudgeRoute {self.name!r} requires an exact JudgeCapability"
            )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("JudgeRoute.priority must be an integer")


@dataclass(frozen=True)
class JudgeRouteReceipt:
    """One bounded routing outcome, safe to retain in traces/benchmarks."""

    route: str | None
    evidence_shape: EvidenceShape
    mode: LatencyMode
    outcome: Literal[
        "decisive",
        "uncertain",
        "provider_error",
        "malformed",
        "deadline",
        "capacity",
        "no_route",
        "cancelled",
    ]
    latency_s: float
    deadline_s: float
    confidence: float | None = None
    exception_type: str | None = None
    #: Which context keys emptied the eligible set.  This is what distinguishes the two ways a
    #: ``no_route`` happens: empty means no route could serve the EVIDENCE SHAPE at all, non-empty
    #: means routes existed for the shape and the request carried context none of them accepts.
    #: Without it both refusals read identically in a trace, and the second one -- the fail-closed
    #: path this field exists to explain -- is the one an operator has to act on.  A distinct
    #: ``outcome`` was considered and rejected: the refusal is still "nothing may serve this", it
    #: reuses the existing no-route path exactly, and a new outcome value would silently escape any
    #: deployment's configured ``terminal_outcomes={"no_route"}``.  Key NAMES only, never a value,
    #: and capped at :data:`_MAX_RECEIPT_CONTEXT_KEYS`.
    refused_context_keys: tuple[str, ...] = ()


@dataclass
class _RouteState:
    route: JudgeRoute
    inflight: int = 0
    successes: int = 0
    failures: int = 0
    latency_ewma_s: float | None = None


@dataclass(frozen=True)
class _RouteResult:
    route: JudgeRoute
    verdict: Any = None
    outcome: str = "provider_error"
    latency_s: float = 0.0
    exception_type: str | None = None
    exception: BaseException | None = None


def _valid_verdict(value: Any) -> bool:
    """Is this route's answer a well-formed :class:`Verdict`? (ADR-001 §7, §10 invariant 4.)

    The pool's whole malformed-detection surface: a ``False`` here becomes a ``malformed`` route
    outcome and the route fails over. A one-line alias for the single authority in
    :mod:`percept.contracts.verdict_validation` — this used to be its own six-term implementation,
    and ``percept.agent``'s twin was a DIFFERENT three-term one that accepted the non-finite and
    out-of-range confidences this side has always refused.

    ``require_exact_type=False`` is deliberate and belongs to THIS call site, not to the authority.
    This is the provider seam: a deployment may hand the pool a ``Verdict``-shaped object of its own
    (``_VerdictSubclass`` in the tests is the benign case), and the pool's job here is to classify a
    route's answer as decisive / uncertain / malformed, not to police the type. The agent-side call
    site passes ``True`` because the contract it enforces is the dataclass itself. Making either the
    global default would silently loosen one boundary or break the other.
    """
    return is_valid_verdict(value, require_exact_type=False)


def _context_refusals(
    capability: JudgeCapability,
    context: dict | None,
) -> frozenset[str]:
    """The keys in ``context`` this capability has not agreed to receive.

    There are two kinds and they refuse identically.  A key NOTHING declares is refused by every
    route -- it is a caller's arbitrary state, and the alternative to refusing it is that it reaches
    a model framed as an established fact.  A PUBLIC key that this route narrowed away is refused by
    this route only, and another route in the pool may still take the request.

    A ``_percept_`` transport key is never refused, and that is structural rather than incidental:
    it is registered by prefix so ``unregistered`` passes it, and it has no ``ContextKeySpec`` so
    the public-key branch never sees it.  Both facts are asked of the registry helpers rather than
    re-spelled here, because the pool itself STAMPS one of these keys
    (:data:`LATENCY_MODE_CONTEXT_KEY`, in :meth:`JudgePool._route_context`) -- a predicate that
    tested ``RESERVED_CONTEXT_KEYS`` membership directly would refuse the pool's own key and take
    every reflex route out of service.
    """
    refused = set(unregistered(context))
    refused.update(
        key
        for key in (context or {})
        if spec_for(key) is not None and key not in capability.context_keys
    )
    return frozenset(refused)


def _evidence_shape(
    *,
    frame: Any,
    frames: list | None,
    audio: Any,
    context: dict | None,
) -> EvidenceShape:
    requested = (
        context.get(_EVIDENCE_SHAPE_CONTEXT_KEY)
        if isinstance(context, dict)
        else None
    )
    if requested is not None:
        if requested not in _EVIDENCE_SHAPES:
            raise ValueError(
                f"unsupported requested evidence shape {requested!r}"
            )
        return requested
    if (
        isinstance(context, dict)
        and context.get("event_chain")
        and (frame is not None or frames)
        and audio is None
    ):
        # Motion-sensitive event-chain watches require a declared temporal
        # route. Its physical transport may be native video or an explicitly
        # configured timestamped image burst, but its logical evidence is a
        # bounded temporal clip.
        return "video_clip"
    if audio is not None and (frame is not None or frames):
        return "fused"
    if audio is not None:
        return "audio"
    if frames:
        return "ordered_images"
    return "single_image"


class JudgePool:
    """A bounded Vision judge router with one authoritative hang-guard deadline.

    ``mode='reflex'`` rejects routes whose declared latency SLO exceeds
    ``deadline_s``.  ``mode='deliberate'`` may use any route that explicitly
    declares deliberate support, but the same configured hard deadline still
    bounds this pool instance.

    ``deadline_s`` defaults to :data:`REFLEX_PROVIDER_DEADLINE_S` — a ceiling
    sized so a judgment COMPLETES (see that constant's derivation), not a
    latency promise.  Percept's latency promise is the measured distribution in
    :mod:`percept.slo` §5, never this number.
    """

    def __init__(
        self,
        routes: Iterable[JudgeRoute],
        *,
        mode: LatencyMode = "reflex",
        deadline_s: float = REFLEX_PROVIDER_DEADLINE_S,
        hedge_after_s: float = 0.2,
        max_parallel: int = 2,
        min_confidence: float = 0.7,
        receipt_limit: int = 256,
        detector: Any = None,
        on_receipt: Callable[[JudgeRouteReceipt], Any] | None = None,
        terminal_outcomes: Iterable[str] = (),
    ) -> None:
        route_list = sorted(
            tuple(routes), key=lambda item: (item.priority, item.name)
        )
        if not route_list:
            raise ValueError("JudgePool requires at least one JudgeRoute")
        names = [route.name for route in route_list]
        if len(set(names)) != len(names):
            raise ValueError("JudgePool route names must be unique")
        if mode not in _LATENCY_MODES:
            raise ValueError("JudgePool.mode must be 'reflex' or 'deliberate'")
        for name, value, allow_zero in (
            ("deadline_s", deadline_s, False),
            ("hedge_after_s", hedge_after_s, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < (0.0 if allow_zero else 1e-12)
            ):
                raise ValueError(f"JudgePool.{name} has an invalid value")
        if (
            isinstance(max_parallel, bool)
            or not isinstance(max_parallel, int)
            or max_parallel < 1
        ):
            raise ValueError("JudgePool.max_parallel must be positive")
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("JudgePool.min_confidence must be in [0, 1]")
        if (
            isinstance(receipt_limit, bool)
            or not isinstance(receipt_limit, int)
            or receipt_limit < 1
        ):
            raise ValueError("JudgePool.receipt_limit must be positive")

        self._routes = route_list
        self._states = {route.name: _RouteState(route) for route in route_list}
        self._mode = mode
        self._deadline_s = float(deadline_s)
        self._hedge_after_s = float(hedge_after_s)
        self._max_parallel = max_parallel
        self._min_confidence = float(min_confidence)
        self._detector = detector
        self._on_receipt = on_receipt
        terminal = frozenset(terminal_outcomes)
        supported_outcomes = {
            "provider_error",
            "malformed",
            "deadline",
            "capacity",
            "no_route",
        }
        if not terminal.issubset(supported_outcomes):
            raise ValueError(
                "JudgePool.terminal_outcomes contains unsupported outcomes"
            )
        self._terminal_outcomes = terminal
        # Both rings are bounded, and ADR-001 §9 says a bound is only half the rule: "a drop forced
        # by a bound is always reported with a reason, never a silent counter bump ... a drop the
        # operator cannot see is worse than the resource exhaustion the bound prevents: it looks
        # exactly like a quiet, working system." A deque at its maxlen discards silently, so every
        # append below goes through a helper that counts the eviction, names its reason, and logs
        # it — the idiom `percept.runtime._bounds.BoundedRecencyMap` already uses. The counters are
        # published on :meth:`snapshot` so `receipts()` can never be mistaken for a COMPLETE record
        # of what this pool decided.
        self._receipts: deque[JudgeRouteReceipt] = deque(maxlen=receipt_limit)
        self._state_lock = threading.Lock()
        self._terminal_errors: deque[BaseException] = deque(maxlen=_TERMINAL_ERROR_LIMIT)
        self._receipts_dropped = 0
        self._receipts_dropped_by_route: dict[str, int] = {}
        self._terminal_errors_dropped = 0

    @property
    def mode(self) -> LatencyMode:
        return self._mode

    @property
    def deadline_s(self) -> float:
        return self._deadline_s

    @property
    def max_inflight(self) -> int:
        """How many CONCURRENT decisions this pool can accept without failing one closed.

        ``JudgeCapability.max_inflight`` bounds occupancy of ONE route across concurrent
        ``judge()`` calls, and :meth:`judge` schedules EVERY eligible route for each decision (that
        is what makes capacity exhaustion on a preferred route fall through to a healthy one).  So D
        simultaneous decisions need D free slots on every route they touch, and the number of
        decisions this pool can serve without any of them landing on the ``"capacity"`` outcome is
        the MINIMUM declared route occupancy — not the sum, and not the maximum.

        This is a DECLARATION, read by a batching caller so it can queue behind the bound instead of
        overrunning it: the pool itself has no queue, so an over-eager caller does not get slower,
        it gets ``"capacity"``-shaped UNKNOWN verdicts.  The pool's own enforcement in
        :meth:`_reserve` is unchanged and remains authoritative.
        """
        return min(
            route.capability.max_inflight for route in self._routes
        )

    def receipts(self) -> tuple[JudgeRouteReceipt, ...]:
        """The retained routing declarations — the most recent ``receipt_limit`` of them.

        NOT necessarily every decision this pool made: the ring is bounded, and under sustained load
        it will have discarded older ones. :meth:`receipts_dropped` (and ``snapshot()``) say how
        many, so this tuple is never mistaken for a complete record.
        """
        with self._state_lock:
            return tuple(self._receipts)

    def receipts_dropped(self) -> int:
        """How many declarations the receipt ring has discarded to stay inside its bound."""
        with self._state_lock:
            return self._receipts_dropped

    def snapshot(self) -> dict[str, Any]:
        """Return bounded operational state without adapter objects or secrets.

        ``receipts_dropped`` / ``terminal_errors_dropped`` are the ADR-001 §9 half a bound on its
        own does not give: the count of declarations each ring had to discard, aggregate and (for
        receipts) per route, so an operator draining :meth:`receipts` after an incident can tell a
        pool that refused nothing from one whose refusals scrolled off the end.
        """
        with self._state_lock:
            routes = {
                name: {
                    "inflight": state.inflight,
                    "max_inflight": state.route.capability.max_inflight,
                    "successes": state.successes,
                    "failures": state.failures,
                    "latency_ewma_s": state.latency_ewma_s,
                    # Which route's declarations were lost, not merely how many: a receipt ring
                    # overrun by a saturated route says nothing about a quiet one, and an operator
                    # reading a gap has to know where it is.
                    "receipts_dropped": self._receipts_dropped_by_route.get(name, 0),
                    "evidence_shapes": sorted(
                        state.route.capability.evidence_shapes
                    ),
                    "modes": sorted(state.route.capability.modes),
                    # Published for the same reason `evidence_shapes` and `modes` are: it is now a
                    # fourth input to eligibility, and an operator reading a `no_route` receipt has
                    # to be able to see which routes would have accepted the key it names. A
                    # declaration, bounded by the registry -- no adapter object, no secret.
                    "context_keys": sorted(state.route.capability.context_keys),
                    "latency_slo_s": state.route.capability.latency_slo_s,
                }
                for name, state in self._states.items()
            }
            receipts_dropped = self._receipts_dropped
            terminal_errors_dropped = self._terminal_errors_dropped
        return {
            "mode": self._mode,
            "deadline_s": self._deadline_s,
            "max_parallel": self._max_parallel,
            "receipt_limit": self._receipts.maxlen,
            "receipts_dropped": receipts_dropped,
            "receipts_dropped_reason": _RECEIPT_BOUND_REASON if receipts_dropped else None,
            "terminal_errors_dropped": terminal_errors_dropped,
            "terminal_errors_dropped_reason": (
                _TERMINAL_ERROR_BOUND_REASON if terminal_errors_dropped else None
            ),
            "routes": routes,
        }

    def take_terminal_error(self) -> BaseException | None:
        """Consume a configured terminal routing failure after cognition degrades it.

        Percept deliberately converts provider exceptions to an UNKNOWN verdict.
        An async runtime or paid qualification lane may still need to stop
        already-accepted pending work. This opt-in latch preserves that signal
        without changing the default fail-closed cognition contract.
        """
        with self._state_lock:
            error = (
                self._terminal_errors.popleft()
                if self._terminal_errors
                else None
            )
        return error

    def _set_terminal_error(self, error: BaseException) -> None:
        evicted: BaseException | None = None
        dropped_total = 0
        with self._state_lock:
            if len(self._terminal_errors) == self._terminal_errors.maxlen:
                evicted = self._terminal_errors[0]
                self._terminal_errors_dropped += 1
                dropped_total = self._terminal_errors_dropped
            self._terminal_errors.append(error)
        if evicted is not None:
            log(
                get_logger(_LOGGER),
                "judge_pool.terminal_error_evicted",
                reason=_TERMINAL_ERROR_BOUND_REASON,
                exception_type=type(evicted).__name__,
                retained=self._terminal_errors.maxlen,
                dropped_total=dropped_total,
            )

    def _route_context(self, context: dict | None) -> dict | None:
        """Stamp the route's latency mode onto the context handed to the adapter.

        A reflex route gets :data:`LATENCY_MODE_CONTEXT_KEY` so a verbosity-aware backend can ask
        for a bounded verdict instead of a prose one. A deliberate route is returned untouched —
        including a ``None`` context, which stays ``None`` — so the deliberate path is byte-for-byte
        the pre-existing call. The caller's dict is never mutated.
        """
        if self._mode != REFLEX_LATENCY_MODE:
            return context
        return {**(context or {}), LATENCY_MODE_CONTEXT_KEY: REFLEX_LATENCY_MODE}

    def _transport_eligible(
        self,
        shape: EvidenceShape,
    ) -> tuple[JudgeRoute, ...]:
        """Routes eligible on TRANSPORT alone -- evidence shape, mode, and the reflex SLO fit.

        Split out from :meth:`_eligible` so a refusal can be ATTRIBUTED.  If this is already empty
        the miss belongs to the evidence shape and no context key caused it; if it is non-empty and
        ``_eligible`` is not, the context is what emptied the set.  A receipt that guessed between
        those two would be worse than one that said nothing.
        """
        return tuple(
            route
            for route in self._routes
            if shape in route.capability.evidence_shapes
            and self._mode in route.capability.modes
            and (
                self._mode != "reflex"
                or route.capability.latency_slo_s <= self._deadline_s
            )
        )

    def _eligible(
        self,
        shape: EvidenceShape,
        context: dict | None = None,
    ) -> tuple[JudgeRoute, ...]:
        """Routes that may serve this request: transport, plus the context they accept.

        The fourth predicate reads the CALLER's context, not the one :meth:`_route_context` hands
        the adapter.  The pool stamps its own transport key on the latter, and route eligibility
        must never turn on something the pool itself added.
        """
        return tuple(
            route
            for route in self._transport_eligible(shape)
            if not _context_refusals(route.capability, context)
        )

    def _refused_context_keys(
        self,
        shape: EvidenceShape,
        context: dict | None,
    ) -> tuple[str, ...]:
        """The context keys that emptied the eligible set, or ``()`` if the evidence shape did.

        The union across every route that WAS transport-eligible: with two routes refusing
        different keys, no route survives and both keys are implicated in why.
        """
        transport_eligible = self._transport_eligible(shape)
        if not transport_eligible:
            return ()
        refused: set[str] = set()
        for route in transport_eligible:
            refused.update(_context_refusals(route.capability, context))
        return tuple(sorted(refused))[:_MAX_RECEIPT_CONTEXT_KEYS]

    def _reserve(self, route: JudgeRoute) -> bool:
        with self._state_lock:
            state = self._states[route.name]
            if state.inflight >= route.capability.max_inflight:
                return False
            state.inflight += 1
            return True

    def _release(
        self,
        route: JudgeRoute,
        *,
        success: bool,
        latency_s: float,
    ) -> None:
        with self._state_lock:
            state = self._states[route.name]
            state.inflight = max(0, state.inflight - 1)
            if success:
                state.successes += 1
            else:
                state.failures += 1
            if latency_s >= 0:
                state.latency_ewma_s = (
                    latency_s
                    if state.latency_ewma_s is None
                    else 0.2 * latency_s + 0.8 * state.latency_ewma_s
                )

    def _record(self, receipt: JudgeRouteReceipt) -> None:
        evicted: JudgeRouteReceipt | None = None
        dropped_total = 0
        with self._state_lock:
            if len(self._receipts) == self._receipts.maxlen:
                # The deque is about to discard this one on append. Read it FIRST: after the append
                # the declaration is gone and nothing can say what it was.
                evicted = self._receipts[0]
                self._receipts_dropped += 1
                route = evicted.route or "-"
                self._receipts_dropped_by_route[route] = (
                    self._receipts_dropped_by_route.get(route, 0) + 1
                )
                dropped_total = self._receipts_dropped
            self._receipts.append(receipt)
        if evicted is not None:
            log(
                get_logger(_LOGGER),
                "judge_pool.receipt_evicted",
                reason=_RECEIPT_BOUND_REASON,
                route=evicted.route,
                outcome=evicted.outcome,
                retained=self._receipts.maxlen,
                dropped_total=dropped_total,
            )
        if self._on_receipt is not None:
            try:
                self._on_receipt(receipt)
            except Exception:
                # Observability must never become the judgment failure.
                pass

    async def _call_route(
        self,
        route: JudgeRoute,
        delay_s: float,
        condition: str,
        frame: Any,
        kwargs: dict[str, Any],
        dispatch_slots: asyncio.Semaphore,
    ) -> _RouteResult:
        if delay_s:
            await asyncio.sleep(delay_s)
        async with dispatch_slots:
            if not self._reserve(route):
                return _RouteResult(route=route, outcome="capacity")
            started = time.monotonic()
            success = False
            try:
                try:
                    verdict = await route.adapter.judge(
                        condition,
                        frame,
                        **kwargs,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - route failure is fail-closed
                    return _RouteResult(
                        route=route,
                        outcome="provider_error",
                        latency_s=time.monotonic() - started,
                        exception_type=type(exc).__name__,
                        exception=exc,
                    )
                if not _valid_verdict(verdict):
                    return _RouteResult(
                        route=route,
                        verdict=verdict,
                        outcome="malformed",
                        latency_s=time.monotonic() - started,
                    )
                success = True
                outcome = (
                    "decisive"
                    if float(verdict.confidence) >= self._min_confidence
                    else "uncertain"
                )
                return _RouteResult(
                    route=route,
                    verdict=verdict,
                    outcome=outcome,
                    latency_s=time.monotonic() - started,
                )
            finally:
                self._release(
                    route,
                    success=success,
                    latency_s=time.monotonic() - started,
                )

    @staticmethod
    def _consume_task(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    def _unknown(
        self,
        shape: EvidenceShape,
        reason: str,
    ) -> Verdict:
        return Verdict(
            satisfied=False,
            confidence=0.0,
            evidence=(
                f"judge pool unavailable ({reason}); mode={self._mode}; "
                f"evidence_shape={shape}; deadline_s={self._deadline_s:g}"
            ),
        )

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        shape = _evidence_shape(
            frame=frame,
            frames=frames,
            audio=audio,
            context=context,
        )
        eligible = self._eligible(shape, context)
        if not eligible:
            # An unaccepted context key is a ROUTING miss, not a new kind of failure: nothing may
            # serve the request, which is what `no_route` already means and already fails closed.
            # Only the receipt and the wording distinguish it, so a deployment that made `no_route`
            # terminal still stops here, and one that did not still degrades to UNKNOWN.
            refused = self._refused_context_keys(shape, context)
            detail = (
                "no route accepts context key(s) " + ", ".join(refused)
                if refused
                else "no eligible route"
            )
            self._record(
                JudgeRouteReceipt(
                    route=None,
                    evidence_shape=shape,
                    mode=self._mode,
                    outcome="no_route",
                    latency_s=0.0,
                    deadline_s=self._deadline_s,
                    refused_context_keys=refused,
                )
            )
            if "no_route" in self._terminal_outcomes:
                error = RuntimeError(
                    f"judge pool has no {shape} route accepting context "
                    f"key(s) {', '.join(refused)}"
                    if refused
                    else f"judge pool has no eligible {shape} route"
                )
                self._set_terminal_error(error)
                raise error
            return self._unknown(shape, detail)

        kwargs = {
            "roi": roi,
            "exemplar": exemplar,
            "corrections": corrections,
            "frames": frames,
            "audio": audio,
            # The evidence shape was already resolved off the CALLER's context above; the adapter
            # additionally needs to know which route it is serving, so verbosity can follow the
            # route instead of being a global downgrade.
            "context": self._route_context(context),
        }
        # Schedule every eligible route so capacity exhaustion on a preferred
        # route can fall through to a healthy lower-priority route. The
        # semaphore bounds physical provider execution for this decision;
        # route-level max_inflight bounds occupancy across concurrent
        # decisions.
        selected = eligible
        if shape == "video_clip":
            # If a deployment publishes both exact temporal transports, prefer
            # native video before the timestamped-image alternative. Both are
            # explicit video_clip routes; this only resolves the transport
            # tie, without inventing a fallback for ordinary image judges.
            selected = sorted(
                eligible,
                key=lambda route: (
                    0
                    if getattr(route.adapter, "TEMPORAL_TRANSPORT", None)
                    == "native_video"
                    else 1,
                    route.priority,
                    route.name,
                ),
            )
        dispatch_slots = asyncio.Semaphore(self._max_parallel)
        tasks: set[asyncio.Task] = set()
        task_routes: dict[asyncio.Task, JudgeRoute] = {}
        for index, route in enumerate(selected):
            task = asyncio.create_task(
                self._call_route(
                    route,
                    min(index * self._hedge_after_s, self._deadline_s),
                    condition,
                    frame,
                    kwargs,
                    dispatch_slots,
                )
            )
            task.add_done_callback(self._consume_task)
            tasks.add(task)
            task_routes[task] = route

        cancelled_receipts: set[asyncio.Task] = set()
        try:
            loop = asyncio.get_running_loop()
            expires_at = loop.time() + self._deadline_s
            uncertain: Verdict | None = None
            while tasks:
                remaining = expires_at - loop.time()
                if remaining <= 0:
                    break
                done, tasks = await asyncio.wait(
                    tasks,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    break
                # ``asyncio.wait`` returns a set. Resolve a simultaneous hedge
                # batch in declared route priority order, record every
                # physical completion, and let any configured terminal outcome
                # win before accepting a decisive result.
                results = [
                    task.result()
                    for task in task_routes
                    if task in done
                ]
                for result in results:
                    confidence = (
                        float(result.verdict.confidence)
                        if _valid_verdict(result.verdict)
                        else None
                    )
                    self._record(
                        JudgeRouteReceipt(
                            route=result.route.name,
                            evidence_shape=shape,
                            mode=self._mode,
                            outcome=result.outcome,
                            latency_s=result.latency_s,
                            deadline_s=self._deadline_s,
                            confidence=confidence,
                            exception_type=result.exception_type,
                        )
                    )
                terminal_result = next(
                    (
                        result
                        for result in results
                        if result.outcome in self._terminal_outcomes
                    ),
                    None,
                )
                if terminal_result is not None:
                    if terminal_result.exception is not None:
                        self._set_terminal_error(
                            terminal_result.exception
                        )
                        raise terminal_result.exception
                    error = RuntimeError(
                        "judge pool terminal route outcome: "
                        f"{terminal_result.outcome} "
                        f"({terminal_result.route.name})"
                    )
                    self._set_terminal_error(error)
                    raise error
                decisive = next(
                    (
                        result
                        for result in results
                        if result.outcome == "decisive"
                    ),
                    None,
                )
                if decisive is not None:
                    return decisive.verdict
                for result in results:
                    if result.outcome == "uncertain":
                        if (
                            uncertain is None
                            or float(result.verdict.confidence)
                            > float(uncertain.confidence)
                        ):
                            uncertain = result.verdict

            if not tasks:
                return uncertain or self._unknown(
                    shape,
                    "all eligible routes were unavailable",
                )
            self._record(
                JudgeRouteReceipt(
                    route=None,
                    evidence_shape=shape,
                    mode=self._mode,
                    outcome="deadline",
                    latency_s=self._deadline_s,
                    deadline_s=self._deadline_s,
                    confidence=(
                        float(uncertain.confidence)
                        if uncertain is not None
                        else None
                    ),
                )
            )
            if "deadline" in self._terminal_outcomes:
                error = TimeoutError(
                    f"judge pool deadline expired after {self._deadline_s:g}s"
                )
                self._set_terminal_error(error)
                raise error
            return uncertain or self._unknown(shape, "deadline expired")
        finally:
            for task in task_routes:
                if task.done():
                    continue
                task.cancel()
                cancelled_receipts.add(task)
                self._record(
                    JudgeRouteReceipt(
                        route=task_routes[task].name,
                        evidence_shape=shape,
                        mode=self._mode,
                        outcome="cancelled",
                        latency_s=0.0,
                        deadline_s=self._deadline_s,
                    )
                )
            if cancelled_receipts:
                # One non-blocking turn lets cancellation-friendly adapters
                # release route occupancy. Cancellation-hostile adapters stay
                # visible as inflight/capacity until they really terminate.
                await asyncio.sleep(0)

    async def detect(self, frame: Any, goals: list[Any]) -> list[Any]:
        if self._detector is None:
            return []
        try:
            result = await asyncio.wait_for(
                self._detector.detect(frame, goals),
                timeout=self._deadline_s,
            )
        except Exception:
            return []
        return result if isinstance(result, list) else []

    async def detect_events(
        self,
        frames: list,
        goals: list[Any],
    ) -> list[Any]:
        if self._detector is None:
            return []
        method = getattr(self._detector, "detect_events", None)
        if not callable(method):
            return []
        try:
            result = await asyncio.wait_for(
                method(frames, goals),
                timeout=self._deadline_s,
            )
        except Exception:
            return []
        return result if isinstance(result, list) else []


__all__ = [
    "EvidenceShape",
    "JudgeCapability",
    "JudgePool",
    "JudgeRoute",
    "JudgeRouteReceipt",
    "LatencyMode",
    "LATENCY_MODE_CONTEXT_KEY",
    "REFLEX_LATENCY_MODE",
    "REFLEX_PROVIDER_DEADLINE_S",
    "REFLEX_DECISION_MARGIN_S",
    "REFLEX_DECISION_DEADLINE_S",
]
