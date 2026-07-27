"""The runtime worker — ``run(core, source) -> incidents``.

The IRA's #1 ask, made generic: own the camera→incidents lifecycle so apps stop writing the capture +
threading + feed glue. Per frame it drives the tier-0 :class:`~percept.edge.tier0.MotionGate`, and on
the rising edge spends a cloud judgment (``core.perceive_judged``), turning each fire into a durable
:class:`~percept.contracts.Incident` (with a pre-roll snippet) handed to the notification sink and yielded.

The **cognition is unchanged** — this *drives* ``vision.judge`` / the three-state gate, it doesn't edit
them. ``source`` is a URI string (file / RTSP / ``webcam:0``), opened on a capture thread, **or** an
iterable of versioned ``SourceEnvelope`` items. Legacy ``(t, gate_frame, judge_frame)`` tuples remain
accepted and are normalized at ingress.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import math
import threading
import time
import uuid
import warnings
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Callable

from percept.events import (
    AudioBoundary,
    AudioUnit,
    FrameWindow,
    frame_nbytes,
    validate_audio_unit,
)
from percept.gate import FIRE, ASK
from percept.incidents import make_incident, snippet_from_window
from percept.interest import (
    InterestCallbackUnsupportedError,
    InterestRunError,
    InterestSubscription,
    LiveInterestAdmission,
    LiveInterestTracker,
)
from percept.observability import get_logger
from percept.runtime.envelope import (
    SourceEnvelope,
    SourceFormat,
    SourceInspection,
    SourceTracker,
    audio_unit_envelope,
    legacy_audio_envelope,
    legacy_video_envelope,
)

_log = get_logger("percept.runtime.worker")


class ConcurrentCoreRunError(RuntimeError):
    """The same cognition core is already owned by another active runtime."""


_ACTIVE_CORE_RUNS_LOCK = threading.Lock()
_ACTIVE_CORE_RUNS: dict[int, Any] = {}


def _acquire_core_run(core: Any) -> tuple[int, Any] | None:
    """Acquire a process-local, thread-safe lease without requiring a hashable/weakref core."""
    if core is None:
        return None
    identity = id(core)
    with _ACTIVE_CORE_RUNS_LOCK:
        active = _ACTIVE_CORE_RUNS.get(identity)
        if active is core:
            raise ConcurrentCoreRunError(
                "percept.run already has an active runtime for this core; "
                "wait for it to finish before reusing the same core"
            )
        # Keeping the active object itself prevents id reuse and retains only active cores.
        _ACTIVE_CORE_RUNS[identity] = core
    return identity, core


def _release_core_run(lease: tuple[int, Any] | None) -> None:
    if lease is None:
        return
    identity, core = lease
    with _ACTIVE_CORE_RUNS_LOCK:
        if _ACTIVE_CORE_RUNS.get(identity) is core:
            del _ACTIVE_CORE_RUNS[identity]


def _exclusive_core_runtime(func):
    """Guard an async-generator runtime before any wrapped-body work can execute."""
    @functools.wraps(func)
    async def leased(core: Any = None, *args: Any, **kwargs: Any):
        lease = _acquire_core_run(core)
        try:
            async for item in func(core, *args, **kwargs):
                yield item
        finally:
            _release_core_run(lease)

    return leased


def _accepts_keyword(callable_obj: Any, name: str) -> bool:
    """Whether a duck-typed callable explicitly accepts ``name`` or arbitrary keywords.

    Signature inspection happens before invocation: an internal ``TypeError`` raised by the callable
    is therefore never mistaken for an unsupported keyword and retried/swallowed.
    """
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _filter_goal_events(events: Any, allowed_goal_ids: tuple[str, ...]) -> list:
    """Delivery isolation for a legacy callback that cannot accept ``goal_ids`` itself."""
    allowed = frozenset(allowed_goal_ids)
    return [
        event
        for event in (events or ())
        if getattr(event, "goal_id", None) in allowed
    ]


@dataclass
class _DecisionTerminal:
    """One scheduler-owned terminal latch shared with a legacy cognition call."""

    terminal: bool = False
    reason: str | None = None

    def expire(self, reason: str) -> None:
        self.reason = reason
        self.terminal = True


_LEGACY_DEADLINE_EXPIRED = object()


# The vocabulary of REASONS a run's effective fps can differ from the requested one.
# Closed set on purpose: a client reading a manifest at 2am must be able to map the value to a
# sentence, not guess. "profile-default" means nothing overrode the request.
FPS_REASONS = ("brief-window-coverage", "source-limit", "profile-default")

#: Accepted ``run(timebase=…)`` values. ``"auto"`` (the default) resolves to MEDIA time
#: for recorded sources (files) and WALL/monotonic time for live devices/streams.
TIMEBASE_CHOICES = ("auto", "media", "wall")


@dataclass(frozen=True)
class EffectiveFps:
    """What capture rate this run ACTUALLY opened with, versus what the caller asked for.

    Raising fps to cover a brief-event plan is not necessarily a bug; doing it INVISIBLY is. The run
    therefore retains the request, the effective value, and which of :data:`FPS_REASONS` explains the
    difference — surfaced on ``stats.effective_config`` and through the one-shot ``effective_config``
    health event. No health WARNING is emitted just for an override (an override may be correct); the
    loud channel stays reserved for a requirement that is actually unmet (``capture_plan_unmet``).
    """

    requested_fps: float
    effective_fps: float
    reason: str

    @property
    def overridden(self) -> bool:
        return self.effective_fps != self.requested_fps

    def as_dict(self) -> dict:
        return {
            "requested_fps": self.requested_fps,
            "effective_fps": self.effective_fps,
            "reason": self.reason,
            "overridden": self.overridden,
        }


@dataclass(frozen=True)
class EffectiveTimebase:
    """Which clock this run stamps frame ``t`` (and therefore ``incident.ts``) on.

    ``requested`` is the caller's ``run(timebase=…)`` (default ``"auto"``); ``resolved`` is the
    concrete ``"media"``/``"wall"`` handed to the capture source; ``reason`` records WHY —
    ``"recorded-source"`` (a file: media time, so an incident timestamp is meaningful against the
    media), ``"live-source"`` (a camera/mic/stream: wall time, so it shares the agent's deadline
    clock), ``"explicit-override"`` (the caller chose), or ``"deterministic-stream"`` (a non-``str``
    source feeds its own explicit ``t`` and is never re-stamped)."""

    requested: str
    resolved: str
    reason: str

    def as_dict(self) -> dict:
        return {"requested": self.requested, "resolved": self.resolved, "reason": self.reason}


@dataclass(frozen=True)
class EffectiveConfig:
    """The recorded effective configuration of one ``run()`` — the answer to "what did it ACTUALLY
    do?" without re-deriving it from the source.

    Additive and runtime-local: it is exposed on :class:`RuntimeStats` (``stats.effective_config``),
    best-effort on the core (``core.effective_config``, so a compiled watch/manifest writer can read
    it off the same object it already holds) and emitted once through ``on_health`` as an
    informational ``effective_config`` event. The structured trace record can serialize :meth:`as_dict`
    verbatim when it lands — no runtime change needed then."""

    fps: EffectiveFps
    video_timebase: EffectiveTimebase
    audio_timebase: EffectiveTimebase | None = None
    # The resolved admission posture this run opened with: whether a SourceProfile-declared coast
    # floor is bound into the runtime (Settings.edge.coast_enabled). False under a strict_admission
    # ExecutionPolicy — a signal-only run states its posture rather than leaving it to be inferred.
    coast_enabled: bool = True

    def as_dict(self) -> dict:
        d = {"fps": self.fps.as_dict(), "video_timebase": self.video_timebase.as_dict()}
        if self.audio_timebase is not None:
            d["audio_timebase"] = self.audio_timebase.as_dict()
        d["coast_enabled"] = self.coast_enabled
        return d


@dataclass
class RuntimeStats:
    """Mutable runtime counters for health dashboards and tests."""

    frames: int = 0
    frames_dropped: int = 0  # frames the bounded capture queue discarded (slow consumer)
    escalations: int = 0
    judge_calls: int = 0
    fires: int = 0
    asks: int = 0
    incidents: int = 0
    sink_errors: int = 0
    audio_chunks: int = 0  # audio chunks pumped into the AudioWindow (fused/acoustic perception)
    audio_wakes: int = 0  # audio onsets that admitted a frame the visual gate would have skipped
    # Admission partition (each admitted frame lands in EXACTLY one bucket, so the totals sum cleanly):
    #   escalations == signal_admits + fixed_admits + coast_admits + audio_wakes
    # - signal_admits: the tuned motion/salience gate escalated (decision.escalate) — the real signal.
    # - coast_admits: a fixed-cadence coast floor forced the judge on a frame the gate would have
    #   skipped (coast_admit and NOT escalate). A strict_admission ExecutionPolicy (coast_enabled=False)
    #   strips a SourceProfile coast at resolve time, so this stays 0 on static content there.
    # - audio_wakes: an audio onset admitted a frame neither the gate nor the coast would (above).
    signal_admits: int = 0  # tuned-signal (motion/salience) escalations
    fixed_admits: int = 0  # fixed-only per-Watch cadence admits (never motion-labelled)
    coast_admits: int = 0  # fixed-cadence coast admits (no tuned signal, no audio)
    boundaries: int = 0  # edge-detector boundaries (rep peaks / onsets) fed to perceive_boundary
    utterances: int = 0  # complete spoken utterances endpointed and handed to on_utterance
    # Frames evicted from the incident snippet's pre-roll ring by its BYTE budget (not by the count
    # bound, which is the ring's normal cadence). Each one also emits a `preroll_capped` health event
    # carrying the reason — ADR-001 §9: a bound-driven drop is never a silent counter bump.
    preroll_frames_dropped: int = 0
    # judge_dispatch="async" accounting. The scheduler is single-flight + one latest-pending, so under
    # overload an admitted frame may be replaced before its judge launches (superseded) or discarded
    # because goals changed after admission (revision_dropped). The honest budget invariant is
    # Absent a source discontinuity, ``judge_calls + judge_superseded +
    # judge_revision_dropped == admitted escalations`` once drained. A discontinuity additionally
    # reports active/pending drops below; an active drop already spent a judge call, while a pending
    # drop did not, so the split is retained rather than hiding that distinction.
    judge_superseded: int = 0  # pending judge candidate replaced by a newer admitted frame
    judge_revision_dropped: int = 0  # pending candidate dropped: goals.revision moved after admission
    judge_deadline_dropped: int = 0  # pending candidate aged past its absolute decision deadline
    judge_deadline_expired: int = 0  # launched provider batch exceeded its remaining decision budget
    judge_source_discontinuity_dropped: int = 0
    judge_source_discontinuity_active_dropped: int = 0
    judge_source_discontinuity_pending_dropped: int = 0
    # The EFFECTIVE configuration this run actually opened with (never only the
    # requested one), populated once at start. `None` until run() has resolved it.
    effective_config: "EffectiveConfig | None" = None
    # The two independent clocks of the most recent accepted envelope.  Either remains ``None`` when
    # the source did not carry it; the runtime never back-fills source time from cadence or elapsed time.
    last_media_t: float | None = None
    last_wall_t: float | None = None
    source_observed: int = 0
    source_excluded: int = 0
    source_discontinuities: int = 0
    source_missing_sequences: int = 0
    # Audio ingress ledger (AUDIO-SUPPRESSED-NO-RECEIPT). Every unit the source offered leaves by
    # exactly one door: accepted (`audio_chunks`), declined because the caller had the runtime
    # muted/paused (`audio_suppressed`), or refused as a malformed carrier (`audio_rejected`, which
    # also receipts an `audio_unit_rejected` health event). Suppression had no term at all — units
    # were `continue`d with no counter, no event, and `source_coverage.excluded` still reading 0 —
    # so an offered-vs-accepted deficit was unattributable.
    audio_suppressed: int = 0
    audio_rejected: int = 0
    # Per-Watch admission accounting. A fixed opportunity is a fixed-cadence condition whose interval
    # elapsed; every such opportunity admits exactly that goal in the in-process topology.
    fixed_opportunities: int = 0
    goal_admissions: int = 0
    admission_by_goal: dict[str, int] = field(default_factory=dict)
    admission_by_cause: dict[str, int] = field(default_factory=dict)
    perceive_dispatches: int = 0


@dataclass(frozen=True)
class RuntimeHealthEvent:
    """Structured runtime health event emitted through ``on_health``."""

    kind: str
    ts: float | None = None
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class _OwnerCommand:
    """One immutable item in the runtime cognition owner's total order."""

    ordinal: int
    kind: str
    payload: Any


class _OwnerCommandQueue:
    """Bounded synchronous owner queue with explicit overflow semantics.

    Publication authors one global monotonic ``ordinal``. The queue never
    silently evicts an older command: reaching ``maxsize`` increments the
    observable ``overflow_count`` and raises :class:`OverflowError` before an
    ordinal is consumed. The one event-loop owner drains the ordered batch
    synchronously before any subscriber callback can run.
    """

    def __init__(self, *, maxsize: int) -> None:
        if type(maxsize) is not int or maxsize < 1:
            raise ValueError("owner queue maxsize must be a positive integer")
        self.maxsize = maxsize
        self.overflow_count = 0
        self._next_ordinal = 1
        self._items: deque[_OwnerCommand] = deque()

    def publish(self, kind: str, payload: Any) -> _OwnerCommand:
        """Synchronously append one item or fail loudly without mutation."""
        if type(kind) is not str or not kind:
            raise ValueError("owner command kind must be non-empty text")
        if len(self._items) >= self.maxsize:
            self.overflow_count += 1
            raise OverflowError("runtime owner command queue overflow")
        row = _OwnerCommand(
            ordinal=self._next_ordinal,
            kind=kind,
            payload=payload,
        )
        self._next_ordinal += 1
        self._items.append(row)
        return row

    def drain(self) -> tuple[_OwnerCommand, ...]:
        """Transfer the current ordered batch to the sole runtime owner."""
        rows = tuple(self._items)
        self._items.clear()
        return rows


@dataclass(frozen=True)
class _MaterializedItem:
    action: str
    fire: Any
    incident: Any = None
    trace: tuple[str, float, str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class _MaterializedReduction:
    at: float
    items: tuple[_MaterializedItem, ...]


def _materialize_reduction(
    fires: Any,
    *,
    at: float,
    gate_source: str,
    snippet_for_fire: Any,
    watchplan_id: str | None,
    label_store: Any,
    disposition_log: Any,
    verdict_ref: Any,
    from_boundary: bool,
) -> _MaterializedReduction:
    """Synchronously freeze every output before a subscriber can run."""
    items = []
    for fire in fires or ():
        action = getattr(fire, "action", None)
        judgment = getattr(fire, "judgment", None)
        source_media_ts = getattr(
            getattr(judgment, "timing", None),
            "source_media_ts",
            None,
        )
        event_at = (
            float(source_media_ts)
            if isinstance(source_media_ts, (int, float))
            and not isinstance(source_media_ts, bool)
            else at
        )
        if action == ASK:
            if disposition_log is not None:
                _log_disposition(
                    disposition_log,
                    incident_id="",
                    fire=fire,
                    disposition="ask",
                    ts=event_at,
                )
            items.append(_MaterializedItem(action=ASK, fire=fire))
            continue
        if action != FIRE:
            continue
        incident = make_incident(
            fire,
            ts=event_at,
            snippet=snippet_for_fire(fire),
            watchplan_id=watchplan_id,
        )
        fused_bundle = getattr(fire, "_fused_evidence_bundle", None)
        if fused_bundle is not None:
            if (
                fused_bundle.judgment_event_id
                != incident.judgment_event_id
                or fused_bundle.evidence_seq != incident.evidence_seq
            ):
                raise RuntimeError(
                    "fused evidence identity diverged from the delivered judgment"
                )
            incident._fused_evidence_bundle = fused_bundle
        if label_store is not None:
            incident.bind_labeler(
                label_store.append,
                verdict_ref=verdict_ref(getattr(fire, "verdict", None)),
                goal_id=getattr(fire, "goal_id", None),
            )
        if disposition_log is not None:
            _log_disposition(
                disposition_log,
                incident_id=incident.id,
                fire=fire,
                disposition="fire",
                ts=at,
            )
        judgment_trace_id = f"{gate_source}@{at}"
        trace = (
            judgment_trace_id,
            at,
            "incident.created",
            {
                "goal_id": incident.concern_id,
                "incident_id": incident.id,
                "from_boundary": from_boundary,
                "entity_id": incident.entity_id,
                "judgment_event_id": incident.judgment_event_id,
                "evidence_origin": incident.evidence_origin,
            },
        )
        items.append(
            _MaterializedItem(
                action=FIRE,
                fire=fire,
                incident=incident,
                trace=trace,
            )
        )
    return _MaterializedReduction(at=at, items=tuple(items))


async def _deliver_materialized(
    materialized: _MaterializedReduction,
    *,
    core: Any,
    on_health: Any,
    on_ask: Any,
    on_incident: Any,
    sink_errors: str,
    stats: Any,
    coverage: Any,
) -> tuple[Any, ...]:
    """Deliver already-frozen outputs without authoring cognition state."""
    delivered = []
    for item in materialized.items:
        if item.action == ASK:
            if on_health is not None:
                await _call_sink(
                    on_health,
                    RuntimeHealthEvent(
                        "ask",
                        ts=materialized.at,
                        data={"goal_id": item.fire.goal_id},
                    ),
                )
            if on_ask is not None:
                await _call_sink_with_policy(
                    on_ask,
                    item.fire,
                    errors=sink_errors,
                    stats=stats,
                    on_health=on_health,
                    health_kind="ask_sink_error",
                )
            continue
        incident = item.incident
        assert incident is not None
        if item.trace is not None:
            emit_trace = getattr(core, "_emit_trace", None)
            if getattr(core, "on_trace", None) is not None and callable(
                emit_trace
            ):
                trace_id, trace_t, stage, data = item.trace
                await emit_trace(trace_id, trace_t, stage, **data)
        if on_health is not None:
            await _call_sink(
                on_health,
                RuntimeHealthEvent(
                    "incident",
                    ts=incident.ts,
                    data={
                        "incident_id": incident.id,
                        "goal_id": incident.concern_id,
                        "entity_id": incident.entity_id,
                    },
                ),
            )
        if on_incident is not None:
            await _call_sink_with_policy(
                on_incident,
                incident,
                errors=sink_errors,
                stats=stats,
                on_health=on_health,
                health_kind="incident_sink_error",
            )
        coverage.incidents += 1
        delivered.append(incident)
    return tuple(delivered)


class _LegacyJudgeScheduler:
    """Off-loop judge dispatch (``run(judge_dispatch="async")``): the capture loop never awaits a
    network judgment.

    The inline path (``async for … perceive_judged(…)`` at the admit site) makes perception wait on
    cognition — one slow VLM turn stalls capture, the gate, the absence checks, and every subscriber
    (the 47s-stall class of bugs). This scheduler moves the judgment onto an ``asyncio`` task with a
    strict budget shape:

    - **single-flight** — at most ONE judgment in flight per source (``run()`` drives one source, so
      one scheduler instance; completion order == submission order, preserving on_incident ordering);
    - **one latest-pending** — while a judgment is in flight, newly admitted frames REPLACE the queued
      candidate (latest wins, ``stats.judge_superseded``): under overload we judge the freshest
      evidence, never build a backlog;
    - **admission-revision pinning** — the goal-registry revision is captured when the frame is
      ADMITTED and travels with it into ``perceive_judged(expected_goals_revision=…)``. A pending
      frame admitted under revision N is never launched once goals moved to N+1
      (``stats.judge_revision_dropped``) — old evidence is not reinterpreted under a newly authored
      watch plan; the in-flight call aborts itself the same way (see ``Percept.perceive_judged``);
    - **fail-loud** — a judge exception is held in ``_fatal`` and re-raised by the next ``drain()``
      on the capture loop, preserving the existing ``runtime_error`` health path; nothing new is
      launched after a failure.

    ``stats.judge_calls`` counts at LAUNCH time (a superseded/dropped candidate never spent a call).
    Results carry the ADMISSION ``t`` so ``incident.ts`` stays the admitting frame's timestamp.
    Lives in this module on purpose: a parity test scans ``inspect.getsource(run)``'s module.

    ADR-001 §9: the in-flight slot (``_active``) and the pending slot (``_pending``) are
    BOUNDED BY CONSTRUCTION — each is a single scalar field, not a collection, so J<=1 and P<=1 always
    hold; overload cannot grow either past 1, it can only REPLACE what's there (reject/supersede).
    Every such replacement is TRACED, never silent: ``submit()``/``_launch_pending()`` append a
    ``(kind, data)`` pair to ``_trace`` (itself drained every frame tick alongside ``_results``, so it
    never accumulates past one frame's worth) that ``run()`` turns into a ``RuntimeHealthEvent`` —
    ``"judge_superseded"`` (latest-wins pending replacement) or ``"judge_revision_dropped"`` (stale
    pending discarded on a goals-revision change) — each carrying an explicit ``reason``.

    Degrade-loud + backpressure, INSIDE this scheduler (not an inline guard: with
    ``judge_dispatch="async"`` the admit site never awaits ``perceive_judged`` directly, so a guard
    planted there is simply never on the path a slow/wedged judge actually takes). Two more
    bound-driven, edge-triggered, TRACED conditions on top of the two above:

    - **stalled active judgment** (a hang) — the in-flight task has been outstanding, from ITS OWN
      launch instant, past ``stall_after_s`` (``_mc_now``-measured: the core's clock when live, the
      admission ``t`` timeline otherwise — same convention as ``_mc_admit``/``_mc_resolve``, never a
      bare wall-clock call). Checked on every ``submit()`` (the scheduler has no timer of its own —
      the frame cadence IS its polling clock), latched so it fires exactly ONCE per stalled task
      (``"judge_scheduler_stalled"``), not once per frame while stalled.
    - **sustained backpressure** (slower than the frame rate) — ``submit()`` superseding the pending
      slot ``backpressure_after`` times IN A ROW with no intervening successful completion means the
      active judgment cannot keep up with admission — a single supersede is normal load-shedding, a
      RUN of them is the judge falling behind. Latched the same way (``"judge_scheduler_backpressure"``),
      carrying the streak length and how long the active judgment has been outstanding.

    Neither condition changes what the scheduler DOES — it is already degrading gracefully (bounded
    single-flight + latest-wins is the graceful degradation); this only makes that degradation LOUD
    instead of a silently-climbing counter nobody is watching. Recovery is symmetric and explicit: the
    next time ``_on_done`` sees the active task complete WITHOUT raising, any latched condition clears
    and, only if something was actually latched, one ``"judge_scheduler_recovered"`` event reports it
    — health never stays latched red after the transient is gone (a fresh launch starts unlatched).
    A judge that RAISES is a separate, pre-existing path: caught inside ``Percept._safe_timed_judge``
    (``_JudgeDegraded`` -> ``"judge_degraded"``/``"judge_persistent_failure"``) before it can
    ever reach this scheduler's task, or — for a failure outside that boundary — the scheduler's own
    fail-loud ``_fatal``/``drain()`` re-raise (``"runtime_error"``). Both stay exactly as they were.
    """

    def __init__(
        self,
        core: Any,
        source: str,
        stats: RuntimeStats | None,
        *,
        perceive: Any = None,
        include_payload: bool = False,
        metrics_collector: Any = None,
        stall_after_s: float = 20.0,
        backpressure_after: int = 3,
        decision_deadline_s: float | None = None,
    ) -> None:
        self._core = core
        self._source = source
        self._perceive = perceive or core.perceive_judged
        self._include_payload = include_payload
        self._stats = stats
        # Optional percept.harness.metrics.MetricsCollector — when given, every admission
        # into this scheduler's active/pending slot is tracked via record_admitted/
        # record_admission_resolved (queue depth + oldest-pending age), keyed by the admission `t`
        # (unique per frame within one run). Purely additive telemetry: the scheduler's own admit/
        # launch/drop decisions are unchanged whether or not a collector is wired.
        self._metrics = metrics_collector
        self._active: "asyncio.Task | None" = None
        self._abandoned_dispatches: set[asyncio.Task] = set()
        self._abandoned_terminal_sources: dict[
            asyncio.Task, tuple[tuple[Any, ...], float | None]
        ] = {}
        self._active_t: float | None = None  # admission t of the in-flight judgment (rides the result)
        self._active_payload: Any = None
        self._active_terminal: _DecisionTerminal | None = None
        self._active_terminal_sources: tuple[Any, ...] = ()
        # (t, judge_frame, admission_revision, admitted_goal_ids,
        # fused_bundles, admitted_wall). The ids are frozen at the same instant
        # as the revision so queued evidence cannot broaden into unrelated
        # Watches later. ``admitted_wall`` makes the decision budget absolute.
        self._pending: (
            tuple[
                float,
                Any,
                Any,
                tuple[str, ...] | None,
                Any,
                float,
            ]
            | None
        ) = None
        self._results: deque = deque()  # completed (t, [FireEvent]) awaiting drain()
        # ADR-001 §9: TRACED bound-driven drops (superseded pending / revision-dropped pending) — a
        # (kind, data) pair per drop, drained every frame tick by run() alongside drain(), so this list
        # is bounded by construction (never holds more than the current tick's drops).
        self._trace: list[tuple[str, dict]] = []
        self._fatal: BaseException | None = None
        self._closing = False
        # Degrade-loud + backpressure (see class docstring). Tunable knobs, defaulted
        # generously so a normal (non-overloaded, non-hung) run never trips either latch.
        self._stall_after_s = stall_after_s
        self._backpressure_after = backpressure_after
        self._active_launch_wall: float | None = None  # _mc_now() at the CURRENT active task's launch
        self._stall_reported = False        # latched: "judge_scheduler_stalled" already fired for it
        self._consec_supersede = 0          # supersessions in a row with no intervening completion
        self._backpressure_reported = False  # latched: "judge_scheduler_backpressure" already fired
        self._perceive_accepts_goal_ids = _accepts_keyword(self._perceive, "goal_ids")
        self._perceive_accepts_source_generation = _accepts_keyword(
            self._perceive, "expected_source_generation"
        )
        self._perceive_accepts_fused_bundles = _accepts_keyword(
            self._perceive, "_fused_bundles_by_goal"
        )
        self._perceive_accepts_decision_terminal = _accepts_keyword(
            self._perceive, "_decision_terminal"
        )
        self._active_filter_goal_ids: tuple[str, ...] | None = None
        self._active_invalidated = False
        self._active_invalidation_reason = "source_discontinuity"
        if decision_deadline_s is not None and (
            isinstance(decision_deadline_s, bool)
            or not isinstance(decision_deadline_s, (int, float))
            or not math.isfinite(float(decision_deadline_s))
            or float(decision_deadline_s) <= 0.0
        ):
            raise ValueError(
                "decision_deadline_s must be a positive finite number or None"
            )
        self._decision_deadline_s = (
            None
            if decision_deadline_s is None
            else float(decision_deadline_s)
        )
        self._active_admitted_wall: float | None = None

    def _mc_now(self, fallback: float) -> float:
        """"Now" for a MetricsCollector stamp: the core's own clock when available (real elapsed time
        between admit and resolve), else the admission ``t`` itself (offline/replay streams without a
        live clock) — never a bare wall-clock call (this module has no clock of its own)."""
        clock = getattr(self._core, "clock", None)
        return clock() if callable(clock) else fallback

    def _mc_admit(self, t: float) -> None:
        if self._metrics is not None:
            self._metrics.record_admitted(t, at=self._mc_now(t))

    def _mc_resolve(self, t: float) -> None:
        if self._metrics is not None:
            self._metrics.record_admission_resolved(t, at=self._mc_now(t))

    # 0/1 by construction — surfaced through the escalation health event for live observability.
    @property
    def active(self) -> int:
        return int(
            self._active is not None or bool(self._abandoned_dispatches)
        )

    @property
    def pending(self) -> int:
        return 0 if self._pending is None else 1

    @property
    def can_accept(self) -> bool:
        """Whether one synchronous owner offer can be installed right now."""
        return self._fatal is None and not self._closing

    def submit(
        self,
        t: float,
        judge_frame: Any,
        goal_ids: tuple[str, ...] | None = None,
        *,
        fused_bundles: Any = None,
    ) -> bool:
        """Admit a frame for judgment. Never blocks; never launches after a failure or during close."""
        if self._fatal is not None or self._closing:
            return False  # a rejected offer spends/reserves nothing
        revision = getattr(self._core.goals, "revision", None)  # ADMISSION-time, never launch-time
        admitted_wall = self._mc_now(t)
        if self._active is None and not self._abandoned_dispatches:
            self._mc_admit(t)  # launching right away — admitted into the funnel from this instant
            self._launch(
                t,
                judge_frame,
                revision,
                goal_ids,
                fused_bundles,
                admitted_wall=admitted_wall,
            )
            return True
        # The scheduler has no timer of its own — every submit() (the frame cadence) is the one
        # chance to notice the CURRENT active judgment has been outstanding too long. Checked before
        # the supersede below so a stalled-AND-overloaded tick reports both, in the order they became
        # true.
        self._check_stall(t)
        if self._pending is not None:
            dropped_t = self._pending[0]
            dropped_job = self._pending[1]
            dropped_token = getattr(dropped_job, "token", None)
            drop_data = {
                "source": self._source,
                "reason": "superseded_by_newer_admission",
                "dropped_t": dropped_t,
                "admitted_t": t,
            }
            if dropped_token is not None:
                drop_data.update(
                    {
                        "admission_id": getattr(
                            dropped_token, "admission_id", None
                        ),
                        "evidence_seq": getattr(
                            dropped_token, "evidence_seq", None
                        ),
                        "goal_id": getattr(dropped_token, "goal_id", None),
                        "slot": "pending",
                    }
                )
                record_drop = getattr(
                    self._core, "record_fused_admission_drop", None
                )
                if callable(record_drop):
                    record_drop(
                        dropped_job,
                        reason="superseded_by_newer_admission",
                        slot="pending",
                    )
            if self._stats is not None:
                self._stats.judge_superseded += 1  # latest wins: the older queued candidate is dropped
            self._trace.append((
                "judge_superseded",
                drop_data,
            ))
            self._mc_resolve(dropped_t)  # leaves the collector's pending map — it stopped waiting here
            funnel = getattr(self._core, "funnel", None)
            if funnel is not None:
                funnel.dropped(reason="superseded_by_newer_admission", camera=self._source)
            # A RUN of supersessions with no intervening completion is the judge falling behind
            # the frame rate (sustained overwrite) — distinct from, and louder than, any one supersede.
            self._consec_supersede += 1
            self._check_backpressure(t)
        self._pending = (
            t,
            judge_frame,
            revision,
            goal_ids,
            fused_bundles,
            admitted_wall,
        )
        self._mc_admit(t)  # queued behind the active judgment — pending in the funnel from this instant
        return True

    def invalidate_source_continuity(self, reasons: tuple[str, ...]) -> None:
        """Prevent pre-discontinuity judgments from mutating post-discontinuity state."""
        reason = ",".join(reasons) or "source_discontinuity"
        # A task can complete between ticks; its result is still pre-boundary work and must not
        # survive in the delivery queue merely because the provider await already returned.
        if self._active is not None and not self._active.done():
            self._active_invalidated = True
            self._active_invalidation_reason = reason
            if self._active_terminal is not None:
                self._active_terminal.expire("source_discontinuity")
            self._active.cancel()
        if self._pending is not None:
            dropped_t = self._pending[0]
            self._mc_resolve(dropped_t)
            self._pending = None
            if self._stats is not None:
                self._stats.judge_source_discontinuity_dropped += 1
                self._stats.judge_source_discontinuity_pending_dropped += 1
            self._trace.append((
                "judge_source_discontinuity_dropped",
                {
                    "source": self._source,
                    "reason": reason,
                    "dropped_t": dropped_t,
                    "slot": "pending",
                },
            ))
            funnel = getattr(self._core, "funnel", None)
            if funnel is not None:
                funnel.dropped(reason="source_discontinuity", camera=self._source)

    def _check_stall(self, now_t: float) -> None:
        if self._stall_reported or self._active is None or self._active_launch_wall is None:
            return
        elapsed = self._mc_now(now_t) - self._active_launch_wall
        if elapsed < self._stall_after_s:
            return
        self._stall_reported = True
        self._trace.append((
            "judge_scheduler_stalled",
            {
                "source": self._source, "reason": "active_judgment_exceeded_stall_threshold",
                "active_t": self._active_t, "elapsed_s": elapsed, "stall_after_s": self._stall_after_s,
            },
        ))

    def _check_backpressure(self, now_t: float) -> None:
        if self._backpressure_reported or self._consec_supersede < self._backpressure_after:
            return
        self._backpressure_reported = True
        elapsed = (
            self._mc_now(now_t) - self._active_launch_wall
            if self._active_launch_wall is not None else None
        )
        self._trace.append((
            "judge_scheduler_backpressure",
            {
                "source": self._source, "reason": "sustained_pending_overwrite",
                "consecutive_supersessions": self._consec_supersede,
                "active_t": self._active_t, "active_elapsed_s": elapsed,
            },
        ))

    def _report_recovery_if_latched(self) -> None:
        """The symmetric half of ``_check_stall``/``_check_backpressure`` — clear whichever
        latch(es) are set and, only when at least one WAS set, trace a single
        ``"judge_scheduler_recovered"`` event naming which condition(s) cleared. A clean run (nothing
        ever latched) stays silent here — recovery is only news when a degradation was reported."""
        recovered = []
        if self._stall_reported:
            recovered.append("stalled")
        if self._backpressure_reported:
            recovered.append("backpressure")
        self._stall_reported = False
        self._backpressure_reported = False
        if not recovered:
            return
        self._trace.append((
            "judge_scheduler_recovered",
            {"source": self._source, "recovered_from": recovered, "resolved_t": self._active_t},
        ))

    def _launch(
        self,
        t: float,
        frame: Any,
        revision: Any,
        goal_ids: tuple[str, ...] | None,
        fused_bundles: Any,
        *,
        admitted_wall: float,
    ) -> None:
        if self._stats is not None:
            self._stats.judge_calls += 1  # counted at launch — a superseded candidate spent nothing
            self._stats.perceive_dispatches += 1
        self._active_t = t
        if self._include_payload:
            self._active_payload = frame
        self._active_launch_wall = self._mc_now(t)  # this task's own clock, for stall detection
        self._active_admitted_wall = admitted_wall
        self._stall_reported = False  # a fresh launch is unlatched — the stall window resets
        kwargs = {"t": t, "expected_goals_revision": revision}
        if self._perceive_accepts_source_generation:
            generation_for = getattr(self._core, "source_continuity_generation", None)
            if callable(generation_for):
                kwargs["expected_source_generation"] = generation_for(self._source)
        if goal_ids is not None and self._perceive_accepts_goal_ids:
            kwargs["goal_ids"] = goal_ids
        if fused_bundles is not None:
            if not self._perceive_accepts_fused_bundles:
                raise ValueError("fused dispatch requires a core accepting frozen fused bundles")
            kwargs["_fused_bundles_by_goal"] = fused_bundles
        terminal = _DecisionTerminal()
        self._active_terminal = terminal
        if self._perceive_accepts_decision_terminal:
            kwargs["_decision_terminal"] = terminal
        self._active_filter_goal_ids = (
            goal_ids
            if goal_ids is not None and not self._perceive_accepts_goal_ids
            else None
        )
        self._active_terminal_sources = self._terminal_sources_for(
            frame,
            goal_ids,
        )
        self._active = asyncio.create_task(
            self._dispatch_active(
                frame,
                kwargs,
                t=t,
                admitted_wall=admitted_wall,
                terminal=terminal,
            )
        )
        self._active.add_done_callback(self._on_done)

    def _terminal_sources_for(
        self,
        frame: Any,
        goal_ids: tuple[str, ...] | None,
    ) -> tuple[Any, ...]:
        roots: list[Any] = []
        job_adapter = getattr(frame, "adapter", None)
        if job_adapter is not None:
            roots.append(job_adapter)
        binding_for = getattr(self._core, "_audio_judge_binding_for", None)
        goal_for = getattr(getattr(self._core, "goals", None), "get", None)
        if callable(binding_for) and callable(goal_for):
            for goal_id in goal_ids or ():
                goal = goal_for(goal_id)
                if goal is None:
                    continue
                try:
                    binding = binding_for(goal)
                except Exception:
                    continue
                adapter = getattr(binding, "adapter", None)
                if adapter is not None:
                    roots.append(adapter)
        root_vision = getattr(self._core, "vision", None)
        if root_vision is not None:
            roots.append(root_vision)
        found: list[Any] = []
        seen: set[int] = set()
        queue = list(roots)
        while queue:
            current = queue.pop(0)
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            if callable(getattr(current, "take_terminal_error", None)):
                found.append(current)
                continue
            for name in ("_adapter", "inner", "adapter"):
                nested = getattr(current, name, None)
                if nested is not None:
                    queue.append(nested)
        return tuple(found)

    def _take_terminal_error(
        self,
        sources: tuple[Any, ...],
        *,
        active_t: float | None,
    ) -> BaseException | None:
        for source in sources:
            take_terminal = getattr(source, "take_terminal_error", None)
            error = take_terminal() if callable(take_terminal) else None
            if error is None:
                continue
            self._trace.append(
                (
                    "judge_terminal_failure",
                    {
                        "source": self._source,
                        "reason": "vision_adapter_terminal_outcome",
                        "active_t": active_t,
                        "exception_type": type(error).__name__,
                    },
                )
            )
            return error
        return None

    def _on_abandoned_done(self, task: asyncio.Task) -> None:
        late_failure: BaseException | None = None
        if not task.cancelled():
            try:
                late_failure = task.exception()
            except asyncio.CancelledError:
                late_failure = None
        sources, active_t = self._abandoned_terminal_sources.pop(
            task,
            ((), None),
        )
        terminal_error = self._take_terminal_error(
            sources,
            active_t=active_t,
        )
        if terminal_error is not None:
            self._fatal = terminal_error
            if self._pending is not None:
                self._mc_resolve(self._pending[0])
            self._pending = None
        elif late_failure is not None:
            self._fatal = late_failure
            if self._pending is not None:
                self._mc_resolve(self._pending[0])
            self._pending = None
            self._trace.append(
                (
                    "judge_abandoned_failure",
                    {
                        "source": self._source,
                        "reason": "late_dispatch_failure",
                        "active_t": active_t,
                        "exception_type": type(late_failure).__name__,
                    },
                )
            )
        self._abandoned_dispatches.discard(task)
        if self._active is None and self._fatal is None:
            self._launch_pending()

    def _track_abandoned(
        self,
        task: asyncio.Task,
        *,
        active_t: float | None,
        terminal_sources: tuple[Any, ...],
    ) -> None:
        if task in self._abandoned_dispatches:
            return
        self._abandoned_dispatches.add(task)
        self._abandoned_terminal_sources[task] = (
            terminal_sources,
            active_t,
        )
        task.add_done_callback(self._on_abandoned_done)

    async def _dispatch_active(
        self,
        frame: Any,
        kwargs: dict[str, Any],
        *,
        t: float,
        admitted_wall: float,
        terminal: _DecisionTerminal,
    ) -> Any:
        dispatch = asyncio.create_task(
            self._perceive(frame, self._source, **kwargs)
        )
        if self._decision_deadline_s is None:
            try:
                return await dispatch
            except asyncio.CancelledError:
                terminal.expire("scheduler_cancelled")
                dispatch.cancel()
                self._track_abandoned(
                    dispatch,
                    active_t=t,
                    terminal_sources=self._active_terminal_sources,
                )
                raise
        age_s = max(0.0, self._mc_now(t) - admitted_wall)
        remaining_s = self._decision_deadline_s - age_s
        if remaining_s > 0.0:
            try:
                done, _ = await asyncio.wait(
                    {dispatch},
                    timeout=remaining_s,
                )
            except asyncio.CancelledError:
                terminal.expire("scheduler_cancelled")
                dispatch.cancel()
                self._track_abandoned(
                    dispatch,
                    active_t=t,
                    terminal_sources=self._active_terminal_sources,
                )
                raise
            if dispatch in done:
                return dispatch.result()
        terminal.expire("decision_deadline_expired")
        dispatch.cancel()
        self._track_abandoned(
            dispatch,
            active_t=t,
            terminal_sources=self._active_terminal_sources,
        )
        elapsed_s = max(0.0, self._mc_now(t) - admitted_wall)
        if self._stats is not None:
            self._stats.judge_deadline_expired += 1
        self._trace.append(
            (
                "judge_deadline_expired",
                {
                    "source": self._source,
                    "reason": "active_decision_deadline_expired",
                    "active_t": t,
                    "age_s": elapsed_s,
                    "decision_deadline_s": self._decision_deadline_s,
                    "slot": "active",
                },
            )
        )
        return _LEGACY_DEADLINE_EXPIRED

    def _launch_pending(self) -> None:
        if self._abandoned_dispatches:
            return
        if self._pending is None:
            return
        if self._fatal is not None or self._closing:
            self._mc_resolve(self._pending[0])  # leaving the funnel un-launched, not a real judgment
            self._pending = None
            return
        (
            t,
            frame,
            revision,
            goal_ids,
            fused_bundles,
            admitted_wall,
        ) = self._pending
        self._pending = None
        current = getattr(self._core.goals, "revision", None)
        if revision != current:
            # Admitted under revision N, goals now at N+1: old evidence must not be judged as-if
            # under the new plan. Drop it and wait for the next frame (which admits under N+1).
            if self._stats is not None:
                self._stats.judge_revision_dropped += 1
            self._trace.append((
                "judge_revision_dropped",
                {
                    "source": self._source, "reason": "goals_revision_changed",
                    "dropped_t": t, "admitted_revision": revision, "current_revision": current,
                },
            ))
            self._mc_resolve(t)
            funnel = getattr(self._core, "funnel", None)
            if funnel is not None:
                funnel.dropped(reason="goals_revision_changed", camera=self._source)
            return
        if self._decision_deadline_s is not None:
            age_s = max(0.0, self._mc_now(t) - admitted_wall)
            if age_s >= self._decision_deadline_s:
                if self._stats is not None:
                    self._stats.judge_deadline_dropped += 1
                self._trace.append(
                    (
                        "judge_deadline_dropped",
                        {
                            "source": self._source,
                            "reason": "pending_admission_deadline_expired",
                            "dropped_t": t,
                            "age_s": age_s,
                            "decision_deadline_s": self._decision_deadline_s,
                            "slot": "pending",
                        },
                    )
                )
                self._mc_resolve(t)
                funnel = getattr(self._core, "funnel", None)
                if funnel is not None:
                    funnel.dropped(
                        reason="decision_deadline_expired",
                        camera=self._source,
                    )
                return
        self._launch(
            t,
            frame,
            revision,
            goal_ids,
            fused_bundles,
            admitted_wall=admitted_wall,
        )

    def _on_done(self, task: "asyncio.Task") -> None:
        # ORDERED checks — cancellation FIRST (close(drain=False) must not become a fatal error),
        # then failure (fatal: never launch pending after a failure), then result + pending launch.
        if task.cancelled():
            self._mc_resolve(self._active_t)  # no longer pending, whether or not it truly judged
            invalidated = self._active_invalidated
            if invalidated:
                if self._stats is not None:
                    self._stats.judge_source_discontinuity_dropped += 1
                    self._stats.judge_source_discontinuity_active_dropped += 1
                self._trace.append((
                    "judge_source_discontinuity_dropped",
                    {
                        "source": self._source,
                        "reason": self._active_invalidation_reason,
                        "dropped_t": self._active_t,
                        "slot": "active",
                    },
                ))
                funnel = getattr(self._core, "funnel", None)
                if funnel is not None:
                    funnel.dropped(reason="source_discontinuity", camera=self._source)
            self._active = None
            self._active_payload = None
            self._active_filter_goal_ids = None
            self._active_invalidated = False
            self._active_terminal = None
            self._active_terminal_sources = ()
            self._active_admitted_wall = None
            if not self._closing:
                self._launch_pending()
            return
        exc = task.exception()
        if exc is not None:
            self._fatal = exc
            self._mc_resolve(self._active_t)
            if self._pending is not None:
                self._mc_resolve(self._pending[0])
            self._pending = None
            self._active = None
            self._active_payload = None
            self._active_filter_goal_ids = None
            self._active_invalidated = False
            self._active_terminal = None
            self._active_terminal_sources = ()
            self._active_admitted_wall = None
            return
        self._mc_resolve(self._active_t)
        result = task.result()
        terminal_error = self._take_terminal_error(
            self._active_terminal_sources,
            active_t=self._active_t,
        )
        if terminal_error is not None:
            self._fatal = terminal_error
            if self._pending is not None:
                self._mc_resolve(self._pending[0])
            self._pending = None
            self._active = None
            self._active_payload = None
            self._active_filter_goal_ids = None
            self._active_invalidated = False
            self._active_terminal = None
            self._active_terminal_sources = ()
            self._active_admitted_wall = None
            return
        if result is _LEGACY_DEADLINE_EXPIRED:
            self._active = None
            self._active_payload = None
            self._active_filter_goal_ids = None
            self._active_invalidated = False
            self._active_terminal = None
            self._active_terminal_sources = ()
            self._active_launch_wall = None
            self._active_admitted_wall = None
            self._launch_pending()
            return
        if self._active_filter_goal_ids is not None:
            result = _filter_goal_events(result, self._active_filter_goal_ids)
        if self._active_invalidated:
            if self._stats is not None:
                self._stats.judge_source_discontinuity_dropped += 1
                self._stats.judge_source_discontinuity_active_dropped += 1
            self._trace.append((
                "judge_source_discontinuity_dropped",
                {
                    "source": self._source,
                    "reason": self._active_invalidation_reason,
                    "dropped_t": self._active_t,
                    "slot": "active",
                },
            ))
            funnel = getattr(self._core, "funnel", None)
            if funnel is not None:
                funnel.dropped(reason="source_discontinuity", camera=self._source)
        else:
            self._results.append(
                (self._active_t, result, self._active_payload)
                if self._include_payload else (self._active_t, result)
            )
        self._active_filter_goal_ids = None
        self._active_payload = None
        self._active_invalidated = False
        self._active = None
        self._active_terminal = None
        self._active_terminal_sources = ()
        # A judgment completing WITHOUT raising is the same "reachable again" proof the agent level
        # (_note_judge_ok) uses — clear both latches and, only if one was actually
        # latched (never spam a clean run), report the recovery ONCE, loud, before anything new
        # launches off this completion.
        self._report_recovery_if_latched()
        self._active_launch_wall = None
        self._active_admitted_wall = None
        self._consec_supersede = 0
        if self._closing:  # shutdown in progress: don't start new work
            if self._pending is not None:
                self._mc_resolve(self._pending[0])
            self._pending = None
            return
        self._launch_pending()

    def drain(self) -> list:
        """Pop completed ``(t, fires)`` results; re-raise a held judge failure into the capture loop
        (the existing fail-loud ``runtime_error`` health path catches it there)."""
        if self._fatal is not None:
            raise self._fatal
        out = []
        while self._results:
            out.append(self._results.popleft())
        return out

    def drain_trace(self) -> list[tuple[str, dict]]:
        """Pop TRACED bound-driven drop events (``(\"judge_superseded\"|\"judge_revision_dropped\",
        data)``) — ADR-001 §9: every drop the scheduler makes is reported here, never silent."""
        out = self._trace
        self._trace = []
        return out

    async def flush(self) -> None:
        """Drain to QUIESCENCE: completing the active task may LAUNCH the pending one, so one await
        is not enough — loop until neither exists. A held failure stops the loop (drain re-raises)."""
        while self._fatal is None and (
            self._active is not None
            or self._pending is not None
            or self._abandoned_dispatches
        ):
            task = self._active
            if task is None:
                if self._abandoned_dispatches:
                    await asyncio.wait(
                        tuple(self._abandoned_dispatches),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    await asyncio.sleep(0)
                    continue
                # pending-without-active never persists (submit/_on_done launch immediately) — defensive.
                self._launch_pending()
                continue
            if task.done():
                # Completed, but _on_done may still be QUEUED: awaiting a done task returns WITHOUT
                # yielding, so spinning here would starve the loop and the callback forever (a real
                # deadlock, found by test). One tick: call_soon callbacks run FIFO, so _on_done has
                # run when we resume and the loop test sees the settled state.
                await asyncio.sleep(0)
                continue
            try:
                # _on_done (added at launch, ahead of this await) runs before we resume, so
                # results/pending/active are settled for the next loop test.
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise  # OUR cancellation (run() teardown) must propagate, not be eaten here
                # the TASK was cancelled: _on_done handled it (not fatal) — keep draining
            except BaseException:  # noqa: BLE001 — routed through _on_done into _fatal; drain re-raises
                pass

    async def close(self, *, drain: bool) -> None:
        """Shut down.

        ``drain=True`` (finite files/tests) first reaches quiescence, including the bounded pending
        candidate, then closes submissions. ``drain=False`` (live cancellation) closes first and
        cancels active work so no new provider call can launch during teardown.
        """
        if drain:
            await self.flush()
            self._closing = True
            return
        self._closing = True
        advance = getattr(self._core, "advance_source_continuity", None)
        if callable(advance):
            advance(self._source)
        self.invalidate_source_continuity(("runtime_shutdown",))
        if self._pending is not None:
            self._mc_resolve(self._pending[0])  # a live shutdown drops the pending candidate outright
        self._pending = None
        task = self._active
        if task is not None:
            if self._active_terminal is not None:
                self._active_terminal.expire("runtime_shutdown")
            task.cancel()
            # A provider may suppress cancellation indefinitely. Its registered done callback owns
            # eventual cleanup; live shutdown itself must remain bounded.
            if task.done():
                await asyncio.sleep(0)
        for abandoned in tuple(self._abandoned_dispatches):
            abandoned.cancel()


class _JudgeScheduler:
    """Bounded single-flight transport lane with one latest-pending batch.

    The runtime owner prepares and atomically launches a batch only after this
    lane accepts it. Provider tasks return frozen completions through ``ready``;
    they never run a whole-core cognition coroutine or reduce state themselves.
    """

    def __init__(
        self,
        core: Any,
        source: str,
        stats: RuntimeStats | None,
        *,
        ready: Any = None,
        stall_after_s: float = 20.0,
        backpressure_after: int = 3,
        decision_deadline_s: float | None = None,
    ) -> None:
        self._core = core
        self._source = source
        self._stats = stats
        self.ready = ready
        self._active: asyncio.Task | None = None
        self._abandoned_dispatches: set[asyncio.Task] = set()
        self._abandoned_terminal_sources: dict[
            asyncio.Task, tuple[Any, ...]
        ] = {}
        self._active_terminal_sources: tuple[Any, ...] = ()
        self._active_t: float | None = None
        self._pending: (
            tuple[float, tuple[Any, ...], int | None, float] | None
        ) = None
        self._results: deque = deque()
        self._trace: list[tuple[str, dict]] = []
        self._fatal: BaseException | None = None
        self._closing = False
        # A core can be driven by more than one finite run over its lifetime.
        # Seed the transport coordinate from the cognition owner's accepted
        # high-water so a fresh scheduler cannot replay ordinal 1 after a
        # prior run already accounted for it.
        self._completion_ordinal = int(
            getattr(core, "_provider_completion_high_water", 0)
        )
        self._split_protocol = callable(
            getattr(core, "launch_judgments", None)
        ) and callable(getattr(core, "dispatch_judgment", None))
        self._legacy_dispatch = (
            None
            if self._split_protocol
            else getattr(core, "perceive_" + "judged", None)
        )
        self._stall_after_s = stall_after_s
        self._backpressure_after = backpressure_after
        self._active_launch_wall: float | None = None
        self._stall_reported = False
        self._backpressure_reported = False
        self._consec_supersede = 0
        if decision_deadline_s is not None and (
            isinstance(decision_deadline_s, bool)
            or not isinstance(decision_deadline_s, (int, float))
            or not math.isfinite(float(decision_deadline_s))
            or float(decision_deadline_s) <= 0
        ):
            raise ValueError(
                "decision_deadline_s must be a positive finite number or None"
            )
        self._decision_deadline_s = (
            float(decision_deadline_s)
            if decision_deadline_s is not None
            else None
        )
        self._active_admitted_wall: float | None = None

    def _mc_now(self, fallback: float) -> float:
        clock = getattr(self._core, "clock", None)
        return clock() if callable(clock) else fallback

    @property
    def active(self) -> int:
        return int(
            self._active is not None or bool(self._abandoned_dispatches)
        )

    @property
    def pending(self) -> int:
        return 0 if self._pending is None else 1

    @property
    def can_accept(self) -> bool:
        return self._fatal is None and not self._closing

    def submit(
        self,
        t: float,
        jobs: Any,
        goal_ids: tuple[str, ...] | None = None,
    ) -> bool:
        """Accept one prepared batch and publish its reservations atomically."""
        if not self.can_accept:
            return False
        prepared = (
            tuple(jobs)
            if self._split_protocol
            else ("legacy", jobs, goal_ids)
        )
        if not prepared:
            return False
        admitted_wall = self._mc_now(t)
        if self._active is None and not self._abandoned_dispatches:
            self._launch(t, prepared, admitted_wall=admitted_wall)
            return True
        self._check_stall(t)
        revision = (
            prepared[0].token.goals_registry_revision
            if self._split_protocol
            else getattr(self._core.goals, "revision", None)
        )
        if self._pending is not None:
            dropped_t, dropped_jobs, _, _ = self._pending
            if self._stats is not None:
                self._stats.judge_superseded += 1
            self._trace.append(
                (
                    "judge_superseded",
                    {
                        "source": self._source,
                        "reason": "superseded_by_newer_admission",
                        "dropped_t": dropped_t,
                        "admitted_t": t,
                        "goal_ids": (
                            tuple(job.token.goal_id for job in dropped_jobs)
                            if self._split_protocol
                            else dropped_jobs[2]
                        ),
                        "slot": "pending",
                    },
                )
            )
            self._consec_supersede += 1
            self._check_backpressure(t)
        self._pending = (t, prepared, revision, admitted_wall)
        return True

    def _launch(
        self,
        t: float,
        jobs: tuple[Any, ...],
        *,
        admitted_wall: float,
    ) -> None:
        launched = (
            tuple(self._core.launch_judgments(jobs, runtime_trace=True))
            if self._split_protocol
            else jobs
        )
        self._active_terminal_sources = self._terminal_sources(
            launched if self._split_protocol else ()
        )
        if self._stats is not None:
            self._stats.perceive_dispatches += (
                len(launched) if self._split_protocol else 1
            )
            # RuntimeStats.judge_calls is the historical scheduler-launch
            # count (one admitted frame batch), not provider subcall spend.
            self._stats.judge_calls += 1
        self._active_t = t
        self._active_launch_wall = self._mc_now(t)
        self._active_admitted_wall = admitted_wall
        self._stall_reported = False
        self._active = asyncio.create_task(
            self._dispatch_with_deadline(
                launched,
                t=t,
                admitted_wall=admitted_wall,
            )
        )
        self._active.add_done_callback(self._on_done)

    def _check_stall(self, now_t: float) -> None:
        if (
            self._stall_reported
            or self._active is None
            or self._active_launch_wall is None
        ):
            return
        elapsed = self._mc_now(now_t) - self._active_launch_wall
        if elapsed < self._stall_after_s:
            return
        self._stall_reported = True
        self._trace.append(
            (
                "judge_scheduler_stalled",
                {
                    "source": self._source,
                    "reason": "active_judgment_exceeded_stall_threshold",
                    "active_t": self._active_t,
                    "elapsed_s": elapsed,
                    "stall_after_s": self._stall_after_s,
                },
            )
        )

    def _check_backpressure(self, now_t: float) -> None:
        if (
            self._backpressure_reported
            or self._consec_supersede < self._backpressure_after
        ):
            return
        self._backpressure_reported = True
        self._trace.append(
            (
                "judge_scheduler_backpressure",
                {
                    "source": self._source,
                    "reason": "sustained_pending_overwrite",
                    "consecutive_supersessions": self._consec_supersede,
                    "active_t": self._active_t,
                    "active_elapsed_s": (
                        self._mc_now(now_t) - self._active_launch_wall
                        if self._active_launch_wall is not None
                        else None
                    ),
                },
            )
        )

    def _launch_pending(self) -> None:
        if self._abandoned_dispatches:
            return
        if self._pending is None or self._closing or self._fatal is not None:
            self._pending = None
            return
        t, jobs, revision, admitted_wall = self._pending
        self._pending = None
        current = getattr(self._core.goals, "revision", None)
        if revision != current:
            if self._stats is not None:
                self._stats.judge_revision_dropped += 1
            self._trace.append(
                (
                    "judge_revision_dropped",
                    {
                        "source": self._source,
                        "reason": "goals_revision_changed",
                        "dropped_t": t,
                        "admitted_revision": revision,
                        "current_revision": current,
                    },
                )
            )
            return
        if self._decision_deadline_s is not None:
            age_s = max(0.0, self._mc_now(t) - admitted_wall)
            if age_s >= self._decision_deadline_s:
                if self._stats is not None:
                    self._stats.judge_deadline_dropped += 1
                self._trace.append(
                    (
                        "judge_deadline_dropped",
                        {
                            "source": self._source,
                            "reason": "pending_admission_deadline_expired",
                            "dropped_t": t,
                            "age_s": age_s,
                            "decision_deadline_s": self._decision_deadline_s,
                            "slot": "pending",
                        },
                    )
                )
                return
        self._launch(t, jobs, admitted_wall=admitted_wall)

    @staticmethod
    def _consume_abandoned(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    @staticmethod
    def _terminal_sources(jobs: tuple[Any, ...]) -> tuple[Any, ...]:
        """Find the exact adapters used by a launched provider batch.

        Goal-bound judges are not necessarily ``core.vision``. Stop at the
        first wrapper in each chain that exposes the terminal-latch protocol;
        wrappers such as NativeVideoVision delegate that protocol inward.
        """
        roots = [getattr(job, "adapter", None) for job in jobs]
        found: list[Any] = []
        seen: set[int] = set()
        queue = [root for root in roots if root is not None]
        while queue:
            current = queue.pop(0)
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            if callable(getattr(current, "take_terminal_error", None)):
                found.append(current)
                continue
            for name in ("_adapter", "inner", "adapter"):
                nested = getattr(current, name, None)
                if nested is not None:
                    queue.append(nested)
        return tuple(found)

    def _take_terminal_error(
        self,
        sources: tuple[Any, ...],
        *,
        active_t: float | None,
    ) -> BaseException | None:
        for source in sources:
            take_terminal = getattr(source, "take_terminal_error", None)
            error = take_terminal() if callable(take_terminal) else None
            if error is None:
                continue
            self._trace.append(
                (
                    "judge_terminal_failure",
                    {
                        "source": self._source,
                        "reason": "vision_adapter_terminal_outcome",
                        "active_t": active_t,
                        "exception_type": type(error).__name__,
                    },
                )
            )
            return error
        return None

    def _on_abandoned_done(self, task: asyncio.Task) -> None:
        late_failure: BaseException | None = None
        if not task.cancelled():
            try:
                late_failure = task.exception()
            except asyncio.CancelledError:
                late_failure = None
        sources = self._abandoned_terminal_sources.pop(task, ())
        terminal_error = self._take_terminal_error(
            sources,
            active_t=self._active_t,
        )
        if terminal_error is not None:
            self._fatal = terminal_error
            self._pending = None
        elif late_failure is not None:
            self._fatal = late_failure
            self._pending = None
            self._trace.append(
                (
                    "judge_abandoned_failure",
                    {
                        "source": self._source,
                        "reason": "late_dispatch_failure",
                        "active_t": self._active_t,
                        "exception_type": type(late_failure).__name__,
                    },
                )
            )
        self._abandoned_dispatches.discard(task)
        if self._active is None and self._fatal is None:
            self._launch_pending()

    def _track_abandoned(self, task: asyncio.Task) -> None:
        if task in self._abandoned_dispatches:
            return
        self._abandoned_dispatches.add(task)
        self._abandoned_terminal_sources[task] = (
            self._active_terminal_sources
        )
        task.add_done_callback(self._on_abandoned_done)

    async def _dispatch_with_deadline(
        self,
        jobs: tuple[Any, ...],
        *,
        t: float,
        admitted_wall: float,
    ) -> tuple[Any, ...] | list[Any]:
        if self._decision_deadline_s is None:
            return await self._dispatch(jobs, t=t)
        age_s = max(0.0, self._mc_now(t) - admitted_wall)
        remaining_s = self._decision_deadline_s - age_s
        if remaining_s <= 0:
            if self._stats is not None:
                self._stats.judge_deadline_expired += 1
            self._trace.append(
                (
                    "judge_deadline_expired",
                    {
                        "source": self._source,
                        "reason": "decision_deadline_expired_before_dispatch",
                        "active_t": t,
                        "age_s": age_s,
                        "decision_deadline_s": self._decision_deadline_s,
                    },
                )
            )
            return ()

        dispatch = asyncio.create_task(self._dispatch(jobs, t=t))
        try:
            done, _ = await asyncio.wait({dispatch}, timeout=remaining_s)
        except asyncio.CancelledError:
            # Source discontinuity/live shutdown cancels the scheduler-owned
            # wrapper. The provider child may suppress cancellation, so retain
            # physical occupancy until it truly terminates.
            dispatch.cancel()
            self._track_abandoned(dispatch)
            raise
        if dispatch in done:
            return dispatch.result()
        dispatch.cancel()
        self._track_abandoned(dispatch)
        elapsed_s = max(0.0, self._mc_now(t) - admitted_wall)
        if self._stats is not None:
            self._stats.judge_deadline_expired += 1
        self._trace.append(
            (
                "judge_deadline_expired",
                {
                    "source": self._source,
                    "reason": "active_decision_deadline_expired",
                    "active_t": t,
                    "age_s": elapsed_s,
                    "decision_deadline_s": self._decision_deadline_s,
                    "slot": "active",
                },
            )
        )
        return ()

    async def _dispatch(
        self,
        jobs: tuple[Any, ...],
        *,
        t: float,
    ) -> tuple[Any, ...] | list[Any]:
        if not self._split_protocol:
            if not callable(self._legacy_dispatch):
                raise AttributeError("core has no scheduler dispatch seam")
            _, frame, goal_ids = jobs
            kwargs = {
                "t": t,
                "expected_goals_revision": getattr(
                    self._core.goals, "revision", None
                ),
            }
            if goal_ids is not None and _accepts_keyword(
                self._legacy_dispatch, "goal_ids"
            ):
                kwargs["goal_ids"] = goal_ids
            return await self._legacy_dispatch(
                frame,
                self._source,
                **kwargs,
            )
        completions = []
        for job in jobs:
            completions.append(await self._core.dispatch_judgment(job))
            terminal_error = self._take_terminal_error(
                self._terminal_sources((job,)),
                active_t=t,
            )
            if terminal_error is not None:
                raise terminal_error
        return tuple(completions)

    def _on_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            self._active = None
            self._active_terminal_sources = ()
            self._launch_pending()
            return
        failure = task.exception()
        if failure is not None:
            self._fatal = failure
            self._active = None
            self._active_terminal_sources = ()
            self._pending = None
            return
        terminal_sources = self._active_terminal_sources
        if not terminal_sources:
            root_vision = getattr(self._core, "vision", None)
            terminal_sources = (
                (root_vision,)
                if callable(
                    getattr(root_vision, "take_terminal_error", None)
                )
                else ()
            )
        terminal_error = self._take_terminal_error(
            terminal_sources,
            active_t=self._active_t,
        )
        if terminal_error is not None:
            self._fatal = terminal_error
            self._active = None
            self._active_terminal_sources = ()
            self._pending = None
            return
        from dataclasses import replace

        if self._split_protocol:
            ready = []
            for completion in task.result():
                self._completion_ordinal += 1
                ready.append(
                    replace(
                        completion,
                        completion_ordinal=self._completion_ordinal,
                    )
                )
            row = (self._active_t, tuple(ready))
        else:
            row = (self._active_t, task.result())
        self._results.append(row)
        callback = self.ready
        if callback is not None:
            callback(row)
        self._active = None
        self._active_terminal_sources = ()
        recovered = []
        if self._stall_reported:
            recovered.append("stalled")
        if self._backpressure_reported:
            recovered.append("backpressure")
        self._stall_reported = False
        self._backpressure_reported = False
        if recovered:
            self._trace.append(
                (
                    "judge_scheduler_recovered",
                    {
                        "source": self._source,
                        "recovered_from": recovered,
                        "resolved_t": self._active_t,
                    },
                )
            )
        self._active_launch_wall = None
        self._active_admitted_wall = None
        self._consec_supersede = 0
        if not self._split_protocol:
            self._launch_pending()

    def owner_reduced(self) -> None:
        """Release the next lane slot only after ready state was owner-reduced."""
        if self._split_protocol and self._active is None:
            self._launch_pending()

    def drain(self) -> list[tuple[float, tuple[Any, ...]]]:
        if self._fatal is not None:
            raise self._fatal
        rows = []
        while self._results:
            rows.append(self._results.popleft())
        return rows

    def drain_trace(self) -> list[tuple[str, dict]]:
        rows = self._trace
        self._trace = []
        return rows

    def invalidate_source_continuity(self, reasons: tuple[str, ...]) -> None:
        """Cancel transport work; the runtime owner advances continuity once."""
        reason = ",".join(
            str(getattr(item, "value", item)) for item in reasons
        ) or "source_discontinuity"
        if self._pending is not None:
            self._pending = None
            if self._stats is not None:
                self._stats.judge_source_discontinuity_dropped += 1
                self._stats.judge_source_discontinuity_pending_dropped += 1
            self._trace.append(
                (
                    "judge_source_discontinuity_dropped",
                    {
                        "source": self._source,
                        "reason": reason,
                        "slot": "pending",
                    },
                )
            )
        if self._active is not None and not self._active.done():
            self._active.cancel()
            if self._stats is not None:
                self._stats.judge_source_discontinuity_dropped += 1
                self._stats.judge_source_discontinuity_active_dropped += 1
            self._trace.append(
                (
                    "judge_source_discontinuity_dropped",
                    {
                        "source": self._source,
                        "reason": reason,
                        "slot": "active",
                    },
                )
            )

    async def flush(self) -> None:
        while self._fatal is None and (
            self._active is not None
            or self._pending is not None
            or self._abandoned_dispatches
        ):
            task = self._active
            if task is None:
                if self._abandoned_dispatches:
                    await asyncio.wait(
                        tuple(self._abandoned_dispatches),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    await asyncio.sleep(0)
                    continue
                if self._split_protocol:
                    return
                self._launch_pending()
                continue
            if task.done():
                await asyncio.sleep(0)
                continue
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
            except BaseException:  # noqa: BLE001
                pass

    async def close(self, *, drain: bool) -> None:
        if drain:
            await self.flush()
            self._closing = True
            return
        self._closing = True
        self._pending = None
        task = self._active
        if task is not None:
            task.cancel()
            if task.done():
                await asyncio.sleep(0)
        for abandoned in tuple(self._abandoned_dispatches):
            abandoned.cancel()


# Public-private compatibility name used by the shared-owner promotion oracle.
_ProviderScheduler = _JudgeScheduler


# Per-path timebase-skew tolerances. The tripwire compares a frame/chunk `t` against the core
# clock; a divergence past the path's tolerance means the two timelines have come apart (a live source
# left on "media", or a wedged pump replaying stale timestamps). Per-path because the two capture queues
# have very different legitimate backlogs: the video queue is ~64 deep @ ~4fps (~16s), while the audio
# queue is ~256 deep @ ~4 chunks/s (~64s) and a chunk sitting behind a blocked pump is exactly an
# expected backlog condition — one global tolerance would false-fire on normal audio lag. Both are far
# below a media/wall confusion (~1e6 s), so a real skew still trips; both are comfortably above the
# queue's own worst-case lag, so a healthy backlog never does.
_VIDEO_SKEW_TOL_S = 30.0   # video queue worst-case ~16s + margin
_AUDIO_SKEW_TOL_S = 120.0  # audio queue worst-case ~64s + margin (a blocked pump is legitimate)

# Media-clock pacing lookahead for a FILE/CONTAINER audio_source. A recorded audio pump (ffmpeg
# demux / wave read) is throttled to the video frame loop's own judged media time so it never streams
# ahead into frames that haven't been judged yet — the fix for a fused judge hearing the file's TAIL no
# matter which frame it was judging (see `_pump_audio`'s `_audio_paced` gate below). The lookahead lets
# the AudioWindow stay filled slightly AHEAD of the current frame (matching `audio_settle_s`-style
# onset-then-settle usage) without letting it race arbitrarily far ahead.
_AUDIO_PACE_LOOKAHEAD_S = 0.5
_RUNTIME_OWNER_QUEUE_MAX = 4_096


async def _assert_timebase(
    t: float,
    clock: "Callable[[], float] | None",
    *,
    tol: float,
    path: str,
    emitted: "set[str]",
    on_health: "Callable | None",
) -> None:
    """CI net: emit ``on_health("timebase_skew", …)`` ONCE per path when a frame/chunk ``t`` and the
    core clock diverge past ``tol``. Only meaningful on a LIVE path (where ``t`` is supposed to BE the
    wall clock); an offline/media source deliberately runs ``t`` on a synthetic timeline and is never
    checked. Idempotent: once a path has emitted, it stays quiet (no per-frame spam). Best-effort — a
    missing clock or ``on_health`` is a no-op."""
    if clock is None or on_health is None or path in emitted:
        return
    now = clock()
    skew = abs(t - now)
    if skew > tol:
        emitted.add(path)
        await _call_sink(on_health, RuntimeHealthEvent(
            "timebase_skew", ts=t,
            data={"path": path, "frame_t": t, "clock": now, "skew_s": skew, "tolerance_s": tol},
        ))


async def _drain_judge_health(core: Any, on_health: "Callable | None", t: float) -> None:
    """Surface every judge-degradation event ``Percept.perceive_judged``/
    ``_observe_and_log`` queued on ``core._judge_health_log`` (agent.py) -- a per-tick
    ``"judge_degraded"`` for each fail-closed-boundary catch (timeout / provider error / malformed /
    partial), plus a ONE-TIME ``"judge_persistent_failure"`` escalation the moment a watch key's
    consecutive-degrade streak crosses ``percept.agent.CONSECUTIVE_DEGRADE_ESCALATE``. This is the
    loud signal the old fail-loud path's ``runtime_error`` health event used to be for a judge
    failure -- degrading no longer crashes the run (see agent.py), so it must stay observable through
    ``on_health`` instead of only a log line. Drains UNCONDITIONALLY (even with ``on_health=None``)
    so the queue never accumulates past one tick's worth, matching ``_JudgeScheduler.drain_trace()``'s
    own drain-regardless pattern above. Percept itself never calls ``on_health`` -- that channel
    belongs to this module; agent.py only queues."""
    log = getattr(core, "_judge_health_log", None)
    if not log:
        return
    entries, log[:] = list(log), []
    if on_health is None:
        return
    for kind, key, cause in entries:
        await _call_sink(on_health, RuntimeHealthEvent(kind, ts=t, data={"key": repr(key), "cause": repr(cause)}))


async def _drain_gate_health(core: Any, on_health: "Callable | None", t: float) -> None:
    """Surface every GATE SUPPRESSION ``percept.gate.Gate`` queued on ``core._gate_health_log``
    (agent.py) as a ``RuntimeHealthEvent("gate_suppressed")`` carrying the gate key, the typed reason
    code (one of ``percept.gate.SUPPRESSION_REASONS`` -- ``start_anchored`` / ``refractory``) and the
    stream time the decision was made at.

    A suppression is a fire the BASELINE rising-edge gate would have delivered, withheld by an opt-in
    precision guard. That is exactly the class of decision an operator must be able to see: without
    this, turning on ``require_false_first`` / ``refractory_s`` makes the runtime quieter with no
    signal saying why. Both guards are off by default, so a default watch drains an empty queue every
    tick and emits nothing (no default behavior change, no new per-frame noise).

    Drains UNCONDITIONALLY (even with ``on_health=None``) so the queue never accumulates past one
    tick's worth -- the same drain-regardless contract :func:`_drain_judge_health` and
    ``_JudgeScheduler.drain_trace()`` follow."""
    log = getattr(core, "_gate_health_log", None)
    if not log:
        return
    entries, log[:] = list(log), []
    if on_health is None:
        return
    for key, reason, at in entries:
        await _call_sink(on_health, RuntimeHealthEvent(
            "gate_suppressed", ts=t, data={"key": repr(key), "reason": reason, "at": at},
        ))


async def _drain_restore_health(core: Any, on_health: "Callable | None") -> None:
    """Surface the ``episode_memory_reset`` event ``Percept.restore`` queued on
    ``core._restore_health_log`` (agent.py). Episode/counting memory is SESSION-ONLY for
    v1.4 (ADR-006): a restart rehydrates goals but NOT the per-goal episode map, so a
    counting goal resumes at zero and may under-count until re-observed. restore() queues one entry
    when it restored active counting goals into an empty episode map; this drains it into a loud
    ``on_health`` event so the amnesia is SEEN, not silently discovered as a wrong count at 2am.

    Runs ONCE, right after ``load_state`` at startup (no frame timestamp yet -> ``ts`` is left
    unset). Drains UNCONDITIONALLY (even with ``on_health=None``) so the queue never lingers, the
    same drain-regardless contract :func:`_drain_judge_health` / :func:`_drain_gate_health` follow.
    Percept itself never calls ``on_health`` -- that channel belongs to this module."""
    log = getattr(core, "_restore_health_log", None)
    if not log:
        return
    entries, log[:] = list(log), []
    if on_health is None:
        return
    for kind, data in entries:
        await _call_sink(on_health, RuntimeHealthEvent(kind, data=data))


def _log_disposition(log: Any, *, incident_id: str, fire: Any, disposition: str, ts: float) -> None:
    """Append one gate disposition to the :class:`~percept.labelstore.DispositionLog`.

    Best-effort: a disposition-log IO error must never kill the watch (it is telemetry, not the trust
    surface). ``incident_id`` is ``""`` for an ASK (asks are not incidents)."""
    from percept.labelstore import verdict_ref as _vr

    try:
        log.append(
            incident_id=incident_id,
            verdict_ref=_vr(getattr(fire, "verdict", None)),
            disposition=disposition,
            ts=float(ts),
        )
    except Exception:  # noqa: BLE001 - telemetry write must not crash the run
        pass


async def _aiter(source: Any, fps: float, **source_kwargs) -> AsyncIterator[SourceEnvelope]:
    """Normalize captured and pushed video inputs to the versioned source envelope."""
    sequence = 0

    def normalize(item: Any) -> SourceEnvelope:
        nonlocal sequence
        if isinstance(item, SourceEnvelope):
            return item
        result = legacy_video_envelope(
            item,
            source_id="video",
            sequence=sequence,
            observed_time=time.monotonic(),
            timebase="observed" if source_kwargs.get("timebase") == "wall" else "source",
        )
        sequence += 1
        return result

    if isinstance(source, str):
        from percept.runtime.source import open_source

        async for item in open_source(source, fps=fps, **source_kwargs):
            yield normalize(item)
    elif hasattr(source, "__aiter__"):
        async for item in source:  # already an async iterator
            yield normalize(item)
    else:  # a sync iterable (a pre-decoded clip / a test stream)
        for item in source:
            yield normalize(item)
            await asyncio.sleep(0)  # cooperate with the event loop


@dataclass(frozen=True)
class _AudioIngress:
    envelope: SourceEnvelope | None = None
    unit: AudioUnit | None = None
    legacy_item: Any = None
    legacy_timebase: str = "source"
    observed_time: float | None = None


async def _audio_aiter(
    source: Any,
    *,
    observed_clock: Callable[[], float] = time.monotonic,
    **source_kwargs,
) -> AsyncIterator[_AudioIngress]:
    """Normalize audio inputs to an envelope plus any explicitly ordered unit."""

    def normalize(item: Any) -> _AudioIngress:
        if isinstance(item, SourceEnvelope):
            return _AudioIngress(item)
        if isinstance(item, AudioUnit):
            # AudioUnit sequence/interval authority belongs to AudioWindow. The worker creates a
            # separate observation-only carrier envelope only after transactional unit preflight.
            return _AudioIngress(unit=item, observed_time=observed_clock())
        timebase = "observed" if source_kwargs.get("timebase") == "wall" else "source"
        return _AudioIngress(
            legacy_item=item,
            legacy_timebase=timebase,
            observed_time=observed_clock(),
        )

    if isinstance(source, str):
        from percept.runtime.audio import open_audio_source

        async for item in open_audio_source(source, **source_kwargs):
            yield normalize(item)
    elif hasattr(source, "__aiter__"):
        async for item in source:
            yield normalize(item)
    else:
        for item in source:
            yield normalize(item)
            await asyncio.sleep(0)


async def _call_sink(fn: Callable, *args) -> None:
    result = fn(*args)
    if inspect.isawaitable(result):
        await result


async def _call_sink_with_policy(
    fn: Callable,
    *args,
    errors: str = "raise",
    stats: RuntimeStats | None = None,
    on_health: Callable | None = None,
    health_kind: str = "sink_error",
) -> None:
    try:
        await _call_sink(fn, *args)
    except Exception as exc:
        if stats is not None:
            stats.sink_errors += 1
        if on_health is not None:
            event = RuntimeHealthEvent(
                health_kind,
                data={"error": repr(exc), "sink": getattr(fn, "__name__", fn.__class__.__name__)},
            )
            await _call_sink(on_health, event)
        if errors == "ignore":
            return
        raise


def _clip_from_chunks(chunks: list, *, t_start: float | None = None, t_end: float | None = None) -> Any:
    """Assemble ONE audio clip from a list of raw audio chunks — the endpointed utterance handed to
    ``on_utterance``. The chunks come from the endpointer's OWN accumulation buffer (started→endpoint),
    NOT from slicing the shared rolling AudioWindow (whose small fused-judge size silently truncated any
    utterance longer than it). Typed :class:`AudioFrame` chunks are concatenated PCM-wise; a raw-bytes/
    tuple test source falls back to :func:`percept.agent._concat_audio`. Returns ``None`` for an empty
    list (nothing to send). ``t_start``/``t_end`` stamp the returned frame's clip bounds (provenance)."""
    from percept.agent import _concat_audio

    if not chunks:
        return None
    clip = _concat_audio(chunks)
    if clip is None or not hasattr(clip, "pcm"):
        return clip  # raw bytes / (mime, bytes), or a rejected mixed-format sequence
    from percept.contracts import AudioFrame

    return AudioFrame(
        pcm=clip.pcm,
        sample_rate=clip.sample_rate,
        channels=clip.channels,
        sample_format=clip.sample_format,
        t_start=t_start,
        t_end=t_end,
    )


# Energy-freshness. A dead/stalled mic leaves the pump's last energy pinned at a (possibly
# speech-level) value; a WakePolicy fusing that channel would then escalate on phantom audio forever.
# The frame loop reads energy only while FRESH — within ~2 chunk periods of the last real reading — so a
# stale reading reads None and the gate degrades to VIDEO-ONLY (a dead mic must never pin the gate open).
_DEFAULT_CHUNK_DT_S = 0.25   # ~4 chunks/s documented mic cadence, until an inter-chunk delta is observed
_ENERGY_STALE_FACTOR = 2.0   # "~2 chunk periods" per the spec
_ENERGY_STALE_FLOOR_S = 0.5  # floor so a fast mic (small dt) still tolerates normal jitter (~2 @ 4/s)


def _energy_fresh(energy_t: float | None, now: float | None, chunk_dt: float | None) -> bool:
    """True when the last real energy reading (at ``energy_t``) is recent enough to feed the gate at
    frame time ``now``. Both must share a timebase (live capture stamps wall on both). Unknown
    ``energy_t`` (no reading yet) is never fresh; an unknown ``now`` can't judge staleness → fresh."""
    if energy_t is None:
        return False
    if now is None:
        return True
    dt = chunk_dt if (chunk_dt and chunk_dt > 0) else _DEFAULT_CHUNK_DT_S
    return (now - energy_t) <= max(_ENERGY_STALE_FACTOR * dt, _ENERGY_STALE_FLOOR_S)


def _amplitude_threshold_to_energy(value: float, *, name: str) -> float:
    """Convert the runtime's RMS/AMPLITUDE threshold knob to MEAN-SQUARE ENERGY, by SQUARING it.

    The two audio knobs a caller passes to :func:`run` — ``audio_wake={"high"/"low"}`` and
    ``utterance_threshold`` — are amplitude; everything downstream of this function
    (:class:`percept.harness.detectors.AcousticOnset`, :class:`percept.runtime._vad.EnergyVad`) is
    mean-square energy. The conversion is NOT linear and the difference is two orders of magnitude at
    the values people actually type: ``utterance_threshold=0.01`` is energy ``1e-4``, and
    ``audio_wake={"high": 0.3}`` is energy ``0.09``. Read every threshold in this module against the
    domain it lives in; a number moved across this boundary unsquared is off by its own magnitude.
    """
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return float(value) ** 2


# --- acoustic wake-gate deafness accounting --------------------------------------------------
#
# AUDIO-GATE-DEAF. The shipped AcousticOnset bar used to sit above anything real audio produces
# (energy 0.09 = RMS 0.3, against a measured corpus maximum of 0.0277), so
# `run(audio_gate="acoustic-onset")` could observe thousands of chunks, wake ZERO times, and emit
# nothing at all: "there was no sound" and "my gate could not hear" were byte-for-byte identical to
# the caller. `percept doctor` already refuses to let the VIDEO eye fail silently that way
# (ZERO_ADMITS_ON_FIXTURE); `zero_wake_coverage` is the audio twin. It reports the same SHAPE of
# fact and leaves the reading to the consumer — a doorbell watch in an empty house correctly never
# wakes, and alarming on that would contradict tier-0's whole economic claim — but it reports the
# gate's SETTING beside what the signal ACTUALLY did, which is the pair the caller needs to tell the
# two apart. It is not in percept.tracing.DEGRADED_HEALTH_KINDS: it records as state="info".

#: Seconds of OBSERVED audio after which a run that has never woken reports once, mid-run. A live
#: watch may never reach its ``runtime_ended``, so an end-of-run-only diagnostic would never reach a
#: 24/7 agent — the exact deployment where a deaf gate costs the most. Latched: at most one
#: ``zero_wake_coverage`` per run, either here or on the way out.
_ZERO_WAKE_DIAGNOSTIC_S = 60.0


class _AcousticGateWitness:
    """Run-local record of what the acoustic wake gate was FED and what it did with it.

    Cheap and bounded — six scalars, no per-chunk retention — so it costs nothing to keep on a
    long-running live watch, which is the one that needs it most.
    """

    __slots__ = (
        "chunks", "wakes", "energy_max", "energy_sum",
        "above_rearm", "first_t", "last_t", "reported",
    )

    def __init__(self) -> None:
        self.chunks = 0
        self.wakes = 0
        self.energy_max = 0.0
        self.energy_sum = 0.0
        self.above_rearm = 0
        self.first_t: float | None = None
        self.last_t: float | None = None
        self.reported = False

    def observe(
        self, t: float, energy: float, *, wakes: int, rearm: float | None = None
    ) -> None:
        """Record ONE energy reading handed to the detector and the onsets it returned."""
        value = float(energy)
        self.chunks += 1
        self.wakes += int(wakes)
        self.energy_sum += value
        if value > self.energy_max:
            self.energy_max = value
        if rearm is not None and value >= rearm:
            self.above_rearm += 1
        if self.first_t is None:
            self.first_t = float(t)
        self.last_t = float(t)

    @property
    def observed_s(self) -> float:
        """Span of gate-observed audio, in the run's own timebase."""
        if self.first_t is None or self.last_t is None:
            return 0.0
        return max(0.0, self.last_t - self.first_t)

    def due(self, *, at_end: bool) -> bool:
        """True at most ONCE per run: this gate was fed audio and never woke on any of it."""
        if self.reported or self.chunks == 0 or self.wakes:
            return False
        return at_end or self.observed_s >= _ZERO_WAKE_DIAGNOSTIC_S

    def payload(
        self,
        *,
        detector: str,
        high: float | None,
        low: float | None,
        at_end: bool,
    ) -> dict[str, Any]:
        """Build the event body and latch. Thresholds are MEAN-SQUARE ENERGY, like the detector's."""
        self.reported = True
        return {
            "detector": detector,
            "chunks_observed": self.chunks,
            "observed_s": round(self.observed_s, 3),
            "wakes": self.wakes,
            "wake_threshold": high,
            "rearm_threshold": low,
            "energy_max": self.energy_max,
            "energy_mean": (self.energy_sum / self.chunks) if self.chunks else 0.0,
            "chunks_at_or_above_rearm": self.above_rearm if low is not None else None,
            # THE number that separates the two readings, and the reason this event exists. It is the
            # loudest thing the run heard as a fraction of the bar it had to clear. Near 1.0: real
            # audio kept just missing a mis-set bar — retune `audio_wake`. Near 0.0: there genuinely
            # was nothing to hear, and the gate saved every model call exactly as intended.
            "loudest_fraction_of_threshold": (
                (self.energy_max / float(high)) if (high and float(high) > 0.0) else None
            ),
            "at_end": at_end,
        }


# Cap so a tiny brief-window prior can't explode the capture rate (mirrors the bench
# _CAPTURE_HEADROOM_FPS): the fovea coast still drives JUDGING, but DECODE never exceeds this.
_BRIEF_CAPTURE_CAP_FPS = 8.0

# Coverage-guarantee tripwire tolerances. A brief goal promises >= its window Nyquist in CAPTURED
# frames; if the OPEN source can't deliver that (a slow live source, or the cap clamping below Nyquist for
# a very short window), the guarantee is silently downgraded. We measure the OBSERVED capture cadence over
# a short window of real frame timestamps and report ONCE when it falls short — the durable rule (report an
# unmeetable guarantee) that generalizes past this one bug.
_CAPTURE_UNMET_MIN_SPAN_S = 1.0  # measure over >=1s of real frames before judging delivery (avoid startup jitter)
_CAPTURE_UNMET_TOL = 0.85        # observed must reach 85% of the promised Nyquist floor or we report


def _capture_plan(goals: list, base_fps: float) -> tuple[float, Any, float]:
    """Derive ``(capture_fps, coast_controller, required_fps)`` from active goals' sensor plans.

    A brief-action *judge* goal (one carrying a ``sense.brief_window_s`` prior) compiles a SENSITIVE
    above-Nyquist cadence via :func:`~percept.sensorplan.sensor_plan_for`; ordinary goals compile no
    cadence floor. So with no brief goal active this returns ``(base_fps, None, 0.0)`` and the worker is
    byte-for-byte the fixed-fps default. When a brief goal IS active we (a) raise the capture rate to
    cover its cadence (capped) and (b) return a fovea :class:`~percept.sampling.SamplingController` whose
    coast floor forces a judge on quiet, sub-threshold frames the motion gate would skip — so a
    sub-second window is still covered. The coast tracks the MOST sensitive active goal.

    ``required_fps`` is the highest brief-goal Nyquist floor (``base_fps`` of its sensitive cadence) —
    the coverage GUARANTEE the open source must actually deliver. The worker compares the observed
    capture cadence against it and emits ``capture_plan_unmet`` when it can't be met: a coverage
    guarantee that can't be honored is REPORTED, never silently downgraded."""
    from percept.sampling import SamplingController, SensePolicy
    from percept.sensorplan import sensor_plan_for

    capture = float(base_fps)
    required = 0.0
    policy: SensePolicy | None = None
    for g in goals:
        try:
            cad = sensor_plan_for(g).cadence
        except Exception:
            continue  # a goal whose sense doesn't compile a plan never raises the floor
        base = cad.get("base_fps")
        if not base:
            continue  # ordinary (non-brief) goal: no cadence floor
        mx = cad.get("max_fps") or base
        capture = max(capture, min(_BRIEF_CAPTURE_CAP_FPS, float(mx)))
        required = max(required, float(base))  # the window Nyquist we PROMISED to cover
        et = cad.get("escalate_threshold")
        pol = SensePolicy(
            base_fps=float(base), max_fps=float(mx),
            escalate_threshold=float(et) if et is not None else SensePolicy().escalate_threshold,
        )
        if policy is None or pol.base_fps > policy.base_fps:
            policy = pol
    return capture, (SamplingController(policy) if policy is not None else None), required


def _goals_want_audio(goals: list) -> bool:
    """True when an active goal's compiled :class:`~percept.sensorplan.SensorPlan` perception modality
    is ``"fused"``/``"audio"`` — the trigger for auto-deriving ``audio_source`` from a video-container
    ``source`` (see ``run()``'s container-demux wiring). A goal whose sense doesn't compile a plan is
    skipped, never treated as wanting audio."""
    from percept.sensorplan import sensor_plan_for

    for g in goals:
        try:
            modality = sensor_plan_for(g).perception.get("modality")
        except Exception:
            continue
        if modality in ("fused", "audio"):
            return True
    return False


# Recorded-media extensions. A `str` source carrying one of these is a RECORDED source:
# it has a real media timeline, so `t` (and every `incident.ts`) must be stamped on it. Anything else
# a str source can be (webcam:, mic:, rtsp://, an opaque live stream) has no media clock at all, so it
# stays on wall/monotonic time. Unknown/exotic URIs therefore keep today's live behaviour.
_RECORDED_SUFFIXES = (
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".m2ts", ".ts", ".gif",
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus",
)
_LIVE_PREFIXES = (
    "webcam:", "webcam://", "mic:", "mic://", "screen://",
    "rtsp://", "rtsps://", "rtp://", "udp://", "tcp://",
)
_LIVE_EXACT = ("webcam", "mic")


def _is_recorded_source(uri: str) -> bool:
    """True when a ``str`` source names RECORDED media (a file path / file URI with a media suffix).

    Deliberately conservative: a device/stream prefix is live no matter what it ends with, and an
    unrecognised URI is treated as LIVE (today's behaviour), so ``timebase="auto"`` can only move a
    source onto media time when there demonstrably IS a media timeline to move it onto."""
    s = uri.strip()
    low = s.lower()
    if low in _LIVE_EXACT or low.startswith(_LIVE_PREFIXES):
        return False
    if low.startswith("file://"):
        low = low[len("file://"):]
    base = low.split("?", 1)[0].split("#", 1)[0]
    return base.endswith(_RECORDED_SUFFIXES)


def _resolve_timebase(
    source: Any,
    requested: str,
    *,
    deterministic_override: bool = True,
) -> EffectiveTimebase:
    """Resolve ``run(timebase=…)`` against ``source``.

    ``"auto"`` (the default) is the fix for a specific bug: a 36s clip decoded in 0.72s
    of wall time used to stamp every ``incident.ts`` with the decode instant, making the timestamps
    useless as audit evidence against the media. Recorded sources now run on MEDIA time; live
    devices/streams stay on WALL time (they must share the agent's deadline clock). An
    explicit ``"media"``/``"wall"`` always wins."""
    if requested not in TIMEBASE_CHOICES:
        raise ValueError(
            f"timebase must be one of {TIMEBASE_CHOICES}, got {requested!r}"
        )
    if requested != "auto" and (isinstance(source, str) or deterministic_override):
        return EffectiveTimebase(requested=requested, resolved=requested, reason="explicit-override")
    if not isinstance(source, str):
        # A deterministic iterable (tests, replay, the eval harness) feeds its OWN explicit `t`; the
        # worker never re-stamps it, so nothing is resolved for it — byte-for-byte unchanged.
        return EffectiveTimebase(requested=requested, resolved="media", reason="deterministic-stream")
    if _is_recorded_source(source):
        return EffectiveTimebase(requested=requested, resolved="media", reason="recorded-source")
    return EffectiveTimebase(requested=requested, resolved="wall", reason="live-source")


def _as_goal(g: Any, i: int) -> Any:
    """Coerce a convenience `goals=[...]` entry into a Goal: a plain condition string, a dict of Goal
    fields, or a Goal passthrough. A string `"the milk is boiling over"` -> Goal(id=f"g{i}", condition=,
    say=) so the one-liner `percept.run(source=, goals=["..."])` needs no Goal import."""
    from percept.adapters import Goal

    if isinstance(g, Goal):
        return g
    if isinstance(g, str):
        return Goal(id=f"g{i}", condition=g, say=g)
    if isinstance(g, dict):
        d = dict(g)
        d.setdefault("id", f"g{i}")
        d.setdefault("say", d.get("condition", ""))
        return Goal(**d)
    raise TypeError(f"goals[{i}] must be a str, dict, or Goal, not {type(g).__name__}")


async def _run_audio_only(
    core: Any,
    *,
    audio_source: Any,
    gate_source: str,
    audio_gate: Any,
    audio_wake: dict | None,
    audio_settle_s: float,
    audio_window_s: float | None,
    judge_dispatch: str,
    metrics_collector: Any,
    judge_stall_after_s: float,
    judge_backpressure_after: int,
    judge_decision_deadline_s: float | None,
    timebase: str,
    wall_clock: Callable[[], float],
    clock: Callable | None,
    paused: Callable | None,
    muted: Callable | None,
    on_incident: Callable | None,
    on_ask: Callable | None,
    on_health: Callable | None,
    sink_errors: str,
    stats: RuntimeStats | None,
    episode_inactivity_s: float | None,
    coverage_staleness_s: float | None,
    regate_on_goal_change: bool,
) -> AsyncIterator:
    """Run exact compiled audio Watches without a synthetic video carrier."""
    from percept.audio_input import (
        audio_frame_from_units,
        audio_snippet_from_units,
        canonicalize_audio_ingress_frame,
        canonicalize_audio_unit,
    )
    from percept.behavior import program_of
    from percept.events import AudioWindow
    from percept.harness.tier0 import compile_watchspec, schedule_active

    if audio_source is None:
        raise ValueError("source=None selects audio-only execution and requires audio_source=")
    if judge_dispatch not in ("inline", "async"):
        raise ValueError("judge_dispatch must be 'inline' or 'async'")
    if timebase not in TIMEBASE_CHOICES:
        raise ValueError(f"timebase must be one of {TIMEBASE_CHOICES!r}")
    resolved_timebase = _resolve_timebase(audio_source, timebase).resolved
    if (
        not isinstance(audio_settle_s, (int, float))
        or isinstance(audio_settle_s, bool)
        or not math.isfinite(audio_settle_s)
        or audio_settle_s < 0
    ):
        raise ValueError("audio_settle_s must be finite and non-negative")

    def source_goals() -> list:
        return [
            goal for goal in core.goals.active()
            if getattr(goal, "source", "camera") == gate_source
        ]

    def preflight() -> tuple[list, dict[str, Any]]:
        goals = source_goals()
        if not goals:
            raise ValueError("audio-only execution requires at least one active Watch")
        bindings: dict[str, Any] = {}
        conditions = []
        for goal in goals:
            if goal.id not in getattr(core, "_watch_runtime_bindings", {}):
                raise ValueError(
                    f"audio-only execution requires a compiled Watch; {goal.id!r} is unbound"
                )
            program = program_of(goal)
            if program.perception.modality != "audio" or goal.state.get("type", "watch") != "watch":
                raise ValueError(
                    f"audio-only execution supports compiled audio watch goals only; "
                    f"{goal.id!r} resolved to {program.perception.modality!r}"
                )
            bindings[goal.id] = core._audio_judge_binding_for(goal)
            compiled = core._watch_runtime_bindings[goal.id][0]
            conditions.extend(
                compile_watchspec([goal], **dict(compiled.watchspec_kwargs)).conditions
            )
        if audio_gate not in (None, "acoustic-onset"):
            raise ValueError(
                "compiled audio-only execution accepts audio_gate='acoustic-onset' or None"
            )
        if audio_gate is None and any(item.gate_on != "fixed" for item in conditions):
            raise ValueError(
                "audio_gate=None is executable only when every audio Watch uses fixed cadence"
            )
        return conditions, bindings

    # No source read or runtime mutation occurs before this complete topology/trust preflight.
    conditions, bindings = preflight()
    capability = min(
        (bound.capability for bound in bindings.values()),
        key=lambda item: (item.max_input_bytes, item.max_provider_bytes),
    )
    window_seconds = float(
        audio_window_s or getattr(core.settings.events, "audio_window_s", 4.0)
    )
    window = AudioWindow(window_seconds)
    core.audio_windows[gate_source] = window
    tracker = SourceTracker()
    onset = None
    if audio_gate == "acoustic-onset":
        from percept.edge.detectors import AcousticOnset

        opts = dict(audio_wake or {})
        for key in ("high", "low"):
            if key in opts:
                opts[key] = _amplitude_threshold_to_energy(
                    opts[key], name=f"audio_wake.{key}"
                )
        onset = AcousticOnset(**opts)
    condition_by_goal = {item.goal_id: item for item in conditions}
    last_admit: dict[str, float] = {}
    wake_at: float | None = None
    carrier_sequence = 0
    goals_revision = getattr(core.goals, "revision", None)
    deadline_bound: set[str] = set()
    suppressed = False
    suppressed_run = 0  # units dropped since the CURRENT suppression window opened
    witness = _AcousticGateWitness()
    cov = stats if stats is not None else RuntimeStats()
    scheduler = (
        _LegacyJudgeScheduler(
            core,
            gate_source,
            cov,
            perceive=core.perceive_audio_judged,
            include_payload=True,
            metrics_collector=metrics_collector,
            stall_after_s=judge_stall_after_s,
            backpressure_after=judge_backpressure_after,
            decision_deadline_s=judge_decision_deadline_s,
        )
        if judge_dispatch == "async" else None
    )

    async def emit_fires(fires: Any, at: float) -> AsyncIterator:
        for fire in fires or ():
            if getattr(fire, "action", None) == ASK:
                cov.asks += 1
                if on_ask is not None:
                    await _call_sink_with_policy(
                        on_ask, fire, errors=sink_errors, stats=stats,
                        on_health=on_health, health_kind="ask_sink_error",
                    )
                continue
            if getattr(fire, "action", None) != FIRE:
                continue
            cov.fires += 1
            units = tuple(getattr(fire, "audio_units", None) or ())
            snippet = (
                audio_snippet_from_units(
                    units, capability=bindings[fire.goal_id].capability
                )
                if units else None
            )
            incident = make_incident(fire, ts=at, snippet=snippet)
            emit_trace = getattr(core, "_emit_trace", None)
            if getattr(core, "on_trace", None) is not None and callable(emit_trace):
                await emit_trace(
                    f"{gate_source}@{at}",
                    at,
                    "incident.created",
                    goal_id=incident.concern_id,
                    incident_id=incident.id,
                    judgment_event_id=incident.judgment_event_id,
                    evidence_origin=incident.evidence_origin,
                )
            if on_incident is not None:
                await _call_sink_with_policy(
                    on_incident, incident, errors=sink_errors, stats=stats,
                    on_health=on_health, health_kind="incident_sink_error",
                )
            cov.incidents += 1
            yield incident

    async def drain_completed(at: float) -> AsyncIterator:
        if scheduler is None:
            return
        completed = scheduler.drain()
        for kind, data in scheduler.drain_trace():
            if on_health is not None:
                await _call_sink(on_health, RuntimeHealthEvent(kind, ts=at, data=data))
        for admitted_t, fires, _evidence in completed:
            async for incident in emit_fires(fires, admitted_t):
                yield incident
        await _drain_judge_health(core, on_health, at)

    def coverage_data() -> dict[str, Any]:
        # THE INGRESS LEDGER MUST RECONCILE (AUDIO-SUPPRESSED-NO-RECEIPT). Every unit the source
        # offered leaves through exactly one of three doors, and each door has a term here:
        #   audio_offered == audio_chunks + audio_suppressed + audio_rejected
        # `audio_chunks` accepted, `audio_suppressed` dropped because the caller had the runtime
        # muted/paused, `audio_rejected` refused as a malformed carrier (each already receipts a
        # per-unit `audio_unit_rejected`). The suppressed term used to exist NOWHERE: units simply
        # vanished while `source_coverage.excluded` still reported 0, so a caller reconciling their
        # own chunk count against ours found a silent deficit with no term to attribute it to.
        return {
            "source": gate_source,
            "goal_ids": tuple(condition_by_goal),
            "audio_chunks": cov.audio_chunks,
            "audio_suppressed": cov.audio_suppressed,
            "audio_rejected": cov.audio_rejected,
            "audio_offered": cov.audio_chunks + cov.audio_suppressed + cov.audio_rejected,
            # What the acoustic wake gate DID, so a zero here has a companion fact rather than being
            # inferred from the absence of admissions.
            "audio_wakes": witness.wakes,
            "admitted": cov.escalations,
            "goal_admissions": cov.goal_admissions,
            "perceive_dispatches": cov.perceive_dispatches,
            "judge_calls": cov.judge_calls,
            "incidents": cov.incidents,
            "source_coverage": {"audio": tracker.coverage.as_dict()},
        }

    async def report_zero_wake(*, at_end: bool) -> None:
        """Emit the latched ``zero_wake_coverage`` diagnostic if this run's gate never woke."""
        if on_health is None or onset is None or not witness.due(at_end=at_end):
            return
        await _call_sink(
            on_health,
            RuntimeHealthEvent(
                "zero_wake_coverage",
                ts=witness.last_t,
                data=witness.payload(
                    detector=str(getattr(onset, "name", audio_gate)),
                    high=getattr(onset, "high", None),
                    low=getattr(onset, "low", None),
                    at_end=at_end,
                ),
            ),
        )

    normal_completion = False
    if on_health is not None:
        await _call_sink(
            on_health,
            RuntimeHealthEvent(
                "runtime_started",
                data={
                    "source": gate_source,
                    "modality": "audio",
                    "timebase": resolved_timebase,
                    "goal_ids": tuple(condition_by_goal),
                },
            ),
        )
        await _call_sink(
            on_health,
            RuntimeHealthEvent("judge_dispatch", data={"mode": judge_dispatch}),
        )
    try:
        async for ingress in _audio_aiter(
            audio_source, timebase=timebase, observed_clock=wall_clock
        ):
            # Evaluated ONCE per unit, short-circuit exactly as before (`paused()` is still only
            # consulted when the muted term is falsy) — a caller's predicate must not be polled an
            # extra time just because the runtime now names which switch swallowed the audio.
            muted_now = muted is not None and muted()
            paused_now = (
                (not muted_now) and paused is not None and paused()
            )
            is_suppressed = bool(muted_now or paused_now)
            suppression_reason = (
                "muted" if muted_now else "paused" if paused_now else "unknown"
            )
            if is_suppressed != suppressed:
                core.advance_source_continuity(gate_source)
                if scheduler is not None:
                    scheduler.invalidate_source_continuity(("audio_suppressed",))
                window.clear()
                last_admit.clear()
                wake_at = None
                reset = getattr(onset, "reset", None)
                if callable(reset):
                    reset()
                # AUDIO-SUPPRESSED-NO-RECEIPT: the EDGES of a suppression window are health events,
                # not the units inside it. A muted mic offers ~4 chunks/s, so a per-unit event would
                # bury the sink for the whole time someone holds the mute button; the per-unit fact
                # is carried by the counter and by source_coverage.excluded instead, and the closing
                # event reports the window's exact size.
                if on_health is not None:
                    if is_suppressed:
                        await _call_sink(
                            on_health,
                            RuntimeHealthEvent(
                                "audio_suppression_started",
                                data={
                                    "reason": suppression_reason,
                                    "source": gate_source,
                                },
                            ),
                        )
                    else:
                        await _call_sink(
                            on_health,
                            RuntimeHealthEvent(
                                "audio_suppression_ended",
                                data={
                                    "suppressed_chunks": suppressed_run,
                                    "source": gate_source,
                                },
                            ),
                        )
                if not is_suppressed:
                    suppressed_run = 0
            if is_suppressed:
                suppressed = True
                suppressed_run += 1
                cov.audio_suppressed += 1
                # A suppressed unit IS an excluded observation: the source offered it and this
                # runtime declined it. `SourceTracker.exclude` is the documented consumer-level hook
                # for exactly that, and routing through it keeps `source_coverage.excluded` equal to
                # everything that did not become an observation.
                tracker.exclude("audio_suppressed")
                continue
            suppressed = False
            event_t = None
            candidate = ingress.unit
            try:
                submitted = ingress.unit
                if submitted is not None:
                    observed = (
                        ingress.observed_time
                        if ingress.observed_time is not None
                        else time.monotonic()
                    )
                    event_t = (
                        submitted.t_end
                        if resolved_timebase == "media"
                        else observed
                    )
                    source_time = submitted.t_end
                    candidate = replace(submitted, event_t=float(event_t))
                    canonical = canonicalize_audio_unit(
                        candidate,
                        capability=capability,
                        source_id="audio",
                        source_epoch=0,
                        source_sequence=carrier_sequence,
                        observed_time=observed,
                        source_time=source_time,
                    )
                    envelope = audio_unit_envelope(
                        canonical,
                        source_id="audio",
                        sequence=carrier_sequence,
                        observed_time=observed,
                    )
                else:
                    envelope = ingress.envelope
                    if envelope is None:
                        item = ingress.legacy_item
                        if type(item) is not tuple or len(item) != 3:
                            raise ValueError(
                                "legacy audio input must be an exact (t, AudioFrame, energy) tuple"
                            )
                        item_t, raw_frame, energy = item
                        observed = (
                            ingress.observed_time
                            if ingress.observed_time is not None
                            else time.monotonic()
                        )
                        source_time = (
                            float(item_t)
                            if ingress.legacy_timebase == "source" else None
                        )
                        event_t = (
                            source_time
                            if resolved_timebase == "media" else
                            float(item_t)
                            if ingress.legacy_timebase == "observed" else observed
                        )
                        if resolved_timebase == "media" and source_time is None:
                            raise ValueError("source_time_unknown")
                        canonical = canonicalize_audio_ingress_frame(
                            raw_frame,
                            event_t=float(event_t),
                            seq=carrier_sequence,
                            energy=energy,
                            capability=capability,
                            source_id="audio",
                            source_epoch=0,
                            source_sequence=carrier_sequence,
                            observed_time=observed,
                            source_time=source_time,
                        )
                        envelope = SourceEnvelope(
                            source_id="audio",
                            modality="audio",
                            epoch=0,
                            sequence=carrier_sequence,
                            observed_time=observed,
                            source_time=source_time,
                            format=SourceFormat(
                                media_type="audio/pcm",
                                sample_rate=canonical.sample_rate,
                                channels=canonical.channels,
                                sample_format=canonical.sample_format,
                            ),
                            payload=canonical.chunk,
                            signal=canonical.energy,
                            audio_direction=canonical.direction,
                            participant_id=canonical.participant_id,
                            legacy_timebase=ingress.legacy_timebase,
                        )
                    else:
                        if type(envelope) is not SourceEnvelope:
                            raise ValueError(
                                "compiled audio ingress requires an exact SourceEnvelope"
                            )
                        if envelope.modality != "audio":
                            raise ValueError("source_modality_mismatch")
                        source_time = envelope.source_time
                        if resolved_timebase == "media" and source_time is None:
                            raise ValueError("source_time_unknown")
                        event_t = (
                            source_time
                            if resolved_timebase == "media" else envelope.observed_time
                        )
                        canonical = canonicalize_audio_ingress_frame(
                            envelope.payload,
                            event_t=float(event_t),
                            seq=envelope.sequence,
                            energy=envelope.signal,
                            capability=capability,
                            source_id=envelope.source_id,
                            source_epoch=envelope.epoch,
                            source_sequence=envelope.sequence,
                            observed_time=envelope.observed_time,
                            source_time=source_time,
                            direction=envelope.audio_direction or "unknown",
                            participant_id=envelope.participant_id,
                        )
                        envelope = replace(
                            envelope,
                            payload=canonical.chunk,
                            signal=canonical.energy,
                            format=SourceFormat(
                                media_type="audio/pcm",
                                sample_rate=canonical.sample_rate,
                                channels=canonical.channels,
                                sample_format=canonical.sample_format,
                            ),
                        )
                require_source_time = (
                    resolved_timebase == "media" and submitted is None
                )
                source_preview = tracker.preview(
                    envelope, require_source_time=require_source_time
                )
                if not source_preview.accepted:
                    raise ValueError(f"audio carrier excluded: {source_preview.events!r}")
                unit_preview = window.preview_unit(
                    canonical, reset=source_preview.resets_continuity
                )
                if not unit_preview.accepted:
                    raise ValueError(
                        f"audio unit admission rejected: {unit_preview.as_dict()!r}"
                    )
            except Exception as exc:  # noqa: BLE001 - isolate one malformed carrier
                cov.audio_rejected += 1
                # The rejection decision was taken on a PROBE (`tracker.preview`), which by contract
                # mutates no coverage, and several of the raises above happen before the tracker is
                # consulted at all — so without this the refusal receipted as a health event but left
                # `source_coverage.excluded` at 0. Record it on the real tracker so observed +
                # excluded accounts for every offered unit.
                tracker.exclude("audio_unit_rejected")
                if on_health is not None:
                    await _call_sink(
                        on_health,
                        RuntimeHealthEvent(
                            "audio_unit_rejected",
                            ts=float(event_t) if isinstance(event_t, (int, float)) else None,
                            data={"error": repr(exc), "seq": getattr(candidate, "seq", None)},
                        ),
                    )
                continue

            inspection = tracker.inspect(
                envelope, require_source_time=require_source_time
            )
            continuity_reset = inspection.resets_continuity or unit_preview.discontinuous
            if continuity_reset:
                core.advance_source_continuity(gate_source)
                if scheduler is not None:
                    scheduler.invalidate_source_continuity(
                        inspection.events or ("audio_unit_discontinuity",)
                    )
                window.clear()
                last_admit.clear()
                wake_at = None
                reset = getattr(onset, "reset", None)
                if callable(reset):
                    reset()
            pushed = window.push_unit(canonical)
            if not pushed.accepted:
                raise RuntimeError("audio preview/commit admission mismatch")
            carrier_sequence += 1
            cov.audio_chunks += 1
            cov.last_media_t = canonical.source_time
            cov.last_wall_t = canonical.observed_time
            at = float(event_t)

            async for incident in drain_completed(at):
                yield incident
            await _drain_gate_health(core, on_health, at)
            current_revision = getattr(core.goals, "revision", goals_revision)
            if current_revision != goals_revision:
                if not regate_on_goal_change:
                    raise ValueError(
                        "audio-only goals changed while regate_on_goal_change=False"
                    )
                conditions, bindings = preflight()
                condition_by_goal = {item.goal_id: item for item in conditions}
                goals_revision = current_revision

            goal_ids = tuple(
                goal.id for goal in source_goals() if goal.id in condition_by_goal
            )
            for goal_id in goal_ids:
                if goal_id in deadline_bound:
                    continue
                goal = core.goals.get(goal_id)
                duration = goal.state.get("deadline_after_s") if goal is not None else None
                if (
                    goal is not None
                    and goal.state.get("rearm") == "deadline"
                    and isinstance(duration, (int, float))
                    and not isinstance(duration, bool)
                    and math.isfinite(duration)
                    and duration >= 0
                    and resolved_timebase == "media"
                ):
                    goal.state["deadline_at"] = at + float(duration)
                    goal.state["deadline_clock_domain"] = "audio_event"
                    deadline_bound.add(goal_id)
            # Every accepted quiet or active unit advances carrier-independent cognition.
            for callback, kwargs in (
                (
                    getattr(core, "check_episode_boundary", None),
                    {"t": at, "inactivity_s": episode_inactivity_s, "goal_ids": goal_ids},
                ),
                (
                    getattr(core, "check_observability", None),
                    {"t": at, "staleness_s": coverage_staleness_s, "goal_ids": goal_ids},
                ),
                (
                    getattr(core, "check_deadlines", None),
                    {"t": at, "goal_ids": goal_ids},
                ),
            ):
                if callback is None:
                    continue
                result = callback(**kwargs)
                if inspect.isawaitable(result):
                    result = await result
                async for incident in emit_fires(result, at):
                    yield incident

            boundaries = onset.push(at, canonical.energy) if onset is not None else ()
            if onset is not None:
                witness.observe(
                    at,
                    canonical.energy,
                    wakes=len(boundaries),
                    rearm=getattr(onset, "low", None),
                )
                # Mid-run, latched: a live audio watch may never reach `runtime_ended`, and a gate
                # that has heard a minute of audio without waking once is a fact the caller should
                # get NOW, not whenever the process happens to stop.
                await report_zero_wake(at_end=False)
            if boundaries:
                wake_at = at + float(audio_settle_s)
            wake_due = wake_at is not None and at >= wake_at
            if wake_due:
                wake_at = None
            civil = clock() if clock is not None else None
            admitted: list[str] = []
            for goal_id in goal_ids:
                condition = condition_by_goal[goal_id]
                if not schedule_active(condition.schedule, civil):
                    continue
                cadence_ready = (
                    condition.cadence_s is None
                    or at - last_admit.get(goal_id, float("-inf")) >= condition.cadence_s
                )
                if cadence_ready and (condition.gate_on == "fixed" or wake_due):
                    admitted.append(goal_id)
            if not admitted:
                continue

            by_id = {goal.id: goal for goal in source_goals()}
            evidence: dict[str, tuple[AudioUnit, ...]] = {}
            for goal_id in admitted:
                seconds = program_of(by_id[goal_id]).perception.audio_s
                units = window.clip_units(seconds or window_seconds, now=canonical.t_end)
                if units:
                    selected = tuple(units)
                    provider_pcm_cap = bindings[goal_id].capability.max_provider_bytes - 44
                    held = 0
                    keep_from = len(selected)
                    for index in range(len(selected) - 1, -1, -1):
                        size = len(selected[index].chunk.pcm)
                        if held + size > provider_pcm_cap:
                            break
                        held += size
                        keep_from = index
                    if keep_from:
                        selected = selected[keep_from:]
                        if on_health is not None:
                            await _call_sink(
                                on_health,
                                RuntimeHealthEvent(
                                    "audio_evidence_capped",
                                    ts=at,
                                    data={
                                        "goal_id": goal_id,
                                        "dropped_units": keep_from,
                                        "provider_bytes_cap": bindings[
                                            goal_id
                                        ].capability.max_provider_bytes,
                                    },
                                ),
                            )
                    # Validate the complete aggregate provider payload before admission cadence,
                    # spend counters, or scheduler state can be mutated.
                    audio_frame_from_units(
                        selected, capability=bindings[goal_id].capability
                    )
                    evidence[goal_id] = selected
            admitted_ids = tuple(goal_id for goal_id in admitted if goal_id in evidence)
            if not admitted_ids:
                continue
            for goal_id in admitted_ids:
                last_admit[goal_id] = at
            cov.escalations += 1
            cov.goal_admissions += len(admitted_ids)
            if scheduler is not None:
                scheduler.submit(at, evidence, admitted_ids)
            else:
                cov.judge_calls += 1
                cov.perceive_dispatches += 1
                perceive_audio = getattr(core, "perceive_audio_judged")
                fires = await perceive_audio(
                    evidence,
                    gate_source,
                    t=at,
                    expected_goals_revision=goals_revision,
                    expected_source_generation=core.source_continuity_generation(gate_source),
                    goal_ids=admitted_ids,
                )
                async for incident in emit_fires(fires, at):
                    yield incident
        normal_completion = True
    except BaseException as exc:
        if on_health is not None and not isinstance(
            exc, (asyncio.CancelledError, GeneratorExit)
        ):
            await _call_sink(
                on_health,
                RuntimeHealthEvent("runtime_error", data={"error": repr(exc)}),
            )
        raise
    finally:
        if scheduler is not None:
            await scheduler.close(drain=normal_completion)
        if not normal_completion and on_health is not None:
            await report_zero_wake(at_end=True)
            await _call_sink(
                on_health, RuntimeHealthEvent("runtime_ended", data=coverage_data())
            )
    if scheduler is not None:
        async for incident in drain_completed(
            cov.last_media_t if cov.last_media_t is not None else 0.0
        ):
            yield incident
    if on_health is not None:
        # Immediately BEFORE runtime_ended, mirroring the video path's zero_admission_coverage.
        await report_zero_wake(at_end=True)
        await _call_sink(
            on_health,
            RuntimeHealthEvent(
                "runtime_ended",
                data=coverage_data(),
            ),
        )


async def _run_worker(
    core: Any = None,
    *,
    source: Any,
    goals: Any = None,
    vision: Any = None,
    on_incident: Callable | None = None,
    on_ask: Callable | None = None,
    on_frame: Callable | None = None,
    on_health: Callable | None = None,
    on_trace: Callable | None = None,
    fps: float = 4.0,
    source_queue_size: int = 64,
    source_reconnects: int = 0,
    source_reconnect_backoff_s: float = 0.5,
    source_rtsp_tcp: bool = True,
    snippet_pre_s: float = 5.0,
    snippet_size: int = 48,
    snippet_max_bytes: int | None = None,
    snippet_kind: str = "frame",
    sink_errors: str = "raise",
    stats: RuntimeStats | None = None,
    gate_source: str = "camera",
    episode_inactivity_s: float | None = None,
    coverage_staleness_s: float | None = None,
    audio_source: Any = None,
    audio_gate: Any = "acoustic-onset",
    audio_wake: dict | None = None,
    audio_settle_s: float = 1.5,
    on_utterance: Callable | None = None,
    allow_scripted_brain: bool = False,
    vad: Any = None,
    utterance_threshold: float = 0.01,
    utterance_hangover_frames: int = 8,
    utterance_min_speech_frames: int = 3,
    utterance_max_s: float = 60.0,
    audio_window_s: float | None = None,
    deaf_until: Callable[[], float] | None = None,
    stale_audio_s: float | None = None,
    wall_clock: Callable[[], float] = time.monotonic,
    wake_policy: Any = None,
    structural_adapter: Callable | None = None,
    semantic_adapter: Callable | None = None,
    energy_provider: Callable | None = None,
    regate_on_goal_change: bool = True,
    paused: Callable | None = None,
    muted: Callable | None = None,
    clock: Callable | None = None,
    persistence: Any = None,
    restore: bool = False,
    autosave_every_s: float | None = None,
    label_store: Any = None,
    disposition_log: Any = None,
    judge_dispatch: str = "async",
    metrics_collector: Any = None,
    judge_stall_after_s: float = 20.0,
    judge_backpressure_after: int = 3,
    judge_decision_deadline_s: float | None = None,
    timebase: str = "auto",
    coverage_floor: float | None = None,
    _interest_tracker: LiveInterestTracker | None = None,
    _interest_subscription: InterestSubscription | None = None,
) -> AsyncIterator:
    """Drive a live ``source`` through the gate + brain, yielding an :class:`Incident` on each rising edge.

    - ``on_incident(incident)`` — the notification sink (SMS / Slack / webhook lives here). Wrapped
      in an :class:`~percept.runtime.sinks.IdempotentSink` bound to a bounded, restart-
      surviving :class:`~percept.runtime.sinks.IdempotencyLedger` keyed by ``incident.id`` (the
      event_id) — a redelivery of an event_id already delivered to terminal-OK is deduped (the sink is
      not called again), while a retry of an in-flight/failed one still reaches it normally.
    - ``on_ask(event)`` — optional refusal sink for low-confidence reads; asks are not incidents.
    - ``on_frame(t, judge_frame)`` — the live-feed hook (wire an MJPEG streamer here).
    - ``on_health(event)`` — optional structured health/telemetry hook.
    - **Behavior routes are per goal, and so is the preflight.** This worker drives the frame routes
      (``watch``/``transition``/``process``) plus ``rhythm``. A goal on a HOST-driven route
      (``count`` → :meth:`~percept.agent.Percept.perceive_boundary`, ``event`` → ``push_frame`` +
      ``perceive_events``, ``remind`` → ``schedule_every`` + ``tick``) armed on this same source key
      is SKIPPED, not fatal: it never reaches tier-0, never spends a judge, and never fires from
      frames, while every other goal on the source runs normally. The skip is reported once per goal
      per run as a :class:`~percept.behavior_routes.UndrivenBehaviorRouteWarning`, a
      ``percept.runtime.worker`` log line, and an ``undriven_behavior_route`` health event, each
      naming the goal and the exact host call. A deployment that wants an undriven route to abort the
      run filters that warning to ``"error"``. Only a route nothing can drive
      (:attr:`~percept.behavior_routes.BehaviorRoute.UNSUPPORTED`) still raises
      :class:`~percept.behavior_routes.UnsupportedBehaviorRouteError` before the source is pulled.
    - ``source`` — captured and pushed inputs share the versioned
      :class:`~percept.runtime.envelope.SourceEnvelope` contract. Duplicate, reordered, corrupt,
      unsupported-version, or reset-without-epoch items are excluded before the Watch gate. Sequence
      gaps, reconnects, and format changes accept the current item but reset temporal perception state
      and report ``coverage_effect="observed_after_discontinuity"``.
    - ``source_*`` — capture queue/reconnect/RTSP options for URI sources.
    - ``snippet_pre_s`` — seconds of pre-roll captured into each incident's snippet.
    - ``snippet_max_bytes`` — BYTE budget for that pre-roll ring (default ``None`` →
      ``settings.events.preroll_max_bytes``, 128 MB; ``0`` disables the byte bound). ``snippet_size``
      bounds the ring by COUNT, which does not bound memory: the same 48-frame ring holds ~0.4 MB of
      8x8 test frames and ~300 MB of 1080p RGB, and profiling named this ring as the whole
      of the resident-memory plateau clients read as a leak. When the budget is exceeded the OLDEST
      frames are evicted (the newest is never dropped) and each eviction emits a ``preroll_capped``
      health event carrying ``reason``/``frame_bytes``/``held_bytes`` and bumps
      ``stats.preroll_frames_dropped`` — a shorter snippet, announced, never silently shorter.
    - ``snippet_kind`` — ``"frame"`` keeps decoded frames; ``"clip"`` emits a portable encoded bundle.
    - ``sink_errors`` — ``"raise"`` (default) or ``"ignore"`` for notification/feed hook failures.
    - ``stats`` — optional mutable :class:`RuntimeStats` counters for health dashboards.
    - ``audio_source`` — optional mic/wav/video-container URI (``"mic:0"``, ``"clip.wav"``,
      ``"clip.mp4"``) **or** an (async) iterable of ``(t, chunk, energy)`` (see
      :func:`percept.runtime.audio.open_audio_source`). When set, the worker pumps each chunk into
      ``core.push_audio`` — **filling the ``AudioWindow`` that ``modality="fused"`` goals judge
      over**. A compiled fused Watch with no exact retained audio fails closed; only legacy/manual
      callers keep compatibility behavior. With ``source=None``, ``audio_source`` is the sole carrier
      for compiled ``sense="audio"`` Watches: every accepted AudioUnit advances cognition, while its
      onset/fixed policy admits exact retained evidence without a synthetic video frame. It also feeds ``energy``
      to the gate, so an off-camera sound (a cry, glass breaking) can escalate a frame the visual motion
      gate would skip. Default ``None`` → the video-only path is byte-for-byte, UNLESS ``source`` is
      itself a video container (``.mp4``/``.mov``/``.mkv``/``.webm``) AND an active goal's modality is
      ``"fused"``/``"audio"`` — then ``audio_source`` is auto-derived as ``source`` (its own audio track,
      demuxed via ffmpeg) so a caller watching an mp4 with a fused goal needs no extra parameter. A
      vision-only goal set never triggers this (no ffmpeg subprocess spawned when nothing wants audio),
      and an explicit ``audio_source`` always wins over the auto-derivation.
    - ``audio_gate`` — the tier-0 audio detector: a name (``"acoustic-onset"``), an instance with
      ``push(t, v) -> [Boundary]``, or ``None`` to only fill the window + feed energy (no onset wake).
    - ``audio_wake`` — kwargs for the mean-square-energy onset detector, e.g.
      ``{"high": 0.3, "low": 0.15, "refractory_s": 5.0}``. The public values remain legacy
      RMS/amplitude thresholds and are squared internally for the mean-square energy stream.
    - ``audio_settle_s`` — seconds to WAIT after an acoustic onset before the woken judge runs. An
      onset marks the START of a sound, and the audio clip is sliced backward from the judge moment —
      judging at the onset itself would hand the model the silence BEFORE the sound. The settle keeps
      the wake but delays the admission so the clip actually contains the sound. ``0.0`` = old behavior.
    - ``on_utterance(audio, t)`` — the **voice-escalation seam**. When the audio pump endpoints a complete
      spoken utterance (energy rose, then fell for ``utterance_hangover_frames``), the utterance's PCM clip
      (an :class:`~percept.contracts.AudioFrame`) is handed here — the hook the voice layer turns into an
      ephemeral reply. Percept owns the mic + the "when is a turn done" decision; the app owns the reply.
      Default ``None`` → no endpointing, the audio path is byte-for-byte. ``vad`` overrides the endpointer
      (any object with ``push(energy) -> event.{started,endpoint}``); else an :class:`EnergyVad` built from
      the ``utterance_*`` knobs is used.

      **READ THE UNITS ON THE ``utterance_*`` KNOBS — they are not the units they look like.**
      ``utterance_threshold`` (default ``0.01``) is an RMS AMPLITUDE and is SQUARED before it reaches
      the endpointer, so it is really energy ``1e-4``. ``utterance_hangover_frames`` (default ``8``) is
      a FRAME COUNT, one frame per capture chunk — at ``AudioSourceConfig.chunk_s=0.25`` that is
      **2.00 seconds of continuous silence** before a turn is declared finished, which continuous
      narration essentially never contains (measured: endpoints stayed at or below 4 across every
      ``utterance_threshold`` from 0.001 to 0.3, so the threshold cannot compensate). Express it in
      seconds with :func:`percept.runtime._vad.hangover_frames_for` — e.g.
      ``utterance_hangover_frames=hangover_frames_for(0.6)`` for conversational endpointing.

      **The utterance is reconstructed from the endpointer's OWN chunk buffer** (accumulated started→
      endpoint), NOT sliced from the shared rolling AudioWindow — so a turn longer than the fused window
      is handed WHOLE, not truncated to the window's last N seconds. ``utterance_max_s`` (default
      ``60``) bounds that buffer so a stuck-open mic can't grow it without limit; a turn exceeding the cap
      is handed capped AND ``on_health("utterance_truncated")`` fires (never a silent drop).
    - ``audio_window_s`` — size (seconds) of the shared AudioWindow the FUSED/audio judge reads (distinct
      from the utterance buffer above). Default ``None`` → the ``settings.events.audio_window_s`` default,
      except a voice-enabled session (``on_utterance`` set) raises it to ``≥30`` so fused judges get real
      acoustic context. Byte-for-byte the settings default when no ``audio_source`` is wired.

      **The handler runs OFF the pump** (spawned as a tracked task, cancelled on exit): a handler that
      takes real time — STT + a voice turn + playback — must not block capture, or the backlog it creates
      is replayed through the endpointer the instant it returns and Percept endpoints its own reply
      (self-echo). Two utterances can therefore overlap; the runtime does NOT serialize them — the app
      owns concurrency (guard at capture time, e.g. a ``speaking`` flag). ``on_utterance`` opens an
      **un-AEC'd** mic and hands raw PCM; if the app plays audio into the same room it MUST call
      ``deaf_until``/``session.deafen(...)`` around playback or the mic will hear it. For a speakers-on,
      always-on deployment wire a voice engine app-side (see ``templates/realtime_voice_watch_agent``,
      whose transport owns AEC) instead of this un-AEC'd mic path.
    - ``deaf_until`` — optional ``() -> deadline`` (in the ``wall_clock`` timebase). While
      ``wall_clock() < deaf_until()`` the endpointer is suppressed (the app is about to / is speaking):
      chunks still fill the AudioWindow (fused judging is unaffected) but are not fed to the VAD, and the
      VAD is reset so a half-built utterance cannot straddle the window. Turn-taking, not perception —
      checked immediately before the VAD, never above ``push_audio``. Default ``None`` → never deafened.
    - ``stale_audio_s`` — optional staleness guard for the VAD path. When the pump has fallen behind
      real time (a slow turn outran the drop-oldest capture queue) draining the backlog would replay
      minutes-old audio through the endpointer as if live. When pump lag (wall-elapsed minus stream-elapsed
      since the first chunk) exceeds this, the chunk is dropped from the VAD path and a
      ``RuntimeHealthEvent("audio_stale")`` is emitted — it is still ``push_audio``'d (the AudioWindow
      wants the history). Default ``None`` → off (byte-for-byte).
    - ``wall_clock`` — monotonic time source for ``deaf_until`` + ``stale_audio_s`` (default
      :func:`time.monotonic`; injectable for tests). Distinct from ``clock`` (civil time for schedules).

    - ``structural_adapter`` / ``semantic_adapter`` — the TIER-0 EXTENSION SEAM: ``(frame) ->
      float in 0..1`` callables that fill the ``structural`` / ``semantic`` channels of the
      :class:`~percept.adapters.Tier0Signal` this run reports, instead of the motion-only ``0.0``.
      This is how you supplement tier 0 with a cheap local model — a CLIP/embedding drift score, a pose
      or privacy classifier, an OCR-diff "the document changed" score — WITHOUT bypassing ``run()`` and
      driving :class:`~percept.edge.tier0.GatedWatchLoop` yourself, which was previously the only way.
      They reach admission two ways: a goal whose condition compiles to ``gate_on="structural"`` /
      ``"semantic"`` escalates on that channel against its own threshold, and a ``wake_policy`` fuses the
      channels into one wake score. Both are ADMISSION signals — they decide *when to spend the cloud
      judge*; they do NOT deliver facts to the judge (see ``docs/reference/registration.md``). Called
      once per frame on the gate frame, so keep them cheap, and keep them total — an adapter that raises
      propagates out of the gate and stops the watch (loudly, not silently). Both default ``None`` → the
      reported signal and every escalation decision are byte-for-byte the motion-only path.
    - ``label_store`` — optional :class:`~percept.labelstore.LabelStore`. When set, every yielded
      incident is bound to it, so ``incident.record_label(good, ...)`` writes a LOCAL
      :class:`~percept.contracts.LabelRecord` (the feedback intake). Default ``None`` → ``record_label``
      raises (nothing bound); the firing path is byte-for-byte either way.
    - ``disposition_log`` — optional :class:`~percept.labelstore.DispositionLog`. When set, every
      gate ``FIRE``/``ASK`` appends a ``{incident_id, verdict_ref, disposition, ts}`` row (the escalation
      meter's raw feed, distinct from ``Verdict.usage``). Default ``None`` → nothing written, byte-for-byte.
    - ``judge_dispatch`` — how an admitted frame's cloud judgment is driven. ``"async"`` (default)
      hands it to a single-flight :class:`_JudgeScheduler` (one in flight + one latest-pending; the
      capture loop NEVER awaits a network judgment — one slow VLM turn no longer stalls capture, the
      gate, or the absence checks). ``"inline"`` awaits ``perceive_judged`` at the admit site —
      byte-for-byte the historical loop, kept for one release as the escape hatch for consumers that
      depend on fires being emitted before the next frame is read. The admit condition is unchanged
      either way; in async mode fires trail the admitting frame by ≤1 frame interval + the judge
      duration (``incident.ts`` stays the admission ``t``), a stale in-flight judgment aborts when
      goals change mid-flight (see ``Percept.perceive_judged`` ``expected_goals_revision``), and
      ``stats.judge_superseded`` / ``stats.judge_revision_dropped`` count the frames
      overload/goal-changes discarded. A ``RuntimeHealthEvent("judge_dispatch")`` announces the mode
      once at start; escalation health events carry live ``judge_active``/``judge_pending`` (each
      0/1) in async mode. The scheduler also degrades LOUD — ``"judge_scheduler_stalled"``
      (the active judgment has run past ``judge_stall_after_s``, a hang) and
      ``"judge_scheduler_backpressure"`` (the pending slot has been superseded
      ``judge_backpressure_after`` times in a row — the judge is slower than the frame rate), each
      latched to fire once, paired with one ``"judge_scheduler_recovered"`` the next time a judgment
      completes cleanly. Async-mode only; ``"inline"`` has no scheduler to degrade.
    - ``judge_stall_after_s`` / ``judge_backpressure_after`` — thresholds for the two health events
      just above (async mode only); the defaults are generous enough that a normal run never trips
      either.
    - ``judge_decision_deadline_s`` — optional absolute async-lane budget measured from admission,
      including pending-slot queue time and provider dispatch. Expired pending work is dropped before
      launch; active work is cancelled and ignored at expiry. Both outcomes emit structured health
      events and counters. A provider that suppresses cancellation retains the one physical lane until
      it really terminates, and its terminal latch prevents a late audio result from mutating cognition.
      ``None`` preserves deliberate/historical behavior; a reflex deployment
      should set this independently of the provider adapter's socket timeout.

    ADMISSION COVERAGE. Every run closes with ``RuntimeHealthEvent("runtime_ended")`` carrying this
    run's counters — including exact fixed/per-goal admission, dispatch, judge, and incident counts —
    so a consumer that kept no state of its own can still say what the session did. When a run observed
    frames but admitted NONE of them, a ``RuntimeHealthEvent("zero_admission_coverage")`` is emitted immediately
    BEFORE it, carrying the same counts plus the ``threshold`` and salience ``stat`` the frames were
    judged against, PLUS ``coverage_floor`` and ``floor_unmet`` (below). By default that event states
    a FACT, not a degradation verdict: a still camera under a motion-gated watch correctly admits
    nothing and saves every model call, which is the point of tier-0 — so with no declared floor it
    stays diagnostic, exactly the way ``percept doctor`` reports ``ZERO_ADMITS_ON_FIXTURE``. If the
    scene was NOT quiet, the eye is mis-selected — ``percept doctor`` will show which statistic would
    have admitted.
    ``runtime_ended.data["source_coverage"]`` separately reports, per modality, accepted observations,
    excluded inputs, accepted discontinuities, and sequence positions known to be missing. These are
    ingress facts and do not imply that a Watch was admitted to a judge.
    - ``coverage_floor`` — an explicit, OPT-IN minimum admission count this run declares it must
      clear (e.g. sourced from a :class:`~percept.sourceprofile.SourceProfile`'s
      ``parameters['coverage_floor']` via :mod:`percept.runtime_compiler`). ``None`` (default)
      declares nothing — ``zero_admission_coverage`` is then BYTE-FOR-BYTE the undeclared-floor
      behavior above: not in :data:`percept.tracing.DEGRADED_HEALTH_KINDS`'s reach, never latches an
      unhealthy interval. When a floor IS declared and this run's admissions fall short of it, the
      SAME event kind carries ``floor_unmet=True`` and :meth:`percept.tracing.TraceRecorder.health_sink`
      classifies that occurrence as ``state="degraded"`` — a declared expectation that was not met,
      not an inferred one (ADR-008 §Corrections(b): "must be explicit, opt-in and declared... never
      inferred from an observed zero").
    - ``metrics_collector`` — optional :class:`~percept.harness.metrics.MetricsCollector`.
      When set (``judge_dispatch="async"`` only — the inline path has no queue to measure), every
      admission into the :class:`_JudgeScheduler`'s active/pending slot is tracked via
      ``record_admitted``/``record_admission_resolved``, so ``collector.queue_depth_series()`` /
      ``max_oldest_pending_age_s()`` reflect this run's real off-loop judge queue (bounded J<=1, P<=1
      by construction — see :class:`_JudgeScheduler`). Default ``None`` → no collector calls, byte-
      for-byte. Independent of ``core.funnel``'s ``seen``/``gated``/``judged``/``refused``/``fired``
      counters (:mod:`percept.observability`), which this run always emits when ``core`` exposes a
      ``.funnel`` (a bare duck-typed core without one is untouched).

    - ``timebase`` — which clock frame ``t`` (and therefore ``incident.ts``) is stamped on.
      ``"auto"`` (default) resolves per source: MEDIA time for a RECORDED source (a file — so an
      incident timestamp is meaningful against the media; a 36s clip decoded in 0.72s used to stamp
      every incident with the decode instant, which made the timestamps useless as audit evidence),
      WALL/monotonic for a LIVE device/stream (``webcam:0``, ``rtsp://…`` — those must share the
      agent's deadline clock). ``"media"``/``"wall"`` force the choice. A non-``str``
      source feeds its own explicit ``t`` and is never re-stamped, whatever this says.
      ``stats.last_media_t`` and ``stats.last_wall_t`` preserve the clocks the source envelope
      actually carried; either remains ``None`` when the source did not supply that clock.

    The capture rate this run actually opens with is RECORDED, never silently substituted —
    ``stats.effective_config`` (an :class:`EffectiveConfig`) carries ``requested_fps``/``effective_fps``
    and the ``reason`` (``brief-window-coverage`` — a brief-event plan needs the cadence;
    ``source-limit`` — a str source's rate is fixed at open, so the headroom cap is taken up front;
    ``profile-default`` — nothing overrode the request), the same record is mirrored onto
    ``core.effective_config`` and emitted once as an informational ``effective_config`` health event.
    An override is not by itself a warning; the loud channel stays reserved for a coverage requirement
    that is actually UNMET (``capture_plan_unmet``).

    Owns capture off the consumer thread; the per-goal cadence + the three-state gate decide *when* and
    *whether* to fire. Yields incidents lazily, so an ``async for incident in run(...)`` is the front door.
    """
    from percept.labelstore import verdict_ref as _verdict_ref
    from percept.behavior import program_of
    from percept.behavior_routes import (
        UndrivenBehaviorRouteWarning,
        drives_rhythm,
        goal_behavior_route,
        require_drivable_routes,
        route_remedy,
        skipped_route_message,
        worker_can_drive,
    )
    from percept.config import load_settings
    from percept.contracts import WatchSpec
    from percept.duration_routes import (
        UNSUPPORTED_DURATION_ROUTE,
        DurationRouteIssue,
        UnsupportedDurationRouteError,
        require_duration_routes,
    )
    from percept.edge.detectors import DetectorBank
    from percept.edge.tier0 import MotionGate, compile_watchspec
    from percept.rhythm import RhythmDrive

    # Convenience: `percept.run(source=..., goals=[...])` with no pre-built core. Build a Percept
    # here and arm each goal, so the one-line quickstart is real. Passing an explicit `core` (the
    # canonical form) takes precedence — and supplying `goals`/`vision` ALONGSIDE a core is
    # now a hard ValueError, not a silent discard. The old behaviour ("ignored when core is given") ran
    # a byte-identical watch that simply never armed the caller's goals: a silent failure that
    # consumes an avoidable debugging cycle. A warning would still permit that run, so this
    # refuses to start instead. Nothing publishes before 1.0 — this is the moment to make it loud.
    if core is not None and (goals or vision is not None):
        raise ValueError(
            "goals= and vision= cannot be supplied when core= is provided; "
            "apply goals to the core or omit core."
        )
    if core is None:
        if not goals:
            raise TypeError(
                "percept.run needs either a core (positional) or goals=[...]; got neither"
            )
        from percept.agent import Percept

        core = Percept.create(vision=vision) if vision is not None else Percept.create()
        for i, g in enumerate(goals):
            core.add_goal(_as_goal(g, i))

    # FATAL seam: a live voice-escalation path (on_utterance) behind a SCRIPTED brain is the
    # demo bug — every spoken "watch X" is silently dropped. Refuse it here, where the language path is
    # actually wired. A code-armed watch (no on_utterance) with a scripted conductor is fine.
    if on_utterance is not None:
        from percept.agent import _guard_live_language_path

        _guard_live_language_path(core, allow_scripted_brain=allow_scripted_brain)

    if source is None:
        unsupported_audio_only = [
            name
            for name, enabled in (
                ("on_frame", on_frame is not None),
                ("on_utterance", on_utterance is not None),
                ("persistence", persistence is not None),
                ("restore", bool(restore)),
                ("autosave_every_s", autosave_every_s is not None),
                ("vad", vad is not None),
                ("deaf_until", deaf_until is not None),
                ("stale_audio_s", stale_audio_s is not None),
                ("structural_adapter", structural_adapter is not None),
                ("semantic_adapter", semantic_adapter is not None),
                ("energy_provider", energy_provider is not None),
                ("wake_policy", wake_policy is not None),
                ("coverage_floor", coverage_floor is not None),
                ("disposition_log", disposition_log is not None),
                ("label_store", label_store is not None),
            )
            if enabled
        ]
        if unsupported_audio_only:
            raise ValueError(
                "source=None audio-only execution does not support these video/voice/persistence "
                f"options: {', '.join(unsupported_audio_only)}"
            )
        if _interest_subscription is not None or _interest_tracker is not None:
            raise ValueError(
                "InterestSubscription currently reports video VID admission only and is not "
                "supported by source=None audio-only execution"
            )
        previous_trace = getattr(core, "on_trace", None)
        if on_trace is not None:
            core.on_trace = on_trace
        try:
            async for incident in _run_audio_only(
                core,
                audio_source=audio_source,
                gate_source=gate_source,
                audio_gate=audio_gate,
                audio_wake=audio_wake,
                audio_settle_s=audio_settle_s,
                audio_window_s=audio_window_s,
                judge_dispatch=judge_dispatch,
                metrics_collector=metrics_collector,
                judge_stall_after_s=judge_stall_after_s,
                judge_backpressure_after=judge_backpressure_after,
                judge_decision_deadline_s=judge_decision_deadline_s,
                timebase=timebase,
                wall_clock=wall_clock,
                clock=clock,
                paused=paused,
                muted=muted,
                on_incident=on_incident,
                on_ask=on_ask,
                on_health=on_health,
                sink_errors=sink_errors,
                stats=stats,
                episode_inactivity_s=episode_inactivity_s,
                coverage_staleness_s=coverage_staleness_s,
                regate_on_goal_change=regate_on_goal_change,
            ):
                yield incident
        finally:
            core.on_trace = previous_trace
        return

    settings = getattr(core, "settings", None) or load_settings()

    def _source_key_goals() -> list:
        """EVERY active Goal bound to this worker's source key — driven here or not."""
        return [
            goal
            for goal in core.goals.active()
            if getattr(goal, "source", "camera") == gate_source
        ]

    def _source_goals() -> list:
        """The Watches on this source key that THIS worker drives, in registry order.

        A host-driven route (``count``/``event``/``remind``) bound to the same key is deliberately
        ABSENT: the frame loop has no driver for it, so including it would either abort the run (the
        old behaviour — one such goal killed every unrelated Watch beside it) or, worse, let it fall
        through into ``_visual_goals`` and spend a per-frame judge on a goal the host counts for free.
        It stays armed and its host API still fires it; ``_preflight_routes`` reports the skip.
        """
        return [
            goal for goal in _source_key_goals() if worker_can_drive(goal_behavior_route(goal))
        ]

    def _rhythm_goals() -> list:
        """Source Watches driven by the COUNTING path (``type="rhythm"``), never by a judge.

        These consume frames like any visual Watch, but the frame becomes a channel MEASUREMENT
        folded into a RhythmTracker — not a tier-0 admission decision and not a judge call. They are
        therefore excluded from every WatchSpec-compiling partition below (``compile_watchspec``
        emits no condition for them, and this worker asserts exactly one condition per compiled goal).
        """
        return [goal for goal in _source_goals() if drives_rhythm(goal_behavior_route(goal))]

    def _visual_goals() -> list:
        """Source Watches whose compiled perception actually consumes video frames FOR A JUDGE."""
        return [
            goal
            for goal in _source_goals()
            if not drives_rhythm(goal_behavior_route(goal))
            and program_of(goal).perception.modality in ("vision", "fused")
        ]

    def _external_only_audio_goals() -> list:
        """Audio Watches compiled for acoustic admission, never visual motion."""
        return [
            goal
            for goal in _source_goals()
            if not drives_rhythm(goal_behavior_route(goal))
            and program_of(goal).perception.modality == "audio"
        ]

    clear_pending_dwell = getattr(core, "clear_pending_dwell", None)

    def _require_duration_runtime() -> None:
        """Validate duration shape and its continuity hook for the CURRENT source goals."""
        source_goals = _source_goals()
        require_duration_routes(
            source_goals,
            driver="percept.run",
            require_worker_cadence=True,
        )
        duration_goal_ids = [
            str(goal.id)
            for goal in source_goals
            if isinstance(getattr(goal, "state", None), Mapping)
            and "hold_s" in goal.state
        ]
        if duration_goal_ids and not callable(clear_pending_dwell):
            raise UnsupportedDurationRouteError(
                (
                    DurationRouteIssue(
                        path="Runtime.clear_pending_dwell",
                        code=UNSUPPORTED_DURATION_ROUTE,
                        goal_id=goal_id,
                        message=(
                            "is required to invalidate sustained-duration evidence across "
                            "run and pause continuity boundaries"
                        ),
                    )
                    for goal_id in duration_goal_ids
                ),
                driver="percept.run",
            )

    # Structured health for a route this run is skipping, drained through on_health with the rest.
    _route_health: list[dict] = []
    _route_reported: dict[str, str] = {}  # goal_id -> route already reported this run

    def _preflight_routes() -> None:
        """Refuse the UNDRIVABLE, report (never abort on) the host-driven.

        Route support is source-local, and it is per GOAL: whether the frame loop can execute goal A
        says nothing about goal B. Refuse before this worker opens or pulls its source, and repeat
        during regating so a Goal armed mid-run cannot reach detection or judging unannounced.

        The split lives in :func:`~percept.behavior_routes.require_drivable_routes`. An UNSUPPORTED
        goal still raises here, before a frame is pulled — nothing can ever fire it. A host-driven
        goal (``count``/``event``/``remind``) is REPORTED and skipped instead: it has a complete,
        shipped driver that this loop simply is not, so aborting the run would take every unrelated
        Watch down with it. Reported once per goal per run on both channels the arm seam uses — the
        warning a test/interactive caller sees (escalate it to an error to restore the old fatal
        behaviour) and the log a 24/7 deployment reads — plus a structured health event.
        """
        for item in require_drivable_routes(_source_key_goals(), driver="percept.run"):
            if _route_reported.get(item.goal_id) == item.route.value:
                continue
            _route_reported[item.goal_id] = item.route.value
            message = skipped_route_message(item, driver="percept.run", source=gate_source)
            warnings.warn(message, UndrivenBehaviorRouteWarning, stacklevel=2)
            _log.warning("%s", message)
            _route_health.append({
                "goal_id": item.goal_id,
                "route": item.route.value,
                "behavior_type": item.authored_type_display,
                "source": gate_source,
                "driver": "percept.run",
                "remedy": route_remedy(item.route, goal_id=item.goal_id),
            })

    _preflight_routes()
    _require_duration_runtime()
    if callable(clear_pending_dwell):
        # Every run is a new observation session even when a caller reuses the same in-process core.
        clear_pending_dwell(gate_source)

    # A compiled SourceProfile coverage floor must not become an inert field. The existing health
    # contract is source/run-level, so it can faithfully inherit the floor for one Watch. With
    # multiple Watches, aggregate admissions could make one Watch appear covered by another's calls;
    # refuse that ambiguous topology until per-Watch coverage counters exist.
    if coverage_floor is None:
        bound_floors = [
            (goal.id, bound[0].coverage_floor)
            for goal in _source_goals()
            if (bound := getattr(core, "_watch_runtime_bindings", {}).get(goal.id)) is not None
            and bound[0].coverage_floor is not None
        ]
        if bound_floors and len(_source_goals()) != 1:
            raise ValueError(
                "compiled SourceProfile coverage_floor is per Watch, but this worker's coverage "
                "health counter is source-aggregate. Run the Watches on separate workers or pass "
                "an explicit source-level coverage_floor; refusing a false per-Watch guarantee."
            )
        if bound_floors:
            coverage_floor = bound_floors[0][1]

    # A SourceProfile timebase is a source-level constraint: one worker cannot interpret the same
    # timestamp as both media and wall time.  Honor the compiled value when run() stayed on "auto";
    # reject incompatible profiles instead of making arming order choose the clock silently.
    _caller_timebase_explicit = timebase != "auto"
    if timebase == "auto":
        bound_timebases = {
            bound[0].timebase
            for goal in _source_goals()
            if (bound := getattr(core, "_watch_runtime_bindings", {}).get(goal.id)) is not None
            and bound[0].timebase not in (None, "auto")
        }
        if len(bound_timebases) > 1:
            raise ValueError(
                "active Watches on one source require incompatible compiled timebases "
                f"{sorted(bound_timebases)}; run them on separate workers or select one timebase."
            )
        if bound_timebases:
            timebase = next(iter(bound_timebases))

    if judge_dispatch not in ("inline", "async"):
        raise ValueError(
            f"judge_dispatch must be 'inline' or 'async', got {judge_dispatch!r}"
        )
    # Non-blocking judge dispatch: in "async" mode the admit site routes to this single-flight
    # scheduler instead of awaiting the network judgment inline. ONE scheduler per run() — run drives
    # one source (the slot is per-source; a future multi-source runner would key a dict of these).
    # metrics_collector rides along so the scheduler can report queue depth / oldest-pending age.
    # Inline mode has no queue to measure, so the collector is simply unused there.
    # ADMISSION COVERAGE. A run that admitted NOTHING was previously indistinguishable from a
    # healthy quiet one: `runtime_ended` carried an empty payload. These are the SAME counters the
    # caller's `stats` already holds when it passed one — `_cov` IS `stats` in that case, so nothing
    # is double-counted; when the caller passed none we keep a throwaway so the end-of-run coverage
    # payload is still honest (and so the async scheduler's judge_calls land somewhere countable).
    _cov = stats if stats is not None else RuntimeStats()
    _owner_queue = _OwnerCommandQueue(maxsize=_RUNTIME_OWNER_QUEUE_MAX)
    _video_source_tracker = SourceTracker()
    _audio_source_tracker = SourceTracker()
    _video_source_id: str | None = None
    _audio_source_id: str | None = None
    _judge_sched = (
        _ProviderScheduler(
            core,
            gate_source,
            _cov,
            stall_after_s=judge_stall_after_s,
            backpressure_after=judge_backpressure_after,
            decision_deadline_s=judge_decision_deadline_s,
        )
        if judge_dispatch == "async" else None
    )
    _audio_judge_sched = None
    # The standard telemetry funnel (seen->gated->judged->refused->fired,
    # percept.observability.Funnel), bound to THIS core's Metrics when it exposes one (the Percept
    # agent always does; a bare duck-typed core without `.funnel` leaves this None and the two calls
    # below become no-ops — byte-for-byte). judged/refused/fired are already emitted inside
    # core.perceive_judged (agent.py) for both dispatch modes; this run() loop is the one place that
    # can see every frame (seen) and every tier-0 admit (gated), mirroring the pattern already
    # proven in the harness loop (percept.harness.tier0._Tier0Loop).
    _funnel = getattr(core, "funnel", None)

    # Batteries-included gate bound. A 24/7 watch must not carry an unbounded gate; install the
    # configured TTL/LRU as the DEFAULT the instant we drive it (a bare Percept stays unbounded — the
    # opt-out — but the runtime is the always-on entry point, so it opts IN). Explicit gate bounds win: we
    # only fill an UNSET (None) bound, never override one the caller already chose. The TTL sweep rides the
    # stream `now` the judge path already passes; never-goals + per-goal singletons are exempt (the trap).
    _gate = getattr(core, "gate", None)
    if _gate is not None:
        gs = getattr(settings, "gate", None)
        ttl = getattr(gs, "key_ttl_s", 0.0) or 0.0
        if getattr(_gate, "key_ttl_s", None) is None and ttl > 0:
            _gate.key_ttl_s = float(ttl)
        mk = getattr(gs, "max_keys", 0) or 0
        if getattr(_gate, "max_keys", None) is None and mk > 0:
            _gate.max_keys = int(mk)

    # Restart durability. Rehydrate goals/entities/action-log from a persistence store BEFORE
    # compiling the gate, so a restarted 24/7 watch resumes its armed goals + refractory state instead
    # of re-arming edges (which would double-fire). Default (persistence=None) is byte-for-byte.
    if persistence is not None and restore:
        load = getattr(core, "load_state", None)
        if load is not None:
            load(persistence)
        # Episode/counting memory is session-only (ADR-006) — restore() rehydrated goals
        # but NOT the episode map, so any counting goal just resumed at zero. Drain the one-shot
        # ``episode_memory_reset`` restore() queued so that amnesia reaches on_health loudly, once.
        await _drain_restore_health(core, on_health)

    # ADR-001 §6: the delivery-path idempotency ledger — bounded, restart-surviving, keyed
    # by event_id (= Incident.id, percept/incidents.py). Built unconditionally (mirrors _funnel/_gate
    # above) so the dedupe-on-redelivery guarantee holds even when persistence isn't wired at all; when
    # `persistence` IS given it is rehydrated here (a corrupt/missing snapshot is a no-op — never blocks
    # a restart) and the autosave loop below persists it back out, so a crash/restart resumes with the
    # same "already delivered" memory instead of starting blank. `on_incident` is wrapped AFTER any
    # restore so a redelivered Incident from a resumed session is checked against the rehydrated state.
    from percept.runtime.sinks import IdempotencyLedger, IdempotentSink

    _idem_ledger = IdempotencyLedger()
    if persistence is not None and restore:
        try:
            _idem_ledger.restore(persistence.load("percept_delivery_ledger"))
        except Exception:  # noqa: BLE001 - a corrupt/missing ledger snapshot must never block restart
            pass
    if on_incident is not None:
        on_incident = IdempotentSink(on_incident, _idem_ledger, clock=wall_clock)

    def _require_source_scoped_callbacks() -> None:
        """Reject a legacy duck core before it can spend or mutate across source boundaries."""
        active_sources = {
            getattr(goal, "source", "camera")
            for goal in core.goals.active()
        }
        if len(active_sources) <= 1:
            return
        callbacks = {
            "perceive_judged": getattr(core, "perceive_judged", None),
            "check_episode_boundary": getattr(core, "check_episode_boundary", None),
            "check_observability": getattr(core, "check_observability", None),
            "check_deadlines": getattr(core, "check_deadlines", None),
        }
        legacy = [
            name
            for name, callback in callbacks.items()
            if callable(callback) and not _accepts_keyword(callback, "goal_ids")
        ]
        if legacy:
            raise ValueError(
                "a core with active goals spanning multiple source keys requires source-scoped "
                f"callbacks accepting goal_ids; legacy callbacks {legacy!r} could spend or mutate "
                "wrong-source state before delivery filtering. Use isolated workers/adapters."
            )

    def _build_gate(prev_gate=None, prev_banks=None, prev_rhythm=None):
        """(Re)compile the WatchSpec → gate + detector banks + rhythm drive + fovea coast from the
        core's CURRENT active goals. Called once at start, and again on a goal-registry revision change
        when regate_on_goal_change is set (so a goal armed mid-run — e.g. by voice — actually reaches the
        tier-0 gate). On a regate the new gate ADOPTS the prior gate's per-goal throttle + prev frame,
        and a surviving goal's detector bank / rhythm eye is REUSED (its running rep-peak / refractory /
        pose-anchor state must not reset mid-rep) — so arming one goal never disturbs the others."""
        _require_source_scoped_callbacks()
        _preflight_routes()
        _require_duration_runtime()
        visual_active = _visual_goals()
        external_audio_active = _external_only_audio_goals()
        conditions = []
        external_conditions = []
        runtime_bindings = getattr(core, "_watch_runtime_bindings", {})

        def _compile_condition(goal):
            bound = runtime_bindings.get(goal.id)
            kwargs = (
                dict(bound[0].watchspec_kwargs)
                if bound is not None
                else {
                    "threshold": settings.edge.motion_gate,
                    "salience_stat": settings.edge.salience_stat,
                    "ego_compensate": settings.edge.ego_compensate,
                    "stat_thresholds": {
                        "mean": settings.edge.motion_gate,
                        "block_max": settings.edge.motion_gate_block_max,
                        "p95": settings.edge.motion_gate_p95,
                    },
                }
            )
            compiled = compile_watchspec([goal], **kwargs).conditions
            if len(compiled) != 1:
                raise ValueError(
                    f"Watch {goal.id!r} did not compile to exactly one admission condition"
                )
            return compiled[0]

        for goal in visual_active:
            conditions.append(_compile_condition(goal))
        for goal in external_audio_active:
            external_conditions.append(_compile_condition(goal))
        s = WatchSpec(conditions=conditions)
        # clock: a civil-time source arms per-goal Schedules; None -> always-armed (default).
        # The tier-0 extension seam, reachable from the supported entrypoint. None -> the gate's
        # structural/semantic channels stay 0.0 and this is byte-for-byte the motion-only path.
        g = MotionGate(
            s,
            external_conditions=external_conditions,
            settings=settings,
            wake_policy=wake_policy,
            clock=clock,
            structural_adapter=structural_adapter,
            semantic_adapter=semantic_adapter,
        )
        if prev_gate is not None:
            g.adopt_state(prev_gate)  # carry throttle + prev for surviving goal_ids
        # One detector bank per goal that selected edge detectors (state['detectors']). The bank
        # emits timed boundaries (rep peaks, acoustic onsets) from the CHEAP motion/energy signals at
        # native fps — never spends a cloud judge. No detectors -> no banks -> byte-for-byte pre-PR.
        # Reuse a surviving goal's bank so a regate mid-rep doesn't drop/double-count a rep.
        b = {}
        for c in s.conditions:
            if not getattr(c, "detectors", None):
                continue
            if prev_banks is not None and c.goal_id in prev_banks:
                b[c.goal_id] = prev_banks[c.goal_id]
            else:
                b[c.goal_id] = DetectorBank(c.detectors)
        # The COUNTING path: one measuring eye per rhythm goal. It never enters the WatchSpec (a
        # rhythm goal is not admitted, it is measured) so it is built from the goal partition, not
        # from `s.conditions`. A surviving goal keeps its eye across a regate (see RhythmDrive.adopt).
        _rhythm = RhythmDrive(core, _rhythm_goals()).adopt(prev_rhythm)
        # One fovea coast controller per Watch. A shared "most sensitive" controller made one
        # Watch's cadence admit every other Watch on the source; compiling each existing
        # SensorPlan independently preserves the already-defined per-concern allocation semantics.
        _coasts = {}
        _req = 0.0
        for goal in visual_active:
            _cap, _coast, _goal_req = _capture_plan([goal], fps)
            if _coast is not None:
                _coasts[goal.id] = _coast
            _req = max(_req, _goal_req)
        return s, g, b, _coasts, _req, _rhythm

    spec, gate, banks, coasts, required_fps, rhythm_drive = _build_gate()
    perceive_boundary = getattr(core, "perceive_boundary", None)
    goals_revision = getattr(core.goals, "revision", None)  # watch for mid-run goal changes

    # The stable id of the PLAN this run is watching under, stamped on every emitted Incident so a
    # client can join a fire back to the exact goal set + policy it was judged against (the plan_hash that
    # doctor/eval already computes but that never reached the wire). The per-goal executor cannot
    # know the whole plan; the runtime that owns the goal set does. Computed from the CURRENT active goals
    # and cached by goal-revision, so a mid-run goal edit re-derives it and an idle run never recomputes.
    _watchplan_cache: dict = {}

    def _current_watchplan_id():
        rev = getattr(core.goals, "revision", None)
        hit = _watchplan_cache.get("v")
        if hit is not None and hit[0] == rev:
            return hit[1]
        try:  # best-effort: an observability id must never break the firing path
            from percept.tracing import PlanIdentity  # noqa: PLC0415 - lazy, avoids an import cycle
            pid = PlanIdentity.from_goals(_source_goals(), revision=rev).plan_hash
        except Exception:  # noqa: BLE001
            pid = None
        _watchplan_cache["v"] = (rev, pid)
        return pid
    # A brief-action goal raises the capture rate + arms a fovea coast so a sub-second
    # window is covered above its Nyquist; no brief goal -> (fps, None) -> byte-for-byte the fixed-fps path.
    # The source-open rate is FIXED once (a live source can't change fps mid-stream), so when a brief goal
    # could be armed MID-RUN (regate on) over a LIVE source we open at the headroom cap now, leaving the
    # capture headroom a voice-armed brief goal will need — gate/coast then ration the actual judging.
    capture_fps, _, _ = _capture_plan(_visual_goals(), fps)
    # The same two overrides as before — but RECORDED, not silent. `reason` names which one
    # moved the rate (a plan cadence, or the fixed-at-open headroom cap a str source needs), so a client
    # reading the effective config can tell a deliberate 5x from an accident.
    _fps_reason = "brief-window-coverage" if capture_fps > fps else "profile-default"
    if regate_on_goal_change and isinstance(source, str):
        if _BRIEF_CAPTURE_CAP_FPS > capture_fps:
            _fps_reason = "source-limit"
        capture_fps = max(capture_fps, _BRIEF_CAPTURE_CAP_FPS)
    effective_fps = EffectiveFps(
        requested_fps=float(fps), effective_fps=float(capture_fps), reason=_fps_reason,
    )
    # Auto-derive `audio_source` from a video-container `source` when the caller didn't pass one
    # explicitly AND an armed goal actually wants audio (fused/audio modality) — a file-driven run over
    # an mp4/mov/mkv/webm gets its fused judge fed the SAME file's audio track with no extra parameter.
    # A caller-supplied `audio_source` always wins (never overridden); a vision-only goal set spawns no
    # ffmpeg demux subprocess at all — the video-only path stays byte-for-byte.
    if audio_source is None and isinstance(source, str):
        from percept.runtime.audio import _is_container as _audio_is_container

        if _audio_is_container(source) and _goals_want_audio(_source_goals()):
            audio_source = source
    runtime_bindings = getattr(core, "_watch_runtime_bindings", {})

    def _live_watch_topology():
        goals = list(_source_goals())
        fused = [
            goal
            for goal in goals
            if goal.id in runtime_bindings
            and program_of(goal).perception.modality == "fused"
        ]
        non_fused = [
            goal
            for goal in goals
            if program_of(goal).perception.modality != "fused"
        ]
        return fused, non_fused

    compiled_audio_goals = [
        goal
        for goal in _source_goals()
        if goal.id in runtime_bindings
        and program_of(goal).perception.modality in ("audio", "fused")
    ]
    compiled_audio_only_goals = [
        goal
        for goal in compiled_audio_goals
        if program_of(goal).perception.modality == "audio"
    ]
    if judge_dispatch == "async" and compiled_audio_only_goals:
        _audio_judge_sched = _LegacyJudgeScheduler(
            core,
            gate_source,
            _cov,
            perceive=core.perceive_audio_judged,
            stall_after_s=judge_stall_after_s,
            backpressure_after=judge_backpressure_after,
            decision_deadline_s=judge_decision_deadline_s,
        )
    compiled_audio_capability = None
    if compiled_audio_goals:
        bound_audio = [
            core._audio_judge_binding_for(goal)
            for goal in compiled_audio_goals
        ]
        compiled_audio_capability = min(
            (bound.capability for bound in bound_audio),
            key=lambda item: (item.max_input_bytes, item.max_provider_bytes),
        )
    if compiled_audio_goals and audio_source is None:
        raise ValueError(
            "compiled audio/fused Watches require audio_source= (or a media container with an "
            "audio track); refusing to run them as silent vision-only Watches."
        )
    if (
        audio_gate is None
        and any(program_of(goal).perception.modality == "audio" for goal in compiled_audio_goals)
    ):
        raise ValueError(
            "a compiled audio-only Watch requires an audio admission gate; audio_gate=None would "
            "leave it with no executable admission path."
        )
    # Resolve the clock BEFORE the capture source is opened; both the video and the audio
    # path get their own record (a "mic:0" live mic and a "session.wav" recording resolve differently
    # even inside one run).
    video_tb = _resolve_timebase(
        source,
        timebase,
        deterministic_override=_caller_timebase_explicit,
    )
    audio_tb = (
        _resolve_timebase(
            audio_source,
            timebase,
            deterministic_override=_caller_timebase_explicit,
        )
        if audio_source is not None
        else None
    )
    _has_fused = any(
        program_of(goal).perception.modality == "fused"
        for goal in compiled_audio_goals
    )
    fused_goals = (
        [
            goal
            for goal in compiled_audio_goals
            if program_of(goal).perception.modality == "fused"
        ]
        if _has_fused
        else None
    )
    if fused_goals and (
        audio_tb is None or video_tb.resolved != audio_tb.resolved
    ):
        raise ValueError(
            "compiled fused Watches require calibrated audio and video in one clock domain; "
            "refusing cross-domain synchronization before source iteration."
        )
    _fused_syncs = None
    _fused_judge_scheds = None
    _fused_local_completions = None
    if fused_goals:
        fused_prepare = getattr(core, "prepare_fused_judgment", None)
        fused_dispatch = getattr(core, "dispatch_fused_judgment", None)
        fused_complete_local = getattr(core, "complete_fused_locally", None)
        fused_reduce = getattr(core, "reduce_fused_completion", None)
        if (
            not callable(fused_prepare)
            or not callable(fused_dispatch)
            or not callable(fused_complete_local)
            or not callable(fused_reduce)
        ):
            raise ValueError(
                "compiled fused Watches require provider-only dispatch and owner reduction seams"
            )
        if goals_revision is None:
            raise ValueError("compiled fused Watches require a revisioned Goal registry")

        def _iterable_clock_claim(media_source: Any) -> tuple[str, float] | None:
            authority = getattr(media_source, "clock_authority", None)
            confidence = getattr(media_source, "clock_confidence", None)
            if authority is None and isinstance(media_source, (list, tuple)) and media_source:
                if any(type(item) is not SourceEnvelope for item in media_source):
                    return None
                claims = {
                    (item.clock_authority, item.clock_confidence)
                    for item in media_source
                }
                if len(claims) != 1:
                    return None
                authority, confidence = next(iter(claims))
            if (
                type(authority) is not str
                or not authority
                or not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not math.isfinite(float(confidence))
            ):
                return None
            return authority, float(confidence)

        _fused_requires_envelope_clock_claim = False
        if video_tb.resolved == "wall":
            _video_clock_authority = _audio_clock_authority = (
                f"runtime.monotonic:{id(core)}"
            )
            _fused_clock_confidence = 1.0
        elif isinstance(source, str) and isinstance(audio_source, str):
            _video_clock_authority = f"media:{source}"
            _audio_clock_authority = f"media:{audio_source}"
            _fused_clock_confidence = (
                1.0 if _video_clock_authority == _audio_clock_authority else 0.0
            )
        else:
            _fused_requires_envelope_clock_claim = True
            video_claim = _iterable_clock_claim(source)
            audio_claim = _iterable_clock_claim(audio_source)
            if video_claim is None or audio_claim is None:
                raise ValueError(
                    "compiled fused Watches over separate iterable lanes require explicit "
                    "matching clock_authority and clock_confidence proof before iteration"
                )
            _video_clock_authority, video_confidence = video_claim
            _audio_clock_authority, audio_confidence = audio_claim
            _fused_clock_confidence = min(video_confidence, audio_confidence)
        if (
            _video_clock_authority != _audio_clock_authority
            or _fused_clock_confidence < 1.0
        ):
            raise ValueError(
                "compiled fused Watches require one qualified clock authority; "
                "distinct media sources are uncalibrated and are rejected before iteration"
            )
        from percept.fused_sync import (
            _AcceptedMediaUnit,
            _FusedDiscontinuity,
            _FusedSyncPolicy,
            _FusedSynchronizer,
        )

        def _new_fused_synchronizer(goal: Any, revision: int) -> _FusedSynchronizer:
            binding = core._audio_judge_binding_for(goal)
            runtime = runtime_bindings.get(goal.id)
            policy = runtime[0].fused_sync_policy if runtime is not None else None
            if type(policy) is not _FusedSyncPolicy:
                raise ValueError("compiled fused Watch has no exact frozen synchronization policy")
            if policy.max_bytes_per_modality > min(
                binding.capability.max_input_bytes,
                binding.capability.max_provider_bytes,
            ):
                raise ValueError("compiled fused policy exceeds the bound provider media limit")
            return _FusedSynchronizer(
                policy,
                watch_application=binding.application_key,
                watch_revision=revision,
            )

        _fused_syncs = {
            goal.id: _new_fused_synchronizer(goal, goals_revision)
            for goal in fused_goals
        }
        _fused_local_completions = deque(
            maxlen=sum(
                synchronizer.policy.max_units_per_modality
                for synchronizer in _fused_syncs.values()
            )
        )
        async def _dispatch_prepared_fused_job(job, _source=None, **_ignored):
            return await fused_dispatch(job)

        # Each fused Watch has bounded admission slots. Distinct goals mutate distinct executor/gate
        # keys; the core's per-task usage correlation is context-local.
        _fused_judge_scheds = {
            goal.id: _LegacyJudgeScheduler(
                core,
                gate_source,
                _cov,
                perceive=_dispatch_prepared_fused_job,
                stall_after_s=judge_stall_after_s,
                backpressure_after=judge_backpressure_after,
                decision_deadline_s=judge_decision_deadline_s,
            )
            for goal in fused_goals
        }

        def _fused_reason(events: tuple[str, ...]) -> _FusedDiscontinuity:
            mapping = {
                "source_gap": _FusedDiscontinuity.GAP,
                "source_reconnect": _FusedDiscontinuity.RECONNECT,
                "source_epoch_gap": _FusedDiscontinuity.EPOCH_GAP,
                "source_format_change": _FusedDiscontinuity.FORMAT_CHANGE,
                "source_capture_overflow": _FusedDiscontinuity.CAPTURE_OVERFLOW,
            }
            for event in events:
                if event in mapping:
                    return mapping[event]
            return _FusedDiscontinuity.GAP

        def _advance_fused_generation(
            reason: _FusedDiscontinuity,
            *,
            revision: int | None = None,
            boundary: float | None = None,
        ) -> None:
            for synchronizer in _fused_syncs.values():
                synchronizer.advance_generation(reason, watch_revision=revision)
                if boundary is not None:
                    synchronizer.set_generation_start(boundary)

        def _queue_fused_bundle(goal_id: str, bundle: Any) -> None:
            at = (
                bundle.video_event_interval[1]
                if bundle.video_event_interval is not None
                else 0.0
            )
            synchronizer = _fused_syncs[goal_id]
            scheduler = (_fused_judge_scheds or {}).get(goal_id)
            if bundle.complete and (
                scheduler is None or not scheduler.can_accept
            ):
                return
            job = fused_prepare(
                bundle,
                goal_id=goal_id,
                source=gate_source,
                source_t=at,
                goals_revision=goals_revision,
                source_generation=core.source_continuity_generation(gate_source),
                fused_generation=synchronizer.generation,
            )
            if job is None:
                return
            if not bundle.complete:
                if len(_fused_local_completions) == _fused_local_completions.maxlen:
                    dropped_at, dropped = _fused_local_completions.popleft()
                    record_drop = getattr(core, "record_fused_admission_drop", None)
                    if callable(record_drop):
                        record_drop(dropped.job, reason="local_completion_bound")
                _fused_local_completions.append(
                    (at, fused_complete_local(job))
                )
                return
            assert scheduler is not None
            if not scheduler.submit(at, job):
                # No await occurs between can_accept, preparation, and submit;
                # rejection here is an invariant violation, not an ignorable drop.
                raise RuntimeError(
                    "fused provider lane rejected an atomically prepared admission"
                )

    effective_config = EffectiveConfig(
        fps=effective_fps, video_timebase=video_tb, audio_timebase=audio_tb,
        coast_enabled=bool(getattr(getattr(settings, "edge", None), "coast_enabled", True)),
    )
    if stats is not None:
        stats.effective_config = effective_config
    try:  # best-effort mirror onto the core, so a manifest writer reads it off the object it holds
        core.effective_config = effective_config
    except Exception:  # noqa: BLE001 - a duck-typed/slotted core must never break the run over telemetry
        pass
    # The incident snippet's rolling pre-roll buffer — the SDK's largest resident structure, and the
    # one profiling identified as the source of the observed "runaway memory" plateau. It is
    # bounded by COUNT (`snippet_size`, 48) and by BYTES: the count alone does not bound memory, since
    # the same 48 frames cost ~0.4 MB of 8x8 test frames and ~300 MB of 1080p RGB. A byte-bound
    # eviction is REPORTED (`preroll_capped` health event + `stats.preroll_frames_dropped`), never a
    # silent counter bump — ADR-001 §9. `snippet_max_bytes=0` disables the byte bound (count only,
    # the pre-0.7.1 behaviour).
    _preroll_cap = (
        int(getattr(getattr(settings, "events", None), "preroll_max_bytes", 0) or 0)
        if snippet_max_bytes is None else int(snippet_max_bytes)
    )
    _preroll_dropped = [0]
    preroll_health: list[tuple[str, dict]] = []

    def _on_preroll_drop(dt: float, frame: Any, reason: str) -> None:
        _preroll_dropped[0] += 1
        _cov.preroll_frames_dropped += 1
        if stats is not None and stats is not _cov:
            stats.preroll_frames_dropped += 1
        preroll_health.append(("preroll_capped", {
            "reason": reason, "dropped_t": dt, "frame_bytes": frame_nbytes(frame),
            "max_bytes": _preroll_cap, "held_frames": len(clips), "held_bytes": clips.nbytes,
            "dropped_total": _preroll_dropped[0],
        }))

    clips = FrameWindow(
        size=snippet_size,
        max_bytes=_preroll_cap or None,
        on_drop=_on_preroll_drop if _preroll_cap else None,
    )
    # Coverage-guarantee tripwire (emit-once). Tracks the observed capture cadence against the brief
    # goal's required Nyquist floor (required_fps); reset when a regate RAISES the requirement.
    _cap_req_seen = 0.0
    _cap_t0: float | None = None
    _cap_n = 0
    _cap_unmet_emitted = False

    # This run's tracer takes precedence; a tracer set via Percept.create(on_trace=) is preserved when
    # run() gets none. Set immediately before the try so the finally ALWAYS restores it — a tracer never
    # leaks into the next run(core), even if the body below raises. (Setup above runs first, so a setup
    # failure happens before the tracer is wired and leaks nothing.)
    _prev_on_trace = getattr(core, "on_trace", None)
    if on_trace is not None:
        core.on_trace = on_trace  # the per-frame decision trace flows out of perceive_judged
    # The gate threshold / salience statistic the admissions were judged against, so a zero-admission
    # report says WHAT the frames failed to clear (not just that they did). Bound before the try so
    # the finally can always read them, even on a setup failure. Threshold is the last one the gate
    # actually applied (it can move on a regate); None until a first frame reaches the gate.
    _cov_threshold: float | None = None
    _cov_stat = getattr(getattr(settings, "edge", None), "salience_stat", None)
    audio_task = None  # bound before try so the finally can always cancel it, even on setup failure
    autosave_task = None
    # In-flight on_utterance handlers, spawned OFF the pump. Tracked so the finally can cancel them —
    # a voice turn must not outlive the run (a dangling STT/Live turn after stop() is a leak).
    _utt_tasks: set = set()

    def _utt_done(task: "asyncio.Task") -> None:
        # Discard from the set AND retrieve the result: a raising handler (sink_errors="raise") re-raises
        # inside this off-pump task; without retrieval asyncio logs "exception never retrieved". The error
        # was already surfaced via on_health in _call_sink_with_policy — the pump stays alive regardless.
        _utt_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _task_watchdog(kind: str) -> "Callable[[asyncio.Task], None]":
        # A raise at the pump/autosave COROUTINE level (not in a handler) died silently —
        # only the per-utterance task had a done-callback. Retrieve the exception (so asyncio doesn't log
        # "exception never retrieved") and surface it through on_health. We do NOT re-raise into the run —
        # the run decides; a dead pump must fail LOUD, not silently. A normal-exit/cancelled task is quiet.
        def _cb(task: "asyncio.Task") -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is None or on_health is None:
                return
            try:
                asyncio.ensure_future(_call_sink(
                    on_health, RuntimeHealthEvent(kind, data={"error": repr(exc)})))
            except RuntimeError:
                pass  # event loop already closed on teardown — nothing to surface into
        return _cb

    _source_health_seen: set[tuple[str, str, int | None, str]] = set()
    dropped: list[int] = [0]  # shared with the capture thread
    source_health: list = []  # capture thread events drained on the video cadence

    async def _record_source_inspection(
        envelope: SourceEnvelope,
        inspection: SourceInspection,
        *,
        ts: float | None,
    ) -> None:
        video_cov = _video_source_tracker.coverage
        audio_cov = _audio_source_tracker.coverage
        _cov.source_observed = video_cov.observed + audio_cov.observed
        _cov.source_excluded = video_cov.excluded + audio_cov.excluded
        _cov.source_discontinuities = video_cov.discontinuities + audio_cov.discontinuities
        _cov.source_missing_sequences = (
            video_cov.missing_sequences + audio_cov.missing_sequences
        )
        if on_health is None:
            return
        # Validation deliberately accepts an arbitrary SourceEnvelope object so malformed producer
        # fields can be excluded and reported.  Never place those arbitrary values directly in a set
        # key or health payload: lists/dicts are unhashable and reprs may be unbounded.
        def _safe_text(value: Any, fallback: str) -> str:
            if isinstance(value, str) and value:
                return value[:256]
            return fallback

        def _safe_int(value: Any) -> int | None:
            return value if type(value) is int else None

        def _safe_time(value: Any) -> float | None:
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
            ):
                return float(value)
            return None

        safe_source_id = _safe_text(envelope.source_id, "<invalid-source-id>")
        safe_modality = _safe_text(envelope.modality, "<invalid-modality>")
        safe_epoch = _safe_int(envelope.epoch)
        for kind in inspection.events:
            marker = (safe_source_id, safe_modality, safe_epoch, kind)
            if kind == "source_time_unknown" and marker in _source_health_seen:
                continue
            _source_health_seen.add(marker)
            await _call_sink(
                on_health,
                RuntimeHealthEvent(
                    kind,
                    ts=ts,
                    data={
                        "version": _safe_int(envelope.version),
                        "source_id": safe_source_id,
                        "modality": safe_modality,
                        "epoch": safe_epoch,
                        "sequence": _safe_int(envelope.sequence),
                        "source_time": _safe_time(envelope.source_time),
                        "observed_time": _safe_time(envelope.observed_time),
                        "coverage_effect": inspection.coverage.kind,
                        "missing_sequences": inspection.coverage.missing_sequences,
                    },
                ),
            )

    # AUDIO-GATE-DEAF: the same deafness witness the audio-only path keeps. A fused/video run that
    # also opened a mic can be just as deaf, and here the symptom is even quieter — the video gate
    # keeps admitting, so nothing at all hints the audio channel is inert. Declared OUT here (not
    # inside the `audio_source is not None` block) because the teardown below reports through it
    # on EVERY run; `_audio_onset_ref` stays None when no gate was built, and the report is a
    # no-op then — a video-only run is byte-for-byte unchanged.
    #
    # Bound BEFORE the `try` below, not inside it. The `finally` calls `_report_zero_wake`, so
    # anything in the guarded body that raises before the definition was reached — a caller's
    # `on_health` sink raising on `runtime_started`, a source that fails on its first pull — would
    # hit an unbound local during teardown and replace the real exception with an UnboundLocalError.
    # Teardown may only reference names that are bound on every path into the `try`.
    _audio_witness = _AcousticGateWitness()
    _audio_onset_ref: list = [None]

    async def _report_zero_wake(*, at_end: bool) -> None:
        """Emit the latched ``zero_wake_coverage`` diagnostic if this gate never woke."""
        _onset = _audio_onset_ref[0]
        if on_health is None or _onset is None:
            return
        if not _audio_witness.due(at_end=at_end):
            return
        await _call_sink(
            on_health,
            RuntimeHealthEvent(
                "zero_wake_coverage",
                ts=_audio_witness.last_t,
                data=_audio_witness.payload(
                    detector=str(getattr(_onset, "name", audio_gate)),
                    high=getattr(_onset, "high", None),
                    low=getattr(_onset, "low", None),
                    at_end=at_end,
                ),
            ),
        )

    try:
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_started", data={"source": repr(source)}))
            # Announce the dispatch mode ONCE so a live dashboard can tell which loop shape it is
            # observing (inline = judge latency shows up as capture stall; async = it shows up as
            # fire-emission lag instead).
            await _call_sink(on_health, RuntimeHealthEvent("judge_dispatch", data={"mode": judge_dispatch}))
            # Announce the EFFECTIVE config once — informational, at the same tier as
            # runtime_started/judge_dispatch. This is a RECORD, not a warning: an fps override may be
            # correct, and the loud channel stays reserved for a requirement that is actually unmet.
            await _call_sink(on_health, RuntimeHealthEvent(
                "effective_config", data=effective_config.as_dict()))
            # Any route this run is SKIPPING rather than driving (the preflight already warned + logged).
            # Structured, so a dashboard can show "armed but not driven here" instead of a client
            # discovering it as a goal that never fires. Regate additions drain on the frame cadence below.
            while _route_health:
                await _call_sink(on_health, RuntimeHealthEvent(
                    "undriven_behavior_route", data=_route_health.pop(0)))
        source_kwargs = {
            "maxsize": source_queue_size,
            "reconnects": source_reconnects,
            "reconnect_backoff_s": source_reconnect_backoff_s,
            "rtsp_tcp": source_rtsp_tcp,
            "dropped": dropped,
            "health": source_health,
        }
        # A captured SourceEnvelope always preserves both clocks it actually has.  This setting selects
        # which clock drives Watch evaluation: observation time for a live source, source time for
        # recorded media. Deterministic legacy tuples preserve their explicit `t` under the selected
        # source/observation interpretation.
        live_video = isinstance(source, str)
        # The normalizer needs the selected clock for custom legacy tuples too. URI sources pass the
        # same value through to their capture adapter.
        source_kwargs["timebase"] = video_tb.resolved
        live_audio = isinstance(audio_source, str)
        audio_kwargs = {"timebase": audio_tb.resolved} if audio_tb else {}
        # Emit-once skew tripwires, one per capture path (frame queue ~64 @4fps ~16s; audio queue
        # ~256 @4/s ~64s — a single global tolerance would false-fire on a legitimately-backlogged audio
        # chunk behind a blocked pump, an expected condition). Only armed on live paths (media is offline).
        _skew_emitted: set[str] = set()
        # The first frame's selected `t`, retained only for legacy cross-modal pacing.  Elapsed
        # observation time is never recorded as source/media provenance.
        _clock_t0: list = [None]
        # The most recent frame timestamp, so the end-of-source gate-suppression drain can stamp its
        # events on the stream timeline. 0.0 until the first frame (an empty source drains nothing).
        _last_t: float = 0.0
        # Episode boundary: a process activity ends precisely on QUIET frames (no escalation), so the
        # boundary check rides the per-frame cadence — only spending a summarize call when it actually fires.
        check_boundary = getattr(core, "check_episode_boundary", None)
        # The absence-native checks, driven every frame (they fire on QUIET frames the
        # gate skips). Both are perception-robust (fire on the LACK of evidence, never a guess) and opt-in
        # per concern — check_observability only for miss_cost=='severe' ("I can't see the baby"), and
        # check_deadlines only for rearm=='deadline' ("overdue"). Default concerns -> both silent,
        # byte-for-byte. Sync (unlike the async boundary check).
        check_coverage = getattr(core, "check_observability", None)
        check_deadlines = getattr(core, "check_deadlines", None)
        boundary_accepts_goal_ids = (
            check_boundary is not None and _accepts_keyword(check_boundary, "goal_ids")
        )
        coverage_accepts_goal_ids = (
            check_coverage is not None and _accepts_keyword(check_coverage, "goal_ids")
        )
        deadlines_accepts_goal_ids = (
            check_deadlines is not None and _accepts_keyword(check_deadlines, "goal_ids")
        )
        # --- autosave: periodically persist state so a crash/restart resumes, not restarts. ----
        # Also persists the delivery idempotency ledger's snapshot() (event_id -> ts of a
        # terminal-OK delivery only — no secrets/media) alongside goals/entities, so a restart that
        # replays a redelivered Incident still dedupes it. Gated on `persistence` alone (not
        # `core.save_state`) — a bare duck-typed core without `save_state` still gets its delivery
        # ledger autosaved, it just skips the core-state half of the save.
        if persistence is not None and autosave_every_s:
            async def _autosave():
                while True:
                    await asyncio.sleep(autosave_every_s)
                    try:
                        if getattr(core, "save_state", None):
                            core.save_state(persistence)
                        persistence.save("percept_delivery_ledger", _idem_ledger.snapshot())
                        if on_health is not None:
                            await _call_sink(on_health, RuntimeHealthEvent("state_saved"))
                    except Exception as exc:  # noqa: BLE001 - a save failure must not kill the watch
                        if on_health is not None:
                            await _call_sink(on_health, RuntimeHealthEvent("state_save_error", data={"error": repr(exc)}))
            autosave_task = asyncio.ensure_future(_autosave())
            autosave_task.add_done_callback(_task_watchdog("autosave_died"))

        # --- audio ingress: fill the AudioWindow + feed the gate an energy/onset signal --------
        # `audio` holds the latest energy (+ the time of that reading and the observed chunk cadence, for
        # the freshness guard) and a one-shot onset-wake flag, shared with the pump task below (same
        # event loop, cooperative — no lock). Default (audio_source=None) leaves it inert: energy stays
        # None and wake stays False, so gate.evaluate and the admit test are byte-for-byte the video path.
        audio = {"energy": None, "energy_t": None, "chunk_dt": None, "wake": False, "wake_at": None}
        _audio_missing_reported: set[str] = set()

        def _advance_source_continuity() -> None:
            advance = getattr(core, "advance_source_continuity", None)
            if callable(advance):
                advance(gate_source)

        def _compiled_audio_ready(goal_id: str) -> bool:
            """Preflight exact retained evidence without allocating a provider aggregate."""
            if goal_id not in getattr(core, "_watch_runtime_bindings", {}):
                return True  # legacy goals retain their documented vision-only degradation
            goal = core.goals.get(goal_id)
            if goal is None:
                return False
            perception = program_of(goal).perception
            if perception.modality not in ("audio", "fused"):
                return True
            window_for = getattr(core, "_audio_window", None)
            if not callable(window_for):
                return False
            units = window_for(gate_source).clip_units(
                perception.audio_s or settings.events.audio_window_s,
                now=t,
            )
            if not units:
                return False
            try:
                from percept.audio_input import validate_audio_units_for_provider

                binding = core._audio_judge_binding_for(goal)
                validate_audio_units_for_provider(
                    tuple(units), capability=binding.capability
                )
            except (RuntimeError, ValueError, TypeError):
                return False
            return True

        def _exact_audio_evidence(
            goal_ids: tuple[str, ...],
            *,
            at: float,
        ) -> dict[str, tuple[Any, ...]]:
            """Freeze exact retained units for audio-only provider admission."""
            from percept.audio_input import validate_audio_units_for_provider

            evidence = {}
            for goal_id in goal_ids:
                goal = core.goals.get(goal_id)
                if goal is None:
                    continue
                perception = program_of(goal).perception
                if perception.modality != "audio":
                    continue
                units = tuple(
                    core._audio_window(gate_source).clip_units(
                        perception.audio_s or settings.events.audio_window_s,
                        now=at,
                    )
                )
                if not units:
                    continue
                binding = core._audio_judge_binding_for(goal)
                validate_audio_units_for_provider(
                    units, capability=binding.capability
                )
                evidence[goal_id] = units
            return evidence

        _prev_chunk_at: list = [None]  # last chunk's t — for the observed inter-chunk cadence (chunk_dt)
        audio_task = None
        # Media-clock pacing for a FILE/CONTAINER audio_source (see `_AUDIO_PACE_LOOKAHEAD_S`).
        # An ffmpeg/wave capture thread streams a whole recorded file's chunks in well under a
        # wall-second — far faster than the video frame loop, which judges at its own (often much
        # slower) pace. Left unchecked, the ENTIRE file lands in the AudioWindow before the first judge
        # call, so the rolling window prunes to the file's final seconds and every fused judge hears the
        # TAIL no matter which frame it is judging. Paced ONLY when `audio_source` is itself a STRING URI
        # that resolved to MEDIA time (`live_audio` — real `open_audio_source` capture — AND
        # `_resolve_timebase`'s "recorded-source" branch, i.e. a wav/mp4/container path, never "mic:").
        # A live mic resolves to WALL time and this stays False for it, so the live path is
        # byte-for-byte unchanged. Just as important: a DETERMINISTIC test/replay `audio_source` (a
        # plain async generator/iterable feeding pre-baked chunks, not a URI) is NOT `live_audio` either
        # — `_resolve_timebase` would otherwise call it "media" too (its non-str "deterministic-stream"
        # branch), which would wrongly pace synthetic fixtures whose `at` values have no real
        # relationship to the video timeline they're paired with. `_video_media_t` is the media time of
        # the most recently processed VIDEO frame; `_video_progress` is the Event the pump awaits (never
        # a busy-wait) until the video has caught up to within the lookahead of a chunk's own media
        # time. `_video_exhausted` unconditionally releases the pump once the video source ends, so a
        # video shorter than its audio track still drains the remainder instead of hanging forever.
        _audio_paced = live_audio and audio_tb is not None and audio_tb.resolved == "media"
        _video_media_t: list = [0.0]
        _video_progress = asyncio.Event()
        _audio_pace_settled = asyncio.Event()
        _owner_ready = asyncio.Event()
        _video_exhausted = [False]
        _audio_eos_seen = [False]
        # Capture-side ordering is transport state only. The pump uses this
        # mirror to validate/publish a canonical command stream while the
        # cognition owner's tracker advances later during ordered drains.
        _audio_capture_tracker = SourceTracker()
        if audio_source is not None:
            # Size the SHARED fused-judge AudioWindow (distinct from the per-utterance buffer below).
            # A voice-enabled session wants richer acoustic context than the small fused default; the
            # utterance itself no longer slices this window, so growing it can't truncate a turn. Only set
            # a fresh window (a restored/pre-existing one keeps its history).
            _fused_win_s = audio_window_s
            if _fused_win_s is None and on_utterance is not None:
                _fused_win_s = max(30.0, float(getattr(settings.events, "audio_window_s", 4.0)))
            if _fused_win_s is not None:
                _windows = getattr(core, "audio_windows", None)
                if isinstance(_windows, dict) and gate_source not in _windows:
                    from percept.events import AudioWindow
                    _windows[gate_source] = AudioWindow(float(_fused_win_s))
            onset = None
            if audio_gate is not None:
                if isinstance(audio_gate, str):
                    from percept.edge.detectors import AcousticOnset
                    if audio_gate == "acoustic-onset":
                        wake_opts = dict(audio_wake or {})
                        for key in ("high", "low"):
                            if key in wake_opts:
                                wake_opts[key] = _amplitude_threshold_to_energy(
                                    wake_opts[key], name=f"audio_wake.{key}"
                                )
                        onset = AcousticOnset(**wake_opts)
                    else:
                        onset = None
                else:
                    onset = audio_gate  # a caller-supplied detector instance
            _audio_onset_ref[0] = onset  # publish the gate to the deafness reporter declared above

            # Voice-escalation endpointer: only built when on_utterance is wired — default None
            # leaves the audio path byte-for-byte. `vad` can be any object with push(energy)->ev.endpoint.
            _vad = None
            _utt = {"start": None}  # start time of the utterance currently being spoken
            _utt_buf: list = []  # (at, chunk) accumulated started→endpoint — the OWN utterance buffer
            _wall_anchor: list = [None]  # (wall0, stream0) at the first VAD chunk — lag reference
            if on_utterance is not None:
                if vad is not None:
                    _vad = vad
                else:
                    from percept.runtime._vad import EnergyVad
                    _vad = EnergyVad(
                        threshold=_amplitude_threshold_to_energy(
                            utterance_threshold, name="utterance_threshold"
                        ),
                        hangover_frames=utterance_hangover_frames,
                        min_speech_frames=utterance_min_speech_frames,
                    )

            def _reset_audio_temporal_state(*, clear_window: bool = False) -> None:
                """Reset every live-audio clock/DSP state at a continuity boundary."""
                if clear_window:
                    window = getattr(core, "audio_windows", {}).get(gate_source)
                    if window is not None and hasattr(window, "clear"):
                        window.clear()
                audio.update(
                    {
                        "energy": None,
                        "energy_t": None,
                        "chunk_dt": None,
                        "wake": False,
                        "wake_at": None,
                    }
                )
                _prev_chunk_at[0] = None
                _wall_anchor[0] = None
                reset_onset = getattr(onset, "reset", None)
                if callable(reset_onset):
                    reset_onset()
                if _vad is not None:
                    _vad.reset()
                _utt["start"] = None
                _utt_buf.clear()

            async def _apply_audio_boundary(command: dict[str, Any]) -> None:
                """Commit one audio continuity boundary on the runtime owner."""
                reasons = tuple(command["reasons"]) or (
                    "audio_unit_discontinuity",
                )
                if _fused_syncs:
                    _advance_fused_generation(
                        _fused_reason(reasons),
                        boundary=command["boundary"],
                    )
                _advance_source_continuity()
                if _judge_sched is not None:
                    _judge_sched.invalidate_source_continuity(reasons)
                if _audio_judge_sched is not None:
                    _audio_judge_sched.invalidate_source_continuity(reasons)
                for scheduler in (_fused_judge_scheds or {}).values():
                    scheduler.invalidate_source_continuity(reasons)
                _reset_audio_temporal_state(clear_window=True)

            async def _apply_audio_rejected(command: dict[str, Any]) -> None:
                """Commit one capture-side rejection to owner coverage."""
                inspection = _audio_source_tracker.inspect(
                    command["envelope"],
                    require_source_time=command["require_source_time"],
                )
                await _record_source_inspection(
                    command["envelope"],
                    inspection,
                    ts=command["at"],
                )

            async def _apply_audio_media(command: dict[str, Any]) -> None:
                """Commit one canonical audio carrier on the sole runtime owner."""
                nonlocal _audio_source_id
                envelope = command["envelope"]
                unit = command["unit"]
                unit_preview = command["unit_preview"]
                at = command["at"]
                chunk = command["chunk"]
                energy = command["energy"]

                if (
                    _fused_syncs
                    and _audio_source_id is not None
                    and envelope.source_id != _audio_source_id
                ):
                    raise ValueError(
                        "one fused runtime worker accepts exactly one audio source_id; "
                        f"got {_audio_source_id!r} then {envelope.source_id!r}"
                    )
                inspection = _audio_source_tracker.inspect(
                    envelope, require_source_time=command["require_source_time"]
                )
                if not inspection.accepted:
                    await _record_source_inspection(envelope, inspection, ts=at)
                    return
                if _fused_syncs and _audio_source_id is None:
                    _audio_source_id = envelope.source_id
                continuity_reset = (
                    inspection.resets_continuity
                    or (
                        compiled_audio_capability is not None
                        and unit_preview.discontinuous
                    )
                )
                if continuity_reset and not command["boundary_published"]:
                    if _fused_syncs:
                        _advance_fused_generation(
                            _fused_reason(inspection.events),
                            boundary=unit.t_start,
                        )
                    _advance_source_continuity()
                    if _judge_sched is not None:
                        reasons = inspection.events or (
                            "audio_unit_discontinuity",
                        )
                        _judge_sched.invalidate_source_continuity(reasons)
                    if _audio_judge_sched is not None:
                        reasons = inspection.events or (
                            "audio_unit_discontinuity",
                        )
                        _audio_judge_sched.invalidate_source_continuity(reasons)
                    for scheduler in (_fused_judge_scheds or {}).values():
                        reasons = inspection.events or (
                            "audio_unit_discontinuity",
                        )
                        scheduler.invalidate_source_continuity(reasons)
                    _reset_audio_temporal_state(clear_window=True)
                discontinuity = (
                    _fused_reason(inspection.events)
                    if _fused_syncs and continuity_reset
                    else _FusedDiscontinuity.NONE
                    if _fused_syncs
                    else None
                )
                accepted_fused_audio = []
                for synchronizer in (_fused_syncs or {}).values():
                    accepted_fused_audio.append(
                        (
                            synchronizer,
                            _AcceptedMediaUnit(
                                modality="audio",
                                payload=unit,
                                source_id=envelope.source_id,
                                source_epoch=envelope.epoch,
                                source_sequence=envelope.sequence,
                                generation=synchronizer.generation,
                                watch_application=synchronizer.watch_application,
                                watch_revision=synchronizer.watch_revision,
                                event_start=unit.t_start,
                                event_end=unit.t_end,
                                observed_start=envelope.observed_time,
                                observed_end=envelope.observed_time,
                                clock_domain=(
                                    "source"
                                    if audio_tb is not None
                                    and audio_tb.resolved == "media"
                                    else "observed"
                                ),
                                clock_authority=_audio_clock_authority,
                                clock_confidence=_fused_clock_confidence,
                                provenance={
                                    "adapter": "runtime.audio",
                                    "format": {
                                        "media_type": envelope.format.media_type,
                                        "sample_rate": envelope.format.sample_rate,
                                        "channels": envelope.format.channels,
                                        "sample_format": envelope.format.sample_format,
                                    },
                                    "direction": unit.direction,
                                    "participant_id": unit.participant_id,
                                    "source_time": envelope.source_time,
                                    "observed_time": envelope.observed_time,
                                },
                                discontinuity=discontinuity,
                            ),
                        )
                    )
                try:
                    ingest = core.push_audio(unit, source=gate_source)
                except Exception as exc:  # noqa: BLE001
                    if _fused_syncs:
                        _advance_fused_generation(
                            _FusedDiscontinuity.WORKER_ERROR
                        )
                    if on_health is not None:
                        await _call_sink(
                            on_health,
                            RuntimeHealthEvent(
                                "audio_unit_rejected",
                                ts=at,
                                data={
                                    "source": gate_source,
                                    "seq": getattr(unit, "seq", None),
                                    "error": repr(exc),
                                },
                            ),
                        )
                    return
                await _record_source_inspection(envelope, inspection, ts=at)
                if (
                    ingest is not None
                    and getattr(ingest, "discontinuous", False)
                    and on_health is not None
                ):
                    data = ingest.as_dict()
                    data.update({"source": gate_source, "seq": unit.seq})
                    await _call_sink(
                        on_health,
                        RuntimeHealthEvent(
                            "audio_discontinuity", ts=at, data=data
                        ),
                    )
                if ingest is not None and not getattr(
                    ingest, "accepted", True
                ):
                    if _fused_syncs:
                        _advance_fused_generation(
                            _FusedDiscontinuity.WORKER_ERROR
                        )
                    return
                fused_accept_failed = False
                for synchronizer, accepted_audio in accepted_fused_audio:
                    try:
                        if not synchronizer.accept(accepted_audio):
                            fused_accept_failed = True
                    except Exception:
                        fused_accept_failed = True
                if fused_accept_failed:
                    _advance_fused_generation(
                        _FusedDiscontinuity.WORKER_ERROR
                    )
                    return
                for goal_id, synchronizer in (_fused_syncs or {}).items():
                    for ready_bundle in synchronizer.drain_ready():
                        _queue_fused_bundle(goal_id, ready_bundle)
                if stats is not None:
                    stats.audio_chunks += 1

                if ingest is not None and getattr(ingest, "reordered", False):
                    return
                if ingest is not None and (
                    getattr(ingest, "missing", 0)
                    or getattr(ingest, "gap_s", 0.0)
                    or getattr(ingest, "format_changed", False)
                ):
                    _reset_audio_temporal_state()

                if _prev_chunk_at[0] is not None:
                    delta = at - _prev_chunk_at[0]
                    if delta > 0:
                        audio["chunk_dt"] = delta
                _prev_chunk_at[0] = at
                if energy is not None:
                    audio["energy"] = float(energy)
                    audio["energy_t"] = at
                if onset is not None and energy is not None:
                    _boundaries = list(onset.push(at, float(energy)))
                    _audio_witness.observe(
                        at,
                        float(energy),
                        wakes=sum(
                            1 for b in _boundaries
                            if getattr(b, "kind", None) == "onset"
                        ),
                        rearm=getattr(onset, "low", None),
                    )
                    for boundary in _boundaries:
                        if (
                            getattr(boundary, "kind", None) == "onset"
                            and audio["wake_at"] is None
                        ):
                            audio["wake_at"] = at + max(
                                0.0, float(audio_settle_s)
                            )
                        if getattr(boundary, "kind", None) == "onset":
                            core._audio_window(gate_source).annotate(
                                unit.seq,
                                AudioBoundary(
                                    "onset", float(boundary.t)
                                ),
                            )
                    # Mid-run and latched — a live watch may never reach `runtime_ended`.
                    await _report_zero_wake(at_end=False)

                if _vad is None or energy is None:
                    return
                suppress = False
                if stale_audio_s is not None or deaf_until is not None:
                    wall_now = command["wall_now"]
                    if stale_audio_s is not None:
                        if _wall_anchor[0] is None:
                            _wall_anchor[0] = (wall_now, at)
                        wall_start, stream_start = _wall_anchor[0]
                        lag = (wall_now - wall_start) - (at - stream_start)
                        if lag > stale_audio_s:
                            suppress = True
                            if on_health is not None:
                                await _call_sink(
                                    on_health,
                                    RuntimeHealthEvent(
                                        "audio_stale",
                                        ts=at,
                                        data={"lag_s": lag},
                                    ),
                                )
                    if deaf_until is not None and wall_now < deaf_until():
                        suppress = True
                if suppress:
                    _vad.reset()
                    _utt["start"] = None
                    _utt_buf.clear()
                    return

                endpoint = _vad.push(float(energy))
                if getattr(endpoint, "started", False):
                    _utt["start"] = at
                    _utt_buf.clear()
                    core._audio_window(gate_source).annotate(
                        unit.seq, AudioBoundary("vad_start", at)
                    )
                if _utt["start"] is not None:
                    _utt_buf.append((at, chunk))
                    cutoff = at - utterance_max_s
                    while _utt_buf and _utt_buf[0][0] < cutoff:
                        _utt_buf.pop(0)
                if not getattr(endpoint, "endpoint", False):
                    return
                core._audio_window(gate_source).annotate(
                    unit.seq, AudioBoundary("endpoint", at)
                )
                start_t = _utt["start"] if _utt["start"] is not None else at
                _utt["start"] = None
                buf = list(_utt_buf)
                _utt_buf.clear()
                first_t = buf[0][0] if buf else start_t
                clip = _clip_from_chunks(
                    [part for _, part in buf],
                    t_start=first_t,
                    t_end=at,
                )
                if clip is None:
                    return
                if stats is not None:
                    stats.utterances += 1
                covered = at - first_t
                expected = at - start_t
                if on_health is not None and covered < expected - 0.25:
                    await _call_sink(
                        on_health,
                        RuntimeHealthEvent(
                            "utterance_truncated",
                            ts=at,
                            data={
                                "covered_s": covered,
                                "expected_s": expected,
                            },
                        ),
                    )
                if on_health is not None:
                    await _call_sink(
                        on_health,
                        RuntimeHealthEvent(
                            "utterance", ts=at, data={"dur_s": expected}
                        ),
                    )
                task = asyncio.ensure_future(
                    _call_sink_with_policy(
                        on_utterance,
                        clip,
                        at,
                        errors=sink_errors,
                        stats=stats,
                        on_health=on_health,
                        health_kind="utterance_sink_error",
                    )
                )
                _utt_tasks.add(task)
                task.add_done_callback(_utt_done)

            async def _pump_audio():
                # Pump every chunk into the AudioWindow (un-strands modality="fused"/"audio") and track
                # energy + onsets. Never lets a bad audio chunk kill the run — the watch survives a mic drop.
                nonlocal _audio_source_id
                synthetic_carrier_sequence = 0

                async def _reject_unit(envelope, unit, at, exc) -> None:
                    if compiled_audio_capability is None:
                        rejected = _audio_source_tracker.exclude(
                            "source_audio_unit_rejected"
                        )
                        await _record_source_inspection(envelope, rejected, ts=at)
                    if on_health is not None:
                        await _call_sink(
                            on_health,
                            RuntimeHealthEvent(
                                "audio_unit_rejected",
                                ts=at,
                                data={
                                    "source": gate_source,
                                    "seq": getattr(unit, "seq", None),
                                    "error": repr(exc),
                                },
                            ),
                        )

                async for ingress in _audio_aiter(audio_source, **audio_kwargs):
                    submitted_unit = ingress.unit
                    envelope = ingress.envelope
                    ingress_clock_authority = getattr(
                        envelope, "clock_authority", None
                    )
                    ingress_clock_confidence = getattr(
                        envelope, "clock_confidence", None
                    )
                    canonical_ingress_unit = None
                    if compiled_audio_capability is not None:
                        from percept.audio_input import (
                            canonicalize_audio_ingress_frame,
                            canonicalize_audio_unit,
                        )

                        observed_time = (
                            ingress.observed_time
                            if ingress.observed_time is not None
                            else time.monotonic()
                        )
                        try:
                            if submitted_unit is not None:
                                at_hint = (
                                    submitted_unit.t_end
                                    if audio_tb is not None
                                    and audio_tb.resolved == "media"
                                    else observed_time
                                )
                                canonical_ingress_unit = canonicalize_audio_unit(
                                    replace(submitted_unit, event_t=float(at_hint)),
                                    capability=compiled_audio_capability,
                                    source_id="audio",
                                    source_epoch=0,
                                    source_sequence=synthetic_carrier_sequence,
                                    observed_time=observed_time,
                                    source_time=submitted_unit.t_end,
                                )
                                envelope = audio_unit_envelope(
                                    canonical_ingress_unit,
                                    source_id="audio",
                                    sequence=synthetic_carrier_sequence,
                                    observed_time=observed_time,
                                )
                            elif envelope is None:
                                item = ingress.legacy_item
                                if type(item) is not tuple or len(item) != 3:
                                    raise ValueError(
                                        "legacy audio input must be an exact "
                                        "(t, AudioFrame, energy) tuple"
                                    )
                                item_t, raw_frame, energy = item
                                source_time = (
                                    float(item_t)
                                    if ingress.legacy_timebase == "source"
                                    else None
                                )
                                at_hint = (
                                    source_time
                                    if audio_tb is not None
                                    and audio_tb.resolved == "media"
                                    else float(item_t)
                                    if ingress.legacy_timebase == "observed"
                                    else observed_time
                                )
                                canonical_ingress_unit = (
                                    canonicalize_audio_ingress_frame(
                                        raw_frame,
                                        event_t=float(at_hint),
                                        seq=synthetic_carrier_sequence,
                                        energy=energy,
                                        capability=compiled_audio_capability,
                                        source_id="audio",
                                        source_epoch=0,
                                        source_sequence=synthetic_carrier_sequence,
                                        observed_time=observed_time,
                                        source_time=source_time,
                                    )
                                )
                                envelope = SourceEnvelope(
                                    source_id="audio",
                                    modality="audio",
                                    epoch=0,
                                    sequence=synthetic_carrier_sequence,
                                    observed_time=observed_time,
                                    source_time=source_time,
                                    format=SourceFormat(
                                        media_type="audio/pcm",
                                        sample_rate=canonical_ingress_unit.sample_rate,
                                        channels=canonical_ingress_unit.channels,
                                        sample_format=canonical_ingress_unit.sample_format,
                                    ),
                                    payload=canonical_ingress_unit.chunk,
                                    signal=canonical_ingress_unit.energy,
                                    audio_direction=canonical_ingress_unit.direction,
                                    participant_id=canonical_ingress_unit.participant_id,
                                    legacy_timebase=ingress.legacy_timebase,
                                )
                            else:
                                if type(envelope) is not SourceEnvelope:
                                    raise ValueError(
                                        "compiled audio ingress requires an exact SourceEnvelope"
                                    )
                                if envelope.modality != "audio":
                                    raise ValueError("source_modality_mismatch")
                                require_media = (
                                    audio_tb is not None
                                    and audio_tb.resolved == "media"
                                )
                                if require_media and envelope.source_time is None:
                                    raise ValueError("source_time_unknown")
                                at_hint = (
                                    envelope.source_time
                                    if require_media
                                    else envelope.observed_time
                                )
                                canonical_ingress_unit = (
                                    canonicalize_audio_ingress_frame(
                                        envelope.payload,
                                        event_t=float(at_hint),
                                        seq=envelope.sequence,
                                        energy=envelope.signal,
                                        capability=compiled_audio_capability,
                                        source_id=envelope.source_id,
                                        source_epoch=envelope.epoch,
                                        source_sequence=envelope.sequence,
                                        observed_time=envelope.observed_time,
                                        source_time=envelope.source_time,
                                        direction=envelope.audio_direction or "unknown",
                                        participant_id=envelope.participant_id,
                                    )
                                )
                                envelope = replace(
                                    envelope,
                                    payload=canonical_ingress_unit.chunk,
                                    signal=canonical_ingress_unit.energy,
                                    format=SourceFormat(
                                        media_type="audio/pcm",
                                        sample_rate=canonical_ingress_unit.sample_rate,
                                        channels=canonical_ingress_unit.channels,
                                        sample_format=canonical_ingress_unit.sample_format,
                                    ),
                                )
                        except Exception as exc:
                            await _reject_unit(
                                envelope, submitted_unit, observed_time, exc
                            )
                            continue
                        submitted_unit = canonical_ingress_unit
                    elif submitted_unit is not None:
                        observed_time = (
                            ingress.observed_time
                            if ingress.observed_time is not None
                            else time.monotonic()
                        )
                        original_unit = submitted_unit
                        try:
                            submitted_unit = validate_audio_unit(original_unit)
                        except Exception as exc:  # noqa: BLE001 - one bad unit cannot kill the pump
                            rejected_envelope = SourceEnvelope(
                                source_id="audio",
                                modality="audio",
                                epoch=0,
                                sequence=synthetic_carrier_sequence,
                                observed_time=observed_time,
                                source_time=None,
                                format=SourceFormat(
                                    media_type="application/x-percept-opaque"
                                ),
                                payload="<invalid-audio-unit>",
                                legacy_timebase="observed",
                            )
                            await _reject_unit(
                                rejected_envelope,
                                original_unit,
                                observed_time,
                                exc,
                            )
                            continue
                        envelope = audio_unit_envelope(
                            submitted_unit,
                            source_id="audio",
                            sequence=synthetic_carrier_sequence,
                            observed_time=observed_time,
                        )
                    elif envelope is None:
                        envelope = legacy_audio_envelope(
                            ingress.legacy_item,
                            source_id="audio",
                            sequence=synthetic_carrier_sequence,
                            observed_time=(
                                ingress.observed_time
                                if ingress.observed_time is not None
                                else time.monotonic()
                            ),
                            timebase=ingress.legacy_timebase,
                        )
                    assert envelope is not None
                    if _fused_syncs and _fused_requires_envelope_clock_claim and (
                        ingress_clock_authority != _audio_clock_authority
                        or ingress_clock_confidence is None
                        or float(ingress_clock_confidence)
                        != _fused_clock_confidence
                    ):
                        raise ValueError(
                            "fused audio envelope changed or omitted its qualified "
                            "clock authority/confidence"
                        )
                    require_source_time = (
                        submitted_unit is None
                        and audio_tb is not None
                        and audio_tb.resolved == "media"
                    )
                    if envelope.modality in ("video", "audio") and envelope.modality != "audio":
                        # Reject a valid envelope routed to the wrong lane before it can create tracker
                        # state or reach a modality-specific gate/window.
                        inspection = (
                            SourceInspection(
                                accepted=False,
                                events=("source_modality_mismatch",),
                                coverage_effect="excluded",
                            )
                            if compiled_audio_capability is not None
                            else _audio_source_tracker.exclude(
                                "source_modality_mismatch"
                            )
                        )
                        wrong_lane_ts = envelope.observed_time
                        if (
                            isinstance(wrong_lane_ts, bool)
                            or not isinstance(wrong_lane_ts, (int, float))
                            or not math.isfinite(wrong_lane_ts)
                        ):
                            wrong_lane_ts = None
                        await _record_source_inspection(
                            envelope, inspection, ts=wrong_lane_ts
                        )
                        continue
                    else:
                        source_preview = _audio_capture_tracker.preview(
                            envelope, require_source_time=require_source_time
                        )
                    at = (
                        submitted_unit.t
                        if submitted_unit is not None
                        else (
                            envelope.source_time
                            if require_source_time
                            and isinstance(envelope.source_time, (int, float))
                            and not isinstance(envelope.source_time, bool)
                            else envelope.observed_time
                        )
                    )
                    if not (
                        isinstance(at, (int, float))
                        and not isinstance(at, bool)
                        and math.isfinite(float(at))
                    ):
                        at = None
                    if not source_preview.accepted:
                        inspection = _audio_capture_tracker.inspect(
                            envelope, require_source_time=require_source_time
                        )
                        _owner_queue.publish(
                            "AudioRejected",
                            {
                                "envelope": envelope,
                                "require_source_time": require_source_time,
                                "at": at,
                                "capture_inspection": inspection,
                            },
                        )
                        _owner_ready.set()
                        continue
                    if at is None:
                        await _reject_unit(
                            envelope,
                            submitted_unit,
                            None,
                            ValueError("audio event time must be finite"),
                        )
                        continue
                    at = float(at)
                    from percept.runtime.audio import normalize_audio_envelope

                    direction = audio_kwargs.get("direction")
                    if (
                        direction is None
                        and isinstance(audio_source, str)
                        and audio_source.startswith("mic")
                    ):
                        direction = "input"
                    candidate_unit = submitted_unit
                    try:
                        if compiled_audio_capability is not None:
                            if canonical_ingress_unit is None:
                                raise RuntimeError(
                                    "compiled audio ingress lost its canonical unit"
                                )
                            unit = canonical_ingress_unit
                        else:
                            unit = normalize_audio_envelope(
                                envelope,
                                event_t=at,
                                unit=submitted_unit,
                                direction=direction,
                                participant_id=audio_kwargs.get("participant_id"),
                            )
                        candidate_unit = unit
                        audio_window_for = getattr(core, "_audio_window", None)
                        if not callable(audio_window_for):
                            raise TypeError(
                                "audio ingress requires an inspectable AudioWindow"
                            )
                        unit_preview = audio_window_for(gate_source).preview_unit(
                            unit,
                            reset=source_preview.resets_continuity,
                        )
                        if not unit_preview.accepted:
                            raise ValueError(
                                f"audio unit admission rejected: {unit_preview.as_dict()!r}"
                            )
                    except Exception as exc:  # noqa: BLE001 - rejected media cannot commit source state
                        await _reject_unit(envelope, candidate_unit, at, exc)
                        continue

                    chunk, energy = envelope.payload, envelope.signal
                    # Hold a FILE/CONTAINER chunk back until the video loop's judged media time has
                    # (almost) reached it — never busy-wait, park on the Event the frame loop sets every
                    # tick. `_audio_paced` is False for a live mic (byte-for-byte unchanged: this loop
                    # body is reached exactly as fast as `_audio_aiter` yields, same as before this gate
                    # existed).
                    while (
                        _audio_paced
                        and not source_preview.resets_continuity
                        and not _video_exhausted[0]
                        and at > _video_media_t[0] + _AUDIO_PACE_LOOKAHEAD_S
                    ):
                        _audio_pace_settled.set()
                        _owner_ready.set()
                        _video_progress.clear()
                        await _video_progress.wait()
                    # A LIVE mic stamps chunks on the wall clock; a divergence past the AUDIO tolerance
                    # (wider than video — a backlogged chunk behind a blocked pump is legitimate) trips once.
                    if live_audio and audio_tb is not None and audio_tb.resolved == "wall":
                        await _assert_timebase(
                            at, getattr(core, "clock", None),
                            tol=_AUDIO_SKEW_TOL_S, path="audio", emitted=_skew_emitted, on_health=on_health,
                        )
                    # MUTED (RuntimeSession.detach_audio_source): keep draining the mic stream but stop
                    # FEEDING it to perception — the fused judge goes video-only and the energy/onset gate
                    # signals freeze, without ending the session. Default (muted=None) is byte-for-byte.
                    if muted is not None and muted():
                        continue
                    boundary_published = (
                        source_preview.resets_continuity
                        or (
                            compiled_audio_capability is not None
                            and unit_preview.discontinuous
                        )
                    )
                    if boundary_published:
                        _owner_queue.publish(
                            "AudioBoundary",
                            {
                                "reasons": source_preview.events,
                                "boundary": unit.t_start,
                            },
                        )
                    _owner_queue.publish(
                        "AudioAccepted",
                        {
                            "envelope": envelope,
                            "unit": unit,
                            "unit_preview": unit_preview,
                            "at": at,
                            "chunk": chunk,
                            "energy": energy,
                            "require_source_time": require_source_time,
                            "boundary_published": boundary_published,
                            "wall_now": (
                                wall_clock()
                                if _vad is not None
                                and energy is not None
                                and (
                                    stale_audio_s is not None
                                    or deaf_until is not None
                                )
                                else None
                            ),
                        },
                    )
                    committed_preview = _audio_capture_tracker.inspect(
                        envelope, require_source_time=require_source_time
                    )
                    if not committed_preview.accepted:
                        raise RuntimeError(
                            "audio capture ordering changed between preview and publish"
                        )
                    if submitted_unit is not None or ingress.legacy_item is not None:
                        synthetic_carrier_sequence += 1
                    _owner_ready.set()
                    continue
                _owner_queue.publish("AudioEOS", {"source": gate_source})
                _audio_pace_settled.set()
                _owner_ready.set()

            audio_task = asyncio.ensure_future(_pump_audio())
            audio_task.add_done_callback(_task_watchdog("audio_pump_died"))
            audio_task.add_done_callback(
                lambda _done: (
                    _owner_ready.set(),
                    _audio_pace_settled.set(),
                )
            )

        async def _drain_owner_commands() -> None:
            """Apply every published command on the sole runtime owner."""
            for command in _owner_queue.drain():
                if command.kind == "AudioBoundary":
                    await _apply_audio_boundary(command.payload)
                    continue
                if command.kind == "AudioAccepted":
                    await _apply_audio_media(command.payload)
                    continue
                if command.kind == "AudioRejected":
                    await _apply_audio_rejected(command.payload)
                    continue
                if command.kind == "AudioEOS":
                    _audio_eos_seen[0] = True
                    continue
                raise RuntimeError(
                    f"unknown runtime owner command {command.kind!r}"
                )

        def _materialize_fires(fires, *, at, from_boundary=False):
            """Freeze one owner reduction and its counters without invoking callbacks."""
            def _snippet_for_fire(fire):
                exact_audio_units = tuple(
                    getattr(fire, "audio_units", None) or ()
                )
                if exact_audio_units:
                    from percept.audio_input import audio_snippet_from_units

                    audio_goal = core.goals.get(fire.goal_id)
                    audio_binding = core._audio_judge_binding_for(audio_goal)
                    return audio_snippet_from_units(
                        exact_audio_units,
                        capability=audio_binding.capability,
                    )
                return snippet_from_window(
                    clips, kind=snippet_kind, pre_s=snippet_pre_s
                )

            materialized = _materialize_reduction(
                fires,
                at=at,
                gate_source=gate_source,
                snippet_for_fire=_snippet_for_fire,
                watchplan_id=_current_watchplan_id(),
                label_store=label_store,
                disposition_log=disposition_log,
                verdict_ref=_verdict_ref,
                from_boundary=from_boundary,
            )
            if stats is not None:
                stats.asks += sum(
                    item.action == ASK for item in materialized.items
                )
                stats.fires += sum(
                    item.action == FIRE for item in materialized.items
                )
            return materialized

        async def _emit_materialized(materialized):
            delivered = await _deliver_materialized(
                materialized,
                core=core,
                on_health=on_health,
                on_ask=on_ask,
                on_incident=on_incident,
                sink_errors=sink_errors,
                stats=stats,
                coverage=_cov,
            )
            for incident in delivered:
                yield incident

        async def _emit_fires(fires, *, at, from_boundary=False):
            """Materialize the whole owner reduction, then deliver it."""
            materialized = _materialize_fires(
                fires,
                at=at,
                from_boundary=from_boundary,
            )
            async for incident in _emit_materialized(materialized):
                yield incident

        async def _emit_fused_completion(completion, *, at):
            token = completion.job.token
            synchronizer = (_fused_syncs or {}).get(token.goal_id)
            live_generation = (
                synchronizer.generation if synchronizer is not None else -1
            )
            fires = fused_reduce(
                completion,
                live_fused_generation=live_generation,
            )
            # Freeze the Incident before any callback, while retaining the
            # causal trace order: cognition rows precede incident.created.
            materialized = _materialize_fires(fires, at=at)
            drain_records = getattr(core, "drain_fused_trace_records", None)
            emit_trace = getattr(core, "_emit_trace", None)
            if callable(drain_records) and callable(emit_trace):
                for trace_id, trace_t, stage, data in drain_records():
                    await emit_trace(trace_id, trace_t, stage, **data)
            await _drain_judge_health(core, on_health, at)
            async for incident in _emit_materialized(materialized):
                yield incident

        async def _prepare_split_jobs(
            frame,
            goal_ids,
            *,
            at,
            revision,
        ):
            """Run observation projection, then the generic route preparer.

            ``prepare_transition_judgment`` and ``prepare_process_judgment``
            are reached through ``prepare_judgment``; provider dispatch and
            ``reduce_judgment_completion`` remain separate owner phases.
            """
            generation = core.source_continuity_generation(gate_source)
            candidate = core.prepare_observation_candidate(
                frame,
                gate_source,
                t=at,
                expected_goals_revision=revision,
                expected_source_generation=generation,
                goal_ids=goal_ids,
            )
            if candidate is None or not core.launch_observation_candidate(
                candidate,
                runtime_trace=True,
            ):
                return ()
            observation = await core.dispatch_observation_candidate(candidate)
            reduction = core.reduce_observation_completion(observation)
            if reduction.failure is not None:
                raise RuntimeError(
                    "observation provider failed: "
                    f"{reduction.failure.exception_type}"
                )
            if reduction.stale_reason is not None:
                return ()
            return tuple(
                job
                for goal_id in goal_ids
                if (
                    job := core.prepare_judgment(
                        reduction,
                        goal_id=goal_id,
                        temporal_window=tuple(clips.frames()),
                    )
                )
                is not None
            )

        async def _emit_judgment_completions(
            completions,
            *,
            at,
            scheduler=None,
        ):
            fires = []
            for completion in sorted(
                completions,
                key=lambda item: (
                    item.completion_ordinal
                    if item.completion_ordinal is not None
                    else -1
                ),
            ):
                fires.extend(core.reduce_judgment_completion(completion))
            if scheduler is not None:
                scheduler.owner_reduced()
            # Reduction and Incident construction are synchronous owner work.
            # Deliver their already-frozen trace rows in causal order before
            # incident.created and subscriber callbacks.
            materialized = _materialize_fires(fires, at=at)
            drain_rows = getattr(core, "drain_judgment_trace_rows", None)
            emit_trace = getattr(core, "_emit_trace", None)
            if callable(drain_rows) and callable(emit_trace):
                for trace_id, trace_t, stage, data in drain_rows():
                    await emit_trace(trace_id, trace_t, stage, **data)
            await _drain_judge_health(core, on_health, at)
            async for incident in _emit_materialized(materialized):
                yield incident

        async def _owner_driven_video():
            """Yield video while servicing owner commands during source stalls."""
            iterator = _aiter(
                source, capture_fps, **source_kwargs
            ).__aiter__()
            next_envelope = asyncio.create_task(anext(iterator))
            owner_wakeup = None
            try:
                while True:
                    owner_wakeup = asyncio.create_task(_owner_ready.wait())
                    done, _pending = await asyncio.wait(
                        (next_envelope, owner_wakeup),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if owner_wakeup in done:
                        _owner_ready.clear()
                        await _drain_owner_commands()
                        # Newly materialized utterance handlers run off-owner;
                        # give them their first turn while capture continues.
                        await asyncio.sleep(0)
                    else:
                        owner_wakeup.cancel()
                        await asyncio.gather(
                            owner_wakeup, return_exceptions=True
                        )
                    if next_envelope not in done:
                        continue
                    try:
                        envelope = next_envelope.result()
                    except StopAsyncIteration:
                        return
                    yield envelope
                    next_envelope = asyncio.create_task(anext(iterator))
            finally:
                if owner_wakeup is not None and not owner_wakeup.done():
                    owner_wakeup.cancel()
                if not next_envelope.done():
                    next_envelope.cancel()
                await asyncio.gather(
                    *(
                        task
                        for task in (owner_wakeup, next_envelope)
                        if task is not None
                    ),
                    return_exceptions=True,
                )

        _was_paused = False
        _fused_was_paused = False
        video_stream = (
            _owner_driven_video()
            if audio_source is not None
            else _aiter(source, capture_fps, **source_kwargs)
        )
        async for envelope in video_stream:
            await _drain_owner_commands()
            if _fused_syncs and _fused_requires_envelope_clock_claim and (
                envelope.clock_authority != _video_clock_authority
                or envelope.clock_confidence is None
                or float(envelope.clock_confidence) != _fused_clock_confidence
            ):
                raise ValueError(
                    "fused video envelope changed or omitted its qualified "
                    "clock authority/confidence"
                )
            require_source_time = video_tb.resolved == "media"
            if envelope.modality in ("video", "audio") and envelope.modality != "video":
                inspection = _video_source_tracker.exclude("source_modality_mismatch")
            else:
                preview = _video_source_tracker.preview(
                    envelope, require_source_time=require_source_time
                )
                if (
                    _fused_syncs
                    and preview.accepted
                    and _video_source_id is not None
                    and envelope.source_id != _video_source_id
                ):
                    raise ValueError(
                        "one fused runtime worker accepts exactly one video source_id; "
                        f"got {_video_source_id!r} then {envelope.source_id!r}"
                    )
                inspection = _video_source_tracker.inspect(
                    envelope, require_source_time=require_source_time
                )
            if inspection.accepted:
                if _video_source_id is None:
                    _video_source_id = envelope.source_id
                elif envelope.source_id != _video_source_id:
                    raise ValueError(
                        "one runtime worker accepts exactly one video source_id; "
                        f"got {_video_source_id!r} then {envelope.source_id!r}"
                    )
                if _interest_tracker is not None:
                    assert _interest_subscription is not None
                    # Record immediately after SourceTracker acceptance: a later hook/gate failure is
                    # closed at this accepted control coordinate, while interval_end remains the last
                    # unit that was actually admitted.
                    _interest_tracker.record_accepted(envelope)
                    if inspection.resets_continuity:
                        InterestSubscription._publish(
                            _interest_subscription,
                            _interest_tracker.discontinuity(envelope, inspection),
                        )
            if inspection.resets_continuity:
                # This must precede health-sink awaits and result draining: both yield control and can
                # let a cancellation-resistant stale judge finish.
                if _fused_syncs:
                    boundary = (
                        envelope.source_time
                        if require_source_time
                        and isinstance(envelope.source_time, (int, float))
                        and not isinstance(envelope.source_time, bool)
                        and math.isfinite(float(envelope.source_time))
                        else envelope.observed_time
                    )
                    _advance_fused_generation(
                        _fused_reason(inspection.events),
                        boundary=float(boundary),
                    )
                _advance_source_continuity()
                if _judge_sched is not None:
                    _judge_sched.invalidate_source_continuity(inspection.events)
                if _audio_judge_sched is not None:
                    _audio_judge_sched.invalidate_source_continuity(
                        inspection.events
                    )
                for scheduler in (_fused_judge_scheds or {}).values():
                    scheduler.invalidate_source_continuity(inspection.events)
            t = (
                envelope.source_time
                if require_source_time
                and isinstance(envelope.source_time, (int, float))
                and not isinstance(envelope.source_time, bool)
                else envelope.observed_time
            )
            if not (
                isinstance(t, (int, float))
                and not isinstance(t, bool)
                and math.isfinite(float(t))
            ):
                t = None
            fused_anchors = {}
            if inspection.accepted and t is not None:
                for goal_id, synchronizer in (_fused_syncs or {}).items():
                    accepted_video = _AcceptedMediaUnit(
                        modality="video",
                        payload=envelope.payload,
                        source_id=envelope.source_id,
                        source_epoch=envelope.epoch,
                        source_sequence=envelope.sequence,
                        generation=synchronizer.generation,
                        watch_application=synchronizer.watch_application,
                        watch_revision=synchronizer.watch_revision,
                        event_start=float(t),
                        event_end=float(t),
                        observed_start=envelope.observed_time,
                        observed_end=envelope.observed_time,
                        clock_domain=(
                            "source" if video_tb.resolved == "media" else "observed"
                        ),
                        clock_authority=_video_clock_authority,
                        clock_confidence=_fused_clock_confidence,
                        provenance={
                            "adapter": "runtime.video",
                            "format": {
                                "media_type": envelope.format.media_type,
                                "width": envelope.format.width,
                                "height": envelope.format.height,
                            },
                            "source_time": envelope.source_time,
                            "observed_time": envelope.observed_time,
                        },
                        discontinuity=(
                            _fused_reason(inspection.events)
                            if inspection.resets_continuity
                            else _FusedDiscontinuity.NONE
                        ),
                        # SourceTracker has proven this carrier's event ordering final.
                        source_watermark=float(t),
                    )
                    if synchronizer.accept(accepted_video):
                        if synchronizer.generation_start_event_time is None:
                            synchronizer.set_generation_start(float(t))
                        fused_anchors[goal_id] = accepted_video
            await _record_source_inspection(envelope, inspection, ts=t)
            # Async judge dispatch: surface everything the off-loop scheduler finished since the last
            # frame — fires route through the SAME _emit_fires machinery at their ADMISSION time `jt`
            # (incident.ts is the admitting frame, not the completion time). drain() also re-raises a
            # held judge failure here, so the fail-loud runtime_error path below is preserved. Runs
            # FIRST (before the paused check): these judgments were spent before any pause.
            if _judge_sched is not None:
                # ADR-001 §9: surface every TRACED bound-driven drop (superseded / revision-dropped
                # pending) as its own health event BEFORE draining results, so a drop is observable
                # even on a tick that also carries a fresh fire. drain_trace() always runs (even with
                # on_health=None) so the trace list never accumulates past one tick's worth.
                for _hk, _hd in _judge_sched.drain_trace():
                    if on_health is not None:
                        await _call_sink(on_health, RuntimeHealthEvent(_hk, ts=t, data=_hd))
                for _jt, _completions in _judge_sched.drain():
                    async for incident in _emit_judgment_completions(
                        _completions,
                        at=_jt,
                        scheduler=_judge_sched,
                    ):
                        yield incident
                # Surface any judge_degraded/judge_persistent_failure events queued by the
                # perceive_judged calls that just drained above (core is shared across dispatch modes).
                await _drain_judge_health(core, on_health, t)
            if _audio_judge_sched is not None:
                for _hk, _hd in _audio_judge_sched.drain_trace():
                    if on_health is not None:
                        await _call_sink(
                            on_health, RuntimeHealthEvent(_hk, ts=t, data=_hd)
                        )
                for _jt, _jfires in _audio_judge_sched.drain():
                    async for incident in _emit_fires(_jfires, at=_jt):
                        yield incident
                await _drain_judge_health(core, on_health, t)
            while _fused_local_completions:
                _local_t, _local_completion = _fused_local_completions.popleft()
                async for incident in _emit_fused_completion(
                    _local_completion, at=_local_t
                ):
                    yield incident
            for scheduler in (_fused_judge_scheds or {}).values():
                for _hk, _hd in scheduler.drain_trace():
                    if on_health is not None:
                        await _call_sink(on_health, RuntimeHealthEvent(_hk, ts=t, data=_hd))
                for _jt, _completion in scheduler.drain():
                    async for incident in _emit_fused_completion(
                        _completion, at=_jt
                    ):
                        yield incident
            if not inspection.accepted:
                continue
            assert t is not None  # accepted envelope validation guarantees a finite lane clock
            gate_frame, judge_frame = envelope.signal, envelope.payload
            if inspection.resets_continuity:
                gate.reset()
                for bank in banks.values():
                    bank.reset()
                rhythm_drive.reset()  # cold-start the measuring eye; the counted reps still happened
                for controller in coasts.values():
                    controller.reset()
                clips.clear()
                _cap_t0 = None
                _cap_n = 0
                _cap_unmet_emitted = False
            # Surface any gate SUPPRESSION queued since the last tick (require_false_first /
            # refractory withholding a fire the baseline gate would have delivered). Drained here —
            # UNCONDITIONALLY, outside the dispatch-mode branch above and BEFORE the per-frame
            # `continue` paths — because the absence/coverage/episode checks below evaluate the gate on
            # every tick, not only on frames that escalate to a judge. Both guards are off by default,
            # so this drains an empty queue and emits nothing unless a profile turned one on.
            _last_t = t
            await _drain_gate_health(core, on_health, t)
            # Preserve both clocks exactly as supplied.  In particular, a live source with no source
            # clock keeps ``last_media_t=None``; elapsed observation time is not relabelled as media.
            if stats is not None:
                stats.last_media_t = envelope.source_time
                stats.last_wall_t = envelope.observed_time
            if _audio_paced:
                # Advance the marker the paced audio pump awaits, to THIS frame's media time —
                # A media-selected `t` is the source position.  For the legacy mixed-clock case, the
                # elapsed observation offset is used only to pace the local pump and is never exposed
                # as source provenance.
                _video_media_t[0] = (
                    t if (video_tb.resolved if live_video else "media") == "media"
                    else ((t - _clock_t0[0]) if _clock_t0[0] is not None else 0.0)
                )
                if audio_task is not None and not audio_task.done():
                    _audio_pace_settled.clear()
                    _owner_ready.clear()
                    _video_progress.set()
                    # Wait until the capture-only publisher reaches the next
                    # beyond-lookahead carrier (or EOS), then commit the
                    # complete ready prefix before this video coordinate can
                    # judge it.
                    while not _audio_pace_settled.is_set():
                        await _owner_ready.wait()
                        _owner_ready.clear()
                        await _drain_owner_commands()
                        if audio_task.done():
                            break
                else:
                    _video_progress.set()
                await _drain_owner_commands()
                if audio_task is not None and audio_task.done():
                    audio_task.result()
            if _clock_t0[0] is None:
                _clock_t0[0] = t
            # On a WALL-timebase source, frame `t` IS the wall clock — if it has drifted from the core
            # clock past the video tolerance, the timelines the gate/deadlines share have come apart. Emit
            # once. A source-time path is intentionally distinct from the core clock and is never checked.
            if live_video and video_tb.resolved == "wall":
                await _assert_timebase(
                    t, getattr(core, "clock", None),
                    tol=_VIDEO_SKEW_TOL_S, path="video", emitted=_skew_emitted, on_health=on_health,
                )
            # Pick up goals armed/cancelled mid-run (e.g. by voice via RuntimeSession.arm)
            # by recompiling the gate+banks when the registry revision moved. Default ON (a goal armed
            # mid-run MUST reach the gate — the proactive premise). Capture fps is fixed for the open
            # source; only the escalation logic re-arms (the important part for arm-at-runtime). When
            # regate is explicitly OFF and goals STILL move, that is the silent-default trap the shipped
            # realtime demo hit (goal lands in core, gate never re-arms, nothing ever fires) — so we fail
            # LOUD (warning + health event) instead of doing nothing.
            if goals_revision is not None:
                rev = getattr(core.goals, "revision", goals_revision)
                if rev != goals_revision:
                    current_fused, _ = _live_watch_topology()
                    if _fused_syncs:
                        _advance_fused_generation(
                            _FusedDiscontinuity.WATCH_REVISION,
                            revision=rev,
                        )
                        for scheduler in (_fused_judge_scheds or {}).values():
                            scheduler.invalidate_source_continuity(
                                ("watch_revision",)
                            )
                    current_fused = {goal.id: goal for goal in current_fused}
                    if current_fused and not fused_goals:
                        raise ValueError(
                            "a fused Watch must be armed before percept.run starts so both media "
                            "lanes and the frozen synchronization policy can be preflighted"
                        )
                    if _fused_syncs is not None:
                        for removed_id in set(_fused_syncs) - set(current_fused):
                            _fused_syncs.pop(removed_id).advance_generation(
                                _FusedDiscontinuity.WATCH_REVISION,
                                watch_revision=rev,
                            )
                            removed_scheduler = (_fused_judge_scheds or {}).pop(
                                removed_id, None
                            )
                            if removed_scheduler is not None:
                                removed_scheduler.invalidate_source_continuity(
                                    ("watch_revision",)
                                )
                        for added_id in set(current_fused) - set(_fused_syncs):
                            added_goal = current_fused[added_id]
                            _fused_syncs[added_id] = _new_fused_synchronizer(
                                added_goal, rev
                            )
                            assert _fused_judge_scheds is not None
                            _fused_judge_scheds[added_id] = _LegacyJudgeScheduler(
                                core,
                                gate_source,
                                _cov,
                                perceive=_dispatch_prepared_fused_job,
                                stall_after_s=judge_stall_after_s,
                                backpressure_after=judge_backpressure_after,
                                decision_deadline_s=judge_decision_deadline_s,
                            )
                    if regate_on_goal_change:
                        # Pass the live gate + banks + rhythm drive so the rebuild ADOPTS per-goal
                        # throttle + prev frame and REUSES surviving goals' detector banks and
                        # measuring eyes (no throttle release / rep drop). Coast + required_fps are
                        # rebuilt too (a mid-run brief goal gets its coast).
                        spec, gate, banks, coasts, required_fps, rhythm_drive = _build_gate(
                            gate, banks, rhythm_drive
                        )
                    else:
                        _log.warning(
                            "goals_changed_without_regate: core.goals.revision moved %s->%s while "
                            "regate_on_goal_change=False — the running gate will NOT re-arm, so the "
                            "new/changed goal cannot fire. Pass regate_on_goal_change=True "
                            "(RuntimeSession.start does this for you).",
                            goals_revision, rev,
                        )
                        if on_health is not None:
                            await _call_sink(on_health, RuntimeHealthEvent(
                                "goals_changed_without_regate", ts=t,
                                data={"from_revision": goals_revision, "to_revision": rev}))
                    goals_revision = rev
            # A brief goal promised >= required_fps CAPTURED frames (its window Nyquist). Measure the
            # OBSERVED capture cadence over a short window of real frame timestamps; if the open source
            # can't meet it, REPORT it once (never silently downgrade). Reset when a regate raises the
            # requirement. No brief goal -> required_fps == 0 -> this whole block is skipped.
            if required_fps > 0.0:
                if required_fps > _cap_req_seen:  # a (new/more-demanding) brief goal -> fresh window
                    _cap_req_seen = required_fps
                    _cap_t0 = t
                    _cap_n = 0
                    _cap_unmet_emitted = False
                _cap_n += 1
                if not _cap_unmet_emitted and _cap_t0 is not None:
                    _span = t - _cap_t0
                    if _span >= _CAPTURE_UNMET_MIN_SPAN_S and _cap_n >= 2:
                        _observed = (_cap_n - 1) / _span
                        if _observed < required_fps * _CAPTURE_UNMET_TOL:
                            _cap_unmet_emitted = True
                            if on_health is not None:
                                await _call_sink(on_health, RuntimeHealthEvent(
                                    "capture_plan_unmet", ts=t,
                                    data={"required_fps": required_fps, "observed_fps": _observed,
                                          "capture_fps": capture_fps}))
            # Surface any capture-thread health events (e.g. a source whose DELIVERED fps diverges
            # from its reported CAP_PROP_FPS — the previously-silent half-rate decimation). Drained on the
            # frame cadence so it fails LOUD through on_health instead of sitting in an unread list.
            if source_health and on_health is not None:
                while source_health:
                    _hk, _hd = source_health.pop(0)
                    await _call_sink(on_health, RuntimeHealthEvent(_hk, ts=t, data=_hd))
            # Same drain for a route a MID-RUN regate found armed on this source and skipped.
            # Drains UNCONDITIONALLY (like the judge/gate health queues) so the list never lingers.
            while _route_health:
                _rd = _route_health.pop(0)
                if on_health is not None:
                    await _call_sink(on_health, RuntimeHealthEvent(
                        "undriven_behavior_route", ts=t, data=_rd))
            if _funnel is not None:
                _funnel.seen(camera=gate_source)  # the ungated baseline: a frame was observed
            _cov.frames += 1
            _cov.frames_dropped = dropped[0]
            clips.push(t, judge_frame)
            is_paused = bool(paused is not None and paused())
            if _fused_syncs and is_paused != _fused_was_paused:
                _advance_fused_generation(
                    _FusedDiscontinuity.PAUSE
                    if is_paused
                    else _FusedDiscontinuity.RESUME
                )
                for scheduler in (_fused_judge_scheds or {}).values():
                    scheduler.invalidate_source_continuity(
                        ("runtime_pause" if is_paused else "runtime_resume",)
                    )
                _fused_was_paused = is_paused
            # A pre-roll frame dropped by the BYTE budget is announced on the frame cadence, with its
            # reason and the live held bytes/frames — the operator sees the snippet getting shorter
            # instead of silently receiving less evidence than the configured pre_s (ADR-001 §9).
            if preroll_health:
                while preroll_health:
                    _pk, _pd = preroll_health.pop(0)
                    if on_health is not None:
                        await _call_sink(on_health, RuntimeHealthEvent(_pk, ts=t, data=_pd))
            if on_frame is not None:
                await _call_sink_with_policy(
                    on_frame, t, judge_frame, errors=sink_errors, stats=stats, on_health=on_health, health_kind="frame_sink_error"
                )

            # SLEEP (RuntimeSession.pause): keep capturing + streaming frames to subscribers (the camera
            # panel stays live) but spend NO gate/judge/incident this tick — a cost-free pause that resumes
            # instantly. Default (paused=None) is byte-for-byte.
            if is_paused != _was_paused:
                # Neither a pause nor a resume is evidence that a condition remained continuously
                # true. Pending dwell is invalidated on both boundaries without re-arming fired state.
                if callable(clear_pending_dwell):
                    clear_pending_dwell(gate_source)
                # Neither side of a pause may diff across the uninspected interval.
                gate.reset()
                for bank in banks.values():
                    bank.reset()
                rhythm_drive.reset()  # neither side of a pause may diff across the uninspected gap
                for controller in coasts.values():
                    controller.reset()
                if is_paused and _interest_tracker is not None:
                    assert _interest_subscription is not None
                    InterestSubscription._publish(
                        _interest_subscription,
                        _interest_tracker.pause(envelope, inspection),
                    )
                _was_paused = is_paused
            if is_paused:
                continue

            # boundary first: it must see quiet frames too (the episode ends when activity stops)
            source_goal_ids = tuple(goal.id for goal in _source_goals())
            if check_boundary is not None:
                boundary_kwargs = {"t": t, "inactivity_s": episode_inactivity_s}
                if boundary_accepts_goal_ids:
                    boundary_kwargs["goal_ids"] = source_goal_ids
                boundary_fires = await check_boundary(**boundary_kwargs)
                if not boundary_accepts_goal_ids:
                    boundary_fires = _filter_goal_events(boundary_fires, source_goal_ids)
                async for incident in _emit_fires(
                    boundary_fires, at=t, from_boundary=True
                ):
                    yield incident
            # Absence-native fires — coverage loss ("I can't see the baby") + overdue deadlines.
            # Sync; ride every frame so they fire on quiet frames the gate skips. from_boundary=True marks
            # that they did NOT pass through perceive_judged (no upstream judge stage in their trace).
            if check_coverage is not None:
                coverage_kwargs = {"t": t, "staleness_s": coverage_staleness_s}
                if coverage_accepts_goal_ids:
                    coverage_kwargs["goal_ids"] = source_goal_ids
                coverage_fires = check_coverage(**coverage_kwargs)
                if not coverage_accepts_goal_ids:
                    coverage_fires = _filter_goal_events(coverage_fires, source_goal_ids)
                async for incident in _emit_fires(
                    coverage_fires, at=t, from_boundary=True
                ):
                    yield incident
            if check_deadlines is not None:
                deadline_kwargs = {"t": t}
                if deadlines_accepts_goal_ids:
                    deadline_kwargs["goal_ids"] = source_goal_ids
                deadline_fires = check_deadlines(**deadline_kwargs)
                if not deadlines_accepts_goal_ids:
                    deadline_fires = _filter_goal_events(deadline_fires, source_goal_ids)
                async for incident in _emit_fires(
                    deadline_fires, at=t, from_boundary=True
                ):
                    yield incident

            # Feed the latest audio energy so a goal with a WakePolicy can fuse it; None (no audio_source)
            # leaves the gate byte-for-byte the visual-only path. Read it ONLY while fresh — a dead/
            # stalled mic leaves energy pinned at its last (speech-level) value, which would otherwise keep
            # escalating the gate forever. A stale reading reads None → the gate degrades to VIDEO-ONLY.
            energy_now = audio["energy"] if _energy_fresh(audio["energy_t"], t, audio["chunk_dt"]) else None
            # Channel B without an audio_source pump (e.g. the browser transport owns the mic and tees it
            # in via _AudioTap → push_audio): the caller supplies tier-0 energy through `energy_provider`,
            # so a WakePolicy can escalate a near-still frame on SOUND. None (the default, or a provider
            # returning None) leaves the gate byte-for-byte the video-only path.
            if energy_now is None and energy_provider is not None:
                try:
                    ep = energy_provider()
                    energy_now = float(ep) if ep is not None else None
                except Exception:  # noqa: BLE001 - a bad energy provider must never crash the watch
                    energy_now = None
            decision = gate.evaluate(
                gate_frame,
                ts=t,
                energy=energy_now,
                include_evidence_id=_interest_tracker is not None,
                include_outcomes=_interest_tracker is not None,
            )

            # Run edge detectors on the cheap motion/energy signals at native fps. Each emitted
            # boundary (a rep peak, an acoustic onset) is a distinct occurrence fed to the brain via
            # perceive_boundary (edge proposes · brain counts) — firing per the goal's accumulate.fire_when
            # WITHOUT a cloud judge. No banks -> this whole block is skipped (byte-for-byte the pre-PR path).
            if banks and perceive_boundary is not None:
                for goal_id, bank in banks.items():
                    for b in bank.push(t, motion=decision.signal.motion, frame=judge_frame, energy=energy_now):
                        if stats is not None:
                            stats.boundaries += 1
                        fired = perceive_boundary(goal_id, t=b.t)
                        if fired is not None:
                            async for incident in _emit_fires([fired], at=b.t):
                                yield incident
            # The RHYTHM sense (percept.rhythm): measure this frame's decomposed channels and let the
            # brain confirm reps causally. Runs on EVERY frame at native rate — never tier-0 gated,
            # never judged — so the count is bounded by the frame rate, not by judgment density. Reps
            # land on goal.state["count"]; an Incident is minted only for an authored count MILESTONE
            # (Goal.state.fire_when), never once per rep. No rhythm goal -> the block is skipped.
            if rhythm_drive:
                rhythm_fires = rhythm_drive.observe(t, judge_frame)
                if rhythm_fires:
                    # from_boundary=True: like the edge-boundary and absence fires, a rhythm milestone
                    # never passed through perceive_judged, so its trace must not claim a judge stage.
                    async for incident in _emit_fires(rhythm_fires, at=t, from_boundary=True):
                        yield incident
            # A brief-action Watch's own fovea coast admits only that Watch. Each controller carries
            # its own cadence state, so a dense Watch cannot spend judgment calls for a sparse one.
            coast_goal_ids = [
                goal_id
                for goal_id, controller in coasts.items()
                if controller.should_process(goal_id, t, float(decision.signal.motion))
            ]
            coast_admit = bool(coast_goal_ids)
            # An audio onset admits a frame the visual gate would skip (off-camera cry / glass break). One-shot:
            # consumed here so a single onset causes one judge, not a run of them (AcousticOnset hysteresis +
            # refractory already debounce the signal itself).
            audio_admit = bool(audio["wake"]) or (audio["wake_at"] is not None and t >= audio["wake_at"])
            audio["wake"] = False
            if audio_admit:
                audio["wake_at"] = None
            admitted_goal_ids = list(decision.goal_ids)
            admitted_causes = list(decision.admission_causes)
            if coast_admit:
                for goal_id in coast_goal_ids:
                    if goal_id not in admitted_goal_ids:
                        admitted_goal_ids.append(goal_id)
                        admitted_causes.append("coast")
            if audio_admit:
                audio_goals = [
                    goal
                    for goal in _source_goals()
                    if getattr(program_of(goal), "perception", None).modality in ("audio", "fused")
                ]
                # Backward compatibility for an explicitly configured audio wake on legacy
                # vision-only goals: when no Watch declares an audio modality, the wake retains its
                # historical all-goals behavior. Once audio/fused Watches exist, only they consume it.
                externally_admitted = gate.admit_external(
                    [goal.id for goal in (audio_goals or _source_goals())],
                    ts=t,
                )
                for goal_id in externally_admitted:
                    if goal_id not in admitted_goal_ids:
                        admitted_goal_ids.append(goal_id)
                        admitted_causes.append("audio")
            # A compiled audio/fused Watch may never fall through to a provider with ``audio=None``:
            # that changes the question into vision-only perception. Drop only the affected Watch,
            # report the missing channel once, and keep this source worker alive for other Watches.
            missing_audio = [
                goal_id
                for goal_id in admitted_goal_ids
                if not _compiled_audio_ready(goal_id)
            ]
            for goal_id in missing_audio:
                missing_index = admitted_goal_ids.index(goal_id)
                admitted_goal_ids.pop(missing_index)
                admitted_causes.pop(missing_index)
                if goal_id not in _audio_missing_reported:
                    _audio_missing_reported.add(goal_id)
                    _log.warning(
                        "compiled_audio_evidence_unavailable: goal=%r source=%r; "
                        "skipping judgment rather than degrading to vision-only",
                        goal_id,
                        gate_source,
                    )
                    if on_health is not None:
                        await _call_sink(
                            on_health,
                            RuntimeHealthEvent(
                                "audio_evidence_unavailable",
                                ts=t,
                                data={"goal_id": goal_id, "source": gate_source},
                            ),
                        )
            if _interest_tracker is not None:
                assert _interest_subscription is not None
                if not decision.evidence_id:
                    raise RuntimeError("live VID tracking requires gate evidence identity")
                final_causes = dict(zip(admitted_goal_ids, admitted_causes))
                live_admissions: list[LiveInterestAdmission] = []
                for outcome in decision.outcomes:
                    final_cause = final_causes.get(outcome.goal_id)
                    if final_cause is None:
                        non_admission_reason = (
                            outcome.non_admission_reason
                            if not outcome.admitted
                            else "not_admitted"
                        )
                        live_admissions.append(
                            LiveInterestAdmission(
                                watch_id=outcome.goal_id,
                                admitted=False,
                                cause=None,
                                non_admission_reason=non_admission_reason,
                                score=outcome.score,
                                threshold=outcome.threshold,
                                evidence_id=decision.evidence_id,
                                admission_id=outcome.admission_id,
                                detector_id=outcome.detector_id,
                            )
                        )
                        continue
                    admission_id = (
                        outcome.admission_id
                        if final_cause in ("motion", "structural", "semantic", "fixed")
                        else f"runtime.admission.{final_cause}"
                    )
                    detector_id = (
                        outcome.detector_id
                        if final_cause in ("motion", "structural", "semantic", "fixed")
                        else f"runtime.{final_cause}"
                    )
                    live_admissions.append(
                        LiveInterestAdmission(
                            watch_id=outcome.goal_id,
                            admitted=True,
                            cause=final_cause,
                            non_admission_reason=None,
                            score=outcome.score,
                            threshold=outcome.threshold,
                            evidence_id=decision.evidence_id,
                            admission_id=admission_id,
                            detector_id=detector_id,
                        )
                    )
                InterestSubscription._publish(
                    _interest_subscription,
                    _interest_tracker.observe(
                        envelope,
                        live_admissions,
                        coverage_kind=inspection.coverage.kind,
                        missing_sequences=inspection.coverage.missing_sequences,
                        source_events=inspection.events,
                    )
                )
            emit_trace = getattr(core, "_emit_trace", None)
            if getattr(core, "on_trace", None) is not None and emit_trace is not None:
                # The tier-0 edge decision — signal, threshold, exact ordered goal ids + aligned
                # admission causes — emitted for EVERY frame, escalated or not, so "why didn't it fire"
                # can be answered even when the gate never spent a cloud call. trace_id matches the one
                # perceive_judged uses below (source+ts), so a dashboard can correlate the two.
                await emit_trace(
                    f"{gate_source}@{t}", t, "edge.decision",
                    motion=round(float(decision.signal.motion), 4),
                    structural=round(float(decision.signal.structural), 4),
                    semantic=round(float(decision.signal.semantic), 4),
                    threshold=round(float(decision.threshold), 4),
                    escalate=bool(decision.escalate),
                    coast_admit=coast_admit,
                    goal_ids=tuple(admitted_goal_ids),
                    admission_causes=tuple(admitted_causes),
                )
            # Remember WHAT this frame had to clear, so an end-of-run zero-admission report can name
            # the bar the scene never crossed (a regate can move it; the last one applied is the one
            # a reader should tune).
            _cov_threshold = float(decision.threshold)
            if not admitted_goal_ids:
                continue  # still / sub-cadence -> no cloud call, no incident
            if _funnel is not None:
                _funnel.gated(camera=gate_source)  # tier-0 escalated this frame — a cloud call will be spent
            _cov.escalations += 1
            for goal_id, cause in zip(admitted_goal_ids, admitted_causes):
                _cov.goal_admissions += 1
                _cov.admission_by_goal[goal_id] = _cov.admission_by_goal.get(goal_id, 0) + 1
                _cov.admission_by_cause[cause] = _cov.admission_by_cause.get(cause, 0) + 1
                if cause == "fixed":
                    _cov.fixed_opportunities += 1
            # Partition the admit into exactly one cause bucket (see RuntimeStats): a tuned-signal
            # escalation, else a fixed-cadence coast admit, else an audio wake. This mirrors the
            # `cause` string computed just below for the escalation health event.
            gate_causes = set(decision.admission_causes)
            if decision.escalate and gate_causes - {"fixed"}:
                _cov.signal_admits += 1
            elif decision.escalate and gate_causes == {"fixed"}:
                _cov.fixed_admits += 1
            elif coast_admit:
                _cov.coast_admits += 1
            if audio_admit and not (decision.escalate or coast_admit):
                _cov.audio_wakes += 1
            if on_health is not None:
                if audio_admit and not (decision.escalate or coast_admit):
                    cause = "audio"
                elif audio_admit and decision.escalate:
                    cause = "both"
                elif coast_admit and not decision.escalate:
                    cause = "coast"
                elif gate_causes == {"fixed"}:
                    cause = "fixed"
                else:
                    cause = "motion"
                _esc_data = {"source": gate_source, "cause": cause}
                if _judge_sched is not None:
                    # Live scheduler-shape proof (each 0/1 by construction): read BEFORE this frame's
                    # submit so the payload reflects the state the admit decision was made under.
                    _esc_data["judge_active"] = _judge_sched.active
                    _esc_data["judge_pending"] = _judge_sched.pending
                await _call_sink(on_health, RuntimeHealthEvent("escalation", ts=t, data=_esc_data))
            # The admit CONDITION above is unchanged by dispatch mode — only what "spend a judgment"
            # means changes. judge_calls is counted at LAUNCH either way (inline: launch == admit;
            # async: a superseded/revision-dropped candidate never spent a call — the scheduler counts).
            fused_admitted = [
                goal_id
                for goal_id in admitted_goal_ids
                if _fused_syncs is not None and goal_id in _fused_syncs
            ]
            for goal_id in fused_admitted:
                anchor = fused_anchors.get(goal_id)
                if anchor is None:
                    continue
                bundle = _fused_syncs[goal_id].bundle_for(anchor)
                for receipt in _fused_syncs[goal_id].drain_bound_drops():
                    if on_health is not None:
                        await _call_sink(
                            on_health,
                            RuntimeHealthEvent(
                                "fused_bounds_drop",
                                ts=t,
                                data={
                                    "goal_id": goal_id,
                                    "dropped_count": receipt.dropped_count,
                                    "latest_anchor_identity": receipt.latest_anchor_identity,
                                    "reason": receipt.reason.value,
                                },
                            ),
                        )
                if not any(
                    reason.value == "watermark_pending"
                    for reason in bundle.coverage_reasons
                ):
                    _queue_fused_bundle(goal_id, bundle)
            audio_goal_ids = tuple(
                goal_id
                for goal_id in admitted_goal_ids
                if (
                    (goal := core.goals.get(goal_id)) is not None
                    and program_of(goal).perception.modality == "audio"
                )
            )
            ordinary_goal_ids = tuple(
                goal_id
                for goal_id in admitted_goal_ids
                if (
                    (_fused_syncs is None or goal_id not in _fused_syncs)
                    and goal_id not in audio_goal_ids
                )
            )
            if audio_goal_ids:
                audio_evidence = _exact_audio_evidence(audio_goal_ids, at=t)
                admitted_audio_ids = tuple(
                    goal_id for goal_id in audio_goal_ids if goal_id in audio_evidence
                )
                if _audio_judge_sched is not None:
                    if admitted_audio_ids:
                        _audio_judge_sched.submit(
                            t, audio_evidence, admitted_audio_ids
                        )
                elif admitted_audio_ids:
                    _cov.perceive_dispatches += 1
                    _cov.judge_calls += 1
                    perceive_audio = getattr(core, "perceive_audio_judged")
                    fires = await perceive_audio(
                        audio_evidence,
                        gate_source,
                        t=t,
                        expected_goals_revision=goals_revision,
                        expected_source_generation=core.source_continuity_generation(
                            gate_source
                        ),
                        goal_ids=admitted_audio_ids,
                    )
                    async for incident in _emit_fires(fires, at=t):
                        yield incident
            if _judge_sched is not None:
                if ordinary_goal_ids:
                    jobs = await _prepare_split_jobs(
                        judge_frame,
                        ordinary_goal_ids,
                        at=t,
                        revision=goals_revision,
                    )
                    if jobs:
                        _judge_sched.submit(t, jobs)
                        # Give a non-blocking provider task one scheduling turn.
                        # Fast local/scripted adapters settle before the next
                        # frame; a slow provider remains off the capture path.
                        await asyncio.sleep(0)
                        await asyncio.sleep(0)
            elif ordinary_goal_ids:
                jobs = await _prepare_split_jobs(
                    judge_frame,
                    ordinary_goal_ids,
                    at=t,
                    revision=goals_revision,
                )
                launched = (
                    core.launch_judgments(jobs, runtime_trace=True)
                    if jobs
                    else ()
                )
                completions = []
                if launched:
                    _cov.judge_calls += 1
                for job in launched:
                    _cov.perceive_dispatches += 1
                    completions.append(await core.dispatch_judgment(job))
                async for incident in _emit_judgment_completions(
                    completions,
                    at=t,
                ):
                    yield incident
        # Source exhausted (finite file / test stream): a judgment may still be IN FLIGHT and one more
        # PENDING behind it. Flush to QUIESCENCE (completing the active may launch the pending — one
        # await is not enough) and emit everything, so a collect-all consumer sees every incident the
        # stream earned. Still inside the try: a held judge failure re-raised by drain() routes through
        # the fail-loud runtime_error path below, exactly like an inline judge failure.
        # The video source is done producing frames — release the paced audio pump
        # unconditionally so a video track shorter than its audio (or one that simply finished) drains
        # whatever it's still holding instead of parking on `_video_progress.wait()` forever.
        if _audio_paced:
            _video_exhausted[0] = True
            _video_progress.set()
            if audio_task is not None:
                await audio_task
        elif audio_task is not None and not live_audio:
            # A deterministic finite audio iterable is part of the same finite
            # replay. Join its capture-only publisher before the final owner
            # drain; a live mic remains cancellation-owned in ``finally``.
            try:
                await audio_task
            except Exception:
                # The watchdog already surfaced capture/publisher failure.
                pass
        await _drain_owner_commands()
        await asyncio.sleep(0)
        for goal_id, synchronizer in (_fused_syncs or {}).items():
            for final_bundle in synchronizer.close():
                _queue_fused_bundle(goal_id, final_bundle)
        while _fused_local_completions:
            _local_t, _local_completion = _fused_local_completions.popleft()
            async for incident in _emit_fused_completion(
                _local_completion, at=_local_t
            ):
                yield incident
        async def _flush_provider_lane(kind, goal_id, scheduler):
            await scheduler.flush()
            return kind, goal_id, scheduler

        provider_flush_tasks = []
        provider_flush_meta = {}
        for goal_id, scheduler in (_fused_judge_scheds or {}).items():
            task = asyncio.create_task(
                _flush_provider_lane("fused", goal_id, scheduler)
            )
            provider_flush_tasks.append(task)
            provider_flush_meta[task] = ("fused", goal_id, scheduler)
        if _audio_judge_sched is not None:
            task = asyncio.create_task(
                _flush_provider_lane("audio", None, _audio_judge_sched)
            )
            provider_flush_tasks.append(task)
            provider_flush_meta[task] = ("audio", None, _audio_judge_sched)
        if _judge_sched is not None:
            task = asyncio.create_task(
                _flush_provider_lane("ordinary", None, _judge_sched)
            )
            provider_flush_tasks.append(task)
            provider_flush_meta[task] = ("ordinary", None, _judge_sched)
        eos_timeout_s = max(0.01, float(judge_stall_after_s))
        eos_deadline = asyncio.get_running_loop().time() + eos_timeout_s
        try:
            for completed in asyncio.as_completed(
                provider_flush_tasks,
                timeout=eos_timeout_s,
            ):
                lane_kind, _goal_id, scheduler = await completed
                for _hk, _hd in scheduler.drain_trace():
                    if on_health is not None:
                        await _call_sink(
                            on_health, RuntimeHealthEvent(_hk, data=_hd)
                        )
                if lane_kind == "fused":
                    for _jt, _completion in scheduler.drain():
                        async for incident in _emit_fused_completion(
                            _completion, at=_jt
                        ):
                            yield incident
                else:
                    for _jt, _result in scheduler.drain():
                        if lane_kind == "ordinary":
                            async for incident in _emit_judgment_completions(
                                _result,
                                at=_jt,
                                scheduler=scheduler,
                            ):
                                yield incident
                        else:
                            async for incident in _emit_fires(
                                _result,
                                at=_jt,
                            ):
                                yield incident
            # Split ordinary transport cannot launch its accepted pending
            # batch until the owner reduces the preceding ready completion.
            # Repeat within the same absolute multi-lane deadline.
            while (
                _judge_sched is not None
                and (_judge_sched.active or _judge_sched.pending)
            ):
                remaining = (
                    eos_deadline - asyncio.get_running_loop().time()
                )
                if remaining <= 0:
                    raise TimeoutError
                await asyncio.wait_for(
                    _flush_provider_lane(
                        "ordinary", None, _judge_sched
                    ),
                    timeout=remaining,
                )
                for _hk, _hd in _judge_sched.drain_trace():
                    if on_health is not None:
                        await _call_sink(
                            on_health,
                            RuntimeHealthEvent(_hk, data=_hd),
                        )
                for _jt, _result in _judge_sched.drain():
                    async for incident in _emit_judgment_completions(
                        _result,
                        at=_jt,
                        scheduler=_judge_sched,
                    ):
                        yield incident
        except TimeoutError:
            # A cancellation-resistant provider lane must not hold completed
            # reflex lanes or end-of-stream delivery hostage.
            closed_schedulers = set()
            for task in provider_flush_tasks:
                if not task.done():
                    lane_kind, goal_id, scheduler = provider_flush_meta[task]
                    if on_health is not None:
                        await _call_sink(
                            on_health,
                            RuntimeHealthEvent(
                                "judge_eos_timeout",
                                ts=_last_t,
                                data={
                                    "lane": lane_kind,
                                    "goal_id": goal_id,
                                    "reason": "end_of_stream_deadline",
                                    "active": scheduler.active,
                                    "pending": scheduler.pending,
                                },
                            ),
                        )
                    task.cancel()
                    # Never await this wrapper: its provider may suppress
                    # cancellation.  The scheduler owns bounded teardown.
                    task.add_done_callback(
                        lambda done: (
                            None
                            if done.cancelled()
                            else done.exception()
                        )
                    )
                    await scheduler.close(drain=False)
                    closed_schedulers.add(scheduler)
            for lane_kind, goal_id, scheduler in provider_flush_meta.values():
                if (
                    scheduler in closed_schedulers
                    or not (scheduler.active or scheduler.pending)
                ):
                    continue
                if on_health is not None:
                    await _call_sink(
                        on_health,
                        RuntimeHealthEvent(
                            "judge_eos_timeout",
                            ts=_last_t,
                            data={
                                "lane": lane_kind,
                                "goal_id": goal_id,
                                "reason": "end_of_stream_deadline",
                                "active": scheduler.active,
                                "pending": scheduler.pending,
                            },
                        ),
                    )
                await scheduler.close(drain=False)
                closed_schedulers.add(scheduler)
            # Closing a lane creates typed cancellation/drop traces. Drain
            # those plus any result that completed immediately before the
            # deadline, including fast ordinary/reflex completions.
            for lane_kind, _goal_id, scheduler in provider_flush_meta.values():
                for _hk, _hd in scheduler.drain_trace():
                    if on_health is not None:
                        await _call_sink(
                            on_health, RuntimeHealthEvent(_hk, ts=_last_t, data=_hd)
                        )
                if lane_kind == "fused":
                    for _jt, _completion in scheduler.drain():
                        async for incident in _emit_fused_completion(
                            _completion, at=_jt
                        ):
                            yield incident
                else:
                    for _jt, _result in scheduler.drain():
                        if lane_kind == "ordinary":
                            async for incident in _emit_judgment_completions(
                                _result,
                                at=_jt,
                                scheduler=scheduler,
                            ):
                                yield incident
                        else:
                            async for incident in _emit_fires(
                                _result,
                                at=_jt,
                            ):
                                yield incident
        await _drain_judge_health(core, on_health, _last_t)
        # The LAST tick's suppressions (and anything the end-of-source flush above produced) would
        # otherwise die with the queue — drain once more so a finite stream reports every suppression
        # it earned, not all-but-the-final-tick's.
        await _drain_gate_health(core, on_health, _last_t)
    except Exception as exc:
        if _fused_syncs:
            _advance_fused_generation(_FusedDiscontinuity.WORKER_ERROR)
            for scheduler in (_fused_judge_scheds or {}).values():
                scheduler.invalidate_source_continuity(("worker_error",))
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_error", data={"error": repr(exc)}))
        raise
    finally:
        if _fused_syncs:
            _advance_fused_generation(_FusedDiscontinuity.CANCELLATION)
            for scheduler in (_fused_judge_scheds or {}).values():
                scheduler.invalidate_source_continuity(("runtime_shutdown",))
        # Async judge dispatch: never start new work past this point; cancel an in-flight judgment and
        # swallow its CancelledError (live shutdown / consumer break / cancellation). The normal
        # end-of-source path already flushed to quiescence above, so this is a no-op there.
        if _judge_sched is not None:
            await _judge_sched.close(drain=False)
        if _audio_judge_sched is not None:
            await _audio_judge_sched.close(drain=False)
        for scheduler in (_fused_judge_scheds or {}).values():
            await scheduler.close(drain=False)
        if audio_task is not None:  # stop the audio pump; a dangling capture thread outlives the run otherwise
            audio_task.cancel()
        for _t in list(_utt_tasks):  # an in-flight voice turn must not outlive the run
            _t.cancel()
        if autosave_task is not None:
            autosave_task.cancel()
        _cov.frames_dropped = dropped[0]
        # One last save on the way out so a clean shutdown loses nothing since the last autosave.
        if persistence is not None and getattr(core, "save_state", None):
            try:
                core.save_state(persistence)
            except Exception:  # noqa: BLE001 - teardown; a failed final save must not mask the real exit
                pass
        core.on_trace = _prev_on_trace  # restore — don't leak this run's tracer into the next run(core)
        if on_health is not None:
            # The audio twin of zero_admission_coverage below: an acoustic gate that was fed audio
            # and never woke says so, with its own thresholds beside the energies it actually saw.
            await _report_zero_wake(at_end=True)
            # A run's COVERAGE, stamped on the way out. `runtime_ended` used to carry an empty
            # payload, which made a run that admitted zero frames byte-for-byte indistinguishable
            # from a healthy one; a consumer had to have kept its own counters to know.
            _cov_data = {
                "frames_seen": _cov.frames,
                "admitted": _cov.escalations,
                # The per-frame admission split (escalations == signal_admits + fixed_admits +
                # coast_admits + audio_wakes): a consumer auditing fixed or strict admission reads
                # the exact bucket directly instead of inferring it from the total.
                "signal_admits": _cov.signal_admits,
                "fixed_admits": _cov.fixed_admits,
                "coast_admits": _cov.coast_admits,
                "audio_wakes": _cov.audio_wakes,
                "fixed_opportunities": _cov.fixed_opportunities,
                "goal_admissions": _cov.goal_admissions,
                "admission_by_goal": dict(_cov.admission_by_goal),
                "admission_by_cause": dict(_cov.admission_by_cause),
                "perceive_dispatches": _cov.perceive_dispatches,
                "judge_calls": _cov.judge_calls,
                "incidents": _cov.incidents,
                "source_coverage": {
                    "video": _video_source_tracker.coverage.as_dict(),
                    "audio": _audio_source_tracker.coverage.as_dict(),
                },
            }
            if _cov.frames > 0 and _cov.escalations == 0:
                # A FACT, not automatically a verdict. A still camera under a motion-gated watch
                # correctly admits nothing and saves every model call; alarming on that would
                # contradict percept's whole economic claim. This states "N frames, zero admissions,
                # against this bar" and leaves the reading to the consumer — the same framing
                # percept-doctor's ZERO_ADMITS_ON_FIXTURE uses.
                #
                # `coverage_floor` (None by default) is the ONE thing that changes the reading: an
                # explicit, opt-in declaration this run must clear (never inferred from the zero
                # itself, per ADR-008 Corrections(b)). Undeclared -> floor_unmet is always False, the
                # SAME diagnostic-only event byte-for-byte as always. Declared and unmet -> floor_unmet
                # is True, and percept.tracing.TraceRecorder.health_sink classifies THIS occurrence
                # (this kind string, this payload) as degraded — the kind itself is unchanged.
                _zero = dict(_cov_data)
                _zero["threshold"] = round(_cov_threshold, 4) if _cov_threshold is not None else None
                _zero["stat"] = _cov_stat
                _zero["coverage_floor"] = coverage_floor
                _zero["floor_unmet"] = coverage_floor is not None and _cov.escalations < coverage_floor
                await _call_sink(on_health, RuntimeHealthEvent("zero_admission_coverage", data=_zero))
            await _call_sink(on_health, RuntimeHealthEvent("runtime_ended", data=_cov_data))


@_exclusive_core_runtime
async def run(
    core: Any = None,
    *,
    source: Any,
    goals: Any = None,
    vision: Any = None,
    on_incident: Callable | None = None,
    on_ask: Callable | None = None,
    on_frame: Callable | None = None,
    on_health: Callable | None = None,
    on_trace: Callable | None = None,
    on_interest: Callable | None = None,
    interest_subscription: InterestSubscription | None = None,
    interest_session_id: str | None = None,
    fps: float = 4.0,
    source_queue_size: int = 64,
    source_reconnects: int = 0,
    source_reconnect_backoff_s: float = 0.5,
    source_rtsp_tcp: bool = True,
    snippet_pre_s: float = 5.0,
    snippet_size: int = 48,
    snippet_max_bytes: int | None = None,
    snippet_kind: str = "frame",
    sink_errors: str = "raise",
    stats: RuntimeStats | None = None,
    gate_source: str = "camera",
    episode_inactivity_s: float | None = None,
    coverage_staleness_s: float | None = None,
    audio_source: Any = None,
    audio_gate: Any = "acoustic-onset",
    audio_wake: dict | None = None,
    audio_settle_s: float = 1.5,
    on_utterance: Callable | None = None,
    allow_scripted_brain: bool = False,
    vad: Any = None,
    utterance_threshold: float = 0.01,
    utterance_hangover_frames: int = 8,
    utterance_min_speech_frames: int = 3,
    utterance_max_s: float = 60.0,
    audio_window_s: float | None = None,
    deaf_until: Callable[[], float] | None = None,
    stale_audio_s: float | None = None,
    wall_clock: Callable[[], float] = time.monotonic,
    wake_policy: Any = None,
    structural_adapter: Callable | None = None,
    semantic_adapter: Callable | None = None,
    energy_provider: Callable | None = None,
    regate_on_goal_change: bool = True,
    paused: Callable | None = None,
    muted: Callable | None = None,
    clock: Callable | None = None,
    persistence: Any = None,
    restore: bool = False,
    autosave_every_s: float | None = None,
    label_store: Any = None,
    disposition_log: Any = None,
    judge_dispatch: str = "async",
    metrics_collector: Any = None,
    judge_stall_after_s: float = 20.0,
    judge_backpressure_after: int = 3,
    judge_decision_deadline_s: float | None = None,
    timebase: str = "auto",
    coverage_floor: float | None = None,
) -> AsyncIterator:
    """Run programmable Watches over a live multimodal stream.

    ``interest_subscription`` exposes the exact per-Watch VID admission lifecycle as immutable
    entered/updated/exited events.  Consume it in a separate task; the producer never executes
    subscriber callbacks.  The channel is bound before the worker's fallible preflight and closed in
    this outermost ``finally`` on EOS, cancellation, or worker failure.

    The implementation delegated below retains every shipped path: ``coast_admit``, ``audio_admit``,
    ``check_episode_boundary``, ``check_observability``, ``check_deadlines``, ``paused``, ``muted``,
    ``judge_dispatch``, and the unconditional ``_drain_gate_health(core, on_health, t)`` drain.
    """
    worker_args = locals().copy()
    worker_args.pop("on_interest")
    worker_args.pop("interest_subscription")
    worker_args.pop("interest_session_id")

    if interest_subscription is not None and type(interest_subscription) is not InterestSubscription:
        raise TypeError("interest_subscription must be an exact InterestSubscription")

    # The disabled path allocates no lifecycle queue/tracker, hashes no media, and changes no provider
    # dispatch.  It is just the existing worker behind the public ownership wrapper.
    if interest_subscription is None:
        if on_interest is not None:
            raise InterestCallbackUnsupportedError(
                "on_interest callbacks are not supported; create InterestSubscription(...) and pass "
                "interest_subscription=..., then consume it with async iteration"
            )
        async for incident in _run_worker(**worker_args):
            yield incident
        return

    valid_session_id = (
        type(interest_session_id) is str and bool(interest_session_id.strip())
    )
    run_id = interest_session_id if valid_session_id else uuid.uuid4().hex
    InterestSubscription._bind(interest_subscription, run_id)
    tracker = LiveInterestTracker(run_id=run_id)
    worker_error: BaseException | None = None
    delivery_error: BaseException | None = None
    terminal_reason = "cancelled"
    try:
        if interest_session_id is not None and not valid_session_id:
            raise ValueError("interest_session_id must be a non-empty string when set")
        if on_interest is not None:
            raise InterestCallbackUnsupportedError(
                "on_interest callbacks are not supported; create InterestSubscription(...) and pass "
                "interest_subscription=..., then consume it with async iteration"
            )
        async for incident in _run_worker(
            **worker_args,
            _interest_tracker=tracker,
            _interest_subscription=interest_subscription,
        ):
            yield incident
        terminal_reason = "end_of_stream"
    except asyncio.CancelledError as exc:
        worker_error = exc
        terminal_reason = "cancelled"
    except GeneratorExit as exc:
        worker_error = exc
        terminal_reason = "cancelled"
    except BaseException as exc:  # noqa: BLE001 - preserve worker and lifecycle failures together
        worker_error = exc
        terminal_reason = "worker_error"
    finally:
        try:
            InterestSubscription._finish(
                interest_subscription,
                tracker.close_all(terminal_reason),
            )
        except BaseException as exc:  # noqa: BLE001 - delivery failure must not mask worker failure
            delivery_error = exc

    if worker_error is not None and delivery_error is not None:
        raise InterestRunError(worker_error, delivery_error) from worker_error
    if worker_error is not None:
        raise worker_error
    if delivery_error is not None:
        raise delivery_error
