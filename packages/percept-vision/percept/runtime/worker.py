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
import inspect
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from percept.events import FrameWindow, frame_nbytes
from percept.gate import FIRE, ASK
from percept.incidents import make_incident, snippet_from_window
from percept.observability import get_logger
from percept.runtime.envelope import (
    SourceEnvelope,
    SourceInspection,
    SourceTracker,
    legacy_audio_envelope,
    legacy_video_envelope,
)

_log = get_logger("percept.runtime.worker")


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
    #   escalations == signal_admits + coast_admits + audio_wakes
    # - signal_admits: the tuned motion/salience gate escalated (decision.escalate) — the real signal.
    # - coast_admits: a fixed-cadence coast floor forced the judge on a frame the gate would have
    #   skipped (coast_admit and NOT escalate). A strict_admission ExecutionPolicy (coast_enabled=False)
    #   strips a SourceProfile coast at resolve time, so this stays 0 on static content there.
    # - audio_wakes: an audio onset admitted a frame neither the gate nor the coast would (above).
    signal_admits: int = 0  # tuned-signal (motion/salience) escalations
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


@dataclass(frozen=True)
class RuntimeHealthEvent:
    """Structured runtime health event emitted through ``on_health``."""

    kind: str
    ts: float | None = None
    data: dict[str, Any] | None = None


class _JudgeScheduler:
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
        metrics_collector: Any = None,
        stall_after_s: float = 20.0,
        backpressure_after: int = 3,
    ) -> None:
        self._core = core
        self._source = source
        self._stats = stats
        # Optional percept.harness.metrics.MetricsCollector — when given, every admission
        # into this scheduler's active/pending slot is tracked via record_admitted/
        # record_admission_resolved (queue depth + oldest-pending age), keyed by the admission `t`
        # (unique per frame within one run). Purely additive telemetry: the scheduler's own admit/
        # launch/drop decisions are unchanged whether or not a collector is wired.
        self._metrics = metrics_collector
        self._active: "asyncio.Task | None" = None
        self._active_t: float | None = None  # admission t of the in-flight judgment (rides the result)
        # (t, judge_frame, admission_revision, admitted_goal_ids). The ids are frozen at the same
        # instant as the revision so a queued frame cannot broaden into unrelated Watches later.
        self._pending: tuple[float, Any, Any, tuple[str, ...] | None] | None = None
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
        self._perceive_accepts_goal_ids = _accepts_keyword(core.perceive_judged, "goal_ids")
        self._active_filter_goal_ids: tuple[str, ...] | None = None
        self._active_invalidated = False
        self._active_invalidation_reason = "source_discontinuity"

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
        return 0 if self._active is None else 1

    @property
    def pending(self) -> int:
        return 0 if self._pending is None else 1

    def submit(
        self, t: float, judge_frame: Any, goal_ids: tuple[str, ...] | None = None
    ) -> None:
        """Admit a frame for judgment. Never blocks; never launches after a failure or during close."""
        if self._fatal is not None or self._closing:
            return  # drain() re-raises the failure on the next loop tick; a closing run spends nothing
        revision = getattr(self._core.goals, "revision", None)  # ADMISSION-time, never launch-time
        if self._active is None:
            self._mc_admit(t)  # launching right away — admitted into the funnel from this instant
            self._launch(t, judge_frame, revision, goal_ids)
            return
        # The scheduler has no timer of its own — every submit() (the frame cadence) is the one
        # chance to notice the CURRENT active judgment has been outstanding too long. Checked before
        # the supersede below so a stalled-AND-overloaded tick reports both, in the order they became
        # true.
        self._check_stall(t)
        if self._pending is not None:
            dropped_t = self._pending[0]
            if self._stats is not None:
                self._stats.judge_superseded += 1  # latest wins: the older queued candidate is dropped
            self._trace.append((
                "judge_superseded",
                {
                    "source": self._source, "reason": "superseded_by_newer_admission",
                    "dropped_t": dropped_t, "admitted_t": t,
                },
            ))
            self._mc_resolve(dropped_t)  # leaves the collector's pending map — it stopped waiting here
            funnel = getattr(self._core, "funnel", None)
            if funnel is not None:
                funnel.dropped(reason="superseded_by_newer_admission", camera=self._source)
            # A RUN of supersessions with no intervening completion is the judge falling behind
            # the frame rate (sustained overwrite) — distinct from, and louder than, any one supersede.
            self._consec_supersede += 1
            self._check_backpressure(t)
        self._pending = (t, judge_frame, revision, goal_ids)
        self._mc_admit(t)  # queued behind the active judgment — pending in the funnel from this instant

    def invalidate_source_continuity(self, reasons: tuple[str, ...]) -> None:
        """Prevent pre-discontinuity judgments from mutating post-discontinuity state."""
        reason = ",".join(reasons) or "source_discontinuity"
        if self._active is not None and not self._active.done():
            self._active_invalidated = True
            self._active_invalidation_reason = reason
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
    ) -> None:
        if self._stats is not None:
            self._stats.judge_calls += 1  # counted at launch — a superseded candidate spent nothing
        self._active_t = t
        self._active_launch_wall = self._mc_now(t)  # this task's own clock, for stall detection
        self._stall_reported = False  # a fresh launch is unlatched — the stall window resets
        kwargs = {"t": t, "expected_goals_revision": revision}
        if goal_ids is not None and self._perceive_accepts_goal_ids:
            kwargs["goal_ids"] = goal_ids
        self._active_filter_goal_ids = (
            goal_ids
            if goal_ids is not None and not self._perceive_accepts_goal_ids
            else None
        )
        self._active = asyncio.create_task(
            self._core.perceive_judged(frame, self._source, **kwargs)
        )
        self._active.add_done_callback(self._on_done)

    def _launch_pending(self) -> None:
        if self._pending is None:
            return
        if self._fatal is not None or self._closing:
            self._mc_resolve(self._pending[0])  # leaving the funnel un-launched, not a real judgment
            self._pending = None
            return
        t, frame, revision, goal_ids = self._pending
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
        self._launch(t, frame, revision, goal_ids)

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
            self._active_filter_goal_ids = None
            self._active_invalidated = False
            if invalidated and not self._closing:
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
            self._active_filter_goal_ids = None
            self._active_invalidated = False
            return
        self._mc_resolve(self._active_t)
        result = task.result()
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
            self._results.append((self._active_t, result))
        self._active_filter_goal_ids = None
        self._active_invalidated = False
        self._active = None
        # A judgment completing WITHOUT raising is the same "reachable again" proof the agent level
        # (_note_judge_ok) uses — clear both latches and, only if one was actually
        # latched (never spam a clean run), report the recovery ONCE, loud, before anything new
        # launches off this completion.
        self._report_recovery_if_latched()
        self._active_launch_wall = None
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
        while self._fatal is None and (self._active is not None or self._pending is not None):
            task = self._active
            if task is None:
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
        """Shut down. ``_closing`` is set FIRST so a completion racing this can't launch new work.
        ``drain=True`` (finite files/tests): let the in-flight judgment finish and keep its result.
        ``drain=False`` (live shutdown/cancellation): cancel it and swallow the CancelledError."""
        self._closing = True
        if drain:
            await self.flush()
            return
        if self._pending is not None:
            self._mc_resolve(self._pending[0])  # a live shutdown drops the pending candidate outright
        self._pending = None
        task = self._active
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 — expected: the cancel we just requested
                pass


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
            timebase=(
                "observed"
                if isinstance(source, str) and source_kwargs.get("timebase") == "wall"
                else "source"
            ),
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


async def _audio_aiter(source: Any, **source_kwargs) -> AsyncIterator[SourceEnvelope]:
    """Normalize captured and pushed audio inputs to the versioned source envelope."""
    sequence = 0

    def normalize(item: Any) -> SourceEnvelope:
        nonlocal sequence
        if isinstance(item, SourceEnvelope):
            return item
        result = legacy_audio_envelope(
            item,
            source_id="audio",
            sequence=sequence,
            observed_time=time.monotonic(),
            timebase=(
                "observed"
                if isinstance(source, str) and source_kwargs.get("timebase") == "wall"
                else "source"
            ),
        )
        sequence += 1
        return result

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
    pcm_frames = [c for c in chunks if hasattr(c, "pcm")]
    if not pcm_frames:
        return _concat_audio(chunks)  # raw bytes / (mime, bytes) test source
    from percept.contracts import AudioFrame

    first = pcm_frames[0]
    return AudioFrame(
        pcm=b"".join(bytes(f.pcm) for f in pcm_frames),
        sample_rate=first.sample_rate,
        channels=first.channels,
        sample_format=first.sample_format,
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


def _resolve_timebase(source: Any, requested: str) -> EffectiveTimebase:
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
    if not isinstance(source, str):
        # A deterministic iterable (tests, replay, the eval harness) feeds its OWN explicit `t`; the
        # worker never re-stamps it, so nothing is resolved for it — byte-for-byte unchanged.
        return EffectiveTimebase(requested=requested, resolved="media", reason="deterministic-stream")
    if requested != "auto":
        return EffectiveTimebase(requested=requested, resolved=requested, reason="explicit-override")
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
    timebase: str = "auto",
    coverage_floor: float | None = None,
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
      ``core.push_audio`` — **filling the ``AudioWindow`` that ``modality="fused"/"audio"`` goals judge
      over** (otherwise it stays empty and fused judges degrade to vision-only) — and feeds ``energy``
      to the gate, so an off-camera sound (a cry, glass breaking) can escalate a frame the visual motion
      gate would skip. Default ``None`` → the video-only path is byte-for-byte, UNLESS ``source`` is
      itself a video container (``.mp4``/``.mov``/``.mkv``/``.webm``) AND an active goal's modality is
      ``"fused"``/``"audio"`` — then ``audio_source`` is auto-derived as ``source`` (its own audio track,
      demuxed via ffmpeg) so a caller watching an mp4 with a fused goal needs no extra parameter. A
      vision-only goal set never triggers this (no ffmpeg subprocess spawned when nothing wants audio),
      and an explicit ``audio_source`` always wins over the auto-derivation.
    - ``audio_gate`` — the tier-0 audio detector: a name (``"acoustic-onset"``), an instance with
      ``push(t, v) -> [Boundary]``, or ``None`` to only fill the window + feed energy (no onset wake).
    - ``audio_wake`` — kwargs for the onset detector, e.g. ``{"high": 0.3, "low": 0.15, "refractory_s": 5.0}``.
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

    ADMISSION COVERAGE. Every run closes with ``RuntimeHealthEvent("runtime_ended")`` carrying this
    run's counters — ``frames_seen`` / ``admitted`` / ``judge_calls`` / ``incidents`` — so a consumer
    that kept no state of its own can still say what the session did. When a run observed frames but
    admitted NONE of them, a ``RuntimeHealthEvent("zero_admission_coverage")`` is emitted immediately
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
    from percept.config import load_settings
    from percept.contracts import WatchSpec
    from percept.edge.detectors import DetectorBank
    from percept.edge.tier0 import MotionGate, compile_watchspec

    # Convenience: `percept.run(source=..., goals=[...])` with no pre-built core. Build a Percept
    # here and arm each goal, so the one-line quickstart is real. Passing an explicit `core` (the
    # canonical form) takes precedence — and supplying `goals`/`vision` ALONGSIDE a core is
    # now a hard ValueError, not a silent discard. The old behaviour ("ignored when core is given") ran
    # a byte-identical watch that simply never armed the caller's goals: the exact silent failure that
    # cost two client teams a full debugging cycle. A warning would still permit that run, so this
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

    settings = getattr(core, "settings", None) or load_settings()

    def _source_goals() -> list:
        """Active Watches bound to this worker's source key, in registry order."""
        return [
            goal
            for goal in core.goals.active()
            if getattr(goal, "source", "camera") == gate_source
        ]

    def _visual_goals() -> list:
        """Source Watches whose compiled perception actually consumes video frames."""
        return [
            goal
            for goal in _source_goals()
            if program_of(goal).perception.modality in ("vision", "fused")
        ]

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
    _video_source_tracker = SourceTracker()
    _audio_source_tracker = SourceTracker()
    _judge_sched = (
        _JudgeScheduler(
            core, gate_source, _cov, metrics_collector=metrics_collector,
            stall_after_s=judge_stall_after_s, backpressure_after=judge_backpressure_after,
        )
        if judge_dispatch == "async" else None
    )
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

    def _build_gate(prev_gate=None, prev_banks=None):
        """(Re)compile the WatchSpec → gate + detector banks + fovea coast from the core's CURRENT active
        goals. Called once at start, and again on a goal-registry revision change when
        regate_on_goal_change is set (so a goal armed mid-run — e.g. by voice — actually reaches the
        tier-0 gate). On a regate the new gate ADOPTS the prior gate's per-goal throttle + prev frame,
        and a surviving goal's detector bank is REUSED (its running rep-peak / refractory state must
        not reset mid-rep) — so arming one goal never disturbs the others."""
        _require_source_scoped_callbacks()
        visual_active = _visual_goals()
        conditions = []
        runtime_bindings = getattr(core, "_watch_runtime_bindings", {})
        for goal in visual_active:
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
            conditions.extend(compile_watchspec([goal], **kwargs).conditions)
        s = WatchSpec(conditions=conditions)
        # clock: a civil-time source arms per-goal Schedules; None -> always-armed (default).
        # The tier-0 extension seam, reachable from the supported entrypoint. None -> the gate's
        # structural/semantic channels stay 0.0 and this is byte-for-byte the motion-only path.
        g = MotionGate(
            s,
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
        return s, g, b, _coasts, _req

    spec, gate, banks, coasts, required_fps = _build_gate()
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
    compiled_audio_goals = [
        goal
        for goal in _source_goals()
        if goal.id in runtime_bindings
        and program_of(goal).perception.modality in ("audio", "fused")
    ]
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
    if audio_source is not None and audio_gate is not None:
        audio_goals = [
            goal
            for goal in _source_goals()
            if program_of(goal).perception.modality in ("audio", "fused")
        ]
        if len(audio_goals) > 1:
            runtime_bindings = getattr(core, "_watch_runtime_bindings", {})

            def _audio_policy_signature(goal: Any) -> str:
                bound = runtime_bindings.get(goal.id)
                watchspec = dict(bound[0].watchspec_kwargs) if bound is not None else {}
                return repr((
                    watchspec,
                    goal.state.get("cadence"),
                    goal.state.get("schedule"),
                    goal.state.get("sense"),
                    goal.state.get("perception"),
                ))

            signatures = {_audio_policy_signature(goal) for goal in audio_goals}
            if len(signatures) > 1:
                raise ValueError(
                    "audio onset admission is shared by this worker, but active audio/fused "
                    "Watches resolve to different admission policies. Run them on separate workers "
                    "or give them one identical policy; refusing cross-Watch wake leakage."
                )
    # Resolve the clock BEFORE the capture source is opened; both the video and the audio
    # path get their own record (a "mic:0" live mic and a "session.wav" recording resolve differently
    # even inside one run).
    video_tb = _resolve_timebase(source, timebase)
    audio_tb = _resolve_timebase(audio_source, timebase) if audio_source is not None else None
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
    # one profiling named as the whole of the "runaway memory" two client teams measured. It is
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

    _source_health_seen: set[tuple[str, str, int, str]] = set()
    dropped: list[int] = [0]  # shared with the capture thread
    source_health: list = []  # capture thread events drained on the video cadence

    async def _record_source_inspection(
        envelope: SourceEnvelope,
        inspection: SourceInspection,
        *,
        ts: float,
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
        for kind in inspection.events:
            marker = (envelope.source_id, envelope.modality, envelope.epoch, kind)
            if kind == "source_time_unknown" and marker in _source_health_seen:
                continue
            _source_health_seen.add(marker)
            await _call_sink(
                on_health,
                RuntimeHealthEvent(
                    kind,
                    ts=ts,
                    data={
                        "version": envelope.version,
                        "source_id": envelope.source_id,
                        "modality": envelope.modality,
                        "epoch": envelope.epoch,
                        "sequence": envelope.sequence,
                        "source_time": envelope.source_time,
                        "observed_time": envelope.observed_time,
                        "coverage_effect": inspection.coverage.kind,
                        "missing_sequences": inspection.coverage.missing_sequences,
                    },
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
        # recorded media. Deterministic legacy tuples carry their explicit source `t`.
        live_video = isinstance(source, str)
        if live_video:
            source_kwargs["timebase"] = video_tb.resolved
        live_audio = isinstance(audio_source, str)
        audio_kwargs = {"timebase": audio_tb.resolved} if live_audio and audio_tb else {}
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
        perceive_accepts_goal_ids = _accepts_keyword(core.perceive_judged, "goal_ids")

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

        def _compiled_audio_ready(goal_id: str) -> bool:
            """True unless this is a compiled audio/fused Watch missing a real audio payload."""
            if goal_id not in getattr(core, "_watch_runtime_bindings", {}):
                return True  # legacy goals retain their documented vision-only degradation
            goal = core.goals.get(goal_id)
            if goal is None:
                return False
            perception = program_of(goal).perception
            if perception.modality not in ("audio", "fused"):
                return True
            clip_for = getattr(core, "_audio_clip", None)
            clip = clip_for(gate_source, perception) if callable(clip_for) else None
            if clip is None:
                return False
            encoded = getattr(clip, "encoded", None)
            pcm = getattr(clip, "pcm", None)
            if encoded is not None or pcm is not None:
                return bool(encoded or pcm)
            if isinstance(clip, tuple) and len(clip) == 2:
                return bool(clip[1])
            if isinstance(clip, (bytes, bytearray, memoryview)):
                return bool(clip)
            return True

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
        _video_exhausted = [False]
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
                    onset = AcousticOnset(**(audio_wake or {})) if audio_gate == "acoustic-onset" else None
                else:
                    onset = audio_gate  # a caller-supplied detector instance

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
                        threshold=utterance_threshold,
                        hangover_frames=utterance_hangover_frames,
                        min_speech_frames=utterance_min_speech_frames,
                    )

            async def _pump_audio():
                # Pump every chunk into the AudioWindow (un-strands modality="fused"/"audio") and track
                # energy + onsets. Never lets a bad audio chunk kill the run — the watch survives a mic drop.
                async for envelope in _audio_aiter(audio_source, **audio_kwargs):
                    require_source_time = audio_tb is not None and audio_tb.resolved == "media"
                    inspection = _audio_source_tracker.inspect(
                        envelope, require_source_time=require_source_time
                    )
                    at = (
                        envelope.source_time
                        if require_source_time and envelope.source_time is not None
                        else envelope.observed_time
                    )
                    await _record_source_inspection(envelope, inspection, ts=at)
                    if not inspection.accepted:
                        continue
                    chunk, energy = envelope.payload, envelope.signal
                    if inspection.resets_continuity:
                        window = getattr(core, "audio_windows", {}).get(gate_source)
                        if window is not None and hasattr(window, "clear"):
                            window.clear()
                        if onset is not None and hasattr(onset, "reset"):
                            onset.reset()
                        if _vad is not None and hasattr(_vad, "reset"):
                            _vad.reset()
                        _utt["start"] = None
                        _utt_buf.clear()
                        _prev_chunk_at[0] = None
                        audio.update({
                            "energy": None,
                            "energy_t": None,
                            "chunk_dt": None,
                            "wake": False,
                            "wake_at": None,
                        })
                        if _judge_sched is not None:
                            _judge_sched.invalidate_source_continuity(inspection.events)
                    # Hold a FILE/CONTAINER chunk back until the video loop's judged media time has
                    # (almost) reached it — never busy-wait, park on the Event the frame loop sets every
                    # tick. `_audio_paced` is False for a live mic (byte-for-byte unchanged: this loop
                    # body is reached exactly as fast as `_audio_aiter` yields, same as before this gate
                    # existed).
                    while (
                        _audio_paced
                        and not _video_exhausted[0]
                        and at > _video_media_t[0] + _AUDIO_PACE_LOOKAHEAD_S
                    ):
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
                    try:
                        core.push_audio(chunk, source=gate_source, t=at)
                    except Exception:  # noqa: BLE001 - audio ingress must not crash the video watch
                        pass
                    if stats is not None:
                        stats.audio_chunks += 1
                    # Track the cadence + the time of the last REAL reading (a None-energy chunk is not
                    # a fresh reading). The frame loop reads energy only while fresh (dead mic → video-only).
                    if _prev_chunk_at[0] is not None:
                        _d = at - _prev_chunk_at[0]
                        if _d > 0:
                            audio["chunk_dt"] = _d
                    _prev_chunk_at[0] = at
                    if energy is not None:
                        audio["energy"] = float(energy)
                        audio["energy_t"] = at
                    if onset is not None and energy is not None:
                        for b in onset.push(at, float(energy)):
                            if getattr(b, "kind", None) == "onset" and audio["wake_at"] is None:
                                # settle: judge AFTER the sound has filled the window, not at its first instant
                                audio["wake_at"] = at + max(0.0, float(audio_settle_s))

                    # Voice escalation: endpoint a complete utterance and hand its clip to on_utterance.
                    # Runs AFTER push_audio so the just-heard chunk is already in the window we clip from.
                    if _vad is not None and energy is not None:
                        # Suppress the endpointer while the app is about to speak (deafen) or while
                        # the pump is replaying a stale capture backlog (staleness). The chunk still filled
                        # the AudioWindow above — this is turn-taking, not perception. Reset the VAD when
                        # suppressed so a half-built utterance cannot straddle (survive) the window.
                        suppress = False
                        if stale_audio_s is not None or deaf_until is not None:
                            wnow = wall_clock()
                            if stale_audio_s is not None:
                                if _wall_anchor[0] is None:
                                    _wall_anchor[0] = (wnow, at)
                                w0, s0 = _wall_anchor[0]
                                if (wnow - w0) - (at - s0) > stale_audio_s:  # pump lag vs real time
                                    suppress = True
                                    if on_health is not None:
                                        await _call_sink(on_health, RuntimeHealthEvent(
                                            "audio_stale", ts=at, data={"lag_s": (wnow - w0) - (at - s0)}))
                            if deaf_until is not None and wnow < deaf_until():
                                suppress = True
                        if suppress:
                            _vad.reset()
                            _utt["start"] = None
                            _utt_buf.clear()  # a suppressed span must not straddle into the next turn
                        else:
                            ev = _vad.push(float(energy))
                            if getattr(ev, "started", False):
                                _utt["start"] = at
                                _utt_buf.clear()  # a fresh turn — drop any pre-start residue
                            # Accumulate the turn into our OWN buffer (started→endpoint), bounded by
                            # utterance_max_s so a stuck-open mic can't grow it without limit.
                            if _utt["start"] is not None:
                                _utt_buf.append((at, chunk))
                                _cutoff = at - utterance_max_s
                                while _utt_buf and _utt_buf[0][0] < _cutoff:
                                    _utt_buf.pop(0)
                            if getattr(ev, "endpoint", False):
                                start_t = _utt["start"] if _utt["start"] is not None else at
                                _utt["start"] = None
                                buf = list(_utt_buf)
                                _utt_buf.clear()
                                first_t = buf[0][0] if buf else start_t
                                # Hand the WHOLE turn from the own buffer (never a window slice).
                                clip = _clip_from_chunks(
                                    [c for _, c in buf], t_start=first_t, t_end=at)
                                if clip is not None:
                                    if stats is not None:
                                        stats.utterances += 1
                                    # Truncation tripwire: if the handed clip covers less than the turn's
                                    # known duration (buffer capped, or leading chunks lost), shout — a
                                    # truncated utterance must NEVER be handed off silently.
                                    _covered = at - first_t
                                    _expected = at - start_t
                                    if on_health is not None and _covered < _expected - 0.25:
                                        await _call_sink(on_health, RuntimeHealthEvent(
                                            "utterance_truncated", ts=at,
                                            data={"covered_s": _covered, "expected_s": _expected}))
                                    if on_health is not None:
                                        await _call_sink(on_health, RuntimeHealthEvent(
                                            "utterance", ts=at, data={"dur_s": _expected}))
                                    # Hand off OFF the pump — a real handler (STT + voice turn + playback)
                                    # blocking here guarantees a capture backlog replayed as self-echo the
                                    # instant it returns. The app owns concurrency (overlaps are possible;
                                    # the runtime does not serialize). Errors still route through the policy.
                                    task = asyncio.ensure_future(_call_sink_with_policy(
                                        on_utterance, clip, at, errors=sink_errors, stats=stats,
                                        on_health=on_health, health_kind="utterance_sink_error",
                                    ))
                                    _utt_tasks.add(task)
                                    task.add_done_callback(_utt_done)

            audio_task = asyncio.ensure_future(_pump_audio())
            audio_task.add_done_callback(_task_watchdog("audio_pump_died"))

        async def _emit_fires(fires, *, at, from_boundary=False):
            """Turn a list of FireEvents (from perceive_judged OR the boundary) into incidents/asks."""
            for fire in fires or []:
                if getattr(fire, "action", None) == ASK:
                    if stats is not None:
                        stats.asks += 1
                    if disposition_log is not None:  # escalation-meter raw feed (asks are not incidents)
                        _log_disposition(disposition_log, incident_id="", fire=fire, disposition="ask", ts=at)
                    if on_health is not None:
                        await _call_sink(on_health, RuntimeHealthEvent("ask", ts=at, data={"goal_id": fire.goal_id}))
                    if on_ask is not None:
                        await _call_sink_with_policy(
                            on_ask, fire, errors=sink_errors, stats=stats, on_health=on_health, health_kind="ask_sink_error"
                        )
                    continue
                if getattr(fire, "action", None) != FIRE:
                    continue
                if stats is not None:
                    stats.fires += 1
                snippet = snippet_from_window(clips, kind=snippet_kind, pre_s=snippet_pre_s)
                incident = make_incident(
                    fire, ts=at, snippet=snippet, watchplan_id=_current_watchplan_id())
                if label_store is not None:  # bind the local label sink so incident.record_label() writes
                    incident.bind_labeler(
                        label_store.append,
                        verdict_ref=_verdict_ref(getattr(fire, "verdict", None)),
                        goal_id=getattr(fire, "goal_id", None),
                    )
                if disposition_log is not None:  # escalation-meter raw feed
                    _log_disposition(disposition_log, incident_id=incident.id, fire=fire, disposition="fire", ts=at)
                emit_trace = getattr(core, "_emit_trace", None)
                if getattr(core, "on_trace", None) is not None and emit_trace is not None:
                    # Routed through the safe core emitter (awaits async, swallows tracer errors); guarded
                    # with getattr since `core` is duck-typed (a custom core may lack _emit_trace).
                    # from_boundary flags incidents that did NOT pass through perceive_judged, so their
                    # trace_id has no upstream judge stages (an honest marker, not an orphan).
                    await emit_trace(
                        f"{gate_source}@{at}", at, "incident.created",
                        goal_id=incident.concern_id, incident_id=incident.id, from_boundary=from_boundary,
                        entity_id=incident.entity_id,
                        # The 1.9 evidence link, carried onto the trace row a client audit resolves: this
                        # incident joins back to EXACTLY ONE judgment.event carrying the same handle.
                        judgment_event_id=incident.judgment_event_id,
                        evidence_origin=incident.evidence_origin,
                    )
                if on_health is not None:
                    await _call_sink(
                        on_health,
                        RuntimeHealthEvent(
                            "incident",
                            ts=at,
                            data={
                                "incident_id": incident.id,
                                "goal_id": incident.concern_id,
                                "entity_id": incident.entity_id,  # which subject fired, not just which goal
                            },
                        ),
                    )
                if on_incident is not None:
                    await _call_sink_with_policy(
                        on_incident, incident, errors=sink_errors, stats=stats, on_health=on_health, health_kind="incident_sink_error",
                    )
                _cov.incidents += 1
                yield incident

        async for envelope in _aiter(source, capture_fps, **source_kwargs):
            require_source_time = (video_tb.resolved if live_video else "media") == "media"
            inspection = _video_source_tracker.inspect(
                envelope, require_source_time=require_source_time
            )
            t = (
                envelope.source_time
                if require_source_time and envelope.source_time is not None
                else envelope.observed_time
            )
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
                for _jt, _jfires in _judge_sched.drain():
                    async for incident in _emit_fires(_jfires, at=_jt):
                        yield incident
                # Surface any judge_degraded/judge_persistent_failure events queued by the
                # perceive_judged calls that just drained above (core is shared across dispatch modes).
                await _drain_judge_health(core, on_health, t)
            if not inspection.accepted:
                continue
            gate_frame, judge_frame = envelope.signal, envelope.payload
            if inspection.resets_continuity:
                gate.reset()
                for bank in banks.values():
                    bank.reset()
                for controller in coasts.values():
                    controller.reset()
                clips.clear()
                _cap_t0 = None
                _cap_n = 0
                _cap_unmet_emitted = False
                if _judge_sched is not None:
                    _judge_sched.invalidate_source_continuity(inspection.events)
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
                _video_progress.set()
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
                    if regate_on_goal_change:
                        # Pass the live gate + banks so the rebuild ADOPTS per-goal throttle + prev
                        # frame and REUSES surviving goals' detector banks (no throttle release / rep
                        # drop). Coast + required_fps are rebuilt too (a mid-run brief goal gets its
                        # coast).
                        spec, gate, banks, coasts, required_fps = _build_gate(gate, banks)
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
            if _funnel is not None:
                _funnel.seen(camera=gate_source)  # the ungated baseline: a frame was observed
            _cov.frames += 1
            _cov.frames_dropped = dropped[0]
            clips.push(t, judge_frame)
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
            if paused is not None and paused():
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
            decision = gate.evaluate(gate_frame, ts=t, energy=energy_now)

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
            if coast_admit:
                for goal_id in coast_goal_ids:
                    if goal_id not in admitted_goal_ids:
                        admitted_goal_ids.append(goal_id)
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
            # A compiled audio/fused Watch may never fall through to a provider with ``audio=None``:
            # that changes the question into vision-only perception. Drop only the affected Watch,
            # report the missing channel once, and keep this source worker alive for other Watches.
            missing_audio = [
                goal_id
                for goal_id in admitted_goal_ids
                if not _compiled_audio_ready(goal_id)
            ]
            for goal_id in missing_audio:
                admitted_goal_ids.remove(goal_id)
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
            emit_trace = getattr(core, "_emit_trace", None)
            if getattr(core, "on_trace", None) is not None and emit_trace is not None:
                # The tier-0 edge decision — the same shape used elsewhere for forensic tracing (motion/structural/
                # semantic/threshold/escalate) — emitted for EVERY frame, escalated or not, so "why didn't
                # it fire" can be answered even when the gate never spent a cloud call. trace_id matches
                # the one perceive_judged uses below (source+ts), so a dashboard can correlate the two.
                await emit_trace(
                    f"{gate_source}@{t}", t, "edge.decision",
                    motion=round(float(decision.signal.motion), 4),
                    structural=round(float(decision.signal.structural), 4),
                    semantic=round(float(decision.signal.semantic), 4),
                    threshold=round(float(decision.threshold), 4),
                    escalate=bool(decision.escalate),
                    coast_admit=coast_admit,
                    goal_ids=tuple(admitted_goal_ids),
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
            # Partition the admit into exactly one cause bucket (see RuntimeStats): a tuned-signal
            # escalation, else a fixed-cadence coast admit, else an audio wake. This mirrors the
            # `cause` string computed just below for the escalation health event.
            if decision.escalate:
                _cov.signal_admits += 1
            elif coast_admit:
                _cov.coast_admits += 1
            if audio_admit and not (decision.escalate or coast_admit):
                _cov.audio_wakes += 1
            if on_health is not None:
                cause = "audio" if (audio_admit and not (decision.escalate or coast_admit)) else \
                    ("both" if (audio_admit and decision.escalate) else ("coast" if coast_admit and not decision.escalate else "motion"))
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
            if _judge_sched is not None:
                _judge_sched.submit(t, judge_frame, tuple(admitted_goal_ids))
            else:
                _cov.judge_calls += 1
                perceive_kwargs = {"t": t}
                if perceive_accepts_goal_ids:
                    perceive_kwargs["goal_ids"] = tuple(admitted_goal_ids)
                fires = await core.perceive_judged(
                    judge_frame, gate_source, **perceive_kwargs
                )
                if not perceive_accepts_goal_ids:
                    fires = _filter_goal_events(fires, tuple(admitted_goal_ids))
                async for incident in _emit_fires(fires, at=t):
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
        if _judge_sched is not None:
            await _judge_sched.flush()
            for _hk, _hd in _judge_sched.drain_trace():
                if on_health is not None:
                    await _call_sink(on_health, RuntimeHealthEvent(_hk, data=_hd))
            for _jt, _jfires in _judge_sched.drain():
                async for incident in _emit_fires(_jfires, at=_jt):
                    yield incident
        # The LAST tick's suppressions (and anything the end-of-source flush above produced) would
        # otherwise die with the queue — drain once more so a finite stream reports every suppression
        # it earned, not all-but-the-final-tick's.
        await _drain_gate_health(core, on_health, _last_t)
    except Exception as exc:
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_error", data={"error": repr(exc)}))
        raise
    finally:
        # Async judge dispatch: never start new work past this point; cancel an in-flight judgment and
        # swallow its CancelledError (live shutdown / consumer break / cancellation). The normal
        # end-of-source path already flushed to quiescence above, so this is a no-op there.
        if _judge_sched is not None:
            await _judge_sched.close(drain=False)
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
            # A run's COVERAGE, stamped on the way out. `runtime_ended` used to carry an empty
            # payload, which made a run that admitted zero frames byte-for-byte indistinguishable
            # from a healthy one; a consumer had to have kept its own counters to know.
            _cov_data = {
                "frames_seen": _cov.frames,
                "admitted": _cov.escalations,
                # The admission split (escalations == signal_admits + coast_admits + audio_wakes): a
                # consumer auditing a strict_admission run reads coast_admits==0 directly here instead
                # of inferring it from the total.
                "signal_admits": _cov.signal_admits,
                "coast_admits": _cov.coast_admits,
                "audio_wakes": _cov.audio_wakes,
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
