"""The Percept orchestrator — the reactive watch loop (Phase 1).

One firing path: for each active goal, judge the plain-language condition over the in-focus
ROI (with any user-asserted corrections), run the verdict through the three-state gate, and emit
fires/asks on the rising edge. Later phases add *beside* this — the spoken turn (Phase 3),
episodes (Phase 4), events (Phase 5) — without editing the gate/graph contracts.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

from . import registry
from .adapters import Box, Decision, Goal, TurnResult, Verdict
from .behavior import BehaviorExecutor, program_of
from .cache import CacheCascade, CacheKey, FrameHashLayer
from .concern import attention_weight, consequence_of, requires_verification
from .config import Settings, load_settings
from .escalation import MAX_PERCEPTION_LEVEL, escalate_perception, should_escalate
from .consent import ActionLog, ActionRecord, ConsentDenied, ConsentProfile
from .entity import EntityGraph
from .episodes import EpisodeJudge, EpisodeLog
from .events import AudioWindow, EventLog, FrameWindow
from .gate import FIRE, Gate, GateAction, classify
from .goals import GoalOp, GoalRegistry
from .observability import Funnel, Metrics, OTelExporter, health, judge_usage
from .rhythm import RhythmTracker
from .routing import RouteEngine
from .scheduler import Scheduler


@dataclass
class FireEvent:
    """What the watch loop emits when the gate acts."""

    goal_id: str
    action: GateAction  # "fire" | "ask"
    text: str
    entity_id: str | None = None
    verdict: Verdict | None = None


def _escalated_perception(base, level: int):
    """Apply the escalation ladder to a base ``PerceptionSpec``: rung 1 adds a temporal strip, rung 2
    adds fused audio. Never downgrades an already-richer authored spec (only escalates the cheap parts)."""
    if level <= 0:
        return base
    spec = base
    if spec.temporal == "single":
        spec = replace(spec, temporal="strip", n_frames=max(3, spec.n_frames))
    if level >= 2 and spec.modality == "vision":
        spec = replace(spec, modality="fused", audio_s=spec.audio_s or 2.0)
    return spec


def _try_make_sprite(frames):
    """Montage the context frames into ONE image (Build 6), or ``None`` if they aren't images (so the
    caller degrades to a plain strip — e.g. L1 mock frames)."""
    try:
        from .sprite import make_sprite

        return make_sprite(frames)
    except Exception:
        return None


def _concat_audio(chunks: list) -> Any:
    """Concatenate audio chunks into one clip: ``(mime, bytes)`` pairs or raw bytes; else pass through."""
    if not chunks:
        return None
    first = chunks[0]
    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], (bytes, bytearray)):
        return (str(first[0]), b"".join(bytes(c[1]) for c in chunks))
    if isinstance(first, (bytes, bytearray)):
        return b"".join(bytes(c) for c in chunks)
    return chunks[-1]  # opaque descriptor — pass the latest through


class DeterministicAuditor:
    """Decides which cache HITs to re-perceive (false-serve audit) — **no RNG** (rule 2).

    False-serve is the cache's analog of false-fire (spec §6.2): re-perceive an
    ``audit_rate`` fraction of HITs through real vision and compare; a divergence is a
    **false-serve** and the fresh read wins. The decision is a deterministic *counter* so the L1
    lane is reproducible: with ``audit_rate = r`` we audit whenever the running HIT count crosses
    the next ``1/r`` boundary (a stratified 1-in-N, not a coin flip). ``r <= 0`` never audits;
    ``r >= 1`` audits every HIT. Inject a custom predicate for other policies.
    """

    def __init__(self, audit_rate: float) -> None:
        self._rate = audit_rate
        self._hits = 0
        self._audited = 0

    def __call__(self, key: object) -> bool:
        if self._rate <= 0.0:
            self._hits += 1
            return False
        if self._rate >= 1.0:
            self._hits += 1
            self._audited += 1
            return True
        self._hits += 1
        # audit when the ideal audited-count (hits*rate) has advanced past what we've done.
        if self._hits * self._rate >= self._audited + 1:
            self._audited += 1
            return True
        return False


def _resolve(seam: str, value: Any, cfg: Settings):
    """Accept either a backend *name* (resolved via the registry) or an adapter instance."""
    if isinstance(value, str):
        return registry.make(seam, value, cfg)
    return value  # already an adapter (handy for tests / custom wiring)


@dataclass
class Percept:
    """The cognition core: backends behind the seams + the memory structures + the gate."""

    vision: Any
    specialist: Any
    conductor: Any
    stt: Any
    tts: Any
    settings: Settings
    metrics: Metrics
    graph: EntityGraph
    gate: Gate
    goals: GoalRegistry
    cache: CacheCascade
    clock: Callable[[], float] = field(default=time.monotonic)
    # Which cache HITs to re-perceive for false-serve auditing. Deterministic by default
    # (no RNG on the L1 path); injectable for custom policies. ``None`` -> never audit.
    audit: Callable[[object], bool] | None = None
    router: RouteEngine | None = None  # local fast-path; None -> always defer (built in create)
    episodes: dict = field(default_factory=dict)  # goal_id -> EpisodeLog (counting goals)
    edge_counts: dict = field(default_factory=dict)  # goal_id -> running tally of edge detector boundaries
    rhythm: dict = field(default_factory=dict)  # goal_id -> RhythmTracker (the SELECTED subject's; brain count)
    _rhythm_multi: dict = field(default_factory=dict)  # (goal_id, entity_id) -> RhythmTracker (per-person streams)
    windows: dict = field(default_factory=dict)  # source -> FrameWindow (temporal perception)
    audio_windows: dict = field(default_factory=dict)  # source -> AudioWindow (fused/acoustic perception)
    perception_cooldown_s: float = 5.0  # the perception-rung refractory (shorter than the LLM's 30s)
    event_log: Any = None  # EventLog — temporal dedup of detected events (built in create)
    scheduler: Any = None  # Scheduler — time-triggered proactive fires (built in create)
    last_supported: dict = field(default_factory=dict)  # goal_id -> t of last CONFIDENT observation (A2)
    last_satisfied: dict = field(default_factory=dict)  # goal_id -> t of last confident-YES (A1 deadline)
    consent: ConsentProfile = field(default_factory=ConsentProfile)  # the act-modality gate
    action_log: ActionLog = field(default_factory=ActionLog)  # audit: every fire + act
    _executor: Any = None  # BehaviorExecutor — the unified firing path (lazily bound to self.gate)
    otel: OTelExporter | None = None  # optional OTel mirror for the funnel; None -> in-process only
    _funnel: Any = None  # Funnel — the standard telemetry funnel (lazily bound to self.metrics)

    # -- construction --------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        vision: Any = None,
        specialist: Any = None,
        conductor: Any = None,
        stt: Any = None,
        tts: Any = None,
        settings: Settings | None = None,
        clock: Callable[[], float] | None = None,
        discover_plugins: bool = True,
        audit: Callable[[object], bool] | None = None,
        identity: Any = None,
        embedder: Callable[[Any, Box], Any] | None = None,
    ) -> "Percept":
        cfg = settings or load_settings()
        # A seam not explicitly passed falls back to the config/env selection (settings.models.*, i.e.
        # the PERCEPT_*_BACKEND vars) so the .env controls which adapter each seam uses. Default: fake.
        vision = cfg.models.vision if vision is None else vision
        specialist = cfg.models.specialist if specialist is None else specialist
        conductor = cfg.models.conductor if conductor is None else conductor
        stt = cfg.models.stt if stt is None else stt
        tts = cfg.models.tts if tts is None else tts
        if discover_plugins:
            registry.discover_entrypoints()
        clk = clock or time.monotonic
        # Cache ships a contract, not a policy (invariant 6): the cascade is EMPTY (always DEFERs)
        # unless the FrameHashLayer is explicitly enabled via settings — then it sits in front of
        # the VLM. The false-serve auditor defaults to deterministic (no RNG on the L1 path).
        layers = (
            [FrameHashLayer(cfg.cache, clock=clk)] if cfg.cache.enabled else []
        )
        auditor = audit if audit is not None else DeterministicAuditor(cfg.cache.audit_rate)
        return cls(
            vision=_resolve("vision", vision, cfg),
            specialist=_resolve("specialist", specialist, cfg),
            conductor=_resolve("conductor", conductor, cfg),
            stt=_resolve("stt", stt, cfg),
            tts=_resolve("tts", tts, cfg),
            settings=cfg,
            metrics=Metrics(),
            graph=EntityGraph(
                cfg.entity.iou_match,
                cfg.entity.max_entities,
                clock=clk,
                identity=identity,
                embedder=embedder,
            ),
            gate=Gate(threshold=cfg.gate.threshold),
            goals=GoalRegistry(),
            cache=CacheCascade(layers=layers, threshold=cfg.cache.threshold),
            clock=clk,
            audit=auditor,
            router=RouteEngine.default(threshold=cfg.routing.confidence_threshold),
            event_log=EventLog(cfg.events.merge_gap_s),
            scheduler=Scheduler(clock=clk),
            consent=ConsentProfile(
                allow_computer_use=cfg.consent.allow_computer_use,
                allowed_actions=cfg.consent.allowed_actions,
            ),
            action_log=ActionLog(),
        )

    # -- goal authoring ------------------------------------------------------
    def add_goal(self, goal: Goal) -> Goal:
        return self.goals.create(goal)

    # -- episode counting (the cup problem) ---------------------------------
    def count_event(
        self, goal_id: str, exemplar: Any, judge: EpisodeJudge, *, t: float | None = None
    ) -> tuple[Any, bool]:
        """Fold a candidate counting-event into the goal's episode log; sync ``goal.state['count']``.

        ``judge`` decides new-episode-vs-continuation against the bounded log (an LLM in real use; a
        stub in L1 — never assert backend intelligence with a fake). Returns ``(episode, is_new)``.
        For goals whose ``state['type'] == 'count'`` (e.g. "how many cups today").
        """
        now = self.clock() if t is None else t
        log = self.episodes.get(goal_id)
        if log is None:
            log = EpisodeLog(
                goal_id,
                window_s=self.settings.episodes.window_s,
                max_episodes=self.settings.episodes.max_episodes,
            )
            self.episodes[goal_id] = log
        episode, is_new = log.observe(now, exemplar, judge)
        goal = self.goals.get(goal_id)
        if goal is not None:
            goal.state["count"] = log.count(now)
        self.metrics.incr("episode.new" if is_new else "episode.continuation")
        return episode, is_new

    # -- the event track (temporal perception) ------------------------------
    def _window(self, source: str) -> FrameWindow:
        window = self.windows.get(source)
        if window is None:
            window = FrameWindow(self.settings.events.window)
            self.windows[source] = window
        return window

    def push_frame(self, frame: Any, source: str = "camera", *, t: float | None = None) -> None:
        """Add a frame to the source's rolling window (for event-only flows; perceive_judged also feeds it)."""
        self._window(source).push(self.clock() if t is None else t, frame)

    def _audio_window(self, source: str) -> AudioWindow:
        win = self.audio_windows.get(source)
        if win is None:
            win = AudioWindow(getattr(self.settings.events, "audio_window_s", 4.0))
            self.audio_windows[source] = win
        return win

    def push_audio(self, chunk: Any, source: str = "camera", *, t: float | None = None) -> None:
        """Feed an audio chunk from the voice loop's mic stream (the same audio already flowing for STT)
        into the rolling AudioWindow — the source for fused/acoustic perception. No new capture path."""
        self._audio_window(source).push(self.clock() if t is None else t, chunk)

    def _audio_clip(self, source: str, spec) -> Any:
        """Assemble one audio clip for a fused/audio ``PerceptionSpec`` from the AudioWindow, else None."""
        if spec.modality not in ("audio", "fused") or not spec.audio_s:
            return None
        return _concat_audio(self._audio_window(source).clip(spec.audio_s))

    async def perceive_events(self, source: str = "camera") -> list[FireEvent]:
        """Detect events over the rolling window, dedup them, and run event goals through the gate.

        A *new* deduped event is a rising edge -> FIRE (once, via the gate's refractory on the
        event id); a continuation is SILENT; a low-confidence event is ASK. Counting falls out as
        ``goal.state['count'] = distinct events of the type``.
        """
        window = self.windows.get(source)
        if window is None or len(window) == 0:
            return []
        detected = await self.vision.detect_events(window.frames(), self.goals.active())
        out: list[FireEvent] = []
        for ev in detected:
            distinct, _is_new = self.event_log.observe(
                ev.type, ev.t, participants=ev.participants, confidence=ev.confidence
            )
            for goal in self.goals.active():
                if goal.state.get("type") != "event" or goal.state.get("event_type") != ev.type:
                    continue
                verdict = Verdict(satisfied=True, confidence=distinct.confidence)
                ev_fire = self._fire_concern(
                    goal, verdict, (goal.id, distinct.id), metric="event.gate"
                )
                goal.state["count"] = self.event_log.count(ev.type)
                if ev_fire is not None:
                    out.append(ev_fire)
        latest = window.latest()
        self._log_fires(out, latest[0] if latest else 0.0)
        return out

    # -- the scheduler (time-triggered proactive fires) ---------------------
    def schedule_every(self, interval_s: float, goal_id: str, *, now: float | None = None):
        return self.scheduler.every(interval_s, goal_id, now=now)

    def schedule_at(self, when_s: float, goal_id: str):
        return self.scheduler.at(when_s, goal_id)

    async def tick(self, *, t: float | None = None) -> list[FireEvent]:
        """Fire due timers via the conductor (timing in the clock; content judged by the model)."""
        now = self.clock() if t is None else t
        out: list[FireEvent] = []
        for timer in self.scheduler.due(now):
            goal = self.goals.get(timer.goal_id)
            say = goal.say if goal is not None else ""
            decision = await self.conductor.run("scheduler_tick", say, {"goal_id": timer.goal_id})
            if decision.route == "proactive_speak":
                out.append(FireEvent(timer.goal_id, FIRE, decision.say or say, None, None))
        self._log_fires(out, now)
        return out

    def apply_goal_ops(self, ops: list[GoalOp]) -> None:
        self.goals.apply(ops)
        # cache invalidation — goal-change (1st in the order): a goal whose condition/lifecycle
        # changed must not be served from a verdict cached under its old meaning.
        for op in ops:
            gid = op.goal.id if op.op == "create" and op.goal is not None else op.goal_id
            if op.op in ("update", "complete", "cancel") and gid is not None:
                self.invalidate_cache(gid)

    # -- unified firing via the general executor ----------------------------
    @property
    def executor(self) -> BehaviorExecutor:
        """The single firing path, lazily bound to THIS agent's gate + metrics (shared state)."""
        if self._executor is None:
            self._executor = BehaviorExecutor(self.gate, self.metrics)
        return self._executor

    @property
    def funnel(self) -> Funnel:
        """The standard telemetry funnel (seen→gated→judged→refused→fired), bound to THIS agent's
        metrics. The harness loop emits seen/gated; the watch path here emits judged/refused/fired and
        the judge latency + token/cost. Lazily built; OTel mirror attached when ``self.otel`` is set."""
        if self._funnel is None:
            self._funnel = Funnel(self.metrics, otel=self.otel)
        return self._funnel

    def _fire_concern(
        self,
        goal: Goal,
        verdict: Verdict,
        base_key: tuple,
        *,
        freshness: float = 1.0,
        t: float = 0.0,
        metric: str = "gate",
    ) -> FireEvent | None:
        """One firing path — delegates to the :class:`~percept.behavior.BehaviorExecutor` (the general
        ``sense·key·accumulate`` executor that now owns the per-concern re-arm key-synthesis, the
        never-contract, the hard-quadrant escalation, and the confusion substrate). This method is a
        thin boundary that adapts the executor's ``Fired`` back to the legacy ``FireEvent`` shape; the
        watch + event paths route through here, so both now run on the unified executor (gate unchanged).
        """
        fired = self.executor.fire_judged(
            goal, verdict, base_key, program_of(goal), freshness=freshness, t=t, metric=metric
        )
        if fired is None:
            return None
        return FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict)

    def perceive_boundary(self, goal_id: str, *, t: float = 0.0) -> FireEvent | None:
        """Consume one EDGE boundary (a detector's timed event — a rep peak, an audio onset) as a
        distinct occurrence and route it through the executor's nth-distinct accumulator (reusing
        :meth:`BehaviorExecutor.fire_episode`). This is the **brain-counts** half of
        edge-proposes·brain-counts·VLM-confirms: the count comes from the cheap edge signal, NOT a
        per-frame VLM call. Fires per the goal's ``accumulate.fire_when`` (e.g. "n == 10"), or on every
        boundary when none is authored — counting reps without spending a single vision token."""
        goal = self.goals.get(goal_id)
        if goal is None:
            return None
        n = self.edge_counts.get(goal_id, 0) + 1
        self.edge_counts[goal_id] = n
        fired = self.executor.fire_episode(goal, program_of(goal), n=n, is_new=True, t=t)
        if fired is None:
            return None
        self.metrics.incr("edge.boundary.fire")
        return FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict)

    def _rhythm_tracker(self, goal) -> RhythmTracker:
        prior = goal.state.get("rhythm") if isinstance(goal.state.get("rhythm"), dict) else {}
        kw = {"amp_gate_frac": prior["amp_gate_frac"]} if "amp_gate_frac" in prior else {}
        if "smooth_s" in prior:  # opt-in rate-invariance smoothing (Build 2); default off keeps legacy counts
            kw["smooth_s"] = prior["smooth_s"]
        if prior.get("canonical_select"):  # opt-in rate-invariant channel selection (Build 3); default off
            kw["canonical_select"] = True
        return RhythmTracker(prior_channel=prior.get("channel"), expected_rom=prior.get("expected_rom"), **kw)

    def _fire_rep(self, goal, n: int, stop_t: float) -> FireEvent | None:
        fired = self.executor.fire_episode(goal, program_of(goal), n=n, is_new=True, t=stop_t)
        if fired is None:
            return None
        self.metrics.incr("rhythm.rep.fire")
        return FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict)

    def perceive_rhythm(
        self, goal_id: str, *, t: float, channels: dict[str, float], entity_id: str | None = None
    ) -> FireEvent | None:
        """The **rhythm sense** (the brain-counts half done right): fold one frame's measured pose/motion
        CHANNELS into the goal's :class:`~percept.rhythm.RhythmTracker` and fire only when a *complete* rep
        is causally confirmed (rest→peak→return-to-rest). The count is the tracker's confirmed-rep tally —
        the eye measures, the brain decides which excursions are reps. No VLM call, no per-action tuning.

        ``channels`` is the eye's decomposed measurement vector for this frame (e.g. the ``pose-channels``
        detector's basis). A confirmed rep IS a distinct episode, so the firing reuses
        :meth:`BehaviorExecutor.fire_episode` (gate + ``fire_when``) — same path, new sense.

        ``entity_id`` makes counting PER-PERSON: each subject gets its own tracker and the brain SELECTS the
        performer (the entity with the most reps on the prior channel — a still bystander reps ~0), so the
        goal's reported count is the performer's. ``None`` (the default) is the legacy single-subject stream
        the benchmark/runtime read directly; behaviour there is unchanged."""
        goal = self.goals.get(goal_id)
        if goal is None or not channels:
            return None
        if entity_id is None:  # legacy single-subject stream — unchanged
            tracker = self.rhythm.get(goal_id)
            if tracker is None:
                tracker = self.rhythm[goal_id] = self._rhythm_tracker(goal)
            rep = tracker.observe(t, channels)
            if rep is None:
                return None
            n = tracker.count
            self.edge_counts[goal_id] = n  # the brain-owned count, read uniformly by the bench/runtime
            goal.state["count"] = n
            return self._fire_rep(goal, n, rep.stop_t)
        # per-entity: the brain selects the performer (most reps for this goal) and reports their count
        key = (goal_id, entity_id)
        tracker = self._rhythm_multi.get(key)
        if tracker is None:
            tracker = self._rhythm_multi[key] = self._rhythm_tracker(goal)
        rep = tracker.observe(t, channels)
        perf_key, perf = max(((k, tr) for k, tr in self._rhythm_multi.items() if k[0] == goal_id),
                             key=lambda kt: kt[1].count)
        self.rhythm[goal_id] = perf  # so core.rhythm[goal_id] / .count read the SELECTED subject uniformly
        n = perf.count
        self.edge_counts[goal_id] = n
        goal.state["count"] = n
        goal.state["subject"] = perf_key[1]  # who the brain is counting (the selected performer entity)
        if rep is None or key != perf_key:  # fire only when THE PERFORMER confirms a rep this frame
            return None
        return self._fire_rep(goal, n, rep.stop_t)

    # -- observability: coverage-loss fires with the concern's consequence (A2, spec §3) ----
    def check_observability(
        self, *, t: float | None = None, staleness_s: float | None = None
    ) -> list[FireEvent]:
        """Fire when a high-consequence concern LOSES perceptual coverage.

        Observability falls out of belief-from-observations: a concern whose last *confident*
        observation is older than ``staleness_s`` has gone blind. Losing coverage of a high-miss-cost
        concern INHERITS its consequence — "I can't see the baby" fires with the baby's urgency. This
        is perception-ROBUST: it fires on the ABSENCE of evidence, never by trusting a verdict.

        De-duped via the UNCHANGED gate: coverage-loss is a ``(C, R)`` fire (C = lost; R = on_false)
        on a dedicated ``__coverage__`` key, so it fires ONCE per loss episode and re-arms when
        coverage returns. Only ``miss_cost == 'severe'`` concerns inherit it (default mild -> silent,
        byte-for-byte). ``last is None`` (never yet observed) is not a loss — acquisition, not blindness.
        """
        now = self.clock() if t is None else t
        bound = staleness_s if staleness_s is not None else self.settings.gate.validity_window_s
        out: list[FireEvent] = []
        for goal in self.goals.active():
            if goal.state.get("miss_cost", "mild") != "severe":
                continue
            last = self.last_supported.get(goal.id)
            lost = last is not None and (now - last) > bound
            # observability is a derived FACT (we KNOW we haven't seen it), not a perceptual guess
            verdict = Verdict(satisfied=lost, confidence=1.0, evidence="coverage")
            action = self.gate.evaluate((goal.id, "__coverage__"), verdict)
            self.metrics.incr(f"coverage.gate.{action}")
            if action == FIRE:
                out.append(FireEvent(goal.id, FIRE, self._coverage_text(goal), None, verdict))
        self._log_fires(out, now)
        return out

    def _coverage_text(self, goal: Goal) -> str:
        return f"I've lost sight of what I'm watching: {goal.condition}"

    def _escalate_text(self, goal: Goal) -> str:
        return f"High stakes either way — confirm before I act: {goal.condition}?"

    def attention_order(self) -> list[Goal]:
        """A3 attention allocation (spec §5): order active concerns by consequence stakes so a
        budget-limited perception loop spends on the highest-consequence concerns first. Pure stakes,
        no confidence — a calibration-free job."""
        return sorted(
            self.goals.active(),
            key=lambda g: attention_weight(consequence_of(g)),
            reverse=True,
        )

    # -- absence-native firing: an expected concern overdue (A1, spec §3,§4) -----------------
    def check_deadlines(self, *, t: float | None = None) -> list[FireEvent]:
        """An EXPECTED concern not satisfied by its deadline fires "overdue".

        Absence IS the firing — there is no positive event to perceive, so this is perception-robust
        by construction: it fires on the LACK of a confident-yes, never on a guess. Rides the existing
        tick/scheduler loop (S2) and feeds a synthetic verdict to the UNCHANGED gate on a dedicated
        ``__overdue__`` key -> fires ONCE per overdue episode, re-arms if the expected event arrives.
        Only ``rearm == 'deadline'`` concerns (with a ``deadline_s``) participate -> byte-for-byte.
        """
        now = self.clock() if t is None else t
        out: list[FireEvent] = []
        for goal in self.goals.active():
            if goal.state.get("rearm") != "deadline":
                continue
            # absence firing (synthetic overdue verdict on the __overdue__ key) lives in the executor;
            # satisfaction is read from the agent's last_satisfied stamp.
            fired = self.executor.fire_absence(
                goal, program_of(goal), now=now, satisfied=goal.id in self.last_satisfied
            )
            if fired is not None:
                out.append(
                    FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict)
                )
        self._log_fires(out, now)
        return out

    # -- Item 4: the episode boundary fires the consolidated process answer -------------------
    async def check_episode_boundary(
        self, *, t: float | None = None, inactivity_s: float | None = None
    ) -> list[FireEvent]:
        """A process activity that has gone quiet (no NEW step for ``inactivity_s``) ENDED — summarize the
        logged deltas, emit the answer, and wipe. Cognition-owned, perception-robust (fires on the
        ABSENCE of new steps, never a guess). Mirrors :meth:`check_observability`: a synthetic verdict
        through the UNCHANGED gate on a dedicated ``__episode__`` key -> fires ONCE per episode and re-arms
        when the next step is logged. Only ``type=='process'`` goals participate -> byte-for-byte.
        Run on the worker's cadence (it must see QUIET frames — the boundary IS an absence of activity)."""
        now = self.clock() if t is None else t
        bound = inactivity_s if inactivity_s is not None else self.settings.gate.validity_window_s
        out: list[FireEvent] = []
        for goal in self.goals.active():
            if goal.state.get("type") != "process":
                continue
            ctx = self.executor.ctx(goal.id)
            ended = (
                bool(ctx.process_log)
                and ctx.last_step_at != float("-inf")
                and (now - ctx.last_step_at) > bound
            )
            verdict = Verdict(satisfied=ended, confidence=1.0, evidence="episode")
            action = self.gate.evaluate((goal.id, "__episode__"), verdict)
            self.metrics.incr(f"episode.gate.{action}")
            if action == FIRE:
                answer = await self.summarize_process(goal.id, wipe=True)  # consolidate + reset
                v = Verdict(satisfied=True, confidence=1.0, evidence=answer, running_answer=answer)
                out.append(FireEvent(goal.id, FIRE, answer or self._episode_text(goal), None, v))
        self._log_fires(out, now)
        return out

    def _episode_text(self, goal: Goal) -> str:
        return f"Here's what I saw for: {goal.condition}"

    def _overdue_text(self, goal: Goal) -> str:
        return f"Still waiting on: {goal.condition}"

    # -- the watch path ------------------------------------------------------
    async def perceive_judged(
        self, frame: Any, source: str = "camera", *, t: float | None = None
    ) -> list[FireEvent]:
        """Judge every active *watch* goal against ``frame`` and emit gate actions on rising edges.

        Event / scheduler goals run through their own paths (:meth:`perceive_events`, :meth:`tick`)
        but feed the **same three-state gate** — one gate, two perception tracks (spec §5.6).
        """
        now = self.clock() if t is None else t
        self._window(source).push(now, frame)  # feed the rolling window for temporal perception
        observations = await self.vision.detect(frame, self.goals.active())
        self.graph.ingest(observations, frame, source, t=now)
        focus = self.graph.in_focus()

        events: list[FireEvent] = []
        for goal in self.goals.active():
            gtype = goal.state.get("type", "watch")
            if gtype == "transition":
                ev = await self._perceive_transition(goal, frame, focus)
                if ev is not None:
                    events.append(ev)
                continue
            if gtype == "process":
                await self._observe_and_log(goal, frame, source, now)  # Item 4: log a delta, never fire here
                continue
            if gtype != "watch":
                continue  # event/count/remind goals are handled by their dedicated paths
            ctx = self.executor.ctx(goal.id)
            # HOW to look = the authored/preset spec, escalated by this concern's current ladder rung
            # (single→strip→+audio) when it's been recurrently ambiguous (set in _deliberate_confused).
            spec = _escalated_perception(program_of(goal).perception, ctx.perception_level)
            roi = self._resolve_roi(goal, focus, spec)
            corrections = focus.asserted() if focus else {}
            strip = self._strip_frames(source, spec)  # temporal strip per the spec; None = single (today)
            audio = self._audio_clip(source, spec)  # audio clip for a fused/audio spec; None otherwise
            self.funnel.judged(concern=goal.id, camera=source)  # a judge call is being spent
            # Item 2 process-QA: opt-in goals carry a per-concern running answer in/out of the judge.
            # GATE-IRRELEVANT — the verdict's satisfied/confidence still drive the (unchanged) gate; the
            # running answer is just threaded through ConcernContext. Default goals pass context=None and
            # are byte-for-byte unchanged.
            process_ctx = (
                {"process_qa": True, "running_answer": ctx.running_answer}
                if goal.state.get("process_qa") else None
            )
            verdict = await self._judge(
                goal, frame, roi, corrections, entity_id=focus.id if focus else None,
                frames=strip, audio=audio, context=process_ctx,
            )
            if process_ctx is not None and getattr(verdict, "running_answer", ""):
                ctx.running_answer = verdict.running_answer  # accumulate the model's rewritten summary
            freshness = self._freshness(focus, now)
            truth = classify(verdict, freshness, self.gate.threshold, self.gate.fresh_min)
            if truth == "unknown":
                self.funnel.refused(concern=goal.id, camera=source)  # abstained — refused to guess (ASK)
            else:
                self.last_supported[goal.id] = now  # A2: a confident read = perceptual coverage
                ctx.perception_level = 0  # decay: the (escalated) look resolved → back to the cheap default
            if truth == "yes":
                self.last_satisfied[goal.id] = now  # A1 deadline: the expected event happened
            base_key = (goal.id, focus.id if focus else None)
            ev_fire = self._fire_concern(goal, verdict, base_key, freshness=freshness, t=now)
            if ev_fire is not None:
                if ev_fire.action == FIRE:
                    self.funnel.fired(concern=goal.id, camera=source)  # a rising-edge fire emitted
                events.append(ev_fire)
        self._log_fires(events, now)
        await self._deliberate_confused(now)  # Posture B: auto-refine recurrently-ambiguous concerns
        return events

    async def _perceive_transition(self, goal, frame, focus) -> FireEvent | None:
        """Fire on a state change A->B, but ONLY after the 'from' state A was witnessed.

        A transition goal carries ``state={"type":"transition","from":<cond A>,"to":<cond B>}``.
        Per frame we judge BOTH states; observing a confident A arms (and re-arms) the goal, and a
        confident B *after* A has been seen fires once. This is what a single-frame judge can't do
        ("waking up" looks like "awake"): you must have seen the *before* state. Joining mid-stream
        already in B (never saw A) correctly never fires.
        """
        from_cond, to_cond = goal.state.get("from"), goal.state.get("to")
        if not from_cond or not to_cond:
            return None
        roi: Box | None = focus.box if focus else None
        corrections = focus.asserted() if focus else {}
        from_v = await self.vision.judge(from_cond, frame, roi=roi, corrections=corrections or None)
        to_v = await self.vision.judge(to_cond, frame, roi=roi, corrections=corrections or None)
        # Perception (judging from/to) stays here; the seen_from/fired state machine + firing now live
        # in the executor, with that state ISOLATED in its per-concern context (never the gate).
        fired = self.executor.fire_transition(
            goal, from_v, to_v, focus_id=focus.id if focus else "scene"
        )
        if fired is None:
            return None
        goal.state["count"] = goal.state.get("count", 0) + 1
        self.metrics.incr("transition.fire")
        return FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict)

    async def _observe_and_log(self, goal: Goal, frame: Any, source: str, now: float) -> None:
        """Item 4 (log→summarize→wipe): observe THIS clip's step delta and append it to the per-concern
        log. NEVER fires here — a process goal's answer is produced by :meth:`summarize_process` at the
        episode boundary / on demand. The gate is untouched (no verdict routed)."""
        ctx = self.executor.ctx(goal.id)
        spec = _escalated_perception(program_of(goal).perception, ctx.perception_level)
        strip = self._strip_frames(source, spec)
        self.funnel.judged(concern=goal.id, camera=source)  # an observe call is being spent
        verdict = await self._judge(
            goal, frame, None, {}, frames=strip, context={"observe_step": True}
        )
        step = (getattr(verdict, "evidence", "") or "").strip()
        if step and step != "not enough evidence":
            ctx.process_log.append(step)  # the VLM never sees this back — cheap append only
            ctx.last_step_at = now

    async def summarize_process(self, goal_id: str, *, wipe: bool = False) -> str:
        """Consolidate a process goal's logged step-deltas into the ordered distinct steps (Item 4).

        On-demand ("what have you seen?") with ``wipe=False``; the episode boundary calls it with
        ``wipe=True`` (end the episode → emit → reset the log). Returns "" if nothing has been logged.
        The summarize is gate-irrelevant — it never routes a verdict; only the boundary does (and through
        the unchanged gate). Backends that don't implement ``summarize_process`` yield "" gracefully.
        """
        ctx = self.executor.ctx(goal_id)
        log = list(ctx.process_log)
        summarizer = getattr(self.vision, "summarize_process", None)
        if not log or summarizer is None:
            return ctx.running_answer
        goal = self.goals.get(goal_id)
        condition = goal.condition if goal is not None else ""
        answer = await summarizer(condition, log)
        ctx.running_answer = answer or ctx.running_answer
        if wipe:
            ctx.process_log = []
        return ctx.running_answer

    def _resolve_roi(self, goal: Goal, focus, spec) -> Box | None:
        """ROI per the ``PerceptionSpec``: ``focus`` = the in-focus entity box (today); ``hint`` = the
        authored ``region_hint`` (reviving the previously-dead field); ``full`` = the whole frame."""
        if spec.roi == "full":
            return None
        if spec.roi == "hint":
            hint = goal.state.get("region_hint")
            if isinstance(hint, Box):
                return hint
            if isinstance(hint, (list, tuple)) and len(hint) == 4:
                return Box(*hint)
            # no usable hint authored -> fall back to the focus box (never worse than today)
        return focus.box if focus else None

    def _strip_frames(self, source: str, spec) -> list | None:
        """Assemble the temporal strip per the ``PerceptionSpec`` from the rolling FrameWindow: the
        earlier-context frames (oldest→newest, EXCLUDING the current frame, which the backend appends
        last). Returns None for single-frame perception — byte-for-byte today."""
        if spec.temporal == "single":
            return None
        earlier = self._window(source).frames()[:-1]  # drop the current frame (passed as `frame`)
        if not earlier:
            return None
        # B1: a clip window (window_s) bounds the strip by TIME — keep only the earlier frames within
        # window_s seconds of the current frame, so "a 4s clip" means the last 4s, not a frame count.
        if getattr(spec, "window_s", None):
            now_t = self._window(source).frames()[-1][0]  # current frame's timestamp
            cutoff = now_t - spec.window_s
            earlier = [(t, f) for t, f in earlier if t >= cutoff]
            if not earlier:
                return None
        want = (spec.n_frames - 1) if spec.temporal in ("strip", "sprite") else len(earlier)
        if want <= 0:
            return None
        if spec.sampling == "uniform" and len(earlier) > want:
            if want == 1:
                picked = [earlier[-1]]
            else:
                step = (len(earlier) - 1) / (want - 1)
                picked = [earlier[round(i * step)] for i in range(want)]
        else:  # "recent": the last `want` earlier frames
            picked = earlier[-want:]
        frames = [f for _, f in picked]
        if spec.temporal == "sprite":  # fold the temporal CONTEXT into ONE montage image (VLM token budget);
            sprite = _try_make_sprite(frames)  # the current frame is still sent separately for fidelity.
            return [sprite] if sprite is not None else frames  # non-image frames degrade to a plain strip
        return frames

    async def _timed_judge(self, *args: Any, **kwargs: Any) -> Verdict:
        """Call the vision judge, timing wall-clock latency and capturing token/cost off the verdict.

        ONE place wraps the real cloud call so every judge path (strip/audio, cache-miss, audit)
        records the same telemetry: ``judge.latency`` (always — we time it here) plus optional
        ``judge.tokens.*`` / ``judge.cost.usd`` when the backend annotated its verdict
        (:func:`~percept.observability.judge_usage`). Backends that don't report usage leave those
        hooks unused; the timing is unconditional."""
        t0 = self.clock()
        verdict = await self.vision.judge(*args, **kwargs)
        self.funnel.record_judge(latency_s=self.clock() - t0, **judge_usage(verdict))
        return verdict

    async def judge_clip(
        self,
        condition: str,
        frames: list,
        *,
        roi: Box | None = None,
        corrections: dict | None = None,
        audio: Any = None,
    ) -> Verdict:
        """Judge a plain-language ``condition`` over a CLIP (multiple frames), returning a ``Verdict``.

        The public temporal-judging seam (B1): a thin, ergonomic wrapper over the existing
        ``Vision.judge(..., frames=...)`` path for callers who already hold a clip (e.g. a concern that
        declared ``window_s``). ``frames`` is the clip oldest→newest; the LAST frame is the anchor passed
        as ``frame`` and the earlier ones are the temporal strip (matching the agent's per-tick
        assembly). A single-element clip degrades to a normal single-frame judge. Timing/usage telemetry
        flows through the same ``_timed_judge`` wrapper as the watch path, so a clip judge is observable
        identically. Raises ``ValueError`` on an empty clip — there is nothing to judge."""
        if not frames:
            raise ValueError("judge_clip requires a non-empty clip (oldest→newest frames)")
        anchor = frames[-1]
        strip = list(frames[:-1]) or None
        return await self._timed_judge(
            condition, anchor, roi=roi, corrections=corrections or None,
            frames=strip, audio=audio,
        )

    async def _judge(
        self,
        goal: Goal,
        frame: Any,
        roi: Box | None,
        corrections: dict[str, str],
        *,
        entity_id: str | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        process_mode = bool(context and (context.get("process_qa") or context.get("observe_step")))
        if frames or audio or process_mode:
            # The strip/audio (escalation) path is rare-by-design and multi-frame; bypass the
            # single-frame cache (its key dHashes ONE frame) so a strip verdict never pollutes — nor is
            # served from — a single-frame slot. The cache stays single-frame-pure. A process-QA judge
            # ALSO bypasses (its running answer/step accumulates per-call — must never be cached/served).
            # ``context`` is passed ONLY for process-QA so a non-process strip judge is byte-for-byte.
            extra = {"context": context} if process_mode else {}
            return await self._timed_judge(
                goal.condition, frame, roi=roi, corrections=corrections or None,
                frames=frames, audio=audio, **extra,
            )
        # A structured key so the FrameHashLayer can ROI-crop + dHash the frame and scope its
        # entry to (goal, entity); the legacy stub cascade ignores the extra fields harmlessly.
        cache_key = CacheKey(
            goal_id=goal.id, frame=frame, roi=roi, entity_id=entity_id
        )
        hit = self.cache.resolve(cache_key)
        if hit.status == "HIT" and hit.verdict is not None:
            self.metrics.incr("cache.hit")
            return await self._audit_or_use(goal, frame, roi, corrections, cache_key, hit.verdict)
        verdict = await self._timed_judge(
            goal.condition, frame, roi=roi, corrections=corrections or None
        )
        self.cache.store(cache_key, verdict, verdict.confidence)
        return verdict

    async def _audit_or_use(
        self,
        goal: Goal,
        frame: Any,
        roi: Box | None,
        corrections: dict[str, str],
        cache_key: Any,
        cached: Verdict,
    ) -> Verdict:
        """Trust-but-verify a cache HIT (spec §4.1, §6.2).

        On the audited fraction (decided by ``self.audit`` — deterministic, no RNG on the L1
        path), re-perceive through real vision and compare to the cached verdict. A divergence in
        the ``satisfied`` bit is a **false-serve**: the fresh read wins and replaces the stale
        cache entry. Un-audited HITs are served as-is. ``audit is None`` -> never audit.
        """
        if self.audit is None or not self.audit(cache_key):
            return cached
        self.metrics.incr("cache.audit")
        fresh = await self._timed_judge(
            goal.condition, frame, roi=roi, corrections=corrections or None
        )
        if fresh.satisfied != cached.satisfied:
            self.metrics.incr("cache.false_serve")  # the cache's analog of false-fire
            self.cache.store(cache_key, fresh, fresh.confidence)  # fresh read wins
            return fresh
        return cached

    def _freshness(self, focus, now: float) -> float:
        """1.0 if no asserted fact or it's still within the validity window, else 0.0 (stale)."""
        if focus is None:
            return 1.0
        confirmed = [a.last_confirmed for a in focus.attributes.values() if a.source == "asserted"]
        if not confirmed:
            return 1.0
        age = now - max(confirmed)
        return 1.0 if age < self.settings.gate.validity_window_s else 0.0

    def _ask_text(self, goal: Goal) -> str:
        return f'About "{goal.condition}" — can you confirm?'

    # -- corrections (voice -> memory) --------------------------------------
    def assert_attribute(
        self, key: str, value: str, *, entity_id: str | None = None, t: float | None = None
    ) -> str | None:
        """Bind a user correction to an entity (defaults to in-focus), then re-open judgment.

        Asserted beats observed; the goal's edge state is cleared so the corrected fact is
        re-judged rather than suppressed by a stale rising edge, and the cache is invalidated.
        """
        now = self.clock() if t is None else t
        target = entity_id
        if target is None:
            focus = self.graph.in_focus()
            target = focus.id if focus else None
        if target is None:
            return None
        self.graph.assert_attribute(target, key, value, t=now)
        for goal in self.goals.active():
            self.gate.reset_goal(goal.id)  # a correction re-opens the edge
        # cache invalidation — assertion-override (2nd in the order goal-change -> assertion ->
        # TTL -> L0): drop every cached verdict resting on the corrected entity so the asserted
        # fact is re-judged, never served stale. Empty/stub cascades are a harmless no-op.
        self.invalidate_cache_entity(target)
        return target

    def invalidate_cache(self, goal_id: str | None = None) -> None:
        """Goal-change invalidation: drop a goal's cached verdicts (``None`` clears all)."""
        for layer in self.cache.layers:
            inv = getattr(layer, "invalidate", None)
            if callable(inv):
                inv(goal_id)

    def invalidate_cache_entity(self, entity_id: str | None) -> None:
        """Assertion-override invalidation: drop cached verdicts resting on an entity."""
        for layer in self.cache.layers:
            inv = getattr(layer, "invalidate_entity", None)
            if callable(inv):
                inv(entity_id)

    def note_mention(self, text: str, *, t: float | None = None) -> str | None:
        """Fuse a spoken mention onto the in-focus entity (voice -> visible-entity)."""
        now = self.clock() if t is None else t
        return self.graph.note_mention(text, t=now)

    # -- the spoken turn (Phase 3) ------------------------------------------
    # Added *beside* perceive_judged, never inside it: the reactive watch loop is untouched.
    # The turn is router -> conductor -> specialist -> TTS, with the same "when in doubt,
    # defer" trust posture as the gate: the local router only fast-paths confident,
    # non-authoring intents; everything else falls to the conductor, the one tier that writes
    # standing state (authors goals).

    async def on_speech_endpoint(
        self, utterance: str, *, frame: Any = None, t: float | None = None
    ) -> TurnResult:
        """Handle one finalized spoken turn end-to-end.

        Pipeline: fuse any spoken mention onto the in-focus entity (voice -> visible-entity)
        -> local route -> if confident, execute locally (stream a reply via the specialist)
        -> else **defer to the conductor**, apply its goal ops, and act on its route /
        proactive-speak. Returns a :class:`~percept.adapters.TurnResult` for the caller/log.
        """
        now = self.clock() if t is None else t
        self.note_mention(utterance, t=now)

        local = self.router.route(utterance) if self.router is not None else None
        if local is not None:
            self.metrics.incr(f"route.local.{local.route}")
            reply = await self._execute(local, utterance, frame=frame)
            return TurnResult(route=local.route, source="router", reply=reply)

        # DEFER: the conductor routes + authors goals (the only tier that writes state).
        self.metrics.incr("route.defer")
        context = self._turn_context()
        decision = await self.conductor.run("speech", utterance, context)
        reply = await self._apply_decision(decision, utterance, frame=frame)
        return TurnResult(route=decision.route, source="conductor", reply=reply)

    async def on_scene_delta(
        self, event: str, context: dict | None = None, *, frame: Any = None
    ) -> TurnResult:
        """A proactive turn: a scene delta (person entered / something appeared) reaches the
        conductor so it can speak *unprompted* — not only fire pre-armed goals.

        ``event`` is a short delta label (e.g. ``"person_entered"``). There is no utterance,
        so the router is bypassed entirely; the conductor decides whether to proactively speak
        and/or author a goal. Apply path is shared with the spoken turn.
        """
        ctx = {**self._turn_context(), **(context or {})}
        decision = await self.conductor.run(event, "", ctx)
        reply = await self._apply_decision(decision, "", frame=frame)
        return TurnResult(route=decision.route, source="conductor", reply=reply)

    async def _apply_decision(
        self, decision: Decision, utterance: str, *, frame: Any = None
    ) -> str:
        """Fold a conductor :class:`Decision` into state + speech.

        Order matters: author goals *first* (standing-state write), then act on the route.
        """
        if decision.goal_ops:
            self.apply_goal_ops(decision.goal_ops)
            self.metrics.incr("goals.authored")
        return await self._execute(decision, utterance, frame=frame)

    async def deliberate(
        self, goal: Goal, *, t: float | None = None, cooldown_s: float = 30.0
    ) -> Decision | None:
        """Posture-B escalation (the consequence×novelty dial): when a concern is recurrently AMBIGUOUS
        and its stakes warrant it, re-invoke the conductor to REFINE the program (or decide to ask the
        human) instead of re-asking every tick. Consequence-gated + rate-limited via the per-concern
        ConfusionSignal + cooldown (the deliberative analog of the gate's refractory). Returns the
        conductor's Decision, or ``None`` when escalation isn't warranted.
        """
        now = self.clock() if t is None else t
        ctx = self.executor.ctx(goal.id)
        if not should_escalate(ctx, consequence_of(goal), now, cooldown_s=cooldown_s):
            return None
        ctx.last_escalated_at = now  # stamp the deliberative refractory (prevents LLM thrash)
        self.metrics.incr("deliberate.escalate")
        context = {
            **self._turn_context(),
            "goal": {"id": goal.id, "condition": goal.condition, "state": dict(goal.state)},
            "reason": "recurrent_ambiguity",
            "recent_confidence": list(ctx.confusion.recent_conf),
        }
        decision = await self.conductor.run("refine_program", goal.condition, context)
        if decision is not None and decision.goal_ops:
            self.apply_goal_ops(decision.goal_ops)  # the LLM refined the program (the in-loop path)
            self.metrics.incr("deliberate.refined")
        return decision

    async def _deliberate_confused(self, now: float) -> None:
        """Posture B in the perceive loop, as a LADDER: a recurrently-ambiguous concern first escalates
        its PERCEPTION (single→strip→+audio — cheap, better-looking) and only once that's maxed and it is
        STILL ambiguous does it re-invoke the LLM (expensive, more-thinking). Both rungs ride the same
        consequence-set ambiguity threshold + a refractory; perception uses a shorter cooldown so it
        always climbs first. A routine/calm concern costs nothing — neither rung fires.
        """
        for goal in list(self.goals.active()):
            ctx = self.executor.ctx(goal.id)
            if ctx.perception_level < MAX_PERCEPTION_LEVEL and escalate_perception(
                ctx, consequence_of(goal), now, cooldown_s=self.perception_cooldown_s
            ):
                self.metrics.incr("perception.escalate")
                continue  # took a perception rung this tick — give the better look a chance before the LLM
            await self.deliberate(goal, t=now)  # perception maxed (or not warranted) → consider the LLM rung

    async def _execute(self, decision: Decision, utterance: str, *, frame: Any = None) -> str:
        """Carry out a routed/conducted :class:`Decision`: speak (stream) or stay silent.

        ``respond_with_vision`` streams grounded on a frame; ``respond_direct`` /
        ``task_with_tools`` stream frame-free; ``proactive_speak`` speaks the conductor's
        ``say`` (or streams if absent); ``ignore`` / ``act`` emit no speech here.
        """
        route = decision.route
        if route == "ignore" or route == "act":
            return ""
        if route == "proactive_speak" and decision.say:
            return self._speak(decision.say)
        grounded = frame if decision.look == "request_frame" else None
        prompt = decision.say or utterance
        return await self._stream_reply(prompt, frame=grounded)

    async def _stream_reply(self, prompt: str, *, frame: Any = None) -> str:
        """Stream specialist tokens to the TTS sink, returning the assembled reply."""
        chunks: list[str] = []
        async for token in self.specialist.stream(prompt, frame=frame):
            self.tts.push(token)
            chunks.append(token)
        self.tts.flush()
        self.metrics.incr("specialist.reply")
        return "".join(chunks).strip()

    def _speak(self, text: str) -> str:
        """Push a fixed line straight to TTS (no model turn)."""
        self.tts.push(text)
        self.tts.flush()
        self.metrics.incr("tts.speak")
        return text

    def _turn_context(self) -> dict:
        """The conductor's read-only view of standing state for a turn / scene delta."""
        focus = self.graph.in_focus()
        return {
            "active_goals": [
                {"id": g.id, "condition": g.condition} for g in self.goals.active()
            ],
            "in_focus": focus.id if focus else None,
        }

    # -- consent + audit (SDK primitives; the strip UI is project 2) --------
    def _log_fires(self, events: list[FireEvent], t: float) -> None:
        """Record every fire/ask to the action log (the audit seed + V1 feedback corpus)."""
        for ev in events:
            self.action_log.record(
                ActionRecord(
                    kind=ev.action,
                    goal_id=ev.goal_id,
                    detail={"text": ev.text, "entity_id": ev.entity_id},
                    t=t,
                )
            )

    def act(
        self, goal_id: str, action: str, detail: dict | None = None, *, t: float | None = None
    ) -> ActionRecord:
        """Perform a computer-use action — **flag-gated, scoped, logged** (spec §8).

        Allowed only when the goal's ``modality == "act"`` AND the consent profile permits the
        action. Every attempt is logged (allowed or blocked — nothing acts silently); a blocked
        attempt raises :class:`ConsentDenied`. The returned record is the event the assistant's
        consent strip renders.
        """
        now = self.clock() if t is None else t
        goal = self.goals.get(goal_id)
        modality_ok = goal is not None and goal.modality == "act"
        source_id = goal.source if goal is not None else None
        allowed = bool(modality_ok and self.consent.permits(action, source_id))
        record = self.action_log.record(
            ActionRecord(
                kind="act" if allowed else "act_blocked",
                goal_id=goal_id,
                detail={"action": action, **(detail or {})},
                t=now,
                allowed=allowed,
            )
        )
        self.metrics.incr("act.allowed" if allowed else "act.blocked")
        if not allowed:
            why = (
                "goal modality is not 'act'"
                if not modality_ok
                else f"action {action!r} not permitted "
                f"(allow_computer_use={self.consent.allow_computer_use})"
            )
            raise ConsentDenied(f"act blocked for goal {goal_id!r}: {why}")
        return record

    def actuate(
        self,
        goal_id: str,
        action: str,
        *,
        expect: str | None = None,
        verify_within: float = 30.0,
        detail: dict | None = None,
        t: float | None = None,
    ) -> ActionRecord:
        """A world-changing action under the spec's full safety contract (§7, A4).

        Goes through :meth:`act` (consent + per-source + modality enforced; a blocked attempt still
        raises :class:`ConsentDenied`), and — for a concern whose false-fire cost is high enough to
        REQUIRE VERIFICATION (A3) — instantiates a **verification concern** that confirms the action's
        effect actually landed. "Every world-changing action instantiates a verification concern"
        (spec §7) is enforced HERE, not left to the caller. OSS stays notify-first: actuate performs
        no side effect beyond the audited ``act()`` and arming the verification concern.
        """
        now = self.clock() if t is None else t
        record = self.act(goal_id, action, detail, t=now)  # consent gate; raises if blocked
        goal = self.goals.get(goal_id)
        if goal is not None and requires_verification(consequence_of(goal)):
            v = self._instantiate_verification(goal, action, expect, now, verify_within)
            record.detail["verification_goal_id"] = v.id
        return record

    def _instantiate_verification(
        self, goal: Goal, action: str, expect: str | None, t: float, verify_within: float
    ) -> Goal:
        """Acting creates a concern that checks the effect LANDED (spec §7) — reusing A1's absence
        machinery: a ``deadline`` concern that fires "couldn't confirm" if the expected post-condition
        isn't observed within ``verify_within``. High miss-cost: an unverified world-change is exactly
        what must not be silently dropped. Confirmation (a confident-yes on the effect) cancels it."""
        vid = f"{goal.id}::verify::{action}"
        v = Goal(
            id=vid,
            condition=expect or f"the effect of {action!r} is confirmed",
            say=f"couldn't confirm the effect of {action!r}",
            source=goal.source,
        )
        v.state["rearm"] = "deadline"
        v.state["deadline_s"] = t + verify_within
        v.state["miss_cost"] = "severe"
        v.state["verifies"] = goal.id
        self.add_goal(v)
        self.action_log.record(
            ActionRecord(
                kind="verify_created",
                goal_id=goal.id,
                detail={"verification_goal_id": vid, "action": action},
                t=t,
            )
        )
        return v

    # -- persistence (endurance: survive a disconnect) ----------------------
    def snapshot(self) -> dict:
        """A JSON-able snapshot of durable state: goals, entities, and the action log."""
        return {
            "version": 1,
            "goals": self.goals.snapshot(),
            "entities": self.graph.snapshot(),
            "action_log": [asdict(r) for r in self.action_log.entries()],
        }

    def restore(self, snap: dict) -> "Percept":
        """Rehydrate goals / entities / action-log from a :meth:`snapshot`."""
        self.goals.restore(snap.get("goals", []))
        self.graph.restore(snap.get("entities", []))
        self.action_log = ActionLog()
        for rd in snap.get("action_log", []):
            self.action_log.record(ActionRecord(**rd))
        return self

    def save_state(self, persistence: Any, key: str = "percept") -> None:
        """Persist the snapshot through a ``Persistence`` backend (in-memory / sqlite)."""
        persistence.save(key, self.snapshot())

    def load_state(self, persistence: Any, key: str = "percept") -> bool:
        """Rehydrate from a ``Persistence`` backend. Returns False if nothing was stored."""
        snap = persistence.load(key)
        if snap is None:
            return False
        self.restore(snap)
        return True

    # -- introspection -------------------------------------------------------
    def health(self) -> dict:
        backends = {
            "vision": type(self.vision).__name__,
            "specialist": type(self.specialist).__name__,
            "conductor": type(self.conductor).__name__,
            "stt": type(self.stt).__name__,
            "tts": type(self.tts).__name__,
        }
        from . import __version__

        return health(__version__, backends, self.metrics)


__all__ = ["Percept", "FireEvent"]
