"""The Percept orchestrator — the reactive watch loop (Phase 1).

One firing path: for each active goal, judge the plain-language condition over the in-focus
ROI (with any user-asserted corrections), run the verdict through the three-state gate, and emit
fires/asks on the rising edge. Later phases add *beside* this — the spoken turn (Phase 3),
episodes (Phase 4), events (Phase 5) — without editing the gate/graph contracts.
"""

from __future__ import annotations

import inspect
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Hashable

from . import registry
from .adapters import Box, Decision, Goal, GoalOrigin, TurnResult, Verdict
from .behavior import BehaviorExecutor, EpisodeFoldResult, ShadowObservation, program_of
from .cache import CacheCascade, CacheKey, FrameHashLayer
from .concern import attention_weight, consequence_of, requires_verification, resolution_ready
from .config import ModeError, Settings, load_settings, missing_credential, resolve_mode
from .contracts import JudgmentEvent, JudgmentTimingBuilder, SubjectRef
from .escalation import MAX_PERCEPTION_LEVEL, escalate_perception, should_escalate
from .consent import ActionLog, ActionRecord, ConsentDenied, ConsentProfile
from .entity import EntityGraph
from .extraction import CONTEXT_KEY as EXTRACTOR_CONTEXT_KEY
from .claims import ClaimState, concern_authority_of
from .episodes import (
    EPISODE_POLICIES,
    ContinuationDecision,
    Episode,
    EpisodeJudge,
    EpisodeLog,
    MutationCheck,
    episode_policy_of,
    subject_is_unknown,
)
from .identity_policy import identity_policy_of
from .events import AudioWindow, EventLog, FrameWindow
from .gate import FIRE, Gate, GateAction, classify
from .goals import GoalOp, GoalRegistry
from .observability import Funnel, Metrics, OTelExporter, TraceEvent, get_logger, health, judge_usage
from .rhythm import RhythmTracker
from .routing import RouteEngine
from .scheduler import Scheduler
from .tracing import STAGE_JUDGMENT_EVENT, STAGE_SHADOW_FOLD


#: LRU ceiling on the judgment evidence-sequencer's per-key counters. Matches the gate's own
#: default ``max_keys`` so the observability map can never outgrow the gate state it mirrors.
_SEQ_MAX_KEYS = 10_000


@dataclass
class FireEvent:
    """What the watch loop emits when the gate acts."""

    goal_id: str
    action: GateAction  # "fire" | "ask"
    text: str
    entity_id: str | None = None
    verdict: Verdict | None = None
    # The frozen :class:`percept.contracts.JudgmentEvent` for this decision (ADR-001 §1): the
    # gate key, the evidence sequence, the action, and the five-stamp two-domain timing the funnel
    # actually took. Populated on the judged watch path (``perceive_judged`` → ``_fire_concern``);
    # ``None`` on every other firing path (episode/rhythm/boundary/transition/absence/process-QA),
    # which take no judge-funnel stamps — an absent event, never an invented one. Appended LAST with
    # a ``None`` default: existing positional constructions and consumers are unchanged.
    judgment: JudgmentEvent | None = None
    # The enriched judgment envelope carried from the executor's ``Fired`` so it survives to the
    # delivered Incident (``make_incident`` reads these off the fire). All OPTIONAL/defaulted: an
    # episode/rhythm/boundary/transition/absence fire that sets none of them is byte-for-byte unchanged.
    reason: str | None = None
    modality: str | None = None
    roi: object | None = None
    provenance: str | None = None


# -- The resolved WatchProfile.plan_mode -> the concern's PerceptionSpec -----------------------
# ``plan_mode`` was carried onto the core before anything read it. This module makes it LIVE: a resolved profile whose
# plan_mode is "window" makes the judge see an ORDERED FRAME WINDOW (``judge(frames=[...])``, oldest→
# newest, the anchor frame passed separately) instead of one still frame. That is the validated win
# (dynamic-goal recall rose five points, and to ~nine-in-ten with gate-relax; see
# percept/profiles_default.py::H0_PROVENANCE for the measured numbers): the eye hands the model TIME,
# not a montage. No number lives here -- this is a plan SHAPE, not a threshold.

#: Plan modes ranked by how much time the judge is given. The profile's plan_mode is a FLOOR, never a
#: downgrade — a concern that AUTHORED a richer spec keeps it (the brain's explicit HOW outranks a
#: context default), and a profile can only raise a cheaper spec up to its own rung.
_PLAN_RANK: dict[str, int] = {"single": 0, "strip": 1, "window": 2}

#: Deprecated plan modes -> their replacement. ``sprite`` folded the context frames into
#: ONE montage image to save VLM tokens; an ordered window beats it unconditionally (see
#: ``percept/profile_resolution.py``: "sprite is strictly worse and must never be selected"), so the plan
#: path no longer builds a montage. ``percept.sprite.make_sprite`` itself is untouched and still importable
#: — only the PLAN path is deprecated, and it degrades to the strictly better mode rather than failing.
_DEPRECATED_PLAN_MODES: dict[str, str] = {"sprite": "window"}

#: Frames a profile-raised strip/window asks for when the authored spec never sized one (same floor the
#: escalation ladder uses, so "the profile widened the plan" and "the ladder widened the plan" agree).
_PLAN_MIN_FRAMES = 3


def _normalize_plan_mode(mode: str) -> str:
    """Map a DEPRECATED plan mode onto its replacement, warning once per call site. A live
    mode is returned unchanged — this is a no-op on every supported plan."""
    replacement = _DEPRECATED_PLAN_MODES.get(mode)
    if replacement is None:
        return mode
    warnings.warn(
        f"plan_mode/temporal {mode!r} is deprecated and now behaves as {replacement!r}: an ordered "
        "frame window beats a sprite montage unconditionally on the dynamic-goal eval set, so the plan "
        "path no longer builds a montage. Set the plan mode to 'window' explicitly.",
        DeprecationWarning,
        stacklevel=3,
    )
    return replacement


def _planned_perception(base, plan_mode: str | None):
    """Overlay the RESOLVED ``WatchProfile.plan_mode`` onto a concern's base ``PerceptionSpec``.

    Two things happen here, in one place:

    1. the deprecation shim — a ``sprite`` plan (authored on the spec OR carried on the profile) is
       normalized to ``window``;
    2. the profile FLOOR — ``plan_mode`` raises a cheaper spec to its rung and never lowers a richer
       authored one (``_PLAN_RANK``).

    ``plan_mode=None`` (no profile, or a profile-less construction) returns the base spec unchanged, so
    the legacy path stays byte-for-byte. An unrecognized mode is ignored rather than trusted — the
    vocabulary lives in :data:`percept.behavior._VALID_TEMPORAL`, not in a string a caller invented."""
    spec = base
    temporal = _normalize_plan_mode(spec.temporal)
    if temporal != spec.temporal:  # authored sprite -> window (shim), before any ranking
        spec = replace(spec, temporal=temporal)
    if not plan_mode:
        return spec
    mode = _normalize_plan_mode(str(plan_mode))
    if mode not in _PLAN_RANK:
        return spec  # unknown/unsupported plan mode: never silently reinterpreted
    if _PLAN_RANK[mode] <= _PLAN_RANK.get(spec.temporal, 0):
        return spec  # the authored spec already sees at least this much time — never downgrade
    n_frames = spec.n_frames if spec.n_frames >= _PLAN_MIN_FRAMES else _PLAN_MIN_FRAMES
    return replace(spec, temporal=mode, n_frames=n_frames)


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


def _trace_reason(truth: str, action: str) -> str:
    """A human-readable WHY for the ``gate.classified`` trace stage — purely observational, derived
    post-hoc from the already-computed ``truth`` (yes/no/unknown) + ``action`` (fire/ask/silent). Never
    read by firing logic (invariant 1: one gate, one firing path); this only explains its output, the
    same forensic shape the eval bench trace used (``eval/bench/track_charades_e2e.py::_silent_reason``)."""
    if action == FIRE:
        return "fire: rising_edge"
    if action == "ask":
        return "ask: low_confidence_or_stale"
    # silent
    if truth == "no":
        return "silent: verdict_not_satisfied"
    if truth == "yes":
        return "silent: already_fired_or_not_rising"
    return "silent: suppressed"


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


def _is_valid_verdict(obj: Any) -> bool:
    """Shape-validate a judge response BEFORE any caller dereferences ``.confidence``/``.satisfied``
    (ADR-001 §10 invariant 4). The backend contract is :class:`Verdict`; anything else — a
    plain/garbled object (malformed), or a ``Verdict``-shaped-but-incomplete response with the wrong
    field types (partial/truncated JSON) — must be rejected here, never dereferenced blindly."""
    return (
        isinstance(obj, Verdict)
        and isinstance(obj.satisfied, bool)
        and isinstance(obj.confidence, (int, float))
    )


_GATE_HEALTH_LOG_MAX = 256
"""Growth bound: the most queued gate-suppression entries :attr:`Percept._gate_health_log`
holds between drains. The runtime loop drains it EVERY tick, so this is never reached in a real
watch; it exists so a core driven WITHOUT ``percept.runtime.worker.run`` (a bare ``perceive_judged``
harness, which never drains) cannot grow the queue without bound over a 24/7 run. Oldest entries are
shed first, and every suppression is already emitted as its own structured ``gate.suppressed`` log
line by :meth:`percept.gate.Gate._note_suppressed`, so shedding never loses the fact itself."""


CONSECUTIVE_DEGRADE_ESCALATE = 5
"""Tunable policy knob: the number of CONSECUTIVE degraded judge ticks (raised /
malformed / partial -- anything :class:`_JudgeDegraded`) for the SAME watch key before the
fail-closed boundary escalates from a per-tick ``"judge_degraded"`` health event to a louder,
ONE-TIME ``"judge_persistent_failure"`` event. Escalating never halts the run (degrade+continue is
the chosen fail-closed policy) -- it only makes an otherwise-silent, persistently-failing watch
LOUD. The streak for a key resets to zero the instant that SAME key produces any well-formed
verdict again (see :meth:`Percept._note_judge_ok`)."""


class _JudgeDegraded(Exception):
    """Internal-only fail-closed signal: the judge call raised (timeout / provider error)
    or returned a malformed/partial response. Raised exclusively by
    :meth:`Percept._safe_timed_judge` and caught by every caller of :meth:`Percept._judge`
    (``perceive_judged``, ``_observe_and_log``) — it must never escape :class:`Percept` itself.
    Carries the ORIGINAL failure (the raised exception, or the bad object returned) for logging only;
    it never reaches ``percept.gate.classify`` or the gate."""

    def __init__(self, cause: Any) -> None:
        self.cause = cause
        super().__init__(f"judge call failed closed: {cause!r}")


class DeterministicAuditor:
    """Decides which cache HITs to re-perceive (false-serve audit) — **no RNG** (rule 2).

    False-serve is the cache's analog of false-fire: re-perceive an
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


def _is_scripted_backend(obj: Any) -> bool:
    """True iff ``obj`` is one of Percept's in-core scripted doubles (``percept.backends.fake``).

    A real vendor adapter (``percept.backends.anthropic`` …) OR any custom/test double counts as
    NOT scripted. The point is to catch a REAL pipeline silently paired with a scripted double — not
    to classify third-party wiring — so identity is by the module (the class is the FACT), never a
    heuristic on the requested name (a registry miss / plugin / stray ``PERCEPT_*_BACKEND`` can land
    you on a scripted double after you asked for real; the string is the request, the class the fact).
    """
    return type(obj).__module__ == "percept.backends.fake"


# One announced vocabulary: "offline" (the mode) selects "scripted" (the backends).
_SCRIPTED_SEAMS = ("vision", "specialist", "conductor", "stt", "tts")
_BANNER_EMITTED = False  # once-per-process guard for the loud stderr banner


def _scripted_seams(core: Any) -> list[str]:
    """The seams whose CONSTRUCTED backend is a scripted double, in a stable order."""
    return [s for s in _SCRIPTED_SEAMS if _is_scripted_backend(getattr(core, s, None))]


def _emit_banner(mode: str, scripted: list[str]) -> None:
    """The loud stderr banner — once per process, suppressed under pytest (895 banners self-defeat)."""
    global _BANNER_EMITTED
    import os
    import sys

    if _BANNER_EMITTED or "pytest" in sys.modules or os.environ.get("PERCEPT_NO_BANNER"):
        return
    _BANNER_EMITTED = True
    header = "OFFLINE" if mode == "offline" else "SCRIPTED BACKENDS"
    sys.stderr.write(
        f"[percept] ⚠ {header} — scripted backends: {', '.join(scripted)}.\n"
        f"          This proves WIRING, not intelligence. A scripted conductor authors NO goals.\n"
    )


def guard_live_language_path(core: Any, *, allow_scripted_brain: bool = False) -> None:
    """FATAL seam: refuse to wire a LIVE voice/language path behind a SCRIPTED brain.

    The conductor is the language-authoring seam — it turns an utterance into ``goal_ops``. A scripted
    conductor's default ``Decision(route="ignore")`` silently discards every spoken "watch/tell me
    when…", so the user wired a voice that can't author. That is the demo bug; refuse it at the
    language↔conductor crossing (``on_speech_endpoint`` / ``arm_from_language``) and at the live-voice
    wiring points ``run(on_utterance=…)`` / ``session.start(on_utterance=…)`` — NOT at ``create()``, which
    cannot yet know a language path is coming. (An app that wires a voice ENGINE — see
    ``templates/realtime_voice_watch_agent`` — calls this itself at ``PerceptBridge.attach``.)

    Skipped when the whole run is declared scripted (``mode="offline"`` — everything is announced
    scripted, including the voice) or when the ugly-on-purpose ``allow_scripted_brain`` hatch is set.
    """
    if allow_scripted_brain or getattr(core, "_allow_scripted_brain", False):
        return
    if getattr(core, "_mode", "live") == "offline":
        return
    conductor = getattr(core, "conductor", None)
    if conductor is not None and _is_scripted_backend(conductor):
        raise ModeError(
            "scripted brain behind a live voice: a language path (on_utterance / an attached "
            "transport) is wired, but the conductor is SCRIPTED — it authors NO goals, so every "
            "spoken 'watch X' / 'tell me when…' is silently dropped. Fix: select a real conductor "
            "(conductor='claude'/'gemini' or PERCEPT_CONDUCTOR_BACKEND=…) with its API key; or, to "
            "test the voice plumbing only, run mode='offline' or pass allow_scripted_brain=True."
        )


# Backwards-compatible alias: this seam was named ``_guard_live_language_path`` before it was made a
# public template import. The underscore name is kept so existing internal callers and
# ``from percept.agent import _guard_live_language_path`` sites keep working; new code imports the
# public ``guard_live_language_path``.
_guard_live_language_path = guard_live_language_path


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
    on_trace: "Callable[[Any], None] | None" = None  # opt-in per-frame decision tracer; None -> no-op
    episodes: dict = field(default_factory=dict)  # goal_id -> EpisodeLog (counting goals; also
    # reused by _document_episode_fold for a content-key watch goal — a goal is never both types,
    # so the map is shared safely)
    # The per-goal high-water mark of the last ``t`` GOVERNED into a content-key concern's episode
    # log (see ``_document_episode_fold``). Session-only, deliberately absent from
    # snapshot()/restore() like ``episodes`` itself.
    _doc_episode_hwm: dict = field(default_factory=dict)  # goal_id -> highest t governed so far
    # CLAIM MODE: the per-goal claim-slot twins of the document-mode fields above.
    # ``_claim_states`` holds the authoritative :class:`~percept.claims.ClaimState` (value + its own
    # revision ledger) per claim-slot goal; ``_claim_hwm`` is the same out-of-order high-water mark as
    # ``_doc_episode_hwm``; ``_claim_episode_id`` remembers the ONE persistent episode a claim-slot
    # concern keeps in the shared ``episodes`` map (a fixed subject_key == goal.id, unlike a
    # content-key concern's many per-document episodes), so :meth:`episode_current` can look it up
    # without reaching into ``EpisodeLog`` internals.
    _claim_states: dict = field(default_factory=dict)  # goal_id -> ClaimState
    _claim_hwm: dict = field(default_factory=dict)  # goal_id -> highest t governed so far
    _claim_episode_id: dict = field(default_factory=dict)  # goal_id -> episode id
    edge_counts: dict = field(default_factory=dict)  # goal_id -> running tally of edge detector boundaries
    rhythm: dict = field(default_factory=dict)  # goal_id -> RhythmTracker (the SELECTED subject's; brain count)
    _rhythm_multi: dict = field(default_factory=dict)  # (goal_id, entity_id) -> RhythmTracker (per-person streams)
    windows: dict = field(default_factory=dict)  # source -> FrameWindow (temporal perception)
    audio_windows: dict = field(default_factory=dict)  # source -> AudioWindow (fused/acoustic perception)
    perception_cooldown_s: float = 5.0  # the perception-rung refractory (shorter than the LLM's 30s)
    event_log: Any = None  # EventLog — temporal dedup of detected events (built in create)
    scheduler: Any = None  # Scheduler — time-triggered proactive fires (built in create)
    last_supported: dict = field(default_factory=dict)  # goal_id -> t of last CONFIDENT observation
    last_satisfied: dict = field(default_factory=dict)  # goal_id -> t of last confident-YES (deadline)
    consent: ConsentProfile = field(default_factory=ConsentProfile)  # the act-modality gate
    action_log: ActionLog = field(default_factory=ActionLog)  # audit: every fire + act
    _executor: Any = None  # BehaviorExecutor — the unified firing path (lazily bound to self.gate)
    # The per-key monotonic evidence counter (percept.sequencing.EvidenceSequencer), stamped at
    # judge-DISPATCH time so every JudgmentEvent carries the same ``evidence_seq`` a GateDecision
    # would. Lazily built and LRU-bounded (``_SEQ_MAX_KEYS``) exactly like the gate's own key maps, so
    # a 24/7 multi-entity watch cannot leak through it. Observability only — nothing branches on it.
    _sequencer: Any = None
    otel: OTelExporter | None = None  # optional OTel mirror for the funnel; None -> in-process only
    _funnel: Any = None  # Funnel — the standard telemetry funnel (lazily bound to self.metrics)
    # The announced-scripted state. Set by create(): the resolved mode ("offline"/"live"),
    # the seams whose CONSTRUCTED backend is a scripted double, and the ugly escape hatch. Read by
    # health() (the single scripted-state payload) and by the language-path FATAL seam. NOT construction inputs the
    # caller passes as dataclass fields — create() fills them.
    _mode: str = "live"
    _scripted: tuple = ()
    _allow_scripted_brain: bool = False
    # -- fail-closed boundary: consecutive-degradation breaker + the health-event queue
    # worker.py's run() loop drains into its on_health(RuntimeHealthEvent) channel every tick (the
    # SAME pattern as _JudgeScheduler.drain_trace() -- Percept queues, the runtime drains + emits;
    # Percept itself never calls on_health directly, that channel belongs to worker.py). Keyed by
    # the SAME ``(goal_id, entity_id)`` key the gate/executor already use ("the reducer key").
    _judge_degrade_streak: dict = field(default_factory=dict)  # key -> consecutive degraded-tick count
    _judge_health_log: list = field(default_factory=list)  # [(kind, key, cause), ...] queued for drain
    # -- consequence-gate GA: the SAME queue-here / drain-in-worker pattern for GATE SUPPRESSION.
    # ``Gate`` calls ``_note_gate_suppressed`` for every decision whose reason is in
    # ``percept.gate.SUPPRESSION_REASONS`` (start_anchored / refractory) -- i.e. a fire the baseline
    # rising-edge gate WOULD have emitted, withheld by an opt-in precision guard. Both guards are off
    # by default, so a default-configured core never appends here at all. worker.py's run() loop drains
    # this into ``on_health(RuntimeHealthEvent("gate_suppressed"))`` every tick; Percept never calls
    # on_health itself. Bounded (see _GATE_HEALTH_LOG_MAX) so a core driven WITHOUT the runtime loop
    # (a bare perceive_judged harness, which never drains) cannot leak on a 24/7 watch.
    _gate_health_log: list = field(default_factory=list)  # [(key, reason, at), ...] queued for drain
    # -- episode-memory reset: the SAME queue-here / drain-in-worker pattern. snapshot() does
    # NOT carry ``episodes`` (the per-goal EpisodeLog map — session-only for v1.4, ADR-006),
    # so ``restore()`` cannot rehydrate grouping/counting memory: after a redeploy every counting goal
    # restarts at zero episodes. That is a documented decision, but a SILENT one — the known failure
    # mode is an operator under-counting after a restart with nothing telling them why. restore() queues
    # ONE ``episode_memory_reset`` entry here when it rehydrates active counting goals with an empty
    # episode map; worker.py's run() drains it into ``on_health(RuntimeHealthEvent("episode_memory_reset"))``
    # right after load_state, so the amnesia is SEEN. Percept never calls on_health itself.
    _restore_health_log: list = field(default_factory=list)  # [(kind, data), ...] queued for drain
    # -- WatchProfile knobs applied at CORE CREATION time (never a mid-run mutation) ----------------
    # The resolved fovea cadence a WatchProfile's ``cadence`` carries (see percept.profile_wiring).
    # ``None`` (the default -- no profile, or a profile-less ``settings=`` construction) means no
    # override: byte-for-byte the behavior before profile resolution existed. Set once, in create(); nothing in this class mutates it
    # afterward -- a live cadence change is a NEW core, not a field write on this one.
    sense_policy: Any = None  # SensePolicy | None -- see percept.sampling.SensePolicy
    # The resolved WatchProfile.plan_mode ("single"/"strip"/"window"; "sprite" is deprecated and shimmed
    # to "window"). LIVE: _planned_perception() floors every concern's PerceptionSpec with it on
    # the judging path, so plan_mode="window" makes the judge see an ordered frame window
    # (``judge(frames=[...])``). ``None`` -- no profile override, byte-for-byte the behavior before profile resolution existed.
    plan_mode: str | None = None
    # The optional EvidenceExtractor seam (percept.extraction.EvidenceExtractor). ``None`` (the
    # default) -- perceive_judged calls it for nothing and every judge call is byte-for-byte
    # unchanged. When set, perceive_judged calls .extract(condition, frame, roi) for each active
    # watch goal right before judging, merges the resulting Enrichment.facts into
    # context["extractor"] (a reserved sub-dict -- see percept.extraction.CONTEXT_KEY), and emits
    # an "extractor.enriched" trace stage. GATE-IRRELEVANT: the gate still reads only the judge's
    # own satisfied/confidence.
    extractor: Any = None
    # The optional DirectJudge routing layer (percept.direct_judge.DirectJudge). ``None`` (the
    # default) -- perceive_judged routes nothing and every judge call is byte-for-byte unchanged.
    # When set, it is consulted right before the judge call: CONFIDENT skips the judge entirely
    # (zero vision calls -- its normalized Verdict is used as-is, and the UNCHANGED Gate still
    # classifies it); UNCERTAIN merges its facts into context["extractor"] and the judge is still
    # called; DEGRADED (or no rule matched) falls through to today's judge call, untouched.
    direct_judge: Any = None

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
        extractor: Any = None,
        direct_judge: Any = None,
        settings: Settings | None = None,
        clock: Callable[[], float] | None = None,
        discover_plugins: bool = True,
        audit: Callable[[object], bool] | None = None,
        identity: Any = None,
        embedder: Callable[[Any, Box], Any] | None = None,
        on_trace: Callable[[Any], None] | None = None,
        on_health: Callable[[str], None] | None = None,
        mode: str | None = None,
        allow_scripted_brain: bool = False,
        sense_policy: Any = None,
        plan_mode: str | None = None,
    ) -> "Percept":
        cfg = settings or load_settings()
        # The mode is the switch: explicit kwarg > PERCEPT_MODE env > live. It is never "offline" by
        # NOT typing something — you fall onto scripted backends only by naming the mode or selecting a
        # scripted seam, which is the whole point (faking must be a choice you make and see).
        resolved_mode = resolve_mode(mode)
        offline = resolved_mode == "offline"
        # Per-seam resolution: explicit kwarg > settings.models.* (env/ModelSettings) > mode default.
        # vision/conductor/specialist now default None (config.py); stt/tts still default "fake". An
        # unset seam becomes "scripted" offline. In LIVE mode an unset VISION is the bare-create() trap
        # (no real perception, no key) -> ModeError; conductor/specialist fall back to scripted and are
        # announced LOUDLY below (the code-armed watch is a legitimate pattern, not a crash).
        def _pick(name: str, kwarg: Any, cfg_value: Any, *, required: bool,
                  scripted_default: str = "scripted") -> Any:
            sel = kwarg if kwarg is not None else cfg_value
            if sel is not None:
                return sel
            if offline:
                return scripted_default
            if required:
                raise ModeError(
                    f"no {name} backend selected and this is a LIVE run — Percept will not silently "
                    f"fall back to a scripted double. Fix: pass mode='offline' for keyless scripted "
                    f"backends (proves WIRING, not intelligence), or select a real backend "
                    f"({name}='claude'/'gemini') and set its API key."
                )
            return scripted_default

        vision = _pick("vision", vision, cfg.models.vision, required=True)
        specialist = _pick("specialist", specialist, cfg.models.specialist, required=False)
        # The conductor is the language-authoring seam: an UNSET one defaults to the HONEST scripted double
        # (ScriptedConductor — no-match -> route="respond_direct"), NEVER FakeConductor whose default
        # route="ignore" is the one value indistinguishable from a real conductor declining. So
        # `mode="offline"` — the door people explore through — no longer hands back a brain that discards
        # every utterance and looks like it worked.
        conductor = _pick("conductor", conductor, cfg.models.conductor, required=False,
                          scripted_default="scripted-conductor")
        stt = cfg.models.stt if stt is None else stt
        tts = cfg.models.tts if tts is None else tts
        # LIVE credential check: a SELECTED real backend name whose key is absent raises ModeError
        # naming the exact env var(s) — never construct-then-fail-silently at first call. Per-backend
        # (not hardcoded to one vendor). Skipped offline (a declared real seam offline needs no key —
        # it is not going to make a live call). Adapter instances the caller wired directly are skipped.
        if not offline:
            for seam_name, sel in (("vision", vision), ("specialist", specialist),
                                   ("conductor", conductor), ("stt", stt), ("tts", tts)):
                keys = missing_credential(sel)
                if keys:
                    raise ModeError(
                        f"{seam_name} backend {sel!r} needs a credential and none is set. Fix: export "
                        f"one of {list(keys)}, or run mode='offline' (keyless scripted backends)."
                    )
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
        obj = cls(
            vision=_resolve("vision", vision, cfg),
            specialist=_resolve("specialist", specialist, cfg),
            conductor=_resolve("conductor", conductor, cfg),
            stt=_resolve("stt", stt, cfg),
            tts=_resolve("tts", tts, cfg),
            extractor=extractor,  # optional; None -> perceive_judged calls nothing, byte-for-byte
            direct_judge=direct_judge,  # optional; None -> every judge call is byte-for-byte
            settings=cfg,
            metrics=Metrics(),
            graph=EntityGraph(
                cfg.entity.iou_match,
                cfg.entity.max_entities,
                clock=clk,
                identity=identity,
                embedder=embedder,
            ),
            gate=Gate(
                threshold=cfg.gate.threshold, clock=clk,
                require_false_first=cfg.gate.require_false_first,
                refractory_s=cfg.gate.refractory_s or None,
            ),
            goals=GoalRegistry(),
            cache=CacheCascade(layers=layers, threshold=cfg.cache.threshold),
            clock=clk,
            audit=auditor,
            on_trace=on_trace,
            router=RouteEngine.default(threshold=cfg.routing.confidence_threshold),
            event_log=EventLog(cfg.events.merge_gap_s),
            scheduler=Scheduler(clock=clk),
            consent=ConsentProfile(
                allow_computer_use=cfg.consent.allow_computer_use,
                allowed_actions=cfg.consent.allowed_actions,
                actuation=getattr(cfg.consent, "actuation", "none"),  # typed envelope declaration
            ),
            action_log=ActionLog(),
            _mode=resolved_mode,
            _allow_scripted_brain=allow_scripted_brain,
            sense_policy=sense_policy,
            plan_mode=plan_mode,
        )
        # When the entity graph evicts a track (it caps at max_entities), sweep that entity's derived
        # state — its ``(goal_id, entity_id)`` gate keys AND any cached verdicts resting on it — so a 24/7
        # churn of entities can't leave orphan edges accumulating in the gate. Wired here (not at graph
        # construction) because the gate is built alongside the graph in the same call.
        obj.graph.on_evict = obj._on_entity_evicted
        # Gate SUPPRESSION (a fire withheld by require_false_first / refractory_s) must be visible,
        # not a silent non-event. Wired here for the same reason as on_evict above — the gate is built
        # in this same call, and the hook needs the finished ``obj`` to queue onto. Queue only; the
        # runtime loop drains it into on_health (this class never calls on_health itself).
        obj.gate.on_suppress = obj._note_gate_suppressed
        # Announce scripted backends LOUDLY at construction, in the one consistent vocabulary.
        # This decides on the CONSTRUCTED CLASS (the fact), not the requested name — a registry miss /
        # plugin / stray PERCEPT_*_BACKEND can land a scripted double after you asked for real.
        scripted = _scripted_seams(obj)
        obj._scripted = tuple(scripted)
        if scripted:
            log = get_logger("percept.agent")
            conductor_scripted = _is_scripted_backend(obj.conductor)
            # create() can only LOG — it CANNOT know whether a language path (on_utterance, or an
            # app-side voice engine) will be wired later. So the FATAL "scripted brain behind a live
            # voice" check lives at the language↔conductor crossing (+ the live-voice wiring points);
            # here we only warn.
            if conductor_scripted:
                # The common, LEGITIMATE code-armed pattern: goals via add_goal(), no voice. Non-fatal.
                log.warning(
                    "scripted_conductor: the conductor is SCRIPTED — language goal-authoring is OFF; "
                    "goals must be code-armed via add_goal(). A spoken 'tell me when…' will be ignored. "
                    "(scripted seams: %s; mode=%s)", ", ".join(scripted), resolved_mode,
                )
                if on_health is not None:
                    on_health("scripted_conductor")
            else:
                log.warning(
                    "scripted_backends: %s are SCRIPTED — this proves WIRING, not intelligence. "
                    "(mode=%s)", ", ".join(scripted), resolved_mode,
                )
            _emit_banner(resolved_mode, scripted)
        return obj

    # -- ADR-008 configuration boundary: thin entry points onto create() ----
    @classmethod
    def configure(
        cls,
        *,
        source_profile: Any = None,
        source_profile_explicit: bool = True,
        domain_pack: Any = None,
        execution_policy: Any = None,
        optimization_lock: Any = None,
        situation: "tuple[Any, Any, Any] | None" = None,
        registry: Any = None,
        settings: Settings | None = None,
        resolver: Any = None,
        **create_kwargs: Any,
    ) -> "Percept":
        """Build a core through the ADR-008 configuration boundary — resolve the six-object
        precedence chain (:class:`percept.resolver.ConfigResolver`), compile it to real runtime
        primitives (:func:`percept.runtime_compiler.runtime_compile`), and hand the result to
        :meth:`create` (unchanged). No goal is armed here — arm one afterward via
        :meth:`add_goal` or :func:`percept.watchplan_apply.apply_watch_plan`.

        This is an ADDITIVE door onto :meth:`create`, not a replacement: a bare ``Percept.create()``
        remains byte-for-byte the pre-ADR-008 path. Every keyword this method does not itself
        consume (``vision=``, ``mode=``, ...) passes straight through to :meth:`create` in
        ``create_kwargs``; an explicit ``create_kwargs['settings']``/``['sense_policy']``/
        ``['plan_mode']`` wins over what resolution produced (the caller building the core is always
        the most specific author).
        """
        from .contracts.watchplan import WatchPlan
        from .resolver import ConfigResolver
        from .runtime_compiler import runtime_compile

        # ConfigResolver.resolve requires a WatchPlan (its condition is never armed — configure()
        # resolves DEPLOYMENT/profile knobs only, no goal). The condition text is discarded; this
        # placeholder is not read by anything downstream of runtime_compile's Settings/sense_policy/
        # plan_mode fields.
        placeholder = WatchPlan(condition="percept.configure — deployment/profile resolution only")
        active_resolver = resolver if resolver is not None else ConfigResolver()
        resolved = active_resolver.resolve(
            placeholder,
            source_profile=source_profile,
            source_profile_explicit=source_profile_explicit,
            domain_pack=domain_pack,
            execution_policy=execution_policy,
            optimization_lock=optimization_lock,
            situation=situation,
            registry=registry,
            settings=settings,
        )
        compiled = runtime_compile(resolved)
        create_kwargs.setdefault("settings", compiled.settings)
        create_kwargs.setdefault("sense_policy", compiled.sense_policy)
        create_kwargs.setdefault("plan_mode", compiled.plan_mode)
        return cls.create(**create_kwargs)

    @classmethod
    def from_bundle(cls, bundle: "dict[str, Any]") -> "Percept":
        """Build a core from a plain mapping of the SAME keywords :meth:`configure` accepts — e.g. a
        deployment bundle loaded from JSON/YAML. Thin: ``Percept.from_bundle(b)`` is exactly
        ``Percept.configure(**b)``, nothing more."""
        return cls.configure(**dict(bundle))

    # -- goal authoring ------------------------------------------------------
    def add_goal(self, goal: Goal) -> Goal:
        # PROVENANCE: the STRUCTURED entry point stamps STRUCTURED_TRUSTED, OVERWRITING
        # whatever the caller supplied — a caller cannot claim interrupt-trust by pre-filling ``origin``.
        # ``replace`` (not in-place set) so this is correct even if Goal is later frozen.
        goal = replace(goal, origin=GoalOrigin.STRUCTURED_TRUSTED)
        self._arm_deadline(goal)  # resolve an authored duration into an absolute stamp at ARM time
        result = self.goals.create(goal)
        # Cache-invalidate on the direct arm() path too (only apply_goal_ops did before). A goal id
        # re-armed with a new condition must not be served a verdict cached under its OLD meaning — the
        # goal-change step of the invalidation order. Runs AFTER _arm_deadline so a legacy
        # deadline_s still raises loudly (never silently cached-then-armed).
        self.invalidate_cache(goal.id)
        return result

    def _arm_deadline(self, goal: Goal) -> None:
        """Resolve a ``deadline`` concern's authored DURATION into an absolute ``deadline_at`` on THIS
        agent's clock, at the moment the goal is armed (one field, one meaning, one clock).

        ``deadline_after_s`` (a duration) -> ``deadline_at = clock() + deadline_after_s``. The executor
        only ever compares against ``deadline_at``, on the same timeline the worker stamps frames on
        (wall for live capture). A legacy absolute ``deadline_s`` in ``goal.state`` (an old author or a
        v1 snapshot) has NO recoverable duration base, so it is refused LOUDLY — never silently reused
        under a new meaning that would fire on the wrong frame or never fire at all."""
        st = goal.state
        if st.get("rearm") != "deadline":
            return
        if st.get("deadline_at") is not None:
            return  # already absolute (e.g. a verification concern that stamped it directly)
        if "deadline_after_s" in st:
            st["deadline_at"] = self.clock() + float(st["deadline_after_s"])
            return
        if "deadline_s" in st:
            raise ValueError(
                f"goal {goal.id!r} carries a legacy 'deadline_s' (ambiguous absolute-vs-duration, the "
                "A-2 clock bug). Re-author as 'deadline_after_s' (a duration from arm). Refusing to arm "
                "a deadline concern whose stamp would fire on the wrong timeline or never fire."
            )

    # -- episode counting (the cup problem) ---------------------------------
    def count_event(
        self, goal_id: str, exemplar: Any, judge: "EpisodeJudge | None" = None, *,
        t: float | None = None,
        subject_key: str | None = None,
        mutation: "MutationCheck | None" = None,
    ) -> tuple[Any, bool]:
        """Fold a candidate counting-event into the goal's episode log; sync ``goal.state['count']``.

        ``judge`` answers the SUBJECT question only — which existing episode is this observation
        about (an LLM in real use; a stub in L1 — never assert backend intelligence with a fake).
        Whether that subject's episode is CONTINUED is a separate decision, evaluated against the
        concern's :class:`~percept.episodes.EpisodePolicy`. Returns ``(episode, is_new)``;
        :meth:`count_event_decision` returns the typed reason as well.
        For goals whose ``state['type'] == 'count'`` (e.g. "how many cups today").
        """
        return self.count_event_decision(
            goal_id, exemplar, judge, t=t, subject_key=subject_key, mutation=mutation
        )[:2]

    def count_event_decision(
        self, goal_id: str, exemplar: Any, judge: "EpisodeJudge | None" = None, *,
        t: float | None = None,
        subject_key: str | None = None,
        mutation: "MutationCheck | None" = None,
    ) -> tuple[Any, bool, ContinuationDecision]:
        """:meth:`count_event` plus the typed :class:`~percept.episodes.ContinuationDecision`.

        The decision separates the three collapsed questions: ``subject_matched`` (was
        this the same subject at all), ``is_new`` + ``reason`` (did it continue, and if not WHY),
        and the mutation seam. ``subject_matched and is_new`` — a same-subject observation that
        opened a SECOND episode — is counted as ``episode.denied.<reason>`` so a client can see the
        split in metrics, not only in logs.
        """
        now = self.clock() if t is None else t
        log = self.episodes.get(goal_id)
        if log is None:
            goal = self.goals.get(goal_id)
            log = EpisodeLog(
                goal_id,
                window_s=self.settings.episodes.window_s,
                max_episodes=self.settings.episodes.max_episodes,
                policy=episode_policy_of(goal) if goal is not None else None,
            )
            self.episodes[goal_id] = log
        decision = log.observe_decision(
            now, exemplar, judge, subject_key=subject_key, mutation=mutation
        )
        goal = self.goals.get(goal_id)
        if goal is not None:
            goal.state["count"] = log.count(now)
        self.metrics.incr("episode.new" if decision.is_new else "episode.continuation")
        if decision.denied:
            self.metrics.incr(f"episode.denied.{decision.reason}")
        return decision.episode, decision.is_new, decision

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
        # PROVENANCE: goal ops arrive from the CONDUCTOR (the language-authoring seam —
        # arm_from_language / _apply_decision), an UNTRUSTED author. Stamp every created goal
        # LANGUAGE_AUTHORED, OVERWRITING whatever it supplied, so its interrupt requests are clamped
        # (see AttentionPolicy). ``update`` ops can never touch ``origin`` (GoalRegistry drops that key).
        for op in ops:
            if op.op == "create" and op.goal is not None:
                op.goal = replace(op.goal, origin=GoalOrigin.LANGUAGE_AUTHORED)
        self.goals.apply(ops)
        for op in ops:
            gid = op.goal.id if op.op == "create" and op.goal is not None else op.goal_id
            # Resolve an authored deadline duration to an absolute stamp at ARM time — a goal armed
            # mid-run (voice) must get its deadline_at on the live clock, not silently stay a duration.
            if op.op in ("create", "update") and gid is not None:
                g = self.goals.get(gid)
                if g is not None:
                    self._arm_deadline(g)
            # cache invalidation — goal-change (1st in the order): a goal whose condition/lifecycle
            # changed must not be served from a verdict cached under its old meaning.
            if op.op in ("update", "complete", "cancel") and gid is not None:
                self.invalidate_cache(gid)

    # -- unified firing via the general executor ----------------------------
    @property
    def executor(self) -> BehaviorExecutor:
        """The single firing path, lazily bound to THIS agent's gate + metrics (shared state)."""
        if self._executor is None:
            self._executor = BehaviorExecutor(self.gate, self.metrics)
        return self._executor

    def _admit_evidence(self, key: tuple) -> int:
        """Reserve this judgment's ``evidence_seq`` at judge-DISPATCH time (ADR-001 §8).

        The live watch path is strictly single-flight, so nothing depends on the sequence today — it
        exists so a :class:`~percept.contracts.JudgmentEvent` emitted here correlates with the
        ``GateDecision`` a future worker-pool reducer would produce for the same key, instead of
        carrying a placeholder. LRU-bounded like the gate's own key maps.
        """
        if self._sequencer is None:
            from .sequencing import EvidenceSequencer  # noqa: PLC0415 — avoid an import cycle at module load
            self._sequencer = EvidenceSequencer()
        seq = self._sequencer.admit(key)
        tracked = self._sequencer.tracked_keys()  # first-admission order
        if len(tracked) > _SEQ_MAX_KEYS:
            for stale in tracked[: len(tracked) - _SEQ_MAX_KEYS]:
                self._sequencer.forget(stale)
        return seq

    @property
    def funnel(self) -> Funnel:
        """The standard telemetry funnel (seen→gated→judged→refused→fired), bound to THIS agent's
        metrics. The harness loop (percept.harness.tier0) AND the production runtime loop
        (percept.runtime.worker.run) both emit seen/gated via ``core.funnel``; the watch path
        here (perceive_judged) emits judged/refused/fired and the judge latency + token/cost, for
        either driver. Lazily built; OTel mirror attached when ``self.otel`` is set."""
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
        timing: JudgmentTimingBuilder | None = None,
        evidence_seq: int = 0,
        fold: "EpisodeFoldResult | None" = None,
    ) -> FireEvent | None:
        """One firing path — delegates to the :class:`~percept.behavior.BehaviorExecutor` (the general
        ``sense·key·accumulate`` executor that now owns the per-concern re-arm key-synthesis, the
        never-contract, the hard-quadrant escalation, and the confusion substrate). This method is a
        thin boundary that adapts the executor's ``Fired`` back to the legacy ``FireEvent`` shape; the
        watch + event paths route through here, so both now run on the unified executor (gate unchanged).

        ``timing``/``evidence_seq`` are the OPTIONAL judgment-funnel stamps the caller collected
        for this unit of evidence; when present the executor freezes them into a
        :class:`~percept.contracts.JudgmentEvent` that rides out on ``FireEvent.judgment``. Callers
        that take no stamps (the event path) pass nothing and get ``judgment=None`` — additive only.

        ``fold`` (OPTIONAL) is the episode-owned idempotency decision to GOVERN this fire with (see
        ``_document_episode_fold``); ``None`` (every caller that doesn't build one) is byte-for-byte
        the gate-only path this method has always run.
        """
        fired = self.executor.fire_judged(
            goal, verdict, base_key, program_of(goal), freshness=freshness, t=t, metric=metric,
            timing=timing, evidence_seq=evidence_seq, fold=fold,
        )
        if fired is None:
            return None
        return FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict,
                         fired.judgment, reason=fired.reason, modality=fired.modality,
                         roi=fired.roi, provenance=fired.provenance)

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

    # -- observability: coverage-loss fires with the concern's consequence ----
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
            # per-goal singleton -> exempt from the R-1 TTL sweep (a dropped edge would re-fire coverage-loss)
            action = self.gate.evaluate((goal.id, "__coverage__"), verdict, no_evict=True)
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
        """Attention allocation: order active concerns by consequence stakes so a
        budget-limited perception loop spends on the highest-consequence concerns first. Pure stakes,
        no confidence — a calibration-free job."""
        return sorted(
            self.goals.active(),
            key=lambda g: attention_weight(consequence_of(g)),
            reverse=True,
        )

    # -- absence-native firing: an expected concern overdue -----------------
    def check_deadlines(self, *, t: float | None = None) -> list[FireEvent]:
        """An EXPECTED concern not satisfied by its deadline fires "overdue".

        Absence IS the firing — there is no positive event to perceive, so this is perception-robust
        by construction: it fires on the LACK of a confident-yes, never on a guess. Rides the existing
        tick/scheduler loop and feeds a synthetic verdict to the UNCHANGED gate on a dedicated
        ``__overdue__`` key -> fires ONCE per overdue episode, re-arms if the expected event arrives.
        Only ``rearm == 'deadline'`` concerns (with a resolved ``deadline_at``) participate -> byte-for-byte.
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
            gtype = goal.state.get("type")
            ctx = self.executor.ctx(goal.id)
            ended = (
                bool(ctx.process_log)
                and ctx.last_step_at != float("-inf")
                and (now - ctx.last_step_at) > bound
            )
            if gtype == "process":
                verdict = Verdict(satisfied=ended, confidence=1.0, evidence="episode")
                action = self.gate.evaluate((goal.id, "__episode__"), verdict, no_evict=True)  # singleton
                self.metrics.incr(f"episode.gate.{action}")
                if action == FIRE:
                    answer = await self.summarize_process(goal.id, wipe=True)  # consolidate + reset
                    v = Verdict(satisfied=True, confidence=1.0, evidence=answer, running_answer=answer)
                    out.append(FireEvent(goal.id, FIRE, answer or self._episode_text(goal), None, v))
                continue
            # process-QA (mvp2) watch goal: a quiet episode end RESOLVES whatever was confidently
            # accumulated, with the coverage floor relaxed by one (``at_boundary``) — but still never
            # resolving an empty/uncertain answer. Same answer-path resolver as the in-tick path.
            if goal.state.get("process_qa") and ended:
                resolved = self._resolve_process_qa(goal, ctx, 1.0, at_boundary=True)
                if resolved is not None:
                    ctx.process_log = []  # end the episode → wipe accumulated coverage, re-arm
                    out.append(resolved)
        self._log_fires(out, now)
        return out

    def _episode_text(self, goal: Goal) -> str:
        return f"Here's what I saw for: {goal.condition}"

    def _overdue_text(self, goal: Goal) -> str:
        return f"Still waiting on: {goal.condition}"

    # -- the watch path ------------------------------------------------------
    async def _emit_trace(self, trace_id: str, t: float, stage: str, *, goal_id: str | None = None,
                          **data: Any) -> None:
        """Invoke ``on_trace`` safely: await an async callback, and never let a bad tracer break the
        loop (observability must not crash perception). Caller guards on ``self.on_trace is not None``
        first so no trace kwargs are built on the default (untraced) hot path."""
        cb = self.on_trace
        if cb is None:
            return
        try:
            result = cb(TraceEvent(trace_id=trace_id, stage=stage, t=t, goal_id=goal_id, data=data))
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 — a tracer is observability; swallow its failures
            pass

    def _shadow_observe(self, verdict: Any) -> "ShadowObservation":
        """SHADOW: the resolved subject + the proposed :class:`ShadowObservation` for a
        watch-path verdict. ADVISORY ONLY — it OBSERVES what the fold would propose and governs
        nothing (no gate call, no re-arm, no suppression). The watch path binds no continuation
        decision, so ``decision=None`` (an ``unbound`` fold, or ``unknown`` when the judge saw a
        subject it could not identify); the subject is the :class:`~percept.contracts.SubjectRef`
        the judge emitted, if any. Never raises — a bad verdict yields an unbound shadow."""
        subject = getattr(verdict, "subject", None)
        if not isinstance(subject, SubjectRef):
            subject = None
        unknown = subject is not None and subject.resolution == "unknown"
        operation = getattr(verdict, "operation", None)
        # An unknown subject carries no key: it is a visible abstention, never a bound identity that
        # could drive a fold, so drop it to None on the ref carrier and mark the fold unknown.
        subject_ref = subject if (subject is not None and subject.resolution == "bound") else None
        return self.executor.shadow_observe(
            None, subject=subject_ref, operation=operation, unknown=unknown
        )

    def _document_episode_fold(
        self, goal: Goal, verdict: Verdict, t: float
    ) -> "EpisodeFoldResult | None":
        """Episode grouping + :class:`~percept.behavior.EpisodeFoldResult`-owned idempotency, GOVERNING,
        scoped to content-key concerns ONLY.

        The shadow-mode advisory (``_shadow_observe``) always folds ``decision=None`` — an ``unbound``
        shadow that changes nothing, for every concern. This method is the narrow exception: when a
        concern SELECTS the content-key resolver (``goal.state['identity_policy'] == 'content_key'``,
        the per-concern selector in ``identity_policy.py`` — screen/document scope, never inferred from
        the source), it maintains a REAL per-goal :class:`~percept.episodes.EpisodeLog` and folds the
        resulting :class:`~percept.episodes.ContinuationDecision` through the SAME pure
        :meth:`~percept.behavior.BehaviorExecutor.fold_episode` the shadow path already exercises — and
        the caller (:meth:`perceive_judged`) now THREADS the answer into ``_fire_concern(fold=...)`` so
        it actually governs the gate interaction (idempotent suppression + re-arm; see
        :func:`percept.behavior.episode_fold`).

        Every other concern — including a claim-slot / conversational (mutation) goal — returns
        ``None`` here, so ``_fire_concern`` runs its byte-for-byte gate-only path and stays shadow-only
        via the UNTOUCHED ``_shadow_observe``/``STAGE_SHADOW_FOLD`` trace. This method never changes
        gate mechanics itself — it only decides whether a fold is built at all.

        Two things this method refuses to do:

        * **fabricate a content key** — a verdict carrying no subject at all is not grouped (returns
          ``None``, the same "nothing to fold" answer an undeclared concern gets); an ``unknown``
          subject folds through :func:`percept.behavior.episode_fold`'s own visible-abstention rule
          (never rearms, never a manufactured key).
        * **let an out-of-order observation corrupt the live log** — this per-goal log only ever
          advances in time (a high-water mark on ``t``). Async/off-loop judge dispatch can resolve
          completions out of chronological order; feeding a stale ``t`` into the log would rewind an
          episode's ``last_seen`` and could fabricate a spurious continuation/new-episode divergence on
          a LATER, in-order observation. A late arrival instead degrades this one tick to the gate-only
          path (``None``) — the same safe answer an undeclared concern always gets — and is counted
          (``episode.document.out_of_order_skipped``), never silently swallowed.
        """
        if identity_policy_of(goal) != "content_key":
            return None
        subject = getattr(verdict, "subject", None)
        if subject is None:
            return None  # nothing to group — gate-only, exactly like an undeclared concern
        hwm = self._doc_episode_hwm.get(goal.id)
        if hwm is not None and t < hwm:
            self.metrics.incr("episode.document.out_of_order_skipped")
            return None
        self._doc_episode_hwm[goal.id] = t
        log = self.episodes.get(goal.id)
        if log is None:
            log = EpisodeLog(
                goal.id,
                window_s=self.settings.episodes.window_s,
                max_episodes=self.settings.episodes.max_episodes,
                policy=episode_policy_of(goal, default=EPISODE_POLICIES["document_presence"]),
            )
            self.episodes[goal.id] = log
        decision = log.observe_decision(t, verdict)
        unknown = subject_is_unknown(verdict)
        return self.executor.fold_episode(
            decision, operation=getattr(verdict, "operation", None), unknown=unknown
        )

    def _claim_episode_fold(
        self, goal: Goal, verdict: Verdict, t: float
    ) -> "EpisodeFoldResult | None":
        """CLAIM MODE — the claim-slot twin of :meth:`_document_episode_fold`.

        Document mode flipped episode grouping + idempotency to GOVERNING for concerns that SELECT the
        content-key resolver; this method flips the SAME seam for concerns that SELECT the claim-slot
        one (``goal.state['identity_policy'] == 'claim_slot'``) — a goal is never both, so the two
        methods are mutually exclusive by construction and either may be called unconditionally by the
        caller. Every other concern (including an undeclared one) still gets ``fold=None`` here, exactly
        as it always has.

        The three pieces this wires together for claim-slot concerns:

        1. **The mutation channel is authority-gated FIRST** (``percept.claims``): the verdict's speech
           act only reaches the episode fold as its EFFECTIVE operation —
           :meth:`~percept.claims.ClaimState.apply` downgrades a ``replace``/``retract``/``close`` from
           an asserter this concern's :class:`~percept.contracts.ConcernAuthority`
           (:func:`~percept.claims.concern_authority_of`, ``goal.state['authority']``) does not permit
           to a non-mutating ``mention`` — TRACED (``claim.authority_downgrade``), never silent — so a
           seller's textbook replace can never move a buyer-authoritative value. A concern that
           authored no authority gets the unrestricted default, byte-for-byte.
        2. **The SAME episode fold document mode governs with**: the effective operation feeds
           :meth:`~percept.behavior.BehaviorExecutor.fold_episode` exactly like a content-key concern's
           verdict does, via the SAME ``episode_mutation_of`` map — so an applied ``replace``/``retract``
           re-arms and fires a superseding revision (``fold.supersedes``), and the gate/incident-delivery
           machinery (``_fire_concern(fold=...)`` → ``BehaviorExecutor.fire_judged`` →
           ``SupersessionSink``) is UNCHANGED. A downgraded (unauthorized) mutation folds as a plain
           ``confirmed`` re-statement — it never re-arms, never supersedes, never moves the value.
        3. **The corrected value + its own revision chain, on the episode** — unlike a content-key
           concern (which never carries a "value"), a claim slot's whole point is a corrected VALUE.
           Whenever the claim state machine actually APPLIES a write (``outcome.applied``), the episode's
           ``value`` is updated (:meth:`~percept.episodes.EpisodeLog.annotate`) and, for a genuine state
           change (``updated``/``retracted``/``closed`` — never the initial ``opened`` row, already
           recorded by ``observe_decision``), a :class:`~percept.episodes.Revision` is appended
           (:meth:`~percept.episodes.EpisodeLog.record_revision`) naming the value the slot became. A
           downgraded or non-mutating act writes nothing — the value and the chain stay exactly as they
           were. :meth:`episode_current` reads both back off the episode.

        Like :meth:`_document_episode_fold`, this method refuses to fabricate: a stale/out-of-order ``t``
        degrades the tick to gate-only (counted, ``episode.claim.out_of_order_skipped``, never silently
        swallowed) rather than risk rewinding the log's high-water mark.
        """
        if identity_policy_of(goal) != "claim_slot":
            return None
        hwm = self._claim_hwm.get(goal.id)
        if hwm is not None and t < hwm:
            self.metrics.incr("episode.claim.out_of_order_skipped")
            return None
        self._claim_hwm[goal.id] = t
        log = self.episodes.get(goal.id)
        if log is None:
            log = EpisodeLog(
                goal.id,
                window_s=self.settings.episodes.window_s,
                max_episodes=self.settings.episodes.max_episodes,
                policy=episode_policy_of(goal, default=EPISODE_POLICIES["conversational_claim"]),
            )
            self.episodes[goal.id] = log
        state = self._claim_states.get(goal.id)
        if state is None:
            state = ClaimState(claim_id=goal.id)
            self._claim_states[goal.id] = state
        authority = concern_authority_of(goal)
        outcome = state.apply(
            getattr(verdict, "operation", "assert"),
            value=getattr(verdict, "value", None),
            normalized_value=getattr(verdict, "normalized_value", None),
            asserted_by=getattr(verdict, "asserted_by", None),
            authority=authority,
        )
        # A fixed subject_key (== goal.id): a claim slot is ONE persistent thing, not many distinct
        # documents, so there is no identity to resolve here — unlike content-key grouping, this never
        # takes the "unknown subject" branch (an explicit subject_key always wins in observe_decision).
        decision = log.observe_decision(t, verdict, subject_key=goal.id)
        self._claim_episode_id[goal.id] = decision.episode.id
        fold = self.executor.fold_episode(decision, operation=outcome.effective)
        if outcome.applied:
            log.annotate(decision.episode, value=outcome.value_after)
            if not decision.is_new and fold.mutation in ("updated", "retracted", "closed"):
                # The initial write's "opened" row was already recorded by observe_decision; only a
                # genuine SUBSEQUENT state change (an accepted replace/retract/close) appends a new one.
                log.record_revision(
                    decision.episode, fold.mutation, t=t, value=outcome.value_after
                )
        return fold

    def episode_current(self, goal_id: str) -> "Episode | None":
        """The current authoritative episode for a claim-slot concern.

        ``.value`` is the corrected value (the last APPLIED assert/replace/retract, never a downgraded
        or merely-mentioned one) and ``.revisions`` is the bounded state-change chain, so a correction
        is never a silent overwrite: the revision it superseded stays on the timeline and the new row
        names what the slot became. Returns ``None`` when this goal has no claim-slot episode yet —
        nothing observed, or the goal never selected ``identity_policy='claim_slot'``.
        """
        log = self.episodes.get(goal_id)
        episode_id = self._claim_episode_id.get(goal_id)
        if log is None or episode_id is None:
            return None
        return log.get(episode_id)

    async def perceive_judged(
        self, frame: Any, source: str = "camera", *, t: float | None = None,
        expected_goals_revision: int | None = None,
    ) -> list[FireEvent]:
        """Judge every active *watch* goal against ``frame`` and emit gate actions on rising edges.

        Event / scheduler goals run through their own paths (:meth:`perceive_events`, :meth:`tick`)
        but feed the **same three-state gate** — one gate, two perception tracks.

        ``expected_goals_revision`` (the non-blocking-dispatch seam): the goal-registry revision this
        frame was ADMITTED under. When set, the call aborts — returning without applying anything —
        the moment ``self.goals.revision`` no longer matches, checked **after every network await and
        before the mutation it feeds**. This is what makes an OFF-LOOP judgment safe: while the vision
        call is in flight the conductor may disarm a goal or replace it under the same id, and a stale
        verdict must not flip a rising-edge latch, write a disposition/action record, stamp
        coverage/deadline satisfaction, or judge old evidence as-if under the newly authored plan.
        Filtering the *returned* fires would be too late — the gate mutates inside this call
        (:meth:`_fire_concern` / the executor), so the abort lives here, before the gate.

        Abort semantics (what is protected vs what may proceed):

        - **Aborted on staleness** — every goal-conditioned application: gate/latch evaluation,
          executor context (running answer / process log / transition state), ``last_supported`` /
          ``last_satisfied`` stamps, action-log records, and the confusion auto-refinement pass.
        - **Allowed to proceed** — frame-level, goal-independent perception: the rolling FrameWindow
          push and the entity-graph ingest of ``vision.detect`` observations. The graph records what
          was IN the frame regardless of which goals asked; that world-state stays valid across a
          goal-registry change (and the verdict cache is keyed by condition text, so a same-id replace
          can't be served a stale write).
        - **Mid-loop staleness** (multi-goal): goals already gated under the matching revision keep
          their (already-applied) fires — swallowing them would strand a flipped latch with no emitted
          incident — and all *subsequent* judgment/mutation is aborted. With a single active goal this
          reduces to the plain contract: abort ⇒ ``[]`` and untouched state.

        ``None`` (the default) never checks — byte-for-byte the inline path.
        """
        now = self.clock() if t is None else t

        def _stale() -> bool:
            return (
                expected_goals_revision is not None
                and self.goals.revision != expected_goals_revision
            )
        # Per-frame decision trace (opt-in via on_trace; no-op otherwise). trace_id correlates this
        # frame's stages; derived from source+timestamp — deterministic WHEN t is supplied (as in
        # tests). With t=None it uses the wall clock, like any production run.
        trace_id = f"{source}@{now}"
        tracing = self.on_trace is not None  # guard so the default (untraced) path builds no trace kwargs
        # The judge-usage record emitted deep inside _timed_judge belongs to THIS frame's trace, so
        # park the id where that wrapper can find it (it takes only the judge's own args). One frame
        # is in judgment at a time per core in both dispatch modes, so this is the current frame's id.
        self._usage_trace_id = trace_id
        self._usage_source_ts = now
        # The judgment clock (ADR-001 §1/§5). Two domains, never mixed: every stamp below is read
        # from THIS core's injected ``clock`` (monotonic in production, FakeClock in tests), while the
        # caller's stream timeline ``t`` — when one was supplied — rides along ONLY as the untrusted
        # ``source_media_ts``. ``ingest_observed_ts`` is stamped here, at local receipt of the frame:
        # the honest thing this method knows. Capture-side stamps upstream of the core (a camera's own
        # capture time, the runtime worker's tier-0 admission) are NOT threaded through this seam, so
        # they are deliberately left out rather than approximated from a later read.
        ingest_observed_at = self.clock()

        if tracing:
            await self._emit_trace(trace_id, now, "source.frame")
        self._window(source).push(now, frame)  # feed the rolling window for temporal perception
        if _stale():
            return []  # admitted under an old revision; spend nothing, apply nothing
        observations = await self.vision.detect(frame, self.goals.active())
        self.graph.ingest(observations, frame, source, t=now)  # goal-independent world-state: proceeds
        focus = self.graph.in_focus()
        if _stale():
            return []  # revision moved during the detect call — nothing goal-conditioned applied yet

        events: list[FireEvent] = []
        for goal in self.goals.active():
            if _stale():
                break  # revision moved during a prior goal's judge — stop before judging under it
            gtype = goal.state.get("type", "watch")
            if gtype == "transition":
                ev = await self._perceive_transition(goal, frame, focus, stale=_stale)
                if ev is not None:
                    events.append(ev)
                continue
            if gtype == "process":
                # Item 4: log a delta, never fire here
                await self._observe_and_log(goal, frame, source, now, stale=_stale)
                continue
            if gtype != "watch":
                continue  # event/count/remind goals are handled by their dedicated paths
            ctx = self.executor.ctx(goal.id)
            # HOW to look = the authored/preset spec, floored by the resolved profile's plan_mode —
            # "window" hands the judge an ordered clip, never a montage — then escalated by this
            # concern's current ladder rung (single→strip→+audio) when it's been recurrently ambiguous
            # (set in _deliberate_confused).
            spec = _escalated_perception(
                _planned_perception(program_of(goal).perception, self.plan_mode), ctx.perception_level
            )
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
            # The EvidenceExtractor seam (percept.extraction): ``None`` by default -- every call
            # below is skipped and every judge call stays byte-for-byte. When set, it runs at the
            # SAME point the judge is about to be called (frame + roi already resolved), and its
            # facts are merged into context under the RESERVED "extractor" sub-dict — visible to
            # the judge and, independently, to the trace (so "did the facts reach the judge" is
            # answerable from the .ptrace alone, not just from reading this call site).
            extractor_ctx = None
            if self.extractor is not None:
                enrichment = self.extractor.extract(goal.condition, frame, roi)
                if tracing:
                    await self._emit_trace(trace_id, now, "extractor.enriched", goal_id=goal.id,
                                           facts=dict(enrichment.facts))
                if not enrichment.is_empty:
                    extractor_ctx = {EXTRACTOR_CONTEXT_KEY: enrichment.to_dict()}
            judge_context = (
                process_ctx if extractor_ctx is None else
                {**(process_ctx or {}), **extractor_ctx}
            )
            # DirectJudge routing (percept.direct_judge): ``None`` by default -- direct_result stays
            # None, the block below is skipped, and every judge call is byte-for-byte unchanged.
            # CONFIDENT is handled inside the try/except below (it substitutes the judge call
            # itself, spending zero vision calls); UNCERTAIN's facts are merged into the SAME
            # reserved "extractor" context key the EvidenceExtractor seam uses, here, BEFORE the
            # judge call, so a direct rule's partial signal reaches the judge exactly like an
            # extractor's would.
            direct_result = None
            if self.direct_judge is not None:
                direct_result = self.direct_judge.route(
                    goal.condition, frame, roi,
                    facts=(extractor_ctx or {}).get(EXTRACTOR_CONTEXT_KEY, {}).get("facts", {}),
                    stakes=goal.state.get("stakes", "medium"),
                )
                if direct_result.outcome == "uncertain" and direct_result.facts:
                    prior = dict((judge_context or {}).get(EXTRACTOR_CONTEXT_KEY) or {})
                    prior["facts"] = {**prior.get("facts", {}), **direct_result.facts}
                    judge_context = {**(judge_context or {}), EXTRACTOR_CONTEXT_KEY: prior}
            # Computed BEFORE the judge call (not just on success) so a degraded tick can be charged
            # to the SAME key the gate/executor use ("the reducer key") — see _note_judge_degraded.
            base_key = (goal.id, focus.id if focus else None)
            # Open this judgment's timing record. ``admitted_ts`` is stamped HERE: tier-0
            # detection and focus resolution are done and this concern is about to spend a judge
            # call, which is exactly "admitted to the judge funnel" (ADR-001 §5). ``source_media_ts``
            # is the caller's stream timeline when one was passed and stays None otherwise — it is
            # never back-filled from the monotonic clock, and never enters latency math.
            judgment_timing = JudgmentTimingBuilder(clock=self.clock)
            judgment_timing.mark_ingest_observed(at=ingest_observed_at)
            judgment_timing.mark_admitted()
            judgment_timing.set_source_media_ts(t)
            evidence_seq = self._admit_evidence(base_key)  # reserved at DISPATCH, per ADR-001 §8
            if tracing:
                await self._emit_trace(trace_id, now, "judge.started", goal_id=goal.id,
                                       temporal=getattr(spec, "temporal", None), roi=roi is not None)
            try:
                judgment_timing.mark_judge_started()
                if direct_result is not None and direct_result.outcome == "confident":
                    # The confident DirectJudge path: use its normalized Verdict AS the judge's
                    # verdict — self._judge / self.vision are never called (zero vision calls).
                    # The Gate stays sole authority: `verdict` flows into the exact same
                    # freshness/classify/gate path below as a real judge call would.
                    verdict = direct_result.verdict
                else:
                    verdict = await self._judge(
                        goal, frame, roi, corrections, entity_id=focus.id if focus else None,
                        frames=strip, audio=audio, context=judge_context,
                    )
                judgment_timing.mark_judge_completed()
            except _JudgeDegraded as degraded:
                # Fail-closed boundary (ADR-001 §10 invariant 4): the judge call raised (timeout /
                # provider error) or returned a malformed/partial response. NEVER read .confidence/
                # .satisfied off that — degrade THIS goal's tick to an explicit abstention (ASK-class:
                # same telemetry a real low-confidence read gets), never a FIRE, and never let the gate
                # see it at all (the gate's armed/dwell edge state must stay exactly as a failed judge
                # call found it — belt-and-suspenders against a fault silently arming/disarming an
                # edge). The exception is swallowed HERE — it must never escape to worker.py's run()
                # loop and end the watch.
                self.metrics.incr("judge.degraded")
                self.funnel.refused(concern=goal.id, camera=source)
                self._note_judge_degraded(base_key, degraded.cause)  # queues on_health for worker.py
                get_logger("percept.agent").warning(
                    "judge.degraded: goal=%r cause=%r — degrading this tick to ASK/SILENT "
                    "(fail-closed)", goal.id, degraded.cause,
                )
                if tracing:
                    await self._emit_trace(
                        trace_id, now, "judge.degraded", goal_id=goal.id, cause=repr(degraded.cause),
                    )
                    await self._emit_trace(trace_id, now, "gate.classified", goal_id=goal.id,
                                           truth="unknown", action="ask",
                                           entity_id=focus.id if focus else None, degraded=True)
                continue  # next goal this SAME frame; the watch loop (and run) continues either way
            self._note_judge_ok(base_key)  # a well-formed verdict resets this key's degrade streak
            if tracing:
                await self._emit_trace(trace_id, now, "judge.finished", goal_id=goal.id,
                                       satisfied=getattr(verdict, "satisfied", None),
                                       confidence=getattr(verdict, "confidence", None))
            if _stale():
                break  # THE choke point: the judge returned, but the plan changed while it was in
                # flight — abort before ANY application (ctx, coverage stamps, gate, disposition).
            resolved: FireEvent | None = None
            if process_ctx is not None and getattr(verdict, "running_answer", ""):
                ctx.running_answer = verdict.running_answer  # accumulate the model's rewritten summary
                # Accumulated-coverage stamp (answer-path only): each evidence-bearing observation that
                # folded into the running answer is a unit of coverage. Reuse the per-concern process_log
                # the boundary already owns, so len(process_log) IS the evidence count — no gate state.
                # Bounded so a 24/7 process_qa concern that never hits a boundary can't leak memory; the
                # resolver only reads it as a >= floor count, so a cap never weakens resolution.
                ctx.process_log.append(verdict.running_answer)
                if len(ctx.process_log) > 256:
                    del ctx.process_log[: len(ctx.process_log) - 256]
                ctx.last_step_at = now
                # RESOLVE the running answer once it's complete/confident enough (general — coverage +
                # confidence, never the question text). De-duped through the UNCHANGED gate on a
                # dedicated __resolved__ key, exactly like the coverage/episode/overdue absence fires.
                resolved = self._resolve_process_qa(goal, ctx, verdict.confidence)
            freshness = self._freshness(focus, now, goal.id)
            truth = classify(verdict, freshness, self.gate.threshold, self.gate.fresh_min)
            if truth == "unknown":
                self.funnel.refused(concern=goal.id, camera=source)  # abstained — refused to guess (ASK)
            else:
                self.last_supported[goal.id] = now  # a confident read = perceptual coverage
                ctx.perception_level = 0  # decay: the (escalated) look resolved → back to the cheap default
            if truth == "yes":
                self.last_satisfied[goal.id] = now  # deadline: the expected event happened
            # base_key computed earlier (before the judge call) — reused here unchanged.
            # GOVERNING for content-key AND claim-slot concerns only; every other concern gets
            # fold=None (byte-for-byte, unchanged). A goal is never both types (identity_policy
            # selects exactly one), so at most one of these returns non-None.
            doc_fold = self._document_episode_fold(goal, verdict, now)
            claim_fold = self._claim_episode_fold(goal, verdict, now)
            ev_fire = self._fire_concern(goal, verdict, base_key, freshness=freshness, t=now,
                                         timing=judgment_timing, evidence_seq=evidence_seq,
                                         fold=doc_fold if doc_fold is not None else claim_fold)
            if ev_fire is not None:
                if ev_fire.action == FIRE:
                    self.funnel.fired(concern=goal.id, camera=source)  # a rising-edge fire emitted
                events.append(ev_fire)
            if resolved is not None:  # the process-QA running answer crossed the bar this tick
                self.funnel.fired(concern=goal.id, camera=source)
                events.append(resolved)
            if tracing:
                # AFTER both fire paths: label a FIRE whenever an incident was actually emitted this tick
                # — either the gate fired OR a resolved process-QA answer crossed the bar (which fires even
                # when the gate abstained/ASK'd on this same tick). Only ask/silent when nothing fired.
                if resolved is not None or (ev_fire is not None and ev_fire.action == FIRE):
                    action = "fire"
                elif ev_fire is not None:
                    action = ev_fire.action
                else:
                    action = "silent"
                # The gate's own INPUTS ride along with its output, so a recorded trace is
                # replayable through the (deterministic) gate without the original frame or any
                # model call — see percept.replay. Purely additive trace data; nothing branches on it.
                await self._emit_trace(trace_id, now, "gate.classified", goal_id=goal.id,
                                       truth=truth, action=action,
                                       reason=_trace_reason(truth, action),
                                       entity_id=focus.id if focus else None,
                                       satisfied=getattr(verdict, "satisfied", None),
                                       confidence=getattr(verdict, "confidence", None),
                                       freshness=freshness,
                                       hold_s=program_of(goal).hold_s)
                # Project the frozen JudgmentEvent (the one production construction site,
                # behavior.py:_judgment) into the trace stream so the SAME evidence this decision
                # judged is resolvable from the .ptrace. Only when THIS tick's fire actually carried
                # one (the judged watch path only — a degraded tick never reaches here, and
                # episode/rhythm/boundary/transition/absence fires take no judge-funnel stamps and
                # build no JudgmentEvent at all). The RECORDER (not this caller) stamps
                # ``data["evidence_seq"]`` as the seq of this SAME trace_id's ``source.frame``
                # record; ``dispatch_seq`` carries the JudgmentEvent's OWN ADR-001 §8 per-key dispatch
                # sequence, a distinct number, so neither meaning is lost (see
                # tracing.STAGE_JUDGMENT_EVENT). Purely additive observability — governs nothing.
                if ev_fire is not None and ev_fire.judgment is not None:
                    await self._emit_trace(trace_id, now, STAGE_JUDGMENT_EVENT, goal_id=goal.id,
                                           action=ev_fire.judgment.action,
                                           dispatch_seq=ev_fire.judgment.evidence_seq,
                                           entity_id=focus.id if focus else None)
                # SHADOW — emit the subject + the fold the model WOULD have proposed, as ADVISORY
                # trace data. It changes NOTHING (computed after the gate already classified, governs
                # no firing) and lands in the ``shadow.fold`` stage so a client's media-OFF .ptrace
                # carries something gradeable. Only built when a tracer is attached (default path does
                # no shadow work at all).
                await self._emit_trace(trace_id, now, STAGE_SHADOW_FOLD, goal_id=goal.id,
                                       **self._shadow_observe(verdict).to_advisory())
        self._log_fires(events, now)
        if _stale():
            return events  # aborted mid-loop: skip auto-refinement (it authors goal changes)
        await self._deliberate_confused(now)  # Posture B: auto-refine recurrently-ambiguous concerns
        return events

    async def _perceive_transition(
        self, goal, frame, focus, *, stale: Callable[[], bool] | None = None
    ) -> FireEvent | None:
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
        if stale is not None and stale():
            return None  # goals changed while the from/to judges were in flight — apply nothing
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

    async def _observe_and_log(
        self, goal: Goal, frame: Any, source: str, now: float,
        *, stale: Callable[[], bool] | None = None,
    ) -> None:
        """Item 4 (log→summarize→wipe): observe THIS clip's step delta and append it to the per-concern
        log. NEVER fires here — a process goal's answer is produced by :meth:`summarize_process` at the
        episode boundary / on demand. The gate is untouched (no verdict routed)."""
        ctx = self.executor.ctx(goal.id)
        spec = _escalated_perception(
            _planned_perception(program_of(goal).perception, self.plan_mode), ctx.perception_level
        )
        strip = self._strip_frames(source, spec)
        self.funnel.judged(concern=goal.id, camera=source)  # an observe call is being spent
        # Process goals have no entity/focus concept here; a 1-tuple key never collides with a watch
        # goal's (goal_id, entity_id) 2-tuple key (see perceive_judged's base_key).
        process_key = (goal.id,)
        try:
            verdict = await self._judge(
                goal, frame, None, {}, frames=strip, context={"observe_step": True}
            )
        except _JudgeDegraded as degraded:
            # A failed/malformed observe-step judge logs NOTHING this tick (never a phantom step
            # off a garbled/timed-out response) and never crashes the run — this goal never touches
            # the gate anyway (see the docstring above), so there is nothing further to degrade.
            self.metrics.incr("judge.degraded")
            self._note_judge_degraded(process_key, degraded.cause)  # queues on_health for worker.py
            get_logger("percept.agent").warning(
                "judge.degraded (observe_step): goal=%r cause=%r — logging nothing this tick "
                "(fail-closed)", goal.id, degraded.cause,
            )
            return
        self._note_judge_ok(process_key)  # a well-formed verdict resets this key's degrade streak
        if stale is not None and stale():
            return  # goals changed while the observe call was in flight — don't log a step under it
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

    def _resolve_process_qa(self, goal: Goal, ctx, confidence: float, *, at_boundary: bool = False
                            ) -> FireEvent | None:
        """Emit a RESOLVED process-QA answer once the accumulated running answer is complete/confident
        enough (Item 2 mvp2). This closes the gap behind ``unresolved_rate == 1.0``: the running answer
        accumulates per escalation but the gate only fires on a confident-YES verdict, so an
        evidence-gathering judge never RESOLVES — the answer the metric can score is never produced.

        GENERAL, not per-question: :func:`~percept.concern.resolution_ready` reads only accumulated
        COVERAGE (``len(ctx.process_log)`` — the evidence-bearing observations folded in), SUBSTANCE, and
        CONFIDENCE. An unseen process question resolves with zero new branches. The gate stays UNCHANGED:
        de-dup rides a synthetic verdict on a dedicated ``(goal, __resolved__)`` key — the same mechanism
        :meth:`check_observability` / :meth:`check_episode_boundary` / :meth:`check_deadlines` already
        use — so it resolves ONCE per accumulation and re-arms only when fresh evidence pushes a new,
        different answer. Precision-guarded: a thin/empty/uncertain running answer never resolves."""
        answer = (ctx.running_answer or "").strip()
        ready = resolution_ready(
            answer, len(ctx.process_log), confidence, at_boundary=at_boundary
        )
        # The answer TEXT used to be folded into the gate key (``(goal, __resolved__, answer)``) — a
        # permanent key per distinct answer, unbounded over a long process. The discriminator now lives in
        # ctx: a *changed* answer (more evidence rewrote it) rolls it over and re-arms the ONE singleton
        # key, so a new answer resolves again while a repeated identical answer fires only once. Marked
        # no_evict (a per-goal singleton, exempt from the TTL sweep).
        key = (goal.id, "__resolved__")
        if ctx.last_resolved_answer is not None and answer != ctx.last_resolved_answer:
            self.gate.rearm(key)
        ctx.last_resolved_answer = answer
        verdict = Verdict(satisfied=ready, confidence=confidence, evidence=answer, running_answer=answer)
        action = self.gate.evaluate(key, verdict, no_evict=True)
        self.metrics.incr(f"process_qa.resolve.{action}")
        if action != FIRE:
            return None
        return FireEvent(goal.id, FIRE, goal.say or answer, None, verdict)

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
        """Assemble the temporal strip/window per the ``PerceptionSpec`` from the rolling FrameWindow:
        the earlier-context frames (oldest→newest, EXCLUDING the current frame, which the backend appends
        last). Returns None for single-frame perception — byte-for-byte today.

        LATEST-WINS: the window is assembled HERE, inside the judging tick, against a FrameWindow
        whose last entry is the frame being judged (``perceive_judged`` pushes before it judges). The
        async scheduler (``percept/runtime/worker.py::_JudgeScheduler``) is single-flight and drops a
        superseded candidate BEFORE it ever reaches this method, so a frame that lost the latest-wins
        race contributes nothing to any window, and the window a launched judgment sees is always
        anchored on its own admitted frame — never on a newer one that arrived mid-flight."""
        spec = replace(spec, temporal=_normalize_plan_mode(spec.temporal))  # shim: sprite -> window
        if spec.temporal == "single":
            return None
        earlier = self._window(source).frames()[:-1]  # drop the current frame (passed as `frame`)
        if not earlier:
            return None
        # A clip window (window_s) bounds the strip by TIME — keep only the earlier frames within
        # window_s seconds of the current frame, so "a 4s clip" means the last 4s, not a frame count.
        if getattr(spec, "window_s", None):
            now_t = self._window(source).frames()[-1][0]  # current frame's timestamp
            cutoff = now_t - spec.window_s
            earlier = [(t, f) for t, f in earlier if t >= cutoff]
            if not earlier:
                return None
        want = (spec.n_frames - 1) if spec.temporal == "strip" else len(earlier)
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
        # Ordered, oldest→newest. The sprite montage that used to collapse these into ONE image is gone
        # from the plan path: the judge sees the real ordered frames.
        return [f for _, f in picked]

    async def _timed_judge(self, *args: Any, **kwargs: Any) -> Verdict:
        """Call the vision judge, timing wall-clock latency and capturing token/cost off the verdict.

        ONE place wraps the real cloud call so every judge path (strip/audio, cache-miss, audit)
        records the same telemetry: ``judge.latency`` (always — we time it here) plus optional
        ``judge.tokens.*`` / ``judge.cost.usd`` when the backend annotated its verdict
        (:func:`~percept.observability.judge_usage`). Backends that don't report usage leave those
        hooks unused; the timing is unconditional.

        THE COST WIRE. The same usage dict is also emitted as a ``judge.usage`` TRACE
        event, which is what carries token/cost out to ``TraceSummary`` (the funnel's counters never
        reached the trace surface, so every traced run reported ``usd: 0.0`` no matter what it spent).
        It is emitted HERE, not at the runtime's judge-result drain, because this wrapper is the only
        site EVERY judge call passes through — inline and async dispatch, cache-miss, strip/audio and
        clip audit alike. A drained runtime result carries FIRES only; metering there would omit
        every judgment that didn't fire and report a confident undercount.
        """
        t0 = self.clock()
        verdict = await self.vision.judge(*args, **kwargs)
        usage = judge_usage(verdict)
        self.funnel.record_judge(latency_s=self.clock() - t0, **usage)
        if usage and self.on_trace is not None:
            await self._emit_trace(
                getattr(self, "_usage_trace_id", None) or f"judge@{t0}",
                getattr(self, "_usage_source_ts", None) if getattr(self, "_usage_source_ts", None) is not None else t0,
                "judge.usage",  # percept.tracing.STAGE_JUDGE_USAGE (string here to keep agent import-light)
                model=self._judge_model(),
                **usage,
            )
        return verdict

    def _judge_model(self) -> str | None:
        """The model id the vision judge bills against, when the adapter exposes one.

        Read off the adapter (public ``model`` first, then the backends' ``_model``); ``None`` for a
        fake/duck-typed judge that names no model. ``None`` means the call is NOT priced — see
        :func:`percept.slo.price_usage_or_none`. We never substitute a default price here: a guessed
        dollar figure printed as a run's cost is worse than no figure at all.
        """
        v = self.vision
        for attr in ("model", "_model"):
            m = getattr(v, attr, None)
            if isinstance(m, str) and m:
                return m
        return None

    async def _safe_timed_judge(self, *args: Any, **kwargs: Any) -> Verdict:
        """Fail-closed boundary around a live judge call (ADR-001 §10 invariant 4).

        A judge call fails one of two ways: it RAISES (timeout / provider error — the designed,
        typed failure signal a real backend's retry-then-raise contract already produces, see
        ``percept.backends._resilience.with_retry``), or it RETURNS something that is not a
        well-formed :class:`Verdict` (malformed / partial — e.g. a plain dict, or a dict-shaped
        truncated-JSON payload missing required fields). Neither shape may reach
        ``percept.gate.classify`` — which dereferences ``.confidence``/``.satisfied`` with no guard
        of its own — un-validated. Both failure shapes are normalized to :class:`_JudgeDegraded` here
        and caught by every caller of :meth:`_judge` (``perceive_judged``, ``_observe_and_log``),
        which degrade THAT TICK to ASK/SILENT (never FIRE, never a crash) and let the watch continue
        to the next frame. A well-formed ``Verdict`` is returned completely unchanged — this wrapper
        is a byte-for-byte no-op on the happy path.
        """
        try:
            verdict = await self._timed_judge(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - ANY judge-call failure fails closed, not just the
            # two typed modes (timeout/provider_error) fake_judge injects — a real backend can raise
            # its own transport exception too, and this boundary must catch all of them alike.
            raise _JudgeDegraded(exc) from exc
        if not _is_valid_verdict(verdict):
            raise _JudgeDegraded(verdict)
        return verdict

    def _note_judge_degraded(self, key: Hashable, cause: Any) -> None:
        """Record a degraded judge tick for ``key`` (the SAME ``(goal_id, entity_id)`` key
        the gate/executor already use) and escalate ONCE if its consecutive-degrade streak crosses
        :data:`CONSECUTIVE_DEGRADE_ESCALATE`. Only QUEUES the health-log entries --
        ``percept/runtime/worker.py``'s ``run()`` loop drains :attr:`_judge_health_log` every tick
        into its own ``on_health(RuntimeHealthEvent)`` channel (see that module); ``Percept`` never
        calls ``on_health`` itself."""
        streak = self._judge_degrade_streak.get(key, 0) + 1
        self._judge_degrade_streak[key] = streak
        self._judge_health_log.append(("judge_degraded", key, cause))
        if streak == CONSECUTIVE_DEGRADE_ESCALATE:  # exactly-once at the crossing, never re-fired
            self._judge_health_log.append(("judge_persistent_failure", key, cause))

    def _note_gate_suppressed(self, decision: Any) -> None:
        """Record one GATE SUPPRESSION (a :class:`~percept.gate.GateOutcome` whose reason is
        in :data:`~percept.gate.SUPPRESSION_REASONS`) for the runtime to surface.

        Only QUEUES -- ``percept/runtime/worker.py``'s ``run()`` loop drains :attr:`_gate_health_log`
        every tick into its ``on_health(RuntimeHealthEvent("gate_suppressed"))`` channel, the same
        contract :meth:`_note_judge_degraded` uses. Bounded: a core driven without that loop never
        drains, so the oldest entries are shed rather than accumulating forever on a 24/7 watch (the
        gate already logged each one structurally, so shedding loses no ground truth)."""
        self._gate_health_log.append(
            (decision.key, decision.reason, decision.at)
        )
        if len(self._gate_health_log) > _GATE_HEALTH_LOG_MAX:
            del self._gate_health_log[: len(self._gate_health_log) - _GATE_HEALTH_LOG_MAX]

    def _note_judge_ok(self, key: Hashable) -> None:
        """A well-formed verdict for ``key`` resets its consecutive-degrade streak -- ANY
        successful judge call (regardless of the verdict's confidence/truth) proves the judge is
        reachable again."""
        self._judge_degrade_streak.pop(key, None)

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

        The public temporal-judging seam: a thin, ergonomic wrapper over the existing
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
        process_mode = bool(
            context
            and (context.get("process_qa") or context.get("observe_step") or context.get(EXTRACTOR_CONTEXT_KEY))
        )
        # PR 1.5: a per-goal reference image ("is this THE mug <exemplar>?") — deictic/individuation
        # grounding the backends already implement but the watch path never supplied. Stable per goal
        # (set in goal.state), so it rides the goal's cache scope; None (the default) is byte-for-byte.
        exemplar = goal.state.get("exemplar")
        if frames or audio or process_mode:
            # The strip/audio (escalation) path is rare-by-design and multi-frame; bypass the
            # single-frame cache (its key dHashes ONE frame) so a strip verdict never pollutes — nor is
            # served from — a single-frame slot. The cache stays single-frame-pure. A process-QA judge
            # ALSO bypasses (its running answer/step accumulates per-call), and so does an
            # extractor-enriched judge (context["extractor"] is per-call, computed off THIS frame —
            # must never be cached/served under a different frame's slot). ``context`` is passed ONLY
            # in process_mode so a plain strip judge with neither is byte-for-byte.
            extra = {"context": context} if process_mode else {}
            # The strip/audio/process-QA judge is on the live watch path (perceive_judged /
            # _observe_and_log) — a malformed/timeout/partial response here must degrade the tick,
            # never crash it, so this goes through the fail-closed boundary (may raise
            # _JudgeDegraded, caught by every caller of _judge).
            return await self._safe_timed_judge(
                goal.condition, frame, roi=roi, exemplar=exemplar, corrections=corrections or None,
                frames=frames, audio=audio, **extra,
            )
        # A structured key so the FrameHashLayer can ROI-crop + dHash the frame and scope its
        # entry to (goal, entity); the legacy stub cascade ignores the extra fields harmlessly.
        cache_key = CacheKey(
            goal_id=goal.id, frame=frame, roi=roi, entity_id=entity_id, condition=goal.condition
        )
        hit = self.cache.resolve(cache_key)
        if hit.status == "HIT" and hit.verdict is not None:
            self.metrics.incr("cache.hit")
            return await self._audit_or_use(goal, frame, roi, corrections, cache_key, hit.verdict)
        # Same fail-closed boundary for the cache-miss (single-frame) judge — the other half of
        # the live watch path. A well-formed Verdict is cached exactly as before; a degraded response
        # raises _JudgeDegraded BEFORE `cache.store` is ever reached, so a failure is never cached.
        verdict = await self._safe_timed_judge(
            goal.condition, frame, roi=roi, exemplar=exemplar, corrections=corrections or None
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
        """Trust-but-verify a cache HIT.

        On the audited fraction (decided by ``self.audit`` — deterministic, no RNG on the L1
        path), re-perceive through real vision and compare to the cached verdict. A divergence in
        the ``satisfied`` bit is a **false-serve**: the fresh read wins and replaces the stale
        cache entry. Un-audited HITs are served as-is. ``audit is None`` -> never audit.
        """
        if self.audit is None or not self.audit(cache_key):
            return cached
        self.metrics.incr("cache.audit")
        # The audit re-perceive is ALSO a live judge call on the watch path — guard it the same
        # way. A degraded audit response must not crash the tick (which already has a perfectly good
        # cached verdict in hand); fall back to serving `cached` unchanged, exactly as an un-audited
        # HIT would. The false-serve upgrade only ever happens off a WELL-FORMED fresh verdict.
        try:
            fresh = await self._safe_timed_judge(
                goal.condition, frame, roi=roi, exemplar=goal.state.get("exemplar"),
                corrections=corrections or None,
            )
        except _JudgeDegraded:
            self.metrics.incr("judge.degraded")
            return cached
        if fresh.satisfied != cached.satisfied:
            self.metrics.incr("cache.false_serve")  # the cache's analog of false-fire
            self.cache.store(cache_key, fresh, fresh.confidence)  # fresh read wins
            return fresh
        return cached

    def _freshness(self, focus, now: float, goal_id: str | None = None) -> float:
        """Freshness in [0,1]: 1.0 = supported/current, 0.0 = stale (-> ASK, invariant 2).

        Two independent staleness signals, combined as the STRICTER (min) — either going to 0.0 drives
        freshness below ``fresh_min`` -> ASK, matching invariant 2's OR ("low confidence OR stale -> ASK"):

        * ASSERTED-fact age (always on): a user-asserted attribute older than ``validity_window_s`` is stale.
        * PERCEPTION recency (A-3b, opt-in via ``gate.stale_ask``): a goal not confidently read within
          ``validity_window_s`` is stale. ``last_supported[goal_id]`` is stamped on every confident read;
          because ``classify`` runs BEFORE that stamp, a goal with NO prior read (missing entry) is treated
          as fresh — the current judgment IS the fresh perception, so a first-ever read must not ASK.

        With ``stale_ask`` OFF (default) the perception term is never consulted and this is byte-for-byte
        the asserted-only path.
        """
        asserted = 1.0
        if focus is not None:
            confirmed = [a.last_confirmed for a in focus.attributes.values() if a.source == "asserted"]
            if confirmed:
                age = now - max(confirmed)
                asserted = 1.0 if age < self.settings.gate.validity_window_s else 0.0
        if not self.settings.gate.stale_ask or goal_id is None:
            return asserted  # OFF: byte-for-byte the asserted-fact freshness
        perception = 1.0
        last = self.last_supported.get(goal_id)
        if last is not None:
            age = now - last
            perception = 1.0 if age < self.settings.gate.validity_window_s else 0.0
        return min(asserted, perception)

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

    def _on_entity_evicted(self, entity_id: str) -> None:
        """Eviction hook: an entity died (graph cap) -> drop its gate keys and its cached verdicts, so no
        orphan state survives it. Ties ``FrameHashLayer.invalidate_entity`` to entity DEATH (it previously
        fired only on assertion-override), closing the adjacent-audit gap."""
        self.gate.reset_entity(entity_id)
        self.invalidate_cache_entity(entity_id)

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
        # FATAL seam: guard HERE, at the language↔conductor crossing, not at the wiring points.
        # A spoken turn that reaches a SCRIPTED brain is silently discarded (FakeConductor -> route="ignore").
        # session.ask() -> on_speech_endpoint bypasses run()/start(), so guarding only the wiring points
        # left a door open; guarding the crossing covers every door with one check. (Router-
        # handled turns above never reach here, so a turn the conductor was never needed for isn't refused.)
        _guard_live_language_path(self)
        self.metrics.incr("route.defer")
        context = self._turn_context()
        decision = await self.conductor.run("speech", utterance, context)
        reply = await self._apply_decision(decision, utterance, frame=frame)
        return TurnResult(route=decision.route, source="conductor", reply=reply)

    async def arm_from_language(self, command: str, *, t: float | None = None) -> Decision:
        """Author + arm a standing watch from RAW LANGUAGE, WITHOUT executing a spoken reply.

        The explicit ``watch``-verb path, distinct from :meth:`on_speech_endpoint` (which also *answers*
        the turn by streaming the specialist). A realtime wire has already said "on it", so the conductor's
        only job here is to author the watchspec; streaming the specialist's frame answer back would poison
        the held voice session and add seconds of latency. Returns the conductor's :class:`Decision` — its
        ``say`` is the confirmation, its ``goal_ops`` are what got armed.
        """
        now = self.clock() if t is None else t
        self.note_mention(command, t=now)
        _guard_live_language_path(self)  # FATAL seam: raw language -> a scripted brain authors nothing
        decision = await self.conductor.run("speech", command, self._turn_context())
        if decision.goal_ops:
            self.apply_goal_ops(decision.goal_ops)
            self.metrics.incr("goals.authored")
        return decision

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
        """Record every fire/ask to the action log (the audit seed and feedback corpus)."""
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
        """Perform a computer-use action — **flag-gated, scoped, logged**.

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
            if not modality_ok:
                why = "goal modality is not 'act'"
            elif getattr(self.consent, "actuation", "none") == "physical":
                # ADR-007: an OUT-of-envelope declaration, not a permission miss. Say so,
                # so a client debugging at 2am reads the refusal instead of hunting an allowlist.
                from .envelope import OUT_OF_ENVELOPE  # noqa: PLC0415 — local, avoids import cycle

                why = (
                    "ConsentProfile.actuation='physical' — "
                    + OUT_OF_ENVELOPE["robotics_control_loop"].sentence()
                )
            else:
                why = (
                    f"action {action!r} not permitted "
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
        """A world-changing action under the full safety contract.

        Goes through :meth:`act` (consent + per-source + modality enforced; a blocked attempt still
        raises :class:`ConsentDenied`), and — for a concern whose false-fire cost is high enough to
        REQUIRE VERIFICATION — instantiates a **verification concern** that confirms the action's
        effect actually landed. "Every world-changing action instantiates a verification concern"
        is enforced HERE, not left to the caller. OSS stays notify-first: actuate performs
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
        """Acting creates a concern that checks the effect LANDED — reusing the deadline/absence-firing
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
        # ``t`` is this agent's clock reading (actuate's ``now``); the worker stamps frames on that SAME
        # clock (wall for live capture), so this absolute deadline is comparable against ``check_deadlines``
        # -> the verification concern can actually fire (A-1: previously a monotonic stamp vs a stream t).
        v.state["deadline_at"] = t + verify_within
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
        """A JSON-able snapshot of durable state: goals, entities, the action log, and (v2) the gate's
        armed edge state + deadlines rebased into EPOCH time.

        Absolute deadlines live on the monotonic clock, which resets on restart — a raw monotonic value
        is meaningless after a reboot. Each ``deadline_at`` is converted to a wall-clock (``time.time``)
        instant here and rebased back into the fresh monotonic frame in :meth:`restore`. The
        ``clock_anchor`` records the ``(epoch, monotonic)`` pair the conversion was taken against.

        DELIBERATELY ABSENT: ``episodes`` (the per-goal :class:`~percept.episodes.EpisodeLog` map). Its
        omission is the ratified v1.4 policy (SESSION-ONLY, ADR-006), not an oversight — an
        :class:`EpisodeLog` is process-lifetime state with no rebasable duration base (see
        ``percept.episodes``). ``test_z2_episode_persistence`` asserts the key stays out of this dict, so
        a future accidental add TRIPS a test and forces the policy to be revisited deliberately rather
        than flipped by accident. The forward seam that will one day carry it is
        :meth:`Episode.to_dict` / :meth:`Episode.from_dict` — present but intentionally NOT wired here."""
        now_mono = self.clock()
        now_epoch = time.time()
        goals = self.goals.snapshot()
        for item in goals:
            st = item.get("goal", {}).get("state", {})
            da = st.get("deadline_at")
            if da is not None:
                st["deadline_at_epoch"] = now_epoch + (da - now_mono)  # monotonic -> epoch (rebasable)
                del st["deadline_at"]
        return {
            "version": 2,
            "clock_anchor": [now_epoch, now_mono],
            "goals": goals,
            "gate": self.gate.snapshot(),
            "entities": self.graph.snapshot(),
            "action_log": [asdict(r) for r in self.action_log.entries()],
        }

    def restore(self, snap: dict) -> "Percept":
        """Rehydrate goals / entities / action-log / gate from a :meth:`snapshot`, rebasing epoch
        deadlines into THIS process's monotonic frame.

        v1 back-compat is a TRAP: a v1 goal carries an old absolute ``deadline_s`` with no recoverable
        duration base, so it cannot be rebased. Accepting it silently would leave the concern with no
        ``deadline_at`` and it would never fire again — the exact silent-inertness this pass exists to
        kill. So a legacy ``deadline_s`` is handled LOUDLY: a warning + metric, then re-resolved as a
        duration-from-restore-now so the concern still fires."""
        # v1 vs v2 is detected structurally (a v2 goal carries deadline_at_epoch; a v1 goal a legacy
        # deadline_s), so a missing/renumbered version field can never silently mis-route a deadline.
        self.goals.restore(snap.get("goals", []))
        self.graph.restore(snap.get("entities", []))
        self.action_log = ActionLog()
        for rd in snap.get("action_log", []):
            self.action_log.record(ActionRecord(**rd))

        now_mono = self.clock()
        now_epoch = time.time()
        for goal in self.goals.all():
            st = goal.state
            if st.get("rearm") != "deadline":
                continue
            if "deadline_at_epoch" in st:  # v2: rebase epoch -> the fresh monotonic frame
                remaining = st.pop("deadline_at_epoch") - now_epoch
                st["deadline_at"] = now_mono + remaining
            elif "deadline_at" not in st and "deadline_after_s" not in st and "deadline_s" in st:
                legacy = st.pop("deadline_s")  # v1 legacy absolute — LOUD, never silent
                try:
                    st["deadline_at"] = now_mono + float(legacy)
                except (TypeError, ValueError):
                    st["deadline_at"] = now_mono
                self.metrics.incr("snapshot.deadline_unmigratable")
                warnings.warn(
                    f"restore(): goal {goal.id!r} carried a legacy absolute 'deadline_s' from a v1 "
                    f"snapshot that cannot be rebased (no duration base). Re-resolved as a duration from "
                    f"restore-time so it still fires — verify the new deadline. Re-author with "
                    f"'deadline_after_s' to remove this warning.",
                    stacklevel=2,
                )

        # Episode memory is SESSION-ONLY (ADR-006) — snapshot() never carried the
        # ``episodes`` map, so every counting goal we just rehydrated will restart at zero episodes and a
        # subject last seen before the restart will open a NEW episode afterwards. That is a decision, not
        # a bug, but it is a failure mode when it is SILENT: an operator watching a counting goal
        # under-counts after a redeploy with nothing saying why. Queue a single loud ``episode_memory_reset``
        # health event whenever we restored active counting goals into an empty episode map, so the amnesia
        # is observable. Queue-only — worker.py drains it into on_health (Percept never calls on_health).
        counting = [g for g in self.goals.all() if g.state.get("type") == "count"]
        if counting and not self.episodes:
            self._restore_health_log.append(
                (
                    "episode_memory_reset",
                    {
                        "reason": "session_only_v1.4",
                        "counting_goals": len(counting),
                        "goal_ids": [g.id for g in counting],
                        "detail": (
                            "episodes are session-only (ADR-006): counting/episode memory was "
                            "NOT restored; these goals restart at zero and may under-count until re-observed."
                        ),
                    },
                )
            )

        self.gate.restore(snap.get("gate"))
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

        payload = health(__version__, backends, self.metrics)
        # The single scripted-state payload — one vocabulary, asked not guessed. ``mode`` is the
        # resolved switch ("offline"/"live"); ``scripted`` lists the seams whose CONSTRUCTED backend is a
        # scripted double (computed at create() from the class, the fact — not the requested name).
        payload["mode"] = self._mode
        payload["scripted"] = list(self._scripted)
        return payload


__all__ = ["Percept", "FireEvent", "guard_live_language_path"]
