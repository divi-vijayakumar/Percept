"""LIVE — the general behavior-program executor: THE single firing path of the whole SDK.

Every fire in the runtime routes through :class:`BehaviorExecutor` here. ``agent.py`` imports
``BehaviorExecutor`` + ``program_of`` and, via its lazily-bound ``self.executor``, dispatches EVERY
concern through one of the four fire methods below:

  * ``_fire_concern``      -> :meth:`BehaviorExecutor.fire_judged`   (agent.py:378)
  * ``_fire_count`` / rep  -> :meth:`BehaviorExecutor.fire_episode`  (agent.py:397, 413)
  * absence coverage       -> :meth:`BehaviorExecutor.fire_absence`  (agent.py:532)
  * state transition       -> :meth:`BehaviorExecutor.fire_transition` (agent.py:736)

It began as a Phase-0 spike but is now wired and load-bearing: ONE executor
+ a general key-synthesis reproduces the legacy firing behaviors (judge / transition / episode /
absence + the on_false / never / each_event / interval / deadline re-arms) byte-for-byte, while
honoring the three review constraints:

  1. STATE ISOLATION — stateful primitives (transition's seen_from/fired) keep cross-tick memory in a
     per-concern :class:`ConcernContext` owned HERE, keyed by concern id, NEVER in ``gate._armed``. The
     gate stays a pure ``key -> armed bit``.
  2. ESCALATION SUBSTRATE — a per-concern :class:`ConfusionSignal` records the quantitative "am I
     confused" signal (rolling confidence + consecutive-unknown) for a later Posture-A->B policy. It is
     ADDITIVE METRIC ONLY and must never alter the firing trace (so the equivalence proof stays valid).
  3. NO eval() — the accumulator's ``fire_when`` micro-expression is parsed with ``ast.parse`` and
     evaluated by a hand-written recursive walker over a strict whitelist. ``eval``/``exec``/``compile``
     are never called on it — the prompt-injection firewall on LLM-authored expressions.

It imports ``gate.classify``/``Gate`` but NEVER edits gate.py — the gate stays a pure ``key -> armed
bit`` and this executor is the only thing that drives it. ``test_program_equivalence`` pins the
byte-for-byte equivalence with the legacy per-sense firing paths it replaced.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Callable, Hashable, Literal

from .adapters import Verdict
from .behavior_routes import is_known_behavior_type
from .concern import consequence_of, is_hard_quadrant
from .contracts import (
    Box,
    EvidenceModality,
    JudgmentEvent,
    JudgmentProvenance,
    JudgmentTimingBuilder,
    SubjectRef,
)
from .episodes import EPISODE_POLICIES, ContinuationDecision
from .duration_routes import duration_policy
from .gate import ASK, FIRE, Gate, GateOutcome, classify
from .observability import get_logger

# --------------------------------------------------------------------------------------------------
# The behavior program — authored data (the LLM fills these; presets compile legacy goal.state).
# --------------------------------------------------------------------------------------------------

Sense = Literal["judge", "transition", "episode", "absence", "rhythm"]
KeyBy = Literal["entity", "episode", "time_bucket", "const"]

# The perception axis — HOW the eye looks (the VLM-side half of the concern program). Every default
# reproduces today's behavior byte-for-byte: one vision-only, focus-cropped frame.
Modality = Literal["vision", "audio", "fused"]
#: ``event_capsule`` (ADDITIVE) is a COMPOSITE evidence shape distinct from a plain strip/window:
#: it assembles (a) pre-roll frames from the existing rolling ring, (b) the episode's own boundary
#: frames (open + most-recent), and (c) an early snapshot a timer captures a fixed N seconds after the
#: episode opened — so a long-running episode still carries early evidence even after the ring has
#: aged past it. Assembly lives in :mod:`percept.capsule` (:func:`~percept.capsule.assemble_event_capsule`);
#: this token is additive-only — no profile defaults to it, no existing value's behavior changes.
#: NOT (yet) IN :data:`percept.watchprofile.PLAN_MODES` — a ``WatchProfile.plan_mode`` cannot select
#: it. ``agent._PLAN_RANK`` (the plan-mode FLOOR every concern's spec is raised to) DOES carry a rung
#: for it, ranked above ``window`` as the superset shape it is — so an authored ``event_capsule``
#: spec is never silently floored/downgraded by an active profile plan_mode, while the reverse (a
#: profile upgrading a cheaper spec UP TO capsule) stays refused until a profile can default to it.
Temporal = Literal["single", "strip", "window", "sprite", "event_capsule"]
RoiMode = Literal["focus", "hint", "full"]
Sampling = Literal["recent", "uniform"]


@dataclass(frozen=True)
class KeySpec:
    """How the re-arm boundary folds into the gate key — the ONE general re-arm mechanism."""

    by: tuple[KeyBy, ...] = ("entity",)
    bucket_s: float | None = None  # time_bucket: fold floor(t/bucket_s) -> a new bucket is a fresh edge
    suppress_false: bool = False  # the never-contract: a confident-false never reaches the gate


@dataclass(frozen=True)
class Accumulator:
    """Optional fold over the event stream; ``fire_when`` gates the gate (count / nth-distinct)."""

    kind: Literal["none", "count_distinct"] = "none"
    window_s: float | None = None
    fire_when: str | None = None  # guarded micro-expr over {n, t}; None => fire each new episode


@dataclass(frozen=True)
class PerceptionSpec:
    """HOW the eye looks for a concern — the VLM-side twin of the behavior program. The brain authors
    it (or a sense-preset supplies it); the agent assembles the matching input and the backend makes
    ONE multimodal call. Default == today: a single vision-only focus-cropped frame, so an
    un-authored concern is byte-for-byte unchanged."""

    modality: Modality = "vision"
    temporal: Temporal = "single"
    n_frames: int = 1  # strip/window width; clamped at authoring time (_MAX_STRIP)
    roi: RoiMode = "focus"  # focus = in-focus entity box (today); hint = authored region; full = whole frame
    sampling: Sampling = "recent"  # which frames from the window: most-recent vs evenly-spaced
    audio_s: float | None = None  # audio clip seconds for the audio/fused modality
    window_s: float | None = None  # time length of the temporal clip (seconds); bounds strip assembly


@dataclass(frozen=True)
class BehaviorProgram:
    """The general behavior a concern runs — sense · key · accumulate · (deadline/cadence) · perception."""

    sense: Sense = "judge"
    predicate: str = ""
    from_predicate: str | None = None
    to_predicate: str | None = None
    key: KeySpec = field(default_factory=KeySpec)
    accumulate: Accumulator = field(default_factory=Accumulator)
    # An ABSOLUTE deadline stamp on the SAME clock the executor compares against (``now``). Resolved at
    # ARM time from the authored ``deadline_after_s`` duration (agent._arm_deadline) — never a raw duration,
    # never a cross-timeline value. (One field, one meaning, one clock.)
    deadline_at: float | None = None
    cadence_s: float | None = None
    hold_s: float | None = None  # R-1 sustained-duration: fire only after the condition holds this long
    perception: PerceptionSpec = field(default_factory=PerceptionSpec)  # HOW to look (defaults = today)


# --------------------------------------------------------------------------------------------------
# Constraint 3 — fire_when is AST-whitelisted and interpreted by hand. eval() is NEVER used.
# --------------------------------------------------------------------------------------------------

_ALLOWED_NAMES = {"n", "t"}
_CMP: dict[type, Callable[[float, float], bool]] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
}


class UnsafeExpression(ValueError):
    """Raised when a ``fire_when`` expression contains a node/name outside the whitelist."""


def _eval_node(node: ast.AST, env: dict):
    """Interpret a whitelisted node. No eval/exec/compile — a hand walker, so nothing escapes."""
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            fn = _CMP.get(type(op))
            if fn is None:
                raise UnsafeExpression(f"disallowed comparator {type(op).__name__}")
            right = _eval_node(comparator, env)
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise UnsafeExpression(f"disallowed name {node.id!r}")
        return env.get(node.id)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise UnsafeExpression(f"non-numeric constant {node.value!r}")
        return node.value
    raise UnsafeExpression(f"disallowed node {type(node).__name__}")


def compile_fire_when(expr: str) -> Callable[[dict], bool]:
    """Parse a ``fire_when`` micro-expression into a safe predicate ``(env) -> bool``.

    Grammar (strict whitelist): boolean ``and``/``or`` of comparisons (``== != > >= < <=``) over the
    names ``{n, t}`` and numeric constants. Anything else — a call, attribute, subscript, import, or a
    name outside ``{n, t}`` — raises :class:`UnsafeExpression`. The tree is INTERPRETED by
    :func:`_eval_node`; ``eval``/``exec``/``compile`` are never invoked on it.
    """
    tree = ast.parse(expr, mode="eval")  # parse only — does not execute
    # Validate the whole tree up-front so a bad node is rejected even on a short-circuited branch.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Load)):
            continue
        if isinstance(node, tuple(_CMP)):
            continue
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                raise UnsafeExpression(f"disallowed name {node.id!r} in {expr!r}")
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise UnsafeExpression(f"non-numeric constant in {expr!r}")
            continue
        raise UnsafeExpression(f"disallowed node {type(node).__name__} in {expr!r}")
    return lambda env: bool(_eval_node(tree.body, env))


# --------------------------------------------------------------------------------------------------
# Constraint 2 — the confusion substrate (additive; never affects the firing trace).
# --------------------------------------------------------------------------------------------------


@dataclass
class ConfusionSignal:
    """Per-concern substrate for Posture-A->B escalation: a quantitative "am I confused".

    A run of low-confidence / ``unknown`` verdicts (clustered ~0.5) is the signal. This is recorded as
    a SIDE METRIC only — it is read by a future escalation policy, never by the firing decision.
    """

    recent_conf: list[float] = field(default_factory=list)
    consecutive_unknown: int = 0
    cap: int = 16

    def record(self, verdict: Verdict, truth: str) -> None:
        self.recent_conf.append(verdict.confidence)
        if len(self.recent_conf) > self.cap:
            del self.recent_conf[: len(self.recent_conf) - self.cap]
        self.consecutive_unknown = self.consecutive_unknown + 1 if truth == "unknown" else 0

    @property
    def confidence_variance(self) -> float:
        xs = self.recent_conf
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    def confused(self, *, min_unknown: int = 3) -> bool:
        return self.consecutive_unknown >= min_unknown


@dataclass
class ConcernContext:
    """Constraint 1 — ISOLATED per-concern runtime memory for stateful primitives. Never the gate."""

    transitions: dict[Hashable, dict] = field(default_factory=dict)  # focus_id -> {seen_from, fired}
    count: int = 0
    confusion: ConfusionSignal = field(default_factory=ConfusionSignal)
    last_escalated_at: float = float("-inf")  # Posture-B deliberative (LLM) refractory (escalation.py)
    perception_level: int = 0  # escalation ladder rung: 0=single · 1=strip · 2=+audio (cheaper than the LLM)
    last_perception_escalate_at: float = float("-inf")  # the perception-rung refractory (shorter than LLM's)
    running_answer: str = ""  # Item 2: per-concern incremental process-QA answer (gate-irrelevant)
    # Item 4 (log -> summarize -> wipe): the per-concern process LOG of cheap per-clip step deltas, plus
    # the time the last step was logged (sustained absence of new steps = the episode boundary). The VLM
    # is NEVER shown this log; a single summarize pass consolidates it on boundary/on-demand, then wipes.
    process_log: list[str] = field(default_factory=list)
    last_step_at: float = float("-inf")
    # The re-arm discriminators that used to be folded into the gate key (making every time
    # bucket / episode number / resolved-answer a permanent key). They live HERE now (constraint 1: the
    # gate stays one bit per key); the executor calls ``gate.rearm(key)`` when one rolls over, restoring
    # the exact firing edge with ONE key per (goal, entity) instead of one per bucket/rep/answer.
    last_bucket: int | None = None       # interval: the elapsed time bucket last seen (int(t // cadence_s))
    last_episode_n: int | None = None    # episode: the last distinct rep number fired
    # The IDENTITY of the last episode fired. The count ``n`` is a weak discriminator — a
    # windowed prune or a max_episodes drop can leave n unchanged across a genuinely different
    # episode, and two different episodes then share one spent edge. When the caller threads the
    # episode id through ``fire_episode(..., episode_id=...)`` the re-arm keys off identity instead;
    # absent it, the ``n`` discriminator is used exactly as before (byte-for-byte).
    last_episode_id: str | None = None
    last_resolved_answer: str | None = None  # process-QA: the last resolved running-answer text


@dataclass
class Fired:
    """Executor firing result — shape-compatible with ``agent.FireEvent`` (action/text/entity_id)."""

    goal_id: str
    action: str
    text: str
    entity_id: object
    verdict: object
    # The frozen ADR-001 §1 record of THIS decision, when the caller threaded the funnel's
    # stamps in (``fire_judged(..., timing=...)``). Appended LAST with a ``None`` default so every
    # existing positional ``Fired(...)`` construction, and every consumer that ignores it, is
    # byte-for-byte unchanged. ``None`` means the caller did not carry judgment timing: for example
    # episode/rhythm/boundary/absence paths, a bare executor test, or a legacy direct
    # ``transition_decide`` call without ``timing``. The live ``Percept._perceive_transition`` caller
    # supplies timing and evidence sequencing, so its fired transition carries a JudgmentEvent marked
    # ``evidence_origin="transition"``. A missing event is NEVER replaced with fabricated timing.
    judgment: JudgmentEvent | None = None
    # Enriched judgment envelope (reason/modality/roi/provenance) + the provisional/superseding
    # link (revision_id/supersedes). All OPTIONAL, defaulted None, appended last: every existing
    # positional Fired(...) and every consumer that ignores them is byte-for-byte unchanged.
    reason: str | None = None
    modality: "EvidenceModality | None" = None
    roi: "Box | None" = None
    provenance: "JudgmentProvenance | None" = None
    revision_id: str | None = None
    supersedes: str | None = None


@dataclass
class TransitionTick:
    """One A→B transition tick's result: the fire (if any) PLUS the facts the agent's
    ``transition.decision`` trace needs, so the agent never reaches into the executor's isolated ctx.

    ``from_truth``/``to_truth`` are the :func:`~percept.gate.classify` outputs for this tick; ``armed``
    is the post-tick ``seen_from`` latch (the goal has witnessed its ``from`` state and is waiting for
    ``to``); ``fired_latch`` is the post-tick ``fired`` latch (this focus already fired and will not
    re-fire until re-armed); ``outcome`` is the typed WHY, one of ``fire`` / ``armed`` / ``rearmed`` /
    ``suppressed_latched`` / ``idle``. ``fired`` is the historical ``Fired | None`` the legacy
    :meth:`BehaviorExecutor.fire_transition` facade returns directly."""

    fired: Fired | None
    from_truth: str
    to_truth: str
    armed: bool
    fired_latch: bool
    outcome: str


# --------------------------------------------------------------------------------------------------
# EPISODE-OWNED IDEMPOTENCY. The gate bit must stop being accidental per-episode memory.
# --------------------------------------------------------------------------------------------------
#
# The A->B->A trace only worked because B happened to leave the single gate key in a fired state. That
# is FRAGILE: if B is judged a new episode and re-arms the shared key, then FAILS its dwell (or becomes
# ASK) and never fires, A's revisit is a continuation that never re-arms — but the gate key is no longer
# in A's fired state, so A can FIRE AGAIN. One subject's evaluation state was silently governing another
# subject's duplicate suppression. The gate is one bit PER KEY; "have I already reported THIS episode?"
# is a property of the EPISODE, and the two are not the same question.
#
# The episode owns idempotency. Before an observation is allowed to (re-)enter the gate, we fold
# it against its episode's emission history and get an :class:`EpisodeFoldResult`:
#
#   * a NEW episode with no emitted incident may keep evaluating (and re-arms its own edge, so a shared
#     key another subject left disarmed cannot suppress its first fire);
#   * once a revision has EMITTED, an UNCHANGED confirmation does NOT re-enter the gate at all
#     (``should_evaluate_gate=False``) — the episode remembers it already spoke, so the gate never has to;
#   * a semantically-accepted UPDATE creates a new revision and MAY re-arm (``supersedes`` names the
#     revision it replaces).
#
# The gate stays MECHANICALLY UNCHANGED (percept/gate.py): the fold only ever calls the existing
# ``gate.rearm(key)`` / ``gate.evaluate(...)``. It remains one bit per key — it just stops being used as
# per-episode memory it was never sized or scoped to be.
#
# INVARIANT (declared, and pinned by tests): if the gate key's TTL expires while the episode is still
# OPEN, an unchanged episode MUST NOT refire merely because gate memory disappeared. Idempotency lives
# on the episode, not on the evictable gate bit, so a memory bound can never leak into user-visible
# behaviour (eval/synthetic/idempotency/ key_ttl_expiry_while_episode_open_does_not_refire).

EpisodeMatch = Literal["new", "continuation", "unbound", "ambiguous", "unknown"]
#: How the observation binds to an episode. ``new`` — it opens a fresh episode; ``continuation`` — it
#: extends an existing open one; ``unbound`` — no subject identity could be established (fall back to
#: gate-only, the legacy behaviour); ``ambiguous`` — the subject resolved to more than one candidate,
#: so idempotency cannot be safely attributed and we also fall back to gate-only; ``unknown``
#: — a subject was SEEN BUT NOT IDENTIFIED (a visible abstention). Like ``unbound`` it degrades to
#: gate-only and NEVER rearms, but it is a DISTINCT observable state so a client can tell
#: "seen-but-unidentified" from "no subject at all". The absence of a subject is ``unbound``, not
#: ``unknown`` — the two are different debugging facts.

EpisodeMutation = Literal["confirmed", "updated", "retracted", "closed", "none"]
#: How this finding modifies the episode (the mutation channel, consumed here). ``confirmed`` — the
#: episode is re-stated unchanged; ``updated`` — an accepted semantic change → a new revision;
#: ``retracted`` — the report is withdrawn; ``closed`` — the episode is terminated; ``none`` — no
#: mutation semantics apply (a birth, or an unbound/ambiguous observation).

EPISODE_MATCHES: tuple[EpisodeMatch, ...] = (
    "new", "continuation", "unbound", "ambiguous", "unknown",
)
EPISODE_MUTATIONS: tuple[EpisodeMutation, ...] = (
    "confirmed", "updated", "retracted", "closed", "none",
)

#: The wiring: a Verdict's LOCAL speech-act :data:`~percept.contracts.Operation` → the
#: episode-fold's :data:`EpisodeMutation`. This is the ONE place the two vocabularies meet, so the fold
#: never has to know about speech acts and the claim channel never has to know about episodes.
#:
#: Only the mutating acts (``replace``/``retract``/``close``) mutate the episode:
#:   * ``replace`` → ``updated``   — an accepted semantic change → a new revision (may re-arm + fire).
#:   * ``retract`` → ``retracted`` — the report is withdrawn (re-arm so a genuine later fire counts).
#:   * ``close``   → ``closed``    — the episode is terminated (no further evaluation).
#: The non-mutating acts (``assert``/``confirm``/``mention``) all fold to ``confirmed`` — an unchanged
#: re-statement, which is exactly the idempotency case this protects. So a plain positive verdict
#: (default ``operation='assert'``) yields ``confirmed``, byte-for-byte the fold default from
#: before this mapping existed.
_OPERATION_TO_MUTATION: dict[str, EpisodeMutation] = {
    "assert": "confirmed",
    "confirm": "confirmed",
    "mention": "confirmed",
    "replace": "updated",
    "retract": "retracted",
    "close": "closed",
}


def episode_mutation_of(operation: str) -> EpisodeMutation:
    """Map a Verdict :data:`~percept.contracts.Operation` to the episode fold's :data:`EpisodeMutation`.

    An unknown/absent operation is treated as the safe non-mutating default ``confirmed`` (a
    re-statement) so a verdict that predates the claim channel folds exactly as it did before this
    mapping existed."""
    return _OPERATION_TO_MUTATION.get(operation, "confirmed")


def revision_id(episode_id: str, n: int) -> str:
    """The stable id of the ``n``-th incident an episode emitted (``<episode_id>#r<n>``). ``supersedes``
    on an update names the previously-emitted revision; the ledger mints the next one on emission."""
    return f"{episode_id}#r{n}"


@dataclass(frozen=True)
class EpisodeFoldResult:
    """The episode-owned idempotency decision for ONE observation.

    It answers, BEFORE the observation is allowed to touch the gate, three things the gate bit cannot
    answer on its own because it is one bit per key, not per episode:

    * :attr:`should_evaluate_gate` — may this observation (re-)enter the gate at all? ``False`` is the
      episode saying "I already reported this and nothing changed" — the fire path returns without ever
      consulting the gate, so neither a shared key nor an evicted key can manufacture a duplicate.
    * :attr:`should_rearm` — should the gate edge be re-armed first? ``True`` for a genuinely new
      episode (its first fire must not be suppressed by a key another subject left disarmed) and for an
      accepted update (a new revision is allowed to fire again).
    * :attr:`supersedes` — for an update, the :func:`revision_id` this new revision replaces.

    :attr:`match` and :attr:`mutation` are the typed WHY, from the closed :data:`EPISODE_MATCHES` /
    :data:`EPISODE_MUTATIONS` vocabularies. This is a pure record — :func:`episode_fold` computes it and
    :meth:`BehaviorExecutor.fire_judged` honours it; the shadow path recomputes it from the same
    function, so it must stay side-effect free.
    """

    episode_id: str | None
    match: EpisodeMatch
    mutation: EpisodeMutation
    should_rearm: bool
    should_evaluate_gate: bool
    supersedes: str | None = None

    def __post_init__(self) -> None:
        if self.match not in EPISODE_MATCHES:
            raise ValueError(f"unknown episode match {self.match!r}; expected one of {EPISODE_MATCHES}")
        if self.mutation not in EPISODE_MUTATIONS:
            raise ValueError(
                f"unknown episode mutation {self.mutation!r}; expected one of {EPISODE_MUTATIONS}"
            )


def episode_fold(
    decision: "ContinuationDecision | None",
    *,
    emitted: int = 0,
    mutation: EpisodeMutation = "confirmed",
    ambiguous: bool = False,
    unknown: bool = False,
) -> EpisodeFoldResult:
    """Fold one observation against its episode's emission history → an :class:`EpisodeFoldResult`.

    Pure and side-effect free (the shadow path calls it). Inputs:

    * ``decision`` — the typed :class:`~percept.episodes.ContinuationDecision` for this observation, or
      ``None`` when no subject could be bound (→ ``unbound``, gate-only fallback).
    * ``emitted`` — how many incidents THIS episode has already emitted (the episode-owned ledger;
      :class:`EpisodeIncidentLedger`). ``0`` means the episode has opened but never fired.
    * ``mutation`` — how this finding modifies the episode (the mutation channel). Defaults to ``confirmed``
      (a plain re-statement), which is the case idempotency is about.
    * ``ambiguous`` — the subject resolved to more than one candidate → ``ambiguous``, gate-only.
    * ``unknown`` — the subject was SEEN BUT NOT IDENTIFIED (``resolution='unknown'``,
      derived by :func:`percept.episodes.subject_is_unknown`). A visible abstention.

    Rules (made executable):

    * unbound / ambiguous → we cannot attribute idempotency to an episode, so degrade to the legacy
      gate-only path: evaluate the gate, never re-arm from here.
    * ``unknown`` → a visible abstention. Even though the episode log MINTED a fresh episode
      for it (two unknowns never collapse), the fold must NOT let that mint REARM the gate — an
      unknown re-arming an edge a bound subject already spoke on is exactly the wrong-suppression this
      rule forbids — and must NOT grant it episode-owned idempotency (two unknowns are never the same
      episode). So it degrades to gate-only like ``unbound``, but keeps its own observable ``unknown``
      match. The SOLE exception is a provisional single-focus continuation of an active BOUND episode
      (``decision.reason == 'unknown_provisional_continuation'``): that is a genuine continuation and
      folds as one below.
    * ``new`` → a fresh episode: evaluate AND re-arm (its own edge, immune to a shared/disarmed key).
    * ``continuation`` + ``confirmed``/``none``:
        - already emitted → do NOT re-enter the gate (idempotent; THE TTL-invariant case);
        - not yet emitted → keep evaluating (a new episode mid-dwell/after an ASK may still fire).
    * ``continuation`` + ``updated`` → a new revision MAY re-arm and fire; ``supersedes`` the prior one.
    * ``continuation`` + ``retracted`` → withdraw and re-arm, so a genuine later re-satisfaction fires.
    * ``continuation`` + ``closed`` → the episode is terminated; no further evaluation for it.
    """
    # A visible abstention degrades to gate-only and NEVER rearms — before every other rule,
    # because an unknown that minted a fresh episode would otherwise be read as ``new`` and
    # re-arm. The provisional single-focus continuation is the one unknown that is a real continuation;
    # it falls through to the continuation handling and folds idempotently like any other.
    if unknown and (
        decision is None or decision.reason != "unknown_provisional_continuation"
    ):
        return EpisodeFoldResult(
            episode_id=decision.episode.id if decision is not None else None,
            match="unknown", mutation="none",
            should_rearm=False, should_evaluate_gate=True, supersedes=None,
        )
    if decision is None:
        return EpisodeFoldResult(
            episode_id=None, match="unbound", mutation="none",
            should_rearm=False, should_evaluate_gate=True, supersedes=None,
        )
    episode_id = decision.episode.id
    if ambiguous:
        return EpisodeFoldResult(
            episode_id=episode_id, match="ambiguous", mutation="none",
            should_rearm=False, should_evaluate_gate=True, supersedes=None,
        )
    if decision.is_new:
        # A birth: nothing prior to mutate. Re-arm so its first fire is never suppressed by a key a
        # different subject (mapping to the same gate key) happened to leave disarmed.
        return EpisodeFoldResult(
            episode_id=episode_id, match="new", mutation="none",
            should_rearm=True, should_evaluate_gate=True, supersedes=None,
        )
    # continuation of an existing episode
    if mutation == "closed":
        return EpisodeFoldResult(
            episode_id=episode_id, match="continuation", mutation="closed",
            should_rearm=False, should_evaluate_gate=False, supersedes=None,
        )
    if mutation == "retracted":
        supersedes = revision_id(episode_id, emitted) if emitted else None
        return EpisodeFoldResult(
            episode_id=episode_id, match="continuation", mutation="retracted",
            should_rearm=True, should_evaluate_gate=True, supersedes=supersedes,
        )
    if mutation == "updated":
        # An accepted update is a new revision. It may re-arm and fire again — but only if the episode
        # had actually emitted something to supersede; before the first emission it is just more
        # evidence for the pending fire (no re-arm, nothing to supersede).
        if emitted:
            return EpisodeFoldResult(
                episode_id=episode_id, match="continuation", mutation="updated",
                should_rearm=True, should_evaluate_gate=True,
                supersedes=revision_id(episode_id, emitted),
            )
        return EpisodeFoldResult(
            episode_id=episode_id, match="continuation", mutation="updated",
            should_rearm=False, should_evaluate_gate=True, supersedes=None,
        )
    # confirmed / none — an unchanged re-statement
    if emitted:
        # THE idempotency case: the episode already reported and nothing changed. Do not re-enter the
        # gate — this is what makes a TTL eviction of the gate key a non-event (invariant above).
        return EpisodeFoldResult(
            episode_id=episode_id, match="continuation", mutation="confirmed",
            should_rearm=False, should_evaluate_gate=False, supersedes=None,
        )
    # Open but not yet emitted (e.g. still accumulating dwell, or the last read was an ASK): keep going.
    return EpisodeFoldResult(
        episode_id=episode_id, match="continuation", mutation="confirmed",
        should_rearm=False, should_evaluate_gate=True, supersedes=None,
    )


class EpisodeIncidentLedger:
    """The episode-owned record needed: how many incidents each episode has EMITTED.

    This is the memory the gate bit was being abused as. It is keyed by episode id (globally unique —
    ``<goal_id>:ep<n>``), so it never collides across subjects that share a gate key, and it is not
    evictable, so a gate-key TTL sweep cannot erase it. :meth:`emitted` feeds :func:`episode_fold`;
    :meth:`record` is called by the executor at the instant an incident is actually delivered and mints
    the new revision id.
    """

    def __init__(self) -> None:
        self._emitted: dict[str, int] = {}

    def emitted(self, episode_id: str) -> int:
        """How many incidents this episode has emitted (0 = opened but never fired)."""
        return self._emitted.get(episode_id, 0)

    def revision(self, episode_id: str) -> "str | None":
        """The id of the episode's most-recent emitted revision, or ``None`` if it never fired."""
        n = self._emitted.get(episode_id, 0)
        return revision_id(episode_id, n) if n else None

    def record(self, episode_id: str) -> str:
        """Record one emitted incident for ``episode_id`` and return the new revision id."""
        n = self._emitted.get(episode_id, 0) + 1
        self._emitted[episode_id] = n
        return revision_id(episode_id, n)


# --------------------------------------------------------------------------------------------------
# SHADOW MODE. Emit the subject + the EpisodeFoldResult the model WOULD have proposed, change NOTHING.
# It OBSERVES what the identity/episode model would do without letting any of it govern the gate,
# re-arm an edge, or suppress a fire. Everything here is a PURE, side-effect-free projection so it can
# be recomputed in shadow and serialized into the trace / onto an incident.
# --------------------------------------------------------------------------------------------------


def _subject_advisory(subject: "SubjectRef | None") -> "dict | None":
    """Serialize a resolved :class:`~percept.contracts.SubjectRef` to a JSON-safe advisory dict.

    ``None`` (the absence of a subject — NOT an ``unknown`` ref) round-trips as ``None``. Every field
    is a scalar so it survives a ``.ptrace`` round-trip with media OFF."""
    if subject is None:
        return None
    return {
        "scheme": subject.scheme,
        "namespace": subject.namespace,
        "key": subject.key,
        "resolution": subject.resolution,
        "scope": subject.scope,
        "strategy": subject.strategy,
        "confidence": subject.confidence,
    }


def _subject_from_advisory(payload: "dict | None") -> "SubjectRef | None":
    """Inverse of :func:`_subject_advisory` — reconstruct the ref from a loaded trace record."""
    if not payload:
        return None
    return SubjectRef(
        scheme=str(payload["scheme"]),
        namespace=str(payload.get("namespace") or ""),
        key=payload.get("key"),
        resolution=payload["resolution"],
        scope=payload.get("scope", "session"),
        strategy=payload.get("strategy", ""),
        confidence=payload.get("confidence"),
    )


def _fold_advisory(fold: "EpisodeFoldResult") -> dict:
    """Serialize an :class:`EpisodeFoldResult` to a JSON-safe advisory dict (all scalars)."""
    return {
        "episode_id": fold.episode_id,
        "match": fold.match,
        "mutation": fold.mutation,
        "should_rearm": fold.should_rearm,
        "should_evaluate_gate": fold.should_evaluate_gate,
        "supersedes": fold.supersedes,
    }


def _fold_from_advisory(payload: "dict") -> "EpisodeFoldResult":
    """Inverse of :func:`_fold_advisory` — reconstruct the fold from a loaded trace record."""
    return EpisodeFoldResult(
        episode_id=payload.get("episode_id"),
        match=payload["match"],
        mutation=payload["mutation"],
        should_rearm=bool(payload["should_rearm"]),
        should_evaluate_gate=bool(payload["should_evaluate_gate"]),
        supersedes=payload.get("supersedes"),
    )


@dataclass(frozen=True)
class ShadowObservation:
    """What the shadow path OBSERVES for one observation: the resolved subject + the proposed fold.

    ADVISORY ONLY. It is computed from the same pure :func:`episode_fold` the executor would honour,
    but nothing here re-arms, suppresses, or alters what fires — a consumer that ignores it sees
    byte-for-byte today's behaviour. Its whole job is to LAND IN THE TRACE (``shadow.fold`` stage)
    and survive a media-OFF ``.ptrace`` round-trip, so a client can grade what the fold proposed
    against their own ground truth.

    :meth:`to_advisory` / :meth:`from_advisory` are the round-trip pair the trace uses; ``subject``
    is ``None`` for the genuine absence of a subject (never an ``unknown`` ref with no key)."""

    subject: "SubjectRef | None"
    fold: "EpisodeFoldResult"

    def to_advisory(self) -> dict:
        """The JSON-safe advisory dict written into a trace record's ``data`` (and onto an incident)."""
        return {"subject": _subject_advisory(self.subject), "fold": _fold_advisory(self.fold)}

    @classmethod
    def from_advisory(cls, payload: "dict") -> "ShadowObservation":
        """Reconstruct a :class:`ShadowObservation` from a loaded trace record's advisory dict."""
        return cls(
            subject=_subject_from_advisory(payload.get("subject")),
            fold=_fold_from_advisory(payload["fold"]),
        )


def ask_text(goal) -> str:  # mirrors agent._ask_text
    return f'About "{goal.condition}" — can you confirm?'


def escalate_text(goal) -> str:  # mirrors agent._escalate_text
    return f"High stakes either way — confirm before I act: {goal.condition}?"


def overdue_text(goal) -> str:  # mirrors agent._overdue_text
    return f"Still waiting on: {goal.condition}"


# --------------------------------------------------------------------------------------------------
# The executor — ONE firing path; the gate is called exactly once per sense, unchanged.
# --------------------------------------------------------------------------------------------------


class _NoMetrics:
    """Default metrics sink — a no-op so the executor runs standalone (tests/spike)."""

    def incr(self, key: str) -> None:  # pragma: no cover - trivial
        pass


class BehaviorExecutor:
    """One executor for every concern. Produces ``(key, verdict)`` and calls the UNCHANGED gate."""

    def __init__(self, gate: Gate, metrics=None) -> None:
        self.gate = gate
        self.metrics = metrics if metrics is not None else _NoMetrics()
        self._ctx: dict[str, ConcernContext] = {}  # isolated per-concern state, keyed by concern id
        self._fire_when_cache: dict[str, Callable[[dict], bool]] = {}
        # Episode-owned idempotency memory (per episode id, NOT per gate key — see the module
        # section above). This is the state the gate bit was being accidentally used as.
        self._incidents = EpisodeIncidentLedger()

    def ctx(self, concern_id: str) -> ConcernContext:
        return self._ctx.setdefault(concern_id, ConcernContext())

    def fold_episode(
        self,
        decision: "ContinuationDecision | None",
        *,
        mutation: EpisodeMutation = "confirmed",
        operation: "str | None" = None,
        ambiguous: bool = False,
        unknown: bool = False,
    ) -> EpisodeFoldResult:
        """Fold an observation's :class:`~percept.episodes.ContinuationDecision` against this
        executor's episode-owned emission ledger → the :class:`EpisodeFoldResult` to thread into
        :meth:`fire_judged` as ``fold=``. Reads the ledger but never mutates it (the emission is only
        recorded when a fire is actually delivered), so the shadow path can call it safely.

        ``operation`` (OPTIONAL) is the Verdict's LOCAL speech-act. When supplied it DERIVES
        the episode ``mutation`` via :func:`episode_mutation_of` (overriding the ``mutation`` kwarg), so
        a caller threads ``verdict.operation`` straight through and the fold consumes the mutation
        channel without ever parsing a value. When ``None`` (default) the explicit ``mutation`` kwarg
        is used, so an existing caller is byte-for-byte unchanged.

        ``unknown`` (OPTIONAL) marks a subject SEEN BUT NOT IDENTIFIED — the caller derives it
        with :func:`percept.episodes.subject_is_unknown` on the verdict. When ``True`` the fold degrades
        to gate-only and NEVER rearms (a visible abstention must not manufacture wrong suppression); it
        does not consult the emission ledger, so it is safe on the transient episode minted for an
        unknown. Default ``False`` is byte-for-byte the path from before this flag existed."""
        if operation is not None:
            mutation = episode_mutation_of(operation)
        emitted = self._incidents.emitted(decision.episode.id) if decision is not None else 0
        return episode_fold(
            decision, emitted=emitted, mutation=mutation, ambiguous=ambiguous, unknown=unknown
        )

    def shadow_observe(
        self,
        decision: "ContinuationDecision | None",
        *,
        subject: "SubjectRef | None" = None,
        mutation: EpisodeMutation = "confirmed",
        operation: "str | None" = None,
        ambiguous: bool = False,
        unknown: bool = False,
    ) -> ShadowObservation:
        """OBSERVE what the fold WOULD propose for this observation — change nothing.

        Resolves to a :class:`ShadowObservation` (the ``subject`` + the proposed
        :class:`EpisodeFoldResult`) by calling the SAME pure :meth:`fold_episode`. It reads the
        emission ledger but never records into it and never touches the gate, so it is safe to run in
        pure shadow beside the live path. The resulting advisory is what the shadow path writes into the
        ``shadow.fold`` trace stage and attaches to an incident — governing nothing."""
        fold = self.fold_episode(
            decision, mutation=mutation, operation=operation, ambiguous=ambiguous, unknown=unknown
        )
        return ShadowObservation(subject=subject, fold=fold)

    def _fire_when(self, expr: str) -> Callable[[dict], bool]:
        pred = self._fire_when_cache.get(expr)
        if pred is None:
            pred = compile_fire_when(expr)
            self._fire_when_cache[expr] = pred
        return pred

    def _interval_rearm(self, ctx: "ConcernContext", key: tuple, spec: KeySpec, t: float) -> None:
        """An ``interval`` re-arm without leaking a key per bucket. The bucket number USED to be folded
        into the gate key (``(goal, entity, floor(t/cadence))``), so a 30-day watch minted 43,200 permanent
        keys. Instead the key is just ``(goal, entity)`` and the elapsed bucket is remembered in the
        per-concern ctx; when it rolls over we ``gate.rearm(key)`` — the SAME fresh edge a new bucket used
        to give, byte-for-byte (see test_interval_firing_trace_byte_for_byte). This is behavior.py's own
        stated invariant: stateful primitives keep cross-tick memory in ConcernContext, never gate._armed."""
        if not ("time_bucket" in spec.by and spec.bucket_s):
            return
        bucket = int(t // spec.bucket_s)
        if ctx.last_bucket is not None and bucket != ctx.last_bucket:
            self.gate.rearm(key)  # bucket rolled over -> fresh armed edge (was: a brand-new key)
        ctx.last_bucket = bucket

    # -- judge / event sense: replicates agent._fire_concern byte-for-byte ----------------------
    def fire_judged(
        self,
        goal,
        verdict: Verdict,
        base_key: tuple,
        program: BehaviorProgram,
        *,
        freshness: float = 1.0,
        t: float = 0.0,
        metric: str = "gate",
        timing: JudgmentTimingBuilder | None = None,
        evidence_seq: int = 0,
        fold: "EpisodeFoldResult | None" = None,
        outcome_sink: "Callable[[GateOutcome], None] | None" = None,
    ) -> Fired | None:
        """Run one judged concern through the (unchanged) gate and return the firing, if any.

        ``fold`` (OPTIONAL, additive) is the episode-owned idempotency decision for this
        observation (see :func:`episode_fold` / :meth:`fold_episode`). When supplied it governs the gate
        interaction WITHOUT changing gate mechanics: ``should_evaluate_gate=False`` returns before the
        gate is ever consulted (an already-emitted episode re-stated unchanged — so a shared or an
        evicted gate key can never manufacture a duplicate), and ``should_rearm=True`` re-arms the edge
        first (a genuinely new episode, or an accepted update). When ``fold is None`` (the default) this
        method is BYTE-FOR-BYTE the path from before this fold existed — the gate bit alone decides, exactly as before.

        ``timing`` (ADR-001 §1) is the OPTIONAL, additive judgment-timing accumulator the caller
        filled as this evidence moved through the funnel (``mark_ingest_observed`` →
        ``mark_admitted`` → ``mark_judge_started`` → ``mark_judge_completed``). When supplied, the
        decision produced here is ALSO frozen into a :class:`~percept.contracts.JudgmentEvent` and
        hung on the returned :class:`Fired` — this method is the runtime's ONE construction site for
        the contract's centre, so the lifecycle/immutability invariants govern the object the live
        path actually makes. It never influences whether or what fires: gate semantics, the
        never-contract, the accumulator and the hard-quadrant escalation are untouched, and a
        failure to build the event degrades to ``judgment=None`` rather than to a changed decision.

        ``outcome_sink`` is an internal observability seam: when supplied, it receives the exact
        typed outcome from this method's sole gate evaluation. It never causes a second evaluation.
        """
        ctx = self.ctx(goal.id)
        truth = classify(verdict, freshness, self.gate.threshold, self.gate.fresh_min)
        ctx.confusion.record(verdict, truth)  # constraint 2: additive, pre-gate, no trace effect
        if program.key.suppress_false and truth == "no":
            return None  # never-contract: a confident-false must not reach the gate
        key = base_key  # the key is ONLY (goal, entity) now — the interval bucket moved to ctx
        self._interval_rearm(ctx, key, program.key, t)
        # Episode-owned idempotency, applied BEFORE the gate. Absent (fold=None) this whole block is
        # skipped and the path is byte-for-byte the gate-only behaviour from before this existed.
        if fold is not None:
            if not fold.should_evaluate_gate:
                # The episode already reported and nothing changed — do not consult the gate at all, so
                # neither a shared key nor a TTL-evicted key can produce a duplicate incident.
                self.metrics.incr(f"{metric}.idempotent_suppressed")
                return None
            if fold.should_rearm:
                # A genuinely new episode (or an accepted update) — re-arm its own edge so a key another
                # subject left disarmed cannot suppress this episode's fire.
                self.gate.rearm(key)
        # Pass the goal's dwell (hold_s) + the frame time. hold_s=None -> the gate ignores `now`
        # and this is byte-for-byte the rising-edge path. A never-goal's key is marked no_evict so a TTL
        # drop can never re-arm it (the fire-once-ever contract; see Gate.evaluate).
        # Take the TYPED outcome (action + typed reason code) rather than the bare action. ``decide`` IS the
        # implementation of ``evaluate`` (``evaluate`` returns ``decide(...).action``), so the edge
        # effects and the action are byte-for-byte identical — we simply also read the ``reason`` the
        # gate already computed, which until now reached only the health/trace channel, never the wire.
        hold_s, max_support_gap_s = duration_policy(goal.state)
        outcome = self.gate.decide(
            key, verdict, freshness=freshness, hold_s=hold_s,
            max_support_gap_s=max_support_gap_s, now=t,
            no_evict=program.key.suppress_false,
        )
        if outcome_sink is not None:
            outcome_sink(outcome)
        action = outcome.action
        # The envelope the executor knows at fire time: the gate reason, how the eye looked, and
        # that a judged read is a DIRECT observation. ``roi`` stays None — the judge does not report the
        # region it fired on, and echoing the ``region_hint`` INPUT would misreport an input as an output.
        env = dict(
            reason=outcome.reason,
            modality=program.perception.modality,
            roi=None,
            provenance="observed",
        )
        self.metrics.incr(f"{metric}.{action}")
        if action == FIRE:
            # accumulate (nth-distinct): a rising edge IS a distinct occurrence, so gate the
            # USER-facing fire on fire_when(n). No fire_when -> byte-for-byte (existing goals never
            # accumulate). This is how "alert me the 3rd distinct time" composes with zero new code.
            if program.accumulate.fire_when:
                ctx.count += 1
                if not self._fire_when(program.accumulate.fire_when)({"n": ctx.count, "t": t}):
                    return None  # counted the occurrence, but it isn't the nth -> no alert
            # An incident is about to be DELIVERED — record it on the episode so the next unchanged
            # confirmation of the same episode is idempotently suppressed (an escalated ASK still counts:
            # the episode spoke to the user). Only reached on a genuine emission, never on a withheld one.
            #
            # Minting the revision here ALSO gives this emission its stable id, and the fold's
            # ``supersedes`` (set only when a later turn was replace/retract on the same episode) names
            # the prior revision this one corrects. Surfacing BOTH on the fire is what lets a superseding
            # revision reach the consumer as a LINKED correction — not a coequal, unlinked peer. On an
            # original ("provisional") fire ``supersedes`` stays ``None``. A superseding emission is
            # metered (``<metric>.superseding_revision``) so the correction is never silent even when a
            # downstream sink drops the link.
            revision = supersedes = None
            if fold is not None and fold.episode_id is not None:
                revision = self._incidents.record(fold.episode_id)
                supersedes = fold.supersedes
            if supersedes is not None:
                self.metrics.incr(f"{metric}.superseding_revision")
            if is_hard_quadrant(consequence_of(goal)):  # a hard-quadrant consequence escalates to ASK
                self.metrics.incr("consequence.escalate")
                return Fired(goal.id, ASK, escalate_text(goal), base_key[-1], verdict,
                             self._judgment(key, evidence_seq, ASK, timing, env),
                             revision_id=revision, supersedes=supersedes, **env)
            return Fired(goal.id, FIRE, goal.say, base_key[-1], verdict,
                         self._judgment(key, evidence_seq, FIRE, timing, env),
                         revision_id=revision, supersedes=supersedes, **env)
        if action == ASK:
            return Fired(goal.id, ASK, ask_text(goal), base_key[-1], verdict,
                         self._judgment(key, evidence_seq, ASK, timing, env), **env)
        return None

    # -- The ONE production construction site of the contract's centre ----------------------
    def _judgment(
        self, key: tuple, evidence_seq: int, action: str,
        timing: JudgmentTimingBuilder | None, env: dict | None = None,
    ) -> JudgmentEvent | None:
        """Freeze this decision into a :class:`~percept.contracts.JudgmentEvent`, or ``None``.

        ``None`` (not a fabricated event) whenever the caller threaded no timing accumulator, or the
        accumulator is missing a stamp the funnel never took — an event whose clock discipline the
        runtime cannot honestly attest is worse than no event. Any failure is swallowed and counted
        (``judgment.build_failed``): this stamp is ADDITIVE, so a broken stamp must never change what fires.

        ``key``/``evidence_seq`` mirror :class:`percept.sequencing.GateDecision`; the freeze stamp
        (``decision_emitted_ts``) is taken HERE, at the instant the decision becomes final, which is
        what makes ``decision_overhead_s()`` (judge-return → decision) a real measurement.
        """
        if timing is None:
            return None
        env = env or {}
        try:
            return JudgmentEvent(
                key=key, evidence_seq=int(evidence_seq), action=action, timing=timing.freeze(),
                reason=env.get("reason"), roi=env.get("roi"),
                modality=env.get("modality"), provenance=env.get("provenance"),
                evidence_origin=env.get("evidence_origin"),
            )
        except Exception as exc:  # never let an observability record perturb the firing path
            self.metrics.incr("judgment.build_failed")
            get_logger("percept.behavior").warning(
                "judgment.build_failed: key=%r action=%r — emitting the fire WITHOUT a "
                "JudgmentEvent (additive-only invariant): %r", key, action, exc,
            )
            return None

    # -- transition sense: replicates agent._perceive_transition; state in ISOLATED ctx ---------
    def fire_transition(
        self, goal, from_v: Verdict, to_v: Verdict, *, focus_id: Hashable = "scene",
        timing: JudgmentTimingBuilder | None = None, evidence_seq: int | None = None,
    ) -> Fired | None:
        """LEGACY FACADE — the historical ``Fired | None`` return every positional caller relies on.

        Delegates to :meth:`transition_decide` (which also carries the per-tick facts the agent's
        ``transition.decision`` trace needs) and hands back only its ``.fired``. The two keyword-only
        params are additive and defaulted: a caller that threads no ``timing`` gets a ``Fired`` whose
        ``.judgment`` is ``None`` (the transition path took no judge-funnel stamps), byte-for-byte the
        pre-1.9 behaviour.
        """
        return self.transition_decide(
            goal, from_v, to_v, focus_id=focus_id, timing=timing, evidence_seq=evidence_seq,
        ).fired

    def transition_decide(
        self, goal, from_v: Verdict, to_v: Verdict, *, focus_id: Hashable = "scene",
        timing: JudgmentTimingBuilder | None = None, evidence_seq: int | None = None,
    ) -> "TransitionTick":
        """Run one A→B transition tick and return BOTH the fire (if any) and the tick facts.

        The state machine is byte-for-byte the historical ``fire_transition``: observing a confident
        ``from`` state arms (and re-arms) the goal; a confident ``to`` state AFTER ``from`` was seen
        fires once. What is new is purely additive observability: when the caller threads the funnel's
        ``timing``/``evidence_seq`` in (the live watch loop does), a fired tick builds a frozen
        :class:`~percept.contracts.JudgmentEvent` through the ONE construction site (:meth:`_judgment`)
        with ``evidence_origin="transition"`` — so the transition path, which historically emitted no
        judgment record at all, now correlates to a delivered Incident exactly like the judged path.
        The gate never ran here, so ``reason`` stays ``None`` (the frozen ``GateReason`` vocabulary is
        not stretched to name a path it never decided).

        The returned :class:`TransitionTick` carries the classify outputs, the post-tick latch state,
        and a typed ``outcome`` so the agent can emit ``transition.decision`` WITHOUT reaching into this
        executor's isolated per-concern ctx.
        """
        thr, fresh_min = self.gate.threshold, self.gate.fresh_min
        ctx = self.ctx(goal.id)
        st = ctx.transitions.setdefault(focus_id, {"seen_from": False, "fired": False})  # NOT gate._armed
        from_truth = classify(from_v, 1.0, thr, fresh_min)
        to_truth = classify(to_v, 1.0, thr, fresh_min)
        prior_seen, prior_fired = st["seen_from"], st["fired"]
        ctx.confusion.record(to_v, to_truth)
        if from_truth == "yes":
            st["seen_from"], st["fired"] = True, False  # in 'from' -> arm (and re-arm)
        fired: Fired | None = None
        if st["seen_from"] and not st["fired"] and to_truth == "yes":
            st["fired"] = True
            ctx.count += 1
            # env: the transition path is a DIRECT visual observation, but the gate never ran — reason
            # stays None (GateReason is frozen) and evidence_origin marks WHICH path produced it.
            env = dict(reason=None, modality="vision", roi=None, provenance="observed",
                       evidence_origin="transition")
            judgment = self._judgment((goal.id, focus_id), evidence_seq, FIRE, timing, env)
            fired = Fired(goal.id, FIRE, goal.say, focus_id, to_v, judgment,
                          reason=None, modality="vision", provenance="observed")
        # The typed WHY of this tick, for transition.decision. Priority: a fire dominates; else an
        # arming tick reports armed (first) vs rearmed (A re-seen while already armed/latched); else a
        # to==yes while the latch is spent is a suppressed re-fire; else idle.
        if fired is not None:
            outcome = "fire"
        elif from_truth == "yes":
            outcome = "rearmed" if (prior_seen or prior_fired) else "armed"
        elif st["seen_from"] and st["fired"] and to_truth == "yes":
            outcome = "suppressed_latched"
        else:
            outcome = "idle"
        return TransitionTick(
            fired=fired, from_truth=from_truth, to_truth=to_truth,
            armed=st["seen_from"], fired_latch=st["fired"], outcome=outcome,
        )

    # -- absence sense: replicates agent.check_deadlines firing for one concern -----------------
    def fire_absence(
        self,
        goal,
        program: BehaviorProgram,
        *,
        now: float,
        satisfied: bool,
        metric: str = "overdue.gate",
    ) -> Fired | None:
        if program.deadline_at is None:
            return None
        overdue = now >= program.deadline_at and not satisfied
        verdict = Verdict(satisfied=overdue, confidence=1.0, evidence="overdue")
        # per-goal singleton -> exempt from TTL (a dropped __overdue__ edge would re-fire the deadline)
        action = self.gate.evaluate((goal.id, "__overdue__"), verdict, no_evict=True)
        self.metrics.incr(f"{metric}.{action}")
        if action == FIRE:
            return Fired(goal.id, FIRE, overdue_text(goal), None, verdict)
        return None

    # -- episode sense + accumulate: the novel-task composition (zero new code) ------------------
    def fire_episode(
        self, goal, program: BehaviorProgram, *, n: int, is_new: bool, t: float = 0.0,
        episode_id: str | None = None,
    ) -> Fired | None:
        """Fire on a NEW distinct episode, gated by the accumulator. ``fire_when`` over ``{n, t}`` is
        what makes "alert me the 3rd distinct time" pure data; default (None) fires each new episode.

        The episode/continuation decision reaches the gate ONLY as a ``rearm()`` call from
        here (gate.py stays mechanically unchanged). ``is_new`` is now the output of the separated
        continuation conjunction in :meth:`percept.episodes.EpisodeLog.observe_decision`, so a
        same-SUBJECT observation that merely continues an open episode never rolls the discriminator
        and never re-arms — which is exactly what stops an A-B-A revisit from minting a spurious
        third fire. ``episode_id`` (optional) is the stronger identity discriminator; see
        :attr:`ConcernContext.last_episode_id`."""
        ctx = self.ctx(goal.id)
        ctx.count = n
        if not is_new:
            return None
        fw = program.accumulate.fire_when
        if fw and not self._fire_when(fw)({"n": n, "t": t}):
            return None
        # ONE key per goal (was ``__episode__{n}`` — a permanent key per rep, unbounded over a long
        # count). A new distinct rep number rolls the discriminator in ctx and re-arms the singleton key —
        # the same fresh edge the per-n key used to give, so every new episode still fires exactly once.
        key = (goal.id, "__episode__")
        if episode_id is None:
            rolled = ctx.last_episode_n is not None and n != ctx.last_episode_n
        else:
            rolled = ctx.last_episode_id is not None and episode_id != ctx.last_episode_id
        if rolled:
            self.gate.rearm(key)
        ctx.last_episode_n = n
        ctx.last_episode_id = episode_id if episode_id is not None else ctx.last_episode_id
        verdict = Verdict(satisfied=True, confidence=1.0, evidence=f"episode {n}")
        action = self.gate.evaluate(key, verdict, no_evict=True)  # per-goal singleton -> exempt from TTL
        if action == FIRE:
            return Fired(goal.id, FIRE, goal.say, n, verdict)
        return None


# --------------------------------------------------------------------------------------------------
# Presets — legacy goal.state compiles to a BehaviorProgram (constructors, not branches).
# --------------------------------------------------------------------------------------------------


_PERCEPTION_FIELDS = {"modality", "temporal", "n_frames", "roi", "sampling", "audio_s", "window_s"}

# sense -> default PerceptionSpec. Empty for now (every sense -> single vision-only frame, byte-for-byte
# with today). A non-single preset (e.g. transition -> strip) is switched on ONLY when the Phase-7
# ablation proves its lift AND its consumer (strip assembly) exists. Authored
# perception always overrides the preset.
_SENSE_PERCEPTION_PRESET: dict[str, PerceptionSpec] = {}


def _perception_for(sense: Sense, authored: object) -> PerceptionSpec:
    """The sense-preset default with any validated authored fields overlaid (authored wins)."""
    base = _SENSE_PERCEPTION_PRESET.get(sense, PerceptionSpec())
    if not isinstance(authored, dict):
        return base
    fields = {k: authored[k] for k in _PERCEPTION_FIELDS if k in authored}
    return replace(base, **fields) if fields else base


def program_of(goal) -> BehaviorProgram:
    """Compile a legacy ``goal.state`` dict into a :class:`BehaviorProgram` via presets, so existing
    behavior is byte-for-byte preserved (the enum values become constructors, not dispatch paths).
    The perception axis is overlaid last: the sense-preset default + any authored override."""
    s = goal.state
    gtype, rearm = s.get("type", "watch"), s.get("rearm", "on_false")
    if gtype == "transition":
        prog = BehaviorProgram(sense="transition", from_predicate=s.get("from"), to_predicate=s.get("to"))
    elif gtype == "count":
        prog = BehaviorProgram(
            sense="episode", predicate=goal.condition, accumulate=Accumulator(kind="count_distinct")
        )
    elif gtype == "rhythm":
        # The rhythm sense: a repeating cycle counted by the brain's RhythmTracker (percept.rhythm). A
        # confirmed rep IS a distinct episode occurrence, so the FIRING mechanics reuse fire_episode (gate +
        # nth-distinct) byte-for-byte; the new sense value only marks the dispatch + carries any fire_when.
        prog = BehaviorProgram(
            sense="rhythm",
            predicate=goal.condition,
            accumulate=Accumulator(kind="count_distinct", fire_when=s.get("fire_when")),
        )
    elif rearm == "deadline":
        prog = BehaviorProgram(sense="absence", predicate=goal.condition, deadline_at=s.get("deadline_at"))
    elif rearm == "never":
        prog = BehaviorProgram(sense="judge", predicate=goal.condition, key=KeySpec(suppress_false=True))
    elif rearm == "interval":
        prog = BehaviorProgram(
            sense="judge",
            predicate=goal.condition,
            key=KeySpec(by=("entity", "time_bucket"), bucket_s=s.get("cadence_s")),
            cadence_s=s.get("cadence_s"),
        )
    else:
        prog = BehaviorProgram(  # on_false default == today; carries an authored fire_when if present
            sense="judge",
            predicate=goal.condition,
            accumulate=Accumulator(fire_when=s.get("fire_when"), window_s=s.get("window_s")),
        )
    # R-1: sustained-duration dwell, authored via goal.state["hold_s"] (same channel as deadline_after_s /
    # cadence_s). Absent -> None -> byte-for-byte the rising-edge gate. Public runtime entry points
    # validate authored duration state before this internal defensive conversion.
    hold = _coerce_hold_s(s.get("hold_s"))
    return replace(prog, perception=_perception_for(prog.sense, s.get("perception")), hold_s=hold)


def _coerce_hold_s(v: object) -> float | None:
    """Internal conversion of a previously validated positive ``goal.state['hold_s']``.

    Structured apply, Tier-0/worker, GatedLoop, and direct judged entry points reject invalid authored
    duration state before :func:`program_of` reaches this helper. ``None`` remains only a defensive
    compatibility fallback for internal/legacy callers, never the supported public contract.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if v > 0 else None


# --------------------------------------------------------------------------------------------------
# Conductor authoring — turn an UNTRUSTED LLM "behavior" block into validated goal.state keys.
# The firewall on authored behavior (sibling of the fire_when AST whitelist): every value is checked
# against a closed vocabulary and coerced to a SAFE default on anything unknown; numerics are clamped.
# An LLM can never author an out-of-vocab behavior, a negative deadline, or a busy-loop cadence.
# --------------------------------------------------------------------------------------------------

_VALID_MISS = {"trivial", "mild", "severe"}
_VALID_FF = {"trivial", "mild", "severe", "irreversible"}
_MIN_CADENCE_S = 1.0  # floor so an authored cadence_s can't busy-loop the scheduler
_VALID_MODALITY = {"vision", "audio", "fused"}
_VALID_TEMPORAL = {  # sprite = N frames -> 1 montage image (VLM-only, deprecated -> window)
    "single", "strip", "window", "sprite",
    "event_capsule",  # composite pre-roll + episode-boundary + early-snapshot evidence shape
}
_VALID_ROI = {"focus", "hint", "full"}
_VALID_SAMPLING = {"recent", "uniform"}
_MAX_STRIP = 6  # ceiling on authored n_frames so an LLM can't author a token-bomb strip
_MAX_AUDIO_S = 5.0  # ceiling on an authored audio clip (token budget)
_MAX_WINDOW_S = 30.0  # ceiling on an authored clip window (seconds) so a strip can't span minutes


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def coerce_authored_behavior(raw) -> dict:
    """Validate a conductor ``behavior`` block into safe ``goal.state`` keys (the dispatch + executor
    already understand them). Anything malformed is DROPPED -> the goal falls back to a plain watch."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    # Execution route. Preserve every closed-vocabulary route so a later worker boundary can either
    # execute it or reject it loudly; an explicit route must never be laundered into the default Watch.
    route_type = raw.get("type")
    if (
        type(route_type) is str
        and route_type == "transition"
        and type(raw.get("from")) is str
        and raw["from"].strip()
        and type(raw.get("to")) is str
        and raw["to"].strip()
    ):
        out.update(type="transition", **{"from": raw["from"], "to": raw["to"]})
    elif type(route_type) is str and route_type != "transition" and is_known_behavior_type(route_type):
        out["type"] = route_type
    # re-arm (interval needs a positive cadence; deadline needs a non-negative deadline)
    rearm = raw.get("rearm")
    if rearm in {"never", "each_event"}:
        out["rearm"] = rearm
    elif rearm == "interval":
        c = _num(raw.get("cadence_s"))
        if c is not None and c > 0:
            out.update(rearm="interval", cadence_s=max(_MIN_CADENCE_S, c))
    elif rearm == "deadline":
        # ``deadline_after_s`` is a DURATION (seconds from arm), NOT an absolute stamp. The agent resolves
        # it to an absolute ``deadline_at`` on its clock at arm time. An authored absolute would fire on the
        # wrong timeline (A-2); keeping the authored field a duration is what makes that impossible.
        d = _num(raw.get("deadline_after_s"))
        if d is not None and d >= 0:
            out.update(rearm="deadline", deadline_after_s=d)
    # accumulate (nth-distinct): fire_when is validated through the SAME AST whitelist that runs at
    # firing time — an unsafe expression is dropped here, so it never reaches goal.state.
    acc = raw.get("accumulate")
    if isinstance(acc, dict) and isinstance(acc.get("fire_when"), str):
        try:
            compile_fire_when(acc["fire_when"])
        except UnsafeExpression:
            pass  # unsafe -> drop; the goal stays a plain watch
        else:
            out["fire_when"] = acc["fire_when"]
            w = _num(acc.get("window_s"))
            if w is not None and w > 0:
                out["window_s"] = w
    # consequence (closed cost vocab; protected_floor can only be raised, never authored false-down)
    cons = raw.get("consequence")
    if isinstance(cons, dict):
        if cons.get("miss_cost") in _VALID_MISS:
            out["miss_cost"] = cons["miss_cost"]
        if cons.get("false_fire_cost") in _VALID_FF:
            out["false_fire_cost"] = cons["false_fire_cost"]
        if cons.get("protected_floor") is True:
            out["protected_floor"] = True
    return out


_MAX_CHANNEL_LEN = 40  # bound an authored channel-hint string (no busy values / token bombs)
_VALID_RHYTHM_MODE = {"discrete", "continuous"}  # rep regime: rest-between (shipped) vs rest-less (research seam)
_MAX_EXPECTED_ROM = 100.0  # bound the authored expected-ROM (channel-native units; generous — channels vary)
_MAX_SMOOTH_S = 1.0  # bound the authored causal-smoothing cold-start span (seconds); >1s over-lags any rep


def coerce_authored_rhythm(raw) -> dict:
    """Validate a conductor ``rhythm`` PRIOR into a safe ``goal.state['rhythm']`` sub-dict — the
    goal-setting half of the rhythm sense (the brain already knows the rep shape at authoring time, so the
    eye never needs a runtime VLM round-trip on rep 1). Closed/bounded vocab, clamped numerics: a
    ``channel`` hint biases :class:`~percept.rhythm.RhythmTracker` channel selection; ``cadence_s`` is a
    rough expected rep period; ``whole_body`` flags a multi-limb action. Anything malformed is DROPPED, so
    the tracker falls back to its unsupervised default (the benchmark runs with NO prior). The hint only
    *biases* selection — it can never force a miscount, because a channel with zero reps is never selected."""
    if not isinstance(raw, dict):
        return {}
    spec: dict = {}
    ch = raw.get("channel")
    if isinstance(ch, str) and 0 < len(ch) <= _MAX_CHANNEL_LEN:
        spec["channel"] = ch
    c = _num(raw.get("cadence_s"))
    if c is not None and c > 0:
        spec["cadence_s"] = min(_MAX_WINDOW_S, c)
    if raw.get("whole_body") is True:
        spec["whole_body"] = True
    # the regime the brain declares for the eye (eye↔brain coordination): a DISCRETE rep has a rest between
    # reps (bench, squat) and the shipped causal counter handles it; a CONTINUOUS rep is rest-less (pommel
    # circles, cycling, strokes) and needs phase counting (research seam — recorded here, not yet counted).
    if raw.get("mode") in _VALID_RHYTHM_MODE:
        spec["mode"] = raw["mode"]
    w = _num(raw.get("window_s"))  # endurance window for windowed rep counts ("reps in the last minute")
    if w is not None and w > 0:
        spec["window_s"] = min(_MAX_WINDOW_S, w)
    # expected ROM: the prominence a REAL rep reaches, used to reject too-small movement (the fidget). UNIT
    # CONTRACT — it is in the PRIOR ``channel``'s native units (same units as Rep.amplitude), so it is only
    # honoured when a channel is named, and is marked ``channel_native`` (the only space supported today) so a
    # frame-normalized number can never be silently fed once other channel families exist.
    rom = _num(raw.get("expected_rom"))
    if rom is not None and rom > 0 and "channel" in spec and raw.get("expected_rom_space", "channel_native") == "channel_native":
        spec["expected_rom"] = min(_MAX_EXPECTED_ROM, rom)
        spec["expected_rom_space"] = "channel_native"
    # amplitude-gate fraction override: the fraction of ``expected_rom`` a rep must reach to be counted.
    # Lower = admit shallower/fatigued reps (fewer fidget rejections). Only meaningful with expected_rom.
    frac = _num(raw.get("amp_gate_frac"))
    if frac is not None and 0.0 < frac <= 1.0:
        spec["amp_gate_frac"] = frac
    # causal smoothing span (seconds): an OPT-IN rate-invariance knob (Build 2). It scales to the measured
    # period, killing the per-sample-noise that COLLAPSES the count at high fps. Off (0/absent) by default —
    # only worth enabling above the Nyquist floor, where it pays off (see RhythmTracker.smooth_s).
    sm = _num(raw.get("smooth_s"))
    if sm is not None and 0.0 < sm <= _MAX_SMOOTH_S:
        spec["smooth_s"] = sm
    # canonical channel selection (Build 3): score channels on the canonical grid instead of the medoid, so
    # the rep-owning channel stops flipping with fps. Opt-in bool, default off (legacy medoid).
    if raw.get("canonical_select") is True:
        spec["canonical_select"] = True
    return {"rhythm": spec} if spec else {}


_MAX_CONTINUITY_GAP_S = 86_400.0  # a day; an authored continuity window beyond that is not a "gap"
_MAX_TERMINAL_STATUSES = 16
_MAX_STATUS_LEN = 40


def coerce_authored_episode(raw) -> dict:
    """Validate a conductor ``episode`` block into a safe ``goal.state['episode']`` sub-dict — the
    CONCERN-LEVEL continuation policy (consumed by
    :func:`~percept.episodes.episode_policy_of`). Sibling of :func:`coerce_authored_rhythm`: closed
    vocabulary (``policy`` must name a shipped :data:`~percept.episodes.EPISODE_POLICIES` preset),
    clamped numerics (a positive ``continuity_gap_s`` under a day), bounded string sets.

    Why this is authored per concern and not a global constant: a document-presence episode, a
    door-open interval and a conversational claim do not share continuation semantics. Anything
    malformed is DROPPED, so the concern falls back to the unbounded legacy policy — a bad hint can
    never silently SPLIT a client's episodes, only fail to tighten them."""
    if not isinstance(raw, dict):
        return {}
    spec: dict = {}
    if raw.get("policy") in EPISODE_POLICIES:
        spec["policy"] = raw["policy"]
    gap = _num(raw.get("continuity_gap_s"))
    if gap is not None and gap > 0:
        spec["continuity_gap_s"] = min(_MAX_CONTINUITY_GAP_S, gap)
    if isinstance(raw.get("allow_reopen"), bool):
        spec["allow_reopen"] = raw["allow_reopen"]
    terminal = raw.get("terminal_statuses")
    if isinstance(terminal, (list, tuple, set, frozenset)):
        statuses = sorted({
            s for s in terminal if isinstance(s, str) and 0 < len(s) <= _MAX_STATUS_LEN
        })[:_MAX_TERMINAL_STATUSES]
        if statuses:
            spec["terminal_statuses"] = statuses
    return {"episode": spec} if spec else {}


_MAX_SENSE_FPS = 60.0  # a per-concern processing rate can't exceed the capture ceiling / busy-loop the eye


def coerce_authored_sense(raw) -> dict:
    """Validate a conductor ``sense`` block into a safe ``goal.state['sense']`` sub-dict — the per-concern
    PROCESSING-density envelope for the fovea (:class:`~percept.sampling.SamplingController`). ``base_fps`` is
    the quiet floor, ``max_fps`` the motion-escalated ceiling, ``escalate_threshold`` the motion level that
    triggers the saccade. Clamped to ``[~0, 60]`` fps with ``base <= max`` and the threshold in ``[0, 1]`` —
    so an LLM can never author a busy-loop rate or an inverted envelope. Malformed -> ``{}`` (the controller
    falls back to its default policy). This is the sibling of :func:`coerce_authored_rhythm`, one tier up:
    rhythm authors HOW to count, sense authors HOW DENSELY to look."""
    if not isinstance(raw, dict):
        return {}
    spec: dict = {}
    base = _num(raw.get("base_fps"))
    if base is not None and base > 0:
        spec["base_fps"] = min(_MAX_SENSE_FPS, base)
    mx = _num(raw.get("max_fps"))
    if mx is not None and mx > 0:
        spec["max_fps"] = min(_MAX_SENSE_FPS, mx)
    if "base_fps" in spec and "max_fps" in spec:
        spec["max_fps"] = max(spec["max_fps"], spec["base_fps"])  # never an inverted envelope
    thr = _num(raw.get("escalate_threshold"))
    if thr is not None and 0.0 <= thr <= 1.0:
        spec["escalate_threshold"] = thr
    return {"sense": spec} if spec else {}


def perception_for_concern(concern) -> dict:
    """Map a :class:`~percept.contracts.Concern`'s clip declaration to a ``goal.state['perception']``
    sub-dict (the same shape :func:`coerce_authored_perception` emits). A concern declaring a positive
    ``window_s`` asks for a time-bounded temporal clip; a bare ``clip=True`` (no window) asks for a
    strip. A concern that declared neither yields ``{}`` -> the single-frame default is untouched. This
    is the seam that lets a loitering-style concern feed the right window to the judge with no new
    dispatch path: its output drops straight into ``goal.state`` and ``program_of`` consumes it."""
    window_s = getattr(concern, "window_s", None)
    if window_s is not None and window_s > 0:
        return {"perception": {"temporal": "window", "window_s": min(_MAX_WINDOW_S, float(window_s))}}
    if getattr(concern, "clip", False):
        return {"perception": {"temporal": "strip"}}
    return {}


def coerce_authored_perception(raw) -> dict:
    """Validate a conductor ``perception`` block into a safe ``goal.state['perception']`` sub-dict — the
    VLM-side HOW, sibling of :func:`coerce_authored_behavior`. Closed vocabularies + clamped numerics:
    an LLM can never author an unknown modality, a token-bomb strip, or an oversized audio clip. Anything
    malformed is DROPPED, so the concern falls back to the safe single-frame default (``PerceptionSpec()``)."""
    if not isinstance(raw, dict):
        return {}
    spec: dict = {}
    if raw.get("modality") in _VALID_MODALITY:
        spec["modality"] = raw["modality"]
    if raw.get("temporal") in _VALID_TEMPORAL:
        spec["temporal"] = raw["temporal"]
    if raw.get("roi") in _VALID_ROI:
        spec["roi"] = raw["roi"]
    if raw.get("sampling") in _VALID_SAMPLING:
        spec["sampling"] = raw["sampling"]
    n = _num(raw.get("n_frames"))
    if n is not None and n >= 1:
        spec["n_frames"] = int(min(_MAX_STRIP, n))
    a = _num(raw.get("audio_s"))
    if a is not None and a > 0:
        spec["audio_s"] = min(_MAX_AUDIO_S, float(a))
    w = _num(raw.get("window_s"))
    if w is not None and w > 0:
        # A clip window implies a temporal strip — author the time bound and (unless overridden)
        # default the temporal mode to "window" so strip assembly is time-bounded, not frame-counted.
        spec["window_s"] = min(_MAX_WINDOW_S, float(w))
        spec.setdefault("temporal", "window")
    return {"perception": spec} if spec else {}


__all__ = [
    "Sense",
    "KeySpec",
    "Accumulator",
    "PerceptionSpec",
    "Modality",
    "Temporal",
    "RoiMode",
    "Sampling",
    "BehaviorProgram",
    "UnsafeExpression",
    "compile_fire_when",
    "ConfusionSignal",
    "ConcernContext",
    "Fired",
    "TransitionTick",
    "EpisodeMatch",
    "EpisodeMutation",
    "EPISODE_MATCHES",
    "EPISODE_MUTATIONS",
    "EpisodeFoldResult",
    "EpisodeIncidentLedger",
    "ShadowObservation",
    "episode_fold",
    "episode_mutation_of",
    "revision_id",
    "BehaviorExecutor",
    "program_of",
    "coerce_authored_behavior",
    "coerce_authored_episode",
    "coerce_authored_perception",
    "coerce_authored_rhythm",
    "coerce_authored_sense",
    "perception_for_concern",
    "ask_text",
    "escalate_text",
    "overdue_text",
]
