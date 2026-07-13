"""LIVE — the general behavior-program executor: THE single firing path of the whole SDK.

Every fire in the runtime routes through :class:`BehaviorExecutor` here. ``agent.py`` imports
``BehaviorExecutor`` + ``program_of`` and, via its lazily-bound ``self.executor``, dispatches EVERY
concern through one of the four fire methods below:

  * ``_fire_concern``      -> :meth:`BehaviorExecutor.fire_judged`   (agent.py:378)
  * ``_fire_count`` / rep  -> :meth:`BehaviorExecutor.fire_episode`  (agent.py:397, 413)
  * absence coverage       -> :meth:`BehaviorExecutor.fire_absence`  (agent.py:532)
  * state transition       -> :meth:`BehaviorExecutor.fire_transition` (agent.py:736)

It began as a Phase-0 spike (see the direction doc) but is now wired and load-bearing: ONE executor
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
from .concern import consequence_of, is_hard_quadrant
from .gate import ASK, FIRE, Gate, classify

# --------------------------------------------------------------------------------------------------
# The behavior program — authored data (the LLM fills these; presets compile legacy goal.state).
# --------------------------------------------------------------------------------------------------

Sense = Literal["judge", "transition", "episode", "absence", "rhythm"]
KeyBy = Literal["entity", "episode", "time_bucket", "const"]

# The perception axis — HOW the eye looks (the VLM-side half of the concern program). Every default
# reproduces today's behavior byte-for-byte: one vision-only, focus-cropped frame.
Modality = Literal["vision", "audio", "fused"]
Temporal = Literal["single", "strip", "window", "sprite"]
RoiMode = Literal["focus", "hint", "full"]
Sampling = Literal["recent", "uniform"]


@dataclass(frozen=True)
class KeySpec:
    """How the re-arm boundary folds into the gate key — the ONE general re-arm mechanism (S2)."""

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
    window_s: float | None = None  # B1: time length of the temporal clip (seconds); bounds strip assembly


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
    # never a cross-timeline value. (R3: one field, one meaning, one clock.)
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


@dataclass
class Fired:
    """Executor firing result — shape-compatible with ``agent.FireEvent`` (action/text/entity_id)."""

    goal_id: str
    action: str
    text: str
    entity_id: object
    verdict: object


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

    def ctx(self, concern_id: str) -> ConcernContext:
        return self._ctx.setdefault(concern_id, ConcernContext())

    def _fire_when(self, expr: str) -> Callable[[dict], bool]:
        pred = self._fire_when_cache.get(expr)
        if pred is None:
            pred = compile_fire_when(expr)
            self._fire_when_cache[expr] = pred
        return pred

    def _synth_key(self, base_key: tuple, key: KeySpec, t: float) -> tuple:
        """Generalizes agent._fire_concern's key-synthesis. Default => base_key (byte-for-byte)."""
        if "time_bucket" in key.by and key.bucket_s:
            return (*base_key, int(t // key.bucket_s))  # new bucket = fresh armed edge (interval)
        return base_key

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
    ) -> Fired | None:
        ctx = self.ctx(goal.id)
        truth = classify(verdict, freshness, self.gate.threshold, self.gate.fresh_min)
        ctx.confusion.record(verdict, truth)  # constraint 2: additive, pre-gate, no trace effect
        if program.key.suppress_false and truth == "no":
            return None  # never-contract: a confident-false must not reach the gate
        key = self._synth_key(base_key, program.key, t)
        # R-1: pass the goal's dwell (hold_s) + the frame time. hold_s=None -> the gate ignores `now`
        # and this is byte-for-byte the rising-edge path.
        action = self.gate.evaluate(key, verdict, freshness=freshness, hold_s=program.hold_s, now=t)
        self.metrics.incr(f"{metric}.{action}")
        if action == FIRE:
            # accumulate (nth-distinct): a rising edge IS a distinct occurrence, so gate the
            # USER-facing fire on fire_when(n). No fire_when -> byte-for-byte (existing goals never
            # accumulate). This is how "alert me the 3rd distinct time" composes with zero new code.
            if program.accumulate.fire_when:
                ctx.count += 1
                if not self._fire_when(program.accumulate.fire_when)({"n": ctx.count, "t": t}):
                    return None  # counted the occurrence, but it isn't the nth -> no alert
            if is_hard_quadrant(consequence_of(goal)):  # A3 hard-quadrant escalates to ASK
                self.metrics.incr("consequence.escalate")
                return Fired(goal.id, ASK, escalate_text(goal), base_key[-1], verdict)
            return Fired(goal.id, FIRE, goal.say, base_key[-1], verdict)
        if action == ASK:
            return Fired(goal.id, ASK, ask_text(goal), base_key[-1], verdict)
        return None

    # -- transition sense: replicates agent._perceive_transition; state in ISOLATED ctx ---------
    def fire_transition(
        self, goal, from_v: Verdict, to_v: Verdict, *, focus_id: Hashable = "scene"
    ) -> Fired | None:
        thr, fresh_min = self.gate.threshold, self.gate.fresh_min
        ctx = self.ctx(goal.id)
        st = ctx.transitions.setdefault(focus_id, {"seen_from": False, "fired": False})  # NOT gate._armed
        ctx.confusion.record(to_v, classify(to_v, 1.0, thr, fresh_min))
        if classify(from_v, 1.0, thr, fresh_min) == "yes":
            st["seen_from"], st["fired"] = True, False  # in 'from' -> arm (and re-arm)
        if st["seen_from"] and not st["fired"] and classify(to_v, 1.0, thr, fresh_min) == "yes":
            st["fired"] = True
            ctx.count += 1
            return Fired(goal.id, FIRE, goal.say, focus_id, to_v)
        return None

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
        action = self.gate.evaluate((goal.id, "__overdue__"), verdict)
        self.metrics.incr(f"{metric}.{action}")
        if action == FIRE:
            return Fired(goal.id, FIRE, overdue_text(goal), None, verdict)
        return None

    # -- episode sense + accumulate: the novel-task composition (zero new code) ------------------
    def fire_episode(
        self, goal, program: BehaviorProgram, *, n: int, is_new: bool, t: float = 0.0
    ) -> Fired | None:
        """Fire on a NEW distinct episode, gated by the accumulator. ``fire_when`` over ``{n, t}`` is
        what makes "alert me the 3rd distinct time" pure data; default (None) fires each new episode."""
        ctx = self.ctx(goal.id)
        ctx.count = n
        if not is_new:
            return None
        fw = program.accumulate.fire_when
        if fw and not self._fire_when(fw)({"n": n, "t": t}):
            return None
        key = (goal.id, f"__episode__{n}")  # a fresh key per distinct episode (each_event analog)
        verdict = Verdict(satisfied=True, confidence=1.0, evidence=f"episode {n}")
        action = self.gate.evaluate(key, verdict)
        if action == FIRE:
            return Fired(goal.id, FIRE, goal.say, n, verdict)
        return None


# --------------------------------------------------------------------------------------------------
# Presets — legacy goal.state compiles to a BehaviorProgram (constructors, not branches).
# --------------------------------------------------------------------------------------------------


_PERCEPTION_FIELDS = {"modality", "temporal", "n_frames", "roi", "sampling", "audio_s", "window_s"}

# sense -> default PerceptionSpec. Empty for now (every sense -> single vision-only frame, byte-for-byte
# with today). A non-single preset (e.g. transition -> strip) is switched on ONLY when the Phase-7
# ablation proves its lift (the S1 discipline) AND its consumer (strip assembly) exists. Authored
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
    # cadence_s). Absent -> None -> byte-for-byte the rising-edge gate. Coerced defensively (untrusted).
    hold = _coerce_hold_s(s.get("hold_s"))
    return replace(prog, perception=_perception_for(prog.sense, s.get("perception")), hold_s=hold)


def _coerce_hold_s(v: object) -> float | None:
    """goal.state['hold_s'] -> a positive float dwell, or None. A non-numeric / non-positive hint -> None
    (no dwell), so a malformed value can never silently suppress firing."""
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
_VALID_TEMPORAL = {"single", "strip", "window", "sprite"}  # sprite = N frames -> 1 montage image (VLM-only)
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
    # tracking pattern (transition needs both from/to in plain language)
    if raw.get("type") == "transition" and isinstance(raw.get("from"), str) and isinstance(raw.get("to"), str):
        out.update(type="transition", **{"from": raw["from"], "to": raw["to"]})
    elif raw.get("type") in {"event", "count"}:
        out["type"] = raw["type"]
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
    """Map a :class:`~percept.contracts.Concern`'s B1 clip declaration to a ``goal.state['perception']``
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
        # B1: a clip window implies a temporal strip — author the time bound and (unless overridden)
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
    "BehaviorExecutor",
    "program_of",
    "coerce_authored_behavior",
    "coerce_authored_perception",
    "coerce_authored_rhythm",
    "coerce_authored_sense",
    "perception_for_concern",
    "ask_text",
    "escalate_text",
    "overdue_text",
]
