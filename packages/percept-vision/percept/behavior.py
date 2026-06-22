"""Phase-0 spike — the general behavior-program executor (NEW module; gate + agent untouched).

Proves the go/no-go for the generalization direction (see the direction doc): ONE executor + a general
key-synthesis reproduces the legacy firing behaviors (judge / transition / absence + the
on_false / never / each_event / interval / deadline re-arms) byte-for-byte, while honoring the three
review constraints:

  1. STATE ISOLATION — stateful primitives (transition's seen_from/fired) keep cross-tick memory in a
     per-concern :class:`ConcernContext` owned HERE, keyed by concern id, NEVER in ``gate._armed``. The
     gate stays a pure ``key -> armed bit``.
  2. ESCALATION SUBSTRATE — a per-concern :class:`ConfusionSignal` records the quantitative "am I
     confused" signal (rolling confidence + consecutive-unknown) for a later Posture-A->B policy. It is
     ADDITIVE METRIC ONLY and must never alter the firing trace (so the equivalence proof stays valid).
  3. NO eval() — the accumulator's ``fire_when`` micro-expression is parsed with ``ast.parse`` and
     evaluated by a hand-written recursive walker over a strict whitelist. ``eval``/``exec``/``compile``
     are never called on it — the prompt-injection firewall on LLM-authored expressions.

This module is NOT wired into agent.py. It imports ``gate.classify``/``Gate`` but never edits gate.py.
If ``test_program_equivalence`` is green, this executor is the seed for the real refactor, one sense at
a time.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable, Hashable, Literal

from .adapters import Verdict
from .concern import consequence_of, is_hard_quadrant
from .gate import ASK, FIRE, Gate, classify

# --------------------------------------------------------------------------------------------------
# The behavior program — authored data (the LLM fills these; presets compile legacy goal.state).
# --------------------------------------------------------------------------------------------------

Sense = Literal["judge", "transition", "episode", "absence"]
KeyBy = Literal["entity", "episode", "time_bucket", "const"]


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
class BehaviorProgram:
    """The general behavior a concern runs — sense · key · accumulate · (deadline/cadence)."""

    sense: Sense = "judge"
    predicate: str = ""
    from_predicate: str | None = None
    to_predicate: str | None = None
    key: KeySpec = field(default_factory=KeySpec)
    accumulate: Accumulator = field(default_factory=Accumulator)
    deadline_s: float | None = None
    cadence_s: float | None = None


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
        action = self.gate.evaluate(key, verdict, freshness=freshness)
        self.metrics.incr(f"{metric}.{action}")
        if action == FIRE:
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
        if program.deadline_s is None:
            return None
        overdue = now >= program.deadline_s and not satisfied
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


def program_of(goal) -> BehaviorProgram:
    """Compile a legacy ``goal.state`` dict into a :class:`BehaviorProgram` via presets, so existing
    behavior is byte-for-byte preserved (the enum values become constructors, not dispatch paths)."""
    s = goal.state
    gtype, rearm = s.get("type", "watch"), s.get("rearm", "on_false")
    if gtype == "transition":
        return BehaviorProgram(sense="transition", from_predicate=s.get("from"), to_predicate=s.get("to"))
    if gtype == "count":
        return BehaviorProgram(
            sense="episode", predicate=goal.condition, accumulate=Accumulator(kind="count_distinct")
        )
    if rearm == "deadline":
        return BehaviorProgram(sense="absence", predicate=goal.condition, deadline_s=s.get("deadline_s"))
    if rearm == "never":
        return BehaviorProgram(sense="judge", predicate=goal.condition, key=KeySpec(suppress_false=True))
    if rearm == "interval":
        return BehaviorProgram(
            sense="judge",
            predicate=goal.condition,
            key=KeySpec(by=("entity", "time_bucket"), bucket_s=s.get("cadence_s")),
            cadence_s=s.get("cadence_s"),
        )
    return BehaviorProgram(sense="judge", predicate=goal.condition)  # on_false default == today


__all__ = [
    "Sense",
    "KeySpec",
    "Accumulator",
    "BehaviorProgram",
    "UnsafeExpression",
    "compile_fire_when",
    "ConfusionSignal",
    "ConcernContext",
    "Fired",
    "BehaviorExecutor",
    "program_of",
    "ask_text",
    "escalate_text",
    "overdue_text",
]
