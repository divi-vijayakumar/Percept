"""The plan conductor — author ONE perception primitive per goal, once, at submission.

What it is for
--------------
``percept`` ships several sense primitives — a frame Watch, a transition, the rhythm counter — and
they are not interchangeable: "Holding a book" holds for a stretch, "Peel onion." completes once, and
"a person completes a squat" repeats and must be COUNTED from the signal rather than judged. Nothing
in the SDK chose between them: ``Goal.state["type"]`` simply defaults to ``"watch"``
(:func:`percept.behavior_routes.authored_behavior_type`), so a caller who did not name a primitive
got the default one whether or not it could express their ask. This module turns that default into a
decision: plain-language goal in, ``Goal.state.*`` out.

Where it sits (and where it must never sit)
-------------------------------------------
**Per goal, ONCE, at submission.** Authoring is a property of the ask, not of a frame: the same goal
text authors the same primitive whether the stream is one second or one day long. Nothing here is on
the reflex path — no frame, no tier-0 decision, and no judged verdict ever reaches this module. A
watch that has been armed never calls it again.

**Its whole output is ``Goal.state.*``.** :attr:`AuthoredPrimitive.explicit` is keyed by the dotted
tunable vocabulary :class:`percept.tunables.TunableSpec` uses, and every key is asserted to live under
``Goal.state.`` (:func:`_guard_authoring_namespace`) before it is returned. That namespace is
WatchPlan-owned, so the authored plan rides in ``WatchPlan.to_dict()["explicit"]`` and is covered by
``plan_hash`` exactly like a hand-authored plan — authoring cannot make a plan unhashable or unpinnable.

Note the deliberate symmetry with :data:`percept.optimizer.GUARDRAIL_FORBIDDEN_PREFIXES`, which denies
the OPTIMIZER the entire ``Goal.state.`` namespace: an optimizer that could rewrite what a goal MEANS
would be authoring it against a score. A conductor authoring it from the user's own words is the
opposite act — it is where the meaning comes FROM — and this module writes ONLY there. Neither
guardrail is relaxed by the other; together they say "a score may not author; an author may not tune".

Reproducibility
---------------
An LLM in the authoring path would destroy benchmark reproducibility if the authoring were re-decided
on every run. :class:`AuthoringCache` is the pin: authoring is content-addressed by
``(prompt version, model, goal text)``, and a cache loaded with ``frozen=True`` REFUSES a miss rather
than calling a provider. A benchmark re-run against a frozen artifact therefore cannot reach the
network and cannot drift, while the artifact itself records exactly which model and prompt produced it.

Safety
------
The model's reply is untrusted. It is parsed and then pushed through the SAME firewalls the existing
conductor seam uses — :func:`percept.behavior.coerce_authored_behavior` (closed route vocabulary,
clamped numerics, ``fire_when`` through the AST whitelist) and
:func:`percept.behavior.coerce_authored_rhythm` — so a malformed or hostile reply degrades to a plain
Watch instead of authoring something unexecutable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .behavior import coerce_authored_behavior, coerce_authored_rhythm
from .contracts.watchplan import WatchPlan

#: The authoring model. Pinned to a dated snapshot, not a floating alias: the alias would silently
#: re-point and a frozen artifact could no longer name what produced it.
CONDUCTOR_MODEL = "claude-haiku-4-5-20251001"

#: Bumped whenever ANY part of the authoring call changes — :data:`AUTHORING_SYSTEM`, the decode
#: settings, the coercion. It is part of the cache key, so such an edit INVALIDATES every pinned
#: decision rather than silently reusing decisions the new call would not have made.
PROMPT_VERSION = "primitive-authoring-1"

#: The whole reply is one small JSON object. A bound here is what keeps the call cheap and makes a
#: runaway generation impossible.
MAX_TOKENS = 320

#: Authoring is a CLASSIFICATION, not a creative act, and it must give the same plan for the same
#: goal text. Sampling at the default temperature made the corpus-wide selection accuracy wander by
#: ~2 points between identical runs; greedy decoding removes that as a source of plan drift on top of
#: (not instead of) the frozen artifact.
TEMPERATURE = 0.0

#: ``Goal.state.`` — the ONLY namespace this module may write. See the module docstring.
AUTHORING_NAMESPACE = "Goal.state."

AUTHORING_SYSTEM = """You author ONE perception primitive for ONE plain-language goal a user wants watched.

Return STRICT JSON and nothing else:
{"behavior": {...}, "rhythm": {...}}
"rhythm" is optional and only ever accompanies the rhythm primitive.

Choose EXACTLY ONE primitive. Decide what KIND of thing the goal is, not what it is about.

1. WATCH — a state or activity that HOLDS over a stretch of time. Ask: "could someone be doing this
   for a while?" If yes, it is a watch.
   -> {"behavior": {}}
   e.g. "Holding a book" / "Sitting in a chair" / "Someone is cooking something" / "Working at a table"

2. TRANSITION — one discrete step that COMPLETES: the world is in state A, then in state B.
   REQUIREMENT: you must be able to name BOTH states concretely from the goal itself. If you cannot
   say what was true before and what is true after, it is NOT a transition.
   -> {"behavior": {"type": "transition", "from": "<plain-language state BEFORE>", "to": "<plain-language state AFTER>"}}
   e.g. "Peel onion." / "The gas cooker is turned on." / "Select the beef." / "Pours nuts into the dough."
   Also a transition: "alert me when <X becomes true>" — from "<X is not yet true>" to "<X is true>".

3. RHYTHM — counting REPETITIONS of a physical cycle the body repeats: reps, strokes, steps, laps,
   stirs. Ask: "is the same bodily motion repeating over and over?"
   -> {"behavior": {"type": "rhythm"}, "rhythm": {"mode": "discrete"}}
   e.g. "a person completes a squat" / "a person completes a jump_jack" / "count my push-ups"
   Only when the goal names a TARGET number, add {"accumulate": {"fire_when": "n == <target>"}}.

4. COUNT — counting how many DISTINCT things or separate occurrences appear. Not a repeating bodily
   cycle: different items, different people, different times something happened.
   -> {"behavior": {"type": "count"}}
   e.g. "Count how many children are visible" / "Count how many times the narrator is shown on screen"
   / "count how many different locations the girl dances in"

5. STATE MONITOR — one thing that flips BETWEEN two or more NAMED states, reported on every flip.
   REQUIREMENT: the goal must name the alternatives it moves between ("between open and closed",
   "enters or leaves", "shifts between the exterior, the interior, or graphics"). "Tell me whenever
   you see X" names no alternatives and is NOT this — it is a transition into X.
   -> {"behavior": {"rearm": "each_event"}}
   e.g. "Monitor the charging case lid and update me whenever its state changes between open and closed"

Tie-breakers:
- "count how many TIMES x happens" is COUNT, not rhythm, unless x is a bodily repetition.
- An imperative instruction ("Cut the onion.", "Check the recipe.", "Peel onion.") is a TRANSITION.
- A bare "-ing" fragment naming an activity with no stated before/after state ("Closing something",
  "Launching something", "Putting a blanket somewhere", "Pouring something into something") is a
  WATCH. The activity is observable for a stretch; you were not told what state it ended in.
- A stative description ("Holding a bag", "Someone is undressing", "Sitting in a chair") is a WATCH.
- When genuinely unsure, return {"behavior": {}} — a plain watch is the safe default.

Never invent fields. Never emit prose, explanation, or code fences."""


class AuthoringError(RuntimeError):
    """The conductor could not author, and refused to guess."""


class AuthoringCacheMiss(AuthoringError):
    """A FROZEN cache was asked for a decision it does not hold.

    Raised instead of calling a provider: that is the whole guarantee a frozen artifact makes — a
    benchmark re-run against it is reproducible because it CANNOT reach the network.
    """


def authoring_key(goal_text: str, *, model: str = CONDUCTOR_MODEL) -> str:
    """The content address of one authoring decision.

    Folds the prompt version and the model in alongside the goal text, so changing either produces a
    different key rather than silently reusing a decision the new prompt/model would not have made.
    """

    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(goal_text.encode("utf-8"))
    return digest.hexdigest()[:32]


def _guard_authoring_namespace(explicit: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if authoring ever tried to write outside ``Goal.state.``.

    A structural check, not a stylistic one: the conductor's whole claim to being safe is that it
    writes the WatchPlan-owned intent namespace and nothing else. A path leaking into ``Settings.*``
    or ``WatchProfile.*`` would be the conductor tuning the deployment, which is exactly the act
    :data:`percept.optimizer.GUARDRAIL_FORBIDDEN_PREFIXES` exists to keep separate.
    """

    for path in explicit:
        if not path.startswith(AUTHORING_NAMESPACE):
            raise AuthoringError(
                f"conductor authored {path!r}, outside its {AUTHORING_NAMESPACE!r} namespace"
            )
    return dict(explicit)


@dataclass(frozen=True)
class AuthoredPrimitive:
    """One goal's authored primitive — the conductor's entire output for that goal.

    ``explicit`` is ready to hand to :class:`~percept.contracts.watchplan.WatchPlan` as-is; it is the
    same dotted-path shape ``percept.resolver`` reads and ``plan_hash`` covers.
    """

    goal_text: str
    behavior_type: str
    explicit: Mapping[str, Any] = field(default_factory=dict)
    #: ``"model"`` (a live call), ``"cache"`` (a pinned decision replayed), or ``"fallback"`` (the
    #: reply could not be used and the safe plain Watch was substituted).
    origin: str = "model"
    model: str = CONDUCTOR_MODEL
    key: str = ""

    def to_watch_plan(self, **kwargs: Any) -> WatchPlan:
        """The armable plan. ``condition`` stays the goal text VERBATIM (never compiled).

        An ``explicit=`` passed here is MERGED over the authored paths and wins on a collision: the
        conductor authors from the user's words, and a caller (a profile, an operator, a test) may
        still be more specific about one field without discarding the rest of the authoring.
        """

        explicit = dict(self.explicit)
        explicit.update(kwargs.pop("explicit", None) or {})
        return WatchPlan(condition=self.goal_text, explicit=explicit, **kwargs)

    def as_record(self) -> dict[str, Any]:
        """The pinnable JSON row (what a frozen artifact stores)."""

        return {
            "key": self.key,
            "goal_text": self.goal_text,
            "behavior_type": self.behavior_type,
            "explicit": dict(self.explicit),
            "origin": self.origin,
            "model": self.model,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, origin: str = "cache") -> "AuthoredPrimitive":
        return cls(
            goal_text=record["goal_text"],
            behavior_type=record["behavior_type"],
            explicit=_guard_authoring_namespace(record.get("explicit") or {}),
            origin=origin,
            model=record.get("model", CONDUCTOR_MODEL),
            key=record.get("key", ""),
        )


# ---------------------------------------------------------------------------------------------------
# The pin — a content-addressed authoring artifact
# ---------------------------------------------------------------------------------------------------


class AuthoringCache:
    """Content-addressed store of authored primitives; the reproducibility pin.

    ``frozen`` is the load-bearing flag. A frozen cache raises :class:`AuthoringCacheMiss` on a miss
    instead of authoring, so a benchmark bound to a frozen artifact provably re-runs against the exact
    primitives the recorded run used — an LLM in the authoring path cannot drift the plan under it.
    """

    def __init__(
        self,
        entries: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        frozen: bool = False,
        prompt_version: str = PROMPT_VERSION,
        model: str = CONDUCTOR_MODEL,
    ) -> None:
        self._entries: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (entries or {}).items()}
        self.frozen = frozen
        self.prompt_version = prompt_version
        self.model = model

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, key: str) -> AuthoredPrimitive | None:
        record = self._entries.get(key)
        return None if record is None else AuthoredPrimitive.from_record(record)

    def put(self, primitive: AuthoredPrimitive) -> None:
        if self.frozen:
            raise AuthoringError("a frozen authoring artifact is read-only")
        self._entries[primitive.key] = primitive.as_record()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "percept-authoring-cache",
            "prompt_version": self.prompt_version,
            "model": self.model,
            "entries": {k: dict(v) for k, v in sorted(self._entries.items())},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path, *, frozen: bool = True) -> "AuthoringCache":
        """Load a pinned artifact. ``frozen`` defaults to True — reading a pin is the reason to load
        one, and a silently-writable pin is not a pin."""

        payload = json.loads(Path(path).read_text())
        if payload.get("kind") != "percept-authoring-cache":
            raise AuthoringError(f"{path!s} is not a percept authoring artifact")
        return cls(
            payload.get("entries") or {},
            frozen=frozen,
            prompt_version=payload.get("prompt_version", PROMPT_VERSION),
            model=payload.get("model", CONDUCTOR_MODEL),
        )


# ---------------------------------------------------------------------------------------------------
# Parsing the untrusted reply
# ---------------------------------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may be fenced or wrapped in prose.

    Delegates to the existing conductor seam's extractor so both conductors are robust to the same
    reply shapes; falls back to the outermost ``{...}`` span if that module is unavailable.
    """

    try:
        from .backends.anthropic_text import _extract_json as _shared

        return _shared(text)
    except Exception:  # noqa: BLE001 - never let reply-shape robustness depend on a backend import
        start, end = text.find("{"), text.rfind("}")
        return text[start : end + 1] if start != -1 and end > start else text


def coerce_authored_primitive(payload: Any, *, goal_text: str, model: str = CONDUCTOR_MODEL,
                              origin: str = "model") -> AuthoredPrimitive:
    """Push an untrusted authoring reply through the shipped firewalls into ``Goal.state.*`` paths.

    Everything the model said is validated by :func:`percept.behavior.coerce_authored_behavior` and
    :func:`percept.behavior.coerce_authored_rhythm` — the same closed vocabulary, clamped numerics and
    ``fire_when`` AST whitelist the live conductor seam already relies on. Whatever survives is
    re-keyed onto the dotted ``Goal.state.*`` vocabulary; whatever does not simply is not authored, so
    the goal degrades to a plain Watch rather than to something unexecutable.
    """

    behavior = coerce_authored_behavior(payload.get("behavior") if isinstance(payload, Mapping) else None)
    behavior_type = behavior.get("type", "watch")
    explicit = {f"{AUTHORING_NAMESPACE}{key}": value for key, value in behavior.items()}
    if behavior_type == "rhythm" and isinstance(payload, Mapping):
        # A rhythm PRIOR is meaningless on any other primitive (it names a channel/cadence of a rep),
        # so it is attached only to the primitive that consumes it.
        prior = coerce_authored_rhythm(payload.get("rhythm")).get("rhythm", {})
        explicit.update(
            {f"{AUTHORING_NAMESPACE}rhythm.{key}": value for key, value in prior.items()}
        )
    return AuthoredPrimitive(
        goal_text=goal_text,
        behavior_type=behavior_type,
        explicit=_guard_authoring_namespace(explicit),
        origin=origin,
        model=model,
        key=authoring_key(goal_text, model=model),
    )


# ---------------------------------------------------------------------------------------------------
# Scoring — selection accuracy against a labelled goal set
# ---------------------------------------------------------------------------------------------------

#: How an authored primitive maps onto the validation corpus's ``goal_kind`` label. Stated as data so
#: the reported accuracy is auditable rather than buried in a comparison.
#:
#: ``count`` and ``rhythm`` both answer "how many", so both score as the corpus's ``count`` kind — the
#: label records what the user asked for, not which of the two counting primitives serves it.
_KIND_OF_TYPE: dict[str, str] = {
    "transition": "transition",
    "rhythm": "count",
    "count": "count",
    "process": "duration",
    "watch": "duration",
}


def goal_kind_of(primitive: AuthoredPrimitive) -> str:
    """The validation-corpus ``goal_kind`` an authored primitive corresponds to.

    A plain Watch that re-arms on EVERY event is the state MONITOR shape ("tell me whenever it
    changes"), which the corpus labels ``state``; a plain Watch otherwise describes something that
    HOLDS, which the corpus labels ``duration``.
    """

    if primitive.behavior_type in ("watch", "process"):
        if primitive.explicit.get(f"{AUTHORING_NAMESPACE}rearm") == "each_event":
            return "state"
    return _KIND_OF_TYPE.get(primitive.behavior_type, "duration")


def selection_accuracy(
    pairs: Iterable[tuple[AuthoredPrimitive, str]],
) -> dict[str, Any]:
    """Per-``goal_kind`` selection accuracy over ``(authored, labelled_kind)`` pairs.

    Returns overall and per-kind ``hit``/``total``/``accuracy`` plus the confusion counts, so a
    regression names WHICH kind moved rather than only that the number fell.
    """

    per_kind: dict[str, dict[str, Any]] = {}
    confusion: dict[str, dict[str, int]] = {}
    hits = total = 0
    for primitive, label in pairs:
        predicted = goal_kind_of(primitive)
        bucket = per_kind.setdefault(label, {"hit": 0, "total": 0})
        bucket["total"] += 1
        total += 1
        if predicted == label:
            bucket["hit"] += 1
            hits += 1
        confusion.setdefault(label, {})
        confusion[label][predicted] = confusion[label].get(predicted, 0) + 1
    for bucket in per_kind.values():
        bucket["accuracy"] = bucket["hit"] / bucket["total"] if bucket["total"] else 0.0
    return {
        "hit": hits,
        "total": total,
        "accuracy": hits / total if total else 0.0,
        "per_kind": per_kind,
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------------------------------
# The conductor
# ---------------------------------------------------------------------------------------------------


class PlanConductor:
    """Authors one perception primitive per goal, once, at submission.

    ``call`` is the provider seam: ``async (system, user) -> raw reply text``. The default builds a
    lazy ``anthropic.AsyncAnthropic`` client (the vendor SDK is never imported at module load) and
    sends one bounded message with a prompt-cached system block. Injecting ``call`` is how tests — and
    any non-Anthropic deployment — drive this with no provider at all.
    """

    def __init__(
        self,
        *,
        model: str = CONDUCTOR_MODEL,
        cache: AuthoringCache | None = None,
        call: Callable[[str, str], Any] | None = None,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.model = model
        self.cache = cache
        self.max_tokens = max_tokens
        self._call = call
        self._client: Any = None

    # -- the provider seam ---------------------------------------------------------------------
    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # lazy: the vendor SDK never loads at import time
            except ImportError as exc:  # pragma: no cover - L2/L3 only
                raise AuthoringError(
                    "PlanConductor needs the 'anthropic' SDK, or an injected call= seam "
                    "(offline runs should pass a frozen AuthoringCache instead)."
                ) from exc
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def _default_call(self, system: str, user: str) -> str:
        client = self._ensure_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=TEMPERATURE,
            system=[
                # The system block is identical for every goal, so prompt caching makes the marginal
                # cost of authoring a corpus essentially the goal text itself.
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(block, "text", "") for block in response.content).strip()

    # -- authoring -----------------------------------------------------------------------------
    async def author(self, goal_text: str) -> AuthoredPrimitive:
        """Author the primitive for one goal. A cache hit spends nothing."""

        key = authoring_key(goal_text, model=self.model)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
            if self.cache.frozen:
                raise AuthoringCacheMiss(
                    f"frozen authoring artifact has no decision for {goal_text!r} (key {key}); "
                    "re-freeze the artifact instead of authoring live inside a pinned run"
                )
        call = self._call or self._default_call
        raw = await call(AUTHORING_SYSTEM, goal_text)
        try:
            payload = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, TypeError):
            # Never guess a primitive off an unreadable reply: the safe default is the plain Watch
            # the runtime would have used anyway, recorded as such so the fallback is never invisible.
            primitive = AuthoredPrimitive(
                goal_text=goal_text, behavior_type="watch", explicit={},
                origin="fallback", model=self.model, key=key,
            )
        else:
            primitive = coerce_authored_primitive(payload, goal_text=goal_text, model=self.model)
        if self.cache is not None and not self.cache.frozen:
            self.cache.put(primitive)
        return primitive

    async def author_all(
        self, goal_texts: Sequence[str], *, concurrency: int = 8
    ) -> list[AuthoredPrimitive]:
        """Author a whole corpus with a bounded number of calls in flight."""

        limiter = asyncio.Semaphore(max(1, concurrency))

        async def _one(text: str) -> AuthoredPrimitive:
            async with limiter:
                return await self.author(text)

        return list(await asyncio.gather(*(_one(text) for text in goal_texts)))


__all__ = [
    "AUTHORING_NAMESPACE",
    "AUTHORING_SYSTEM",
    "CONDUCTOR_MODEL",
    "MAX_TOKENS",
    "PROMPT_VERSION",
    "TEMPERATURE",
    "AuthoredPrimitive",
    "AuthoringCache",
    "AuthoringCacheMiss",
    "AuthoringError",
    "PlanConductor",
    "authoring_key",
    "coerce_authored_primitive",
    "goal_kind_of",
    "selection_accuracy",
]
