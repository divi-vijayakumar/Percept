"""DirectJudge — route a condition to a cheap, vision-free answer, WITHOUT touching gate authority.

Some conditions do not need a vision call at all: an OCR-extracted field already answers
"does the invoice total exceed $500", a detector's own boundary already answers "did motion
cross this line". :class:`DirectJudge` is the routing layer in front of the normal judge call
that recognizes those cases and answers directly — a real cost win (zero vision spend) on the
CONFIDENT path, with two honest escape hatches back to the judge on anything short of confident.

**The Gate remains sole authority.** DirectJudge never fires anything itself; it produces a
:class:`Verdict` (on the confident path) exactly like a vision backend would, and the SAME
unmodified :class:`~percept.gate.Gate` classifies it. This module adds a cheaper way to PRODUCE a
verdict, never a second way to DECIDE one.

Three outcomes, one honest fallback ladder:

- **confident** — the rule answered outright; its normalized :class:`Verdict` is used as-is, no
  vision call spent.
- **uncertain** — the rule has partial signal (structured facts) but not enough to answer;
  those facts are handed to the NORMAL judge call as extractor-shaped context (see
  :mod:`percept.extraction`), so the vision call that does happen is at least better-informed.
- **degraded** — no rule applies, or the rule itself failed; falls through to the normal judge
  path completely unchanged (fail-closed: a broken direct rule can only cost a vision call it
  would have spent anyway, never produce a wrong confident answer).

**Public routing is typed, not the compiler's internal vocabulary.** A rule declares
``applies_to`` as a :class:`DirectRouting` — a ``condition_namespace`` prefix and/or a
:mod:`percept.catalog` component ``role``. ``goal.state['kind']`` (the runtime compiler's
internal target once a goal is compiled) is NEVER read here and MUST NOT become how a client
selects a direct rule — that field is free to change shape across compiler internals; the public
contract is ``applies_to``.

**Stakes-derived verify cadence — WHAT IT DOES, AND WHAT IT DOES NOT.**
:attr:`DirectJudge.verify_cadence` maps a caller-supplied ``stakes`` label to "spend a real judge
call every N confident answers" (``0`` = never force). A HIGH-stakes condition gets a tighter
cadence than a LOW-stakes one by default; a caller may override the whole map.

It **forces the call. It does not check the rule.** The Nth confident answer is downgraded to
``uncertain``, so the caller runs a real judge — and the rule's own verdict is discarded at that
point. Nothing here receives the judge's answer, nothing compares the two, and no divergence is
counted. So a rule that is confidently wrong produces a periodic real call whose result agrees or
disagrees with it *unobserved*.

**Verification is therefore still the caller's job**, and a caller relying on this as a correctness
backstop is relying on something that is not implemented. Contrast
:class:`~percept.cache.framehash.FrameHashLayer`'s audit, which is the shape a real backstop takes:
re-perceive a sampled fraction, compare ``satisfied``, count the divergence, and let the fresh read
win. Closing that gap here means feeding the judge's answer back and comparing it; that is a
DECISION, not an oversight, and it is recorded in the backlog rather than hidden behind this
docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from .contracts import Box, Verdict, verdict_defect

DirectOutcome = Literal["confident", "uncertain", "degraded"]

#: The closed outcome vocabulary a DirectRule may report. Anything else is a caller bug.
DIRECT_OUTCOMES: tuple[str, ...] = ("confident", "uncertain", "degraded")

#: The default stakes -> verify-cadence map (confident answers between forced real judge calls;
#: ``0`` = never force). Caller-overridable wholesale via ``DirectJudge(verify_cadence=...)`` —
#: these are sane defaults, not a tuned claim.
DEFAULT_VERIFY_CADENCE: Mapping[str, int] = {"low": 0, "medium": 20, "high": 5}


@dataclass(frozen=True)
class DirectRouting:
    """The PUBLIC matcher a :class:`DirectRule` declares as its ``applies_to``.

    ``condition_namespace`` matches a concern's plain-language ``condition`` by prefix (e.g.
    ``"screen.ocr."`` matches ``"screen.ocr.total_exceeds"``). ``role`` matches a
    :mod:`percept.catalog` component role (``ROLE_DETECTOR``, ...) when the caller supplies one at
    :meth:`DirectJudge.route` time. Either, both, or neither may be set; unset means "match any"
    on that axis — a rule with neither set matches every call, so at most one such catch-all rule
    should be registered.

    Deliberately NOT ``goal.state['kind']``: that key is the runtime compiler's internal compile
    target (see ``percept.runtime_compiler``) and is free to change shape within a MINOR; a public
    routing contract cannot be built on it.
    """

    condition_namespace: str | None = None
    role: str | None = None

    def matches(self, *, condition: str, role: str | None = None) -> bool:
        if self.condition_namespace is not None and not condition.startswith(self.condition_namespace):
            return False
        if self.role is not None and self.role != role:
            return False
        return True


def _coerce_verdict(spec: "Verdict | Mapping[str, Any] | None") -> Verdict | None:
    """Normalize a rule's answer into a real :class:`Verdict` — a dict or a Verdict both work, so
    a rule author can return a plain ``{"satisfied": True, "confidence": 0.95}`` without importing
    the dataclass. ``None`` stays ``None`` (checked by :class:`DirectResult`).

    Normalizing is not validating, and this used to do only the first: ``Verdict(**dict(spec))``
    accepts a third-party rule's raw dict with zero shape checking, because a dataclass constructor
    does not type-check its fields. A rule returning ``{"satisfied": True, "confidence": "high"}``
    or a NaN produced a ``Verdict``-typed object carrying nonsense that the gate then read as a
    confident yes. This is a judge seam — ADR-001 §7 says the backend contract is ``Verdict`` and
    nothing else, and a rule answering instead of a model is no more trusted than the model — so
    the one authority (:func:`percept.contracts.verdict_defect`) is asked here, as close to the
    third-party value as it is possible to get.

    Raises ``ValueError`` naming the failed term. :meth:`DirectJudge.route` catches whatever a rule
    raises and returns ``outcome="degraded"``, so on the normal path this costs the vision call the
    rule would have saved — fail-closed, exactly as the module docstring promises for a broken rule.
    """
    if spec is None:
        return None
    verdict = spec if isinstance(spec, Verdict) else Verdict(**dict(spec))
    defect = verdict_defect(verdict, require_exact_type=True)
    if defect is not None:
        raise ValueError(
            f"DirectRule returned a malformed Verdict ({defect}): {verdict!r}. A direct rule's "
            "answer goes to the gate unchanged, so it must satisfy the same contract a vision "
            "backend does (see percept.contracts.VERDICT_DEFECTS)."
        )
    return verdict


@dataclass(frozen=True)
class DirectResult:
    """One :class:`DirectRule` evaluation. ``verdict`` is REQUIRED (and normalized) iff
    ``outcome == "confident"``; ``facts`` carries whatever partial signal an ``"uncertain"``
    outcome wants forwarded to the normal judge as extractor-shaped context."""

    outcome: DirectOutcome
    verdict: "Verdict | Mapping[str, Any] | None" = None
    facts: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate the outcome vocabulary and the verdict SHAPE at construction.

        This used to check only that a ``confident`` result carried a non-``None`` verdict — the
        presence of an answer, never its well-formedness. A rule's verdict reaches
        ``percept.gate.classify`` unchanged, so presence alone is not the contract (ADR-001 §10
        invariant 4); :func:`_coerce_verdict` now enforces the full one. Any verdict is checked, not
        just a ``confident`` one: a ``Verdict`` this object hands out is a ``Verdict``, and a rule
        attaching a malformed one to an ``uncertain`` result is reporting a bug worth raising on
        rather than a value that happens to go unread today.
        """
        if self.outcome not in DIRECT_OUTCOMES:
            raise ValueError(f"DirectResult.outcome must be one of {DIRECT_OUTCOMES}, got {self.outcome!r}")
        verdict = _coerce_verdict(self.verdict)
        if self.outcome == "confident" and verdict is None:
            raise ValueError("DirectResult(outcome='confident') requires a verdict")
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "facts", dict(self.facts))


@runtime_checkable
class DirectRule(Protocol):
    applies_to: DirectRouting

    def evaluate(
        self, condition: str, frame: Any, roi: Box | None, facts: Mapping[str, Any]
    ) -> DirectResult:
        """Answer ``condition`` directly if possible. MUST NOT raise on a normal miss — return
        ``DirectResult(outcome="degraded")`` instead (mirrors the extractor/enricher no-raise
        convention); :class:`DirectJudge` also catches a raise defensively and degrades."""
        ...


class DirectJudge:
    """Routes a condition to the first matching :class:`DirectRule`, applying stakes-derived
    verify cadence on the confident path. Registration order is match priority (first match
    wins) — mirrors every other name/priority registry in this package."""

    def __init__(
        self,
        rules: "Any" = (),
        *,
        verify_cadence: Mapping[str, int] | None = None,
    ) -> None:
        self._rules: list[DirectRule] = list(rules)
        self.verify_cadence: dict[str, int] = dict(
            DEFAULT_VERIFY_CADENCE if verify_cadence is None else verify_cadence
        )
        # confident-streak counter, per cadence key (default: the condition text itself).
        self._streak: dict[str, int] = {}

    def register(self, rule: DirectRule) -> None:
        self._rules.append(rule)

    def _rule_for(self, condition: str, role: str | None) -> DirectRule | None:
        for rule in self._rules:
            if rule.applies_to.matches(condition=condition, role=role):
                return rule
        return None

    def route(
        self,
        condition: str,
        frame: Any,
        roi: Box | None = None,
        *,
        role: str | None = None,
        facts: Mapping[str, Any] | None = None,
        stakes: str = "medium",
        key: str | None = None,
    ) -> DirectResult:
        """Route one judge-shaped call. Returns a :class:`DirectResult` — the caller (typically
        ``Percept.perceive_judged``) is responsible for using ``outcome == "confident"``'s
        ``verdict`` AS the judge's verdict (spending zero vision calls), forwarding
        ``outcome == "uncertain"``'s ``facts`` into the normal judge call's context, and taking
        ``outcome == "degraded"`` as "call the normal judge path, unchanged"."""
        rule = self._rule_for(condition, role)
        if rule is None:
            return DirectResult(outcome="degraded", reason="no DirectRule.applies_to matched")
        try:
            result = rule.evaluate(condition, frame, roi, facts or {})
        except Exception as exc:  # noqa: BLE001 — a broken rule degrades, it never crashes the tick
            return DirectResult(outcome="degraded", reason=f"DirectRule raised: {exc!r}")
        if result.outcome != "confident":
            return result
        cadence_key = key if key is not None else condition
        cadence = self.verify_cadence.get(stakes, 0)
        streak = self._streak.get(cadence_key, 0) + 1
        self._streak[cadence_key] = streak
        if cadence and streak % cadence == 0:
            # Stakes-derived forced re-verify: the Nth confident answer on THIS cadence is
            # deliberately downgraded to "uncertain" so the caller spends one real judge call.
            #
            # ``result.verdict`` IS DISCARDED HERE, and that is the limit of the mechanism: the
            # judge's answer is never returned to this object, so the rule is never checked against
            # it. This forces a call; it does not verify anything. See the module docstring.
            #
            # (Also note the counter is cumulative, not consecutive — it is incremented only on
            # ``confident`` and never reset by an intervening ``uncertain``/``degraded``, so this is
            # "every Nth confident answer ever" rather than "N in a row".)
            return DirectResult(
                outcome="uncertain", facts=result.facts,
                reason=f"stakes-derived verify cadence reached (stakes={stakes!r}, every {cadence})",
            )
        return result

    def reset(self, key: str | None = None) -> None:
        """Clear the confident-streak counters — all of them, or just ``key``'s."""
        if key is None:
            self._streak.clear()
        else:
            self._streak.pop(key, None)


__all__ = [
    "DIRECT_OUTCOMES",
    "DEFAULT_VERIFY_CADENCE",
    "DirectOutcome",
    "DirectRouting",
    "DirectResult",
    "DirectRule",
    "DirectJudge",
]
