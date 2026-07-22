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

**Stakes-derived verify cadence.** A confident streak is not exempt from ever seeing a real
judge: :attr:`DirectJudge.verify_cadence` maps a caller-supplied ``stakes`` label to "verify for
real every N confident answers" (``0`` = never force). A HIGH-stakes condition gets a tighter
cadence than a LOW-stakes one by default; a caller may override the whole map. This is what keeps
an unconditionally-confident-but-wrong rule from running forever unverified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from .contracts import Box, Verdict

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
    the dataclass. ``None`` stays ``None`` (checked by :class:`DirectResult`)."""
    if spec is None or isinstance(spec, Verdict):
        return spec
    return Verdict(**dict(spec))


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
            # Stakes-derived forced re-verify: the Nth confident-in-a-row answer on THIS cadence
            # is deliberately downgraded to "uncertain" so the caller spends one real judge call
            # instead of trusting the direct rule forever unverified.
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
