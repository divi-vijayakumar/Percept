"""``ScriptedJudge`` — a fully scriptable, key-free :class:`~percept.contracts.Vision` double.

Why this exists beside :class:`percept.backends.fake.FakeVision` (which is UNCHANGED — this is
purely additive): ``FakeVision`` is scripted by *frame token* against a fixed built-in dict, so a
test can only say "this token means true". It cannot say "on call 3 return this exact verdict", and
it cannot carry fields that do not exist on :class:`~percept.contracts.Verdict` yet.

The unit-layer capability fixtures (``percept.harness.synthetic``) need both:

* **per-call scripting** — a stream of N frames/turns where each step returns a *different* verdict,
  in order, so a temporal capability (continuation vs new episode, episode counting, intervals) can
  be exercised deterministically;
* **forward-compatible additive fields** — the fixtures label things the judge does not yet return
  as real ``Verdict`` fields (a durable ``subject_key``, a ``binding`` state). :func:`verdict_from_spec`
  sets them as plain attributes on the verdict instance and records them in ``verdict.scripted_extras``,
  so a fixture written today keeps working *unchanged* the day the field becomes a real dataclass field
  — which is exactly what happened to the speech-act ``operation`` (a real field since card Y2): it now
  flows through the constructor to ``verdict.operation`` with no fixture edit.

Standing rule 2 still holds: this is scripted WIRING, not intelligence. It proves our logic reads
what a judge returns. It never proves a model would return it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from ..contracts import Box, ClipEvent, Goal, Observation, Verdict

__all__ = [
    "ScriptedJudge",
    "ScriptedExhausted",
    "verdict_from_spec",
    "observation_from_spec",
]

#: Fields that live on ``Verdict`` today. Anything else in a spec is an ADDITIVE field the fixtures
#: label ahead of the implementation; it is attached to the instance instead of passed to the ctor.
_VERDICT_FIELDS = frozenset(f.name for f in dataclasses.fields(Verdict))


class ScriptedExhausted(RuntimeError):
    """Raised when the script runs out and ``on_exhausted="raise"`` (the default).

    Loud by construction: a silently-recycled verdict would turn a fixture bug into a passing test.
    """


def observation_from_spec(spec: "Mapping[str, Any] | Observation") -> Observation:
    """Build an :class:`Observation` from a JSON-ish dict (``box`` as ``[x, y, w, h]``)."""
    if isinstance(spec, Observation):
        return spec
    box = spec.get("box") or [0.0, 0.0, 1.0, 1.0]
    if isinstance(box, Box):
        b = box
    elif isinstance(box, Mapping):
        b = Box(float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"]))
    else:
        b = Box(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return Observation(
        label=str(spec.get("label", "")),
        box=b,
        confidence=float(spec.get("confidence", 1.0)),
        attrs={str(k): str(v) for k, v in (spec.get("attrs") or {}).items()},
    )


def verdict_from_spec(spec: "Mapping[str, Any] | Verdict") -> Verdict:
    """Build a :class:`Verdict` from a JSON-ish dict, tolerating fields that do not exist yet.

    Known dataclass fields go through the constructor. UNKNOWN keys (e.g. ``subject_key``,
    ``operation``, ``binding`` — the P0 additions the fixtures already label) are ``setattr``-ed onto
    the instance and mirrored in ``verdict.scripted_extras``. When P0 promotes one of those keys to a
    real field, this function starts passing it to the constructor instead and every existing fixture
    keeps working with no edit.
    """
    if isinstance(spec, Verdict):
        return spec
    known: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, value in spec.items():
        if key in _VERDICT_FIELDS:
            known[key] = value
        else:
            extras[key] = value
    events = known.get("events")
    if events:
        known["events"] = [
            e if isinstance(e, ClipEvent)
            else ClipEvent(
                t_start=float(e["t_start"]),
                t_end=None if e.get("t_end") is None else float(e["t_end"]),
                label=str(e.get("label", "")),
            )
            for e in events
        ]
    known.setdefault("satisfied", False)
    known.setdefault("confidence", 0.0)
    verdict = Verdict(**known)
    for key, value in extras.items():
        try:
            setattr(verdict, key, value)
        except (AttributeError, dataclasses.FrozenInstanceError):  # pragma: no cover - Verdict is mutable today
            pass
    # Always present, even when empty, so a caller can read it without a hasattr dance.
    try:
        setattr(verdict, "scripted_extras", dict(extras))
    except (AttributeError, dataclasses.FrozenInstanceError):  # pragma: no cover
        pass
    return verdict


class ScriptedJudge:
    """A :class:`~percept.contracts.Vision` implementation that returns exactly what you tell it to.

    Three (composable) ways to script it, checked in this order on every ``judge`` call:

    1. ``hook`` — ``hook(condition, frame, context) -> spec | Verdict | None``; ``None`` falls through.
    2. ``by_token`` — ``{str(frame): spec}``, the ``FakeVision`` style, for order-independent scripts.
    3. ``script`` — a SEQUENCE consumed one entry per call, in order. This is the temporal mode the
       synthetic corpora use: step *i* of the fixture returns entry *i*.

    ``on_exhausted`` decides what happens when a sequence script runs dry: ``"raise"`` (default —
    loud), ``"repeat"`` (re-serve the last entry) or ``"default"`` (serve ``default_verdict``).

    Every call is recorded in :attr:`calls`, so a test can assert on what the runtime actually asked
    (condition text, whether a temporal strip / audio / caller ``context`` was assembled) as well as
    on what it did with the answer.
    """

    def __init__(
        self,
        script: "Sequence[Mapping[str, Any] | Verdict] | None" = None,
        *,
        by_token: "Mapping[str, Mapping[str, Any] | Verdict] | None" = None,
        hook: "Callable[[str, Any, dict | None], Any] | None" = None,
        detect_script: "Mapping[str, Iterable[Any]] | None" = None,
        events_script: "Mapping[str, Iterable[Any]] | None" = None,
        default_verdict: "Mapping[str, Any] | Verdict | None" = None,
        on_exhausted: Literal["raise", "repeat", "default"] = "raise",
    ) -> None:
        self._script: list[Any] = list(script or [])
        self._by_token = dict(by_token or {})
        self._hook = hook
        self._detect = {k: list(v) for k, v in (detect_script or {}).items()}
        self._events = {k: list(v) for k, v in (events_script or {}).items()}
        self._default = default_verdict if default_verdict is not None else {
            "satisfied": False, "confidence": 0.0, "evidence": "scripted: default",
        }
        self._on_exhausted = on_exhausted
        self._cursor = 0
        #: One dict per ``judge`` call, in order — the interrogation record.
        self.calls: list[dict[str, Any]] = []
        #: One dict per ``detect`` call, in order.
        self.detect_calls: list[dict[str, Any]] = []

    # -- introspection ------------------------------------------------------------------------
    @property
    def cursor(self) -> int:
        """How many sequence entries have been consumed."""
        return self._cursor

    @property
    def remaining(self) -> int:
        """How many sequence entries are left unconsumed (0 when there is no sequence script)."""
        return max(0, len(self._script) - self._cursor)

    def reset(self) -> None:
        """Rewind the sequence cursor and clear the call record (the script itself is kept)."""
        self._cursor = 0
        self.calls.clear()
        self.detect_calls.clear()

    def extend(self, script: "Sequence[Mapping[str, Any] | Verdict]") -> None:
        """Append more entries to the sequence script (does not move the cursor)."""
        self._script.extend(script)

    # -- construction from a corpus ------------------------------------------------------------
    @classmethod
    def from_steps(
        cls,
        steps: "Sequence[Mapping[str, Any]]",
        *,
        key: str = "verdict",
        **kwargs: Any,
    ) -> "ScriptedJudge":
        """Build a sequence-scripted judge from a synthetic corpus case's ``steps``.

        Each step contributes its ``verdict`` block (a step with no verdict contributes an explicit
        not-satisfied entry, so the 1:1 step↔call alignment the fixtures rely on is preserved).
        """
        script = [
            s.get(key) or {"satisfied": False, "confidence": 0.0, "evidence": "scripted: no verdict on step"}
            for s in steps
        ]
        return cls(script, **kwargs)

    # -- Vision protocol ------------------------------------------------------------------------
    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        self.calls.append(
            {
                "condition": condition,
                "frame": frame,
                "roi": roi,
                "corrections": corrections,
                "strip_len": len(frames) if frames else 0,
                "has_audio": audio is not None,
                "context": context,
            }
        )
        if self._hook is not None:
            hooked = self._hook(condition, frame, context)
            if hooked is not None:
                return verdict_from_spec(hooked)
        token = str(frame)
        if token in self._by_token:
            return verdict_from_spec(self._by_token[token])
        if self._cursor < len(self._script):
            spec = self._script[self._cursor]
            self._cursor += 1
            return verdict_from_spec(spec)
        if self._script and self._on_exhausted == "repeat":
            return verdict_from_spec(self._script[-1])
        if self._on_exhausted == "raise":
            raise ScriptedExhausted(
                f"ScriptedJudge script exhausted after {self._cursor} call(s) "
                f"(condition={condition!r}, frame={token!r}). Lengthen the script or pass "
                f"on_exhausted='repeat'/'default'."
            )
        return verdict_from_spec(self._default)

    async def detect(self, frame: Any, goals: "list[Goal]") -> "list[Observation]":
        self.detect_calls.append({"frame": frame, "goals": list(goals or [])})
        return [observation_from_spec(o) for o in self._detect.get(str(frame), [])]

    async def detect_events(self, frames: list, goals: "list[Goal]") -> list:
        token = str(frames[-1]) if frames else ""
        return list(self._events.get(token, []))
