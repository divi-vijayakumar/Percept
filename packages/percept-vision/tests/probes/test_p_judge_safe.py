"""P-JUDGE-SAFE-1 (ADR-001 §10 invariant 4; the release criteria) — SCAFFOLD PROBE.

Property under test: **no malformed / truncated / timed-out / partial judge response may ever
cause a FIRE.** Every such response must resolve to ASK, SILENT, or a provider-failure
classification. This is a probe/test harness, not a for-record benchmark — it observes the REAL
judge->gate path (``Percept.perceive_judged``, ``percept/agent.py``) under fault injection.

Fault injector: :class:`~percept.backends.fake_judge.FakeJudge` (a key-free, deterministic judge
double; see that module's docstring). ``ErrorInjectionConfig(rate=1.0, modes=(<mode>,))`` forces
EVERY call to fail with exactly one named mode, so each test below drives the real path through
ONE fully-deterministic failure shape.

``_ProbeVision`` below adapts ``FakeJudge`` (a judge-ONLY double) to the full
``percept.contracts.Vision`` protocol Percept.create() expects (``detect``/``detect_events`` are
no-ops — this probe only cares what a judge() failure does to the gate). It lives entirely in this
file; nothing outside this module was touched.

Two things this module asserts, kept deliberately SEPARATE:

1. ``TestJudgeSafeNeverFires`` — the HARD safety invariant this card exists to prove. Every mode,
   every run, MUST resolve to something other than FIRE. This must always pass; a failure here is
   RELEASE-BLOCKING (a fail-open bug) and must be reported prominently, not papered over.
2. ``TestJudgeSafeOutcomeClassification`` — asserts *how* each mode resolves: gracefully, in-loop,
   with the run continuing (never an uncaught crash). Originally this class only documented a gap
   (see the HISTORY note below); the implementation closed that gap, so it now asserts the FIXED behavior and
   will fail loudly again if a regression reopens it.

HISTORY (now fixed — kept for context): before A30, the judge->gate path neither caught
a raised judge-call exception (timeout / provider error) nor validated the shape of a returned
verdict (malformed / partial) before handing it to ``percept.gate.classify``. ``timeout`` /
``provider_error`` raised a designed, typed exception (``asyncio.TimeoutError`` / ``ProviderError``)
that nonetheless propagated UNCAUGHT out of ``Percept.perceive_judged``, and ``malformed`` /
``partial`` (a plain, non-``Verdict`` dict) crashed with a bare, unvalidated-input
``AttributeError`` the moment ``percept.gate.classify`` read ``verdict.confidence``. Either way the
caller (``percept/runtime/worker.py``'s ``run()`` loop) caught the exception at the top level,
emitted a ``runtime_error`` health event, and RE-RAISED — ending the whole watch run. The hard
"never FIRE" invariant held throughout (the raise always preceded ``Gate.evaluate``), but the
second half of invariant 4 ("must resolve to ASK/SILENT/provider-failure") did not.

CURRENT STATE (post-A30): ``Percept._safe_timed_judge`` (agent.py) wraps every live judge call on
the watch path — catching a raised exception AND shape-validating a returned response
(``_is_valid_verdict``) — and raises the internal ``_JudgeDegraded`` signal on either failure.
``perceive_judged`` (and ``_observe_and_log``, the process-goal path) catch ``_JudgeDegraded``
BEFORE ``classify``/the gate ever see it: the tick degrades to an explicit ASK-class abstention
(``funnel.refused`` — the same telemetry a real low-confidence read gets), the gate's armed/dwell
edge state is left completely untouched (the exception is caught before ``Gate.evaluate`` is ever
reached), and ``perceive_judged`` returns its normal ``list[FireEvent]`` — it never raises. The
watch run continues onto the next frame exactly as if nothing had gone wrong, for all four modes
alike.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from percept.adapters import Goal
from percept.agent import Percept
from percept.backends.fake_judge import ErrorInjectionConfig, FakeJudge, LatencyProfile, ProviderError

# -- the failure surface (fake_judge.py's `_ERROR_MODES`) ------------------------------------------
# One tuple drives every parametrized test below. Adding a new mode to fake_judge's
# ErrorInjectionConfig later is a ONE-LINE extension here (just append its name) -- both test
# classes iterate this tuple, nothing else needs to change.
FAILURE_MODES: tuple[str, ...] = ("timeout", "provider_error", "malformed", "partial")

# Exception types a REAL judge call may raise (timeout / provider error) that
# `Percept._safe_timed_judge` (agent.py) catches and normalizes to the internal
# `_JudgeDegraded` signal before `perceive_judged` ever sees them -- retained here (and exercised
# indirectly through FakeJudge) so this probe still documents which raised shapes the boundary is
# proven to catch. Mirrors the raise-after-retries contract real backends already use
# (percept/backends/_resilience.py: CircuitOpen / the last real exception after with_retry's
# attempts are exhausted) -- extend this set once a real adapter's own transport exception type is
# wired into the same probe (see the TODO hook at the bottom of this file).
KNOWN_PROVIDER_FAILURE_EXCEPTIONS: tuple[type[BaseException], ...] = (asyncio.TimeoutError, ProviderError)


class _ProbeVision:
    """Adapts a judge-only double (:class:`FakeJudge`) to the full ``percept.contracts.Vision``
    protocol that ``Percept.create()``/``perceive_judged`` expect, so the probe can drive the REAL
    production judge->gate path end to end instead of re-implementing any of it. ``detect``/
    ``detect_events`` are no-ops -- P-JUDGE-SAFE-1 only cares what a ``judge()`` failure does to the
    gate, never the (goal-independent) entity-detection side of a frame. Lives entirely in this
    test module; nothing in ``percept/backends/fake_judge.py`` was touched."""

    def __init__(self, judge_double: FakeJudge) -> None:
        self._judge_double = judge_double

    async def judge(self, *args: Any, **kwargs: Any):
        return await self._judge_double.judge(*args, **kwargs)

    async def detect(self, frame: Any, goals: list) -> list:
        return []

    async def detect_events(self, frames: list, goals: list) -> list:
        return []


def _build_agent(mode: str, *, seed: int = 0) -> tuple[Percept, FakeJudge]:
    """One watch goal, one Percept core, wired to a FakeJudge that ALWAYS fails with ``mode``
    (``rate=1.0``) -- deterministic, key-free, zero real vendor traffic."""
    judge = FakeJudge(
        seed=seed,
        latency=LatencyProfile(kind="fixed", value_s=0.0),
        errors=ErrorInjectionConfig(rate=1.0, modes=(mode,)),
        simulate_sleep=False,
    )
    agent = Percept.create(vision=_ProbeVision(judge), discover_plugins=False)
    agent.add_goal(Goal(id="g", condition="the pot is boiling over", say="the pot is boiling over!"))
    return agent, judge


async def _drive_one_frame(agent: Percept, *, t: float = 1.0):
    """Feed exactly one frame through the REAL production judge->gate path
    (``Percept.perceive_judged``, ``percept/agent.py``).

    Returns ``(events, exc)``:
      * on a normal return, ``events`` is the ``list[FireEvent]`` and ``exc`` is ``None``. Post-A30
        this is what EVERY injected failure mode does too (the fail-closed boundary degrades the
        tick before it ever reaches ``BehaviorExecutor.fire_judged`` / ``Gate.evaluate``).
      * on a raised exception, ``events`` is ``None`` and ``exc`` is the exception instance -- this
        would now indicate a REGRESSION for any of ``FAILURE_MODES`` (see the module docstring's
        CURRENT STATE note).
    """
    try:
        events = await agent.perceive_judged("frame-token", t=t)
        return events, None
    except Exception as exc:  # noqa: BLE001 -- the probe deliberately observes ANY exception shape
        return None, exc


class TestJudgeSafeNeverFires(unittest.TestCase):
    """THE hard safety property this card exists to prove (ADR-001 §10 invariant 4): whatever a
    judge backend returns -- garbled, truncated, timed out, or erroring -- the real judge->gate
    path must never emit a FIRE. Every sub-test here MUST pass; if one doesn't, that is a
    RELEASE-BLOCKING finding (a fail-open bug), not a scaffold nit."""

    def test_never_fires_for_any_injected_failure_mode(self) -> None:
        for mode in FAILURE_MODES:
            with self.subTest(mode=mode):
                agent, _judge = _build_agent(mode)
                events, exc = asyncio.run(_drive_one_frame(agent))
                if events is not None:
                    fired = [e for e in events if e.action == "fire"]
                    self.assertEqual(
                        fired,
                        [],
                        f"RELEASE-BLOCKING: injected judge failure mode {mode!r} reached FIRE: "
                        f"{fired!r} -- a malformed/timed-out/partial/erroring judge response must "
                        f"NEVER fire (ADR-001 §10 invariant 4).",
                    )
                else:
                    # No FireEvent list was even produced -- perceive_judged raised before it could
                    # reach the gate. Structurally that means FIRE was unreachable this tick (see
                    # the call order agent.py:913-954: the raise always precedes
                    # BehaviorExecutor.fire_judged / Gate.evaluate for every mode covered here).
                    self.assertIsNotNone(
                        exc, f"{mode!r}: perceive_judged returned neither events nor an exception"
                    )

    def test_gate_edge_state_is_unmutated_by_a_failed_judge_call(self) -> None:
        """Belt-and-suspenders: an injected failure must not silently arm/disarm the rising edge
        either (that would let a LATER clean call fire/suppress incorrectly because of a fault that
        should never have touched gate state at all)."""
        for mode in FAILURE_MODES:
            with self.subTest(mode=mode):
                agent, _judge = _build_agent(mode)
                snapshot_before = agent.gate.snapshot()
                asyncio.run(_drive_one_frame(agent))
                snapshot_after = agent.gate.snapshot()
                self.assertEqual(
                    snapshot_before,
                    snapshot_after,
                    f"{mode!r}: gate edge state changed as a side effect of a failed judge call",
                )


class TestJudgeSafeOutcomeClassification(unittest.TestCase):
    """Asserts *how* (not just *whether*) each mode resolves: gracefully, in-loop, with the run
    continuing -- not the hard "never FIRE" invariant (see ``TestJudgeSafeNeverFires`` for that).
    the implementation (the fail-closed judge->classify boundary, ``Percept._safe_timed_judge`` /
    ``_JudgeDegraded`` in agent.py) landed, so EVERY mode -- raised (timeout/provider_error) or a
    malformed/partial *return* alike -- must now resolve to a graceful ASK/SILENT/provider-failure
    abstention: ``perceive_judged`` returns its normal (possibly-empty) ``list[FireEvent]``, never
    raises, and the gate is never even touched. See the module docstring's CURRENT STATE note."""

    def test_every_failure_mode_resolves_gracefully_and_the_run_continues(self) -> None:
        """THE fixed behavior this class exists to pin down: no injected failure mode may ever
        escape ``perceive_judged`` as an uncaught exception (that is what used to end the whole
        watch run pre-A30) -- every one degrades THAT TICK to ASK/SILENT/provider-failure and lets
        the watch continue onto the next frame, never a FIRE."""
        for mode in FAILURE_MODES:
            with self.subTest(mode=mode):
                agent, _judge = _build_agent(mode)
                events, exc = asyncio.run(_drive_one_frame(agent))
                self.assertIsNone(
                    exc,
                    f"REGRESSION: {mode!r} escaped perceive_judged as an uncaught "
                    f"exception ({exc!r}) instead of degrading the tick gracefully -- this is the "
                    f"exact fail-closed-boundary gap A30 fixed; the watch run must continue, never "
                    f"crash, on a malformed/timed-out/partial/erroring judge response.",
                )
                self.assertIsNotNone(
                    events, f"{mode!r}: expected perceive_judged's normal FireEvent-list return"
                )
                fired = [e for e in events if e.action == "fire"]
                self.assertEqual(
                    fired, [], f"{mode!r}: a degraded judge response must never FIRE"
                )

    def test_timeout_and_provider_error_are_recognized_designed_failure_shapes(self) -> None:
        """Documentary: FakeJudge's ``timeout``/``provider_error`` injection raises the SAME typed,
        documented exceptions a real backend's retry-then-raise contract would
        (``asyncio.TimeoutError`` / ``ProviderError`` -- ``KNOWN_PROVIDER_FAILURE_EXCEPTIONS``).
        ``Percept._safe_timed_judge`` (agent.py) catches ANY exception generically -- it
        does not need to special-case these two types -- but this pins the specific raised shape the
        probe drives through that boundary, at the source (``FakeJudge.judge`` itself, bypassing
        ``Percept``), so a future change to FakeJudge's injection surface is caught here too."""
        for mode in ("timeout", "provider_error"):
            with self.subTest(mode=mode):
                judge = FakeJudge(
                    seed=0, latency=LatencyProfile(kind="fixed", value_s=0.0),
                    errors=ErrorInjectionConfig(rate=1.0, modes=(mode,)), simulate_sleep=False,
                )
                with self.assertRaises(KNOWN_PROVIDER_FAILURE_EXCEPTIONS) as caught:
                    asyncio.run(judge.judge("the pot is boiling over", "frame-token"))
                self.assertIsInstance(caught.exception, KNOWN_PROVIDER_FAILURE_EXCEPTIONS)


# -- TODO hook: point the SAME assertions at each real shipped adapter -----------------------------
#
# This scaffold exercises ONLY FakeJudge (no real vendor call, no key, ever). To extend
# P-JUDGE-SAFE-1 to a real shipped adapter (anthropic / gemini / openrouter) WITHOUT touching this
# file's structure:
#
#   1. Add a `vision=` factory that wraps the real backend's client so its OWN transport failures
#      (HTTP timeout, 5xx, a truncated/garbled JSON response body) are injectable the same way
#      FakeJudge injects them -- e.g. a thin subclass or monkeypatch of the backend's `_call`/
#      `_client` boundary (see percept/backends/anthropic.py, gemini.py, openrouter.py) that raises/
#      returns a bad shape on cue, mirroring FakeJudge's `ErrorInjectionConfig` surface.
#   2. Extend `KNOWN_PROVIDER_FAILURE_EXCEPTIONS` with that adapter's own typed transport exception
#      (e.g. the anthropic SDK's `APITimeoutError`/`APIStatusError`, mirroring how
#      `percept.backends._resilience.with_retry` already re-raises it after retries).
#   3. Reuse `_build_agent`/`_drive_one_frame`/`TestJudgeSafeNeverFires` UNCHANGED against the new
#      `_ProbeVision`-wrapped real backend -- the assertions do not know or care which Vision
#      implementation they're driving.
#
# Do NOT add a live-key-dependent test to this module -- that belongs in a separate, opt-in,
# credential-gated suite (see how test_anthropic_backend.py / test_gemini_backend.py already skip
# without a key) so `pytest` stays keyless and CI-safe by default.


if __name__ == "__main__":
    unittest.main()
