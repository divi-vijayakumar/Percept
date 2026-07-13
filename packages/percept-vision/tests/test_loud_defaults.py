"""Loud-failure audit (AGENT-AUDIT): the dangerous defaults must fail LOUD, not silent.

Every bug in this hardening pass shares one shape — a default that runs, looks correct, and
silently does nothing. These tests pin the two dangerous defaults in ``Percept.create`` and
``run()`` so a fifth one cannot hide:

  * E3 — ``conductor="fake"`` behind a REAL vision backend. FakeConductor's default
    ``Decision(route="ignore")`` discards every standing-goal request, so a voice-authored
    "watch X" never arms. Must warn + report ``health()["degraded"]``.
  * E5 — ``regate_on_goal_change=False``. A goal armed mid-run bumps the registry revision but
    the compiled-once gate never re-arms → no rising edge → no fire, silently, for the whole
    session (the shipped realtime-demo bug). The default is now True; when it IS off and goals
    move, the runtime must say so out loud.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import unittest

import percept
from percept.adapters import Goal, Verdict
from percept.backends.fake import FakeConductor, FakeVision
from percept.edge.tier0 import gray_frame
from percept.observability import get_logger
from percept.runtime import run
from percept.runtime.worker import RuntimeHealthEvent

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)


class _RealishVision:
    """A NON-fake vision double: defined outside percept.backends.fake, so create() must treat it
    as a real backend (the class of adapter that makes a FakeConductor a silent trap)."""

    async def judge(self, *a, **k):  # pragma: no cover - never called by create()
        return Verdict(satisfied=False, confidence=0.0)

    async def detect(self, *a, **k):  # pragma: no cover
        return []


class _CaptureWarnings:
    """Attach a capturing handler to the 'percept' logger (get_logger sets propagate=False, so
    pytest caplog can't see it — capture directly)."""

    def __init__(self, name: str = "percept") -> None:
        self.records: list[logging.LogRecord] = []
        self._logger = get_logger(name)

        class _H(logging.Handler):
            def emit(_self, record):  # noqa: N805
                self.records.append(record)

        self._handler = _H(level=logging.WARNING)

    def __enter__(self):
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)

    def warnings(self) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------- E3: degraded stub conductor ----
class DegradedStubConductorTests(unittest.TestCase):
    def test_real_vision_default_conductor_is_degraded(self):
        # Real vision + the DEFAULT (fake) conductor: health() must report the degradation LOUDLY.
        a = percept.Percept.create(vision=_RealishVision(), discover_plugins=False)
        h = a.health()
        self.assertTrue(h["degraded"], "real vision + FakeConductor must report degraded=True")
        self.assertEqual(h["conductor"], "fake")
        self.assertIn("reason", h)

    def test_real_vision_fake_conductor_warns_and_emits_health(self):
        events: list = []
        with _CaptureWarnings("percept.agent") as cap:
            percept.Percept.create(
                vision=_RealishVision(), on_health=events.append, discover_plugins=False
            )
        self.assertIn("degraded_stub_conductor", events)
        self.assertTrue(
            any("degraded_stub_conductor" in w or "FakeConductor" in w for w in cap.warnings()),
            f"expected a degraded-conductor warning, got {cap.warnings()}",
        )

    def test_all_fake_is_not_degraded(self):
        # The all-fake default (the L1 test lane) must NOT be flagged — a fake vision is obviously
        # fake, so a fake conductor beside it is expected, not a trap. Guards a false positive.
        a = percept.Percept.create(vision="fake", discover_plugins=False)
        self.assertFalse(a.health()["degraded"])

    def test_real_vision_real_conductor_is_not_degraded(self):
        # A real conductor beside real vision is the correct production wiring — not degraded.
        class _RealishConductor:
            async def run(self, *a, **k):  # pragma: no cover
                from percept.adapters import Decision

                return Decision(route="ignore")

        a = percept.Percept.create(
            vision=_RealishVision(), conductor=_RealishConductor(), discover_plugins=False
        )
        self.assertFalse(a.health()["degraded"])


# ------------------------------------------------------------------ E5: regate default + health --
def _core_with_scripted_baby():
    vision = FakeVision(script={
        "baby": Verdict(satisfied=True, confidence=0.95, evidence="the baby is fuzzy"),
        "x": Verdict(satisfied=False, confidence=0.9, evidence="nothing"),
    })
    core = percept.Percept.create(vision=vision, conductor=FakeConductor(), discover_plugins=False)
    return core


async def _run_arming_midflight(*, pass_regate, regate_value=None, on_health=None):
    core = _core_with_scripted_baby()

    async def source():
        yield (0.0, STILL, "x")                       # nothing armed yet — compiles the gate once
        core.add_goal(Goal(id="fuzzy", condition="the baby is fuzzy",
                           say="Heads up — the baby's getting fuzzy."))  # bumps goals.revision
        for i in range(1, 9):
            frame = MOVED if i % 2 else STILL
            yield (float(i), frame, "baby")

    kw = {}
    if pass_regate:
        kw["regate_on_goal_change"] = regate_value
    if on_health is not None:
        kw["on_health"] = on_health
    return [inc async for inc in run(core, source=source(), **kw)]


class RegateDefaultTests(unittest.TestCase):
    def test_regate_default_is_true(self):
        default = inspect.signature(run).parameters["regate_on_goal_change"].default
        self.assertIs(default, True, "regate_on_goal_change must default to True (E5)")

    def test_default_run_rearms_and_fires(self):
        # With the flipped default, run(core) with NO regate arg must re-arm on a mid-run goal and fire.
        incidents = asyncio.run(_run_arming_midflight(pass_regate=False))
        self.assertTrue(
            any(getattr(i, "concern_id", None) == "fuzzy" for i in incidents),
            "the flipped default must re-arm the gate so a runtime-authored goal fires",
        )

    def test_regate_off_emits_goals_changed_health_event(self):
        # When regate is explicitly OFF and goals move, the runtime must fire a loud health event —
        # not silently swallow the revision change (that is exactly the silent-default shape).
        events: list = []
        asyncio.run(_run_arming_midflight(
            pass_regate=True, regate_value=False, on_health=events.append))
        kinds = [e.kind for e in events if isinstance(e, RuntimeHealthEvent)]
        self.assertIn("goals_changed_without_regate", kinds)

    def test_regate_on_does_not_emit_the_off_warning(self):
        # The health event is for the OFF path only; with regate ON there is nothing to warn about.
        events: list = []
        asyncio.run(_run_arming_midflight(
            pass_regate=True, regate_value=True, on_health=events.append))
        kinds = [e.kind for e in events if isinstance(e, RuntimeHealthEvent)]
        self.assertNotIn("goals_changed_without_regate", kinds)


if __name__ == "__main__":
    unittest.main()
