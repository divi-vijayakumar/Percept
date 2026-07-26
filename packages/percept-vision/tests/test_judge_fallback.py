"""FallbackVision wraps a primary Vision backend with an optional secondary, fail-closed.

The judge seam must NEVER raise: a primary outage (including a tripped CircuitBreaker's
``CircuitOpen``) hands off to an optional fallback backend, and if both are unavailable the seam
degrades to a typed low-confidence ``Verdict`` (``satisfied=False, confidence=0.0``) rather than
letting a backend exception cross into the agent's hot path. No network: fakes only.
"""
from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Observation, Verdict
from percept.backends._resilience import CircuitOpen
from percept.backends.fallback import FallbackVision, wrap_if_configured


class _OkVision:
    """Always succeeds with a scripted verdict/detections."""

    def __init__(self, verdict: Verdict | None = None, observations: list | None = None) -> None:
        self._verdict = verdict or Verdict(satisfied=True, confidence=0.9, evidence="ok")
        self._observations = observations or []
        self.judge_calls = 0
        self.detect_calls = 0

    async def judge(self, condition, frame, **kwargs):
        self.judge_calls += 1
        return self._verdict

    async def detect(self, frame, goals):
        self.detect_calls += 1
        return self._observations

    async def detect_events(self, frames, goals):
        return []


class _RaisingVision:
    """Always raises the given exception (default: a plain RuntimeError)."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("boom")

    async def judge(self, condition, frame, **kwargs):
        raise self._exc

    async def detect(self, frame, goals):
        raise self._exc

    async def detect_events(self, frames, goals):
        raise self._exc


def _run(coro):
    return asyncio.run(coro)


class FallbackVisionJudgeTests(unittest.TestCase):
    def test_primary_success_passes_through_unchanged(self):
        primary = _OkVision(Verdict(satisfied=True, confidence=0.8, evidence="primary said yes"))
        fallback = _OkVision(Verdict(satisfied=False, confidence=0.1, evidence="fallback said no"))
        fv = FallbackVision(primary, fallback)
        v = _run(fv.judge("x?", b"frame"))
        self.assertEqual(v.evidence, "primary said yes")
        self.assertEqual(primary.judge_calls, 1)
        self.assertEqual(fallback.judge_calls, 0)  # never touched when primary is healthy

    def test_primary_error_delegates_to_fallback(self):
        primary = _RaisingVision(RuntimeError("vendor 500"))
        fallback = _OkVision(Verdict(satisfied=True, confidence=0.95, evidence="fallback saved it"))
        fv = FallbackVision(primary, fallback)
        v = _run(fv.judge("x?", b"frame"))
        self.assertTrue(v.satisfied)
        self.assertEqual(v.evidence, "fallback saved it")

    def test_circuit_open_on_primary_delegates_to_fallback(self):
        # CircuitOpen is just another exception here — no special-casing needed, but this pins it
        # explicitly since a tripped breaker is the primary real-world trigger for this seam.
        primary = _RaisingVision(CircuitOpen("circuit breaker open"))
        fallback = _OkVision(Verdict(satisfied=True, confidence=0.7, evidence="fallback took over"))
        fv = FallbackVision(primary, fallback)
        v = _run(fv.judge("x?", b"frame"))
        self.assertEqual(v.evidence, "fallback took over")

    def test_primary_error_no_fallback_degrades_to_low_confidence_verdict(self):
        primary = _RaisingVision(RuntimeError("no fallback configured"))
        fv = FallbackVision(primary, fallback=None)
        v = _run(fv.judge("x?", b"frame"))
        self.assertIsInstance(v, Verdict)
        self.assertFalse(v.satisfied)
        self.assertEqual(v.confidence, 0.0)
        self.assertIn("unavailable", v.evidence)

    def test_both_primary_and_fallback_fail_never_raises(self):
        primary = _RaisingVision(RuntimeError("primary down"))
        fallback = _RaisingVision(RuntimeError("fallback down too"))
        fv = FallbackVision(primary, fallback)
        v = _run(fv.judge("x?", b"frame"))  # must NOT raise — fail-closed all the way
        self.assertIsInstance(v, Verdict)
        self.assertFalse(v.satisfied)
        self.assertEqual(v.confidence, 0.0)

    def test_kwargs_forwarded_to_primary(self):
        class _RecordingVision(_OkVision):
            async def judge(self, condition, frame, **kwargs):
                self.seen = (condition, frame, kwargs)
                return await super().judge(condition, frame, **kwargs)

        primary = _RecordingVision()
        fv = FallbackVision(primary)
        goal_context = {"plate": "7XYZ123"}
        _run(fv.judge("cond", b"f", roi=None, exemplar=b"ex", corrections={"a": "b"},
                      frames=[b"a", b"f"], audio=b"aud", context=goal_context))
        condition, frame, kwargs = primary.seen
        self.assertEqual(condition, "cond")
        self.assertEqual(frame, b"f")
        self.assertEqual(kwargs["exemplar"], b"ex")
        self.assertEqual(kwargs["corrections"], {"a": "b"})
        self.assertEqual(kwargs["frames"], [b"a", b"f"])
        self.assertEqual(kwargs["audio"], b"aud")
        self.assertEqual(kwargs["context"], goal_context)


class FallbackVisionDetectTests(unittest.TestCase):
    def test_detect_passes_through_on_success(self):
        obs = [Observation(label="cup", box=None, confidence=0.9)]  # box unused by this test
        primary = _OkVision(observations=obs)
        fv = FallbackVision(primary)
        out = _run(fv.detect(b"frame", [Goal(id="g", condition="c", say="s")]))
        self.assertEqual(out, obs)

    def test_detect_delegates_to_fallback_on_primary_error(self):
        obs = [Observation(label="cup", box=None, confidence=0.9)]
        primary = _RaisingVision()
        fallback = _OkVision(observations=obs)
        fv = FallbackVision(primary, fallback)
        out = _run(fv.detect(b"frame", []))
        self.assertEqual(out, obs)

    def test_detect_degrades_to_empty_list_never_raises(self):
        primary = _RaisingVision()
        fallback = _RaisingVision()
        fv = FallbackVision(primary, fallback)
        out = _run(fv.detect(b"frame", []))
        self.assertEqual(out, [])

    def test_detect_events_degrades_to_empty_list_never_raises(self):
        fv = FallbackVision(_RaisingVision(), _RaisingVision())
        out = _run(fv.detect_events([b"a", b"b"], []))
        self.assertEqual(out, [])


class WrapIfConfiguredTests(unittest.TestCase):
    def test_no_vision_fallback_setting_returns_primary_unchanged(self):
        from types import SimpleNamespace

        primary = _OkVision()
        settings = SimpleNamespace(models=SimpleNamespace())  # no vision_fallback attr at all
        out = wrap_if_configured(primary, settings)
        self.assertIs(out, primary)

    def test_none_vision_fallback_returns_primary_unchanged(self):
        from types import SimpleNamespace

        primary = _OkVision()
        settings = SimpleNamespace(models=SimpleNamespace(vision_fallback=None))
        out = wrap_if_configured(primary, settings)
        self.assertIs(out, primary)

    def test_configured_vision_fallback_wraps_via_injected_resolver(self):
        from types import SimpleNamespace

        primary = _RaisingVision()
        fallback = _OkVision(Verdict(satisfied=True, confidence=0.9, evidence="from resolved fallback"))
        settings = SimpleNamespace(models=SimpleNamespace(vision_fallback="gemini"))
        calls = []

        def _resolver(name, s):
            calls.append((name, s))
            return fallback

        out = wrap_if_configured(primary, settings, make_vision=_resolver)
        self.assertIsInstance(out, FallbackVision)
        v = _run(out.judge("x?", b"frame"))
        self.assertEqual(v.evidence, "from resolved fallback")
        self.assertEqual(calls, [("gemini", settings)])


if __name__ == "__main__":
    unittest.main()
