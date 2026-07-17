"""Probe-harness ``fake_judge`` (Viability A12) — determinism + error-injection, L1 no-key.

Two things this proves, both required before P-BOUND-1/latency/delivery probes (or
P-JUDGE-SAFE-1/P-QUAL-1 later) can trust this double:

  * same seed + same call sequence -> identical latencies AND identical verdicts (reproducible
    architecture-isolation, not a second source of randomness in the spike's numbers).
  * a configured error rate reliably produces the configured injected failure mode (timeout /
    provider_error raise; malformed / partial return a non-Verdict shape) so the runtime's
    fail-closed handling has something deterministic to exercise.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.backends.fake_judge import (
    ErrorInjectionConfig,
    FakeJudge,
    LatencyProfile,
    ProviderError,
)
from percept.contracts import Verdict


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_calls_yields_identical_latencies_and_verdicts(self):
        async def _drive(judge: FakeJudge) -> list[Verdict]:
            out = []
            for token in ("sip-coffee", "sip-water", "empty", "sip-coffee"):
                out.append(await judge.judge("drinking coffee?", token))
            return out

        latency = LatencyProfile(kind="uniform", low_s=0.01, high_s=0.05)
        j1 = FakeJudge(seed=42, latency=latency, simulate_sleep=False)
        j2 = FakeJudge(seed=42, latency=latency, simulate_sleep=False)

        out1 = asyncio.run(_drive(j1))
        out2 = asyncio.run(_drive(j2))

        self.assertEqual(j1.latencies, j2.latencies)
        self.assertTrue(all(lat > 0 for lat in j1.latencies))
        self.assertEqual(out1, out2)
        self.assertEqual([v.satisfied for v in out1], [True, False, False, True])

    def test_different_seed_yields_different_latencies(self):
        latency = LatencyProfile(kind="lognormal", mu=-3.0, sigma=0.5)
        j1 = FakeJudge(seed=1, latency=latency, simulate_sleep=False)
        j2 = FakeJudge(seed=2, latency=latency, simulate_sleep=False)

        asyncio.run(j1.judge("c", "sip-coffee"))
        asyncio.run(j2.judge("c", "sip-coffee"))

        self.assertNotEqual(j1.latencies, j2.latencies)

    def test_fixed_latency_profile_is_exact(self):
        j = FakeJudge(seed=0, latency=LatencyProfile(kind="fixed", value_s=0.25), simulate_sleep=False)
        asyncio.run(j.judge("c", "sip-coffee"))
        asyncio.run(j.judge("c", "sip-coffee"))
        self.assertEqual(j.latencies, [0.25, 0.25])

    def test_simulate_sleep_actually_waits(self):
        j = FakeJudge(seed=0, latency=LatencyProfile(kind="fixed", value_s=0.02), simulate_sleep=True)

        async def _drive():
            start = asyncio.get_event_loop().time()
            await j.judge("c", "sip-coffee")
            return asyncio.get_event_loop().time() - start

        elapsed = asyncio.run(_drive())
        self.assertGreaterEqual(elapsed, 0.02)


class TestErrorInjection(unittest.TestCase):
    def test_rate_zero_never_injects(self):
        j = FakeJudge(seed=7, errors=ErrorInjectionConfig(rate=0.0), simulate_sleep=False)

        async def _drive():
            for _ in range(20):
                await j.judge("c", "sip-coffee")

        asyncio.run(_drive())
        self.assertEqual(j.injected_modes, [None] * 20)

    def test_rate_one_timeout_always_raises_timeout_error(self):
        j = FakeJudge(
            seed=3,
            errors=ErrorInjectionConfig(rate=1.0, modes=("timeout",)),
            simulate_sleep=False,
        )
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(j.judge("c", "sip-coffee"))
        self.assertEqual(j.injected_modes, ["timeout"])

    def test_rate_one_provider_error_always_raises_provider_error(self):
        j = FakeJudge(
            seed=3,
            errors=ErrorInjectionConfig(rate=1.0, modes=("provider_error",)),
            simulate_sleep=False,
        )
        with self.assertRaises(ProviderError):
            asyncio.run(j.judge("c", "sip-coffee"))

    def test_rate_one_malformed_returns_non_verdict_shape(self):
        j = FakeJudge(
            seed=3,
            errors=ErrorInjectionConfig(rate=1.0, modes=("malformed",)),
            simulate_sleep=False,
        )
        out = asyncio.run(j.judge("c", "sip-coffee"))
        self.assertNotIsInstance(out, Verdict)
        self.assertIsInstance(out, dict)

    def test_rate_one_partial_returns_dict_missing_required_fields(self):
        j = FakeJudge(
            seed=3,
            errors=ErrorInjectionConfig(rate=1.0, modes=("partial",)),
            simulate_sleep=False,
        )
        out = asyncio.run(j.judge("c", "sip-coffee"))
        self.assertIsInstance(out, dict)
        self.assertNotIn("confidence", out)

    def test_injected_error_rate_is_reproducible_given_same_seed(self):
        cfg = ErrorInjectionConfig(rate=0.5, modes=("timeout", "provider_error", "malformed", "partial"))

        async def _drive(judge: FakeJudge) -> list[str | None]:
            modes = []
            for _ in range(30):
                try:
                    await judge.judge("c", "sip-coffee")
                    modes.append(None)
                except (asyncio.TimeoutError, ProviderError):
                    modes.append(judge.injected_modes[-1])
            return modes

        j1 = FakeJudge(seed=99, errors=cfg, simulate_sleep=False)
        j2 = FakeJudge(seed=99, errors=cfg, simulate_sleep=False)
        seq1 = asyncio.run(_drive(j1))
        seq2 = asyncio.run(_drive(j2))

        self.assertEqual(seq1, seq2)
        self.assertEqual(j1.injected_modes, j2.injected_modes)
        # rate=0.5 over 30 calls should inject roughly a third to two thirds of the time
        # (deterministic value pinned via the seed, just sanity-bounding it here).
        injected_count = sum(1 for m in j1.injected_modes if m is not None)
        self.assertTrue(5 <= injected_count <= 25)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            ErrorInjectionConfig(rate=1.5)
        with self.assertRaises(ValueError):
            ErrorInjectionConfig(modes=("not-a-real-mode",))
        with self.assertRaises(ValueError):
            LatencyProfile(kind="not-a-real-kind")


class TestDropsIntoJudgeSeam(unittest.TestCase):
    def test_clean_call_returns_a_real_verdict_with_expected_fields(self):
        j = FakeJudge(seed=0, simulate_sleep=False)
        out = asyncio.run(j.judge("drinking coffee?", "sip-coffee"))
        self.assertIsInstance(out, Verdict)
        self.assertTrue(out.satisfied)
        self.assertEqual(out.confidence, 0.9)
        self.assertTrue(out.evidence)

    def test_custom_script_and_vision_delegation_still_works(self):
        script = {"custom-token": Verdict(satisfied=True, confidence=0.77, evidence="custom")}
        j = FakeJudge(seed=0, script=script, simulate_sleep=False)
        out = asyncio.run(j.judge("c", "custom-token"))
        self.assertEqual(out, Verdict(satisfied=True, confidence=0.77, evidence="custom"))


if __name__ == "__main__":
    unittest.main()
