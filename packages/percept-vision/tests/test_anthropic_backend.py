"""Hardened Claude vision judge (E1) — resilience + prompt-cache contract, L1 with a fake client.

No real key, no real SDK: an injected fake ``client`` stands in for ``anthropic.Anthropic`` and
records what was sent. These pin the production-judge hardening:

  * a transient cloud error is retried and the judge still returns a ``Verdict`` (retry + breaker);
  * the stable per-concern instruction prefix is sent as a ``system`` block carrying an ephemeral
    ``cache_control`` breakpoint (prompt caching — the per-frame call only re-bills the suffix);
  * the explicit per-call timeout is applied via the SDK's ``with_options`` when present;
  * a run of failures trips the breaker OPEN (short-circuits instead of hammering the hot path).
"""

import asyncio
import unittest

from percept.adapters import Verdict
from percept.backends._resilience import CircuitOpen
from percept.backends.anthropic import AnthropicVision, production_judge


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        if self._owner.fail_first and len(self._owner.calls) == 1:
            raise RuntimeError("transient 529 overloaded")
        return self._owner.response


class _FakeClient:
    """Minimal stand-in for ``anthropic.Anthropic``: records ``messages.create`` kwargs and can be
    scripted to fail the first attempt. ``with_options`` returns a timeout-bound view (the same
    recorder) so the adapter's explicit-timeout path is exercised without a real SDK."""

    def __init__(self, *, response, fail_first=False):
        self.response = response
        self.fail_first = fail_first
        self.calls = []
        self.timeout = None
        self.messages = _Messages(self)

    def with_options(self, *, timeout=None, **_):
        self.timeout = timeout
        return self  # same recorder; the adapter only needs a client exposing messages.create


def _resp(text):
    """An Anthropic-shaped Messages response: ``.content`` is a list of text blocks."""
    return {"content": [{"type": "text", "text": text}]}


_VERDICT_JSON = '{"satisfied": true, "confidence": 0.91, "evidence": "mug at lips"}'


class TestRetryReturnsVerdict(unittest.TestCase):
    def test_transient_error_is_retried_then_succeeds(self):
        client = _FakeClient(response=_resp(_VERDICT_JSON), fail_first=True)
        v = AnthropicVision(client=client)
        out = asyncio.run(v.judge("is the user drinking?", b"\x89PNG-bytes"))
        self.assertIsInstance(out, Verdict)
        self.assertTrue(out.satisfied)
        self.assertAlmostEqual(out.confidence, 0.91)
        self.assertEqual(len(client.calls), 2)  # one failure, one success — proves the retry

    def test_clean_call_makes_a_single_request(self):
        client = _FakeClient(response=_resp(_VERDICT_JSON))
        v = AnthropicVision(client=client)
        out = asyncio.run(v.judge("is the user drinking?", b"\x89PNG-bytes"))
        self.assertTrue(out.satisfied)
        self.assertEqual(len(client.calls), 1)


class TestPromptCachePrefix(unittest.TestCase):
    def test_system_prefix_carries_ephemeral_cache_control(self):
        client = _FakeClient(response=_resp(_VERDICT_JSON))
        v = AnthropicVision(client=client)
        asyncio.run(v.judge("is the user drinking?", b"\x89PNG-bytes"))
        system = client.calls[0]["system"]
        # The stable instruction prefix is sent as a block list (not a bare string) so it can carry
        # a cache breakpoint — the per-frame variable content stays in messages, after the prefix.
        self.assertIsInstance(system, list)
        self.assertEqual(system[-1]["cache_control"], {"type": "ephemeral"})
        self.assertIn("perception module", system[-1]["text"])

    def test_variable_frame_content_is_not_in_the_cached_prefix(self):
        client = _FakeClient(response=_resp(_VERDICT_JSON))
        v = AnthropicVision(client=client)
        asyncio.run(v.judge("is the user drinking?", b"\x89PNG-bytes"))
        # The narrow per-frame question lives in the user turn, never in the cached system prefix.
        system_text = client.calls[0]["system"][-1]["text"]
        self.assertNotIn("is the user drinking?", system_text)
        user_content = client.calls[0]["messages"][0]["content"]
        self.assertTrue(any("is the user drinking?" in str(b) for b in user_content))


class TestTimeoutApplied(unittest.TestCase):
    def test_explicit_timeout_passed_through_with_options(self):
        client = _FakeClient(response=_resp(_VERDICT_JSON))
        v = AnthropicVision(client=client)
        asyncio.run(v.judge("is the user drinking?", b"\x89PNG-bytes"))
        self.assertIsNotNone(client.timeout)  # the explicit per-call timeout was applied
        self.assertGreater(client.timeout, 0)


class TestProductionJudgeWrapper(unittest.TestCase):
    def test_returns_hardened_judge_that_caches_and_retries(self):
        client = _FakeClient(response=_resp(_VERDICT_JSON), fail_first=True)
        v = production_judge(client=client)
        self.assertIsInstance(v, AnthropicVision)
        out = asyncio.run(v.judge("is the user drinking?", b"\x89PNG-bytes"))
        self.assertTrue(out.satisfied)
        self.assertEqual(len(client.calls), 2)  # retried the transient failure
        self.assertEqual(
            client.calls[-1]["system"][-1]["cache_control"], {"type": "ephemeral"}
        )


class _AlwaysFailMessages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        raise RuntimeError("hard down")


class TestBreakerTripsOpen(unittest.TestCase):
    def test_run_of_failures_opens_the_breaker(self):
        # Always-failing client: each judge() exhausts its retries and feeds the breaker; once it
        # has seen enough failures the breaker goes OPEN and the next call short-circuits with
        # CircuitOpen instead of hammering the hot path.
        client = _FakeClient(response=_resp(_VERDICT_JSON))
        client.messages = _AlwaysFailMessages(client)
        v = AnthropicVision(client=client)

        async def _drive():
            # Default breaker trips at 5 failures; each judge() makes 3 attempts. Two judge() calls
            # (6 failures) cross the threshold, after which the breaker is OPEN.
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    await v.judge("is the user drinking?", b"\x89PNG-bytes")
            with self.assertRaises(CircuitOpen):
                await v.judge("is the user drinking?", b"\x89PNG-bytes")

        asyncio.run(_drive())


if __name__ == "__main__":
    unittest.main()
