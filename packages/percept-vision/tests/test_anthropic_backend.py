"""Hardened Claude vision judge — resilience + prompt-cache contract, L1 with a fake client.

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
from percept.backends.anthropic import AnthropicVision, _parse_verdict, production_judge


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

    def test_event_chain_prompt_round_trips_phase_and_hides_control_keys(self):
        response = _resp(
            '{"satisfied": true, "confidence": 0.93, "evidence": "visible change", '
            '"action_present": true, "phase": "during"}'
        )
        client = _FakeClient(response=response)
        verdict = asyncio.run(
            AnthropicVision(client=client).judge(
                "one discrete pointing occurrence",
                b"current",
                frames=[b"before"],
                context={
                    "event_chain": True,
                    "_percept_frames_are_earlier": True,
                },
            )
        )
        self.assertEqual((verdict.phase, verdict.action_present), ("during", True))
        user_content = client.calls[0]["messages"][0]["content"]
        prompt = " ".join(
            block["text"] for block in user_content if block.get("type") == "text"
        )
        self.assertIn("pre (not begun)", prompt)
        self.assertIn('"action_present", "phase"', prompt)
        self.assertNotIn("_percept_frames_are_earlier", prompt)


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


class VerdictParseSalvage(unittest.TestCase):
    """A thinking model can spend its budget and truncate the STRICT-JSON verdict mid-``evidence``.
    The decisive fields (``satisfied``/``confidence``) come first and survive — recover them instead
    of discarding a real read as ``unparsed`` -> confidence 0.0 -> a spurious ASK. Shared by every
    backend that parses via ``anthropic._parse_verdict`` (claude/gemini/openrouter)."""

    def test_truncated_mid_evidence_recovers_the_verdict(self):
        v = _parse_verdict('{"satisfied": false, "confidence": 0.4, "evidence": "While a')
        self.assertFalse(v.satisfied)
        self.assertEqual(v.confidence, 0.4)  # NOT 0.0 — the read is recovered, no false ASK
        self.assertEqual(v.evidence, "While a")

    def test_truncated_at_evidence_key_recovers_satisfied_and_confidence(self):
        v = _parse_verdict('```json\n{"satisfied": false, "confidence": 0.8, "evidence":')
        self.assertFalse(v.satisfied)
        self.assertEqual(v.confidence, 0.8)

    def test_wellformed_and_fenced_replies_are_unchanged(self):
        v = _parse_verdict('{"satisfied": true, "confidence": 0.92, "evidence": "running"}')
        self.assertTrue(v.satisfied)
        self.assertEqual(v.confidence, 0.92)
        self.assertEqual(v.evidence, "running")
        fenced = _parse_verdict('```json\n{"satisfied": true, "confidence": 0.9, "evidence": "x"}\n```')
        self.assertTrue(fenced.satisfied)
        self.assertEqual(fenced.confidence, 0.9)

    def test_out_of_range_confidence_is_clamped(self):
        self.assertEqual(_parse_verdict('{"satisfied": true, "confidence": 1.4, "evidence": "z').confidence, 1.0)

    def test_unrecoverable_reply_stays_conservative_ASK(self):
        # No decisive fields -> the invariant-1 default (0.0 -> the gate ASKs, never a confident-wrong fire).
        v = _parse_verdict("I cannot answer that from this image.")
        self.assertFalse(v.satisfied)
        self.assertEqual(v.confidence, 0.0)
        self.assertTrue(v.evidence.startswith("unparsed:"))

    def test_satisfied_without_confidence_stays_conservative(self):
        # A verdict needs BOTH decisive fields; missing confidence -> conservative default, not a guess.
        self.assertEqual(_parse_verdict('{"satisfied": true, "evidence": "...').confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
