"""— cost must be a MEASURED number on a real run, or ``—``, but never a false zero.

The failure this file exists to stop: a client run that provably spent 134,946 input + 5,231 output
tokens across 108 judge calls reported ``{"input_tokens": 0.0, "output_tokens": 0.0, "usd": 0.0,
"usage_reported_calls": 0}``. The token counts existed in the Funnel; nothing carried them to the
trace/cost surface, so a ``cost_usd`` bar went green on a run that spent real money.

The rule under test, in three parts:

1. a real offline run whose judge reports usage produces NON-ZERO tokens and cost in
   ``TraceSummary`` (the wire actually exists end-to-end: judge -> trace -> summary);
2. a run whose judge reports nothing produces ``None`` ("—"), NEVER ``0.0`` — unknown is safe, a
   zero is a wrong number that a bar can pass;
3. tokens against a model absent from ``PRICE_TABLE`` produce ``None`` dollars, never a
   ``DEFAULT_PRICE``-derived figure dressed up as a real price.

Everything here is OFFLINE and KEYLESS: scripted ``FakeVision`` judges, list sources, no network.
"""

from __future__ import annotations

import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.runtime import run
from percept.slo import PRICE_TABLE, price_usage, price_usage_or_none
from percept.tracing import RunTraceManifest, TraceRecorder, TracePolicy

_W, _H = 8, 8
_STILL = gray_frame(_W, _H, 0)
_MOVED = gray_frame(_W, _H, 255)

# (t, gate_frame, judge_frame) — the same O1 fixture the rest of the trace tests drive: the motion
# gate escalates on still->moved and moved->still, so this stream really spends judge calls.
_SOURCE = [
    (0.0, _STILL, "clear"),
    (1.0, _MOVED, "intruder"),
    (2.0, _STILL, "clear"),
]
# A stream that NEVER moves: tier-0 suppresses everything, so zero judge calls are spent.
_STILL_SOURCE = [(float(i), _STILL, "clear") for i in range(4)]

_PRICED_MODEL = "claude-opus-4-8"  # in percept.slo.PRICE_TABLE
_UNPRICED_MODEL = "some-vendor/uncatalogued-vlm-9"  # deliberately NOT in the table

_IN_TOKENS, _OUT_TOKENS = 1249.0, 48.0


class _MeteredVision(FakeVision):
    """A scripted judge that ALSO reports its token usage, like a real backend does.

    ``Verdict.usage`` is the existing, frozen-contract-compatible annotation hook
    (``percept.observability.judge_usage`` reads it) — this fake fills it exactly the way
    ``percept/backends/anthropic.py`` does, so the test drives the production wire, not a shim.
    """

    def __init__(self, *a, model: str | None = _PRICED_MODEL, usage: dict | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        self.model = model  # what Percept._judge_model() reads off the adapter
        self._usage = {"input_tokens": _IN_TOKENS, "output_tokens": _OUT_TOKENS} if usage is None else usage
        self.calls = 0

    async def judge(self, *a, **kw) -> Verdict:
        verdict = await super().judge(*a, **kw)
        self.calls += 1
        return Verdict(
            satisfied=verdict.satisfied, confidence=verdict.confidence, evidence=verdict.evidence,
            usage=dict(self._usage),
        )


def _core(vision) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert"))
    return core


def _recorder() -> TraceRecorder:
    return TraceRecorder(RunTraceManifest.create(source="camera", policy=TracePolicy(level="full")))


async def _traced_run(vision, source=_SOURCE):
    """One real offline run through ``percept.runtime.run`` with a TraceRecorder attached."""
    core = _core(vision)
    rec = _recorder()
    async for _ in run(core, source=source, on_trace=rec.tracer(), on_health=rec.health_sink()):
        pass
    rec.close()
    return rec.summary()


class RunPathCostTests(unittest.IsolatedAsyncioTestCase):
    """THE DELIVERABLE: cost on a real run, end to end."""

    async def test_metered_run_reports_nonzero_tokens_and_cost(self):
        vision = _MeteredVision()
        summary = await _traced_run(vision)

        self.assertGreater(vision.calls, 0, "fixture must actually spend judge calls")
        # every judge call the run made reached the cost surface — no silent metering gap
        self.assertEqual(summary.usage_reported_calls, vision.calls)
        self.assertEqual(summary.judge_completed, vision.calls)

        self.assertEqual(summary.input_tokens, _IN_TOKENS * vision.calls)
        self.assertEqual(summary.output_tokens, _OUT_TOKENS * vision.calls)

        # NON-ZERO dollars, priced from the real table — this is the number that used to be 0.0
        expected = price_usage(
            {"input_tokens": _IN_TOKENS, "output_tokens": _OUT_TOKENS}, model=_PRICED_MODEL,
        ) * vision.calls
        self.assertIsNotNone(summary.cost_usd)
        self.assertGreater(summary.cost_usd, 0.0)
        self.assertAlmostEqual(summary.cost_usd, expected, places=12)
        self.assertEqual(summary.priced_calls, vision.calls)

        # and it survives the JSON surface a client actually reads
        cost = summary.to_dict()["cost"]
        self.assertGreater(cost["usd"], 0.0)
        self.assertEqual(cost["usage_reported_calls"], vision.calls)
        self.assertNotEqual(cost["text"], "—")

    async def test_unmetered_run_reports_none_never_zero(self):
        """A judge that reports no usage: spend is UNKNOWN. The old behaviour printed 0.0 here."""
        vision = FakeVision(
            script={
                "intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person"),
                "clear": Verdict(satisfied=False, confidence=0.95, evidence="empty"),
            }
        )
        summary = await _traced_run(vision)

        self.assertGreater(summary.judge_completed, 0, "the run really did spend judge calls")
        self.assertEqual(summary.usage_reported_calls, 0)
        self.assertIsNone(summary.cost_usd)
        self.assertIsNone(summary.input_tokens)
        self.assertIsNone(summary.output_tokens)
        # the exact regression: a cost bar must NOT be able to read this as free
        self.assertNotEqual(summary.cost_usd, 0.0)
        self.assertEqual(summary.cost_text(), "—")
        self.assertEqual(summary.tokens_text(), "—")
        self.assertIsNone(summary.dollars_per_stream_hour)
        self.assertIsNone(summary.to_dict()["cost"]["usd"])

    async def test_unpriceable_model_reports_tokens_but_no_dollars(self):
        """Tokens are known, the model is not in PRICE_TABLE: dollars are ``None``, not a guess.

        Never a DEFAULT_PRICE-derived figure presented as a real price — a client computed
        ~$16/stream-hour that way and correctly refused to report it.
        """
        self.assertNotIn(_UNPRICED_MODEL, PRICE_TABLE)
        vision = _MeteredVision(model=_UNPRICED_MODEL)
        summary = await _traced_run(vision)

        self.assertEqual(summary.usage_reported_calls, vision.calls)
        self.assertEqual(summary.input_tokens, _IN_TOKENS * vision.calls)  # what we know, we report
        self.assertIsNone(summary.cost_usd)                                # what we don't, we don't
        self.assertEqual(summary.priced_calls, 0)
        self.assertEqual(summary.cost_text(), "—")
        # and specifically NOT the DEFAULT_PRICE figure price_usage() would have produced
        guess = price_usage({"input_tokens": _IN_TOKENS}, model=_UNPRICED_MODEL)
        self.assertGreater(guess, 0.0)
        self.assertNotEqual(summary.cost_usd, guess)

    async def test_backend_reported_cost_is_ground_truth_even_without_a_price_table_entry(self):
        """OpenRouter-style: the backend bills its own ``cost_usd``. That number wins outright."""
        vision = _MeteredVision(
            model=_UNPRICED_MODEL,
            usage={"input_tokens": _IN_TOKENS, "output_tokens": _OUT_TOKENS, "cost_usd": 0.0123},
        )
        summary = await _traced_run(vision)
        self.assertAlmostEqual(summary.cost_usd, 0.0123 * vision.calls, places=12)
        self.assertEqual(summary.priced_calls, vision.calls)

    async def test_a_run_that_spent_no_judge_call_reports_a_measured_zero(self):
        """The ONE licensed zero: no judge call was made, so $0 is measured, not assumed."""
        vision = _MeteredVision()
        summary = await _traced_run(vision, source=_STILL_SOURCE)
        self.assertEqual(vision.calls, 0)
        self.assertEqual(summary.judge_completed, 0)
        self.assertEqual(summary.cost_usd, 0.0)
        self.assertEqual(summary.input_tokens, 0.0)

    async def test_dollars_per_stream_hour_is_derived_from_the_run(self):
        """price_usage reaches a $/stream-hour rate on the RUN path, not just in the SLO harness."""
        vision = _MeteredVision()
        summary = await _traced_run(vision)
        rate = summary.dollars_per_stream_hour
        self.assertIsNotNone(rate)
        self.assertGreater(rate, 0.0)
        self.assertAlmostEqual(
            rate, summary.cost_usd / (summary.watch_duration_s / 3600.0), places=9
        )


class PricingRuleTests(unittest.TestCase):
    """The pricing rule itself, in isolation."""

    def test_unknown_model_prices_to_none(self):
        self.assertIsNone(price_usage_or_none({"input_tokens": 1_000_000.0}, model=_UNPRICED_MODEL))
        self.assertIsNone(price_usage_or_none({"input_tokens": 1_000_000.0}, model=None))

    def test_known_model_prices_its_tokens(self):
        got = price_usage_or_none({"input_tokens": 1_000_000.0}, model=_PRICED_MODEL)
        self.assertAlmostEqual(got, PRICE_TABLE[_PRICED_MODEL].input_per_mtok, places=12)

    def test_explicit_cost_wins_over_the_table(self):
        got = price_usage_or_none(
            {"input_tokens": 1_000_000.0, "cost_usd": 0.42}, model=_PRICED_MODEL
        )
        self.assertEqual(got, 0.42)


class DirectDriverSeamTests(unittest.TestCase):
    """``record_verdict_usage`` — the seam for a caller that drives ``perceive_judged`` itself."""

    def test_record_verdict_usage_prices_through_the_same_rule(self):
        rec = _recorder()
        v = Verdict(satisfied=True, confidence=0.9, evidence="",
                    usage={"input_tokens": _IN_TOKENS, "output_tokens": _OUT_TOKENS})
        self.assertIsNotNone(rec.record_verdict_usage(v, source_ts=1.0, model=_PRICED_MODEL))
        rec.close()
        summary = rec.summary()
        self.assertEqual(summary.usage_reported_calls, 1)
        self.assertAlmostEqual(
            summary.cost_usd,
            price_usage({"input_tokens": _IN_TOKENS, "output_tokens": _OUT_TOKENS}, model=_PRICED_MODEL),
            places=12,
        )

    def test_record_verdict_usage_without_a_model_reports_tokens_only(self):
        rec = _recorder()
        v = Verdict(satisfied=True, confidence=0.9, evidence="", usage={"input_tokens": 10.0})
        rec.record_verdict_usage(v, source_ts=1.0)
        rec.close()
        summary = rec.summary()
        self.assertEqual(summary.input_tokens, 10.0)
        self.assertIsNone(summary.cost_usd)

    def test_unannotated_verdict_records_nothing(self):
        rec = _recorder()
        self.assertIsNone(rec.record_verdict_usage(Verdict(satisfied=False, confidence=0.1), source_ts=1.0))
        rec.close()
        self.assertEqual(rec.summary().usage_reported_calls, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
