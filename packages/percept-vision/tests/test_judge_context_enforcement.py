"""Route selection consumes the judge-context registry, so an undeclared key never reaches a model.

WHAT THIS PINS. ``percept.judge_context_producers`` declares the eight public judge-context keys and
the ``_percept_`` transport prefix, but declaring is not enforcing: until selection consulted it, the
registry was a description of a rule nobody applied. An arbitrary caller key -- ``{"invoice_number":
...}``, ``{"conformance": "probe"}`` -- had no structural barrier between it and the prompt, where it
rendered under a sentence telling the model to trust it over its own read.

WHY AT SELECTION AND NOWHERE ELSE. A judge is a capability, and a capability is refused where the
route is chosen. ``percept.conformance`` deliberately passes an unregistered probe key straight into
an adapter to prove the seam accepts a caller's context; enforcement inside a backend's ``judge()``
would fail the published conformance suite. The two facts are consistent precisely because they live
at different layers, and :class:`TestTheSeamBelowThePoolIsUntouched` holds that line.

FAIL CLOSED, NO NEW PATH. A refused context empties the eligible set, which is what ``no_route``
already means: it already fails closed to an UNKNOWN verdict, already writes a receipt, and already
honours a deployment's ``terminal_outcomes``. Only ``JudgeRouteReceipt.refused_context_keys`` and the
wording distinguish the two ways of getting there.

Written on ``unittest.IsolatedAsyncioTestCase`` so BOTH runners execute it -- the pool's other tests
are pytest-style functions that ``make test``'s unittest discovery does not collect. No provider, no
network, no key.
"""

from __future__ import annotations

import asyncio
import unittest

from percept import Verdict
from percept.backends.judge_pool import (
    LATENCY_MODE_CONTEXT_KEY,
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.judge_context_producers import (
    PRIVATE_KEY_PREFIX,
    RESERVED_CONTEXT_KEYS,
)

_SHAPE_KEY = "_percept_evidence_shape"


class _Judge:
    """Records every call and the exact context it was handed."""

    def __init__(self, verdict: Verdict | None = None) -> None:
        self.verdict = verdict or Verdict(True, 0.99, "served")
        self.calls = 0
        self.contexts: list[dict | None] = []

    async def judge(self, _condition, _frame, **kwargs):
        self.calls += 1
        self.contexts.append(kwargs.get("context"))
        return self.verdict


def _cap(*shapes: str, **kwargs) -> JudgeCapability:
    kwargs.setdefault("latency_slo_s", 0.05)
    kwargs.setdefault("modes", frozenset({"reflex", "deliberate"}))
    return JudgeCapability(frozenset(shapes or ("single_image",)), **kwargs)


def _pool(*routes: JudgeRoute, **kwargs) -> JudgePool:
    kwargs.setdefault("deadline_s", 0.5)
    kwargs.setdefault("hedge_after_s", 0.0)
    return JudgePool(list(routes), **kwargs)


class TestTheDeclarationIsCheckedAtConstruction(unittest.TestCase):
    def test_the_default_is_the_registry_so_no_existing_route_had_to_change(self):
        """The decision this field turns on.

        ``frozenset()`` would have refused everything the runtime emits and broken every existing
        declaration; permissive would have left the leak open. The registry's public set admits what
        the runtime actually sends and refuses what nothing declares -- which is the leak that
        mattered -- and costs an existing constructor nothing.
        """
        self.assertEqual(_cap("single_image").context_keys, RESERVED_CONTEXT_KEYS)

    def test_a_route_declaring_a_key_nothing_emits_fails_to_construct(self):
        # A typo in a declaration is cheaper to learn here than at selection, where it presents as
        # an unexplained no_route on a request that looks perfectly ordinary.
        with self.assertRaises(ValueError) as caught:
            _cap("single_image", context_keys=frozenset({"claim_mode", "clam_mode"}))
        self.assertIn("clam_mode", str(caught.exception))
        self.assertNotIn("claim_mode", str(caught.exception))

    def test_a_non_string_key_is_refused_before_the_prefix_rule_can_be_asked(self):
        # `is_registered` calls `str.startswith`; an int member would raise AttributeError from
        # inside the registry rather than a ValueError from the declaration that is actually wrong.
        with self.assertRaises(ValueError):
            _cap("single_image", context_keys=frozenset({1}))  # type: ignore[arg-type]

    def test_narrowing_to_nothing_is_legal_because_it_is_a_real_deployment_choice(self):
        # Unlike evidence_shapes, empty is meaningful: a judge that accepts NO structured context
        # still judges frames. It is a narrowing, not a misconfiguration.
        self.assertEqual(_cap("single_image", context_keys=frozenset()).context_keys, frozenset())

    def test_a_declaration_is_normalised_to_a_frozenset_like_the_other_fields(self):
        cap = _cap("single_image", context_keys={"claim_mode", "event_chain"})
        self.assertIsInstance(cap.context_keys, frozenset)

    def test_a_private_key_may_be_declared_even_though_it_need_not_be(self):
        # Registered by prefix, so the declaration is redundant rather than wrong. Refusing it would
        # make the registry's own rule a construction error.
        cap = _cap("single_image", context_keys=frozenset({f"{PRIVATE_KEY_PREFIX}latency_mode"}))
        self.assertIn(f"{PRIVATE_KEY_PREFIX}latency_mode", cap.context_keys)


class TestSelectionRefusesWhatNothingDeclares(unittest.IsolatedAsyncioTestCase):
    async def test_an_arbitrary_caller_key_never_reaches_the_adapter(self):
        """THE test. ``conformance`` is the literal key that renders into the prompt today."""
        judge = _Judge()
        pool = _pool(JudgeRoute("only", judge, _cap("single_image")))

        verdict = await pool.judge("cond", b"frame", context={"conformance": "probe"})

        self.assertEqual(judge.calls, 0, "an unregistered key reached the provider")
        self.assertFalse(verdict.satisfied)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIn("conformance", verdict.evidence)

    async def test_one_undeclared_key_refuses_the_request_even_beside_declared_ones(self):
        # The predicate is over the whole context, not "did anything legitimate come along too".
        judge = _Judge()
        pool = _pool(JudgeRoute("only", judge, _cap("single_image")))

        await pool.judge(
            "cond",
            b"frame",
            context={"claim_mode": True, "invoice_number": "INV-2291"},
        )

        self.assertEqual(judge.calls, 0)
        self.assertEqual(pool.receipts()[-1].refused_context_keys, ("invoice_number",))

    async def test_every_registered_public_key_is_admitted_by_the_default(self):
        """The default has to admit the whole runtime, or enforcement is an outage.

        Shape is pinned so this asserts context admission only -- a truthy ``event_chain`` otherwise
        resolves the request to ``video_clip`` and the miss would be the shape's.
        """
        judge = _Judge()
        pool = _pool(JudgeRoute("only", judge, _cap("single_image")))
        context = {key: "x" for key in RESERVED_CONTEXT_KEYS}
        context[_SHAPE_KEY] = "single_image"

        verdict = await pool.judge("cond", b"frame", context=context)

        self.assertEqual(judge.calls, 1)
        self.assertTrue(verdict.satisfied)

    async def test_an_absent_or_empty_context_is_the_untouched_base_case(self):
        judge = _Judge()
        pool = _pool(JudgeRoute("only", judge, _cap("single_image")))

        self.assertTrue((await pool.judge("cond", b"frame")).satisfied)
        self.assertTrue((await pool.judge("cond", b"frame", context={})).satisfied)
        self.assertEqual(judge.calls, 2)


class TestThePrivatePrefixKeepsWorking(unittest.IsolatedAsyncioTestCase):
    """The pool stamps a ``_percept_`` key ITSELF. A predicate that refused it takes reflex down."""

    async def test_the_pools_own_stamped_key_does_not_make_its_route_ineligible(self):
        judge = _Judge()
        pool = _pool(
            JudgeRoute("reflex", judge, _cap("single_image")),
            mode="reflex",
        )

        verdict = await pool.judge("cond", b"frame")

        self.assertTrue(verdict.satisfied)
        self.assertEqual(
            judge.contexts[0].get(LATENCY_MODE_CONTEXT_KEY),
            "reflex",
            "reflex routing lost its own latency-mode stamp",
        )

    async def test_selection_reads_the_callers_context_not_the_stamped_one(self):
        # Eligibility must never turn on something the pool added after choosing the route. A
        # route narrowed to nothing still serves a reflex request, because the stamp is transport.
        judge = _Judge()
        pool = _pool(
            JudgeRoute("narrow", judge, _cap("single_image", context_keys=frozenset())),
            mode="reflex",
        )

        self.assertTrue((await pool.judge("cond", b"frame")).satisfied)

    async def test_a_private_key_that_does_not_exist_yet_is_admitted_by_rule(self):
        # The prefix rule holds for keys added later -- that is why it is a rule and not a list, and
        # why selection asks the registry helpers instead of testing RESERVED_CONTEXT_KEYS.
        judge = _Judge()
        pool = _pool(JudgeRoute("only", judge, _cap("single_image")))

        verdict = await pool.judge(
            "cond",
            b"frame",
            context={f"{PRIVATE_KEY_PREFIX}a_key_added_next_year": True},
        )

        self.assertEqual(judge.calls, 1)
        self.assertTrue(verdict.satisfied)


class TestANarrowedRouteIsARoutingFactNotAFailure(unittest.IsolatedAsyncioTestCase):
    async def test_a_narrowed_route_is_skipped_and_a_broad_sibling_serves(self):
        """Narrowing one route must not fail the request the pool can still answer."""
        narrow = _Judge(Verdict(True, 0.99, "narrow"))
        broad = _Judge(Verdict(True, 0.99, "broad"))
        pool = _pool(
            JudgeRoute(
                "narrow",
                narrow,
                _cap("single_image", context_keys=RESERVED_CONTEXT_KEYS - {"claim_mode"}),
                priority=0,
            ),
            JudgeRoute("broad", broad, _cap("single_image"), priority=1),
        )

        verdict = await pool.judge("cond", b"frame", context={"claim_mode": True})

        self.assertEqual(verdict.evidence, "broad")
        self.assertEqual(narrow.calls, 0)
        self.assertEqual(broad.calls, 1)

    async def test_a_narrowed_route_still_serves_everything_it_did_declare(self):
        narrow = _Judge()
        pool = _pool(
            JudgeRoute(
                "narrow",
                narrow,
                _cap("single_image", context_keys=frozenset({"claim_mode"})),
            )
        )

        self.assertTrue((await pool.judge("c", b"f", context={"claim_mode": True})).satisfied)
        self.assertEqual(narrow.calls, 1)


class TestTheReceiptSaysWhichRefusalHappened(unittest.IsolatedAsyncioTestCase):
    async def test_a_context_refusal_names_the_keys_on_a_no_route_receipt(self):
        pool = _pool(JudgeRoute("only", _Judge(), _cap("single_image")))

        await pool.judge("cond", b"frame", context={"conformance": "probe"})

        receipt = pool.receipts()[-1]
        self.assertEqual(receipt.outcome, "no_route")
        self.assertEqual(receipt.refused_context_keys, ("conformance",))
        self.assertIsNone(receipt.route)

    async def test_a_plain_shape_miss_names_no_context_key(self):
        """Attribution has to be exact or the receipt misdirects the operator reading it.

        Nothing here is transport-eligible, so the context never got a chance to refuse anything --
        even though this request also carries an undeclared key.
        """
        pool = _pool(JudgeRoute("strip", _Judge(), _cap("ordered_images")))

        await pool.judge("cond", b"frame", context={"conformance": "probe"})

        receipt = pool.receipts()[-1]
        self.assertEqual(receipt.outcome, "no_route")
        self.assertEqual(receipt.refused_context_keys, ())

    async def test_the_named_keys_are_the_union_across_routes_that_were_transport_eligible(self):
        # Two routes each refusing a different key: neither survives, and both keys explain why.
        pool = _pool(
            JudgeRoute(
                "a",
                _Judge(),
                _cap("single_image", context_keys=RESERVED_CONTEXT_KEYS - {"claim_mode"}),
            ),
            JudgeRoute(
                "b",
                _Judge(),
                _cap("single_image", context_keys=RESERVED_CONTEXT_KEYS - {"process_qa"}),
            ),
        )

        await pool.judge(
            "cond",
            b"frame",
            context={"claim_mode": True, "process_qa": True},
        )

        self.assertEqual(
            pool.receipts()[-1].refused_context_keys,
            ("claim_mode", "process_qa"),
        )

    async def test_the_named_keys_stay_bounded_for_an_arbitrary_caller_dict(self):
        # A receipt is retained; the caller's dict size is not the pool's to control.
        pool = _pool(JudgeRoute("only", _Judge(), _cap("single_image")))

        await pool.judge(
            "cond",
            b"frame",
            context={f"undeclared_{index:02d}": index for index in range(40)},
        )

        named = pool.receipts()[-1].refused_context_keys
        self.assertLessEqual(len(named), 8)
        self.assertEqual(named, tuple(sorted(named)))

    async def test_a_successful_call_carries_no_refusal(self):
        pool = _pool(JudgeRoute("only", _Judge(), _cap("single_image")))

        await pool.judge("cond", b"frame", context={"claim_mode": True})

        self.assertEqual(pool.receipts()[-1].outcome, "decisive")
        self.assertEqual(pool.receipts()[-1].refused_context_keys, ())


class TestItReusesTheExistingFailClosedPath(unittest.IsolatedAsyncioTestCase):
    async def test_the_default_is_an_unknown_verdict_exactly_like_any_other_no_route(self):
        pool = _pool(JudgeRoute("only", _Judge(), _cap("single_image")))

        verdict = await pool.judge("cond", b"frame", context={"conformance": "probe"})

        self.assertFalse(verdict.satisfied)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIn("judge pool unavailable", verdict.evidence)

    async def test_a_deployment_that_made_no_route_terminal_still_stops(self):
        """No new failure path: a context refusal IS a no_route, so it inherits the configured one.

        A distinct outcome value would have silently escaped this, which is the argument for putting
        the distinction in a receipt field instead.
        """
        pool = _pool(
            JudgeRoute("only", _Judge(), _cap("single_image")),
            terminal_outcomes={"no_route"},
        )

        with self.assertRaises(RuntimeError) as caught:
            await pool.judge("cond", b"frame", context={"conformance": "probe"})

        self.assertIn("conformance", str(caught.exception))
        self.assertIsInstance(pool.take_terminal_error(), RuntimeError)


class TestTheSeamBelowThePoolIsUntouched(unittest.TestCase):
    """Enforcement is at selection, so the conformance probe still reaches a bare adapter.

    ``percept.conformance`` passes ``{"conformance": "probe"}`` DIRECTLY into an adapter, bypassing
    every pool, to prove the seam accepts a caller's context. That is why the refusal cannot live in
    a backend: it would fail the published conformance suite. Both facts are true at once only
    because they sit at different layers, and this test is what says so out loud.
    """

    def test_an_adapter_called_directly_still_accepts_an_unregistered_key(self):
        judge = _Judge()

        verdict = asyncio.run(
            judge.judge("cond", b"frame", context={"conformance": "probe"})
        )

        self.assertTrue(verdict.satisfied)
        self.assertEqual(judge.contexts[-1], {"conformance": "probe"})


class TestTheDeclarationIsVisibleToAnOperator(unittest.TestCase):
    def test_snapshot_publishes_the_accepted_keys_beside_the_other_eligibility_inputs(self):
        # A no_route receipt names a refused key; the operator then needs to see which routes would
        # have taken it. Every other eligibility input is already published here.
        pool = _pool(
            JudgeRoute(
                "only",
                _Judge(),
                _cap("single_image", context_keys=frozenset({"claim_mode"})),
            )
        )

        route = pool.snapshot()["routes"]["only"]

        self.assertEqual(route["context_keys"], ["claim_mode"])
        self.assertNotIn("adapter", route)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
