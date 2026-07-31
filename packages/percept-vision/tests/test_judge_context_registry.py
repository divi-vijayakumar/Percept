"""The judge-context registry is TOTAL, and adding a key is a decision rather than an accident.

The registry replaces three hand-copied deny-lists that had already drifted apart — ``openrouter``
excluded ``process_qa``, ``anthropic`` and ``gemini`` did not — plus a fourth, separately incomplete
set in ``percept.hosted_byok``. Four copies of one fact is how a key ends up leaking into a prompt
nobody meant it to reach.

The load-bearing test here is :meth:`TestTheRegistryIsTotal.test_every_key_the_runtime_emits_is_declared`.
It fails when someone adds a context key without classifying it, which is the entire point: the cost
of an unclassified key is that it renders to the model as a fact, under a sentence telling the model
to trust it over its own read.

No provider, no network, no key.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from percept.extraction import CONTEXT_KEY as EXTRACTOR_CONTEXT_KEY
from percept.judge_context import JUDGE_OPEN_EPISODE_KEY, JUDGE_STATIC_CONTEXT_KEY
from percept.judge_context_producers import (
    NON_RENDERABLE_KEYS,
    PRIVATE_KEY_PREFIX,
    PUBLIC_CONTEXT_KEYS,
    RESERVED_CONTEXT_KEYS,
    is_registered,
    spec_for,
    unregistered,
)

_PERCEPT = Path(__file__).resolve().parents[1] / "percept"


class TestTheRegistryIsTotal(unittest.TestCase):
    def test_every_key_the_runtime_emits_is_declared(self):
        """THE forcing function. A context key written anywhere in the package must be classified.

        Scans for literal ``context[...] = `` / ``context.update({...})`` style writes in agent.py,
        which is where every judge context is assembled. A key found there but absent from the
        registry is exactly the failure this file exists to catch.
        """
        src = (_PERCEPT / "agent.py").read_text()
        # `context["k"] = ...` and `"k": ...` inside a context.update({...}) call.
        written = set(re.findall(r'context\[\s*"([a-z_][a-z0-9_]*)"\s*\]\s*=', src))
        undeclared = {k for k in written if not is_registered(k)}
        self.assertEqual(
            undeclared, set(),
            "agent.py writes judge-context key(s) the registry does not declare: "
            f"{sorted(undeclared)}. Classify each in percept/judge_context_producers.py — an "
            "unclassified key renders to the model as a fact it is told to trust.",
        )

    def test_the_named_constants_are_the_registered_ones(self):
        # Imported from their defining modules, never re-spelled here: a registry that restates a
        # constant is a fifth copy of the thing this file exists to stop.
        for key in (JUDGE_OPEN_EPISODE_KEY, JUDGE_STATIC_CONTEXT_KEY, EXTRACTOR_CONTEXT_KEY):
            with self.subTest(key=key):
                self.assertIn(key, RESERVED_CONTEXT_KEYS)
                self.assertIsNotNone(spec_for(key))

    def test_every_spec_carries_a_note(self):
        # The note is why the classification is what it is. A spec without one is a guess.
        for spec in PUBLIC_CONTEXT_KEYS:
            with self.subTest(key=spec.key):
                self.assertTrue(spec.note.strip(), f"{spec.key} is classified with no reason given")

    def test_no_duplicate_keys(self):
        keys = [spec.key for spec in PUBLIC_CONTEXT_KEYS]
        self.assertEqual(len(keys), len(set(keys)))


class TestWhatIsAndIsNotRegistered(unittest.TestCase):
    def test_an_arbitrary_caller_key_is_not_registered(self):
        # The leak in one assertion. `conformance` is the key percept.conformance passes straight
        # into an adapter; today it renders into the prompt.
        self.assertFalse(is_registered("conformance"))
        self.assertEqual(unregistered({"conformance": "probe"}), frozenset({"conformance"}))

    def test_private_keys_are_admitted_by_prefix_not_by_enumeration(self):
        # Including one that does not exist yet -- the prefix rule holds for future keys, which is
        # why it is a rule and not a list.
        self.assertTrue(is_registered(f"{PRIVATE_KEY_PREFIX}latency_mode"))
        self.assertTrue(is_registered(f"{PRIVATE_KEY_PREFIX}a_key_added_next_year"))

    def test_a_wholly_declared_context_has_nothing_unregistered(self):
        ctx = {k: object() for k in RESERVED_CONTEXT_KEYS}
        ctx[f"{PRIVATE_KEY_PREFIX}latency_mode"] = "reflex"
        self.assertEqual(unregistered(ctx), frozenset())

    def test_empty_and_none_are_trivially_accounted_for(self):
        self.assertEqual(unregistered(None), frozenset())
        self.assertEqual(unregistered({}), frozenset())


class TestTreatmentsSayHowAKeyIsSpoken(unittest.TestCase):
    def test_only_the_extractor_is_framed_as_a_specialist_report(self):
        """The behavioural bug the registry fixes.

        The generic render announces context as "a specialist model has reported these structured
        facts (trust them over your own read)". That is true of the extractor and of nothing else --
        the open episode is this system's own established history, and the static context is a
        policy document. Both were being framed as a third party's authoritative claim.
        """
        specialist = {s.key for s in PUBLIC_CONTEXT_KEYS if s.treatment == "specialist_facts"}
        self.assertEqual(specialist, {EXTRACTOR_CONTEXT_KEY})
        self.assertNotIn(JUDGE_OPEN_EPISODE_KEY, specialist)
        self.assertNotIn(JUDGE_STATIC_CONTEXT_KEY, specialist)

    def test_non_renderable_is_everything_except_the_specialist_report(self):
        self.assertEqual(NON_RENDERABLE_KEYS, RESERVED_CONTEXT_KEYS - {EXTRACTOR_CONTEXT_KEY})

    def test_running_answer_is_declared_at_all(self):
        # It was in NO backend's deny-list, so on the ordinary judge path it rendered as a fact.
        self.assertIn("running_answer", RESERVED_CONTEXT_KEYS)
        self.assertEqual(spec_for("running_answer").treatment, "early_return")

    def test_the_episode_is_keyed_and_the_rubric_bypasses(self):
        # These two answer-affecting keys are handled differently on purpose, and the registry has
        # to record which: keying keeps a continuing occurrence cacheable, bypassing does not.
        self.assertTrue(spec_for(JUDGE_OPEN_EPISODE_KEY).fingerprinted)
        self.assertTrue(spec_for(JUDGE_STATIC_CONTEXT_KEY).answer_affecting)
        self.assertFalse(spec_for(JUDGE_STATIC_CONTEXT_KEY).fingerprinted)

    def test_every_key_is_answer_affecting_or_says_why_not(self):
        # Today all eight are. The assertion is here so that a future key declared NOT
        # answer-affecting is a visible, reviewed line rather than a default.
        for spec in PUBLIC_CONTEXT_KEYS:
            with self.subTest(key=spec.key):
                self.assertTrue(spec.answer_affecting)


class TestTheHostedSeamGap(unittest.TestCase):
    def test_hosted_byok_reserved_set_is_currently_incomplete(self):
        """Records the live gap the registry closes, so closing it is a deliberate edit.

        ``percept.hosted_byok`` maintains its own reserved-key set to refuse a hosted caller that
        supplies runtime-owned context. It omits two keys the runtime really does own, so a hosted
        caller can pass them today. This asserts the CURRENT state; when hosted_byok derives its set
        from the registry, this test flips to an equality assertion.
        """
        from percept.hosted_byok import _RESERVED_CONTEXT_KEYS as hosted

        missing = RESERVED_CONTEXT_KEYS - hosted
        self.assertEqual(
            missing, {JUDGE_OPEN_EPISODE_KEY, "event_chain"},
            "the hosted reserved-key gap changed; derive it from RESERVED_CONTEXT_KEYS and make "
            "this an equality assertion",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
