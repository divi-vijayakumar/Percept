"""What each judge-context key is SAID to be, in all three real backends at once.

The three adapters each rendered ``Vision.judge(context={...})`` through a hand-copied deny-list, and
the lists disagreed — ``openrouter`` excluded ``process_qa``, ``anthropic`` and ``gemini`` did not, and
``running_answer`` was in none of them. Only the ``_percept_`` PREFIX was structural; every public key
was filtered BY NAME, so anything nobody remembered fell through into one sentence:

    A specialist model has reported these structured facts (trust them over your own read): …

That sentence is honest about the extractor and about nothing else. ``judge_open_episode`` is the
watch's OWN accumulated record of an occurrence it is already tracking — announcing it as an outside
report invites the model to read its own earlier verdict as fresh corroboration. ``judge_static_context``
is a deployment's standing rubric, up to a megabyte of policy, and it was being inlined into that
semicolon-joined sentence as though it were one observed fact.

**The load-bearing test is** :meth:`TestOnlyTheExtractorIsASpecialistReport.test_the_facts_sentence_carries_the_extractor_and_nothing_else`.
It is ONE test over THREE backends on purpose: the bug was divergence, so a guard that could pass for
two adapters and fail for the third would reproduce the original failure mode.

No provider, no network, no key — every completion is an injected fake, exactly like the existing
backend tests.
"""

from __future__ import annotations

import asyncio
import re
import sys
import types
import unittest
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

from percept.backends import anthropic as an
from percept.backends import gemini as gm
from percept.backends import openrouter as orr
from percept.extraction import CONTEXT_KEY as EXTRACTOR_CONTEXT_KEY
from percept.judge_context import JUDGE_OPEN_EPISODE_KEY, JUDGE_STATIC_CONTEXT_KEY
from percept.judge_context_producers import (
    NON_RENDERABLE_KEYS,
    PRIVATE_KEY_PREFIX,
    PUBLIC_CONTEXT_KEYS,
    spec_for,
)

COND = "the lid is closed"
FRAME = b"\x89PNG\r\n\x1a\n-fake-bytes"

# Distinctive, period-free values, so "did this text reach the facts sentence?" is decidable by
# substring and the sentence's own terminating ". " cannot be confused with one inside a value.
EXTRACTOR_FACT = "plate=7ABC123"
EPISODE_TEXT = (
    "you are continuing an already-established episode (open) about person:alice, "
    "first seen at t=1_000, confirmed across 4 observations"
)
RUBRIC_TEXT = "RUBRIC-A spill counts only if the liquid pools on the floor"
RUNNING_ANSWER_TEXT = "so far the technician has opened the panel"

#: Everything the ordinary judge path can legitimately carry at once. ``process_qa``/``observe_step``
#: are deliberately absent: they SELECT A DIFFERENT REQUEST on the Gemini adapter, so including them
#: would test the process-QA branch rather than the ordinary prompt. They are covered instead by
#: :class:`TestTheRendererIsTotalOverTheRegistry`, against the renderer all three backends share.
FULL_CONTEXT = {
    EXTRACTOR_CONTEXT_KEY: EXTRACTOR_FACT,
    JUDGE_OPEN_EPISODE_KEY: EPISODE_TEXT,
    JUDGE_STATIC_CONTEXT_KEY: RUBRIC_TEXT,
    "running_answer": RUNNING_ANSWER_TEXT,
    "claim_mode": True,
    "event_chain": False,
    f"{PRIVATE_KEY_PREFIX}latency_mode": "reflex",
}


# ==================================================================================================
# the three real prompt builders, each reduced to "context in, assembled prompt text out"
# ==================================================================================================


def _anthropic_prompt(context) -> str:
    v = an.AnthropicVision(client=object())
    content = v._build_content(
        COND, FRAME, roi=None, exemplar=None, corrections=None, frames=None,
        context=context, claim_mode=bool(context and context.get("claim_mode")),
    )
    return content[-1]["text"]


def _openrouter_prompt(context) -> str:
    v = orr.OpenRouterVision(client=object())
    content = v._build_content(
        COND, FRAME, exemplar=None, corrections=None, frames=None,
        context=context, claim_mode=bool(context and context.get("claim_mode")),
    )
    return content[-1]["text"]


class _GeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None


class _GeminiModels:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _GeminiResponse(self._text)


class _GeminiClient:
    def __init__(self, text: str) -> None:
        self.models = _GeminiModels(text)
        self.aio = SimpleNamespace(models=self.models)


def _lazy_google() -> dict[str, types.ModuleType]:
    """The ``google.genai`` surface this adapter touches, and nothing else (invariant 7: the SDK is
    lazy-imported inside methods, so the module under test imports without it)."""

    class Part:
        @staticmethod
        def from_bytes(**kwargs: object) -> object:
            return ("part", kwargs)

    class GenerateContentConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class HttpOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    google_types = types.ModuleType("google.genai.types")
    genai.types = google_types
    google.genai = genai
    google_types.Part = Part
    google_types.GenerateContentConfig = GenerateContentConfig
    google_types.HttpOptions = HttpOptions
    return {"google": google, "google.genai": genai, "google.genai.types": google_types}


def _gemini_prompt(context) -> str:
    # Gemini assembles its question inside ``judge`` rather than a separate builder, so the prompt is
    # read back off the (fake) transport call it would have made.
    client = _GeminiClient('{"satisfied": false, "confidence": 0.1, "evidence": "x"}')
    adapter = gm.GeminiVision(client=client)
    with patch.dict(sys.modules, _lazy_google()):
        asyncio.run(adapter.judge(COND, FRAME, context=context))
    assert len(client.models.calls) == 1
    return "".join(p for p in client.models.calls[0]["contents"] if isinstance(p, str))


#: Every shipping backend that parses the strict-JSON verdict contract. Adding a fourth without adding
#: it here is the drift this file exists to stop.
BACKENDS = {
    "anthropic": _anthropic_prompt,
    "gemini": _gemini_prompt,
    "openrouter": _openrouter_prompt,
}

#: The facts a backend announced as a specialist model's report. Deliberately matched on the SHAPE of
#: the claim ("a specialist model … reported … : <payload>.") rather than on either adapter's exact
#: wording, so the guard survives a reword and still fails on a leak.
_SPECIALIST_PAYLOAD = re.compile(
    r"A specialist model (?:has )?reported these structured facts[^:]*:\s*(.*?)\.\s"
)


def _specialist_payload(prompt: str) -> str | None:
    """What the prompt claims a specialist reported, or ``None`` if it makes no such claim."""
    match = _SPECIALIST_PAYLOAD.search(prompt)
    return match.group(1) if match else None


# ==================================================================================================
# THE REGRESSION GUARD — one test, three backends
# ==================================================================================================


class TestOnlyTheExtractorIsASpecialistReport(unittest.TestCase):
    def test_the_facts_sentence_carries_the_extractor_and_nothing_else(self):
        """The bug, in one assertion, on every backend.

        Handed the open episode, a deployment rubric, a running answer and the extractor's report at
        once, the sentence that tells the model to trust a specialist over its own read must contain
        the extractor's facts EXACTLY — not the watch's own history, not a policy document, and not
        the accumulating answer that was in no adapter's deny-list.
        """
        for name, prompt_of in BACKENDS.items():
            with self.subTest(backend=name):
                prompt = prompt_of(dict(FULL_CONTEXT))
                payload = _specialist_payload(prompt)
                self.assertIsNotNone(
                    payload,
                    f"{name} dropped the extractor's report entirely; a specialist's structured "
                    "facts are the one thing that sentence is for",
                )
                self.assertEqual(
                    payload, f"{EXTRACTOR_CONTEXT_KEY}={EXTRACTOR_FACT}",
                    f"{name} announced something other than the extractor's report as a "
                    "specialist's facts",
                )
                # Named individually so a regression reads as "the episode leaked" rather than as a
                # string diff of a paragraph.
                self.assertNotIn(EPISODE_TEXT, payload, f"{name}: the open episode leaked")
                self.assertNotIn(RUBRIC_TEXT, payload, f"{name}: the standing rubric leaked")
                self.assertNotIn(
                    JUDGE_OPEN_EPISODE_KEY, payload, f"{name}: the open-episode KEY leaked"
                )
                self.assertNotIn(
                    JUDGE_STATIC_CONTEXT_KEY, payload, f"{name}: the static-context KEY leaked"
                )

    def test_no_non_renderable_key_appears_as_a_fact_anywhere(self):
        """``key=value`` is the fact syntax. No key with its own treatment may be spelled that way.

        Derived from the registry rather than restated, so a key classified tomorrow is covered here
        on the same commit.
        """
        for name, prompt_of in BACKENDS.items():
            prompt = prompt_of(dict(FULL_CONTEXT))
            for key in sorted(NON_RENDERABLE_KEYS & set(FULL_CONTEXT)):
                with self.subTest(backend=name, key=key):
                    self.assertNotIn(
                        f"{key}=", prompt,
                        f"{name} rendered {key!r} as a fact; it is declared "
                        f"{spec_for(key).treatment!r} and has its own treatment",
                    )

    def test_the_running_answer_never_reaches_the_ordinary_prompt(self):
        # The key that was in NO backend's deny-list. It belongs to the process-QA request; on the
        # ordinary path it is not a fact about the frame at all.
        for name, prompt_of in BACKENDS.items():
            with self.subTest(backend=name):
                self.assertNotIn(RUNNING_ANSWER_TEXT, prompt_of(dict(FULL_CONTEXT)))

    def test_the_episode_and_the_rubric_still_reach_the_model(self):
        """Not rendering them as facts is not the same as dropping them — both are answer-affecting.

        The rubric additionally must be DELIMITED rather than inlined: it can be a megabyte of
        arbitrary policy bytes, and run into a semicolon-joined sentence any imperative inside it
        would be indistinguishable from this prompt's own instructions.
        """
        for name, prompt_of in BACKENDS.items():
            with self.subTest(backend=name):
                prompt = prompt_of(dict(FULL_CONTEXT))
                self.assertIn(EPISODE_TEXT, prompt, f"{name} dropped the open episode")
                self.assertIn(RUBRIC_TEXT, prompt, f"{name} dropped the standing rubric")
                self.assertIn("--- BEGIN STANDING RUBRIC ---", prompt)
                self.assertIn("--- END STANDING RUBRIC ---", prompt)

    def test_the_two_new_blocks_are_identical_on_every_backend(self):
        """The three prompts may differ; what they SAY these two keys are may not.

        Each adapter's question stem, evidence framing and reply contract are its own. The provenance
        claim about the episode and the rubric is not — divergence there is the original bug.
        """
        rendered = {
            name: prompt_of({JUDGE_OPEN_EPISODE_KEY: EPISODE_TEXT, JUDGE_STATIC_CONTEXT_KEY: RUBRIC_TEXT})
            for name, prompt_of in BACKENDS.items()
        }
        blocks = {}
        for name, prompt in rendered.items():
            start = prompt.index("The deployment that installed this watch")
            end = prompt.index("Use it only to tell a continuation")
            blocks[name] = prompt[start:end]
        self.assertEqual(len(set(blocks.values())), 1, f"backends disagree: {blocks}")


class TestTheExtractorOnlyPromptIsUnchanged(unittest.TestCase):
    """The extractor's sentence shipped correct and must not move.

    A prompt edit moves eval results, so the fix is scoped to the keys that were framed dishonestly.
    With ONLY the extractor in context, each backend's prompt is byte-for-byte what it was before the
    registry existed — including OpenRouter's shorter phrasing, which is different from the other two
    and stays different.
    """

    def test_anthropic_and_gemini_keep_the_long_sentence(self):
        expected = (
            f"A specialist model has reported these structured facts (trust them over your own "
            f"read): {EXTRACTOR_CONTEXT_KEY}={EXTRACTOR_FACT}. "
        )
        for name in ("anthropic", "gemini"):
            with self.subTest(backend=name):
                prompt = BACKENDS[name]({EXTRACTOR_CONTEXT_KEY: EXTRACTOR_FACT})
                self.assertIn(expected, prompt)

    def test_openrouter_keeps_its_own_shorter_sentence(self):
        prompt = BACKENDS["openrouter"]({EXTRACTOR_CONTEXT_KEY: EXTRACTOR_FACT})
        self.assertIn(
            f"A specialist model reported these structured facts (trust them): "
            f"{EXTRACTOR_CONTEXT_KEY}={EXTRACTOR_FACT}. ",
            prompt,
        )

    def test_an_empty_context_adds_nothing_on_any_backend(self):
        for name, prompt_of in BACKENDS.items():
            with self.subTest(backend=name):
                self.assertEqual(prompt_of(None), prompt_of({}))


class TestTheRendererIsTotalOverTheRegistry(unittest.TestCase):
    """The renderer itself, over keys the ordinary judge path cannot carry together.

    ``process_qa`` and ``observe_step`` select a different request on the adapters that implement
    them, so they cannot be probed through a live ``judge`` call — but a backend that does NOT
    implement that branch (Anthropic has no process-QA path) used to render them as facts. Asserted
    against the shared renderer, which is the same function object in all three adapters.
    """

    def test_all_three_backends_render_through_one_function(self):
        self.assertIs(gm._judge_context_sentences, an._judge_context_sentences)
        self.assertIs(orr._judge_context_sentences, an._judge_context_sentences)

    def test_every_declared_treatment_is_handled(self):
        # Fails when a treatment is added to the registry without a branch here, rather than letting
        # the new one silently fall through to the specialist default.
        handled = {"own_block", "early_return", "episode", "policy", "specialist_facts", "transport"}
        self.assertEqual({spec.treatment for spec in PUBLIC_CONTEXT_KEYS} - handled, set())

    def test_no_early_return_key_is_ever_rendered(self):
        for spec in PUBLIC_CONTEXT_KEYS:
            if spec.treatment != "early_return":
                continue
            with self.subTest(key=spec.key):
                text = an._judge_context_sentences({spec.key: "SENTINEL-VALUE"})
                self.assertEqual(text, "", f"{spec.key} reached the ordinary prompt")

    def test_a_private_key_that_does_not_exist_yet_is_never_rendered(self):
        text = an._judge_context_sentences({f"{PRIVATE_KEY_PREFIX}invented_next_year": "SENTINEL"})
        self.assertEqual(text, "")

    def test_a_caller_key_the_registry_does_not_declare_is_still_a_caller_fact(self):
        # Shipped behaviour, deliberately kept: ``context={...}`` is the documented door for a
        # caller's own structured facts, and ``percept.conformance`` passes an undeclared probe key
        # straight into an adapter to prove the seam accepts one. Refusing it is route selection's
        # job (``judge_context_producers.unregistered``), not an adapter's.
        text = an._judge_context_sentences({"conformance": "probe"})
        self.assertIn("conformance=probe", text)


class TestAReadOnlyContextIsStillAContext(unittest.TestCase):
    """The other divergence: OpenRouter guarded on ``isinstance(context, dict)``.

    Handed a ``MappingProxyType``, it rendered NOTHING while Anthropic rendered the facts in full —
    a silent drop of the extractor's report on one backend only. Latent today (the routed path hands
    down a real dict), which is exactly why it needs a test rather than a comment.
    """

    def test_a_mappingproxy_renders_identically_everywhere(self):
        proxy = MappingProxyType({EXTRACTOR_CONTEXT_KEY: EXTRACTOR_FACT})
        for name, prompt_of in BACKENDS.items():
            with self.subTest(backend=name):
                self.assertEqual(
                    _specialist_payload(prompt_of(proxy)),
                    f"{EXTRACTOR_CONTEXT_KEY}={EXTRACTOR_FACT}",
                )

    def test_a_non_mapping_context_renders_nothing_rather_than_raising(self):
        self.assertEqual(an._judge_context_sentences(["not", "a", "mapping"]), "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
