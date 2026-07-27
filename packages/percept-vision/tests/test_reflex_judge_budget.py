"""The reflex judge request is TERSE and BOUNDED; the deliberate one is unchanged.

Streaming decomposition of the sub-1.5s reflex gate showed the miss was in the GENERATION half, not
in ingestion: a shipped ``max_tokens=1024`` "headroom" plus a prose ``evidence`` field let a route
spend seconds emitting text *after* it already knew the answer. These tests pin the fix as a
property of the ROUTE — ``JudgePool(mode='reflex')`` stamps a reserved context key, and a
verbosity-aware backend asks only for what the gate reads.

No provider or network calls: every completion is an injected fake, exactly like the existing
openrouter/gemini backend tests.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from percept.adapters import Verdict
from percept.backends.anthropic import (
    _JUDGE_SYSTEM,
    _JUDGE_SYSTEM_REFLEX,
    _REFLEX_MAX_FRAMES,
    _REFLEX_MAX_TOKENS,
    _REFLEX_STRUCTURED_MAX_TOKENS,
    _parse_verdict,
)
from percept.backends.gemini import GeminiVision
from percept.backends.judge_pool import (
    LATENCY_MODE_CONTEXT_KEY,
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.backends.openrouter import OpenRouterVision
from percept.gate import ASK, FIRE, Gate

_DELIBERATE_MAX_TOKENS = 1024


# --------------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def _settings(model: str = "qwen/qwen3-vl-8b-instruct"):
    return SimpleNamespace(models=SimpleNamespace(vision_model=model))


def _openrouter(content: str) -> tuple[OpenRouterVision, _FakeClient]:
    client = _FakeClient(content)
    return OpenRouterVision(_settings(), client=client), client


def _route(adapter, *, shape: str = "single_image") -> JudgeRoute:
    return JudgeRoute(
        "primary",
        adapter,
        JudgeCapability(
            evidence_shapes=frozenset({shape}),
            latency_slo_s=1.5,
            modes=frozenset({"reflex", "deliberate"}),
        ),
    )


def _pool(adapter, *, mode: str, shape: str = "single_image") -> JudgePool:
    return JudgePool(
        [_route(adapter, shape=shape)],
        mode=mode,
        deadline_s=1.5,
        max_parallel=1,
    )


def _sent(client: _FakeClient) -> dict:
    assert len(client.chat.completions.calls) == 1
    return client.chat.completions.calls[0]


def _system_of(sent: dict) -> str:
    return next(m["content"] for m in sent["messages"] if m["role"] == "system")


def _prompt_of(sent: dict) -> str:
    user = next(m for m in sent["messages"] if m["role"] == "user")
    return " ".join(
        block["text"] for block in user["content"] if block.get("type") == "text"
    )


TERSE_REPLY = json.dumps({"satisfied": True, "confidence": 0.87})
VERBOSE_REPLY = json.dumps(
    {"satisfied": True, "confidence": 0.87, "evidence": "a hand closes the lid"}
)


# --------------------------------------------------------------------------------------------
# the pool publishes the route's latency mode
# --------------------------------------------------------------------------------------------


class _RecordingJudge:
    def __init__(self) -> None:
        self.contexts: list[dict | None] = []

    async def judge(self, _condition, _frame, **kwargs):
        self.contexts.append(kwargs.get("context"))
        return Verdict(satisfied=True, confidence=0.9, evidence="ok")


class JudgePoolPublishesLatencyMode(unittest.TestCase):
    def test_reflex_pool_stamps_the_mode_without_mutating_the_caller_context(self):
        judge = _RecordingJudge()
        caller_context = {"extractor": {"plate": "AB-12"}}
        asyncio.run(
            _pool(judge, mode="reflex").judge("x?", b"jpeg", context=caller_context)
        )
        self.assertEqual(
            judge.contexts[0][LATENCY_MODE_CONTEXT_KEY],
            "reflex",
        )
        # the specialist fact still rides along, and the caller's dict is untouched
        self.assertEqual(judge.contexts[0]["extractor"], {"plate": "AB-12"})
        self.assertEqual(caller_context, {"extractor": {"plate": "AB-12"}})

    def test_reflex_pool_stamps_the_mode_when_the_caller_passed_no_context(self):
        judge = _RecordingJudge()
        asyncio.run(_pool(judge, mode="reflex").judge("x?", b"jpeg"))
        self.assertEqual(judge.contexts[0], {LATENCY_MODE_CONTEXT_KEY: "reflex"})

    def test_deliberate_pool_leaves_the_context_exactly_as_the_caller_sent_it(self):
        judge = _RecordingJudge()
        asyncio.run(_pool(judge, mode="deliberate").judge("x?", b"jpeg"))
        self.assertIsNone(judge.contexts[0])

        judge = _RecordingJudge()
        caller_context = {"event_chain": True}
        asyncio.run(
            _pool(judge, mode="deliberate", shape="video_clip").judge(
                "x?", b"jpeg", context=caller_context
            )
        )
        self.assertEqual(judge.contexts[0], {"event_chain": True})
        self.assertNotIn(LATENCY_MODE_CONTEXT_KEY, judge.contexts[0])


# --------------------------------------------------------------------------------------------
# OpenRouter: the shipped judge request
# --------------------------------------------------------------------------------------------


class OpenRouterReflexRequest(unittest.TestCase):
    def test_reflex_request_is_bounded_and_asks_for_no_prose(self):
        adapter, client = _openrouter(TERSE_REPLY)
        asyncio.run(_pool(adapter, mode="reflex").judge("the lid is closed", b"jpeg"))

        sent = _sent(client)
        self.assertEqual(sent["max_tokens"], _REFLEX_MAX_TOKENS)
        self.assertLessEqual(sent["max_tokens"], 32)
        self.assertEqual(_system_of(sent), _JUDGE_SYSTEM_REFLEX)

        prompt = _prompt_of(sent)
        self.assertIn('STRICT JSON {"satisfied", "confidence"}', prompt)
        self.assertNotIn('"evidence"}', prompt)
        # neither half of the request may still solicit a written reason
        self.assertNotIn("<short reason>", _system_of(sent))
        self.assertIn("no prose", prompt)
        # ...but the grounded question itself is unchanged
        self.assertIn("the lid is closed", prompt)

    def test_reflex_request_still_demands_confidence(self):
        # the Gate is the sole firing authority and compares confidence to its threshold; a
        # request that made confidence optional would silently turn every verdict into an ASK.
        adapter, client = _openrouter(TERSE_REPLY)
        asyncio.run(_pool(adapter, mode="reflex").judge("x?", b"jpeg"))
        sent = _sent(client)
        self.assertIn('"confidence"', _prompt_of(sent))
        self.assertIn('"confidence": <0..1>', _system_of(sent))
        self.assertIn("REQUIRED", _system_of(sent))

    def test_deliberate_request_is_byte_for_byte_the_direct_request(self):
        pooled_adapter, pooled_client = _openrouter(VERBOSE_REPLY)
        asyncio.run(_pool(pooled_adapter, mode="deliberate").judge("the lid is closed", b"jpeg"))

        direct_adapter, direct_client = _openrouter(VERBOSE_REPLY)
        asyncio.run(direct_adapter.judge("the lid is closed", b"jpeg"))

        self.assertEqual(_sent(pooled_client), _sent(direct_client))
        sent = _sent(pooled_client)
        self.assertEqual(sent["max_tokens"], _DELIBERATE_MAX_TOKENS)
        self.assertEqual(_system_of(sent), _JUDGE_SYSTEM)
        self.assertIn('{"satisfied", "confidence", "evidence"}', _prompt_of(sent))

    def test_reflex_occurrence_request_keeps_every_field_cognition_reads(self):
        # phase/action_present/events are NOT prose: behavior.normalize_event_chain_verdict
        # abstains on an inconsistent tuple and the occurrence ledger counts events. A reflex
        # occurrence route therefore drops only the free-text evidence field.
        adapter, client = _openrouter(
            json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.93,
                    "action_present": True,
                    "phase": "during",
                    "events": [{"t_start": 0.1, "t_end": 0.4, "label": "lid closes"}],
                }
            )
        )
        verdict = asyncio.run(
            _pool(adapter, mode="reflex", shape="video_clip").judge(
                "one discrete closing",
                b"latest",
                frames=[b"earlier"],
                context={"event_chain": True},
            )
        )

        sent = _sent(client)
        self.assertEqual(sent["max_tokens"], _REFLEX_STRUCTURED_MAX_TOKENS)
        self.assertLess(sent["max_tokens"], _DELIBERATE_MAX_TOKENS)
        prompt = _prompt_of(sent)
        self.assertIn('"action_present", "phase", "events"', prompt)
        self.assertIn("List every grounded occurrence", prompt)
        self.assertIn('omit "evidence"', prompt)
        self.assertNotIn('"confidence", "evidence"', prompt)
        # and the structured contract still round-trips into the verdict
        self.assertEqual((verdict.phase, verdict.action_present), ("during", True))
        self.assertEqual(len(verdict.events), 1)

    def test_reflex_claim_mode_keeps_the_correction_channel_and_its_budget(self):
        # A claim watch exists to catch a CORRECTION; the ledger reads operation/value off the
        # reply. Trimming that to hit a deadline would silently drop the thing being watched for,
        # so claim mode opts out of terseness even on a reflex route.
        adapter, client = _openrouter(
            json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.9,
                    "evidence": "buyer restates the price",
                    "operation": "replace",
                    "value": "$25K",
                }
            )
        )
        verdict = asyncio.run(
            _pool(adapter, mode="reflex").judge(
                "the price is stated",
                b"jpeg",
                context={"claim_mode": True},
            )
        )

        sent = _sent(client)
        self.assertEqual(sent["max_tokens"], _DELIBERATE_MAX_TOKENS)
        self.assertEqual(_system_of(sent), _JUDGE_SYSTEM)
        prompt = _prompt_of(sent)
        self.assertIn('{"satisfied", "confidence", "evidence"}', prompt)
        self.assertIn("CLAIMED VALUE", prompt)
        self.assertEqual((verdict.operation, verdict.value), ("replace", "$25K"))

    def test_a_directly_constructed_adapter_is_untouched_by_the_mechanism(self):
        # No pool, no stamp: an SDK user calling the adapter straight keeps the verbose contract.
        adapter, client = _openrouter(VERBOSE_REPLY)
        asyncio.run(adapter.judge("x?", b"jpeg"))
        self.assertEqual(_sent(client)["max_tokens"], _DELIBERATE_MAX_TOKENS)
        self.assertEqual(_system_of(_sent(client)), _JUDGE_SYSTEM)


# --------------------------------------------------------------------------------------------
# Gemini: the same policy, so the two providers stay coherent
# --------------------------------------------------------------------------------------------


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
    return {
        "google": google,
        "google.genai": genai,
        "google.genai.types": google_types,
    }


def _gemini_config(mode: str, reply: str) -> dict:
    client = _GeminiClient(reply)
    adapter = GeminiVision(client=client)
    with patch.dict(sys.modules, _lazy_google()):
        asyncio.run(_pool(adapter, mode=mode).judge("the lid is closed", b"jpeg"))
    assert len(client.models.calls) == 1
    return client.models.calls[0]["config"].kwargs


class GeminiMatchesTheOpenRouterPolicy(unittest.TestCase):
    def test_reflex_request_is_bounded_and_asks_for_no_prose(self):
        config = _gemini_config("reflex", TERSE_REPLY)
        self.assertEqual(config["max_output_tokens"], _REFLEX_MAX_TOKENS)
        self.assertEqual(config["system_instruction"], _JUDGE_SYSTEM_REFLEX)

    def test_deliberate_request_keeps_the_historical_headroom_and_prose(self):
        config = _gemini_config("deliberate", VERBOSE_REPLY)
        self.assertEqual(config["max_output_tokens"], _DELIBERATE_MAX_TOKENS)
        self.assertEqual(config["system_instruction"], _JUDGE_SYSTEM)


# --------------------------------------------------------------------------------------------
# the reflex FRAME bound: the same route policy applied to the INGESTION half
# --------------------------------------------------------------------------------------------
#
# The output budget above fixed the GENERATION half of the reflex miss. The remaining miss is
# ingestion: at 5 frames the measured provider p95 (1.473s) all but exhausts the 1.5s pool deadline,
# so under concurrency the unit fails closed to UNKNOWN and the gate can only ASK. A measured frame
# sweep put the recall optimum at 3, bracketed on both sides. These tests pin the bound as a ROUTE
# property — reflex only, most-recent-first, deliberate untouched.

CONTEXT_FRAMES = [b"f1", b"f2", b"f3", b"f4"]  # earlier context, oldest -> newest
CURRENT_FRAME = b"f5"  # the moment actually being judged; always retained


def _shape_for(frames) -> str:
    return "ordered_images" if frames else "single_image"


def _openrouter_images(mode: str, *, frames) -> list[bytes]:
    """The image payload of one pooled OpenRouter judge call, decoded back to frame bytes."""
    adapter, client = _openrouter(TERSE_REPLY)
    asyncio.run(
        _pool(adapter, mode=mode, shape=_shape_for(frames)).judge(
            "the lid is closed", CURRENT_FRAME, frames=list(frames)
        )
    )
    user = next(m for m in _sent(client)["messages"] if m["role"] == "user")
    return [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in user["content"]
        if block.get("type") == "image_url"
    ]


def _gemini_call(mode: str, *, frames, context: dict | None = None, shape: str | None = None):
    client = _GeminiClient(TERSE_REPLY)
    adapter = GeminiVision(client=client)
    with patch.dict(sys.modules, _lazy_google()):
        asyncio.run(
            _pool(adapter, mode=mode, shape=shape or _shape_for(frames)).judge(
                "the lid is closed",
                CURRENT_FRAME,
                frames=list(frames),
                context=context,
            )
        )
    assert len(client.models.calls) == 1
    return client.models.calls[0]


def _gemini_images(mode: str, *, frames) -> list[bytes]:
    """The same payload for Gemini: the fake ``Part.from_bytes`` hands back its own kwargs."""
    contents = _gemini_call(mode, frames=frames)["contents"]
    return [
        part[1]["data"]
        for part in contents
        if isinstance(part, tuple) and part and part[0] == "part"
    ]


def _gemini_prompt(
    mode: str,
    *,
    frames,
    context: dict | None = None,
    shape: str | None = None,
) -> str:
    contents = _gemini_call(mode, frames=frames, context=context, shape=shape)["contents"]
    return " ".join(part for part in contents if isinstance(part, str))


def _openrouter_prompt(
    mode: str,
    *,
    frames,
    context: dict | None = None,
    shape: str = "video_clip",
) -> str:
    adapter, client = _openrouter(TERSE_REPLY)
    asyncio.run(
        _pool(adapter, mode=mode, shape=shape).judge(
            "one discrete closing",
            CURRENT_FRAME,
            frames=list(frames),
            context=dict(context) if context is not None else None,
        )
    )
    return _prompt_of(_sent(client))


class ReflexFrameBound(unittest.TestCase):
    def test_a_reflex_request_sends_at_most_three_frames_when_given_five(self):
        for name, images in (
            ("openrouter", _openrouter_images("reflex", frames=CONTEXT_FRAMES)),
            ("gemini", _gemini_images("reflex", frames=CONTEXT_FRAMES)),
        ):
            with self.subTest(adapter=name):
                self.assertEqual(len(images), 3)
                self.assertEqual(len(images), _REFLEX_MAX_FRAMES)

    def test_a_deliberate_request_still_sends_all_five(self):
        # No deadline to buy headroom inside, so the extra frames are free recall. Dropping them
        # here would be a pure loss.
        for name, images in (
            ("openrouter", _openrouter_images("deliberate", frames=CONTEXT_FRAMES)),
            ("gemini", _gemini_images("deliberate", frames=CONTEXT_FRAMES)),
        ):
            with self.subTest(adapter=name):
                self.assertEqual(images, [b"f1", b"f2", b"f3", b"f4", b"f5"])

    def test_the_retained_frames_are_the_most_recent_ones_in_order(self):
        # Recency carries the change signal, and the prompt promises "consecutive moments in TIME
        # ORDER (oldest first)" — so the bound drops the OLDEST context and never reorders or
        # resamples what it keeps.
        for name, images in (
            ("openrouter", _openrouter_images("reflex", frames=CONTEXT_FRAMES)),
            ("gemini", _gemini_images("reflex", frames=CONTEXT_FRAMES)),
        ):
            with self.subTest(adapter=name):
                self.assertEqual(images, [b"f3", b"f4", b"f5"])
                self.assertEqual(images[-1], CURRENT_FRAME)

    def test_a_reflex_request_under_the_bound_is_unchanged(self):
        # No padding, no error, no resampling: a 1- or 2-frame strip is already inside the bound.
        for size, expected in (
            (0, [b"f5"]),
            (1, [b"f4", b"f5"]),
            (2, [b"f3", b"f4", b"f5"]),  # exactly at the bound
        ):
            frames = CONTEXT_FRAMES[len(CONTEXT_FRAMES) - size :] if size else []
            with self.subTest(context_frames=size, adapter="openrouter"):
                self.assertEqual(_openrouter_images("reflex", frames=frames), expected)
            with self.subTest(context_frames=size, adapter="gemini"):
                self.assertEqual(_gemini_images("reflex", frames=frames), expected)

    def test_both_adapters_apply_the_identical_bound(self):
        # One shared policy, not two coincidentally-similar ones: at every strip length the two
        # providers must ingest the same evidence AND that evidence must be the bounded window.
        for size in range(len(CONTEXT_FRAMES) + 1):
            frames = CONTEXT_FRAMES[:size]
            reflex_expected = [*frames, CURRENT_FRAME][-_REFLEX_MAX_FRAMES:]
            deliberate_expected = [*frames, CURRENT_FRAME]
            for mode, expected in (
                ("reflex", reflex_expected),
                ("deliberate", deliberate_expected),
            ):
                with self.subTest(mode=mode, context_frames=size):
                    openrouter_images = _openrouter_images(mode, frames=frames)
                    gemini_images = _gemini_images(mode, frames=frames)
                    self.assertEqual(openrouter_images, gemini_images)
                    self.assertEqual(openrouter_images, expected)

    def test_the_prompt_states_the_number_of_frames_actually_sent(self):
        # A bound that trimmed the payload but left the question claiming five images would be
        # lying to the model about its own evidence.
        adapter, client = _openrouter(TERSE_REPLY)
        asyncio.run(
            _pool(adapter, mode="reflex", shape="ordered_images").judge(
                "the lid is closed", CURRENT_FRAME, frames=list(CONTEXT_FRAMES)
            )
        )
        self.assertIn("These 3 images are consecutive moments", _prompt_of(_sent(client)))
        self.assertIn(
            "These 3 images are consecutive moments",
            _gemini_prompt("reflex", frames=CONTEXT_FRAMES),
        )
        self.assertIn(
            "These 5 images are consecutive moments",
            _gemini_prompt("deliberate", frames=CONTEXT_FRAMES),
        )

    def test_timestamp_labels_are_sliced_with_the_frames_and_keep_the_clip_origin(self):
        # agent._fire_clip_occurrences decodes a reported t_start against
        # _percept_temporal_span_s — the FULL captured span — so the retained labels must stay in
        # the clip's original frame of reference. Re-basing them to 0 here would shift every
        # reported occurrence earlier by exactly the dropped prefix.
        adapter, client = _openrouter(TERSE_REPLY)
        clip_context = {
            "event_chain": True,
            "_percept_temporal_image_timestamps_s": (0.0, 0.4, 0.8, 1.2, 1.6),
        }
        asyncio.run(
            _pool(adapter, mode="reflex", shape="video_clip").judge(
                "one discrete closing",
                CURRENT_FRAME,
                frames=list(CONTEXT_FRAMES),
                context=clip_context,
            )
        )
        user = next(m for m in _sent(client)["messages"] if m["role"] == "user")
        labels = [
            block["text"]
            for block in user["content"]
            if block.get("type") == "text" and block["text"].startswith("Temporal frame")
        ]
        self.assertEqual(
            labels,
            [
                "Temporal frame 1/3 at t=0.8s.",
                "Temporal frame 2/3 at t=1.2s.",
                "Temporal frame 3/3 at t=1.6s.",
            ],
        )
        self.assertEqual(
            len([b for b in user["content"] if b.get("type") == "image_url"]), 3
        )

        # ...and the deliberate clip is untouched: five frames, five labels, origin at 0.
        deliberate_adapter, deliberate_client = _openrouter(TERSE_REPLY)
        asyncio.run(
            _pool(deliberate_adapter, mode="deliberate", shape="video_clip").judge(
                "one discrete closing",
                CURRENT_FRAME,
                frames=list(CONTEXT_FRAMES),
                context=clip_context,
            )
        )
        user = next(
            m for m in _sent(deliberate_client)["messages"] if m["role"] == "user"
        )
        self.assertEqual(
            [
                block["text"]
                for block in user["content"]
                if block.get("type") == "text"
                and block["text"].startswith("Temporal frame")
            ],
            [
                "Temporal frame 1/5 at t=0s.",
                "Temporal frame 2/5 at t=0.4s.",
                "Temporal frame 3/5 at t=0.8s.",
                "Temporal frame 4/5 at t=1.2s.",
                "Temporal frame 5/5 at t=1.6s.",
            ],
        )

    def test_a_directly_constructed_adapter_keeps_every_frame(self):
        # No pool, no stamp, no bound — an SDK user calling the adapter straight is unaffected.
        adapter, client = _openrouter(TERSE_REPLY)
        asyncio.run(
            adapter.judge("x?", CURRENT_FRAME, frames=list(CONTEXT_FRAMES))
        )
        user = next(m for m in _sent(client)["messages"] if m["role"] == "user")
        self.assertEqual(
            len([b for b in user["content"] if b.get("type") == "image_url"]), 5
        )

    def test_a_deliberate_temporal_request_is_byte_for_byte_the_direct_request(self):
        # The whole payload, not just the image count: images, per-frame timestamp labels, the
        # question text and the output budget must all be indistinguishable from the pre-bound
        # request. The bound is a reflex-route property and nothing else.
        clip_context = {
            "event_chain": True,
            "_percept_temporal_image_timestamps_s": (0.0, 0.4, 0.8, 1.2, 1.6),
            "_percept_temporal_span_s": 1.6,
        }
        pooled_adapter, pooled_client = _openrouter(VERBOSE_REPLY)
        asyncio.run(
            _pool(pooled_adapter, mode="deliberate", shape="video_clip").judge(
                "one discrete closing",
                CURRENT_FRAME,
                frames=list(CONTEXT_FRAMES),
                context=dict(clip_context),
            )
        )
        direct_adapter, direct_client = _openrouter(VERBOSE_REPLY)
        asyncio.run(
            direct_adapter.judge(
                "one discrete closing",
                CURRENT_FRAME,
                frames=list(CONTEXT_FRAMES),
                context=dict(clip_context),
            )
        )
        self.assertEqual(_sent(pooled_client), _sent(direct_client))
        self.assertEqual(_sent(pooled_client)["max_tokens"], _DELIBERATE_MAX_TOKENS)


# --------------------------------------------------------------------------------------------
# a bounded occurrence request still answers on the CLIP'S timeline
# --------------------------------------------------------------------------------------------
#
# The bound drops the oldest frames, so the images cover only the TAIL of the captured window. The
# reply is decoded against the whole window — agent._fire_clip_occurrences derives
# clip_start_t = source_t - _percept_temporal_span_s and behavior.fire_clip_events reports each
# occurrence at clip_start_t + t_start — so a request that announces the full span while showing
# only its tail gets its occurrences attributed EARLY by up to the dropped prefix.
#
# The OpenRouter timestamped transport already pins this per frame (t= labels in clip coordinates).
# Nothing else does: the Gemini adapter has no per-frame labels at all, so EVERY Gemini occurrence
# request is on the un-timestamped path. These tests pin the origin being stated there instead.

# 5 frames uniformly sampled across a 1.6s window; the reflex bound keeps the last 3, i.e. the
# window from t=0.8s onward. 0.8 is exactly what the timestamped transport labels the first
# retained frame (see test_timestamp_labels_are_sliced_with_the_frames_and_keep_the_clip_origin).
CLIP_SPAN_S = 1.6
CLIP_OFFSETS_S = (0.0, 0.4, 0.8, 1.2, 1.6)
RETAINED_ORIGIN_S = 0.8


def _clip_context(**extra) -> dict:
    return {
        "event_chain": True,
        "_percept_temporal_span_s": CLIP_SPAN_S,
        "_percept_temporal_offsets_s": CLIP_OFFSETS_S,
        **extra,
    }


_ORIGIN_SENTENCE = (
    "The images shown cover only the later part of that window: the first image is at "
    "t=0.8s and the last at t=1.6s. Report every time on the full 1.6-second timeline — "
    "t=0 is the START of the window, not the first image shown."
)


class BoundedClipStatesItsOrigin(unittest.TestCase):
    def test_an_untimestamped_reflex_occurrence_request_states_the_retained_origin(self):
        for name, prompt in (
            (
                "openrouter",
                _openrouter_prompt("reflex", frames=CONTEXT_FRAMES, context=_clip_context()),
            ),
            (
                "gemini",
                _gemini_prompt(
                    "reflex",
                    frames=CONTEXT_FRAMES,
                    context=_clip_context(),
                    shape="video_clip",
                ),
            ),
        ):
            with self.subTest(adapter=name):
                # the span the decoder uses is still announced, unchanged...
                self.assertIn("The evidence spans 1.6 seconds.", prompt)
                # ...and the evidence is now placed inside it
                self.assertIn(_ORIGIN_SENTENCE, prompt)
                self.assertIn("These 3 images are consecutive moments", prompt)

    def test_the_stated_origin_matches_what_the_timestamped_transport_labels(self):
        # One clip, two transports, one answer: whether the retained window's start reaches the
        # model as a per-frame ``t=`` label or as prose, it must be the SAME instant, because both
        # replies are decoded against the same _percept_temporal_span_s.
        adapter, client = _openrouter(TERSE_REPLY)
        asyncio.run(
            _pool(adapter, mode="reflex", shape="video_clip").judge(
                "one discrete closing",
                CURRENT_FRAME,
                frames=list(CONTEXT_FRAMES),
                context=_clip_context(
                    _percept_temporal_image_timestamps_s=CLIP_OFFSETS_S,
                ),
            )
        )
        user = next(m for m in _sent(client)["messages"] if m["role"] == "user")
        first_label = next(
            block["text"]
            for block in user["content"]
            if block.get("type") == "text" and block["text"].startswith("Temporal frame")
        )
        self.assertEqual(first_label, f"Temporal frame 1/3 at t={RETAINED_ORIGIN_S:.6g}s.")
        self.assertIn(
            f"the first image is at t={RETAINED_ORIGIN_S:.6g}s",
            _gemini_prompt(
                "reflex",
                frames=CONTEXT_FRAMES,
                context=_clip_context(),
                shape="video_clip",
            ),
        )

    def test_a_timestamped_request_is_not_given_the_prose_origin(self):
        # Per-frame labels already pin the timeline exactly; restating it in prose would be a
        # second, redundant frame of reference on the path the commit deliberately left correct.
        prompt = _openrouter_prompt(
            "reflex",
            frames=CONTEXT_FRAMES,
            context=_clip_context(_percept_temporal_image_timestamps_s=CLIP_OFFSETS_S),
        )
        self.assertIn("The evidence spans 1.6 seconds.", prompt)
        self.assertNotIn("The images shown cover only", prompt)

    def test_the_exact_per_frame_provenance_beats_the_uniform_assumption(self):
        # Frames are not always evenly spaced across the window (window_s clipping, "recent"
        # sampling). _percept_temporal_offsets_s is the agent's real selection provenance, so the
        # stated origin follows it rather than span * dropped / (n - 1).
        uneven = {
            "event_chain": True,
            "_percept_temporal_span_s": 2.0,
            "_percept_temporal_offsets_s": (0.0, 1.2, 1.5, 1.7, 2.0),
        }
        for name, prompt in (
            ("openrouter", _openrouter_prompt("reflex", frames=CONTEXT_FRAMES, context=uneven)),
            (
                "gemini",
                _gemini_prompt(
                    "reflex", frames=CONTEXT_FRAMES, context=uneven, shape="video_clip"
                ),
            ),
        ):
            with self.subTest(adapter=name):
                self.assertIn("the first image is at t=1.5s and the last at t=2s", prompt)
                self.assertNotIn("t=0.8s", prompt)

    def test_without_provenance_the_origin_falls_back_to_the_sampling_the_prompt_claims(self):
        # No offsets: the request itself tells the model the images are consecutive moments across
        # the announced span, so the honest reading of "5 moments over 1.6s" is 0.4s apart and the
        # retained window starts at 0.8s. Silence would leave the model with no origin at all.
        bare = {"event_chain": True, "_percept_temporal_span_s": CLIP_SPAN_S}
        for name, prompt in (
            ("openrouter", _openrouter_prompt("reflex", frames=CONTEXT_FRAMES, context=bare)),
            (
                "gemini",
                _gemini_prompt("reflex", frames=CONTEXT_FRAMES, context=bare, shape="video_clip"),
            ),
        ):
            with self.subTest(adapter=name):
                self.assertIn(_ORIGIN_SENTENCE, prompt)

    def test_a_request_that_dropped_nothing_keeps_the_historical_sentence(self):
        # Deliberate keeps every frame, and a strip already inside the bound has no prefix to
        # account for. Both must read exactly as they did before the bound existed.
        short_frames = CONTEXT_FRAMES[-2:]  # 2 context + current == the bound
        short_context = {
            "event_chain": True,
            "_percept_temporal_span_s": CLIP_SPAN_S,
            "_percept_temporal_offsets_s": (0.0, 0.4, 0.8),
        }
        cases = (
            ("openrouter-deliberate", _openrouter_prompt("deliberate", frames=CONTEXT_FRAMES, context=_clip_context())),
            (
                "gemini-deliberate",
                _gemini_prompt(
                    "deliberate", frames=CONTEXT_FRAMES, context=_clip_context(), shape="video_clip"
                ),
            ),
            (
                "openrouter-inside-bound",
                _openrouter_prompt("reflex", frames=short_frames, context=short_context),
            ),
            (
                "gemini-inside-bound",
                _gemini_prompt(
                    "reflex", frames=short_frames, context=short_context, shape="video_clip"
                ),
            ),
        )
        for name, prompt in cases:
            with self.subTest(case=name):
                self.assertIn("The evidence spans 1.6 seconds.", prompt)
                self.assertNotIn("The images shown cover only", prompt)

    def test_a_deliberate_untimestamped_clip_is_byte_for_byte_the_direct_request(self):
        # The un-timestamped clip request is where the origin sentence lives, so pin that the
        # deliberate route there is still indistinguishable from an unpooled direct call.
        context = _clip_context()
        pooled_adapter, pooled_client = _openrouter(VERBOSE_REPLY)
        asyncio.run(
            _pool(pooled_adapter, mode="deliberate", shape="video_clip").judge(
                "one discrete closing",
                CURRENT_FRAME,
                frames=list(CONTEXT_FRAMES),
                context=dict(context),
            )
        )
        direct_adapter, direct_client = _openrouter(VERBOSE_REPLY)
        asyncio.run(
            direct_adapter.judge(
                "one discrete closing",
                CURRENT_FRAME,
                frames=list(CONTEXT_FRAMES),
                context=dict(context),
            )
        )
        self.assertEqual(_sent(pooled_client), _sent(direct_client))

    def test_an_unusable_span_still_produces_no_timeline_sentence(self):
        # A clip with no announced span is never decoded into occurrences at all
        # (agent._fire_clip_occurrences returns [] without one), so there is no timeline to state.
        for name, prompt in (
            (
                "openrouter",
                _openrouter_prompt("reflex", frames=CONTEXT_FRAMES, context={"event_chain": True}),
            ),
            (
                "gemini",
                _gemini_prompt(
                    "reflex",
                    frames=CONTEXT_FRAMES,
                    context={"event_chain": True},
                    shape="video_clip",
                ),
            ),
        ):
            with self.subTest(adapter=name):
                self.assertNotIn("The evidence spans", prompt)
                self.assertNotIn("The images shown cover only", prompt)


# --------------------------------------------------------------------------------------------
# the parser accepts both shapes and still fails CLOSED
# --------------------------------------------------------------------------------------------


class ParserAcceptsTerseAndVerbose(unittest.TestCase):
    def test_terse_shape_parses_into_a_usable_verdict(self):
        verdict = _parse_verdict('{"satisfied":false,"confidence":0.9}')
        self.assertFalse(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.9)
        # evidence is absent, not invalid: the contract says Verdict.evidence is a str
        self.assertEqual(verdict.evidence, "")
        self.assertIsInstance(verdict.evidence, str)

    def test_verbose_shape_still_parses_unchanged(self):
        verdict = _parse_verdict(
            '{"satisfied":true,"confidence":0.82,"evidence":"the lid is down"}'
        )
        self.assertTrue(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.82)
        self.assertEqual(verdict.evidence, "the lid is down")

    def test_a_provider_that_ignores_the_terse_ask_is_still_parsed(self):
        # the ask is an instruction, not a guarantee — prose must never become a parse miss
        verdict = _parse_verdict(
            'Sure! ```json\n{"satisfied": true, "confidence": 0.75, '
            '"evidence": "the hand withdraws"}\n```'
        )
        self.assertTrue(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.75)

    def test_a_reply_truncated_by_the_small_budget_salvages_both_decisive_fields(self):
        verdict = _parse_verdict('{"satisfied": true, "confidence": 0.88, "evidence": "a hand wa')
        self.assertTrue(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.88)

    def test_malformed_replies_fail_closed_to_unknown_not_to_a_verdict(self):
        # ASK is the only safe landing: it neither fires nor touches the gate's rising edge.
        for text in (
            "",
            "not json at all",
            "{",
            "[]",
            '{"satisfied": true}',  # no confidence -> undecidable, never a confident yes
            '{"confidence": 0.99}',  # no answer    -> must never read as a confident NO
            '{"confidence": 0.99, "evidence": "the lid is up"}',
        ):
            with self.subTest(text=text):
                verdict = _parse_verdict(text)
                self.assertEqual(verdict.confidence, 0.0)
                self.assertEqual(
                    Gate(threshold=0.7).decide("k", verdict).action,
                    ASK,
                )

    def test_an_answerless_reply_never_re_arms_the_gate(self):
        # Regression: a stray {"confidence": ...} used to parse as a CONFIDENT NO, which re-arms
        # the rising edge — so the next true read could mint a duplicate fire off a parse miss.
        gate = Gate(threshold=0.7, require_false_first=True)
        self.assertEqual(
            gate.decide("lid", _parse_verdict('{"confidence": 0.99}')).action,
            ASK,
        )
        outcome = gate.decide(
            "lid", _parse_verdict('{"satisfied": true, "confidence": 0.95}')
        )
        self.assertNotEqual(outcome.action, FIRE)


# --------------------------------------------------------------------------------------------
# confidence survives the reflex path all the way to the firing authority
# --------------------------------------------------------------------------------------------


class ConfidenceReachesTheGate(unittest.TestCase):
    def _reflex_verdict(self, reply: str) -> Verdict:
        adapter, _ = _openrouter(reply)
        return asyncio.run(_pool(adapter, mode="reflex").judge("the lid is closed", b"jpeg"))

    def test_a_confident_terse_yes_fires(self):
        verdict = self._reflex_verdict(TERSE_REPLY)
        self.assertAlmostEqual(verdict.confidence, 0.87)
        gate = Gate(threshold=0.7)
        gate.decide("lid", Verdict(satisfied=False, confidence=0.99, evidence=""))
        self.assertEqual(gate.decide("lid", verdict).action, FIRE)

    def test_an_unconfident_terse_yes_asks_instead_of_firing(self):
        verdict = self._reflex_verdict(json.dumps({"satisfied": True, "confidence": 0.55}))
        self.assertAlmostEqual(verdict.confidence, 0.55)
        outcome = Gate(threshold=0.7).decide("lid", verdict)
        self.assertEqual(outcome.action, ASK)
        self.assertEqual(outcome.reason, "low_confidence")

    def test_a_terse_verdict_is_a_valid_pool_verdict_not_a_malformed_one(self):
        # JudgePool._valid_verdict requires evidence to be a str; the terse shape leaves it "",
        # so a reflex reply must route as decisive rather than degrade to the UNKNOWN fallback.
        adapter, _ = _openrouter(TERSE_REPLY)
        pool = _pool(adapter, mode="reflex")
        asyncio.run(pool.judge("the lid is closed", b"jpeg"))
        receipt = pool.receipts()[-1]
        self.assertEqual(receipt.outcome, "decisive")
        self.assertAlmostEqual(receipt.confidence, 0.87)


if __name__ == "__main__":
    unittest.main()
