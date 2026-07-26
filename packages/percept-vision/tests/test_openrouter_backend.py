"""OpenRouter vision backend (L1) — imports without the SDK/key, formats images as OpenAI image_url
blocks, parses the strict-JSON verdict, and degrades a bad reply to low-confidence unknown. No network:
the client is INJECTED, mirroring the anthropic/gemini L2-fixture pattern."""
from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from percept import registry
from percept.adapters import Verdict
from percept.backends.openrouter import OpenRouterVision, _data_url, _text_of


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def _settings(model: str = "qwen/qwen2.5-vl-72b-instruct"):
    return SimpleNamespace(models=SimpleNamespace(vision_model=model))


class OpenRouterBackendTests(unittest.TestCase):
    def test_constructs_without_sdk_or_key(self):
        # an injected client means no SDK import + no key needed to build the adapter (L2 boundary)
        v = OpenRouterVision(_settings(), client=object())
        self.assertEqual(v._model, "qwen/qwen2.5-vl-72b-instruct")

    def test_data_url_handles_frame_shapes(self):
        self.assertTrue(_data_url(b"abc").startswith("data:image/jpeg;base64,"))
        self.assertTrue(_data_url(("image/png", b"abc")).startswith("data:image/png;base64,"))
        block = {"type": "image", "source": {"media_type": "image/jpeg", "data": "QUJD"}}
        self.assertEqual(_data_url(block), "data:image/jpeg;base64,QUJD")
        with self.assertRaises(TypeError):
            _data_url("not-an-image")

    def test_text_of_reads_openai_shape(self):
        resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
        self.assertEqual(_text_of(resp), "hi")
        self.assertEqual(_text_of({"choices": [{"message": {"content": "yo"}}]}), "yo")

    def test_judge_parses_verdict_and_sends_ordered_images(self):
        content = json.dumps({"satisfied": True, "confidence": 0.9, "evidence": "a hand waves"})
        client = _FakeClient(content)
        v = OpenRouterVision(_settings(), client=client)
        verdict = asyncio.run(v.judge("someone waves", b"current", frames=[b"earlier", b"current"]))
        self.assertIsInstance(verdict, Verdict)
        self.assertTrue(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.9)
        sent = client.chat.completions.calls[0]
        self.assertEqual(sent["model"], "qwen/qwen2.5-vl-72b-instruct")
        user = [m for m in sent["messages"] if m["role"] == "user"][0]
        imgs = [c for c in user["content"] if c.get("type") == "image_url"]
        self.assertEqual(len(imgs), 2)  # earlier-context frame + current frame (current last)
        self.assertTrue(any(m["role"] == "system" for m in sent["messages"]))

    def test_native_video_is_one_video_url_and_never_an_image_strip(self):
        content = json.dumps(
            {
                "satisfied": True,
                "confidence": 0.94,
                "evidence": "one complete putting occurrence",
                "action_present": False,
                "phase": "post",
            }
        )
        client = _FakeClient(content)
        verdict = asyncio.run(
            OpenRouterVision(_settings(), client=client).judge(
                "one discrete putting occurrence",
                ("video/mp4", b"native-mp4"),
                context={
                    "event_chain": True,
                    "_percept_evidence_shape": "video_clip",
                },
            )
        )
        self.assertEqual(verdict.phase, "post")
        user = next(
            message
            for message in client.chat.completions.calls[0]["messages"]
            if message["role"] == "user"
        )
        videos = [
            block for block in user["content"] if block.get("type") == "video_url"
        ]
        images = [
            block for block in user["content"] if block.get("type") == "image_url"
        ]
        self.assertEqual(len(videos), 1)
        self.assertEqual(images, [])
        self.assertTrue(
            videos[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        )
        prompt = " ".join(
            block["text"]
            for block in user["content"]
            if block.get("type") == "text"
        )
        self.assertIn("short native video clip", prompt)

    def test_native_video_rejects_mixed_ordered_image_fallback(self):
        with self.assertRaisesRegex(ValueError, "exact evidence shape"):
            asyncio.run(
                OpenRouterVision(
                    _settings(),
                    client=_FakeClient("{}"),
                ).judge(
                    "one discrete action",
                    ("video/mp4", b"native-mp4"),
                    frames=[b"earlier"],
                    context={"event_chain": True},
                )
            )

    def test_event_chain_prompt_and_structured_phase_round_trip(self):
        content = json.dumps(
            {
                "satisfied": True,
                "confidence": 0.93,
                "evidence": "hand and object visibly move through the action",
                "action_present": True,
                "phase": "during",
                "events": [
                    {"t_start": 0.1, "t_end": 0.3, "label": "first"},
                    {"t_start": 0.6, "t_end": 0.8, "label": "second"},
                ],
            }
        )
        client = _FakeClient(content)
        verdict = asyncio.run(
            OpenRouterVision(_settings(), client=client).judge(
                "one discrete putting occurrence",
                b"current",
                frames=[b"before"],
                context={"event_chain": True},
            )
        )
        self.assertEqual((verdict.phase, verdict.action_present), ("during", True))
        self.assertEqual(
            [(event.t_start, event.t_end, event.label) for event in verdict.events],
            [(0.1, 0.3, "first"), (0.6, 0.8, "second")],
        )
        user = next(
            message
            for message in client.chat.completions.calls[0]["messages"]
            if message["role"] == "user"
        )
        prompt = " ".join(
            block["text"] for block in user["content"] if block.get("type") == "text"
        )
        self.assertIn("pre (the occurrence has not begun)", prompt)
        self.assertIn("Do not infer an action from a static", prompt)
        self.assertIn('"action_present", "phase"', prompt)
        self.assertIn("List every grounded occurrence", prompt)

    def test_bad_reply_degrades_to_low_confidence_unknown(self):
        v = OpenRouterVision(_settings(), client=_FakeClient("not json at all"))
        verdict = asyncio.run(v.judge("x?", b"jpeg"))
        self.assertFalse(verdict.satisfied)
        self.assertEqual(verdict.confidence, 0.0)

    def test_process_qa_returns_rewritten_running_answer(self):
        content = json.dumps({"updated_summary": "peel; slice", "satisfied": False, "confidence": 0.8})
        v = OpenRouterVision(_settings(), client=_FakeClient(content))
        verdict = asyncio.run(v.judge("steps?", b"cur", frames=[b"a", b"cur"],
                                      context={"process_qa": True, "running_answer": "peel"}))
        self.assertEqual(verdict.running_answer, "peel; slice")
        self.assertAlmostEqual(verdict.confidence, 0.8)

    def test_process_qa_parse_miss_preserves_prior(self):
        v = OpenRouterVision(_settings(), client=_FakeClient("garbled"))
        verdict = asyncio.run(v.judge("steps?", b"cur", context={"process_qa": True, "running_answer": "peel"}))
        self.assertEqual(verdict.running_answer, "peel")  # never regress the accumulation

    def test_registry_resolves_openrouter(self):
        self.assertIn("openrouter", registry._REGISTRY["vision"])


if __name__ == "__main__":
    unittest.main()
