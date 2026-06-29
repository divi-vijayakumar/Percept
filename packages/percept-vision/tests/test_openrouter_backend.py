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

    def test_bad_reply_degrades_to_low_confidence_unknown(self):
        v = OpenRouterVision(_settings(), client=_FakeClient("not json at all"))
        verdict = asyncio.run(v.judge("x?", b"jpeg"))
        self.assertFalse(verdict.satisfied)
        self.assertEqual(verdict.confidence, 0.0)

    def test_registry_resolves_openrouter(self):
        self.assertIn("openrouter", registry._REGISTRY["vision"])


if __name__ == "__main__":
    unittest.main()
