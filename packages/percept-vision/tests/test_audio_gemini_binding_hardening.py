from __future__ import annotations

import asyncio
import gc
import sys
import types
import unittest
from unittest.mock import patch

from percept.backends import gemini as gemini_backend
from percept.backends.gemini import GeminiVision, _GeminiBase
from percept.contracts import AudioFrame
from percept.audio_input import prepare_audio_judge_binding


class _Response:
    text = '{"satisfied": true, "confidence": 0.99, "evidence": "heard"}'
    usage_metadata = None


class _Models:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_content(self, **kwargs: object) -> _Response:
        self.calls += 1
        return _Response()


class _Client:
    def __init__(self) -> None:
        self.models = _Models()
        self.aio = types.SimpleNamespace(models=self.models)


def _audio() -> AudioFrame:
    return AudioFrame(
        pcm=b"\0\0",
        sample_rate=1,
        channels=1,
        sample_format="s16le",
        t_start=0.0,
        t_end=1.0,
    )


async def _dispatch(binding: object) -> object:
    return await binding.dispatch("someone speaks", None, audio=_audio())


def _lazy_google(created: list[_Client]) -> dict[str, types.ModuleType]:
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

    def make_client(**kwargs: object) -> _Client:
        client = _Client()
        created.append(client)
        return client

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    google_types = types.ModuleType("google.genai.types")
    genai.Client = make_client
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


class GeminiBindingHardeningTests(unittest.TestCase):
    def test_post_apply_client_swap_fails_before_poison_client_is_called(self) -> None:
        accepted = _Client()
        poison = _Client()
        adapter = GeminiVision(client=accepted)
        binding = prepare_audio_judge_binding(
            adapter,
            mode="audio_only",
            application_key="watch",
        )

        adapter._client = poison
        with self.assertRaisesRegex(RuntimeError, "state was mutated"):
            asyncio.run(_dispatch(binding))
        self.assertEqual(accepted.models.calls, 0)
        self.assertEqual(poison.models.calls, 0)

    def test_post_apply_nested_client_route_cannot_redirect_captured_provider(self) -> None:
        accepted = _Client()
        poison = _Client()
        adapter = GeminiVision(client=accepted)
        binding = prepare_audio_judge_binding(
            adapter,
            mode="audio_only",
            application_key="watch",
        )

        adapter._client.aio.models = poison.models
        result = asyncio.run(_dispatch(binding))
        self.assertTrue(result.satisfied)
        self.assertEqual(accepted.models.calls, 1)
        self.assertEqual(poison.models.calls, 0)

    def test_post_apply_instance_state_and_method_shadows_fail_closed(self) -> None:
        mutations = (
            lambda adapter: setattr(adapter, "_model", "poison-model"),
            lambda adapter: setattr(adapter, "_settings", object()),
            lambda adapter: setattr(adapter, "_breaker", object()),
            lambda adapter: setattr(adapter, "judge", lambda *a, **k: None),
            lambda adapter: setattr(adapter, "_ensure_client", lambda: _Client()),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                client = _Client()
                adapter = GeminiVision(client=client)
                binding = prepare_audio_judge_binding(
                    adapter,
                    mode="audio_only",
                    application_key="watch",
                )
                mutate(adapter)
                with self.assertRaises(RuntimeError):
                    asyncio.run(_dispatch(binding))
                self.assertEqual(client.models.calls, 0)

    def test_post_apply_class_judge_and_ensure_client_mutations_fail_closed(self) -> None:
        async def poisoned_judge(*args: object, **kwargs: object) -> object:
            raise AssertionError("mutated class judge ran")

        def poisoned_ensure(*args: object, **kwargs: object) -> object:
            raise AssertionError("mutated ensure-client ran")

        cases = (
            (GeminiVision, "judge", poisoned_judge),
            (_GeminiBase, "_ensure_client", poisoned_ensure),
        )
        for owner, name, poison in cases:
            with self.subTest(name=name):
                client = _Client()
                adapter = GeminiVision(client=client)
                binding = prepare_audio_judge_binding(
                    adapter,
                    mode="audio_only",
                    application_key="watch",
                )
                original = owner.__dict__[name]
                setattr(owner, name, poison)
                try:
                    with self.assertRaisesRegex(RuntimeError, "dependency was mutated"):
                        asyncio.run(_dispatch(binding))
                finally:
                    setattr(owner, name, original)
                self.assertEqual(client.models.calls, 0)

    def test_post_apply_module_helper_mutation_fails_closed(self) -> None:
        client = _Client()
        adapter = GeminiVision(client=client)
        binding = prepare_audio_judge_binding(
            adapter,
            mode="audio_only",
            application_key="watch",
        )
        original = gemini_backend._audio_part

        def poisoned(*args: object, **kwargs: object) -> object:
            raise AssertionError("mutated module helper ran")

        gemini_backend._audio_part = poisoned
        try:
            with self.assertRaisesRegex(RuntimeError, "dependency was mutated"):
                asyncio.run(_dispatch(binding))
        finally:
            gemini_backend._audio_part = original
        self.assertEqual(client.models.calls, 0)

    def test_trusted_lazy_client_initialization_is_allowed_once(self) -> None:
        created: list[_Client] = []
        with patch.dict(sys.modules, _lazy_google(created)):
            adapter = GeminiVision(client=None)
            binding = prepare_audio_judge_binding(
                adapter,
                mode="audio_only",
                application_key="watch",
            )
            first = asyncio.run(_dispatch(binding))
            second = asyncio.run(_dispatch(binding))

        self.assertTrue(first.satisfied)
        self.assertTrue(second.satisfied)
        self.assertEqual(len(created), 1)
        self.assertIs(adapter._client, created[0])
        self.assertEqual(created[0].models.calls, 2)

    def test_two_lazy_bindings_share_one_trusted_initialization_in_both_orders(self) -> None:
        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                created: list[_Client] = []
                with patch.dict(sys.modules, _lazy_google(created)):
                    adapter = GeminiVision(client=None)
                    bindings = [
                        prepare_audio_judge_binding(
                            adapter,
                            mode="audio_only",
                            application_key=f"watch-{index}",
                        )
                        for index in range(2)
                    ]
                    first = asyncio.run(_dispatch(bindings[order[0]]))
                    second = asyncio.run(_dispatch(bindings[order[1]]))
                self.assertTrue(first.satisfied)
                self.assertTrue(second.satisfied)
                self.assertEqual(len(created), 1)
                self.assertEqual(created[0].models.calls, 2)

    def test_concurrent_first_dispatches_share_one_lazy_client(self) -> None:
        created: list[_Client] = []

        async def drive(bindings: list[object]) -> list[object]:
            return await asyncio.gather(*(_dispatch(item) for item in bindings))

        with patch.dict(sys.modules, _lazy_google(created)):
            adapter = GeminiVision(client=None)
            bindings = [
                prepare_audio_judge_binding(
                    adapter,
                    mode="audio_only",
                    application_key=f"watch-{index}",
                )
                for index in range(2)
            ]
            results = asyncio.run(drive(bindings))
        self.assertTrue(all(item.satisfied for item in results))
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].models.calls, 2)

    def test_post_lazy_init_client_replacement_fails_for_every_binding(self) -> None:
        created: list[_Client] = []
        poison = _Client()
        with patch.dict(sys.modules, _lazy_google(created)):
            adapter = GeminiVision(client=None)
            first = prepare_audio_judge_binding(
                adapter, mode="audio_only", application_key="first"
            )
            second = prepare_audio_judge_binding(
                adapter, mode="audio_only", application_key="second"
            )
            self.assertTrue(asyncio.run(_dispatch(first)).satisfied)
            adapter._client = poison
            with self.assertRaisesRegex(RuntimeError, "state was mutated"):
                asyncio.run(_dispatch(second))
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].models.calls, 1)
        self.assertEqual(poison.models.calls, 0)

    def test_rolled_back_preparation_does_not_poison_later_adapter_state(self) -> None:
        adapter = GeminiVision(client=None)
        abandoned = prepare_audio_judge_binding(
            adapter, mode="audio_only", application_key="rolled-back"
        )
        del abandoned
        gc.collect()
        accepted = _Client()
        adapter._client = accepted

        binding = prepare_audio_judge_binding(
            adapter, mode="audio_only", application_key="later-valid"
        )
        result = asyncio.run(_dispatch(binding))

        self.assertTrue(result.satisfied)
        self.assertEqual(accepted.models.calls, 1)


if __name__ == "__main__":
    unittest.main()
