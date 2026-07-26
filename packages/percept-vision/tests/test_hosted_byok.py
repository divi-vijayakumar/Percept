"""Hosted BYOK conformance/adversarial suite — fake transports only, never network or real keys."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import percept
from percept import (
    HOSTED_BYOK_CONTRACT_VERSION,
    HostedByokAdapterFactory,
    HostedByokCancelled,
    HostedByokContractError,
    HostedByokDeadlineExceeded,
    HostedByokGeminiClient,
    HostedByokInvocationError,
    HostedByokRequest,
    HostedByokUnsupportedError,
    Goal,
    Percept,
)
from percept.adapters import Verdict
from percept.config import Settings

SECRET_A = "sk-secret-a-never-disclose"
SECRET_B = "sk-secret-b-never-disclose"
_HOSTED_ENV_KEYS = {
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_AISTUDIO_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "PERCEPT_VISION_MODEL",
    "PERCEPT_VISION_TIMEOUT_S",
    "PERCEPT_STRIP_PROMPT",
}


def _request(
    *,
    provider: str = "openrouter",
    model: str = "qwen/qwen3-vl-32b-instruct",
    credential: str = SECRET_A,
    lease: str = "lease-a",
    principal: str = "principal-a",
    **kwargs,
) -> HostedByokRequest:
    return HostedByokRequest(
        provider=provider,
        model=model,
        credential=credential,
        credential_lease_id=lease,
        principal_id=principal,
        **kwargs,
    )


class _Backend:
    def __init__(self, result: Verdict | None = None) -> None:
        self.result = result or Verdict(True, 0.9, "fake")
        self.calls: list[tuple] = []

    async def judge(self, condition, frame, **kwargs):
        self.calls.append((condition, frame, kwargs))
        return self.result


class _OpenRouterCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps({"satisfied": True, "confidence": 0.91, "evidence": "fake"})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class _OpenRouterClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_OpenRouterCompletions())
        self.closed = False

    async def close(self):
        self.closed = True


class _GeminiPart:
    @staticmethod
    def from_bytes(*, data, mime_type):
        return ("part", mime_type, bytes(data))


class _GeminiConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _GeminiTypes:
    Part = _GeminiPart
    GenerateContentConfig = _GeminiConfig


class _GeminiModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps({"satisfied": False, "confidence": 0.83, "evidence": "fake"})
        return SimpleNamespace(text=text, usage_metadata=None)


class _GeminiClient:
    def __init__(self) -> None:
        self.aio = SimpleNamespace(models=_GeminiModels())
        self.closed = False

    async def close(self):
        self.closed = True


def test_request_is_immutable_redacted_unhashable_and_versioned():
    request = _request()
    assert SECRET_A not in repr(request)
    assert "credential=<redacted>" in repr(request)
    assert request.contract_version == HOSTED_BYOK_CONTRACT_VERSION
    with pytest.raises(FrozenInstanceError):
        request.model = "other/model"
    with pytest.raises(TypeError):
        hash(request)
    with pytest.raises(HostedByokContractError, match="contract version"):
        _request(contract_version="2.0.0")


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("gemini", "gemini-3.5-flash"),
        ("openrouter", "qwen/qwen3-vl-32b-instruct"),
    ],
)
def test_explicit_credential_path_reads_no_environment(monkeypatch, provider, model):
    seen: list[tuple[str, str]] = []
    original_get = __import__("os")._Environ.get
    for key in _HOSTED_ENV_KEYS:
        monkeypatch.setenv(key, "poisoned-shared-process-value")

    def forbidden_environment_read(environ, key, *args, **kwargs):
        if key in _HOSTED_ENV_KEYS:
            raise AssertionError(f"explicit hosted BYOK path read {key}")
        return original_get(environ, key, *args, **kwargs)

    monkeypatch.setattr("os._Environ.get", forbidden_environment_read)

    def make_backend(request, timeout_s):
        seen.append((request.credential, request.model))
        return _Backend()

    verdict = asyncio.run(
        HostedByokAdapterFactory(backend_factories={provider: make_backend}).judge(
            _request(provider=provider, model=model), "condition", b"frame"
        )
    )
    assert verdict.satisfied
    assert seen == [(SECRET_A, model)]


def test_openrouter_client_seam_routes_exact_model_and_explicit_credential(monkeypatch):
    clients: list[_OpenRouterClient] = []
    received: list[tuple[str, str, float]] = []
    original_get = __import__("os")._Environ.get
    for key in _HOSTED_ENV_KEYS:
        monkeypatch.setenv(key, "poisoned-shared-process-value")

    def forbidden_environment_read(environ, key, *args, **kwargs):
        if key in _HOSTED_ENV_KEYS:
            raise AssertionError(f"environment fallback: {key}")
        return original_get(environ, key, *args, **kwargs)

    monkeypatch.setattr("os._Environ.get", forbidden_environment_read)

    def client_factory(request, timeout_s):
        received.append((request.credential, request.model, timeout_s))
        client = _OpenRouterClient()
        clients.append(client)
        return client

    verdict = asyncio.run(
        HostedByokAdapterFactory(openrouter_client_factory=client_factory).judge(
            _request(timeout_s=4.0), "someone waves", b"jpeg"
        )
    )
    assert verdict.satisfied
    assert received == [(SECRET_A, "qwen/qwen3-vl-32b-instruct", 4.0)]
    assert clients[0].chat.completions.calls[0]["model"] == "qwen/qwen3-vl-32b-instruct"
    assert clients[0].closed


def test_gemini_client_seam_routes_exact_model_and_explicit_credential(monkeypatch):
    clients: list[_GeminiClient] = []
    received: list[tuple[str, str, float]] = []
    original_get = __import__("os")._Environ.get
    for key in _HOSTED_ENV_KEYS:
        monkeypatch.setenv(key, "poisoned-shared-process-value")

    def forbidden_environment_read(environ, key, *args, **kwargs):
        if key in _HOSTED_ENV_KEYS:
            raise AssertionError(f"environment fallback: {key}")
        return original_get(environ, key, *args, **kwargs)

    monkeypatch.setattr("os._Environ.get", forbidden_environment_read)

    def client_factory(request, timeout_s):
        received.append((request.credential, request.model, timeout_s))
        client = _GeminiClient()
        clients.append(client)
        return HostedByokGeminiClient(client, _GeminiTypes)

    verdict = asyncio.run(
        HostedByokAdapterFactory(gemini_client_factory=client_factory).judge(
            _request(provider="gemini", model="gemini-3.5-flash", timeout_s=3.0),
            "someone waves",
            b"png",
        )
    )
    assert not verdict.satisfied
    assert received == [(SECRET_A, "gemini-3.5-flash", 3.0)]
    assert clients[0].aio.models.calls[0]["model"] == "gemini-3.5-flash"
    assert clients[0].closed


def test_concurrent_principals_and_rotations_get_fresh_backends_without_bleed():
    built: list[tuple[int, str, str, str]] = []

    class TenantBackend:
        def __init__(self, request):
            self.request = request

        async def judge(self, condition, frame, **kwargs):
            await asyncio.sleep(0)
            return Verdict(
                True,
                0.9,
                f"{self.request.principal_id}/{self.request.credential_lease_id}",
            )

    def factory(request, timeout_s):
        backend = TenantBackend(request)
        built.append(
            (
                id(backend),
                request.principal_id,
                request.credential_lease_id,
                request.credential,
            )
        )
        return backend

    async def run():
        hosted = HostedByokAdapterFactory(backend_factories={"openrouter": factory})
        requests = [
            _request(credential=SECRET_A, lease="lease-a1", principal="alice"),
            _request(credential=SECRET_B, lease="lease-b1", principal="bob"),
            _request(credential=SECRET_B, lease="lease-a2", principal="alice"),
        ]
        return await asyncio.gather(
            *(hosted.judge(request, "condition", b"frame") for request in requests)
        )

    verdicts = asyncio.run(run())
    assert [v.evidence for v in verdicts] == [
        "alice/lease-a1",
        "bob/lease-b1",
        "alice/lease-a2",
    ]
    assert len({item[0] for item in built}) == 3
    assert {(p, lease, secret) for _, p, lease, secret in built} == {
        ("alice", "lease-a1", SECRET_A),
        ("bob", "lease-b1", SECRET_B),
        ("alice", "lease-a2", SECRET_B),
    }


def test_client_factory_is_called_again_for_sequential_rotation():
    clients: list[_OpenRouterClient] = []

    def client_factory(request, timeout_s):
        client = _OpenRouterClient()
        clients.append(client)
        return client

    async def run():
        hosted = HostedByokAdapterFactory(openrouter_client_factory=client_factory)
        await hosted.judge(_request(lease="lease-1"), "a", b"x")
        await hosted.judge(_request(credential=SECRET_B, lease="lease-2"), "b", b"y")

    asyncio.run(run())
    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert all(client.closed for client in clients)


def test_credential_is_redacted_from_backend_error_and_exception_chain():
    leaky_error = type(SECRET_A, (RuntimeError,), {})

    class LeakyBackend:
        async def judge(self, condition, frame, **kwargs):
            raise leaky_error(f"Authorization: Bearer {SECRET_A}")

    async def run():
        hosted = HostedByokAdapterFactory(
            backend_factories={"openrouter": lambda request, timeout: LeakyBackend()}
        )
        await hosted.judge(_request(), "condition", b"frame")

    with pytest.raises(HostedByokInvocationError) as raised:
        asyncio.run(run())
    assert SECRET_A not in str(raised.value)
    assert SECRET_A not in repr(raised.value)
    assert raised.value.__cause__ is None
    assert str(raised.value) == "hosted_byok_provider_failure"


def test_injected_backend_reuse_is_rejected_across_sequential_leases():
    shared = _Backend()
    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: shared}
    )

    async def run():
        await hosted.judge(_request(lease="lease-1"), "condition", b"frame")
        await hosted.judge(_request(credential=SECRET_B, lease="lease-2"), "condition", b"frame")

    with pytest.raises(HostedByokContractError, match="hosted_byok_injected_component_reused"):
        asyncio.run(run())


def test_injected_client_reuse_is_rejected_across_principals():
    shared = _OpenRouterClient()
    hosted = HostedByokAdapterFactory(openrouter_client_factory=lambda request, timeout: shared)

    async def run():
        await hosted.judge(_request(principal="alice"), "condition", b"frame")
        await hosted.judge(_request(principal="bob", lease="lease-b"), "condition", b"frame")

    with pytest.raises(HostedByokContractError, match="hosted_byok_injected_component_reused"):
        asyncio.run(run())


def test_timeout_cancels_provider_work():
    cancelled = asyncio.Event()

    class SlowBackend:
        async def judge(self, condition, frame, **kwargs):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: SlowBackend()}
    )
    with pytest.raises(HostedByokDeadlineExceeded):
        asyncio.run(hosted.judge(_request(timeout_s=0.01), "condition", b"frame"))
    assert cancelled.is_set()


def test_absolute_deadline_fails_before_constructing_backend():
    constructed = False

    def factory(request, timeout_s):
        nonlocal constructed
        constructed = True
        return _Backend()

    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": factory}, time_fn=lambda: 100.0
    )
    with pytest.raises(HostedByokDeadlineExceeded, match="before provider"):
        asyncio.run(hosted.judge(_request(deadline_unix_s=99.0), "condition", b"frame"))
    assert not constructed


def test_provider_construction_time_counts_against_monotonic_deadline():
    monotonic_values = iter((10.0, 12.1))

    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: _Backend()},
        monotonic_fn=lambda: next(monotonic_values),
    )
    with pytest.raises(HostedByokDeadlineExceeded, match="during provider construction"):
        asyncio.run(hosted.judge(_request(timeout_s=2.0), "condition", b"frame"))


def test_invocation_cancellation_signal_cancels_provider_work():
    cancellation = asyncio.Event()
    provider_cancelled = asyncio.Event()

    class SlowBackend:
        async def judge(self, condition, frame, **kwargs):
            try:
                await asyncio.sleep(60)
            finally:
                provider_cancelled.set()

    async def run():
        hosted = HostedByokAdapterFactory(
            backend_factories={"openrouter": lambda request, timeout: SlowBackend()}
        )
        task = asyncio.create_task(
            hosted.judge(_request(cancellation=cancellation), "condition", b"frame")
        )
        await asyncio.sleep(0)
        cancellation.set()
        await task

    with pytest.raises(HostedByokCancelled):
        asyncio.run(run())
    assert provider_cancelled.is_set()


def test_cancellation_during_construction_spends_zero_provider_calls():
    cancellation = asyncio.Event()
    judge_calls = 0

    class CountingBackend:
        async def judge(self, condition, frame, **kwargs):
            nonlocal judge_calls
            judge_calls += 1
            return Verdict(True, 0.9, "must not run")

    def factory(request, timeout):
        cancellation.set()
        return CountingBackend()

    hosted = HostedByokAdapterFactory(backend_factories={"openrouter": factory})
    with pytest.raises(HostedByokCancelled):
        asyncio.run(hosted.judge(_request(cancellation=cancellation), "condition", b"frame"))
    assert judge_calls == 0


def test_cancellation_is_bounded_when_provider_temporarily_suppresses_cancel():
    cancellation = asyncio.Event()
    release = asyncio.Event()

    class SuppressingBackend:
        async def judge(self, condition, frame, **kwargs):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await release.wait()
                return Verdict(False, 0.0, "late")

    async def run():
        hosted = HostedByokAdapterFactory(
            backend_factories={"openrouter": lambda request, timeout: SuppressingBackend()},
            cancellation_grace_s=0.01,
        )
        task = asyncio.create_task(
            hosted.judge(_request(cancellation=cancellation), "condition", b"frame")
        )
        await asyncio.sleep(0)
        started = asyncio.get_running_loop().time()
        cancellation.set()
        with pytest.raises(HostedByokCancelled):
            await asyncio.wait_for(task, 0.2)
        elapsed = asyncio.get_running_loop().time() - started
        release.set()
        await asyncio.sleep(0)
        return elapsed

    assert asyncio.run(run()) < 0.15


def test_hanging_client_close_is_separately_bounded():
    release = asyncio.Event()

    class HangingCloseClient(_OpenRouterClient):
        async def close(self):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await release.wait()

    async def run():
        hosted = HostedByokAdapterFactory(
            openrouter_client_factory=lambda request, timeout: HangingCloseClient(),
            close_timeout_s=0.01,
        )
        started = asyncio.get_running_loop().time()
        verdict = await asyncio.wait_for(hosted.judge(_request(), "condition", b"frame"), 0.2)
        elapsed = asyncio.get_running_loop().time() - started
        release.set()
        await asyncio.sleep(0)
        return verdict, elapsed

    verdict, elapsed = asyncio.run(run())
    assert verdict.satisfied
    assert elapsed < 0.15


@pytest.mark.parametrize("fatal", [KeyboardInterrupt, SystemExit])
def test_process_control_base_exceptions_are_not_swallowed(fatal):
    def factory(request, timeout):
        raise fatal()

    hosted = HostedByokAdapterFactory(backend_factories={"openrouter": factory})
    with pytest.raises(fatal):
        asyncio.run(hosted.judge(_request(), "condition", b"frame"))


@pytest.mark.parametrize(
    ("byok_request", "audio", "match"),
    [
        (_request(provider="unknown"), None, "unsupported hosted BYOK provider"),
        (_request(model="not-namespaced"), None, "OpenRouter"),
        (
            _request(provider="gemini", model="qwen/model"),
            None,
            "Gemini",
        ),
        (
            _request(capability="vision.detect"),
            None,
            "does not support capability",
        ),
        (_request(), b"audio", "does not support fused audio"),
    ],
)
def test_unsupported_provider_model_or_capability_fails_before_factory(byok_request, audio, match):
    constructed = False

    def factory(request, timeout_s):
        nonlocal constructed
        constructed = True
        return _Backend()

    hosted = HostedByokAdapterFactory(
        backend_factories={
            "gemini": factory,
            "openrouter": factory,
        }
    )
    with pytest.raises(HostedByokUnsupportedError, match=match):
        asyncio.run(hosted.judge(byok_request, "condition", b"frame", audio=audio))
    assert not constructed


def test_fused_capability_is_gemini_only_and_requires_both_modalities():
    hosted = HostedByokAdapterFactory(
        backend_factories={"gemini": lambda request, timeout: _Backend()}
    )
    request = _request(
        provider="gemini",
        model="gemini-3.5-flash",
        capability="vision.judge.fused",
    )
    with pytest.raises(HostedByokUnsupportedError, match="both image and audio"):
        asyncio.run(hosted.judge(request, "condition", b"frame"))
    verdict = asyncio.run(hosted.judge(request, "condition", b"frame", audio=b"audio"))
    assert verdict.satisfied


def test_request_bound_adapter_runs_through_real_percept_with_fake_transport():
    clients: list[_OpenRouterClient] = []

    def client_factory(request, timeout_s):
        client = _OpenRouterClient()
        clients.append(client)
        return client

    hosted = HostedByokAdapterFactory(openrouter_client_factory=client_factory)
    vision = hosted.bind(_request())
    assert vision.CONTRACT_VERSION == HOSTED_BYOK_CONTRACT_VERSION
    core = Percept.create(
        vision=vision,
        settings=Settings(),
        mode="offline",
        discover_plugins=False,
    )
    core.add_goal(Goal(id="entry", condition="someone enters", say="entry"))

    events = asyncio.run(core.perceive_judged(b"jpeg", t=1.0))
    assert events
    assert len(clients) == 1
    assert clients[0].chat.completions.calls[0]["model"] == "qwen/qwen3-vl-32b-instruct"
    assert clients[0].closed


def test_public_exports_are_cloud_usable():
    expected = {
        "HOSTED_BYOK_CONTRACT_VERSION",
        "HostedByokRequest",
        "HostedByokAdapterFactory",
        "HostedByokVision",
        "HostedByokCancelled",
        "HostedByokDeadlineExceeded",
        "HostedByokUnsupportedError",
    }
    assert expected <= set(percept.__all__)
    for name in expected:
        assert getattr(percept, name) is not None
