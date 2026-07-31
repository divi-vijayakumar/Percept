"""Hosted BYOK conformance/adversarial suite — fake transports only, never network or real keys."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import textwrap
import time
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from collections import UserDict

import pytest

from percept.judge_context_producers import RESERVED_CONTEXT_KEYS

import percept
from percept import (
    HOSTED_BYOK_CONTRACT_VERSION,
    HostedByokAdapterFactory,
    HostedByokCancelled,
    HostedByokContractError,
    HostedByokDeadlineExceeded,
    HostedByokGeminiClient,
    HostedByokInvocationError,
    HostedByokTransportSafety,
    HostedByokRequest,
    HostedByokSubprocessSupervisor,
    HostedByokUnsupportedError,
    Goal,
    Percept,
)
from percept.adapters import Verdict
from percept.contracts import AudioFrame, WatchPlan
from percept.config import Settings
from percept.harness.tier0 import gray_frame
from percept.runtime.worker import run
from percept.runtime.envelope import legacy_audio_envelope, legacy_video_envelope
from percept.audio_input import prepare_audio_judge_binding
from percept.watchplan_apply import apply_watch_plan

SECRET_A = "sk-secret-a-never-disclose"
SECRET_B = "sk-secret-b-never-disclose"
_HOSTED_ENV_KEYS = {
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_AISTUDIO_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "PERCEPT_VISION_MODEL",
    "PERCEPT_VISION_TIMEOUT_S",
    "PERCEPT_STRIP_PROMPT",
}
_HOSTILE_STARTUP_ENV_KEYS = _HOSTED_ENV_KEYS | {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SSLKEYLOGFILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "AWS_CA_BUNDLE",
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
    "NODE_EXTRA_CA_CERTS",
    "GOOGLE_API_USE_MTLS_ENDPOINT",
    "GOOGLE_API_USE_CLIENT_CERTIFICATE",
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


def _write_production_transport_fakes(
    root: Path,
    *,
    behavior: str,
    effect_path: Path,
    call_path: Path,
) -> None:
    package_dir = Path(__file__).resolve().parents[1] / "percept"
    (root / "percept").symlink_to(package_dir, target_is_directory=True)
    (root / "fake_transport_common.py").write_text(
        textwrap.dedent(
            f"""
            import asyncio
            import os
            from pathlib import Path
            import socket
            import subprocess
            import sys
            import threading
            import time

            BLOCKED = {sorted(_HOSTILE_STARTUP_ENV_KEYS)!r}
            BEHAVIOR = {behavior!r}
            EFFECT_PATH = Path({str(effect_path)!r})
            CALL_PATH = Path({str(call_path)!r})

            def late_effect():
                delay = 2.25 if BEHAVIOR in {{"timeout", "slow-close"}} else 0.25
                time.sleep(delay)
                EFFECT_PATH.write_text("late-thread-or-detached-effect")

            def verify_and_attempt_detached(secret, trust_env):
                if secret in "\\0".join(sys.argv):
                    raise RuntimeError("credential appeared in worker argv")
                if any(secret in key or secret in value for key, value in os.environ.items()):
                    raise RuntimeError("credential appeared in worker environment")
                if any(key in os.environ for key in BLOCKED):
                    raise RuntimeError("hostile startup environment survived")
                if trust_env is not False:
                    raise RuntimeError("transport still trusts process environment")
                CALL_PATH.write_text("provider-called")
                threading.Thread(target=late_effect, daemon=False).start()
                try:
                    pid = os.fork()
                except OSError:
                    pid = None
                if pid is not None:
                    if pid == 0:
                        try:
                            os.setsid()
                        except OSError:
                            pass
                        late_effect()
                        os._exit(0)
                    raise RuntimeError("detached child creation was not contained")
                try:
                    escaped = subprocess.Popen(["/bin/true"])
                except OSError:
                    escaped = None
                if escaped is not None:
                    escaped.kill()
                    escaped.wait()
                    raise RuntimeError("subprocess creation was not contained")
                if sys.platform.startswith("linux"):
                    try:
                        os.execve("/bin/true", ["true"], {{}})
                    except OSError:
                        pass
                    else:
                        raise RuntimeError("exec was not contained")

            async def exercise_required_thread_paths():
                loop = asyncio.get_running_loop()
                addresses = await loop.run_in_executor(
                    None, socket.getaddrinfo, "localhost", 443
                )
                if not addresses:
                    raise RuntimeError("default-executor DNS path failed")

            async def maybe_wait():
                if BEHAVIOR in {{"timeout", "cancel"}}:
                    await asyncio.sleep(60)

            async def maybe_close():
                if BEHAVIOR == "slow-close":
                    await asyncio.sleep(60)
            """
        )
    )
    (root / "httpx.py").write_text(
        textwrap.dedent(
            """
            class AsyncClient:
                def __init__(self, *, timeout, trust_env):
                    self.timeout = timeout
                    self._trust_env = trust_env
                    self.closed = False

                async def aclose(self):
                    self.closed = True
            """
        )
    )
    (root / "openai.py").write_text(
        textwrap.dedent(
            """
            import json
            from types import SimpleNamespace
            from fake_transport_common import (
                exercise_required_thread_paths,
                maybe_close,
                maybe_wait,
                verify_and_attempt_detached,
            )

            class _Completions:
                def __init__(self, owner):
                    self._owner = owner

                async def create(self, **kwargs):
                    del kwargs
                    await exercise_required_thread_paths()
                    verify_and_attempt_detached(
                        self._owner._secret, self._owner._http_client._trust_env
                    )
                    await maybe_wait()
                    content = json.dumps(
                        {"satisfied": True, "confidence": 0.99, "evidence": "contained"}
                    )
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                        usage=None,
                    )

            class AsyncOpenAI:
                def __init__(self, *, api_key, http_client, **kwargs):
                    del kwargs
                    self._secret = api_key
                    self._http_client = http_client
                    self.chat = SimpleNamespace(completions=_Completions(self))

                async def close(self):
                    await maybe_close()
                    await self._http_client.aclose()
            """
        )
    )
    google = root / "google"
    genai = google / "genai"
    genai.mkdir(parents=True)
    (google / "__init__.py").write_text("")
    (genai / "types.py").write_text(
        textwrap.dedent(
            """
            class HttpOptions:
                def __init__(self, *, timeout, client_args, async_client_args):
                    self.timeout = timeout
                    self.client_args = client_args
                    self.async_client_args = async_client_args

            class Part:
                @staticmethod
                def from_bytes(*, data, mime_type):
                    return (mime_type, data)

            class GenerateContentConfig:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
            """
        )
    )
    (genai / "__init__.py").write_text(
        textwrap.dedent(
            """
            import json
            from types import SimpleNamespace
            from . import types
            from fake_transport_common import (
                exercise_required_thread_paths,
                maybe_close,
                maybe_wait,
                verify_and_attempt_detached,
            )

            class _Models:
                def __init__(self, owner):
                    self._owner = owner

                async def generate_content(self, **kwargs):
                    del kwargs
                    await exercise_required_thread_paths()
                    trust_env = self._owner._http_options.async_client_args.get("trust_env")
                    verify_and_attempt_detached(self._owner._secret, trust_env)
                    await maybe_wait()
                    text = json.dumps(
                        {"satisfied": True, "confidence": 0.99, "evidence": "contained"}
                    )
                    return SimpleNamespace(text=text, usage_metadata=None)

            class Client:
                def __init__(self, *, api_key, http_options):
                    self._secret = api_key
                    self._http_options = http_options
                    self.aio = SimpleNamespace(models=_Models(self))

                async def aclose(self):
                    await maybe_close()
                    return None
            """
        )
    )


def _run_production_composition_harness(
    tmp_path: Path,
    *,
    provider: str,
    scenario: str,
    python_executable: str = sys.executable,
    large_context: bool = False,
    poison_launch_globals: bool = False,
) -> dict:
    root = tmp_path / f"{provider}-{scenario}-root"
    root.mkdir()
    effect_path = tmp_path / f"{provider}-{scenario}-late-effect"
    call_path = tmp_path / f"{provider}-{scenario}-provider-called"
    _write_production_transport_fakes(
        root,
        behavior=(
            scenario
            if scenario in {"timeout", "cancel"}
            else ("slow-close" if scenario == "teardown" else "success")
        ),
        effect_path=effect_path,
        call_path=call_path,
    )
    model = "gemini-3.5-flash" if provider == "gemini" else "qwen/qwen3-vl-32b-instruct"
    script = textwrap.dedent(
        f"""
        import asyncio
        import json
        import time
        from pathlib import Path
        from percept import (
            HostedByokAdapterFactory,
            HostedByokCancelled,
            HostedByokContractError,
            HostedByokDeadlineExceeded,
            HostedByokRequest,
            HostedByokSubprocessSupervisor,
            Percept,
        )
        from percept.config import Settings

        async def main():
            cancellation = asyncio.Event() if {scenario!r} == "cancel" else None
            timeout_s = (
                0.001
                if {large_context!r}
                else (2.0 if {scenario!r} in {{"timeout", "teardown"}} else 5.0)
            )
            request = HostedByokRequest(
                provider={provider!r},
                model={model!r},
                credential="sk-production-composition-secret",
                credential_lease_id="lease-production",
                principal_id="principal-production",
                timeout_s=timeout_s,
                cancellation=cancellation,
            )
            supervisor = HostedByokSubprocessSupervisor()
            vision = HostedByokAdapterFactory(
                process_supervisor=supervisor
            ).bind(request)
            if {poison_launch_globals!r}:
                import percept.hosted_byok as hosted_module

                def poison(*args, **kwargs):
                    del args, kwargs
                    raise AssertionError("live mutable launch global was used")

                hosted_module._select_containment_capability = poison
                hosted_module._clean_child_environment = poison
                hosted_module._clean_child_import_roots = poison
                hosted_module._launch_contained_worker = poison
                hosted_module._terminate_and_wait_contained_worker = poison
                hosted_module._make_private_supervisor_dispatch = poison
                hosted_module._make_worker_exchange = poison
                hosted_module._make_private_wire_codec = poison
                hosted_module._make_private_input_validator = poison
                hosted_module._validate_hosted_call_inputs = (
                    lambda *args, **kwargs: None
                )
                hosted_module._input_contains_credential = (
                    lambda *args, **kwargs: False
                )
                hosted_module._wire_value = poison
                hosted_module._wire_unvalue = poison
                hosted_module._reject_duplicate_json_pairs = poison
                hosted_module._wire_dumps = poison
                hosted_module._wire_loads = poison
                hosted_module._positive_duration = poison
                hosted_module._verdict_is_credential_free = poison
                hosted_module.subprocess.Popen = poison
                hosted_module.socket.socketpair = poison
                hosted_module.platform.machine = poison
                hosted_module.hmac.compare_digest = poison
                hosted_module.os.urandom = poison
                hosted_module.sys.executable = "/poisoned/interpreter"
                hosted_module.os.defpath = "/poisoned/path"
                hosted_module.os.environ["OPENAI_API_KEY"] = "poisoned-after-snapshot"
                hosted_module._CHILD_BOOTSTRAP = "raise SystemExit(97)"
            core = Percept.create(
                vision=vision,
                settings=Settings(),
                mode="offline",
                discover_plugins=False,
            )
            context = {{"blob": "x" * (16 * 1024 * 1024)}} if {large_context!r} else None
            if {scenario!r} == "child-credential-input":
                deadline = time.monotonic() + 5.0
                dispatch = supervisor._dispatch
                payload = dispatch.wire_dumps(
                    (
                        request,
                        request.credential,
                        b"jpeg",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        deadline,
                    )
                )
                child_outcome = await dispatch.exchange(payload, None, deadline)
                outcome = (
                    "child-contract"
                    if child_outcome[0] == "contract"
                    else "wrong-child-outcome"
                )
            else:
                task = asyncio.create_task(
                    core.vision.judge(
                        (
                            "sk-production-composition-secret"
                            if {scenario!r} == "credential-input"
                            else "someone enters"
                        ),
                        b"jpeg",
                        context=context,
                    )
                )
                if {scenario!r} == "cancel":
                    marker = Path({str(call_path)!r})
                    for _ in range(500):
                        if marker.exists():
                            break
                        await asyncio.sleep(0.01)
                    else:
                        raise RuntimeError("provider was not reached before cancellation")
                    cancellation.set()
                try:
                    verdict = await task
                except HostedByokDeadlineExceeded:
                    outcome = "deadline"
                except HostedByokCancelled:
                    outcome = "cancelled"
                except HostedByokContractError:
                    outcome = "contract"
                else:
                    outcome = "success" if verdict.satisfied else "wrong-verdict"
            await asyncio.sleep(0.4)
            print(json.dumps({{
                "outcome": outcome,
                "provider_called": Path({str(call_path)!r}).exists(),
                "late_effect": Path({str(effect_path)!r}).exists(),
            }}))

        asyncio.run(main())
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    for key in _HOSTILE_STARTUP_ENV_KEYS:
        environment[key] = (
            str(tmp_path / "ssl-keylog.txt")
            if key == "SSLKEYLOGFILE"
            else "http://inherited-attacker.invalid:4444"
        )
    completed = subprocess.run(
        [python_executable, "-c", script],
        cwd="/",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


class _ClockedReplay(list):
    """A replay lane that declares the shared clock a compiled fused Watch demands.

    A compiled fused Watch over two SEPARATE iterable lanes refuses to assume the lanes share a
    clock merely because one worker consumes them: it requires an explicit, matching calibration
    claim on the lane AND on every envelope it yields. This fixture supplies both.
    """

    clock_authority = "test.hosted_byok.replay"
    clock_confidence = 1.0

    @classmethod
    def video(cls, items) -> "_ClockedReplay":
        return cls(
            replace(
                legacy_video_envelope(
                    item, source_id="video", sequence=sequence, observed_time=float(item[0])
                ),
                clock_authority=cls.clock_authority,
                clock_confidence=cls.clock_confidence,
            )
            for sequence, item in enumerate(items)
        )

    @classmethod
    def audio(cls, items) -> "_ClockedReplay":
        return cls(
            replace(
                legacy_audio_envelope(
                    item, source_id="audio", sequence=sequence, observed_time=float(item[0])
                ),
                clock_authority=cls.clock_authority,
                clock_confidence=cls.clock_confidence,
            )
            for sequence, item in enumerate(items)
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
    assert received[0][:2] == (SECRET_A, "qwen/qwen3-vl-32b-instruct")
    assert received[0][2] == pytest.approx(4.0, abs=0.01)
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
    assert received[0][:2] == (SECRET_A, "gemini-3.5-flash")
    assert received[0][2] == pytest.approx(3.0, abs=0.01)
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


@pytest.mark.parametrize(
    ("field_name", "leaky_value"),
    [
        ("evidence", f"Bearer {SECRET_A}"),
        ("phase", SECRET_A),
        ("running_answer", f"summary:{SECRET_A}"),
        ("usage", {"nested": [{"credential": SECRET_A}]}),
        ("usage", {SECRET_A: 1}),
        ("events", [{"label": SECRET_A}]),
        ("subject", {"nested": (SECRET_A,)}),
        ("operation", SECRET_A),
        ("value", {"secret": bytearray(SECRET_A.encode())}),
        ("normalized_value", [SECRET_A]),
        ("target_revision_id", SECRET_A),
        ("asserted_by", SECRET_A),
    ],
)
def test_credential_in_any_verdict_carrier_fails_closed_without_echo(
    field_name, leaky_value
):
    verdict = Verdict(True, 0.99, "safe")
    setattr(verdict, field_name, leaky_value)
    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: _Backend(verdict)}
    )
    with pytest.raises(HostedByokInvocationError) as raised:
        asyncio.run(hosted.judge(_request(), "condition", b"frame"))
    assert str(raised.value) == "hosted_byok_provider_failure"
    assert SECRET_A not in str(raised.value)
    assert raised.value.__cause__ is None


def test_verdict_subclasses_and_opaque_mapping_carriers_fail_closed():
    class VerdictSubclass(Verdict):
        pass

    class OpaqueUsage(UserDict):
        pass

    for result in (
        VerdictSubclass(True, 0.99, "safe"),
        Verdict(True, 0.99, "safe", usage=OpaqueUsage({"safe": 1})),
    ):
        hosted = HostedByokAdapterFactory(
            backend_factories={"openrouter": lambda request, timeout, result=result: _Backend(result)}
        )
        with pytest.raises(HostedByokInvocationError, match="hosted_byok_provider_failure"):
            asyncio.run(hosted.judge(_request(), "condition", b"frame"))


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_real_adapter_transport_cannot_return_credential_in_evidence(provider):
    if provider == "gemini":
        client = _GeminiClient()

        async def generate_content(**kwargs):
            client.aio.models.calls.append(kwargs)
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "satisfied": True,
                        "confidence": 0.99,
                        "evidence": f"leaked {SECRET_A}",
                    }
                ),
                usage_metadata=None,
            )

        client.aio.models.generate_content = generate_content
        hosted = HostedByokAdapterFactory(
            gemini_client_factory=lambda request, timeout: HostedByokGeminiClient(
                client, _GeminiTypes
            )
        )
        request = _request(provider="gemini", model="gemini-3.5-flash")
    else:
        client = _OpenRouterClient()

        async def create(**kwargs):
            client.chat.completions.calls.append(kwargs)
            text = json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.99,
                    "evidence": f"leaked {SECRET_A}",
                }
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
            )

        client.chat.completions.create = create
        hosted = HostedByokAdapterFactory(
            openrouter_client_factory=lambda request, timeout: client
        )
        request = _request()

    with pytest.raises(HostedByokInvocationError) as raised:
        asyncio.run(hosted.judge(request, "condition", b"frame"))
    assert str(raised.value) == "hosted_byok_provider_failure"
    assert client.closed


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


def test_event_loop_blocking_provider_result_is_rejected_after_monotonic_deadline():
    effects: list[str] = []

    class BlockingBackend:
        async def judge(self, condition, frame, **kwargs):
            time.sleep(0.03)
            effects.append("provider-returned")
            return Verdict(True, 0.99, "too late")

    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: BlockingBackend()}
    )
    with pytest.raises(HostedByokDeadlineExceeded, match="completed after"):
        asyncio.run(hosted.judge(_request(timeout_s=0.01), "condition", b"frame"))
    assert effects == ["provider-returned"]


def test_absolute_deadline_fails_before_constructing_backend():
    constructed = False

    def factory(request, timeout_s):
        nonlocal constructed
        constructed = True
        return _Backend()

    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": factory}, time_fn=lambda: 100.0
    )
    with pytest.raises(HostedByokDeadlineExceeded, match="public judge entry"):
        asyncio.run(hosted.judge(_request(deadline_unix_s=99.0), "condition", b"frame"))
    assert not constructed


def test_provider_construction_time_counts_against_monotonic_deadline():
    monotonic_values = iter((10.0, 10.0, 10.0, 12.1))

    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: _Backend()},
        monotonic_fn=lambda: next(monotonic_values),
    )
    with pytest.raises(HostedByokDeadlineExceeded, match="during provider construction"):
        asyncio.run(hosted.judge(_request(timeout_s=2.0), "condition", b"frame"))


def test_invocation_cancellation_signal_cancels_provider_work():
    cancellation = asyncio.Event()
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    class SlowBackend:
        async def judge(self, condition, frame, **kwargs):
            provider_started.set()
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
        await provider_started.wait()
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


def test_cancellation_holds_ownership_until_cancel_suppressing_provider_finishes():
    cancellation = asyncio.Event()
    release = asyncio.Event()
    started = asyncio.Event()
    effects: list[str] = []

    class SuppressingBackend:
        async def judge(self, condition, frame, **kwargs):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await release.wait()
                effects.append("provider-finished")
                return Verdict(False, 0.0, "late")

    async def run():
        hosted = HostedByokAdapterFactory(
            backend_factories={"openrouter": lambda request, timeout: SuppressingBackend()},
        )
        task = asyncio.create_task(
            hosted.judge(_request(cancellation=cancellation), "condition", b"frame")
        )
        await started.wait()
        cancellation.set()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert effects == []
        release.set()
        with pytest.raises(HostedByokCancelled):
            await task
        assert effects == ["provider-finished"]
        await asyncio.sleep(0)
        assert effects == ["provider-finished"]
        assert {pending for pending in asyncio.all_tasks() if pending is not asyncio.current_task()} == set()

    asyncio.run(run())


def test_timeout_waits_for_cancel_suppressing_provider_and_consumes_late_error():
    effects: list[str] = []

    class SuppressingBackend:
        async def judge(self, condition, frame, **kwargs):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await asyncio.sleep(0.03)
                effects.append("provider-finished")
                raise RuntimeError(f"late provider detail {SECRET_A}")

    async def run():
        loop_errors = []
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        try:
            hosted = HostedByokAdapterFactory(
                backend_factories={
                    "openrouter": lambda request, timeout: SuppressingBackend()
                },
            )
            started = loop.time()
            with pytest.raises(HostedByokDeadlineExceeded):
                await hosted.judge(_request(timeout_s=0.01), "condition", b"frame")
            elapsed = loop.time() - started
            assert elapsed >= 0.03
            assert effects == ["provider-finished"]
            await asyncio.sleep(0.02)
            assert effects == ["provider-finished"]
            assert loop_errors == []
            assert {
                pending
                for pending in asyncio.all_tasks()
                if pending is not asyncio.current_task()
            } == set()
        finally:
            loop.set_exception_handler(original_handler)

    asyncio.run(run())


def test_external_task_cancel_holds_ownership_until_provider_finishes():
    release = asyncio.Event()
    started = asyncio.Event()
    effects: list[str] = []

    class SuppressingBackend:
        async def judge(self, condition, frame, **kwargs):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await release.wait()
                effects.append("provider-finished")
                return Verdict(False, 0.0, "ignored")

    async def run():
        hosted = HostedByokAdapterFactory(
            backend_factories={"openrouter": lambda request, timeout: SuppressingBackend()}
        )
        task = asyncio.create_task(hosted.judge(_request(), "condition", b"frame"))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert effects == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert effects == ["provider-finished"]
        assert {
            pending
            for pending in asyncio.all_tasks()
            if pending is not asyncio.current_task()
        } == set()

    asyncio.run(run())


def test_async_client_close_is_shielded_and_completed_before_cancel_returns():
    release = asyncio.Event()
    close_started = asyncio.Event()
    close_cancelled = False
    close_effects: list[str] = []

    class HangingCloseClient(_OpenRouterClient):
        async def close(self):
            nonlocal close_cancelled
            close_started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                close_cancelled = True
                await release.wait()
            self.closed = True
            close_effects.append("closed")

    async def run():
        hosted = HostedByokAdapterFactory(
            openrouter_client_factory=lambda request, timeout: HangingCloseClient(),
        )
        task = asyncio.create_task(hosted.judge(_request(), "condition", b"frame"))
        await close_started.wait()
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert not close_cancelled
        assert close_effects == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert close_effects == ["closed"]
        await asyncio.sleep(0)
        assert close_effects == ["closed"]
        assert {
            pending
            for pending in asyncio.all_tasks()
            if pending is not asyncio.current_task()
        } == set()

    asyncio.run(run())


def test_sync_client_close_runs_inline_and_finishes_before_return():
    effects: list[str] = []

    class SyncCloseClient(_OpenRouterClient):
        def close(self):
            self.closed = True
            effects.append("closed")

    verdict = asyncio.run(
        HostedByokAdapterFactory(
            openrouter_client_factory=lambda request, timeout: SyncCloseClient()
        ).judge(_request(), "condition", b"frame")
    )
    assert verdict.satisfied
    assert effects == ["closed"]


def test_raising_client_close_contains_internal_worker_before_fail_closed_return():
    release = asyncio.Event()
    close_started = asyncio.Event()
    effects: list[str] = []

    class RaisingCloseClient(_OpenRouterClient):
        async def close(self):
            close_started.set()

            async def internal_worker():
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    await release.wait()
                    effects.append("internal-worker-finished")

            asyncio.create_task(internal_worker())
            raise RuntimeError("close failure after internal worker launch")

    async def run_case():
        task = asyncio.create_task(
            HostedByokAdapterFactory(
                openrouter_client_factory=lambda request, timeout: RaisingCloseClient()
            ).judge(_request(), "condition", b"frame")
        )
        await close_started.wait()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert effects == []
        release.set()
        with pytest.raises(HostedByokInvocationError, match="hosted_byok_provider_failure"):
            await task
        assert effects == ["internal-worker-finished"]
        await asyncio.sleep(0)
        assert effects == ["internal-worker-finished"]
        assert {
            pending for pending in asyncio.all_tasks() if pending is not asyncio.current_task()
        } == set()

    asyncio.run(run_case())


@pytest.mark.parametrize("close_value", [object(), property(lambda self: (_ for _ in ()).throw(RuntimeError()))])
def test_malformed_backend_close_fails_closed(close_value):
    class BadCloseBackend(_Backend):
        close = close_value

    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: BadCloseBackend()}
    )
    with pytest.raises(HostedByokInvocationError, match="hosted_byok_provider_failure"):
        asyncio.run(hosted.judge(_request(), "condition", b"frame"))


def test_cancelled_close_gets_fail_closed_only_after_scope_teardown():
    class CancelledCloseBackend(_Backend):
        def close(self):
            raise asyncio.CancelledError()

    hosted = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: CancelledCloseBackend()}
    )
    with pytest.raises(HostedByokInvocationError, match="hosted_byok_provider_failure"):
        asyncio.run(hosted.judge(_request(), "condition", b"frame"))


def test_overlapping_invocations_keep_task_scopes_isolated_and_restore_loop_factory():
    first_done = asyncio.Event()
    allow_second_worker = asyncio.Event()
    second_worker_started = asyncio.Event()
    second_worker_finished = asyncio.Event()
    release_second = asyncio.Event()
    external_started = asyncio.Event()
    release_external = asyncio.Event()

    class Backend:
        def __init__(self, principal):
            self.principal = principal

        async def judge(self, condition, frame, **kwargs):
            if self.principal == "first":
                await asyncio.sleep(0)
                first_done.set()
                return Verdict(True, 0.9, "first")
            await first_done.wait()
            await allow_second_worker.wait()

            async def worker():
                second_worker_started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    await release_second.wait()
                    second_worker_finished.set()

            asyncio.create_task(worker())
            await second_worker_started.wait()
            await asyncio.sleep(60)

    async def run_case():
        loop = asyncio.get_running_loop()
        assert loop.get_task_factory() is None
        hosted = HostedByokAdapterFactory(
            backend_factories={"openrouter": lambda request, timeout: Backend(request.principal_id)}
        )
        first = asyncio.create_task(hosted.judge(_request(principal="first"), "first", b"frame"))
        second = asyncio.create_task(hosted.judge(_request(principal="second"), "second", b"frame"))
        assert (await first).evidence == "first"
        # The first scope has restored nothing yet: the overlapping second invocation still owns
        # the shared dispatcher and may create its worker after the first result is accepted.
        assert loop.get_task_factory() is not None
        allow_second_worker.set()
        await second_worker_started.wait()

        async def unrelated_host_task():
            external_started.set()
            await release_external.wait()
            return "host-owned"

        external_during_scope = asyncio.create_task(unrelated_host_task())
        await external_started.wait()
        second.cancel()
        await asyncio.sleep(0.02)
        assert not second.done()
        assert not second_worker_finished.is_set()
        release_second.set()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert second_worker_finished.is_set()
        assert not external_during_scope.done()
        release_external.set()
        assert await external_during_scope == "host-owned"
        assert loop.get_task_factory() is None
        marker = []
        external = asyncio.create_task(asyncio.sleep(0, result=marker))
        external.add_done_callback(lambda _task: marker.append("ok"))
        await external
        await asyncio.sleep(0)
        assert marker == ["ok"]

    asyncio.run(run_case())


def test_existing_loop_task_factory_is_not_clobbered_and_fails_before_construction():
    async def run_case():
        loop = asyncio.get_running_loop()
        constructed = False

        def existing_factory(factory_loop, coroutine, context=None):
            if context is None:
                return asyncio.tasks.Task(coroutine, loop=factory_loop)
            return asyncio.tasks.Task(coroutine, loop=factory_loop, context=context)

        loop.set_task_factory(existing_factory)
        try:
            hosted = HostedByokAdapterFactory(
                backend_factories={
                    "openrouter": lambda request, timeout: (_ for _ in ()).throw(
                        AssertionError("must not construct")
                    )
                }
            )
            with pytest.raises(HostedByokContractError, match="task factory"):
                await hosted.judge(_request(), "condition", b"frame")
            assert loop.get_task_factory() is existing_factory
            assert not constructed
        finally:
            loop.set_task_factory(None)

    asyncio.run(run_case())


def test_cloud_foreign_transport_supervisor_gate_is_machine_checkable_before_construction():
    request = _request()
    assert HostedByokAdapterFactory().transport_safety(request) == HostedByokTransportSafety(
        provider="openrouter",
        transport_class="production_process_isolated",
        asyncio_task_contained=False,
        requires_process_supervisor=True,
    )
    with pytest.raises(HostedByokContractError, match="subprocess supervisor"):
        asyncio.run(HostedByokAdapterFactory().judge(request, "condition", b"frame"))

    injected = HostedByokAdapterFactory(
        backend_factories={"openrouter": lambda request, timeout: _Backend()}
    )
    safety = injected.transport_safety(request)
    assert safety == HostedByokTransportSafety(
        provider="openrouter",
        transport_class="test_only_in_process",
        asyncio_task_contained=True,
        requires_process_supervisor=False,
    )
    assert asyncio.run(injected.judge(request, "condition", b"frame")).satisfied
    with pytest.raises(AttributeError, match="immutable"):
        injected.transport_safety = lambda request: HostedByokTransportSafety(
            request.provider, "forged", False, False
        )


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("success", "success"),
        ("timeout", "deadline"),
        ("cancel", "cancelled"),
        ("teardown", "deadline"),
    ],
)
def test_public_production_composition_contains_detached_descendants_and_environment(
    tmp_path, provider, scenario, expected
):
    result = _run_production_composition_harness(
        tmp_path,
        provider=provider,
        scenario=scenario,
    )
    assert result == {
        "outcome": expected,
        "provider_called": True,
        "late_effect": False,
    }


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_public_composition_uses_sealed_launch_snapshot_after_global_poison(
    tmp_path, provider
):
    result = _run_production_composition_harness(
        tmp_path,
        provider=provider,
        scenario="success",
        poison_launch_globals=True,
    )
    assert result == {
        "outcome": "success",
        "provider_called": True,
        "late_effect": False,
    }


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_frozen_validation_graph_rejects_credential_before_provider_after_global_poison(
    tmp_path, provider
):
    result = _run_production_composition_harness(
        tmp_path,
        provider=provider,
        scenario="credential-input",
        poison_launch_globals=True,
    )
    assert result == {
        "outcome": "contract",
        "provider_called": False,
        "late_effect": False,
    }


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_clean_child_revalidates_decoded_credential_input_before_provider_construction(
    tmp_path, provider
):
    result = _run_production_composition_harness(
        tmp_path,
        provider=provider,
        scenario="child-credential-input",
        poison_launch_globals=True,
    )
    assert result == {
        "outcome": "child-contract",
        "provider_called": False,
        "late_effect": False,
    }


@pytest.mark.skipif(
    not (
        sys.platform.startswith("linux")
        and hasattr(os, "uname")
        and os.uname().machine.lower() in {"x86_64", "amd64"}
    ),
    reason="requires the genuine Linux/AMD64 seccomp boundary",
)
@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
@pytest.mark.parametrize(
    "scenario,expected",
    [("success", "success"), ("timeout", "deadline"), ("cancel", "cancelled")],
)
def test_linux_amd64_public_composition_allows_thread_dns_and_contains_late_effects(
    tmp_path, provider, scenario, expected
):
    result = _run_production_composition_harness(
        tmp_path,
        provider=provider,
        scenario=scenario,
    )
    assert result == {
        "outcome": expected,
        "provider_called": True,
        "late_effect": False,
    }


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_entry_deadline_rejects_large_context_before_production_provider(tmp_path, provider):
    result = _run_production_composition_harness(
        tmp_path,
        provider=provider,
        scenario="large-context",
        large_context=True,
    )
    assert result == {
        "outcome": "deadline",
        "provider_called": False,
        "late_effect": False,
    }


def test_non_executable_wire_rejects_reduce_bombs_both_directions():
    from percept.hosted_byok import _wire_dumps, _wire_loads

    reduce_calls = 0

    class ReduceBomb:
        def __reduce__(self):
            nonlocal reduce_calls
            reduce_calls += 1
            return (str, ("executed",))

    with pytest.raises(HostedByokContractError, match="unsupported carrier"):
        _wire_dumps({"bomb": ReduceBomb()})
    assert reduce_calls == 0
    poisoned = b'{"t":"dto:Verdict","v":{"__reduce__":{"t":"str","v":"boom"}}}'
    with pytest.raises(HostedByokInvocationError):
        _wire_loads(poisoned)
    assert reduce_calls == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", f"openrouter-{SECRET_A}"),
        ("model", f"vendor/{SECRET_A}"),
        ("credential_lease_id", f"lease-{SECRET_A}"),
        ("principal_id", f"principal-{SECRET_A}"),
        ("capability", f"vision.judge-{SECRET_A}"),
        ("contract_version", f"1.0.0-{SECRET_A}"),
    ],
)
def test_request_visible_metadata_cannot_alias_the_credential(field, value):
    values = {
        "provider": "openrouter",
        "model": "qwen/qwen3-vl-32b-instruct",
        "credential": SECRET_A,
        "credential_lease_id": "lease-a",
        "principal_id": "principal-a",
    }
    values[field] = value
    with pytest.raises(HostedByokContractError, match="must not contain the credential"):
        HostedByokRequest(**values)


def test_wire_round_trips_exact_request_media_audio_roi_and_verdict_dtos():
    from percept.hosted_byok import _wire_dumps, _wire_loads

    request = _request(
        provider="gemini",
        model="gemini-3.5-flash",
        capability="vision.judge.fused",
    )
    values = (
        request,
        "condition",
        ("image/jpeg", b"jpeg"),
        percept.Box(0.1, 0.2, 0.3, 0.4),
        AudioFrame(pcm=b"\x01\x00", sample_rate=16_000),
        Verdict(True, 0.9, "safe", usage={"input_tokens": 3}),
    )
    decoded = _wire_loads(_wire_dumps(values))
    assert type(decoded) is tuple
    assert type(decoded[0]) is HostedByokRequest
    assert decoded[0].credential == SECRET_A
    assert decoded[2] == ("image/jpeg", b"jpeg")
    assert type(decoded[3]) is percept.Box
    assert type(decoded[4]) is AudioFrame
    assert type(decoded[5]) is Verdict


@pytest.mark.parametrize(
    "payload",
    [
        b'{"t":"str","t":"int","v":"x"}',
        b'{"t":"str","v":"unterminated"',
        b'{"t":"float","v":NaN}',
        b'{"v":"x","t":"str"}',
        b'{"t":"dict","v":[[{"t":"str","v":"x"},{"t":"int","v":1}],'
        b'[{"t":"str","v":"x"},{"t":"int","v":2}]]}',
    ],
)
def test_wire_rejects_duplicate_truncated_nonfinite_and_schema_poison(payload):
    from percept.hosted_byok import _wire_loads

    with pytest.raises(HostedByokInvocationError):
        _wire_loads(payload)


def test_wire_rejects_oversized_payload_before_json_decode():
    from percept.hosted_byok import _wire_loads

    with pytest.raises(HostedByokInvocationError):
        _wire_loads(b"x" * (32 * 1024 * 1024 + 1))


class _FramedPeerProcess:
    def __init__(self):
        self.returncode = None
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self):
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _PostSpawnSocket:
    def __init__(
        self,
        *,
        descriptor=73,
        setblocking_failure=False,
        close_failure=False,
    ):
        self.descriptor = descriptor
        self.setblocking_failure = setblocking_failure
        self.close_failure = close_failure
        self.setblocking_calls = 0
        self.close_calls = 0

    def fileno(self):
        return self.descriptor

    def setblocking(self, value):
        assert value is False
        self.setblocking_calls += 1
        if self.setblocking_failure:
            raise OSError("injected parent setblocking failure")

    def close(self):
        self.close_calls += 1
        if self.close_failure:
            raise OSError("injected socket close failure")


@pytest.mark.parametrize(
    ("setblocking_failure", "parent_close_failure", "child_close_failure"),
    [
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
        (False, False, True),
        (False, True, True),
    ],
    ids=[
        "setblocking",
        "setblocking-parent-close",
        "setblocking-child-close",
        "setblocking-both-closes",
        "child-close",
        "child-and-parent-close",
    ],
)
def test_every_post_spawn_setup_failure_kills_and_joins_exact_process(
    monkeypatch,
    setblocking_failure,
    parent_close_failure,
    child_close_failure,
):
    import percept.hosted_byok as hosted_module

    parent_socket = _PostSpawnSocket(
        setblocking_failure=setblocking_failure,
        close_failure=parent_close_failure,
    )
    child_socket = _PostSpawnSocket(close_failure=child_close_failure)
    processes = []

    def fake_popen(*args, **kwargs):
        del args, kwargs
        process = _FramedPeerProcess()
        processes.append(process)
        return process

    # Restore the shared stdlib module attributes before asyncio creates its loop; the
    # supervisor must retain only the exact fake authorities captured at construction.
    with monkeypatch.context() as patch:
        patch.setattr(hosted_module.subprocess, "Popen", fake_popen)
        patch.setattr(
            hosted_module.socket,
            "socketpair",
            lambda: (parent_socket, child_socket),
        )
        supervisor = HostedByokSubprocessSupervisor()

    with pytest.raises(HostedByokInvocationError):
        asyncio.run(
            supervisor._execute_impl(
                _request(),
                "condition",
                b"frame",
                deadline_monotonic=time.monotonic() + 2.0,
                roi=None,
                exemplar=None,
                corrections=None,
                frames=None,
                audio=None,
                context=None,
            )
        )

    assert len(processes) == 1
    process = processes[0]
    assert process.killed
    assert process.waited
    assert process.poll() is not None
    assert parent_socket.close_calls >= 1
    assert child_socket.close_calls >= 1


def _framed_exchange_for_test(
    parent_socket,
    process,
    *,
    verify_ready=None,
    wire_loads=None,
):
    from percept.hosted_byok import (
        _MAX_PROCESS_IPC_BYTES,
        _make_worker_exchange,
    )

    parent_socket.setblocking(False)

    def launch():
        return process, parent_socket, b"c" * 32

    def terminate(target):
        if target.poll() is None:
            target.kill()
        target.wait()

    if verify_ready is None:
        def default_verify_ready(ready, challenge):
            return ready == b"ready" and challenge == b"c" * 32

        verify_ready = default_verify_ready
    if wire_loads is None:
        def default_wire_loads(payload):
            return payload

        wire_loads = default_wire_loads

    return _make_worker_exchange(
        launch=launch,
        terminate=terminate,
        verify_ready=verify_ready,
        wire_loads=wire_loads,
        sleep=asyncio.sleep,
        monotonic=time.monotonic,
        poll_interval_s=0.001,
        max_payload_bytes=_MAX_PROCESS_IPC_BYTES,
        invocation_error=HostedByokInvocationError,
        deadline_error=HostedByokDeadlineExceeded,
        cancellation_error=HostedByokCancelled,
        failure_code="hosted_byok_provider_failure",
    )


class _FaultingFramedSocket:
    def __init__(self, peer, *, send_failure=False, close_failure=False):
        self.peer = peer
        self.send_failure = send_failure
        self.close_failure = close_failure

    def setblocking(self, value):
        return self.peer.setblocking(value)

    def recv(self, size):
        return self.peer.recv(size)

    def send(self, payload):
        if self.send_failure:
            raise OSError("injected framed send failure")
        return self.peer.send(payload)

    def close(self):
        try:
            self.peer.close()
        finally:
            if self.close_failure:
                raise OSError("injected framed close failure")


@pytest.mark.parametrize(
    ("failure_stage", "expected_error"),
    [
        ("ready-decode", RuntimeError),
        ("ready-verify", RuntimeError),
        ("request-send", HostedByokInvocationError),
        ("peer-close", HostedByokInvocationError),
    ],
)
def test_post_spawn_readiness_send_and_cleanup_exceptions_always_join_worker(
    failure_stage,
    expected_error,
):
    async def exercise():
        parent_socket, writer = socket.socketpair()
        process = _FramedPeerProcess()
        peer = _FaultingFramedSocket(
            parent_socket,
            send_failure=failure_stage == "request-send",
            close_failure=failure_stage == "peer-close",
        )

        def decode(payload):
            if failure_stage == "ready-decode":
                raise RuntimeError("injected ready decode failure")
            return payload

        def verify(ready, challenge):
            if failure_stage == "ready-verify":
                raise RuntimeError("injected ready verification failure")
            if failure_stage == "peer-close":
                return False
            return ready == b"ready" and challenge == b"c" * 32

        exchange = _framed_exchange_for_test(
            peer,
            process,
            wire_loads=decode,
            verify_ready=verify,
        )
        writer.sendall(struct.pack("!I", 5) + b"ready")
        try:
            with pytest.raises(expected_error):
                await exchange(b"request", None, time.monotonic() + 1.0)
        finally:
            writer.close()
        assert process.killed
        assert process.waited
        assert process.poll() is not None

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("wire_bytes", "close_writer", "expected_error", "needs_heartbeat"),
    [
        (b"", False, HostedByokDeadlineExceeded, True),
        (b"\x00\x00", False, HostedByokDeadlineExceeded, True),
        (struct.pack("!I", 32) + b"partial", False, HostedByokDeadlineExceeded, True),
        (
            struct.pack("!I", 32 * 1024 * 1024 + 1),
            False,
            HostedByokInvocationError,
            False,
        ),
        (b"\x00\x00", True, HostedByokInvocationError, False),
        (
            struct.pack("!I", 32) + b"partial",
            True,
            HostedByokInvocationError,
            False,
        ),
    ],
    ids=[
        "stall",
        "partial-header",
        "partial-body",
        "oversize-header",
        "eof-header",
        "eof-body",
    ],
)
def test_nonblocking_framed_ready_rejects_stalls_oversize_and_eof(
    wire_bytes,
    close_writer,
    expected_error,
    needs_heartbeat,
):
    async def exercise():
        parent_socket, writer = socket.socketpair()
        process = _FramedPeerProcess()
        exchange = _framed_exchange_for_test(parent_socket, process)
        ticks = 0
        stopped = False

        async def heartbeat():
            nonlocal ticks
            while not stopped:
                ticks += 1
                await asyncio.sleep(0.001)

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            if wire_bytes:
                writer.sendall(wire_bytes)
            if close_writer:
                writer.close()
            with pytest.raises(expected_error):
                await exchange(b"request", None, time.monotonic() + 0.04)
        finally:
            stopped = True
            await heartbeat_task
            writer.close()
        assert process.killed
        assert process.waited
        if needs_heartbeat:
            assert ticks >= 3

    asyncio.run(exercise())


def test_partial_framed_body_honors_cancellation_and_joins_worker():
    async def exercise():
        parent_socket, writer = socket.socketpair()
        process = _FramedPeerProcess()
        exchange = _framed_exchange_for_test(parent_socket, process)
        cancellation = asyncio.Event()
        writer.sendall(struct.pack("!I", 64) + b"partial")

        async def cancel_soon():
            await asyncio.sleep(0.02)
            cancellation.set()

        cancellation_task = asyncio.create_task(cancel_soon())
        try:
            with pytest.raises(HostedByokCancelled):
                await exchange(
                    b"request",
                    cancellation,
                    time.monotonic() + 1.0,
                )
        finally:
            await cancellation_task
            writer.close()
        assert process.killed
        assert process.waited

    asyncio.run(exercise())


def test_partial_framed_body_honors_task_cancellation_and_joins_worker():
    async def exercise():
        parent_socket, writer = socket.socketpair()
        process = _FramedPeerProcess()
        exchange = _framed_exchange_for_test(parent_socket, process)
        writer.sendall(struct.pack("!I", 64) + b"partial")
        task = asyncio.create_task(
            exchange(b"request", None, time.monotonic() + 1.0)
        )
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            writer.close()
        assert process.killed
        assert process.waited

    asyncio.run(exercise())


def test_partial_outcome_body_cannot_block_deadline_or_worker_join():
    async def exercise():
        parent_socket, writer = socket.socketpair()
        writer.setblocking(False)
        process = _FramedPeerProcess()
        exchange = _framed_exchange_for_test(parent_socket, process)
        request_seen = asyncio.Event()

        async def receive_exact(size):
            result = bytearray()
            loop = asyncio.get_running_loop()
            while len(result) < size:
                chunk = await loop.sock_recv(writer, size - len(result))
                if not chunk:
                    raise AssertionError("parent closed before sending the request")
                result.extend(chunk)
            return bytes(result)

        async def adversary():
            loop = asyncio.get_running_loop()
            await loop.sock_sendall(writer, struct.pack("!I", 5) + b"ready")
            request_size = struct.unpack("!I", await receive_exact(4))[0]
            assert await receive_exact(request_size) == b"request"
            request_seen.set()
            await loop.sock_sendall(writer, struct.pack("!I", 64) + b"partial")

        adversary_task = asyncio.create_task(adversary())
        try:
            with pytest.raises(HostedByokDeadlineExceeded):
                await exchange(b"request", None, time.monotonic() + 0.05)
            assert request_seen.is_set()
        finally:
            await adversary_task
            writer.close()
        assert process.killed
        assert process.waited

    asyncio.run(exercise())


def test_nonreading_worker_cannot_block_framed_request_send_or_deadline():
    async def exercise():
        parent_socket, writer = socket.socketpair()
        process = _FramedPeerProcess()
        exchange = _framed_exchange_for_test(parent_socket, process)
        writer.sendall(struct.pack("!I", 5) + b"ready")
        ticks = 0
        stopped = False

        async def heartbeat():
            nonlocal ticks
            while not stopped:
                ticks += 1
                await asyncio.sleep(0.001)

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            with pytest.raises(HostedByokDeadlineExceeded):
                await exchange(
                    b"x" * (4 * 1024 * 1024),
                    None,
                    time.monotonic() + 0.05,
                )
        finally:
            stopped = True
            await heartbeat_task
            writer.close()
        assert ticks >= 3
        assert process.killed
        assert process.waited

    asyncio.run(exercise())


def test_credential_tainted_or_malicious_inputs_spend_zero_constructions():
    constructions = 0

    class Malicious:
        def __reduce__(self):
            raise AssertionError("reduce must never run")

    def factory(request, timeout):
        nonlocal constructions
        constructions += 1
        return _Backend()

    hosted = HostedByokAdapterFactory(backend_factories={"openrouter": factory})
    attempts = (
        {"condition": SECRET_A, "frame": b"frame"},
        {"condition": "condition", "frame": SECRET_A.encode()},
        {"condition": "condition", "frame": b"frame", "context": {"safe": SECRET_A}},
        {"condition": "condition", "frame": b"frame", "corrections": {"safe": SECRET_A}},
        {"condition": "condition", "frame": b"frame", "roi": percept.Box(SECRET_A, 0, 1, 1)},
        {"condition": "condition", "frame": b"frame", "context": {"bomb": Malicious()}},
    )
    for attempt in attempts:
        call = dict(attempt)
        condition = call.pop("condition")
        frame = call.pop("frame")
        with pytest.raises(HostedByokContractError):
            asyncio.run(hosted.judge(_request(), condition, frame, **call))
    assert constructions == 0


def test_request_rejects_hostile_primitive_subclasses_before_serialization():
    reduce_calls = 0

    class HostileStr(str):
        def __reduce__(self):
            nonlocal reduce_calls
            reduce_calls += 1
            return (str, ("forged",))

    with pytest.raises(HostedByokContractError, match="provider"):
        _request(provider=HostileStr("openrouter"))
    assert reduce_calls == 0


def test_private_transport_classifier_ignores_public_method_poison(monkeypatch):
    factory = HostedByokAdapterFactory()
    monkeypatch.setattr(
        HostedByokAdapterFactory,
        "transport_safety",
        lambda self, request: HostedByokTransportSafety(
            request.provider, "forged", False, False
        ),
    )
    with pytest.raises(HostedByokContractError, match="subprocess supervisor"):
        asyncio.run(factory.judge(_request(), "condition", b"frame"))
    with pytest.raises(TypeError, match="sealed"):
        type("ForgedFactory", (HostedByokAdapterFactory,), {})


def test_exact_subprocess_supervisor_and_factory_configuration_are_immutable():
    supervisor = HostedByokSubprocessSupervisor()
    factory = HostedByokAdapterFactory(process_supervisor=supervisor)
    assert factory.transport_safety(_request()).production_ready
    assert factory.transport_safety(_request()).process_isolation_version == "2.3.0"
    with pytest.raises(AttributeError, match="immutable"):
        supervisor._poll_interval_s = 99
    with pytest.raises(FrozenInstanceError):
        supervisor._dispatch.exchange = lambda *args: None
    with pytest.raises(AttributeError, match="immutable"):
        factory._process_supervisor = None
    with pytest.raises(TypeError, match="sealed"):
        type("ForgedSupervisor", (HostedByokSubprocessSupervisor,), {})


def test_production_supervisor_rejects_virtual_clock_seams():
    supervisor = HostedByokSubprocessSupervisor()
    with pytest.raises(HostedByokContractError, match="exact system"):
        HostedByokAdapterFactory(process_supervisor=supervisor, monotonic_fn=lambda: 1.0)
    with pytest.raises(HostedByokContractError, match="exact system"):
        HostedByokAdapterFactory(process_supervisor=supervisor, time_fn=lambda: 1.0)


def test_platform_containment_capability_selection_is_exact_and_fails_closed():
    from percept.hosted_byok import _select_containment_capability

    assert _select_containment_capability(
        system="linux", machine="x86_64"
    ) == ("linux_seccomp_amd64_no_descendants", [])
    assert _select_containment_capability(
        system="linux", machine="AMD64"
    ) == ("linux_seccomp_amd64_no_descendants", [])
    for system, machine in (
        ("linux", "aarch64"),
        ("linux", "riscv64"),
        ("linux2", "x86_64"),
        ("win32", "AMD64"),
        ("freebsd", "x86_64"),
    ):
        with pytest.raises(HostedByokContractError, match="exact platform"):
            _select_containment_capability(system=system, machine=machine)


def test_child_boundary_attestation_does_not_trust_parent_argv():
    from percept.hosted_byok import _establish_no_descendant_boundary

    forged_kind = (
        "linux_seccomp_amd64_no_descendants"
        if sys.platform == "darwin"
        else "darwin_sandbox_no_descendants"
    )
    with pytest.raises(HostedByokInvocationError):
        _establish_no_descendant_boundary(forged_kind)


def test_factory_uses_frozen_supervisor_dispatch_after_class_poison(monkeypatch):
    supervisor = HostedByokSubprocessSupervisor()
    factory = HostedByokAdapterFactory(process_supervisor=supervisor)

    async def poison(*args, **kwargs):
        raise AssertionError("mutable public supervisor dispatch was used")

    monkeypatch.setattr(HostedByokSubprocessSupervisor, "execute", poison)
    with pytest.raises(HostedByokDeadlineExceeded):
        asyncio.run(
            factory.judge(
                _request(timeout_s=0.001),
                "condition",
                b"frame",
            )
        )


def test_supervisor_validator_freezes_transitive_helpers_and_type_authorities(monkeypatch):
    import percept.hosted_byok as hosted_module

    supervisor = HostedByokSubprocessSupervisor()
    validate_inputs = supervisor._dispatch.validate_inputs
    roi = percept.Box(0, 0, 1, 1)

    def poison(*args, **kwargs):
        del args, kwargs
        raise AssertionError("live validation dependency was used")

    monkeypatch.setattr(hosted_module, "_validate_hosted_call_inputs", poison)
    monkeypatch.setattr(hosted_module, "_input_contains_credential", poison)
    monkeypatch.setattr(hosted_module, "_wire_value", poison)
    monkeypatch.setattr(hosted_module, "dataclass_fields", poison)
    monkeypatch.setattr(hosted_module, "math", SimpleNamespace(isfinite=poison))
    monkeypatch.setattr(hosted_module, "HostedByokRequest", object)
    monkeypatch.setattr(hosted_module, "Box", object)
    monkeypatch.setattr(hosted_module, "AudioFrame", object)

    with pytest.raises(HostedByokContractError, match="must not contain the credential"):
        validate_inputs(
            _request(),
            condition="condition",
            frame=b"frame",
            roi=roi,
            exemplar=None,
            corrections=None,
            frames=None,
            audio=None,
            context={"nested": [SECRET_A]},
        )


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_acquired_client_closes_when_backend_constructor_fails(monkeypatch, provider):
    clients = []
    if provider == "gemini":
        import percept.backends.gemini as backend_module

        request = _request(provider="gemini", model="gemini-3.5-flash")

        def client_factory(request, timeout):
            client = _GeminiClient()
            clients.append(client)
            return HostedByokGeminiClient(client, _GeminiTypes)

        hosted = HostedByokAdapterFactory(gemini_client_factory=client_factory)
    else:
        import percept.backends.openrouter as backend_module

        request = _request()

        def client_factory(request, timeout):
            client = _OpenRouterClient()
            clients.append(client)
            return client

        hosted = HostedByokAdapterFactory(openrouter_client_factory=client_factory)

    def fail_constructor(self, *args, **kwargs):
        raise RuntimeError(f"constructor leaked {SECRET_A}")

    adapter_type = (
        backend_module.GeminiVision
        if provider == "gemini"
        else backend_module.OpenRouterVision
    )
    monkeypatch.setattr(adapter_type, "__init__", fail_constructor)
    with pytest.raises(HostedByokInvocationError) as raised:
        asyncio.run(hosted.judge(request, "condition", b"frame"))
    assert str(raised.value) == "hosted_byok_provider_failure"
    assert len(clients) == 1
    assert clients[0].closed


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_acquired_client_closes_when_backend_refuses_exact_model(monkeypatch, provider):
    clients = []
    if provider == "gemini":
        import percept.backends.gemini as backend_module

        adapter_type = backend_module.GeminiVision
        request = _request(provider="gemini", model="gemini-3.5-flash")

        def client_factory(request, timeout):
            client = _GeminiClient()
            clients.append(client)
            return HostedByokGeminiClient(client, _GeminiTypes)

        hosted = HostedByokAdapterFactory(gemini_client_factory=client_factory)
    else:
        import percept.backends.openrouter as backend_module

        adapter_type = backend_module.OpenRouterVision
        request = _request()

        def client_factory(request, timeout):
            client = _OpenRouterClient()
            clients.append(client)
            return client

        hosted = HostedByokAdapterFactory(openrouter_client_factory=client_factory)

    original_init = adapter_type.__init__

    def substitute_model(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._model = "provider/fallback"

    monkeypatch.setattr(adapter_type, "__init__", substitute_model)
    with pytest.raises(HostedByokUnsupportedError, match="no fallback"):
        asyncio.run(hosted.judge(request, "condition", b"frame"))
    assert len(clients) == 1
    assert clients[0].closed


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
        (_request(), b"audio", "forbids audio evidence"),
        (
            _request(provider="gemini", model="gemini-3.5-flash"),
            b"audio",
            "forbids audio evidence",
        ),
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
    with pytest.raises(HostedByokUnsupportedError, match="both image and audio"):
        asyncio.run(hosted.judge(request, "condition", None, audio=b"audio"))
    verdict = asyncio.run(hosted.judge(request, "condition", b"frame", audio=b"audio"))
    assert verdict.satisfied


@pytest.mark.parametrize(
    ("byok_request", "frame", "audio"),
    [
        (
            _request(provider="gemini", model="gemini-3.5-flash"),
            b"frame",
            b"audio",
        ),
        (
            _request(provider="gemini", model="gemini-3.5-flash"),
            None,
            b"audio",
        ),
        (
            _request(
                provider="gemini",
                model="gemini-3.5-flash",
                capability="vision.judge.fused",
            ),
            b"frame",
            None,
        ),
        (
            _request(
                provider="gemini",
                model="gemini-3.5-flash",
                capability="vision.judge.fused",
            ),
            None,
            b"audio",
        ),
        (_request(), b"frame", b"audio"),
        (_request(), None, b"audio"),
        (_request(capability="vision.judge.fused"), b"frame", b"audio"),
    ],
)
def test_exact_evidence_capability_negatives_spend_zero_fake_backend_calls(
    byok_request, frame, audio
):
    constructions = 0
    dispatches = 0

    class CountingBackend(_Backend):
        async def judge(self, condition, frame, **kwargs):
            nonlocal dispatches
            dispatches += 1
            return await super().judge(condition, frame, **kwargs)

    def factory(request, timeout):
        nonlocal constructions
        constructions += 1
        return CountingBackend()

    hosted = HostedByokAdapterFactory(
        backend_factories={"gemini": factory, "openrouter": factory}
    )
    with pytest.raises(HostedByokUnsupportedError):
        asyncio.run(hosted.judge(byok_request, "condition", frame, audio=audio))
    assert (constructions, dispatches) == (0, 0)


def test_exact_evidence_capability_negatives_spend_zero_real_adapter_transport_calls():
    gemini_clients = 0
    openrouter_clients = 0

    def gemini_factory(request, timeout):
        nonlocal gemini_clients
        gemini_clients += 1
        return HostedByokGeminiClient(_GeminiClient(), _GeminiTypes)

    def openrouter_factory(request, timeout):
        nonlocal openrouter_clients
        openrouter_clients += 1
        return _OpenRouterClient()

    hosted = HostedByokAdapterFactory(
        gemini_client_factory=gemini_factory,
        openrouter_client_factory=openrouter_factory,
    )
    attempts = (
        (
            _request(provider="gemini", model="gemini-3.5-flash"),
            b"frame",
            b"audio",
        ),
        (
            _request(
                provider="gemini",
                model="gemini-3.5-flash",
                capability="vision.judge.fused",
            ),
            b"frame",
            None,
        ),
        (_request(), b"frame", b"audio"),
        (_request(capability="vision.judge.fused"), b"frame", b"audio"),
    )
    for request, frame, audio in attempts:
        with pytest.raises(HostedByokUnsupportedError):
            asyncio.run(hosted.judge(request, "condition", frame, audio=audio))
    assert (gemini_clients, openrouter_clients) == (0, 0)


@pytest.mark.parametrize(
    ("byok_request", "kwargs"),
    [
        (_request(), {"frame": ("audio/wav", b"audio")}),
        (_request(), {"frame": b"frame", "frames": [("audio/wav", b"audio")]}),
        (_request(), {"frame": b"frame", "exemplar": ("audio/wav", b"audio")}),
        (
            _request(
                provider="gemini",
                model="gemini-3.5-flash",
                capability="vision.judge.fused",
            ),
            {"frame": b"frame", "audio": ("image/jpeg", b"not-audio")},
        ),
        (
            _request(
                provider="gemini",
                model="gemini-3.5-flash",
                capability="vision.judge.fused",
            ),
            {
                "frame": b"frame",
                "audio": b"audio",
                "context": {"process_qa": True},
            },
        ),
        (
            _request(
                provider="gemini",
                model="gemini-3.5-flash",
                capability="vision.judge.fused",
            ),
            {
                "frame": b"frame",
                "audio": b"audio",
                "context": {"observe_step": True},
            },
        ),
    ],
)
def test_media_and_context_routes_cannot_widen_or_discard_declared_capability(
    byok_request, kwargs
):
    constructions = 0

    def factory(request, timeout):
        nonlocal constructions
        constructions += 1
        return _Backend()

    hosted = HostedByokAdapterFactory(
        backend_factories={"gemini": factory, "openrouter": factory}
    )
    call_kwargs = dict(kwargs)
    frame = call_kwargs.pop("frame")
    with pytest.raises(HostedByokUnsupportedError):
        asyncio.run(hosted.judge(byok_request, "condition", frame, **call_kwargs))
    assert constructions == 0


@pytest.mark.parametrize(
    "audio",
    [
        AudioFrame(pcm=b"", encoded=b"RIFF", mime=None),
        AudioFrame(pcm=b"", encoded=b"RIFF", mime="image/jpeg"),
        AudioFrame(pcm=b"raw", sample_rate=0),
        AudioFrame(pcm=b"raw", channels=0),
        AudioFrame(pcm=b"raw", sample_format="not-pcm"),
        AudioFrame(pcm=b"raw", mime="audio/wav"),
        AudioFrame(pcm=b"raw", t_start=2.0, t_end=1.0),
    ],
)
def test_fused_typed_audio_rejects_invalid_mime_metadata_or_modality_before_construction(audio):
    constructions = 0

    def factory(request, timeout):
        nonlocal constructions
        constructions += 1
        return _Backend()

    with pytest.raises(HostedByokUnsupportedError):
        asyncio.run(
            HostedByokAdapterFactory(backend_factories={"gemini": factory}).judge(
                _request(
                    provider="gemini",
                    model="gemini-3.5-flash",
                    capability="vision.judge.fused",
                ),
                "condition",
                b"frame",
                audio=audio,
            )
        )
    assert constructions == 0


# Every key the runtime owns, taken from the registry rather than restated -- a key added there must
# be refused here, and a hand-copied list is exactly how this fell two keys behind.
@pytest.mark.parametrize("reserved_key", sorted(RESERVED_CONTEXT_KEYS))
def test_reserved_context_key_presence_rejects_even_false_values_before_construction(reserved_key):
    constructions = 0

    def factory(request, timeout):
        nonlocal constructions
        constructions += 1
        return _Backend()

    with pytest.raises(HostedByokUnsupportedError, match="reserved context"):
        asyncio.run(
            HostedByokAdapterFactory(backend_factories={"gemini": factory}).judge(
                _request(
                    provider="gemini",
                    model="gemini-3.5-flash",
                    capability="vision.judge.fused",
                ),
                "condition",
                b"frame",
                audio=AudioFrame(pcm=b"\x01\x00", sample_rate=16_000),
                context={reserved_key: False},
            )
        )
    assert constructions == 0


def test_fake_backend_receives_frames_context_and_exact_declared_modalities():
    backends: list[_Backend] = []

    def factory(request, timeout):
        backend = _Backend()
        backends.append(backend)
        return backend

    hosted = HostedByokAdapterFactory(
        backend_factories={"gemini": factory, "openrouter": factory}
    )
    plain_context = {"specialist": "plain"}
    asyncio.run(
        hosted.judge(
            _request(),
            "plain",
            ("image/jpeg", b"anchor"),
            frames=[("image/jpeg", b"earlier")],
            context=plain_context,
        )
    )
    assert backends[0].calls[0][2]["frames"] == [("image/jpeg", b"earlier")]
    assert backends[0].calls[0][2]["context"] is plain_context
    assert backends[0].calls[0][2]["audio"] is None

    fused_context = {"specialist": "fused"}
    audio = ("audio/wav", b"RIFF-audio")
    asyncio.run(
        hosted.judge(
            _request(
                provider="gemini",
                model="gemini-3.5-flash",
                capability="vision.judge.fused",
            ),
            "fused",
            ("image/jpeg", b"anchor"),
            frames=[("image/jpeg", b"earlier")],
            audio=audio,
            context=fused_context,
        )
    )
    assert backends[1].calls[0][2]["frames"] == [("image/jpeg", b"earlier")]
    assert backends[1].calls[0][2]["context"] is fused_context
    assert backends[1].calls[0][2]["audio"] == audio


def test_real_gemini_frames_and_context_keep_audio_in_fused_transport():
    clients: list[_GeminiClient] = []

    def client_factory(request, timeout):
        client = _GeminiClient()
        clients.append(client)
        return HostedByokGeminiClient(client, _GeminiTypes)

    hosted = HostedByokAdapterFactory(gemini_client_factory=client_factory)
    verdict = asyncio.run(
        hosted.judge(
            _request(
                provider="gemini",
                model="gemini-3.5-flash",
                capability="vision.judge.fused",
            ),
            "condition",
            ("image/jpeg", b"anchor"),
            frames=[("image/png", b"earlier")],
            audio=("audio/wav", b"RIFF-audio"),
            context={"specialist": "trusted"},
        )
    )
    assert not verdict.satisfied
    contents = clients[0].aio.models.calls[0]["contents"]
    assert ("part", "image/png", b"earlier") in contents
    assert ("part", "image/jpeg", b"anchor") in contents
    assert ("part", "audio/wav", b"RIFF-audio") in contents
    assert clients[0].closed


def test_real_openrouter_frames_and_context_remain_image_only():
    clients: list[_OpenRouterClient] = []

    def client_factory(request, timeout):
        client = _OpenRouterClient()
        clients.append(client)
        return client

    verdict = asyncio.run(
        HostedByokAdapterFactory(openrouter_client_factory=client_factory).judge(
            _request(),
            "condition",
            ("image/jpeg", b"anchor"),
            frames=[("image/png", b"earlier")],
            context={"specialist": "trusted"},
        )
    )
    assert verdict.satisfied
    payload = clients[0].chat.completions.calls[0]
    content = payload["messages"][1]["content"]
    urls = [
        item["image_url"]["url"]
        for item in content
        if item.get("type") == "image_url"
    ]
    assert len(urls) == 2
    assert all(url.startswith("data:image/") for url in urls)
    assert clients[0].closed


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


def test_plain_bound_adapter_cannot_arm_audio_or_fused_watch():
    client_calls = 0

    def client_factory(request, timeout_s):
        nonlocal client_calls
        client_calls += 1
        return HostedByokGeminiClient(_GeminiClient(), _GeminiTypes)

    vision = HostedByokAdapterFactory(gemini_client_factory=client_factory).bind(
        _request(provider="gemini", model="gemini-3.5-flash")
    )
    for sense in ("audio", "fused"):
        core = Percept.create(
            vision=vision,
            settings=Settings(),
            mode="offline",
            discover_plugins=False,
        )
        receipt = apply_watch_plan(
            core,
            WatchPlan(id=sense, condition="speaker points", sense=sense),
        )
        assert receipt.status == "rejected"
        assert receipt.dropped_fields[0].code == "audio_input_backend_unsupported"
        assert core.goals.all() == []
    assert client_calls == 0


def test_fused_bound_adapter_arms_real_watchplan_and_runtime_uses_fake_transport():
    clients: list[_GeminiClient] = []

    def client_factory(request, timeout_s):
        client = _GeminiClient()
        clients.append(client)
        return HostedByokGeminiClient(client, _GeminiTypes)

    request = _request(
        provider="gemini",
        model="gemini-3.5-flash",
        capability="vision.judge.fused",
    )
    vision = HostedByokAdapterFactory(gemini_client_factory=client_factory).bind(request)
    core = Percept.create(
        vision=vision,
        settings=Settings(),
        mode="offline",
        discover_plugins=False,
    )
    receipt = apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="speaker points", sense="fused"),
    )
    assert receipt.status == "applied"
    binding = core._audio_judge_bindings["fused"]
    assert binding.adapter is vision
    assert binding.mode == "fused"
    assert binding.capability.modes == frozenset({"fused"})

    still = gray_frame(48, 36, 0)
    changed = gray_frame(48, 36, 255)
    # A compiled fused Watch over two SEPARATE iterable lanes now demands an explicit, matching
    # clock-calibration proof before it will iterate either lane — the runtime refuses to assume
    # two independent iterables share a clock merely because one worker consumes them. The replay
    # fixture declares the shared authority; nothing about the hosted boundary changes.
    video = _ClockedReplay.video(
        [
            (0.0, still, ("image/jpeg", b"prime")),
            (1.0, changed, ("image/jpeg", b"changed")),
        ]
    )
    # 0.5 s of real PCM, not a 6 ms token: a fused Watch now requires the audio unit to actually
    # COVER the frame's evidence window within the policy's skew bound before it will dispatch a
    # fused judgment, and honestly degrades to ASK when it does not.
    audio = _ClockedReplay.audio(
        [
            (
                0.5,
                AudioFrame(pcm=b"\x01\x00" * 8_000, sample_rate=16_000),
                0.5,
            )
        ]
    )

    async def collect():
        return [
            incident
            async for incident in run(
                core,
                source=video,
                audio_source=audio,
                audio_gate=None,
                judge_dispatch="inline",
            )
        ]

    assert asyncio.run(collect()) == []
    assert len(clients) == 1
    assert clients[0].closed
    call = clients[0].aio.models.calls[0]
    assert call["model"] == "gemini-3.5-flash"
    assert ("part", "image/jpeg", b"changed") in call["contents"]
    assert any(part[1] == "audio/wav" for part in call["contents"] if isinstance(part, tuple))


def test_compiled_fused_binding_captures_factory_route_before_poison_mutation(monkeypatch):
    routed: list[str] = []

    def accepted_route(request, timeout):
        routed.append("accepted")
        return _Backend(Verdict(True, 0.9, "sealed"))

    hosted = HostedByokAdapterFactory(backend_factories={"gemini": accepted_route})
    vision = hosted.bind(
        _request(
            provider="gemini",
            model="gemini-3.5-flash",
            capability="vision.judge.fused",
        )
    )
    binding = prepare_audio_judge_binding(
        vision,
        mode="fused",
        application_key="compiled-test",
    )
    with pytest.raises(TypeError):
        hosted._backend_factories["gemini"] = lambda request, timeout: _Backend()
    with pytest.raises(AttributeError, match="immutable"):
        hosted._build_invocation = lambda *args: _Backend()
    monkeypatch.setattr(
        HostedByokAdapterFactory,
        "_build_invocation",
        lambda *args: (_ for _ in ()).throw(AssertionError("poisoned class helper")),
    )

    verdict = asyncio.run(
        binding.dispatch(
            "condition",
            b"frame",
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate=16_000),
        )
    )
    assert verdict.evidence == "sealed"
    assert routed == ["accepted"]


def test_compiled_fused_binding_captures_exact_gemini_class_before_module_replacement(monkeypatch):
    import percept.backends.gemini as gemini_module

    clients = []

    def client_factory(request, timeout):
        client = _GeminiClient()
        clients.append(client)
        return HostedByokGeminiClient(client, _GeminiTypes)

    hosted = HostedByokAdapterFactory(gemini_client_factory=client_factory)
    vision = hosted.bind(
        _request(
            provider="gemini",
            model="gemini-3.5-flash",
            capability="vision.judge.fused",
        )
    )
    binding = prepare_audio_judge_binding(
        vision,
        mode="fused",
        application_key="sealed-gemini-class",
    )

    class PoisonGemini:
        def __init__(self, *args, **kwargs):
            raise AssertionError("live GeminiVision import was used")

    monkeypatch.setattr(gemini_module, "GeminiVision", PoisonGemini)
    verdict = asyncio.run(
        binding.dispatch(
            "condition",
            b"frame",
            audio=AudioFrame(pcm=b"\x01\x00", sample_rate=16_000),
        )
    )
    assert not verdict.satisfied
    assert len(clients) == 1


def test_public_exports_are_cloud_usable():
    expected = {
        "HOSTED_BYOK_CONTRACT_VERSION",
        "HOSTED_BYOK_PROCESS_ISOLATION_VERSION",
        "HostedByokRequest",
        "HostedByokAdapterFactory",
        "HostedByokVision",
        "HostedByokCancelled",
        "HostedByokDeadlineExceeded",
        "HostedByokProcessSupervisor",
        "HostedByokSubprocessSupervisor",
        "HostedByokTransportSafety",
        "HostedByokUnsupportedError",
    }
    assert expected <= set(percept.__all__)
    for name in expected:
        assert getattr(percept, name) is not None
