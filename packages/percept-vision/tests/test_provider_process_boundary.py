"""External provider lifetime: no Python scheduling escape can outlive return."""
from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fallback import FallbackVision
from percept.backends.gemini import GeminiVision
from percept.backends.openrouter import OpenRouterVision
from percept.config import ModelSettings
from percept.contracts import (
    AudioFrame,
    Box,
    ClipEvent,
    Observation,
    SubjectRef,
)
from percept.events import Event
from percept.harness.tier0 import Frame, gray_frame
from percept.runtime import provider_process
from percept.runtime.provider_process import (
    ProviderProcessError,
    ProviderWatchdogExpired,
    register_process_provider,
    run_provider_process,
)
from percept.runtime.worker import _ACTIVE_CORE_RUNS
from percept.runtime.worker import run as worker_run

_FORBIDDEN_CHILD_ENVIRONMENT = (
    "ALL_PROXY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_CONFIG_FILE",
    "AWS_PROFILE",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "CLOUDSDK_CONFIG",
    "CURL_CA_BUNDLE",
    "DEEPGRAM_API_KEY",
    "DYLD_INSERT_LIBRARIES",
    "GEMINI_API_KEY",
    "GOOGLE_AISTUDIO_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_API_USE_CLIENT_CERTIFICATE",
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
    "GRPC_PROXY_EXP",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "KUBECONFIG",
    "LD_PRELOAD",
    "NODE_EXTRA_CA_CERTS",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "PERCEPT_STRIP_PROMPT",
    "PERCEPT_VISION_TIMEOUT_S",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "VIRTUAL_ENV",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
)


def _present_forbidden_environment() -> tuple[str, ...]:
    return tuple(
        name for name in _FORBIDDEN_CHILD_ENVIRONMENT if name in os.environ
    )


_FORBIDDEN_AT_TEST_MODULE_IMPORT = _present_forbidden_environment()


def _production_boundary_core(vision: Any) -> Percept:
    """Build a core without the repository test harness's shared-memory grant."""
    core = Percept.create(vision=vision, discover_plugins=False)
    core.__dict__.pop(
        "_reviewed_test_in_process_provider_capability",
        None,
    )
    return core


def _late_write(path: str, label: str, delay_s: float = 0.2) -> None:
    time.sleep(delay_s)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(label + "\n")


async def _late_async_write(path: str, label: str) -> None:
    await asyncio.sleep(0.2)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(label + "\n")


class _EscapeProvider:
    def __init__(self, effects: str) -> None:
        self.effects = effects

    async def __call__(self) -> tuple[str, bool]:
        loop = asyncio.get_running_loop()

        # A provider-created process inherits the containment process group.
        try:
            descendant = os.fork()
        except OSError:
            # The production sandbox denies process descendants outright.
            descendant = -1
        fork_denied = descendant == -1
        if descendant == 0:  # pragma: no cover - only on a weaker platform
            _late_write(self.effects, "forked-descendant")
            os._exit(0)

        # Bypass create_task and ambient Context directly.
        asyncio.Task(
            _late_async_write(self.effects, "direct-task"),
            loop=loop,
        )
        empty = contextvars.Context()
        if sys.version_info >= (3, 11):
            loop.create_task(
                _late_async_write(self.effects, "empty-context-task"),
                context=empty,
            )
        else:
            empty.run(
                loop.create_task,
                _late_async_write(self.effects, "empty-context-task"),
            )

        # Replace the loop factory and create another descendant through it.
        def replacement_factory(
            factory_loop: asyncio.AbstractEventLoop,
            coroutine: Any,
            **_kwargs: Any,
        ) -> asyncio.Task[Any]:
            return asyncio.Task(coroutine, loop=factory_loop)

        loop.set_task_factory(replacement_factory)
        loop.create_task(
            _late_async_write(self.effects, "replacement-factory")
        )

        # A scheduled callback may itself create non-async work.
        loop.call_soon(
            lambda: threading.Thread(
                target=_late_write,
                args=(self.effects, "call-soon-thread"),
                daemon=False,
            ).start()
        )
        loop.call_soon(
            lambda: threading.Thread(
                target=_late_write,
                args=(self.effects, "empty-context-call-soon"),
                daemon=False,
            ).start(),
            context=empty,
        )
        loop.run_in_executor(
            None,
            _late_write,
            self.effects,
            "executor-thread",
        )

        try:
            await asyncio.sleep(0)
            return ("frozen-result", fork_denied)
        finally:
            # Teardown remains contained even when the root creates new work
            # while leaving its own finally block.
            asyncio.Task(
                _late_async_write(self.effects, "finally-direct-task"),
                loop=loop,
            )
            empty.run(
                loop.create_task,
                _late_async_write(self.effects, "finally-empty-context"),
            )


register_process_provider(
    _EscapeProvider,
    module="tests.test_provider_process_boundary",
    qualname="_EscapeProvider",
    constructor_fields=("effects",),
)


class _ResistantProvider:
    def __init__(self, effects: str) -> None:
        self.effects = effects

    async def __call__(self) -> None:
        threading.Thread(
            target=_late_write,
            args=(self.effects, "resistant-thread"),
            daemon=False,
        ).start()
        while True:
            try:
                await asyncio.Event().wait()
            except BaseException:  # noqa: BLE001 - adversarial provider
                continue


register_process_provider(
    _ResistantProvider,
    module="tests.test_provider_process_boundary",
    qualname="_ResistantProvider",
    constructor_fields=("effects",),
)


class _SelfCancellingProvider:
    async def __call__(self) -> str:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        return "still-owned-by-parent"


register_process_provider(
    _SelfCancellingProvider,
    module="tests.test_provider_process_boundary",
    qualname="_SelfCancellingProvider",
)


def test_terminal_result_kills_every_async_thread_and_process_escape(
    tmp_path: Path,
) -> None:
    effects = str(tmp_path / "late-effects.txt")

    async def drive() -> None:
        assert await run_provider_process(
            _EscapeProvider(effects),
            "__call__",
            (),
            {},
            watchdog_s=1.0,
        ) == ("frozen-result", True)
        assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()

    asyncio.run(drive())
    time.sleep(0.35)
    assert not Path(effects).exists()


def test_watchdog_kills_and_reaps_indefinitely_resistant_provider(
    tmp_path: Path,
) -> None:
    effects = str(tmp_path / "watchdog-late-effects.txt")

    async def drive() -> None:
        started = time.monotonic()
        with pytest.raises(ProviderWatchdogExpired):
            await run_provider_process(
                _ResistantProvider(effects),
                "__call__",
                (),
                {},
                watchdog_s=0.05,
            )
        assert time.monotonic() - started < 0.75
        assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()

    asyncio.run(drive())
    time.sleep(0.35)
    assert not Path(effects).exists()


def test_parent_cancellation_kills_and_reaps_before_propagation(
    tmp_path: Path,
) -> None:
    effects = str(tmp_path / "cancel-late-effects.txt")

    async def drive() -> None:
        invocation = asyncio.create_task(
            run_provider_process(
                _ResistantProvider(effects),
                "__call__",
                (),
                {},
                watchdog_s=5.0,
            )
        )
        await asyncio.sleep(0.05)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()

    asyncio.run(drive())
    time.sleep(0.35)
    assert not Path(effects).exists()


def test_cancellation_during_subprocess_launch_still_obtains_and_reaps_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_spawn = asyncio.create_subprocess_exec

    async def drive() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_spawn(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            return await original_spawn(*args, **kwargs)

        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            delayed_spawn,
        )
        invocation = asyncio.create_task(
            run_provider_process(
                _SelfCancellingProvider(),
                "__call__",
                (),
                {},
                watchdog_s=1.0,
            )
        )
        await entered.wait()
        invocation.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()

    asyncio.run(drive())


def test_child_cannot_cancel_the_live_parent_owner_task() -> None:
    async def drive() -> None:
        owner = asyncio.current_task()
        assert owner is not None

        assert await run_provider_process(
            _SelfCancellingProvider(),
            "__call__",
            (),
            {},
            watchdog_s=1.0,
        ) == "still-owned-by-parent"
        await asyncio.sleep(0)
        assert not owner.cancelled()

    asyncio.run(drive())


class _UnmarkedCustomVision:
    def __init__(self, effects: str) -> None:
        self.effects = effects
        self.parent_visible_calls = 0

    async def detect(self, _frame: Any, _goals: Any) -> list[Any]:
        return []

    async def judge(
        self,
        _condition: str,
        _frame: Any,
        **_kwargs: Any,
    ) -> Verdict:
        self.parent_visible_calls += 1
        threading.Thread(
            target=_late_write,
            args=(self.effects, "adapter-thread"),
            daemon=False,
        ).start()
        return Verdict(
            satisfied=True,
            confidence=0.99,
            evidence="isolated",
        )


register_process_provider(
    _UnmarkedCustomVision,
    module="tests.test_provider_process_boundary",
    qualname="_UnmarkedCustomVision",
    constructor_fields=("effects",),
)


def test_worker_releases_core_only_after_provider_process_is_dead(
    tmp_path: Path,
) -> None:
    effects = str(tmp_path / "worker-late-effects.txt")
    vision = _UnmarkedCustomVision(effects)
    assert not hasattr(vision, "RUNTIME_EXECUTION_BOUNDARY")
    core = _production_boundary_core(vision)
    core.add_goal(Goal(id="g", condition="motion", say="seen"))
    still = gray_frame(8, 8, 0)
    moving_pixels = [0] * 64
    for index in range(40):
        moving_pixels[index] = 255
    moving = Frame(w=8, h=8, pixels=tuple(moving_pixels))

    async def source():
        yield (0.0, still, "f0")
        await asyncio.sleep(0)
        yield (1.0, moving, "f1")
        await asyncio.sleep(0)

    async def drive() -> list[Any]:
        return [
            incident
            async for incident in worker_run(
                core,
                source=source(),
                judge_dispatch="async",
            )
        ]

    incidents = asyncio.run(drive())
    assert len(incidents) == 1
    assert incidents[0].verdict.evidence == "isolated"
    assert vision.parent_visible_calls == 0
    assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()
    assert id(core) not in _ACTIVE_CORE_RUNS

    time.sleep(0.35)
    assert not Path(effects).exists()


class _BoundarySpoofingVision(_UnmarkedCustomVision):
    RUNTIME_EXECUTION_BOUNDARY = "in_process"


def test_custom_adapter_cannot_self_authorize_in_process_execution(
    tmp_path: Path,
) -> None:
    vision = _BoundarySpoofingVision(str(tmp_path / "must-not-run.txt"))
    core = _production_boundary_core(vision)

    async def drive() -> None:
        with pytest.raises(
            ProviderProcessError,
            match="not explicitly registered",
        ):
            await core._invoke_provider_call(
                vision.judge,
                ("condition", "frame"),
                {},
                operation="judge",
                adapter=vision,
            )
        assert vision.parent_visible_calls == 0

    asyncio.run(drive())


class _HostileAdapter:
    def __init__(self, effects: str) -> None:
        object.__setattr__(self, "effects", effects)

    def _effect(self, label: str) -> None:
        with open(
            object.__getattribute__(self, "effects"),
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(label + "\n")

    def __getattribute__(self, name: str) -> Any:
        if name not in {"effects", "_effect"}:
            object.__getattribute__(self, "_effect")("getattribute")
        return object.__getattribute__(self, name)

    @property
    def __dict__(self) -> dict[str, Any]:
        self._effect("dict-property")
        return {}

    def __reduce__(self) -> Any:
        self._effect("reduce")
        return (_HostileAdapter, (self.effects,))

    def __repr__(self) -> str:
        self._effect("repr")
        return "hostile"


class _HostileValue:
    def __init__(self, effects: str) -> None:
        object.__setattr__(self, "effects", effects)

    def _effect(self, label: str) -> None:
        with open(
            object.__getattribute__(self, "effects"),
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(label + "\n")

    def __getattribute__(self, name: str) -> Any:
        if name not in {"effects", "_effect"}:
            object.__getattribute__(self, "_effect")("getattribute")
        return object.__getattribute__(self, name)

    def __reduce__(self) -> Any:
        self._effect("reduce")
        return (_HostileValue, (self.effects,))

    def __repr__(self) -> str:
        self._effect("repr")
        return "hostile"


class _HostileString(str):
    effects: str

    def __new__(cls, value: str, effects: str) -> "_HostileString":
        instance = super().__new__(cls, value)
        instance.effects = effects
        return instance

    def __reduce__(self) -> Any:
        with open(self.effects, "a", encoding="utf-8") as stream:
            stream.write("reduce\n")
        return (_HostileString, (str(self), self.effects))

    def __repr__(self) -> str:
        with open(self.effects, "a", encoding="utf-8") as stream:
            stream.write("repr\n")
        return "hostile"


class _PoisonedResult:
    def __init__(self, effects: str) -> None:
        self.effects = effects

    def __reduce__(self) -> Any:
        with open(self.effects, "a", encoding="utf-8") as stream:
            stream.write("reduce\n")
        return (_PoisonedResult, (self.effects,))


class _PoisonedResultProvider:
    def __init__(self, effects: str) -> None:
        self.effects = effects

    async def __call__(self) -> Any:
        return _PoisonedResult(self.effects)


register_process_provider(
    _PoisonedResultProvider,
    module="tests.test_provider_process_boundary",
    qualname="_PoisonedResultProvider",
    constructor_fields=("effects",),
)


class _DtoProvider:
    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Box | None = None,
        audio: AudioFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> Verdict:
        valid = (
            condition == "condition"
            and frame == ("image/jpeg", b"jpeg")
            and type(roi) is Box
            and type(audio) is AudioFrame
            and context == {"strict": True}
        )
        return Verdict(
            satisfied=valid,
            confidence=0.9,
            evidence="typed",
            usage={"input_tokens": 3},
            events=[ClipEvent(0.25, 0.5, "motion")],
            subject=SubjectRef(
                scheme="content",
                namespace="camera",
                key="subject-1",
                resolution="bound",
                scope="source",
                strategy="test",
                confidence=0.8,
            ),
            value={"raw": b"value"},
        )

    async def detect(
        self,
        _frame: Any,
        _goals: list[Goal],
    ) -> list[Observation]:
        return [Observation("thing", Box(0.1, 0.2, 0.3, 0.4), 0.75)]

    async def detect_events(
        self,
        _frames: list[Any],
        _goals: list[Goal],
    ) -> list[Event]:
        return [Event("motion", 1.25, ["subject-1"], 0.8, "event-1")]


register_process_provider(
    _DtoProvider,
    module="tests.test_provider_process_boundary",
    qualname="_DtoProvider",
)


class _EnvironmentProbeProvider:
    def __init__(self, credential: str, mode: str) -> None:
        self.credential = credential
        self.mode = mode
        self.import_forbidden = _FORBIDDEN_AT_TEST_MODULE_IMPORT
        self.constructor_forbidden = _present_forbidden_environment()

    def _is_sealed(self) -> bool:
        return not (
            self.import_forbidden
            or self.constructor_forbidden
            or _present_forbidden_environment()
        )

    async def __call__(self) -> dict[str, Any]:
        return {
            "credential_received": (
                self.credential == "reviewed-explicit-credential"
            ),
            "environment": dict(os.environ),
            "import_forbidden": list(self.import_forbidden),
            "constructor_forbidden": list(self.constructor_forbidden),
            "call_forbidden": list(_present_forbidden_environment()),
        }

    async def judge(
        self,
        _condition: str,
        _frame: Any,
        **_kwargs: Any,
    ) -> Verdict:
        sealed = self._is_sealed()
        if self.mode == "primary":
            if sealed:
                raise RuntimeError("exercise registered fallback")
            return Verdict(False, 0.0, "primary environment was not sealed")
        return Verdict(
            sealed,
            1.0 if sealed else 0.0,
            "registered-fallback",
        )


register_process_provider(
    _EnvironmentProbeProvider,
    module="tests.test_provider_process_boundary",
    qualname="_EnvironmentProbeProvider",
    constructor_fields=("credential", "mode"),
    credential_fields=("credential",),
)


class _CredentialReflectingProvider:
    def __init__(
        self,
        credential: str,
        mode: str,
        reflected: str = "nonsecret",
    ) -> None:
        self.credential = credential
        self.mode = mode
        self.reflected = reflected

    async def __call__(self) -> Any:
        if self.mode == "plain":
            return {
                f"key::{self.credential}": {
                    "value": self.credential,
                    "bytes": self.credential.encode(
                        "utf-8",
                        errors="surrogatepass",
                    ),
                }
            }
        if self.mode == "error":
            raise RuntimeError("fixed provider failure")
        return "nonsecret"

    async def judge(
        self,
        _condition: str,
        _frame: Any,
        **_kwargs: Any,
    ) -> Verdict:
        if self.mode == "error":
            raise RuntimeError("fixed provider failure")
        if self.mode == "verdict":
            return Verdict(
                satisfied=True,
                confidence=1.0,
                evidence=f"reflected::{self.credential}",
                usage={
                    f"key::{self.credential}": self.credential.encode(
                        "utf-8",
                        errors="surrogatepass",
                    )
                },
            )
        return Verdict(True, 1.0, self.reflected)


register_process_provider(
    _CredentialReflectingProvider,
    module="tests.test_provider_process_boundary",
    qualname="_CredentialReflectingProvider",
    constructor_fields=("credential", "mode", "reflected"),
    credential_fields=("credential",),
)


class _CredentialLookingDataProvider:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


register_process_provider(
    _CredentialLookingDataProvider,
    module="tests.test_provider_process_boundary",
    qualname="_CredentialLookingDataProvider",
    constructor_fields=("api_key",),
)


def _install_hostile_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    sentinels = {
        name: f"sentinel::{name}::$()\\nPYTHONPATH=/attacker"
        for name in _FORBIDDEN_CHILD_ENVIRONMENT
    }
    for name, value in sentinels.items():
        monkeypatch.setenv(name, value)
    return sentinels


def test_registered_child_environment_is_exact_and_credential_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = _install_hostile_parent_environment(monkeypatch)

    async def drive() -> None:
        result = await run_provider_process(
            _EnvironmentProbeProvider(
                "reviewed-explicit-credential",
                "return",
            ),
            "__call__",
            (),
            {},
            watchdog_s=1.0,
        )
        assert result == {
            "credential_received": True,
            "environment": provider_process._provider_child_environment(),
            "import_forbidden": [],
            "constructor_forbidden": [],
            "call_forbidden": [],
        }

    asyncio.run(drive())
    assert all(os.environ[name] == value for name, value in sentinels.items())


def test_registered_fallback_graph_is_constructed_only_after_env_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_hostile_parent_environment(monkeypatch)
    adapter = FallbackVision(
        _EnvironmentProbeProvider("typed-primary", "primary"),
        _EnvironmentProbeProvider("typed-fallback", "fallback"),
    )

    async def drive() -> None:
        verdict = await run_provider_process(
            adapter,
            "judge",
            ("condition", ("image/jpeg", b"jpeg")),
            {},
            watchdog_s=1.0,
        )
        assert verdict.satisfied is True
        assert verdict.evidence == "registered-fallback"

    asyncio.run(drive())


def _assert_credential_absent_from_error(
    error: BaseException,
    *credentials: str,
) -> None:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered = f"{type(current).__name__}: {current}"
        assert all(
            not credential or credential not in rendered
            for credential in credentials
        )
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


@pytest.mark.parametrize("mode", ["plain", "verdict"])
def test_actual_registered_child_cannot_return_selected_credential(
    mode: str,
) -> None:
    credential = f"registered-{mode}-credential-7e91"

    async def drive() -> None:
        operation = "__call__" if mode == "plain" else "judge"
        args = () if mode == "plain" else ("condition", None)
        with pytest.raises(
            ProviderProcessError,
            match="credential-bearing completion",
        ) as caught:
            await run_provider_process(
                _CredentialReflectingProvider(credential, mode),
                operation,
                args,
                {},
                watchdog_s=1.0,
            )
        _assert_credential_absent_from_error(caught.value, credential)

    asyncio.run(drive())


def test_nested_fallback_tracks_every_registered_credential() -> None:
    primary_credential = "nested-primary-credential-a1"
    middle_credential = "nested-middle-credential-b2"
    final_credential = "nested-final-credential-c3"
    adapter = FallbackVision(
        _CredentialReflectingProvider(
            primary_credential,
            "error",
        ),
        FallbackVision(
            _CredentialReflectingProvider(
                middle_credential,
                "error",
            ),
            _CredentialReflectingProvider(
                final_credential,
                "return",
                reflected=primary_credential,
            ),
        ),
    )
    request = provider_process._build_request(
        adapter,
        "judge",
        ("condition", None),
        {},
    )
    assert request.credentials == (
        primary_credential,
        middle_credential,
        final_credential,
    )
    assert repr(request) == "_BuiltProviderRequest(<redacted>)"
    assert all(
        credential not in repr(request)
        for credential in request.credentials
    )

    async def drive() -> None:
        with pytest.raises(
            ProviderProcessError,
            match="credential-bearing completion",
        ) as caught:
            await run_provider_process(
                adapter,
                "judge",
                ("condition", None),
                {},
                watchdog_s=1.0,
            )
        _assert_credential_absent_from_error(
            caught.value,
            *request.credentials,
        )

    asyncio.run(drive())


def test_builtin_fallback_cannot_reflect_selected_credential() -> None:
    credential = (
        "GeminiVision requires image evidence, audio evidence, or both"
    )
    adapter = FallbackVision(
        GeminiVision(
            ModelSettings(vision_model="gemini-3.5-flash"),
            api_key=credential,
        )
    )

    async def drive() -> None:
        with pytest.raises(
            ProviderProcessError,
            match="credential-bearing completion",
        ) as caught:
            await run_provider_process(
                adapter,
                "judge",
                ("condition", None),
                {},
                watchdog_s=5.0,
            )
        _assert_credential_absent_from_error(caught.value, credential)

    asyncio.run(drive())


def test_registered_credential_contract_does_not_infer_field_names() -> None:
    ordinary_value = "api-key-shaped-ordinary-data"
    request = provider_process._build_request(
        _CredentialLookingDataProvider(ordinary_value),
        "__call__",
        (),
        {},
    )
    assert request.credentials == ()


@pytest.mark.parametrize(
    ("constructor_fields", "credential_fields", "message"),
    [
        (
            ("credential",),
            ["credential"],
            "credential fields must be exact identifiers",
        ),
        (
            ("credential",),
            ("not-valid!",),
            "credential fields must be exact identifiers",
        ),
        (
            ("credential",),
            ("credential", "credential"),
            "credential fields must be unique",
        ),
        (
            ("credential",),
            ("missing",),
            "credential fields must be constructor fields",
        ),
    ],
)
def test_registered_credential_field_contract_is_explicit(
    constructor_fields: tuple[str, ...],
    credential_fields: Any,
    message: str,
) -> None:
    class RegistrationCandidate:
        pass

    with pytest.raises(ValueError, match=message):
        register_process_provider(
            RegistrationCandidate,
            module="tests.test_provider_process_boundary",
            qualname="RegistrationCandidate",
            constructor_fields=constructor_fields,
            credential_fields=credential_fields,
        )


@pytest.mark.parametrize(
    "credential",
    [
        pytest.param("", id="empty"),
        pytest.param(7, id="non-string"),
        pytest.param("x" * 65_537, id="oversized"),
    ],
)
def test_registered_credential_value_fails_before_launch(
    credential: Any,
) -> None:
    with pytest.raises(
        ProviderProcessError,
        match="credential field is outside the reviewed schema",
    ) as caught:
        provider_process._build_request(
            _CredentialReflectingProvider(credential, "plain"),
            "__call__",
            (),
            {},
        )
    _assert_credential_absent_from_error(caught.value, str(credential))


def test_credential_matching_child_error_type_is_not_reflected() -> None:
    credential = "RuntimeError"

    async def drive() -> None:
        with pytest.raises(
            ProviderProcessError,
            match="credential-bearing completion",
        ) as caught:
            await run_provider_process(
                _CredentialReflectingProvider(credential, "error"),
                "__call__",
                (),
                {},
                watchdog_s=1.0,
            )
        _assert_credential_absent_from_error(caught.value, credential)

    asyncio.run(drive())


def _encoded_result_payload(value: Any) -> bytes:
    return provider_process._encode_json_bytes(
        {
            "schema": provider_process._SCHEMA_VERSION,
            "status": "ok",
            "value": provider_process._encode_provider_result(value),
        },
        limit=provider_process._MAX_RESULT_BYTES,
    )


@pytest.mark.parametrize(
    "make_result",
    [
        pytest.param(lambda secret: secret, id="primitive-string"),
        pytest.param(
            lambda secret: secret.encode("utf-8"),
            id="primitive-bytes",
        ),
        pytest.param(lambda secret: {secret: "value"}, id="mapping-key"),
        pytest.param(
            lambda secret: {"key": bytearray(secret.encode("utf-8"))},
            id="mapping-byte-value",
        ),
        pytest.param(
            lambda secret: AudioFrame(pcm=secret.encode("utf-8")),
            id="audio-frame",
        ),
        pytest.param(
            lambda secret: Goal(
                id=secret,
                condition="condition",
                say="say",
            ),
            id="goal",
        ),
        pytest.param(
            lambda secret: ClipEvent(0.0, 1.0, secret),
            id="clip-event",
        ),
        pytest.param(
            lambda secret: SubjectRef(
                scheme="content",
                namespace="camera",
                key=secret,
                resolution="bound",
                scope="source",
                strategy="test",
            ),
            id="subject-ref",
        ),
        pytest.param(
            lambda secret: Observation(
                secret,
                Box(0.0, 0.0, 1.0, 1.0),
                1.0,
            ),
            id="observation",
        ),
        pytest.param(
            lambda secret: Event(
                "motion",
                0.0,
                [secret],
                1.0,
                "event",
            ),
            id="event",
        ),
        pytest.param(
            lambda secret: Verdict(
                True,
                1.0,
                "safe",
                usage={secret: secret.encode("utf-8")},
            ),
            id="verdict-recursive",
        ),
    ],
)
def test_selected_credential_is_rejected_in_every_result_carrier(
    make_result: Any,
) -> None:
    credential = "recursive-result-credential-f4"
    with pytest.raises(
        ProviderProcessError,
        match="credential-bearing completion",
    ) as caught:
        provider_process._decode_provider_result(
            _encoded_result_payload(make_result(credential)),
            credentials=(credential,),
        )
    _assert_credential_absent_from_error(caught.value, credential)


def test_protocol_field_names_are_not_inferred_as_credentials() -> None:
    result = provider_process._decode_provider_result(
        _encoded_result_payload(Verdict(True, 1.0, "safe", value=b"safe")),
        credentials=("verdict", "evidence", "plain", "bytes", "v"),
    )
    assert result.evidence == "safe"
    assert result.value == b"safe"


def test_secret_is_rejected_before_malformed_dto_construction() -> None:
    credential = "malformed-dto-credential-e5"
    payload = provider_process._encode_json_bytes(
        {
            "schema": provider_process._SCHEMA_VERSION,
            "status": "ok",
            "value": {
                "$": "plain",
                "v": {
                    "$": "goal",
                    "id": credential,
                    "condition": "condition",
                    "say": "say",
                    "source": "camera",
                    "modality": "speak",
                    "state": {"$": "dict", "v": []},
                    "expiry": "until_cancelled",
                    "origin": "invalid-origin",
                },
            },
        },
        limit=provider_process._MAX_RESULT_BYTES,
    )
    with pytest.raises(
        ProviderProcessError,
        match="credential-bearing completion",
    ) as caught:
        provider_process._decode_provider_result(
            payload,
            credentials=(credential,),
        )
    _assert_credential_absent_from_error(caught.value, credential)


def test_builtin_credential_uses_typed_recipe_not_child_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "ambient-provider-credential",
    )
    settings = ModelSettings(vision_model="vendor/model")
    ambient = OpenRouterVision(settings)
    explicit = OpenRouterVision(
        settings,
        api_key="explicit-provider-credential",
    )

    ambient_spec = provider_process._adapter_spec(ambient)
    explicit_spec = provider_process._adapter_spec(explicit)
    assert ambient_spec["credential"] == "ambient-provider-credential"
    assert explicit_spec["credential"] == "explicit-provider-credential"

    monkeypatch.delenv("OPENROUTER_API_KEY")
    reconstructed = provider_process._construct_adapter(explicit_spec)
    assert type(reconstructed) is OpenRouterVision
    assert reconstructed.__dict__["_api_key"] == (
        "explicit-provider-credential"
    )
    assert "OPENROUTER_API_KEY" not in os.environ
    with pytest.raises(
        ProviderProcessError,
        match="requires an explicit reviewed credential",
    ):
        provider_process._adapter_spec(OpenRouterVision(settings))


def test_typed_media_and_provider_dtos_round_trip_without_codecs() -> None:
    goal = Goal(id="g", condition="condition", say="seen")

    async def drive() -> None:
        adapter = _DtoProvider()
        verdict = await run_provider_process(
            adapter,
            "judge",
            ("condition", ("image/jpeg", b"jpeg")),
            {
                "roi": Box(0.0, 0.0, 1.0, 1.0),
                "audio": AudioFrame(pcm=b"\x00\x00"),
                "context": {"strict": True},
            },
            watchdog_s=1.0,
        )
        assert verdict.satisfied is True
        assert verdict.events == [ClipEvent(0.25, 0.5, "motion")]
        assert verdict.subject is not None
        assert verdict.subject.key == "subject-1"
        assert verdict.value == {"raw": b"value"}

        observations = await run_provider_process(
            adapter,
            "detect",
            (gray_frame(2, 2, 0), [goal]),
            {},
            watchdog_s=1.0,
        )
        assert observations == [
            Observation("thing", Box(0.1, 0.2, 0.3, 0.4), 0.75)
        ]

        events = await run_provider_process(
            adapter,
            "detect_events",
            ([gray_frame(2, 2, 0)], [goal]),
            {},
            watchdog_s=1.0,
        )
        assert events == [
            Event("motion", 1.25, ["subject-1"], 0.8, "event-1")
        ]

    asyncio.run(drive())


def test_hostile_unregistered_adapter_has_zero_parent_effect(
    tmp_path: Path,
) -> None:
    effects = tmp_path / "hostile-adapter.txt"

    async def drive() -> None:
        with pytest.raises(
            ProviderProcessError,
            match="not explicitly registered",
        ):
            await run_provider_process(
                _HostileAdapter(str(effects)),
                "__call__",
                (),
                {},
                watchdog_s=1.0,
            )

    asyncio.run(drive())
    assert not effects.exists()


@pytest.mark.parametrize("location", ["frame", "context"])
def test_hostile_frame_or_context_has_zero_parent_effect(
    tmp_path: Path,
    location: str,
) -> None:
    effects = tmp_path / f"hostile-{location}.txt"
    adapter_effects = tmp_path / f"adapter-{location}.txt"
    hostile = _HostileValue(str(effects))
    args = ("condition", hostile if location == "frame" else "frame")
    kwargs = {} if location == "frame" else {"context": {"value": hostile}}

    async def drive() -> None:
        with pytest.raises(
            ProviderProcessError,
            match="outside the reviewed wire schema",
        ):
            await run_provider_process(
                _UnmarkedCustomVision(str(adapter_effects)),
                "judge",
                args,
                kwargs,
                watchdog_s=1.0,
            )

    asyncio.run(drive())
    assert not effects.exists()
    assert not adapter_effects.exists()


def test_string_subclass_reducer_is_rejected_without_parent_effect(
    tmp_path: Path,
) -> None:
    effects = tmp_path / "hostile-string.txt"
    adapter_effects = tmp_path / "string-adapter.txt"

    async def drive() -> None:
        with pytest.raises(
            ProviderProcessError,
            match="outside the reviewed wire schema",
        ):
            await run_provider_process(
                _UnmarkedCustomVision(str(adapter_effects)),
                "judge",
                (_HostileString("condition", str(effects)), "frame"),
                {},
                watchdog_s=1.0,
            )

    asyncio.run(drive())
    assert not effects.exists()
    assert not adapter_effects.exists()


def test_child_poisoned_result_never_executes_a_reducer(
    tmp_path: Path,
) -> None:
    effects = tmp_path / "poisoned-result.txt"

    async def drive() -> None:
        result = await run_provider_process(
            _PoisonedResultProvider(str(effects)),
            "__call__",
            (),
            {},
            watchdog_s=1.0,
        )
        assert result.exception_type == "unsupported completion"

    asyncio.run(drive())
    assert not effects.exists()


def test_malformed_completion_json_fails_closed() -> None:
    with pytest.raises(
        ProviderProcessError,
        match="malformed JSON",
    ):
        provider_process._decode_provider_result(b'{"schema":1')


def test_oversized_request_fails_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    effects = tmp_path / "oversized-adapter.txt"
    monkeypatch.setattr(provider_process, "_MAX_REQUEST_BYTES", 32)

    async def drive() -> None:
        with pytest.raises(
            ProviderProcessError,
            match="process boundary limit",
        ):
            await run_provider_process(
                _UnmarkedCustomVision(str(effects)),
                "judge",
                ("condition", b"x" * 64),
                {},
                watchdog_s=1.0,
            )

    asyncio.run(drive())
    assert not effects.exists()


@pytest.mark.parametrize("payload", [b"", b"\x00\x00", b"x"])
def test_truncated_completion_frame_fails_closed(payload: bytes) -> None:
    async def drive() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(payload)
        reader.feed_eof()
        with pytest.raises(
            ProviderProcessError,
            match="exited before returning",
        ):
            await provider_process._read_frame(
                reader,
                time.monotonic() + 1.0,
            )

    asyncio.run(drive())


def test_oversized_completion_frame_fails_before_body_read() -> None:
    async def drive() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(
            provider_process._FRAME_HEADER.pack(
                provider_process._MAX_RESULT_BYTES + 1
            )
        )
        reader.feed_eof()
        with pytest.raises(
            ProviderProcessError,
            match="oversized completion",
        ):
            await provider_process._read_frame(
                reader,
                time.monotonic() + 1.0,
            )

    asyncio.run(drive())
