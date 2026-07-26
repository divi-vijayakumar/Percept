"""Versioned, invocation-scoped BYOK boundary for hosted Percept deployments.

This module is intentionally separate from the environment-configured backend registry. A hosted
service receives one opaque credential lease from its caller and must not let process environment,
another tenant's client, or a prior rotation participate in that invocation. The boundary therefore:

* accepts an immutable, explicitly versioned request;
* validates an exact provider/model/capability tuple before constructing vendor objects;
* constructs a fresh backend and client for every :meth:`HostedByokAdapterFactory.judge` call;
* applies the request's cancellation/deadline around the complete provider call; and
* discards and best-effort closes all invocation-local provider objects afterward.

For composition with :class:`percept.Percept`, :meth:`HostedByokAdapterFactory.bind` returns a
request-bound ``Vision`` adapter. It contains no provider resources or result cache: every adapter
``judge`` call still creates a fresh client/backend through this boundary.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol, runtime_checkable

from .adapters import Verdict
from .config import Settings

HOSTED_BYOK_CONTRACT_VERSION = "1.0.0"
HOSTED_BYOK_CAPABILITIES: Mapping[str, frozenset[str]] = {
    "gemini": frozenset({"vision.judge", "vision.judge.fused"}),
    "openrouter": frozenset({"vision.judge"}),
}
_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_CANCELLATION_GRACE_S = 0.05
_DEFAULT_CLOSE_TIMEOUT_S = 0.25
_PROVIDER_FAILURE_CODE = "hosted_byok_provider_failure"
_INJECTED_REUSE_CODE = "hosted_byok_injected_component_reused"


class HostedByokError(RuntimeError):
    """Base class for fail-closed hosted-BYOK failures."""


class HostedByokContractError(HostedByokError, ValueError):
    """The request cannot be executed under this contract."""


class HostedByokUnsupportedError(HostedByokContractError):
    """The provider/model/capability tuple is not supported exactly as requested."""


class HostedByokDeadlineExceeded(HostedByokError, TimeoutError):
    """The invocation timeout or absolute deadline elapsed."""


class HostedByokCancelled(asyncio.CancelledError):
    """The request's invocation-scoped cancellation signal fired."""


class HostedByokInvocationError(HostedByokError):
    """A provider/client/backend failed; vendor details are withheld to protect credentials."""


@runtime_checkable
class HostedByokCancellationSignal(Protocol):
    """Minimal cancellation protocol; :class:`asyncio.Event` satisfies it."""

    def is_set(self) -> bool: ...

    async def wait(self) -> Any: ...


@dataclass(frozen=True, slots=True, repr=False)
class HostedByokRequest:
    """Immutable routing, tenancy, credential, and timing context for exactly one invocation.

    ``deadline_unix_s`` is an absolute UNIX timestamp. ``timeout_s`` is relative to entry into
    :meth:`HostedByokAdapterFactory.judge`; when both are present, the earlier bound wins.

    The credential is excluded from repr/equality/hash. The whole request is deliberately
    unhashable so it cannot accidentally become a cache key. Callers must issue a new
    ``credential_lease_id`` whenever the opaque credential rotates.
    """

    provider: str
    model: str
    credential: str = field(repr=False, compare=False, hash=False)
    credential_lease_id: str
    principal_id: str
    capability: str = "vision.judge"
    timeout_s: float | None = None
    deadline_unix_s: float | None = None
    cancellation: HostedByokCancellationSignal | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    contract_version: str = HOSTED_BYOK_CONTRACT_VERSION

    __hash__ = None

    def __post_init__(self) -> None:
        for name in ("provider", "model", "credential_lease_id", "principal_id", "capability"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise HostedByokContractError(
                    f"{name} must be a non-empty, whitespace-trimmed string."
                )
        if not isinstance(self.credential, str) or not self.credential:
            raise HostedByokContractError("credential must be a non-empty opaque string.")
        if self.contract_version != HOSTED_BYOK_CONTRACT_VERSION:
            raise HostedByokContractError(
                f"unsupported hosted BYOK contract version {self.contract_version!r}; "
                f"this SDK implements {HOSTED_BYOK_CONTRACT_VERSION!r}."
            )
        for name in ("timeout_s", "deadline_unix_s"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise HostedByokContractError(f"{name} must be a finite positive number when set.")
        if self.cancellation is not None and not isinstance(
            self.cancellation, HostedByokCancellationSignal
        ):
            raise HostedByokContractError(
                "cancellation must implement is_set() and async wait() (asyncio.Event is supported)."
            )

    def __repr__(self) -> str:
        return (
            "HostedByokRequest("
            f"provider={self.provider!r}, model={self.model!r}, credential=<redacted>, "
            f"credential_lease_id={self.credential_lease_id!r}, principal_id={self.principal_id!r}, "
            f"capability={self.capability!r}, timeout_s={self.timeout_s!r}, "
            f"deadline_unix_s={self.deadline_unix_s!r}, "
            f"contract_version={self.contract_version!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class HostedByokGeminiClient:
    """Injected Gemini client plus its ``google.genai.types``-compatible transport namespace."""

    client: Any = field(repr=False)
    transport_types: Any = field(repr=False)

    def __repr__(self) -> str:
        return "HostedByokGeminiClient(<redacted>)"


HostedByokBackendFactory = Callable[[HostedByokRequest, float], Any]
HostedByokGeminiClientFactory = Callable[[HostedByokRequest, float], HostedByokGeminiClient]
HostedByokOpenRouterClientFactory = Callable[[HostedByokRequest, float], Any]


@dataclass(slots=True)
class _Invocation:
    backend: Any
    client: Any = None


class HostedByokAdapterFactory:
    """Execute hosted BYOK vision judgments without environment or cross-tenant fallback.

    The optional factories are the reviewed fake/client/backend test seams. Every callback is invoked
    once per judgment and **must return a fresh object**. ``backend_factories`` may contain only the two
    supported provider names; it is primarily useful for provider-independent conformance fixtures.
    Client factories exercise the real Percept provider adapters while replacing only vendor transport.
    """

    def __init__(
        self,
        *,
        backend_factories: Mapping[str, HostedByokBackendFactory] | None = None,
        gemini_client_factory: HostedByokGeminiClientFactory | None = None,
        openrouter_client_factory: HostedByokOpenRouterClientFactory | None = None,
        time_fn: Callable[[], float] = time.time,
        monotonic_fn: Callable[[], float] = time.monotonic,
        cancellation_grace_s: float = _DEFAULT_CANCELLATION_GRACE_S,
        close_timeout_s: float = _DEFAULT_CLOSE_TIMEOUT_S,
    ) -> None:
        overrides = dict(backend_factories or {})
        unknown = sorted(set(overrides) - set(HOSTED_BYOK_CAPABILITIES))
        if unknown:
            raise HostedByokContractError(
                f"unsupported hosted BYOK backend factory provider(s): {unknown!r}."
            )
        self._backend_factories = overrides
        self._gemini_client_factory = gemini_client_factory
        self._openrouter_client_factory = openrouter_client_factory
        self._time_fn = time_fn
        self._monotonic_fn = monotonic_fn
        self._cancellation_grace_s = _positive_duration(
            "cancellation_grace_s", cancellation_grace_s
        )
        self._close_timeout_s = _positive_duration("close_timeout_s", close_timeout_s)
        # Weak identity claims enforce the fresh-object contract without retaining clients/backends
        # (and therefore without turning this factory into a resource cache).
        self._injected_claims: dict[int, weakref.ReferenceType[Any]] = {}

    def bind(self, request: HostedByokRequest) -> "HostedByokVision":
        """Return a request-bound public Vision adapter accepted by ``Percept.create(vision=...)``.

        The bound object stores no provider client/backend/result. Each ``judge`` call still enters
        this factory and creates a fresh invocation-local transport.
        """

        self._validate_route(request)
        return HostedByokVision(self, request)

    async def judge(
        self,
        request: HostedByokRequest,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        """Perform one isolated vision judgment and return the normal public :class:`Verdict`."""

        invocation: _Invocation | None = None
        provider_task: asyncio.Task | None = None
        cancellation_task: asyncio.Task | None = None
        try:
            timeout_s = self._preflight(request, frame=frame, audio=audio)
            expires_at = float(self._monotonic_fn()) + timeout_s
            invocation = self._build_invocation(request, timeout_s)
            remaining_s = expires_at - float(self._monotonic_fn())
            if remaining_s <= 0:
                raise HostedByokDeadlineExceeded(
                    "hosted BYOK invocation deadline elapsed during provider construction."
                )
            if request.cancellation is not None and request.cancellation.is_set():
                raise HostedByokCancelled()
            judge = getattr(invocation.backend, "judge", None)
            if not callable(judge):
                raise HostedByokUnsupportedError(
                    f"provider {request.provider!r} has no vision.judge capability."
                )
            # The backend's descriptor lookup is injectable code too. Re-check at the last possible
            # point before task creation so a construction/lookup race can spend zero provider calls.
            if request.cancellation is not None and request.cancellation.is_set():
                raise HostedByokCancelled()
            provider_task = asyncio.create_task(
                judge(
                    condition,
                    frame,
                    roi=roi,
                    exemplar=exemplar,
                    corrections=corrections,
                    frames=frames,
                    audio=audio,
                    context=context,
                )
            )
            wait_for: set[asyncio.Task] = {provider_task}
            if request.cancellation is not None:
                cancellation_task = asyncio.create_task(request.cancellation.wait())
                wait_for.add(cancellation_task)
            done, _ = await asyncio.wait(
                wait_for, timeout=remaining_s, return_when=asyncio.FIRST_COMPLETED
            )
            if request.cancellation is not None and request.cancellation.is_set():
                await self._stop_task(provider_task, self._cancellation_grace_s)
                raise HostedByokCancelled()
            if cancellation_task is not None and cancellation_task in done:
                try:
                    await cancellation_task
                except Exception:
                    await self._stop_task(provider_task, self._cancellation_grace_s)
                    raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            if provider_task not in done:
                await self._stop_task(provider_task, self._cancellation_grace_s)
                raise HostedByokDeadlineExceeded(
                    "hosted BYOK invocation exceeded its effective deadline."
                )
            try:
                result = await provider_task
            except Exception:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            if not isinstance(result, Verdict):
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            return result
        except (HostedByokError, HostedByokCancelled):
            raise
        except asyncio.CancelledError:
            if provider_task is not None:
                await self._stop_task(provider_task, self._cancellation_grace_s)
            raise
        except Exception:
            # Never forward a vendor exception: an SDK/client may echo headers, constructor args, or
            # even an attacker-controlled exception class name. A fixed code is the whole surface.
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
        finally:
            if cancellation_task is not None:
                await self._stop_task(cancellation_task, self._cancellation_grace_s)
            if invocation is not None:
                await self._close_invocation(invocation, timeout_s=self._close_timeout_s)

    def _preflight(self, request: HostedByokRequest, *, frame: Any, audio: Any) -> float:
        self._validate_route(request)
        if request.capability == "vision.judge.fused" and (frame is None or audio is None):
            raise HostedByokUnsupportedError(
                "vision.judge.fused requires both image and audio evidence."
            )
        if request.provider == "openrouter" and audio is not None:
            raise HostedByokUnsupportedError(
                "OpenRouter hosted BYOK does not support fused audio input."
            )
        if request.provider == "openrouter" and frame is None:
            raise HostedByokUnsupportedError(
                "OpenRouter hosted BYOK vision.judge requires image evidence."
            )
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()

        bounds = [_DEFAULT_TIMEOUT_S]
        if request.timeout_s is not None:
            bounds.append(float(request.timeout_s))
        if request.deadline_unix_s is not None:
            bounds.append(float(request.deadline_unix_s) - float(self._time_fn()))
        timeout_s = min(bounds)
        if timeout_s <= 0:
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation deadline elapsed before provider construction."
            )
        return timeout_s

    @staticmethod
    def _validate_route(request: HostedByokRequest) -> None:
        if not isinstance(request, HostedByokRequest):
            raise HostedByokContractError(
                "request must be an immutable HostedByokRequest; implicit configuration is unsupported."
            )
        capabilities = HOSTED_BYOK_CAPABILITIES.get(request.provider)
        if capabilities is None:
            raise HostedByokUnsupportedError(
                f"unsupported hosted BYOK provider {request.provider!r}; "
                f"supported providers are {sorted(HOSTED_BYOK_CAPABILITIES)!r}."
            )
        if request.capability not in capabilities:
            raise HostedByokUnsupportedError(
                f"provider {request.provider!r} does not support capability "
                f"{request.capability!r} in hosted BYOK contract "
                f"{HOSTED_BYOK_CONTRACT_VERSION}."
            )
        if request.provider == "gemini":
            if not request.model.startswith("gemini-"):
                raise HostedByokUnsupportedError(
                    f"model {request.model!r} is not an exact Gemini model id."
                )
        elif (
            request.model.count("/") != 1
            or request.model.startswith("/")
            or request.model.endswith("/")
            or any(ch.isspace() for ch in request.model)
        ):
            raise HostedByokUnsupportedError(
                f"model {request.model!r} is not an exact OpenRouter 'vendor/model' id."
            )

    def _build_invocation(self, request: HostedByokRequest, timeout_s: float) -> _Invocation:
        override = self._backend_factories.get(request.provider)
        if override is not None:
            try:
                backend = override(request, timeout_s)
            except Exception:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            self._claim_injected(backend)
            return _Invocation(backend=backend)
        if request.provider == "gemini":
            return self._build_gemini(request, timeout_s)
        if request.provider == "openrouter":
            return self._build_openrouter(request, timeout_s)
        raise HostedByokUnsupportedError(f"unsupported hosted BYOK provider {request.provider!r}.")

    def _build_gemini(self, request: HostedByokRequest, timeout_s: float) -> _Invocation:
        from .backends.gemini import GeminiVision, _require_genai

        settings = _settings_for_model(request.model)
        if self._gemini_client_factory is not None:
            try:
                binding = self._gemini_client_factory(request, timeout_s)
            except Exception:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            if not isinstance(binding, HostedByokGeminiClient):
                raise HostedByokContractError(
                    "gemini_client_factory must return HostedByokGeminiClient."
                )
            self._claim_injected(binding.client)
        else:
            genai, types = _require_genai()
            client = genai.Client(
                api_key=request.credential,
                http_options=types.HttpOptions(timeout=max(1, int(timeout_s * 1000))),
            )
            binding = HostedByokGeminiClient(client=client, transport_types=types)
        backend = GeminiVision(
            settings,
            client=binding.client,
            transport_types=binding.transport_types,
            strip_prompt="change",
        )
        if backend._model != request.model:
            raise HostedByokUnsupportedError(
                f"Gemini refused exact model {request.model!r}; no fallback is permitted."
            )
        return _Invocation(backend=backend, client=binding.client)

    def _build_openrouter(self, request: HostedByokRequest, timeout_s: float) -> _Invocation:
        from .backends.openrouter import (
            OpenRouterVision,
            _OPENROUTER_BASE_URL,
            _require_openai,
        )

        settings = _settings_for_model(request.model)
        if self._openrouter_client_factory is not None:
            try:
                client = self._openrouter_client_factory(request, timeout_s)
            except Exception:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            self._claim_injected(client)
        else:
            AsyncOpenAI = _require_openai()
            client = AsyncOpenAI(
                base_url=_OPENROUTER_BASE_URL,
                api_key=request.credential,
                timeout=timeout_s,
                default_headers={
                    "HTTP-Referer": "https://github.com/divi-vijayakumar/Percept",
                    "X-Title": "Percept",
                },
            )
        backend = OpenRouterVision(settings, client=client, timeout_s=timeout_s)
        if backend._model != request.model:
            raise HostedByokUnsupportedError(
                f"OpenRouter refused exact model {request.model!r}; no fallback is permitted."
            )
        return _Invocation(backend=backend, client=client)

    def _claim_injected(self, component: Any) -> None:
        """Reject callback-returned object reuse without retaining the object."""

        component_id = id(component)
        prior = self._injected_claims.get(component_id)
        if prior is not None and prior() is component:
            raise HostedByokContractError(_INJECTED_REUSE_CODE)
        try:
            self_ref = weakref.ref(self)

            def release(claimed: weakref.ReferenceType[Any], *, key: int = component_id) -> None:
                owner = self_ref()
                if owner is not None and owner._injected_claims.get(key) is claimed:
                    owner._injected_claims.pop(key, None)

            claimed = weakref.ref(component, release)
        except TypeError:
            raise HostedByokContractError(
                "injected hosted BYOK components must support weak references."
            ) from None
        self._injected_claims[component_id] = claimed

    @staticmethod
    async def _stop_task(task: asyncio.Task, timeout_s: float) -> None:
        if not task.done():
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        if task not in done:
            return
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    @staticmethod
    async def _close_invocation(invocation: _Invocation, *, timeout_s: float) -> None:
        seen: set[int] = set()
        for component in (invocation.backend, invocation.client):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            try:
                closer = getattr(component, "aclose", None) or getattr(component, "close", None)
            except Exception:
                continue
            if not callable(closer):
                continue

            async def close_one() -> None:
                result = await asyncio.to_thread(closer)
                if inspect.isawaitable(result):
                    await result

            close_task = asyncio.create_task(close_one())
            done, _ = await asyncio.wait({close_task}, timeout=timeout_s)
            if close_task not in done:
                close_task.cancel()
                continue
            try:
                close_task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                # Teardown must not hide the judgment result/failure. Objects are discarded regardless.
                pass


@dataclass(frozen=True, slots=True, repr=False)
class HostedByokVision:
    """Request-bound Vision adapter for direct use with :class:`percept.Percept`.

    The adapter holds only the immutable request and factory. ``detect``/``detect_events`` honestly
    return no proposals because contract 1.0 exposes semantic judgment only; every ``judge`` delegates
    to a fresh invocation through the factory.
    """

    _factory: HostedByokAdapterFactory = field(repr=False)
    _request: HostedByokRequest = field(repr=False)
    CONTRACT_VERSION: ClassVar[str] = HOSTED_BYOK_CONTRACT_VERSION

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        return await self._factory.judge(
            self._request,
            condition,
            frame,
            roi=roi,
            exemplar=exemplar,
            corrections=corrections,
            frames=frames,
            audio=audio,
            context=context,
        )

    async def detect(self, frame: Any, goals: list[Any]) -> list:
        return []

    async def detect_events(self, frames: list, goals: list[Any]) -> list:
        return []

    def __repr__(self) -> str:
        return (
            "HostedByokVision("
            f"provider={self._request.provider!r}, model={self._request.model!r}, "
            f"credential_lease_id={self._request.credential_lease_id!r}, "
            f"principal_id={self._request.principal_id!r})"
        )


def _settings_for_model(model: str) -> Settings:
    """Construct explicit defaults without calling the environment-backed ``load_settings``."""

    defaults = Settings()
    return replace(defaults, models=replace(defaults.models, vision_model=model))


def _positive_duration(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise HostedByokContractError(f"{name} must be a finite positive number.")
    return float(value)


__all__ = [
    "HOSTED_BYOK_CONTRACT_VERSION",
    "HOSTED_BYOK_CAPABILITIES",
    "HostedByokRequest",
    "HostedByokCancellationSignal",
    "HostedByokGeminiClient",
    "HostedByokAdapterFactory",
    "HostedByokVision",
    "HostedByokBackendFactory",
    "HostedByokGeminiClientFactory",
    "HostedByokOpenRouterClientFactory",
    "HostedByokError",
    "HostedByokContractError",
    "HostedByokUnsupportedError",
    "HostedByokDeadlineExceeded",
    "HostedByokCancelled",
    "HostedByokInvocationError",
]
