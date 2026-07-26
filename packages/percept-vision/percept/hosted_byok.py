"""Versioned, invocation-scoped BYOK boundary for hosted Percept deployments.

This module is intentionally separate from the environment-configured backend registry. A hosted
service receives one opaque credential lease from its caller and must not let process environment,
another tenant's client, or a prior rotation participate in that invocation. The boundary therefore:

* accepts an immutable, explicitly versioned request;
* validates an exact provider/model/capability tuple before constructing vendor objects;
* constructs a fresh backend and client for every :meth:`HostedByokAdapterFactory.judge` call;
* applies the request's cancellation/deadline to result acceptance and cancellation initiation;
* does not return until provider work and invocation-local teardown have actually completed; and
* discards every invocation-local provider object afterward.

For composition with :class:`percept.Percept`, :meth:`HostedByokAdapterFactory.bind` returns a
request-bound ``Vision`` adapter. It contains no provider resources or result cache: every adapter
``judge`` call still creates a fresh client/backend through this boundary.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import ctypes
import errno
import hmac
import inspect
import json
import math
import os
import platform
import secrets
import signal
import site
import socket
import stat
import subprocess
import sys
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from multiprocessing.connection import Connection
from types import FunctionType, MappingProxyType, MethodType
from typing import Any, ClassVar, Protocol, runtime_checkable

from .adapters import AudioFrame, Verdict
from .config import Settings
from .contracts import Box, ClipEvent, SubjectRef

HOSTED_BYOK_CONTRACT_VERSION = "1.0.0"
HOSTED_BYOK_PROCESS_ISOLATION_VERSION = "2.1.0"
HOSTED_BYOK_CAPABILITIES: Mapping[str, frozenset[str]] = {
    "gemini": frozenset({"vision.judge", "vision.judge.fused"}),
    "openrouter": frozenset({"vision.judge"}),
}
_DEFAULT_TIMEOUT_S = 120.0
_PROVIDER_FAILURE_CODE = "hosted_byok_provider_failure"
_INJECTED_REUSE_CODE = "hosted_byok_injected_component_reused"
_MAX_PROCESS_IPC_BYTES = 32 * 1024 * 1024
_BOUNDARY_PROFILE = "(version 1)(allow default)(deny process-fork)"
_CHILD_BOOTSTRAP = (
    "import json,sys;"
    "sys.path[:0]=json.loads(sys.argv[1]);"
    "from percept.hosted_byok import _hosted_byok_clean_child_main;"
    "_hosted_byok_clean_child_main(int(sys.argv[2]),sys.argv[3],sys.argv[4])"
)
_PROVIDER_ENV_KEYS = frozenset(
    {
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
)
_RESERVED_CONTEXT_KEYS = frozenset(
    {
        "process_qa",
        "observe_step",
        "running_answer",
        "claim_mode",
        "extractor",
        "judge_static_context",
    }
)


_ACTIVE_INVOCATION_SCOPE: contextvars.ContextVar["_InvocationTaskScope | None"] = (
    contextvars.ContextVar("hosted_byok_active_invocation_scope", default=None)
)
_TASK_FACTORY_INSTALLATIONS: weakref.WeakKeyDictionary[Any, "_TaskFactoryInstallation"] = (
    weakref.WeakKeyDictionary()
)


class _TaskFactoryInstallation:
    """One shared dispatcher per loop so concurrent hosted calls remain independently scoped."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.previous_factory = loop.get_task_factory()
        self.active_scopes = 0

        def factory(
            factory_loop: asyncio.AbstractEventLoop,
            coroutine: Any,
            context: contextvars.Context | None = None,
        ) -> asyncio.Task[Any]:
            if context is None:
                task = asyncio.tasks.Task(coroutine, loop=factory_loop)
                scope = _ACTIVE_INVOCATION_SCOPE.get()
            else:
                task = asyncio.tasks.Task(coroutine, loop=factory_loop, context=context)
                scope = context.get(_ACTIVE_INVOCATION_SCOPE, None)
            if scope is not None:
                scope.tasks.add(task)
            return task

        self.factory = factory


class _InvocationTaskScope:
    """Own tasks spawned by a provider/client while this invocation is active.

    A vendor coroutine can create its own worker task and then return while that worker is still
    able to send a request.  Task cancellation alone cannot discover that worker.  The event-loop
    task-factory hook records every task inheriting this invocation's context, and teardown waits
    for *all* of them (including workers spawned while another worker handles cancellation).
    """

    def __init__(self) -> None:
        self.tasks: set[asyncio.Future[Any]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._installation: _TaskFactoryInstallation | None = None
        self._token: contextvars.Token[_InvocationTaskScope | None] | None = None

    def enter(self) -> None:
        loop = asyncio.get_running_loop()
        installation = _TASK_FACTORY_INSTALLATIONS.get(loop)
        if installation is None and loop.get_task_factory() is not None:
            # A hosting application task factory may have security/accounting semantics we cannot
            # safely replace.  Failing before construction is preferable to a false containment
            # guarantee for credential-bearing work.
            raise HostedByokContractError(
                "hosted BYOK requires an unmodified event-loop task factory for task containment."
            )
        if installation is None:
            installation = _TaskFactoryInstallation(loop)
            loop.set_task_factory(installation.factory)
            _TASK_FACTORY_INSTALLATIONS[loop] = installation
        elif loop.get_task_factory() is not installation.factory:
            raise HostedByokContractError(
                "hosted BYOK task containment was replaced while an invocation was active."
            )
        installation.active_scopes += 1
        self._loop = loop
        self._installation = installation
        self._token = _ACTIVE_INVOCATION_SCOPE.set(self)

    async def settle(self) -> bool:
        """Cancel and join every descendant task, including cancellation-suppressing workers."""

        interrupted = False
        while True:
            owner = asyncio.current_task()
            pending = [task for task in tuple(self.tasks) if task is not owner and not task.done()]
            if not pending:
                return interrupted
            for task in pending:
                task.cancel()
            for task in pending:
                interrupted |= await _settle_owned_task(task, cancel=False)

    def exit(self) -> None:
        if self._loop is not None and self._installation is not None:
            self._installation.active_scopes -= 1
            if self._installation.active_scopes == 0:
                self._loop.set_task_factory(self._installation.previous_factory)
                _TASK_FACTORY_INSTALLATIONS.pop(self._loop, None)
        if self._token is not None:
            _ACTIVE_INVOCATION_SCOPE.reset(self._token)
        self._loop = None
        self._installation = None
        self._token = None


class HostedByokError(RuntimeError):
    """Base class for fail-closed hosted-BYOK failures."""


class HostedByokContractError(HostedByokError, ValueError):
    """The request cannot be executed under this contract."""


class HostedByokUnsupportedError(HostedByokContractError):
    """The provider/model/capability tuple is not supported exactly as requested."""


class HostedByokDeadlineExceeded(HostedByokError, TimeoutError):
    """The invocation timeout or absolute deadline elapsed."""


class HostedByokCancelled(HostedByokError):
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
        for name in (
            "provider",
            "model",
            "credential_lease_id",
            "principal_id",
            "capability",
            "contract_version",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise HostedByokContractError(
                    f"{name} must be a non-empty, whitespace-trimmed string."
                )
        if type(self.credential) is not str or not self.credential:
            raise HostedByokContractError("credential must be a non-empty opaque string.")
        visible_fields = (
            self.provider,
            self.model,
            self.credential_lease_id,
            self.principal_id,
            self.capability,
            self.contract_version,
        )
        if any(self.credential in value for value in visible_fields):
            raise HostedByokContractError(
                "hosted BYOK routing and identity fields must not contain the credential."
            )
        if self.contract_version != HOSTED_BYOK_CONTRACT_VERSION:
            raise HostedByokContractError(
                f"unsupported hosted BYOK contract version {self.contract_version!r}; "
                f"this SDK implements {HOSTED_BYOK_CONTRACT_VERSION!r}."
            )
        for name in ("timeout_s", "deadline_unix_s"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise HostedByokContractError(f"{name} must be a finite positive number when set.")
        if self.cancellation is not None and type(self.cancellation) is not asyncio.Event:
            raise HostedByokContractError(
                "cancellation must be an exact asyncio.Event."
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


@runtime_checkable
class HostedByokProcessSupervisor(Protocol):
    """Typing protocol implemented by the exact SDK subprocess supervisor."""

    async def execute(self, *args: Any, **kwargs: Any) -> Verdict: ...


@dataclass(frozen=True, slots=True)
class HostedByokTransportSafety:
    """Machine-readable execution classification for one exact request route."""

    provider: str
    transport_class: str
    asyncio_task_contained: bool
    requires_process_supervisor: bool
    process_isolation_version: str = HOSTED_BYOK_PROCESS_ISOLATION_VERSION
    production_ready: bool = False


def _wire_value(value: Any) -> dict[str, Any]:
    """Encode exact approved carriers into a non-executable tagged JSON value."""

    value_type = type(value)
    if value is None:
        return {"t": "none", "v": None}
    if value_type is bool:
        return {"t": "bool", "v": value}
    if value_type is int:
        return {"t": "int", "v": value}
    if value_type is float and math.isfinite(value):
        return {"t": "float", "v": value}
    if value_type is str:
        return {"t": "str", "v": value}
    if value_type in (bytes, bytearray, memoryview):
        return {
            "t": {bytes: "bytes", bytearray: "bytearray", memoryview: "memoryview"}[value_type],
            "v": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if value_type in (list, tuple, set, frozenset):
        encoded = [_wire_value(item) for item in value]
        if value_type in (set, frozenset):
            encoded.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"t": value_type.__name__, "v": encoded}
    if value_type is dict:
        pairs = [[_wire_value(key), _wire_value(item)] for key, item in value.items()]
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return {"t": "dict", "v": pairs}
    dto_types = (HostedByokRequest, AudioFrame, Box, ClipEvent, SubjectRef, Verdict)
    if value_type in dto_types:
        values = {
            item.name: _wire_value(getattr(value, item.name))
            for item in dataclass_fields(value_type)
            if not (value_type is HostedByokRequest and item.name == "cancellation")
        }
        return {"t": f"dto:{value_type.__name__}", "v": values}
    raise HostedByokContractError("hosted BYOK IPC contains an unsupported carrier.")


def _wire_unvalue(value: Any) -> Any:
    if type(value) is not dict or set(value) != {"t", "v"} or type(value["t"]) is not str:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    tag, payload = value["t"], value["v"]
    if tag == "none" and payload is None:
        return None
    if tag == "bool" and type(payload) is bool:
        return payload
    if tag == "int" and type(payload) is int:
        return payload
    if tag == "float" and type(payload) is float and math.isfinite(payload):
        return payload
    if tag == "str" and type(payload) is str:
        return payload
    if tag in ("bytes", "bytearray", "memoryview") and type(payload) is str:
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
        return {"bytes": raw, "bytearray": bytearray(raw), "memoryview": memoryview(raw)}[tag]
    if tag in ("list", "tuple", "set", "frozenset") and type(payload) is list:
        decoded = [_wire_unvalue(item) for item in payload]
        if tag == "list":
            return decoded
        if tag == "tuple":
            return tuple(decoded)
        try:
            unique = set(decoded)
        except TypeError:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
        if len(unique) != len(decoded):
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        return unique if tag == "set" else frozenset(unique)
    if tag == "dict" and type(payload) is list:
        decoded_dict = {}
        for pair in payload:
            if type(pair) is not list or len(pair) != 2:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            key, item = _wire_unvalue(pair[0]), _wire_unvalue(pair[1])
            try:
                if key in decoded_dict:
                    raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
                decoded_dict[key] = item
            except TypeError:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
        return decoded_dict
    dto_map = {
        "dto:HostedByokRequest": HostedByokRequest,
        "dto:AudioFrame": AudioFrame,
        "dto:Box": Box,
        "dto:ClipEvent": ClipEvent,
        "dto:SubjectRef": SubjectRef,
        "dto:Verdict": Verdict,
    }
    dto_type = dto_map.get(tag)
    if dto_type is not None and type(payload) is dict:
        allowed = {item.name for item in dataclass_fields(dto_type)}
        if dto_type is HostedByokRequest:
            allowed.remove("cancellation")
        if set(payload) != allowed:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        kwargs = {name: _wire_unvalue(item) for name, item in payload.items()}
        try:
            return dto_type(**kwargs)
        except Exception:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        result[key] = value
    return result


def _wire_dumps(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            _wire_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except HostedByokError:
        raise
    except Exception:
        raise HostedByokContractError("hosted BYOK IPC encoding failed.") from None
    if len(encoded) > _MAX_PROCESS_IPC_BYTES:
        raise HostedByokContractError("hosted BYOK process payload exceeds the IPC limit.")
    return encoded


def _wire_loads(encoded: bytes) -> Any:
    if type(encoded) is not bytes or len(encoded) > _MAX_PROCESS_IPC_BYTES:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    try:
        parsed = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except Exception:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    decoded = _wire_unvalue(parsed)
    try:
        canonical = _wire_dumps(decoded)
    except HostedByokError:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    if canonical != encoded:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    return decoded


class HostedByokSubprocessSupervisor:
    """Exact, immutable authenticated no-descendant supervisor for production Hosted BYOK calls."""

    __slots__ = ("_poll_interval_s", "_boundary_kind", "_execute_impl", "_locked")

    def __init__(self, *, poll_interval_s: float = 0.002) -> None:
        if (
            type(poll_interval_s) not in (int, float)
            or not math.isfinite(float(poll_interval_s))
            or float(poll_interval_s) <= 0
        ):
            raise HostedByokContractError("poll_interval_s must be a finite positive number.")
        boundary_kind, _ = _select_containment_capability()
        object.__setattr__(self, "_poll_interval_s", float(poll_interval_s))
        object.__setattr__(self, "_boundary_kind", boundary_kind)
        implementation = HostedByokSubprocessSupervisor.__dict__["execute"]
        frozen = FunctionType(
            implementation.__code__,
            dict(implementation.__globals__),
            name=implementation.__name__,
            argdefs=implementation.__defaults__,
            closure=implementation.__closure__,
        )
        frozen.__kwdefaults__ = implementation.__kwdefaults__
        object.__setattr__(self, "_execute_impl", MethodType(frozen, self))
        object.__setattr__(self, "_locked", True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("HostedByokSubprocessSupervisor is sealed.")

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("HostedByokSubprocessSupervisor is immutable.")
        object.__setattr__(self, name, value)

    async def execute(
        self,
        request: HostedByokRequest,
        condition: str,
        frame: Any,
        *,
        deadline_monotonic: float,
        roi: Any,
        exemplar: Any,
        corrections: dict | None,
        frames: list | None,
        audio: Any,
        context: dict | None,
    ) -> Verdict:
        if type(self) is not HostedByokSubprocessSupervisor:
            raise HostedByokContractError("process supervisor must be the exact SDK implementation.")
        deadline_monotonic = _positive_duration("deadline_monotonic", deadline_monotonic)
        _validate_hosted_call_inputs(
            request,
            condition=condition,
            frame=frame,
            roi=roi,
            exemplar=exemplar,
            corrections=corrections,
            frames=frames,
            audio=audio,
            context=context,
        )
        # Deliver an already-queued caller cancellation before any credential-bearing child exists.
        await asyncio.sleep(0)
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()
        if time.monotonic() >= deadline_monotonic:
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation exceeded its effective deadline."
            )
        worker_values = (
            request,
            condition,
            frame,
            roi,
            exemplar,
            corrections,
            frames,
            audio,
            context,
            deadline_monotonic,
        )
        worker_payload = _wire_dumps(worker_values)
        await asyncio.sleep(0)
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()
        if time.monotonic() >= deadline_monotonic:
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation exceeded its effective deadline."
            )
        process, connection, challenge = _launch_contained_worker(self._boundary_kind)
        outcome: Any = None
        try:
            ready = None
            while True:
                if request.cancellation is not None and request.cancellation.is_set():
                    raise HostedByokCancelled()
                if connection.poll():
                    try:
                        ready = _wire_loads(
                            connection.recv_bytes(maxlength=_MAX_PROCESS_IPC_BYTES + 1)
                        )
                    except (EOFError, OSError):
                        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
                    break
                if process.poll() is not None:
                    raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
                if time.monotonic() >= deadline_monotonic:
                    raise HostedByokDeadlineExceeded(
                        "hosted BYOK invocation exceeded its effective deadline."
                    )
                await asyncio.sleep(self._poll_interval_s)
            if (
                type(ready) is not tuple
                or len(ready) != 4
                or ready[0] != "ready"
                or ready[1] != HOSTED_BYOK_PROCESS_ISOLATION_VERSION
                or type(ready[2]) is not bytes
                or not hmac.compare_digest(ready[2], challenge)
                or ready[3] != self._boundary_kind
            ):
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            await asyncio.sleep(0)
            if request.cancellation is not None and request.cancellation.is_set():
                raise HostedByokCancelled()
            if time.monotonic() >= deadline_monotonic:
                raise HostedByokDeadlineExceeded(
                    "hosted BYOK invocation exceeded its effective deadline."
                )
            try:
                connection.send_bytes(worker_payload)
            except (OSError, ValueError):
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            while True:
                if request.cancellation is not None and request.cancellation.is_set():
                    raise HostedByokCancelled()
                if connection.poll():
                    try:
                        outcome = _wire_loads(
                            connection.recv_bytes(maxlength=_MAX_PROCESS_IPC_BYTES + 1)
                        )
                    except (EOFError, OSError):
                        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
                    break
                if process.poll() is not None:
                    raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
                if time.monotonic() >= deadline_monotonic:
                    raise HostedByokDeadlineExceeded(
                        "hosted BYOK invocation exceeded its effective deadline."
                    )
                await asyncio.sleep(self._poll_interval_s)
        finally:
            connection.close()
            _terminate_and_wait_contained_worker(process)
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()
        if time.monotonic() >= deadline_monotonic:
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation exceeded its effective deadline."
            )
        if type(outcome) is not tuple or len(outcome) != 2:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        kind, payload = outcome
        if type(kind) is not str:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        if kind == "ok" and type(payload) is Verdict:
            if not _verdict_is_credential_free(payload, request.credential):
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            return payload
        if type(payload) is not str or request.credential in payload:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        if kind == "deadline":
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation exceeded its effective deadline."
            )
        if kind == "cancelled" and not payload:
            raise HostedByokCancelled()
        if kind == "unsupported":
            raise HostedByokUnsupportedError(payload)
        if kind == "contract":
            raise HostedByokContractError(payload)
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)


@dataclass(slots=True)
class _Invocation:
    backend: Any = None
    client: Any = None


async def _settle_owned_task(task: asyncio.Future[Any], *, cancel: bool) -> bool:
    """Optionally cancel one owned task and wait for its actual terminal state."""

    interrupted = False
    if cancel and not task.done():
        task.cancel()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                interrupted = True
            continue
        except Exception:
            break
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
    return interrupted


def _classify_transport_safety(
    factory: "HostedByokAdapterFactory", request: HostedByokRequest
) -> HostedByokTransportSafety:
    if type(factory) is not HostedByokAdapterFactory:
        raise HostedByokContractError("factory must be the exact SDK HostedByokAdapterFactory.")
    HostedByokAdapterFactory._validate_route(request)
    injected = (
        request.provider in factory._backend_factories
        or (request.provider == "gemini" and factory._gemini_client_factory is not None)
        or (request.provider == "openrouter" and factory._openrouter_client_factory is not None)
    )
    if injected:
        return HostedByokTransportSafety(
            provider=request.provider,
            transport_class="test_only_in_process",
            asyncio_task_contained=True,
            requires_process_supervisor=False,
        )
    return HostedByokTransportSafety(
        provider=request.provider,
        transport_class="production_process_isolated",
        asyncio_task_contained=False,
        requires_process_supervisor=True,
        production_ready=type(factory._process_supervisor) is HostedByokSubprocessSupervisor,
    )


class HostedByokAdapterFactory:
    """Execute hosted BYOK vision judgments without environment or cross-tenant fallback.

    The optional factories are the reviewed fake/client/backend test seams. Every callback is invoked
    once per judgment and **must return a fresh object**. ``backend_factories`` may contain only the two
    supported provider names; it is primarily useful for provider-independent conformance fixtures.
    Client factories exercise the real Percept provider adapters while replacing only vendor transport.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("HostedByokAdapterFactory is sealed.")

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("HostedByokAdapterFactory configuration is immutable.")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        backend_factories: Mapping[str, HostedByokBackendFactory] | None = None,
        gemini_client_factory: HostedByokGeminiClientFactory | None = None,
        openrouter_client_factory: HostedByokOpenRouterClientFactory | None = None,
        time_fn: Callable[[], float] = time.time,
        monotonic_fn: Callable[[], float] = time.monotonic,
        process_supervisor: HostedByokProcessSupervisor | None = None,
    ) -> None:
        from .backends.gemini import GeminiVision, _require_genai
        from .backends.openrouter import OpenRouterVision, _OPENROUTER_BASE_URL, _require_openai

        overrides = dict(backend_factories or {})
        unknown = sorted(set(overrides) - set(HOSTED_BYOK_CAPABILITIES))
        if unknown:
            raise HostedByokContractError(
                f"unsupported hosted BYOK backend factory provider(s): {unknown!r}."
            )
        self._backend_factories = MappingProxyType(overrides)
        self._gemini_client_factory = gemini_client_factory
        self._openrouter_client_factory = openrouter_client_factory
        self._time_fn = time_fn
        self._monotonic_fn = monotonic_fn
        self._gemini_backend_class = GeminiVision
        self._gemini_require_genai = _require_genai
        self._openrouter_backend_class = OpenRouterVision
        self._openrouter_require_openai = _require_openai
        self._openrouter_require_httpx = _require_httpx_async_client
        self._openrouter_base_url = _OPENROUTER_BASE_URL
        if process_supervisor is not None and type(process_supervisor) is not HostedByokSubprocessSupervisor:
            raise HostedByokContractError(
                "process_supervisor must be an exact HostedByokSubprocessSupervisor."
            )
        if process_supervisor is not None and (
            time_fn is not time.time or monotonic_fn is not time.monotonic
        ):
            raise HostedByokContractError(
                "production Hosted BYOK requires the exact system wall and monotonic clocks."
            )
        self._process_supervisor = process_supervisor
        # Weak identity claims enforce the fresh-object contract without retaining clients/backends
        # (and therefore without turning this factory into a resource cache).
        self._injected_claims: dict[int, weakref.ReferenceType[Any]] = {}
        self._locked = True

    def transport_safety(self, request: HostedByokRequest) -> HostedByokTransportSafety:
        """Classify whether this route needs a Cloud process supervisor before it may execute."""

        return _classify_transport_safety(self, request)

    def _assert_transport_permitted(self, request: HostedByokRequest) -> None:
        safety = _classify_transport_safety(self, request)
        if not safety.requires_process_supervisor:
            return
        supervisor = self._process_supervisor
        if type(supervisor) is not HostedByokSubprocessSupervisor:
            raise HostedByokContractError(
                "hosted BYOK production transport requires an exact subprocess supervisor."
            )

    def bind(self, request: HostedByokRequest) -> "HostedByokVision":
        """Return a request-bound public Vision adapter accepted by ``Percept.create(vision=...)``.

        The bound object stores no provider client/backend/result. Each ``judge`` call still enters
        this factory and creates a fresh invocation-local transport.
        """

        self._validate_route(request)
        return HostedByokVision(self, request)

    def seal_fused_dispatch(self, request: HostedByokRequest) -> Callable[..., Any]:
        """Capture an immutable fused dispatch for compiled Watches.

        Compiled runtime routes must not reread this factory's mutable maps, helpers, or methods
        after Watch preparation.  The snapshot deliberately owns only scalar routing state and
        copied callback references; it owns no provider client/backend and still constructs one per
        call through the normal boundary.
        """

        self._validate_route(request)
        if request.provider != "gemini" or request.capability != "vision.judge.fused":
            raise HostedByokUnsupportedError(
                "compiled hosted BYOK dispatch requires Gemini vision.judge.fused."
            )
        method_names = (
            "judge",
            "_deadline_at_entry",
            "_preflight",
            "_build_invocation",
            "_build_gemini",
            "_build_openrouter",
            "_claim_injected",
            "transport_safety",
            "_assert_transport_permitted",
            "_judge_in_process",
        )
        static_names = ("_validate_route", "_settle_task", "_close_invocation")
        captured = {}
        for name in (*method_names, *static_names):
            implementation = HostedByokAdapterFactory.__dict__[name]
            captured[name] = (
                implementation.__func__ if isinstance(implementation, staticmethod) else implementation
            )
        snapshot = object.__new__(HostedByokAdapterFactory)
        snapshot._backend_factories = MappingProxyType(dict(self._backend_factories))
        snapshot._gemini_client_factory = self._gemini_client_factory
        snapshot._openrouter_client_factory = self._openrouter_client_factory
        snapshot._time_fn = self._time_fn
        snapshot._monotonic_fn = self._monotonic_fn
        snapshot._gemini_backend_class = self._gemini_backend_class
        snapshot._gemini_require_genai = self._gemini_require_genai
        snapshot._openrouter_backend_class = self._openrouter_backend_class
        snapshot._openrouter_require_openai = self._openrouter_require_openai
        snapshot._openrouter_require_httpx = self._openrouter_require_httpx
        snapshot._openrouter_base_url = self._openrouter_base_url
        snapshot._process_supervisor = self._process_supervisor
        snapshot._injected_claims = {}

        def freeze(implementation: Any, *, globals_extra: dict[str, Any] | None = None) -> Any:
            if not isinstance(implementation, FunctionType):
                raise HostedByokContractError("hosted BYOK trusted dispatch was not a function.")
            global_snapshot = dict(implementation.__globals__)
            if globals_extra:
                global_snapshot.update(globals_extra)
            frozen = FunctionType(
                implementation.__code__,
                global_snapshot,
                name=implementation.__name__,
                argdefs=implementation.__defaults__,
                closure=implementation.__closure__,
            )
            frozen.__kwdefaults__ = implementation.__kwdefaults__
            return frozen

        frozen_settings = freeze(_settings_for_model)
        frozen_class = type("_SealedHostedByokAdapterFactory", (), {})
        # ``_close_invocation`` calls the task-settler by class name in the normal implementation;
        # bind that name to this frozen surrogate before copying it so class-level helper poisoning
        # after Watch preparation cannot redirect teardown.
        frozen_static = {
            "_validate_route": freeze(captured["_validate_route"]),
            "_settle_task": freeze(captured["_settle_task"]),
        }
        for name, implementation in frozen_static.items():
            setattr(frozen_class, name, staticmethod(implementation))
        frozen_static["_close_invocation"] = freeze(
            captured["_close_invocation"],
            globals_extra={"HostedByokAdapterFactory": frozen_class},
        )
        setattr(frozen_class, "_close_invocation", staticmethod(frozen_static["_close_invocation"]))
        globals_extra = {
            "_settings_for_model": frozen_settings,
        }
        for name in method_names:
            setattr(snapshot, name, MethodType(freeze(captured[name], globals_extra=globals_extra), snapshot))
        for name, implementation in frozen_static.items():
            setattr(snapshot, name, implementation)
        snapshot._locked = True

        async def dispatch(*args: Any, **kwargs: Any) -> Verdict:
            return await snapshot.judge(request, *args, **kwargs)

        return dispatch

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
        """Dispatch production routes only through the exact contained-process supervisor."""

        if type(self) is not HostedByokAdapterFactory:
            raise HostedByokContractError("factory must be the exact SDK HostedByokAdapterFactory.")
        entered_monotonic = float(self._monotonic_fn())
        if type(request) is not HostedByokRequest:
            raise HostedByokContractError(
                "request must be an immutable HostedByokRequest; implicit configuration is unsupported."
            )
        deadline_monotonic = self._deadline_at_entry(
            request, entered_monotonic=entered_monotonic
        )
        _validate_hosted_call_inputs(
            request,
            condition=condition,
            frame=frame,
            roi=roi,
            exemplar=exemplar,
            corrections=corrections,
            frames=frames,
            audio=audio,
            context=context,
        )
        self._preflight(
            request,
            deadline_monotonic=deadline_monotonic,
            frame=frame,
            exemplar=exemplar,
            frames=frames,
            audio=audio,
            context=context,
        )
        safety = _classify_transport_safety(self, request)
        if safety.requires_process_supervisor:
            self._assert_transport_permitted(request)
            supervisor = self._process_supervisor
            assert type(supervisor) is HostedByokSubprocessSupervisor
            return await supervisor._execute_impl(
                request,
                condition,
                frame,
                deadline_monotonic=deadline_monotonic,
                roi=roi,
                exemplar=exemplar,
                corrections=corrections,
                frames=frames,
                audio=audio,
                context=context,
            )
        if self._process_supervisor is not None:
            raise HostedByokContractError(
                "injected Hosted BYOK transports are test-only and cannot enter production mode."
            )
        return await self._judge_in_process(
            request,
            condition,
            frame,
            roi=roi,
            exemplar=exemplar,
            corrections=corrections,
            frames=frames,
            audio=audio,
            context=context,
            deadline_monotonic=deadline_monotonic,
        )

    async def _judge_in_process(
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
        deadline_monotonic: float | None = None,
    ) -> Verdict:
        """Perform one isolated vision judgment and return the normal public :class:`Verdict`.

        A timeout or cancellation stops accepting a provider result and requests task cancellation
        immediately. It does not release invocation ownership: Python cannot forcibly terminate an
        in-process coroutine that suppresses ``CancelledError``, so this await completes only after
        the provider coroutine and all teardown have actually settled.
        """

        invocation: _Invocation | None = None
        provider_task: asyncio.Future | None = None
        cancellation_task: asyncio.Future | None = None
        task_scope: _InvocationTaskScope | None = None
        cleanup_interrupted = False
        cleanup_failed = False
        try:
            if deadline_monotonic is None:
                deadline_monotonic = self._deadline_at_entry(request)
            timeout_s = self._preflight(
                request,
                deadline_monotonic=deadline_monotonic,
                frame=frame,
                exemplar=exemplar,
                frames=frames,
                audio=audio,
                context=context,
            )
            expires_at = deadline_monotonic
            task_scope = _InvocationTaskScope()
            task_scope.enter()
            # Allocate the record before construction. Client factories and vendor constructors put
            # their client into it immediately, so a later backend-constructor or exact-model failure
            # is still closed by the unconditional teardown below.
            invocation = _Invocation()
            self._build_invocation(request, timeout_s, invocation)
            remaining_s = expires_at - float(self._monotonic_fn())
            if remaining_s <= 0:
                raise HostedByokDeadlineExceeded(
                    "hosted BYOK invocation deadline elapsed during provider construction."
                )
            if request.cancellation is not None and request.cancellation.is_set():
                raise HostedByokCancelled()
            # Python 3.10 has no public pending-cancellation counter. Yield once after synchronous
            # construction so a queued caller cancellation is delivered before provider dispatch.
            await asyncio.sleep(0)
            judge = getattr(invocation.backend, "judge", None)
            if not callable(judge):
                raise HostedByokUnsupportedError(
                    f"provider {request.provider!r} has no vision.judge capability."
                )
            # The backend's descriptor lookup is injectable code too. Re-check at the last possible
            # point before task creation so a construction/lookup race can spend zero provider calls.
            if request.cancellation is not None and request.cancellation.is_set():
                raise HostedByokCancelled()
            provider_task = asyncio.ensure_future(
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
            wait_for: set[asyncio.Future] = {provider_task}
            if request.cancellation is not None:
                cancellation_task = asyncio.ensure_future(request.cancellation.wait())
                wait_for.add(cancellation_task)
            done, _ = await asyncio.wait(
                wait_for, timeout=remaining_s, return_when=asyncio.FIRST_COMPLETED
            )
            if request.cancellation is not None and request.cancellation.is_set():
                cleanup_interrupted |= await self._settle_task(provider_task, cancel=True)
                raise HostedByokCancelled()
            if cancellation_task is not None and cancellation_task in done:
                try:
                    cancellation_task.result()
                except (asyncio.CancelledError, Exception):
                    cleanup_interrupted |= await self._settle_task(provider_task, cancel=True)
                    raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
                # ``wait()`` returning without the signal becoming set violates the public protocol.
                # Fail closed instead of treating the early wake as a deadline or dispatch result.
                cleanup_interrupted |= await self._settle_task(provider_task, cancel=True)
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            if provider_task not in done:
                cleanup_interrupted |= await self._settle_task(provider_task, cancel=True)
                raise HostedByokDeadlineExceeded(
                    "hosted BYOK invocation exceeded its effective deadline."
                )
            if float(self._monotonic_fn()) >= expires_at:
                # A provider can block the event loop synchronously and return a completed task after
                # ``asyncio.wait``'s timeout should have fired. Consume that terminal result, but never
                # accept it after the monotonic deadline.
                cleanup_interrupted |= await self._settle_task(provider_task, cancel=False)
                raise HostedByokDeadlineExceeded(
                    "hosted BYOK provider completed after its effective deadline."
                )
            try:
                result = provider_task.result()
            except asyncio.CancelledError:
                # A provider that cancels itself is a provider failure, not caller cancellation.
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            except Exception:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            if type(result) is not Verdict:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            if not _verdict_is_credential_free(result, request.credential):
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            return result
        except (HostedByokError, HostedByokCancelled):
            raise
        except asyncio.CancelledError:
            if provider_task is not None:
                cleanup_interrupted |= await self._settle_task(provider_task, cancel=True)
            raise
        except Exception:
            # Never forward a vendor exception: an SDK/client may echo headers, constructor args, or
            # even an attacker-controlled exception class name. A fixed code is the whole surface.
            if provider_task is not None:
                cleanup_interrupted |= await self._settle_task(provider_task, cancel=True)
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
        finally:
            if cancellation_task is not None:
                cleanup_interrupted |= await self._settle_task(cancellation_task, cancel=True)
            if invocation is not None:
                interrupted, failed = await self._close_invocation(invocation)
                cleanup_interrupted |= interrupted
                cleanup_failed |= failed
            if task_scope is not None:
                cleanup_interrupted |= await task_scope.settle()
                task_scope.exit()
            if cleanup_failed:
                # A close getter/non-callable/raising close means we cannot report a successful,
                # cancelled, or timed-out invocation until all descendant work is dead.  The scope
                # above has joined it; this fixed error is the fail-closed public result.
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            if cleanup_interrupted:
                # Cancellation received while result/deadline teardown was already in progress is
                # delivered only after all owned work has completed. This deliberately overrides a
                # pending result or typed timeout: the caller explicitly cancelled its own await.
                raise asyncio.CancelledError()

    def _deadline_at_entry(
        self,
        request: HostedByokRequest,
        *,
        entered_monotonic: float | None = None,
    ) -> float:
        if entered_monotonic is None:
            entered_monotonic = float(self._monotonic_fn())
        bounds = [entered_monotonic + _DEFAULT_TIMEOUT_S]
        if request.timeout_s is not None:
            bounds.append(entered_monotonic + float(request.timeout_s))
        if request.deadline_unix_s is not None:
            entered_unix = float(self._time_fn())
            bounds.append(
                entered_monotonic + float(request.deadline_unix_s) - entered_unix
            )
        deadline_monotonic = min(bounds)
        if deadline_monotonic <= entered_monotonic:
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation deadline elapsed at public judge entry."
            )
        return deadline_monotonic

    def _preflight(
        self,
        request: HostedByokRequest,
        *,
        deadline_monotonic: float,
        frame: Any,
        exemplar: Any,
        frames: list | None,
        audio: Any,
        context: dict | None,
    ) -> float:
        self._validate_route(request)
        if not _is_hosted_image(frame):
            if request.capability == "vision.judge.fused":
                raise HostedByokUnsupportedError(
                    "vision.judge.fused requires both image and audio evidence as exact non-empty "
                    "media carriers."
                )
            raise HostedByokUnsupportedError(
                "vision.judge requires exact non-empty image evidence."
            )
        if exemplar is not None and not _is_hosted_image(exemplar):
            raise HostedByokUnsupportedError(
                "hosted BYOK exemplar evidence must be an exact non-empty image."
            )
        if frames is not None and (
            type(frames) is not list or any(not _is_hosted_image(item) for item in frames)
        ):
            raise HostedByokUnsupportedError(
                "hosted BYOK frames must be an exact list of non-empty image evidence."
            )
        if context is not None and type(context) is not dict:
            raise HostedByokUnsupportedError("hosted BYOK context must be an exact dict.")
        if context is not None and _RESERVED_CONTEXT_KEYS.intersection(context):
            raise HostedByokUnsupportedError(
                "hosted BYOK contract 1.0 rejects reserved context control keys."
            )
        if request.capability == "vision.judge":
            if audio is not None:
                raise HostedByokUnsupportedError(
                    "vision.judge forbids audio evidence; declare vision.judge.fused exactly."
                )
        else:
            if not _is_hosted_audio(audio):
                raise HostedByokUnsupportedError(
                    "vision.judge.fused requires both image and audio evidence as exact non-empty "
                    "media carriers."
                )
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()

        timeout_s = deadline_monotonic - float(self._monotonic_fn())
        if timeout_s <= 0:
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation deadline elapsed before provider construction."
            )
        return timeout_s

    @staticmethod
    def _validate_route(request: HostedByokRequest) -> None:
        if type(request) is not HostedByokRequest:
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

    def _build_invocation(
        self,
        request: HostedByokRequest,
        timeout_s: float,
        invocation: _Invocation,
    ) -> None:
        override = self._backend_factories.get(request.provider)
        if override is not None:
            try:
                backend = override(request, timeout_s)
            except Exception:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
            invocation.backend = backend
            self._claim_injected(backend)
            return
        if request.provider == "gemini":
            self._build_gemini(request, timeout_s, invocation)
            return
        if request.provider == "openrouter":
            self._build_openrouter(request, timeout_s, invocation)
            return
        raise HostedByokUnsupportedError(f"unsupported hosted BYOK provider {request.provider!r}.")

    def _build_gemini(
        self,
        request: HostedByokRequest,
        timeout_s: float,
        invocation: _Invocation,
    ) -> None:
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
        else:
            genai, types = self._gemini_require_genai()
            client = genai.Client(
                api_key=request.credential,
                http_options=types.HttpOptions(
                    timeout=max(1, int(timeout_s * 1000)),
                    client_args={"trust_env": False},
                    async_client_args={"trust_env": False},
                ),
            )
            binding = HostedByokGeminiClient(client=client, transport_types=types)
        invocation.client = binding.client
        if self._gemini_client_factory is not None:
            self._claim_injected(binding.client)
        backend = self._gemini_backend_class(
            settings,
            client=binding.client,
            transport_types=binding.transport_types,
            strip_prompt="change",
        )
        invocation.backend = backend
        if backend._model != request.model:
            raise HostedByokUnsupportedError(
                f"Gemini refused exact model {request.model!r}; no fallback is permitted."
            )

    def _build_openrouter(
        self,
        request: HostedByokRequest,
        timeout_s: float,
        invocation: _Invocation,
    ) -> None:
        settings = _settings_for_model(request.model)
        if self._openrouter_client_factory is not None:
            try:
                client = self._openrouter_client_factory(request, timeout_s)
            except Exception:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
        else:
            AsyncOpenAI = self._openrouter_require_openai()
            AsyncHttpClient = self._openrouter_require_httpx()
            http_client = AsyncHttpClient(timeout=timeout_s, trust_env=False)
            invocation.client = http_client
            client = AsyncOpenAI(
                base_url=self._openrouter_base_url,
                api_key=request.credential,
                timeout=timeout_s,
                http_client=http_client,
                default_headers={
                    "HTTP-Referer": "https://github.com/divi-vijayakumar/Percept",
                    "X-Title": "Percept",
                },
            )
        invocation.client = client
        if self._openrouter_client_factory is not None:
            self._claim_injected(client)
        backend = self._openrouter_backend_class(settings, client=client, timeout_s=timeout_s)
        invocation.backend = backend
        if backend._model != request.model:
            raise HostedByokUnsupportedError(
                f"OpenRouter refused exact model {request.model!r}; no fallback is permitted."
            )

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
    async def _settle_task(task: asyncio.Future, *, cancel: bool) -> bool:
        """Optionally cancel ``task`` and await its actual terminal state without detaching it.

        ``asyncio.shield`` keeps repeated cancellation of the owning ``judge`` task from silently
        abandoning this child. The boolean tells the caller whether an additional caller cancellation
        arrived during settlement; child cancellation itself does not count.
        """

        return await _settle_owned_task(task, cancel=cancel)

    @staticmethod
    async def _close_invocation(invocation: _Invocation) -> tuple[bool, bool]:
        """Close every acquired component to actual completion, without threads or orphan tasks."""

        interrupted = False
        failed = False
        seen: set[int] = set()
        for component in (invocation.backend, invocation.client):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            try:
                aclose = getattr(component, "aclose", None)
            except asyncio.CancelledError:
                failed = True
                continue
            except Exception:
                failed = True
                continue
            if aclose is not None:
                closer = aclose
            else:
                try:
                    closer = getattr(component, "close", None)
                except asyncio.CancelledError:
                    failed = True
                    continue
                except Exception:
                    failed = True
                    continue
            if closer is None:
                continue
            if not callable(closer):
                failed = True
                continue
            try:
                result = closer()
                if inspect.isawaitable(result):
                    close_task = asyncio.ensure_future(result)
                    interrupted |= await HostedByokAdapterFactory._settle_task(
                        close_task,
                        cancel=False,
                    )
                    if close_task.cancelled() or close_task.exception() is not None:
                        failed = True
            except asyncio.CancelledError:
                failed = True
            except Exception:
                failed = True
        return interrupted, failed


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


def _select_containment_capability(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> tuple[str, list[str]]:
    selected_system = sys.platform if system is None else system
    selected_machine = platform.machine().lower() if machine is None else machine.lower()
    if selected_system == "darwin":
        path = "/usr/bin/sandbox-exec"
        prefix = [path, "-p", _BOUNDARY_PROFILE]
        try:
            metadata = os.stat(path, follow_symlinks=False)
        except OSError:
            raise HostedByokContractError(
                "hosted BYOK trusted process-containment executable is unavailable."
            ) from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise HostedByokContractError(
                "hosted BYOK process-containment executable failed ownership validation."
            )
        return "darwin_sandbox_no_descendants", prefix
    if selected_system == "linux" and selected_machine in {"x86_64", "amd64"}:
        return "linux_seccomp_amd64_no_descendants", []
    raise HostedByokContractError(
        "hosted BYOK process containment is unavailable for this exact platform capability."
    )


def _clean_child_environment() -> dict[str, str]:
    """Return an allowlisted startup environment with no transport or credential controls."""

    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }


def _clean_child_import_roots() -> list[str]:
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates: list[str] = [package_root]
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass
    roots: list[str] = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in roots and os.path.isdir(normalized):
            roots.append(normalized)
    return roots


def _launch_contained_worker(
    boundary_kind: str,
) -> tuple[subprocess.Popen[bytes], Connection, bytes]:
    selected_kind, prefix = _select_containment_capability()
    if boundary_kind != selected_kind:
        raise HostedByokContractError(
            "hosted BYOK process-containment capability changed after supervisor construction."
        )
    challenge = secrets.token_bytes(32)
    parent_socket, child_socket = socket.socketpair()
    child_fd = child_socket.fileno()
    command = [
        *prefix,
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _CHILD_BOOTSTRAP,
        json.dumps(_clean_child_import_roots(), separators=(",", ":")),
        str(child_fd),
        challenge.hex(),
        boundary_kind,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(child_fd,),
            cwd="/",
            env=_clean_child_environment(),
        )
    except Exception:
        parent_socket.close()
        child_socket.close()
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    child_socket.close()
    return process, Connection(parent_socket.detach()), challenge


def _terminate_and_wait_contained_worker(process: subprocess.Popen[bytes]) -> None:
    """Establish the terminal worker state; the authenticated boundary proves descendants are absent."""

    try:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        process.wait()
    except Exception:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    if process.poll() is None:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)


def _scrub_provider_environment() -> None:
    for key in _PROVIDER_ENV_KEYS:
        os.environ.pop(key, None)


def _install_linux_seccomp_amd64() -> None:
    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class SockFprog(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ushort),
            ("filters", ctypes.POINTER(SockFilter)),
        ]

    audit_arch_x86_64 = 0xC000003E
    seccomp_ret_kill_process = 0x80000000
    seccomp_ret_allow = 0x7FFF0000
    seccomp_ret_errno = 0x00050000
    clone_thread = 0x00010000
    filter_specs = [
        (0x20, 0, 0, 4),  # load seccomp_data.arch
        (0x15, 1, 0, audit_arch_x86_64),
        (0x06, 0, 0, seccomp_ret_kill_process),
        (0x20, 0, 0, 0),  # load seccomp_data.nr
        (0x35, 0, 1, 0x40000000),  # x32 ABI syscalls must not bypass the x86_64 table
        (0x06, 0, 0, seccomp_ret_errno | errno.ENOSYS),
        (0x15, 0, 5, 56),  # clone: inspect flags; non-clone skips the block
        (0x20, 0, 0, 16),  # low 32 bits of clone flags
        (0x54, 0, 0, clone_thread),
        (0x15, 0, 1, 0),
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x06, 0, 0, seccomp_ret_allow),  # CLONE_THREAD remains permitted
        (0x15, 0, 1, 57),  # fork
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x15, 0, 1, 58),  # vfork
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x15, 0, 1, 435),  # clone3; ENOSYS makes pthreads fall back to clone
        (0x06, 0, 0, seccomp_ret_errno | errno.ENOSYS),
        (0x15, 0, 1, 59),  # execve
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x15, 0, 1, 322),  # execveat
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x06, 0, 0, seccomp_ret_allow),
    ]
    filters = (SockFilter * len(filter_specs))(
        *(SockFilter(*spec) for spec in filter_specs)
    )
    program = SockFprog(
        length=len(filter_specs),
        filters=ctypes.cast(filters, ctypes.POINTER(SockFilter)),
    )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    libc.syscall.restype = ctypes.c_long
    if libc.syscall(317, 1, 1, ctypes.byref(program)) != 0:  # seccomp FILTER + TSYNC
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)


def _establish_no_descendant_boundary(boundary_kind: str) -> None:
    """Install the exact platform boundary, prove fork denial, and prove threads remain usable."""

    if boundary_kind == "darwin_sandbox_no_descendants":
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            if resource.getrlimit(resource.RLIMIT_NPROC) != (0, 0):
                raise RuntimeError
        except Exception:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    elif boundary_kind == "linux_seccomp_amd64_no_descendants":
        _install_linux_seccomp_amd64()
    else:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    try:
        pid = os.fork()
    except OSError:
        pid = None
    if pid is None:
        import threading

        completed: list[bool] = []
        thread = threading.Thread(target=lambda: completed.append(True))
        thread.start()
        thread.join()
        if completed == [True]:
            return
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    if pid == 0:
        os._exit(91)
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    finally:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)


def _hosted_byok_clean_child_main(
    connection_fd: int,
    challenge_hex: str,
    boundary_kind: str,
) -> None:
    """Clean-start child: attest containment before receiving the credential-bearing request."""

    connection = Connection(connection_fd)
    try:
        if type(challenge_hex) is not str:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        challenge = bytes.fromhex(challenge_hex)
        if len(challenge) != 32:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        _scrub_provider_environment()
        _establish_no_descendant_boundary(boundary_kind)
        connection.send_bytes(
            _wire_dumps(
                (
                    "ready",
                    HOSTED_BYOK_PROCESS_ISOLATION_VERSION,
                    challenge,
                    boundary_kind,
                )
            )
        )
        worker_payload = connection.recv_bytes(maxlength=_MAX_PROCESS_IPC_BYTES + 1)

        async def run() -> Verdict:
            values = _wire_loads(worker_payload)
            if type(values) is not tuple or len(values) != 10:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            request, condition, frame, roi, exemplar, corrections, frames, audio, context, deadline = (
                values
            )
            if (
                type(request) is not HostedByokRequest
                or type(deadline) is not float
                or time.monotonic() >= deadline
            ):
                raise HostedByokDeadlineExceeded(
                    "hosted BYOK invocation exceeded its effective deadline."
                )
            factory = HostedByokAdapterFactory()
            return await factory._judge_in_process(
                request,
                condition,
                frame,
                roi=roi,
                exemplar=exemplar,
                corrections=corrections,
                frames=frames,
                audio=audio,
                context=context,
                deadline_monotonic=deadline,
            )

        try:
            result = asyncio.run(run())
            connection.send_bytes(_wire_dumps(("ok", result)))
        except HostedByokDeadlineExceeded as exc:
            connection.send_bytes(_wire_dumps(("deadline", str(exc))))
        except HostedByokCancelled:
            connection.send_bytes(_wire_dumps(("cancelled", "")))
        except HostedByokUnsupportedError as exc:
            connection.send_bytes(_wire_dumps(("unsupported", str(exc))))
        except HostedByokContractError as exc:
            connection.send_bytes(_wire_dumps(("contract", str(exc))))
        except BaseException:
            connection.send_bytes(_wire_dumps(("failure", _PROVIDER_FAILURE_CODE)))
    except BaseException:
        pass
    finally:
        connection.close()


def _settings_for_model(model: str) -> Settings:
    """Construct explicit defaults without calling the environment-backed ``load_settings``."""

    defaults = Settings()
    return replace(defaults, models=replace(defaults.models, vision_model=model))


def _require_httpx_async_client() -> Any:
    try:
        from httpx import AsyncClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("hosted OpenRouter requires the OpenAI HTTP transport.") from exc
    return AsyncClient


def _positive_duration(name: str, value: float) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise HostedByokContractError(f"{name} must be a finite positive number.")
    return float(value)


def _validate_hosted_call_inputs(
    request: HostedByokRequest,
    *,
    condition: Any,
    frame: Any,
    roi: Any,
    exemplar: Any,
    corrections: Any,
    frames: Any,
    audio: Any,
    context: Any,
) -> None:
    if type(request) is not HostedByokRequest:
        raise HostedByokContractError(
            "request must be an immutable HostedByokRequest; implicit configuration is unsupported."
        )
    if type(condition) is not str or not condition or condition != condition.strip():
        raise HostedByokContractError("condition must be an exact non-empty trimmed string.")
    if roi is not None and (
        type(roi) is not Box
        or any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in (roi.x, roi.y, roi.w, roi.h)
        )
    ):
        raise HostedByokContractError("roi must be an exact finite Box when supplied.")
    if corrections is not None and (
        type(corrections) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in corrections.items())
    ):
        raise HostedByokContractError("corrections must be an exact string-to-string dict.")
    if context is not None:
        if type(context) is not dict or any(type(key) is not str for key in context):
            raise HostedByokContractError("context must be an exact string-keyed dict.")
        # Run the exact-carrier encoder before any child exists. It never calls repr/getters/reduce.
        _wire_value(context)
    checked = (condition, frame, roi, exemplar, corrections, frames, audio, context)
    if any(_input_contains_credential(value, request.credential) for value in checked):
        raise HostedByokContractError("hosted BYOK call inputs must not contain the credential.")


def _input_contains_credential(value: Any, credential: str) -> bool:
    credential_bytes = credential.encode("utf-8", errors="surrogatepass")
    value_type = type(value)
    if value is None or value_type in (bool, int, float):
        return False
    if value_type is str:
        return credential in value
    if value_type in (bytes, bytearray, memoryview):
        return credential_bytes in bytes(value)
    if value_type in (list, tuple, set, frozenset):
        return any(_input_contains_credential(item, credential) for item in value)
    if value_type is dict:
        return any(
            _input_contains_credential(key, credential)
            or _input_contains_credential(item, credential)
            for key, item in value.items()
        )
    if value_type in (AudioFrame, Box):
        return any(
            _input_contains_credential(getattr(value, item.name), credential)
            for item in dataclass_fields(value_type)
        )
    return True


def _is_hosted_image(value: Any) -> bool:
    """Whether ``value`` is one exact image carrier supported by both hosted adapters."""

    if type(value) in (bytes, bytearray):
        return bool(value)
    if (
        type(value) is tuple
        and len(value) == 2
        and type(value[0]) is str
        and value[0].strip().lower().startswith("image/")
        and type(value[1]) in (bytes, bytearray)
    ):
        return bool(value[1])
    if type(value) is dict and set(value) == {"type", "source"} and value.get("type") == "image":
        source = value.get("source")
        return (
            type(source) is dict
            and set(source) == {"type", "media_type", "data"}
            and source.get("type") == "base64"
            and type(source.get("media_type")) is str
            and source["media_type"].strip().lower().startswith("image/")
            and type(source.get("data")) is str
            and bool(source["data"])
        )
    return False


def _is_hosted_audio(value: Any) -> bool:
    """Whether ``value`` is one exact non-empty Gemini audio carrier."""

    if type(value) is bytes:
        return bool(value)
    if (
        type(value) is tuple
        and len(value) == 2
        and type(value[0]) is str
        and value[0] == value[0].strip()
        and value[0].lower().startswith("audio/")
        and type(value[1]) is bytes
    ):
        return bool(value[1])
    if type(value) is AudioFrame:
        if (
            type(value.sample_rate) is not int
            or value.sample_rate <= 0
            or type(value.channels) is not int
            or value.channels <= 0
            or type(value.sample_format) is not str
            or value.sample_format not in {"s16le", "f32le", "u8"}
            or not _audio_bounds_are_valid(value)
        ):
            return False
        if value.encoded is None:
            # Raw PCM has no source MIME: the trusted Gemini adapter constructs an audio/wav
            # container from these validated layout fields.
            return type(value.pcm) is bytes and bool(value.pcm) and value.mime is None
        return (
            type(value.pcm) is bytes
            and not value.pcm
            and type(value.encoded) is bytes
            and bool(value.encoded)
            and type(value.mime) is str
            and value.mime == value.mime.strip()
            and value.mime.lower().startswith("audio/")
        )
    return False


def _audio_bounds_are_valid(value: AudioFrame) -> bool:
    for bound in (value.t_start, value.t_end):
        if bound is not None and (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(float(bound))
        ):
            return False
    return value.t_start is None or value.t_end is None or value.t_end >= value.t_start


def _verdict_is_credential_free(verdict: Verdict, credential: str) -> bool:
    """Recursively inspect the complete public Verdict carrier without rendering attacker values."""

    secret_bytes = credential.encode("utf-8", errors="surrogatepass")
    seen: set[int] = set()

    def contains(value: Any) -> bool:
        if value is None or type(value) in (bool, int, float):
            return False
        if type(value) is str:
            return credential in value
        if type(value) in (bytes, bytearray, memoryview):
            return secret_bytes in bytes(value)

        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)

        if type(value) in (ClipEvent, SubjectRef):
            return any(contains(getattr(value, item.name)) for item in dataclass_fields(value))
        if type(value) is dict:
            return any(contains(key) or contains(item) for key, item in value.items())
        if type(value) in (list, tuple, set, frozenset):
            return any(contains(item) for item in value)
        # Hosted provider adapters parse JSON-like data into SDK DTOs. An opaque custom carrier
        # cannot be proven credential-free without invoking attacker-controlled rendering/access.
        raise TypeError("unsupported hosted BYOK Verdict carrier")

    if type(verdict) is not Verdict:
        return False
    try:
        return not any(contains(getattr(verdict, item.name)) for item in dataclass_fields(Verdict))
    except Exception:
        return False


__all__ = [
    "HOSTED_BYOK_CONTRACT_VERSION",
    "HOSTED_BYOK_PROCESS_ISOLATION_VERSION",
    "HOSTED_BYOK_CAPABILITIES",
    "HostedByokRequest",
    "HostedByokCancellationSignal",
    "HostedByokProcessSupervisor",
    "HostedByokSubprocessSupervisor",
    "HostedByokTransportSafety",
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
